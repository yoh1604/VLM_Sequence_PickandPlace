import base64
import json
import re
import traceback

import ollama
from openai import OpenAI

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


class LogicValidator:
    """
    LogicValidator hanya bertugas sebagai gatekeeper:
    - PASS jika action_plan aman/logis
    - FAIL jika action_plan tidak aman/tidak logis
    - TIDAK BOLEH mengubah, menambah, menghapus, atau menyusun ulang action_plan

    Executor harus selalu memakai action_plan asli dari Planner.
    """

    def __init__(self, api_key=None, model_name=None, provider=None):
        self.api_key = api_key
        self.model_name = model_name
        self.cloud_provider = None
        self.provider = (provider or "auto").lower()
        self.client = None

        if self.model_name:
            self.mode = "LOCAL"
            print(f"LogicValidator berjalan di mode LOKAL dengan model: {self.model_name}")
        elif self.api_key:
            self.mode = "CLOUD"
            print("LogicValidator berjalan di mode CLOUD")
        else:
            raise ValueError("Harus memasukkan api_key atau model_name")

        if self.api_key is not None:
            if self.provider in ["gemini", "google", "gcp", "auto"] and genai is not None:
                try:
                    self.client = genai.Client(api_key=self.api_key)
                    self.cloud_model = "gemini-2.5-pro"
                    self.cloud_provider = "gemini"
                    print("☁️ LogicValidator diinisialisasi untuk mode CLOUD (Gemini).")
                except Exception as e:
                    if self.provider in ["gemini", "google", "gcp"]:
                        raise RuntimeError(
                            "Gemini initialization failed. Make sure google-genai is installed "
                            "and the Gemini API key is valid."
                        ) from e

            if self.cloud_provider is None:
                self.client = OpenAI(api_key=self.api_key)
                self.cloud_model = "gpt-4o"
                self.cloud_provider = "openai"
                print("☁️ LogicValidator diinisialisasi untuk mode CLOUD (OpenAI).")
        else:
            print(f"🔌 LogicValidator diinisialisasi untuk mode LOKAL (Model: {self.model_name}).")

    # -------------------------------------------------------------------------
    # Basic utilities
    # -------------------------------------------------------------------------

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _extract_planner_action_plan(self, planner_json):
        """
        Ambil action_plan asli dari Planner.
        Ini adalah satu-satunya plan yang boleh dieksekusi.
        """
        if isinstance(planner_json, dict):
            plan = planner_json.get("action_plan") or planner_json.get("final_action_plan")
            return plan if isinstance(plan, list) else []

        return planner_json if isinstance(planner_json, list) else []

    def _safe_json_loads(self, raw_text):
        """
        Parse JSON secara defensif.
        Berguna untuk local model yang kadang membungkus JSON dengan teks tambahan.
        """
        if isinstance(raw_text, dict):
            return raw_text

        if raw_text is None:
            raise ValueError("Empty model response")

        text = str(raw_text).strip()
        text = text.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if not json_match:
                raise
            return json.loads(json_match.group(0))

    def _normalize_validation_result(self, validation_result, planner_json):
        """
        Normalisasi output validator.

        PENTING:
        - Validator boleh memberi validation_status dan feedback.
        - Validator TIDAK BOLEH menentukan final_action_plan.
        - final_action_plan selalu dipaksa menjadi action_plan asli dari Planner.
        """
        original_plan = self._extract_planner_action_plan(planner_json)

        if not isinstance(validation_result, dict):
            return {
                "validation_status": "FAIL",
                "feedback": f"Validator returned invalid result type: {type(validation_result).__name__}",
                "final_action_plan": original_plan,
            }

        status = str(validation_result.get("validation_status", "FAIL")).upper().strip()
        if status not in ["PASS", "FAIL"]:
            status = "FAIL"

        feedback = validation_result.get("feedback", "")
        if not isinstance(feedback, str):
            feedback = str(feedback)

        return {
            "validation_status": status,
            "feedback": feedback,
            "final_action_plan": original_plan,
        }

    def _error_result(self, error_type, error_message, planner_json):
        """
        Error juga harus berbentuk dict supaya main_teleop.py tidak crash.
        """
        return {
            "error": error_type,
            "validation_status": "FAIL",
            "feedback": error_message,
            "final_action_plan": self._extract_planner_action_plan(planner_json),
        }

    # -------------------------------------------------------------------------
    # Cloud/local model callers
    # -------------------------------------------------------------------------

    def _call_gemini_vision_json(self, image_path, prompt):
        """
        Gemini SDK baru: google-genai.
        Install:
            pip install -U google-genai
        """
        if types is None:
            raise RuntimeError(
                "google.genai.types tidak tersedia. Install ulang dengan: pip install -U google-genai"
            )

        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        response = self.client.models.generate_content(
            model=self.cloud_model,
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )

        return self._safe_json_loads(response.text)

    def _call_openai_vision_json(self, image_path, prompt):
        base64_image = self.encode_image(image_path)

        response = self.client.chat.completions.create(
            model=self.cloud_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        return self._safe_json_loads(response.choices[0].message.content)

    def _call_ollama_vision_json(self, image_path, prompt):
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path],
                }
            ],
            format="json",
        )

        raw_output = response["message"]["content"]
        return self._safe_json_loads(raw_output)

    # -------------------------------------------------------------------------
    # Main validation
    # -------------------------------------------------------------------------

    def validate_strategy(self, image_path, user_query, planner_json):
        """
        Evaluasi rencana dari Planner.

        Kontrak output:
        {
            "validation_status": "PASS" atau "FAIL",
            "feedback": "...",
            "final_action_plan": planner_json["action_plan"]  # selalu asli dari Planner
        }

        Validator tidak boleh membuat corrected plan.
        """
        original_plan = self._extract_planner_action_plan(planner_json)

        prompt = f"""
You are a strictly logical Robotic Safety Validator for a UR5e robot picking objects on a table.

Your task:
Verify whether the Planner's proposed action_plan is safe and logically valid.

IMPORTANT CONTRACT:
- You are ONLY a validator.
- You MUST NOT rewrite the action_plan.
- You MUST NOT add steps.
- You MUST NOT remove steps.
- You MUST NOT reorder steps.
- You MUST NOT create a corrected plan.
- You only return PASS or FAIL with feedback.
- The executor will use the Planner's original action_plan, not your generated plan.

Primary Target Object:
{user_query}

Planner JSON:
{json.dumps(planner_json, indent=2)}

Planner action_plan that must be judged:
{json.dumps(original_plan, indent=2)}

Validation rules:
1. Validate whether the action_plan matches the user's primary target semantically.
2. Do not fail only because of strict string mismatch.
   Example: if user says "cam" but the visible object is a "soda can", this can be valid.
3. Do not fail because of bbox, pixel coordinates, grasp_pixel_2d, 3D coordinates, or grasp-point placement.
   Coordinates are only hints and will be recomputed later by YOLO/SAM/depth.
4. A step with target_role="search_occluder" is not the final user target. It is only a temporary search/removal object.
5. If the Planner's own action_plan contains obstacle removal before picking the primary target, judge whether that sequence is safe.
6. If the Planner's own action_plan directly picks the primary target and does not include obstacle removal, do not invent an obstacle-removal step.
7. visual_analysis is context only. It is NOT an executable plan.
8. Table, countertop, counter, floor, wall, cabinet, shelf, surface, and background are support/background items and must not be treated as removable targets.
9. Return FAIL only if the Planner's existing action_plan is clearly unsafe, irrelevant, or logically impossible.
10. If the plan is reasonable, return PASS.

Output format:
Return ONLY a valid JSON object.
Do not use markdown.
Do not include corrected action_plan.
Do not include final_action_plan.

Exact schema:
{{
  "validation_status": "PASS" or "FAIL",
  "feedback": "Detailed explanation of the validation decision."
}}
"""

        if self.api_key is not None:
            print("☁️ [LogicValidator] Menggunakan mode CLOUD...")
            try:
                if self.cloud_provider == "gemini":
                    result = self._call_gemini_vision_json(image_path, prompt)
                elif self.cloud_provider == "openai":
                    result = self._call_openai_vision_json(image_path, prompt)
                else:
                    raise RuntimeError(f"Unknown cloud provider: {self.cloud_provider}")

                return self._normalize_validation_result(result, planner_json)

            except Exception as e:
                print(f"❌ Safety check gagal: {type(e).__name__}: {repr(e)}")
                traceback.print_exc()
                return self._error_result(
                    "validator_exception",
                    f"{type(e).__name__}: {str(e)}",
                    planner_json,
                )

        print(f"🔌 [LogicValidator] Menggunakan mode LOKAL ({self.model_name})...")
        try:
            result = self._call_ollama_vision_json(image_path, prompt)
            return self._normalize_validation_result(result, planner_json)

        except Exception as e:
            print(f"❌ Safety check gagal: {type(e).__name__}: {repr(e)}")
            traceback.print_exc()
            return self._error_result(
                "validator_local_exception",
                f"{type(e).__name__}: {str(e)}",
                planner_json,
            )

    # -------------------------------------------------------------------------
    # Remaining-plan validation
    # -------------------------------------------------------------------------

    def _remaining_plan_should_continue(self, remaining_plan):
        """
        Fast-path untuk kasus obstacle/search sudah berhasil dan sisa plan masih masuk akal.
        Ini tidak mengubah plan, hanya memberi izin lanjut.
        """
        if not isinstance(remaining_plan, list) or len(remaining_plan) == 0:
            return False

        for step in remaining_plan:
            if not isinstance(step, dict):
                continue

            action = str(step.get("action", "")).lower()
            explanation = str(step.get("explanation", "")).lower()
            target = str(step.get("target", "")).lower()

            if action in ["pick", "search", "find"] or any(
                keyword in explanation or keyword in target
                for keyword in ["target", "hidden", "search", "reveal", "remaining"]
            ):
                return True

        return False

    def validate_remaining_plan(self, image_path, original_query, remaining_plan):
        """
        Closed-loop validation untuk sisa antrean.

        Return tetap tuple:
            (True/False, feedback)

        Fungsi ini juga tidak boleh mengubah remaining_plan.
        """
        print("\n[Validator Engine] Memulai evaluasi keamanan sisa antrean...")

        validation_prompt = f"""
You are a strict Closed-Loop Robotic Safety Validator.

Original User Query:
{original_query}

Remaining Action Plan:
{json.dumps(remaining_plan, indent=2)}

Your task:
Evaluate whether it is safe and logical for the robot to continue executing the remaining action plan.

IMPORTANT CONTRACT:
- You are ONLY a validator.
- You MUST NOT rewrite the remaining plan.
- You MUST NOT add, remove, or reorder steps.
- Return only PASS or FAIL with feedback.

Rules:
1. Return PASS if the environment is safe, the previous action succeeded, and the remaining plan makes sense.
2. If the previous step removed an obstacle/occluder and the remaining plan is to search for or pick the original target, this is a valid continuation.
3. Do not fail because coordinates, grasp points, bbox, or pixels are wrong. Those are recomputed later.
4. Return FAIL only if something is clearly wrong or unsafe, for example the target fell off the table or the remaining action is irrelevant.

Return ONLY a valid JSON object.
Do not use markdown.

Exact schema:
{{
  "validation_status": "PASS" or "FAIL",
  "feedback": "brief explanation of your decision based on the image"
}}
"""

        if self._remaining_plan_should_continue(remaining_plan):
            return True, "Remaining plan is a valid continuation after obstacle removal/search."

        if self.api_key is not None:
            try:
                if self.cloud_provider == "gemini":
                    response = self._call_gemini_vision_json(image_path, validation_prompt)
                elif self.cloud_provider == "openai":
                    response = self._call_openai_vision_json(image_path, validation_prompt)
                else:
                    raise RuntimeError(f"Unknown cloud provider: {self.cloud_provider}")

            except Exception as e:
                traceback.print_exc()
                return False, f"Error validasi remaining plan: {type(e).__name__}: {e}"

        else:
            try:
                response = self._call_ollama_vision_json(image_path, validation_prompt)
            except Exception as e:
                traceback.print_exc()
                return False, f"Error validasi remaining plan lokal: {type(e).__name__}: {e}"

        if isinstance(response, dict):
            status = str(response.get("validation_status", "FAIL")).upper().strip()
            feedback = response.get("feedback", "Tidak ada alasan spesifik.")
            return status == "PASS", feedback

        return False, "Respons validator tidak dikenali."
