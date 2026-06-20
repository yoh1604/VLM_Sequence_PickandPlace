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


class VLMEngine:
    """
    VLM Planner untuk membuat action_plan robot.

    Provider yang didukung:
    - Gemini: provider="gemini"
    - OpenAI: provider="openai"
    - Auto: provider="auto"
    - Local Ollama: pakai model_name, tanpa api_key

    Catatan penting:
    - Planner membuat action_plan.
    - Validator hanya memverifikasi PASS/FAIL dan tidak boleh mengubah action_plan.
    - Nama target harus YOLO-World friendly: pendek, visual, dan konsisten.
    """

    BACKGROUND_TARGET_KEYWORDS = [
        "table",
        "counter",
        "countertop",
        "surface",
        "floor",
        "wall",
        "background",
        "kitchen island",
        "shelf",
        "cabinet",
    ]

    def __init__(self, api_key=None, model_name=None, provider=None):
        self.api_key = api_key
        self.model_name = model_name
        self.cloud_provider = None
        self.provider = (provider or "auto").lower()
        self.client = None

        if self.model_name:
            self.mode = "LOCAL"
            print(f"VLM Planner berjalan di mode LOKAL dengan model: {self.model_name}")
        elif self.api_key:
            self.mode = "CLOUD"
            print("VLM Planner berjalan di mode CLOUD")
        else:
            raise ValueError("Harus memasukkan api_key atau model_name")

        if self.api_key is not None:
            if self.provider in ["gemini", "google", "gcp", "auto"] and genai is not None:
                try:
                    self.client = genai.Client(api_key=self.api_key)
                    self.cloud_model = "gemini-2.5-pro"
                    self.cloud_provider = "gemini"
                    print("☁️ VLM Planner diinisialisasi untuk mode CLOUD (Gemini).")
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
                print("☁️ VLM Planner diinisialisasi untuk mode CLOUD (OpenAI).")
        else:
            print(f"🔌 VLM Planner diinisialisasi untuk mode LOKAL (Model: {self.model_name}).")

    # -------------------------------------------------------------------------
    # Basic utilities
    # -------------------------------------------------------------------------

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _safe_json_loads(self, raw_text):
        if isinstance(raw_text, dict):
            return raw_text

        if raw_text is None:
            raise ValueError("Empty VLM response")

        text = str(raw_text).strip()
        text = text.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if not json_match:
                raise
            return json.loads(json_match.group(0))

    def _is_background_target(self, target_name):
        target_lower = str(target_name or "").lower()
        return any(keyword in target_lower for keyword in self.BACKGROUND_TARGET_KEYWORDS)

    def _normalize_object_name(self, name):
        """
        Normalisasi ringan agar nama target lebih stabil untuk YOLO-World.
        Tidak mengubah semantik besar; hanya membersihkan casing/spasi.
        """
        if name is None:
            return ""

        cleaned = str(name).strip().lower()
        cleaned = re.sub(r"[^a-z0-9\s\-]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _normalize_search_plan(self, planner_json):
        """
        Normalisasi hasil Planner:
        - Hapus action step yang menargetkan background/support surface.
        - Pastikan target/action_plan memakai nama bersih.
        - Pastikan metadata role/intent minimal ada.
        """
        if not isinstance(planner_json, dict):
            return planner_json

        if "resolved_primary_target" in planner_json:
            planner_json["resolved_primary_target"] = self._normalize_object_name(
                planner_json.get("resolved_primary_target")
            )

        visual_analysis = planner_json.get("visual_analysis", [])
        if isinstance(visual_analysis, list):
            for item in visual_analysis:
                if not isinstance(item, dict):
                    continue

                if "object" in item:
                    item["object"] = self._normalize_object_name(item.get("object"))

                if self._is_background_target(item.get("object", "")):
                    item["status"] = "clear_background"
                    item["target_role"] = "background"

                bbox = item.get("bbox")
                if not (
                    isinstance(bbox, list)
                    and len(bbox) == 4
                    and all(isinstance(v, (int, float)) for v in bbox)
                ):
                    item["bbox"] = [0, 0, 0, 0]
                else:
                    item["bbox"] = [
                        max(0, min(1000, int(round(v))))
                        for v in bbox
                    ]

        action_plan = planner_json.get("action_plan")
        if not isinstance(action_plan, list):
            planner_json["action_plan"] = []
            return planner_json

        filtered_plan = []
        for step in action_plan:
            if not isinstance(step, dict):
                continue

            action = self._normalize_object_name(step.get("action", ""))
            target = self._normalize_object_name(step.get("target", ""))

            step["action"] = action
            step["target"] = target

            if action in ["pick", "remove", "clear"] and self._is_background_target(target):
                print(
                    f"⚠️ Menghapus langkah tidak valid: robot tidak boleh memindahkan "
                    f"background/support surface '{target}'."
                )
                continue

            grasp = step.get("grasp_pixel_2d")
            if not (
                isinstance(grasp, list)
                and len(grasp) == 2
                and all(isinstance(v, (int, float)) for v in grasp)
            ):
                step["grasp_pixel_2d"] = [500, 500]
            else:
                step["grasp_pixel_2d"] = [
                    max(0, min(1000, int(round(v))))
                    for v in grasp
                ]

            filtered_plan.append(step)

        planner_json["action_plan"] = filtered_plan
        action_plan = filtered_plan

        has_clearing_step = any(
            isinstance(step, dict)
            and str(step.get("action", "")).lower() in ["remove", "clear"]
            for step in action_plan
        )

        if not has_clearing_step:
            planner_json["plan_intent"] = "pick_target"
            for step in action_plan:
                if isinstance(step, dict):
                    step.setdefault("target_role", "primary_target")
                    step.setdefault("step_intent", "pick_primary_target")
            return planner_json

        planner_json["plan_intent"] = "search_for_hidden_or_blocked_target"

        final_primary_pick_index = None
        for index in range(len(action_plan) - 1, -1, -1):
            step = action_plan[index]
            if isinstance(step, dict) and str(step.get("action", "")).lower() in ["pick", "search", "find"]:
                final_primary_pick_index = index
                break

        for index, step in enumerate(action_plan):
            if not isinstance(step, dict):
                continue

            action = str(step.get("action", "")).lower()
            if index == final_primary_pick_index:
                step.setdefault("target_role", "primary_target")
                step.setdefault("step_intent", "pick_or_search_primary_target")
            elif action in ["pick", "remove", "clear"]:
                step.setdefault("target_role", "search_occluder")
                step.setdefault("step_intent", "clear_occluder_to_reveal_primary_target")

        return planner_json

    # -------------------------------------------------------------------------
    # Prompt builder
    # -------------------------------------------------------------------------

    def _build_planner_prompt(self, user_query, local=False):
        """
        Prompt ini dibuat agar output target lebih reliable untuk YOLO-World:
        - nama objek pendek
        - warna + kategori
        - hindari frasa panjang/relatif
        - action_plan target harus sama persis dengan visual_analysis.object
        """
        ollama_rules = ""
        if local:
            ollama_rules = """
LOCAL MODEL STRICTNESS:
- You are an automated JSON-only API.
- Do not include conversational text.
- Do not wrap JSON in markdown.
- Start exactly with "{" and end exactly with "}".
"""

        prompt = f"""
You are an expert robotic vision planner for a UR5e robot.
Your job is to inspect the image and create a safe JSON action_plan.
Look at the lowest point of the objects in the image. Objects whose bottom edges are lower down are physically closer to the camera and must be cleared first if they overlap target behind them."

Primary user request:
"{user_query}"

{ollama_rules}

CRITICAL OUTPUT CONTRACT:
- Output only valid JSON.
- Do not use markdown.
- Do not include comments in JSON.
- The executable command source is action_plan only.
- visual_analysis is context only.
- The validator will not rewrite your action_plan, so your action_plan must be correct.

YOLO-WORLD NAMING RULES FOR RELIABLE DETECTION:
1. Every executable target name must be easy for YOLO-World to detect.
2. Use short visual noun phrases: color + generic object category.
3. Prefer 2 to 4 words only.
4. Use lowercase names.
5. Avoid brand names unless the brand text is clearly visible and useful.
6. Avoid vague names such as "object", "item", "thing", "stuff", "container" alone.
7. Avoid relative phrases such as "left object", "front object", "thing near lemon".
8. Avoid overly specific descriptions that YOLO-World may not understand.
9. Good names:
   - "yellow lemon"
   - "red soda can"
   - "white milk carton"
   - "green bottle"
   - "brown bread"
   - "blue cup"
   - "black bowl"
   - "silver spoon"
10. Bad names:
   - "target object"
   - "obstacle"
   - "front occluder"
   - "red cylindrical container with label"
   - "the thing in front of the lemon"
11. The string in action_plan[i].target MUST exactly match one object string in visual_analysis[j].object, except when the target is hidden.
12. If the user says a generic target like "lemon", but the visible object is yellow, output "yellow lemon".
13. If the user says "can", output a visually grounded name like "red soda can" if that is the visible object.
14. If a target is hidden, still use the best YOLO-friendly target name inferred from the user query, e.g. "yellow lemon", not "hidden target".

SCENE ANALYSIS RULES:
1. List every visible tabletop object in visual_analysis.
2. Include support/background objects only when useful, and mark them as:
   "status": "clear_background",
   "target_role": "background"
3. Table, countertop, counter, floor, wall, cabinet, shelf, surface, and background are never executable targets.
4. Do not put background/support objects in action_plan.
5. Look at the lowest point of the objects in the image. Objects whose bottom edges are lower down are physically closer to the camera and must be cleared first ONLY IF they overlap the target (user_query) behind them."

TARGET IDENTIFICATION RULES:
1. Resolve the user's request into one concrete object name from the scene when possible.
2. If the user query has a typo, correct it using visible objects.
   Example: "cam" can mean "can" if a soda can is visible.
3. If the user query expresses intent, map it to a logical visible object.
   Example: "I am thirsty" -> "red soda can", "water bottle", or "white milk carton".
4. resolved_primary_target must be the YOLO-friendly object name.
5. The final primary target step must use the same string as resolved_primary_target whenever the target is visible.

OBSTRUCTION RULES:
1. If the primary target is clear, action_plan MUST contain exactly one step:
   pick the primary target.
2. Do not add obstacle-removal steps unless an object physically blocks robot access to the primary target.
3. Treat the primary target as obstructed if another object covers or overlaps the lower part, side part, or front surface area of the primary target.
4. An obstacle must physically overlap, cover, or block the reachable grasp area/path to the primary target.
5. If the target is obstructed, use a sequence:
   - pick the occluding object
   - remove the occluding object
   - pick or search/pick the primary target
6. If the target is hidden, plan a search/removal sequence using the most likely visible occluder, then include a final step to find/pick the original primary target.
7. target_role:
   - "primary_target" only for the object requested by the user
   - "search_occluder" only for objects moved to reveal/search for the primary target
   - "background" only for non-executable background/support surfaces
8. plan_intent:
   - "pick_target" if directly picking the primary target
   - "search_for_hidden_or_blocked_target" if moving occluders first

MULTI-OCCLUDER / ACCESS PATH RULES:
1. The robot approaches the scene from the front, similar to the camera viewpoint.
2. A target is not reachable if there are objects in the frontal access path between the robot/camera and the primary target.
3. If multiple objects form a blocking chain in front of the primary target, remove them from front to back.
4. Do not stop after removing only the nearest/front object if another object still blocks the primary target's lower, central, side, or graspable body region.
5. For a bottle-shaped target, the lower or central body is usually the graspable region. If this region is blocked by a can, box, fruit, or another object, remove that object before picking the bottle.
6. If a small object is in front of another occluding object and blocks the robot's access path to that occluder or target, remove the small front object first.
7. The correct sequence is:
   * remove the nearest/front occluder in the access path
   * remove the next object that still blocks the primary target
   * pick the primary target only after its graspable region is reachable
8. If the primary target is a water bottle and the scene contains a yellow lemon in front, a red soda can behind/near it, and the red soda can blocks the bottle body, the action_plan should be:
   * pick/remove yellow lemon
   * pick/remove red soda can
   * pick blue water bottle

   
   FORWARD-LOOKING CAMERA REASONING:
1. The camera observes the scene from the front, not from top-down.
2. In a forward-looking view, an object located lower in the image may be physically in front of another object.
3. If a lower/front object overlaps the bottom part of the requested target, it likely blocks the robot's frontal approach or grasp access.
4. The planner must reason about occlusion using image overlap and visible object ordering:
   - front/lower object = possible occluder
   - rear/upper object = possible target behind it
5. If the target is behind another object and the front object blocks the target's lower or central grasp region, remove the front object first.
6. Look at the lowest point of the objects in the image. Objects whose bottom edges are lower down are physically closer to the camera and must be cleared first if they overlap target behind them.
7. The robotic arm approaches from the front. Never plan a grasp on an object if another object stands taller or overlap 3/4 part and directly in front of it, even if the top of the rear object is visible.

STRICT JSON SCHEMA:
{{
  "resolved_primary_target": "yolo_friendly_object_name",
  "target_visibility": "clear / obstructed / not_exist / hidden",
  "plan_intent": "pick_target / search_for_hidden_or_blocked_target",
  "visual_analysis": [
    {{
      "object": "color generic_object",
      "description": "short visual description",
      "status": "target / obstructing / clear / clear_background",
      "target_role": "primary_target / search_occluder / background"
    }}
  ],
  "action_plan": [
    {{
      "step": 1,
      "action": "pick / remove / search",
      "target": "must_exactly_match_visual_analysis_object_unless_hidden",
      "target_role": "primary_target / search_occluder",
      "step_intent": "pick_primary_target / clear_occluder_to_reveal_primary_target / pick_or_search_primary_target",
      "explanation": "brief reason",
    }}
  ]
}}

FINAL SELF-CHECK BEFORE OUTPUT:
- Is every action_plan target YOLO-friendly?
- Does every visible action_plan target exactly match a visual_analysis object?
- If the target is clear and its graspable lower/front body is accessible, is action_plan exactly one step?
- If the lower/front/graspable part of the primary target is blocked by object A, did you remove object A before picking the target?
- Did you avoid using table/counter/background as an executable target?
- Is the JSON valid?
"""
        return prompt

    # -------------------------------------------------------------------------
    # Model callers
    # -------------------------------------------------------------------------

    def _call_gemini_vision_json(self, image_path, prompt):
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
                temperature=0.1,
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
            temperature=0.1,
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

        return self._safe_json_loads(response["message"]["content"])

    # -------------------------------------------------------------------------
    # Public APIs
    # -------------------------------------------------------------------------

    def get_strategy_local(self, img_path, user_query):
        print("🧠 Menghubungi Ollama VLM Lokal...")

        prompt = self._build_planner_prompt(user_query, local=True)

        try:
            parsed_data = self._call_ollama_vision_json(img_path, prompt)
            return self._normalize_search_plan(parsed_data)

        except Exception as e:
            print(f"❌ Error saat memanggil Ollama: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None

    def get_strategy(self, img_path, user_query):
        print("🧠 Menghubungi VLM Planner CLOUD...")

        prompt = self._build_planner_prompt(user_query, local=False)

        try:
            if self.cloud_provider == "gemini":
                parsed_data = self._call_gemini_vision_json(img_path, prompt)
            elif self.cloud_provider == "openai":
                parsed_data = self._call_openai_vision_json(img_path, prompt)
            else:
                return {"error": f"Unknown cloud provider: {self.cloud_provider}"}

            print(f"[DEBUG] Raw Parsed Response: {json.dumps(parsed_data, indent=2)}")
            return self._normalize_search_plan(parsed_data)

        except Exception as e:
            print(f"❌ Error saat memanggil VLM Cloud: {type(e).__name__}: {e}")
            traceback.print_exc()
            return {"error": f"{type(e).__name__}: {str(e)}"}
