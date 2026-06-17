import os
import json
import subprocess
from pathlib import Path


class GraspNetRunner:
    """
    Runner untuk menjalankan GraspNet baseline dari environment terpisah.

    Environment utama:
    - ur5_pickplace

    Environment GraspNet:
    - anygrasp_py310

    Output:
    - best_grasp_camera.json
    - grasp_candidates_camera.json
    """

    def __init__(
        self,
        project_dir,
        conda_env_name="anygrasp_py310",
        graspnet_dir=None,
        checkpoint_path=None,
    ):
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.conda_env_name = conda_env_name

        if graspnet_dir is None:
            self.graspnet_dir = self.project_dir / "models" / "graspnet-baseline"
        else:
            self.graspnet_dir = Path(graspnet_dir).expanduser().resolve()

        if checkpoint_path is None:
            self.checkpoint_path = self.graspnet_dir / "logs" / "log_rs" / "checkpoint.tar"
        else:
            self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()

        self.demo_script = self.graspnet_dir / "demo_d455.py"

    def check_paths(self):
        if not self.graspnet_dir.exists():
            raise FileNotFoundError(f"Folder GraspNet tidak ditemukan: {self.graspnet_dir}")

        if not self.demo_script.exists():
            raise FileNotFoundError(f"demo_d455.py tidak ditemukan: {self.demo_script}")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint GraspNet tidak ditemukan: {self.checkpoint_path}")

    def run(
        self,
        rgb_path,
        depth_path,
        intrinsics_path,
        mask_path,
        output_dir,
        num_point=10000,
        num_view=300,
        collision_thresh=-1,
        voxel_size=0.01,
        max_center_dist=0.10,
        mask_dilate_iter=1,
        no_vis=True,
        timeout_sec=180,
    ):
        self.check_paths()

        rgb_path = Path(rgb_path).expanduser().resolve()
        depth_path = Path(depth_path).expanduser().resolve()
        intrinsics_path = Path(intrinsics_path).expanduser().resolve()
        mask_path = Path(mask_path).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()

        output_dir.mkdir(parents=True, exist_ok=True)

        best_output = output_dir / "best_grasp_camera.json"
        candidates_output = output_dir / "grasp_candidates_camera.json"

        required_files = {
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "intrinsics_path": intrinsics_path,
            "mask_path": mask_path,
        }

        for name, path in required_files.items():
            if not path.exists():
                raise FileNotFoundError(f"{name} tidak ditemukan: {path}")

        graspnet_python = "/home/b401/miniconda3/envs/anygrasp_py310/bin/python"

        cmd = [
            graspnet_python,
            str(self.demo_script),
            "--checkpoint_path",
            str(self.checkpoint_path),
            "--rgb_path",
            str(rgb_path),
            "--depth_path",
            str(depth_path),
            "--intrinsics_path",
            str(intrinsics_path),
            "--mask_path",
            str(mask_path),
            "--output_dir",
            str(output_dir),
            "--best_output",
            str(best_output),
            "--candidates_output",
            str(candidates_output),
            "--num_point",
            str(num_point),
            "--num_view",
            "300",
            "--collision_thresh",
            str(collision_thresh),
            "--voxel_size",
            str(voxel_size),
            "--max_center_dist",
            str(max_center_dist),
            "--mask_dilate_iter",
            str(mask_dilate_iter),
        ]
        # cmd = [
        #     "conda",
        #     "run",
        #     "-n",
        #     self.conda_env_name,
        #     "python",
        #     str(self.demo_script),
        #     "--checkpoint_path",
        #     str(self.checkpoint_path),
        #     "--rgb_path",
        #     str(rgb_path),
        #     "--depth_path",
        #     str(depth_path),
        #     "--intrinsics_path",
        #     str(intrinsics_path),
        #     "--mask_path",
        #     str(mask_path),
        #     "--output_dir",
        #     str(output_dir),
        #     "--best_output",
        #     str(best_output),
        #     "--candidates_output",
        #     str(candidates_output),
        #     "--num_point",
        #     str(num_point),
        #     "--num_view",
        #     str(num_view),
        #     "--collision_thresh",
        #     str(collision_thresh),
        #     "--voxel_size",
        #     str(voxel_size),
        #     "--max_center_dist",
        #     str(max_center_dist),
        #     "--mask_dilate_iter",
        #     str(mask_dilate_iter),
        # ]

        if no_vis:
            cmd.append("--no_vis")

        env = os.environ.copy()

        graspnet_env = Path("/home/b401/miniconda3/envs/anygrasp_py310")
        torch_lib = graspnet_env / "lib" / "python3.10" / "site-packages" / "torch" / "lib"
        conda_lib = graspnet_env / "lib"
        conda_target_lib = graspnet_env / "targets" / "x86_64-linux" / "lib"

        old_ld = env.get("LD_LIBRARY_PATH", "")

        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["CUDA_HOME"] = str(graspnet_env)
        env["PATH"] = f"{graspnet_env / 'bin'}:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = (
        f"{torch_lib}:"
        f"{conda_lib}:"
        f"{conda_target_lib}:"
        f"{old_ld}"
        )
        # env = os.environ.copy()

        # # Penting untuk extension PyTorch/knn/pointnet2.
        # torch_lib = (
        #     Path.home()
        #     / "miniconda3"
        #     / "envs"
        #     / self.conda_env_name
        #     / "lib"
        #     / "python3.10"
        #     / "site-packages"
        #     / "torch"
        #     / "lib"
        # )

        # conda_lib = Path.home() / "miniconda3" / "envs" / self.conda_env_name / "lib"
        # conda_target_lib = (
        #     Path.home()
        #     / "miniconda3"
        #     / "envs"
        #     / self.conda_env_name
        #     / "targets"
        #     / "x86_64-linux"
        #     / "lib"
        # )

        # old_ld = env.get("LD_LIBRARY_PATH", "")
        # env["LD_LIBRARY_PATH"] = (
        #     f"{torch_lib}:{conda_lib}:{conda_target_lib}:{old_ld}"
        # )

        cuda_check_cmd = [
            graspnet_python,
            "-c",
            (
                "import torch; "
                "print('SUBPROCESS torch:', torch.__version__); "
                "print('SUBPROCESS cuda:', torch.cuda.is_available()); "
                "print('SUBPROCESS device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU_ONLY')"
            )
        ]

        cuda_check = subprocess.run(
            cuda_check_cmd,
            cwd=str(self.graspnet_dir),
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )

        print("\n========== GRASPNET CUDA CHECK ==========")
        print(cuda_check.stdout)
        print(cuda_check.stderr)
        print("=========================================\n")

        print("\n========== RUN GRASPNET SUBPROCESS ==========")
        print("Command:")
        print(" ".join(cmd))
        print("Working dir:", self.graspnet_dir)
        print("Output best:", best_output)
        print("============================================\n")

        result = subprocess.run(
            cmd,
            cwd=str(self.graspnet_dir),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )

        print("\n========== GRASPNET STDOUT ==========")
        print(result.stdout)

        print("\n========== GRASPNET STDERR ==========")
        print(result.stderr)

        if result.returncode != 0:
            raise RuntimeError(
                f"GraspNet gagal dengan return code {result.returncode}."
            )

        if not best_output.exists():
            raise FileNotFoundError(
                f"GraspNet selesai tapi best_grasp_camera.json tidak ditemukan: {best_output}"
            )

        with open(best_output, "r") as f:
            best_grasp = json.load(f)

        print("\n========== GRASPNET RESULT ==========")
        print(json.dumps(best_grasp, indent=2, ensure_ascii=False))
        print("====================================\n")

        return best_grasp