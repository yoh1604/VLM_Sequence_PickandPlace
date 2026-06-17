#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import rospy
import moveit_commander
from geometry_msgs.msg import Pose


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# OPTIONAL CAPTURE CONFIG
# ============================================================

def try_load_capture_config():
    """
    capture_config.py bersifat opsional.

    Di ROS/system Python, import capture_config.py bisa gagal kalau dependency
    seperti python-dotenv tidak ada. Karena itu script ini tetap bisa jalan
    dengan --target_json eksplisit.
    """
    try:
        sys.path.append(str(PROJECT_DIR))
        import capture_config as cfg
        return cfg
    except Exception as e:
        print("[WARN] Gagal import capture_config.py:", e)
        return None


CFG = try_load_capture_config()


def resolve_path(path_like):
    path = Path(str(path_like)).expanduser()

    if not path.is_absolute():
        path = PROJECT_DIR / path

    return path.resolve()


def get_default_target_json():
    """
    Default target JSON.

    Prioritas:
    1. capture_config.TOOL0_PREGRASP_TARGET_JSON
    2. capture_config.VISION_OUTPUT_DIR/tool0_pregrasp_target.json
    3. outputs/test_default/vision_output/tool0_pregrasp_target.json
    """
    if CFG is not None:
        if hasattr(CFG, "TOOL0_PREGRASP_TARGET_JSON"):
            return resolve_path(CFG.TOOL0_PREGRASP_TARGET_JSON)

        if hasattr(CFG, "VISION_OUTPUT_DIR"):
            return resolve_path(
                Path(CFG.VISION_OUTPUT_DIR) / "tool0_pregrasp_target.json"
            )

    return resolve_path(
        PROJECT_DIR / "outputs" / "test_default" / "vision_output" / "tool0_pregrasp_target.json"
    )


# ============================================================
# JSON LOADER
# ============================================================

def load_target(path):
    path = resolve_path(path)

    if not path.exists():
        raise FileNotFoundError(f"Target JSON tidak ditemukan: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if not data.get("success", False):
        raise RuntimeError("Target JSON success=False")

    if "translation_tool0_pregrasp" not in data:
        raise KeyError(
            "Target JSON tidak punya key 'translation_tool0_pregrasp'. "
            "Pastikan file ini hasil dari convert_gripper_tip_to_tool0_target.py."
        )

    if "quaternion_tool0_xyzw" not in data:
        raise KeyError(
            "Target JSON tidak punya key 'quaternion_tool0_xyzw'. "
            "Pastikan file ini hasil dari convert_gripper_tip_to_tool0_target.py."
        )

    pos = data["translation_tool0_pregrasp"]
    quat = data["quaternion_tool0_xyzw"]

    if len(pos) != 3:
        raise ValueError("translation_tool0_pregrasp harus berisi 3 angka.")

    if len(quat) != 4:
        raise ValueError("quaternion_tool0_xyzw harus berisi 4 angka.")

    return pos, quat, data, path


# ============================================================
# MOVEIT HELPERS
# ============================================================

def normalize_plan_result(plan_result):
    """
    MoveIt Commander ROS Noetic kadang return:
    - RobotTrajectory
    - tuple(success, trajectory, planning_time, error_code)

    Fungsi ini membuat handling lebih stabil.
    """
    if isinstance(plan_result, tuple):
        success = bool(plan_result[0])
        trajectory = plan_result[1]
        return success, trajectory

    trajectory = plan_result

    try:
        success = len(trajectory.joint_trajectory.points) > 0
    except Exception:
        success = trajectory is not None

    return success, trajectory


def print_pose(title, pose):
    print(f"\n========== {title} ==========")
    print("position:")
    print("  x:", pose.position.x)
    print("  y:", pose.position.y)
    print("  z:", pose.position.z)
    print("orientation xyzw:")
    print("  x:", pose.orientation.x)
    print("  y:", pose.orientation.y)
    print("  z:", pose.orientation.z)
    print("  w:", pose.orientation.w)
    print("=======================================\n")


def print_current_pose(group, eef_link):
    try:
        current = group.get_current_pose(end_effector_link=eef_link).pose
        print_pose(f"CURRENT {eef_link} POSE", current)
    except Exception as e:
        print(f"[WARN] Gagal membaca current pose {eef_link}:", e)


def make_pose(pos, quat):
    pose = Pose()

    pose.position.x = float(pos[0])
    pose.position.y = float(pos[1])
    pose.position.z = float(pos[2])

    pose.orientation.x = float(quat[0])
    pose.orientation.y = float(quat[1])
    pose.orientation.z = float(quat[2])
    pose.orientation.w = float(quat[3])

    return pose


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Move UR5 tool0 ke pre-grasp target dari tool0_pregrasp_target.json"
    )

    parser.add_argument(
        "--target_json",
        default=None,
        help=(
            "Path ke tool0_pregrasp_target.json. "
            "Disarankan diisi eksplisit jika beda conda/env."
        ),
    )

    parser.add_argument(
        "--group_name",
        default="manipulator",
        help="Nama MoveIt group. Default: manipulator",
    )

    parser.add_argument(
        "--reference_frame",
        default="base",
        help="Pose reference frame. Default: base",
    )

    parser.add_argument(
        "--eef_link",
        default="tool0",
        help="End-effector link yang digerakkan. Default: tool0",
    )

    parser.add_argument(
        "--velocity",
        type=float,
        default=0.10,
        help="Max velocity scaling factor. Default: 0.10",
    )

    parser.add_argument(
        "--acceleration",
        type=float,
        default=0.10,
        help="Max acceleration scaling factor. Default: 0.10",
    )

    parser.add_argument(
        "--planning_time",
        type=float,
        default=10.0,
        help="Planning time in seconds. Default: 10.0",
    )

    parser.add_argument(
        "--planning_attempts",
        type=int,
        default=10,
        help="Number of planning attempts. Default: 10",
    )

    parser.add_argument(
        "--position_tolerance",
        type=float,
        default=0.01,
        help="Goal position tolerance in meters. Default: 0.01",
    )

    parser.add_argument(
        "--orientation_tolerance",
        type=float,
        default=0.05,
        help="Goal orientation tolerance in radians. Default: 0.05",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Kalau flag ini diberikan, robot langsung execute tanpa menunggu ENTER. "
            "Untuk test awal sebaiknya jangan pakai flag ini."
        ),
    )

    args = parser.parse_args()

    target_json = (
        resolve_path(args.target_json)
        if args.target_json is not None
        else get_default_target_json()
    )

    print("\n========== MOVE TO TOOL0 PREGRASP ==========")
    print("PROJECT_DIR:", PROJECT_DIR)

    if CFG is not None:
        print("capture_config: LOADED")
        print("capture_config TEST_NAME:", getattr(CFG, "TEST_NAME", "N/A"))
        print("capture_config VISION_OUTPUT_DIR:", getattr(CFG, "VISION_OUTPUT_DIR", "N/A"))
    else:
        print("capture_config: NOT LOADED")

    print("TARGET_JSON:", target_json)
    print("group_name:", args.group_name)
    print("reference_frame:", args.reference_frame)
    print("eef_link:", args.eef_link)
    print("velocity:", args.velocity)
    print("acceleration:", args.acceleration)
    print("===========================================\n")

    # ------------------------------------------------------------
    # Init ROS + MoveIt
    # ------------------------------------------------------------

    rospy.init_node("move_to_tool0_pregrasp", anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)

    robot = moveit_commander.RobotCommander()
    group = moveit_commander.MoveGroupCommander(args.group_name)

    group.set_pose_reference_frame(args.reference_frame)
    group.set_end_effector_link(args.eef_link)

    group.set_max_velocity_scaling_factor(float(args.velocity))
    group.set_max_acceleration_scaling_factor(float(args.acceleration))

    group.set_planning_time(float(args.planning_time))
    group.set_num_planning_attempts(int(args.planning_attempts))

    group.set_goal_position_tolerance(float(args.position_tolerance))
    group.set_goal_orientation_tolerance(float(args.orientation_tolerance))

    print("[INFO] Robot groups:", robot.get_group_names())
    print("[INFO] Planning frame:", group.get_planning_frame())
    print("[INFO] End effector link:", group.get_end_effector_link())

    print_current_pose(group, args.eef_link)

    # ------------------------------------------------------------
    # Load target JSON
    # ------------------------------------------------------------

    pos, quat, data, loaded_path = load_target(target_json)

    print("\n========== TARGET JSON LOADED ==========")
    print("Path:", loaded_path)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print("=======================================\n")

    target_pose = make_pose(pos, quat)
    print_pose("TARGET TOOL0 PREGRASP POSE", target_pose)

    # ------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------

    group.clear_pose_targets()
    group.set_pose_target(target_pose, end_effector_link=args.eef_link)

    print("[PLAN] Planning to tool0 pre-grasp...")
    plan_result = group.plan()

    success, trajectory = normalize_plan_result(plan_result)

    print("[PLAN] success:", success)

    if not success:
        group.clear_pose_targets()
        raise RuntimeError("Planning gagal. Robot tidak dieksekusi.")

    try:
        n_points = len(trajectory.joint_trajectory.points)
        print("[PLAN] trajectory points:", n_points)
    except Exception:
        pass

    # ------------------------------------------------------------
    # Execute confirmation
    # ------------------------------------------------------------

    print("\nPERIKSA SEBELUM EXECUTE:")
    print("- Ini hanya PRE-GRASP, bukan turun ke objek.")
    print("- Target yang dikirim adalah tool0, sudah dikompensasi dari gripper_tip.")
    print("- Pastikan robot, meja, objek, dan manusia aman.")
    print("- Setelah execute, ujung gripper seharusnya berada di atas objek.")
    print()

    if not args.execute:
        input("Tekan ENTER untuk EXECUTE, atau CTRL+C untuk batal... ")

    print("[EXECUTE] Moving to tool0 pre-grasp...")
    ok = group.execute(trajectory, wait=True)

    group.stop()
    group.clear_pose_targets()

    print("[EXECUTE] result:", ok)

    print_current_pose(group, args.eef_link)

    print("\n========== SELESAI ==========")
    print("Cek fisik:")
    print("1. Apakah ujung gripper berada di atas objek?")
    print("2. Apakah jaraknya sesuai pregrasp_z, misalnya 10 cm?")
    print("3. Kalau terlalu rendah, ulang convert dengan --pregrasp_z 0.15.")
    print("4. Kalau XY meleset, cek offset tool0_to_gripper_tip dan scene terbaru.")
    print("=============================\n")


if __name__ == "__main__":
    main()