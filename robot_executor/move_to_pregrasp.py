#!/usr/bin/env python3

import sys
import json
import argparse
from pathlib import Path

import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped


PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_WAYPOINT_FILE = PROJECT_DIR / "configs" / "waypoints_ur5.json"
DEFAULT_GRASP_FILE = (
    PROJECT_DIR
    / "outputs"
    / "test_grasp"
    / "vision_output"
    / "best_grasp_base.json"
)


def resolve_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


def load_json(path, name="JSON"):
    path = resolve_path(path)

    if not path.exists():
        raise FileNotFoundError(f"{name} tidak ditemukan: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    return data, path


def get_plan_result(plan):
    """
    Kompatibel dengan beberapa versi MoveIt Noetic.
    group.plan() bisa return tuple atau RobotTrajectory.
    """
    if isinstance(plan, tuple):
        success = bool(plan[0])
        trajectory = plan[1]
    else:
        trajectory = plan
        success = (
            hasattr(trajectory, "joint_trajectory")
            and len(trajectory.joint_trajectory.points) > 0
        )

    return success, trajectory


def print_robot_info(group, robot):
    print("\n========== MOVEIT INFO ==========")
    print("Planning frame:", group.get_planning_frame())
    print("End effector link:", group.get_end_effector_link())
    print("Available groups:", robot.get_group_names())
    print("Active joints:", group.get_active_joints())
    print("Current joints:", [round(x, 6) for x in group.get_current_joint_values()])

    pose = group.get_current_pose().pose
    print(
        "Current pose position:",
        round(pose.position.x, 6),
        round(pose.position.y, 6),
        round(pose.position.z, 6),
    )
    print(
        "Current pose orientation:",
        round(pose.orientation.x, 6),
        round(pose.orientation.y, 6),
        round(pose.orientation.z, 6),
        round(pose.orientation.w, 6),
    )
    print("================================\n")


def print_pose(label, pose):
    print(f"\n========== {label} ==========")
    print(
        "position:",
        pose.position.x,
        pose.position.y,
        pose.position.z,
    )
    print(
        "orientation:",
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    print("==============================\n")


def move_joint_waypoint(group, waypoint_name, joints, execute=False):
    """
    Joint waypoint movement.
    Saat execute=True, pakai group.go(joints, wait=True), seperti go_to_idle.py.
    Saat execute=False, hanya planning.
    """
    if len(joints) != 6:
        raise ValueError(
            f"Waypoint {waypoint_name} harus punya 6 joint, sekarang {len(joints)}"
        )

    print(f"\n========== JOINT TARGET: {waypoint_name} ==========")
    print("Target joints:", joints)
    print("Execute:", execute)
    print("=================================================\n")

    group.clear_pose_targets()

    if execute:
        rospy.loginfo(f"Moving to {waypoint_name}: {joints}")

        success = group.go(joints, wait=True)

        group.stop()
        group.clear_pose_targets()

        print(f"[EXECUTE] {waypoint_name} success:", success)

        if not success:
            raise RuntimeError(f"Execute ke {waypoint_name} gagal.")

        rospy.sleep(0.5)
        return True

    group.set_joint_value_target(joints)

    plan = group.plan()
    success, _ = get_plan_result(plan)

    print(f"[PLAN ONLY] {waypoint_name} success:", success)

    if not success:
        raise RuntimeError(f"Planning ke {waypoint_name} gagal.")

    return True


def load_pregrasp_position(grasp_json):
    grasp_data, grasp_path = load_json(grasp_json, "BEST_GRASP_BASE_JSON")

    if not grasp_data.get("success", False):
        raise RuntimeError("best_grasp_base.json success=False")

    if "pre_grasp_preview" in grasp_data:
        p = grasp_data["pre_grasp_preview"]["translation_base"]
        print("[INFO] Menggunakan pre_grasp_preview.translation_base dari JSON.")
    else:
        p = grasp_data["translation_base"]
        p = [p[0], p[1], p[2] + 0.10]
        print("[WARN] pre_grasp_preview tidak ada. Pakai translation_base + 0.10 m.")

    if len(p) != 3:
        raise ValueError("pre_grasp position harus 3 angka.")

    print("[INFO] Grasp file:", grasp_path)
    print("[INFO] Pre-grasp position:", p)

    return [float(p[0]), float(p[1]), float(p[2])]


def plan_or_execute_pregrasp_fixed_orientation(
    group,
    pregrasp_position,
    orientation_source_pose,
    base_frame,
    execute=False,
):
    """
    Dynamic pre-grasp:
    - position dari best_grasp_base.json
    - orientation dari pose TCP setelah OBSERVATION

    Ini mencegah wrist bebas naik seperti position-only target.
    """
    target = PoseStamped()
    target.header.frame_id = base_frame
    target.header.stamp = rospy.Time.now()

    target.pose.position.x = float(pregrasp_position[0])
    target.pose.position.y = float(pregrasp_position[1])
    target.pose.position.z = float(pregrasp_position[2])

    target.pose.orientation = orientation_source_pose.orientation

    print("\n========== PRE-GRASP POSE TARGET ==========")
    print("Frame:", base_frame)
    print(
        "Target position:",
        target.pose.position.x,
        target.pose.position.y,
        target.pose.position.z,
    )
    print(
        "Target orientation from OBSERVATION/current TCP:",
        target.pose.orientation.x,
        target.pose.orientation.y,
        target.pose.orientation.z,
        target.pose.orientation.w,
    )
    print("Execute:", execute)
    print("==========================================\n")

    group.clear_pose_targets()
    group.set_pose_reference_frame(base_frame)

    # Toleransi dibuat agak longgar untuk test awal.
    group.set_goal_position_tolerance(0.015)
    group.set_goal_orientation_tolerance(0.25)

    group.set_pose_target(target)

    plan = group.plan()
    success, trajectory = get_plan_result(plan)

    print("[PLAN] pre_grasp fixed-orientation success:", success)

    if not success:
        raise RuntimeError(
            "Planning ke pre_grasp fixed-orientation gagal. "
            "Coba naikkan z pre-grasp, longgarkan tolerance, atau pakai --skip_idle."
        )

    if execute:
        print("[EXECUTE] Moving to pre_grasp fixed-orientation ...")

        ok = group.execute(trajectory, wait=True)

        group.stop()
        group.clear_pose_targets()

        print("[EXECUTE] pre_grasp result:", ok)

        if not ok:
            raise RuntimeError("Execute ke pre_grasp gagal.")

        rospy.sleep(0.5)

    else:
        print("[PLAN ONLY] pre_grasp fixed-orientation. Tidak dieksekusi.")

    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--move_group", default="manipulator")
    parser.add_argument("--base_frame", default="base")
    parser.add_argument("--waypoints", default=str(DEFAULT_WAYPOINT_FILE))
    parser.add_argument("--grasp_base_json", default=str(DEFAULT_GRASP_FILE))

    parser.add_argument("--idle_name", default="IDLE")
    parser.add_argument("--observation_name", default="OBSERVATION")

    parser.add_argument(
        "--skip_idle",
        action="store_true",
        help="Lewati gerak ke IDLE. Default: current -> IDLE -> OBSERVATION -> pre_grasp.",
    )

    parser.add_argument(
        "--skip_observation",
        action="store_true",
        help="Lewati gerak ke OBSERVATION. Tidak disarankan.",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Kalau tidak diberi flag ini, script hanya plan-only.",
    )

    parser.add_argument("--velocity_scale", type=float, default=0.05)
    parser.add_argument("--acceleration_scale", type=float, default=0.05)

    args = parser.parse_args()

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("move_to_pregrasp_node", anonymous=True)

    robot = moveit_commander.RobotCommander()
    group = moveit_commander.MoveGroupCommander(args.move_group)

    group.set_max_velocity_scaling_factor(args.velocity_scale)
    group.set_max_acceleration_scaling_factor(args.acceleration_scale)
    group.set_planning_time(15.0)
    group.set_num_planning_attempts(20)
    group.allow_replanning(True)

    print_robot_info(group, robot)

    waypoints, waypoint_path = load_json(args.waypoints, "WAYPOINTS_JSON")
    print("[INFO] Waypoint file:", waypoint_path)

    if args.idle_name not in waypoints:
        raise KeyError(f"Waypoint {args.idle_name} tidak ditemukan di {waypoint_path}")

    if args.observation_name not in waypoints:
        raise KeyError(
            f"Waypoint {args.observation_name} tidak ditemukan di {waypoint_path}"
        )

    print("[INFO] IDLE loaded:", waypoints[args.idle_name])
    print("[INFO] OBSERVATION loaded:", waypoints[args.observation_name])

    # 1. current -> IDLE
    if not args.skip_idle:
        move_joint_waypoint(
            group=group,
            waypoint_name=args.idle_name,
            joints=waypoints[args.idle_name],
            execute=args.execute,
        )
    else:
        print("[INFO] Skip IDLE.")

    # 2. IDLE/current -> OBSERVATION
    if not args.skip_observation:
        move_joint_waypoint(
            group=group,
            waypoint_name=args.observation_name,
            joints=waypoints[args.observation_name],
            execute=args.execute,
        )
    else:
        print("[WARN] Skip OBSERVATION. Orientation source dari current TCP.")

    # 3. Ambil orientation dari pose setelah OBSERVATION.
    rospy.sleep(0.5)
    observation_pose = group.get_current_pose().pose
    print_pose("ORIENTATION SOURCE POSE", observation_pose)

    # 4. Load posisi pre-grasp dari best_grasp_base.json.
    pregrasp_position = load_pregrasp_position(args.grasp_base_json)

    # 5. Plan / execute ke dynamic pre-grasp dengan orientasi OBSERVATION.
    plan_or_execute_pregrasp_fixed_orientation(
        group=group,
        pregrasp_position=pregrasp_position,
        orientation_source_pose=observation_pose,
        base_frame=args.base_frame,
        execute=args.execute,
    )

    print("\n========== SELESAI ==========")
    if args.execute:
        print(
            "Robot sudah bergerak current/IDLE -> OBSERVATION -> pre_grasp. "
            "Cek fisik apakah gripper berada di atas objek."
        )
    else:
        print(
            "Plan-only selesai. Jika trajectory aman di RViz, jalankan ulang dengan --execute."
        )
    print("============================\n")

    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()