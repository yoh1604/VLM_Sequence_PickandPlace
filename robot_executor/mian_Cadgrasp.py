#!/usr/bin/env python3

import argparse
import copy
import json
import math
import socket
import sys
import time
from pathlib import Path

import rospy
import moveit_commander
from geometry_msgs.msg import Pose
from moveit_msgs.msg import DisplayTrajectory


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent


def try_load_capture_config():
    try:
        sys.path.append(str(PROJECT_DIR))
        import capture_config as cfg
        return cfg
    except Exception as e:
        print("[WARN] Gagal import capture_config.py:", e)
        return None


CFG = try_load_capture_config()


# ============================================================
# PATH / JSON
# ============================================================

def resolve_path(path_like):
    path = Path(str(path_like)).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


def load_waypoints(path):
    path = resolve_path(path)

    if not path.exists():
        raise FileNotFoundError(f"Waypoints JSON tidak ditemukan: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if "OBSERVATION" not in data:
        raise KeyError(f"Waypoint OBSERVATION tidak ada di {path}")

    if len(data["OBSERVATION"]) != 6:
        raise ValueError("Waypoint OBSERVATION harus 6 joint.")

    print("[OK] Loaded waypoints from:", path)
    return data


def get_default_target_json():
    if CFG is not None:
        if hasattr(CFG, "TOOL0_PREGRASP_TARGET_JSON"):
            return resolve_path(CFG.TOOL0_PREGRASP_TARGET_JSON)

        if hasattr(CFG, "VISION_OUTPUT_DIR"):
            return resolve_path(
                Path(CFG.VISION_OUTPUT_DIR) / "tool0_pregrasp_target.json"
            )

    return resolve_path(
        PROJECT_DIR
        / "outputs"
        / "test_default"
        / "vision_output"
        / "tool0_pregrasp_target.json"
    )


def load_target(path):
    path = resolve_path(path)

    if not path.exists():
        raise FileNotFoundError(f"Target JSON tidak ditemukan: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if not data.get("success", False):
        raise RuntimeError("Target JSON success=False")

    if "translation_tool0_pregrasp" not in data:
        raise KeyError("Target JSON tidak punya translation_tool0_pregrasp.")

    if "quaternion_tool0_xyzw" not in data:
        raise KeyError("Target JSON tidak punya quaternion_tool0_xyzw.")

    pos = data["translation_tool0_pregrasp"]
    quat = data["quaternion_tool0_xyzw"]

    if len(pos) != 3:
        raise ValueError("translation_tool0_pregrasp harus 3 angka.")

    if len(quat) != 4:
        raise ValueError("quaternion_tool0_xyzw harus 4 angka.")

    return pos, quat, data, path


# ============================================================
# ROBOTIQ SOCKET
# ============================================================

class RobotiqSocket:
    def __init__(self, robot_ip, port=63352, timeout=3.0):
        self.robot_ip = robot_ip
        self.port = port
        self.timeout = timeout
        self.sock = None

    def connect(self):
        print(f"[GRIPPER] Connecting to {self.robot_ip}:{self.port}")
        self.sock = socket.create_connection(
            (self.robot_ip, self.port),
            timeout=self.timeout,
        )
        self.sock.settimeout(self.timeout)
        print("[GRIPPER] Connected")

    def close_socket(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None
            print("[GRIPPER] Socket closed")

    def send_cmd(self, cmd):
        if self.sock is None:
            raise RuntimeError("Socket gripper belum connect.")

        msg = cmd.strip() + "\n"
        self.sock.sendall(msg.encode("utf-8"))

        try:
            resp = self.sock.recv(1024).decode("utf-8", errors="ignore").strip()
        except socket.timeout:
            resp = ""

        print(f"[GRIPPER] >> {cmd}")
        print(f"[GRIPPER] << {resp}")
        return resp

    def set_var(self, name, value):
        return self.send_cmd(f"SET {name} {value}")

    def activate(self):
        print("[GRIPPER] Activate")
        self.set_var("ACT", 1)
        self.set_var("GTO", 1)
        self.set_var("SPE", 255)
        self.set_var("FOR", 150)
        time.sleep(0.5)

    def open(self, speed=255, force=150):
        print("[GRIPPER] Open")
        self.set_var("SPE", int(speed))
        self.set_var("FOR", int(force))
        self.set_var("POS", 0)
        self.set_var("GTO", 1)
        time.sleep(1.0)

    def close(self, position=180, speed=150, force=80):
        print("[GRIPPER] Close")
        self.set_var("SPE", int(speed))
        self.set_var("FOR", int(force))
        self.set_var("POS", int(position))
        self.set_var("GTO", 1)
        time.sleep(1.2)


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_plan_result(plan_result):
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
        return current
    except Exception as e:
        print("[WARN] Gagal membaca current pose:", e)
        return None


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


def publish_display_trajectory(robot, trajectory, label="trajectory", wait_time=1.0):
    try:
        pub = rospy.Publisher(
            "/move_group/display_planned_path",
            DisplayTrajectory,
            queue_size=20,
        )

        rospy.sleep(0.5)

        display = DisplayTrajectory()
        display.trajectory_start = robot.get_current_state()
        display.trajectory.append(trajectory)

        pub.publish(display)

        print(f"[RVIZ] Published trajectory: {label}")
        rospy.sleep(float(wait_time))

    except Exception as e:
        print(f"[WARN] Gagal publish trajectory ke RViz:", e)


# ============================================================
# JOINT WRAPPING
# ============================================================

def wrap_to_nearest(angle, reference):
    return float(reference) + math.atan2(
        math.sin(float(angle) - float(reference)),
        math.cos(float(angle) - float(reference)),
    )


def unwrap_trajectory_to_nearest_start(trajectory, start_joints):
    new_traj = copy.deepcopy(trajectory)

    if len(new_traj.joint_trajectory.points) == 0:
        return new_traj

    prev = [float(v) for v in start_joints]

    for point in new_traj.joint_trajectory.points:
        pos = [float(v) for v in point.positions]
        new_pos = []

        for a, ref in zip(pos, prev):
            new_a = wrap_to_nearest(a, ref)
            new_pos.append(new_a)

        point.positions = tuple(new_pos)
        prev = new_pos

    return new_traj


def get_trajectory_final_positions(trajectory):
    try:
        if len(trajectory.joint_trajectory.points) == 0:
            return None
        return list(trajectory.joint_trajectory.points[-1].positions)
    except Exception:
        return None


def print_trajectory_joint_summary(group, start_joints, trajectory, label):
    names = group.get_active_joints()
    final_joints = get_trajectory_final_positions(trajectory)

    print(f"\n========== TRAJECTORY JOINT SUMMARY: {label} ==========")
    print("[JOINTS] active joints:", names)
    print("[JOINTS] start:", start_joints)

    if final_joints is None:
        print("[JOINTS] final: None")
        print("======================================================\n")
        return

    final_nearest = [
        wrap_to_nearest(float(f), float(s))
        for s, f in zip(start_joints, final_joints)
    ]

    print("[JOINTS] final raw     :", final_joints)
    print("[JOINTS] final nearest :", final_nearest)

    for i, name in enumerate(names):
        if i >= len(final_joints):
            continue

        raw_delta = abs(float(final_joints[i]) - float(start_joints[i]))
        wrapped_delta = abs(float(final_nearest[i]) - float(start_joints[i]))

        print(
            f"[JOINTS] {name}: "
            f"raw_delta={raw_delta:.4f}, "
            f"wrapped_delta={wrapped_delta:.4f}"
        )

    print("======================================================\n")


def joint_delta_ok(
    group,
    start_joints,
    final_joints,
    max_base_delta=1.80,
    max_wrist_delta=1.00,
):
    if final_joints is None:
        return False

    names = group.get_active_joints()

    final_nearest = [
        wrap_to_nearest(float(f), float(s))
        for s, f in zip(start_joints, final_joints)
    ]

    deltas = [
        abs(float(f_near) - float(s))
        for s, f_near in zip(start_joints, final_nearest)
    ]

    print("[CHECK] active joints:", names)
    print("[CHECK] start joints:", start_joints)
    print("[CHECK] final joints raw:", final_joints)
    print("[CHECK] final joints nearest:", final_nearest)

    for name, delta in zip(names, deltas):
        print(f"[CHECK] wrapped delta {name}: {delta:.4f} rad")

    if len(deltas) >= 1 and deltas[0] > float(max_base_delta):
        print(
            f"[REJECT] shoulder_pan delta terlalu besar: "
            f"{deltas[0]:.4f} > {max_base_delta:.4f}"
        )
        return False

    # Sekarang cek semua wrist, bukan hanya wrist_3.
    for idx in [3, 4, 5]:
        if len(deltas) > idx and deltas[idx] > float(max_wrist_delta):
            print(
                f"[REJECT] {names[idx]} delta terlalu besar: "
                f"{deltas[idx]:.4f} > {max_wrist_delta:.4f}"
            )
            return False

    print("[OK] Joint delta accepted.")
    return True


# ============================================================
# MOTION
# ============================================================

def go_joint(group, joints, label):
    print(f"\n[JOINT] Moving to {label}")
    print("[JOINT] target:", joints)
    print("[JOINT] current:", group.get_current_joint_values())

    if len(joints) != 6:
        raise ValueError(f"Waypoint {label} harus 6 joint.")

    group.stop()
    group.clear_pose_targets()
    group.set_start_state_to_current_state()
    rospy.sleep(0.3)

    ok = group.go([float(v) for v in joints], wait=True)

    group.stop()
    group.clear_pose_targets()
    rospy.sleep(0.8)

    print(f"[JOINT] {label} result:", ok)

    if not ok:
        raise RuntimeError(f"Gerak ke {label} gagal.")

    return ok


def apply_orientation_mode(group, pose, target_quat, eef_link, orientation_mode):
    if orientation_mode == "target":
        print("[ORIENTATION] Pakai quaternion dari target JSON.")
        pose.orientation.x = float(target_quat[0])
        pose.orientation.y = float(target_quat[1])
        pose.orientation.z = float(target_quat[2])
        pose.orientation.w = float(target_quat[3])
        return pose

    if orientation_mode in ["current", "observation"]:
        print("[ORIENTATION] Pakai orientasi CURRENT robot.")
        cur = group.get_current_pose(end_effector_link=eef_link).pose
        pose.orientation = copy.deepcopy(cur.orientation)
        return pose

    raise ValueError(f"orientation_mode tidak dikenal: {orientation_mode}")


def safety_check_target_from_current(
    group,
    target_pose,
    eef_link="tool0",
    max_xyz_distance=0.80,
):
    cur = group.get_current_pose(end_effector_link=eef_link).pose

    dx = float(target_pose.position.x - cur.position.x)
    dy = float(target_pose.position.y - cur.position.y)
    dz = float(target_pose.position.z - cur.position.z)

    dist_xy = math.sqrt(dx * dx + dy * dy)
    dist_xyz = math.sqrt(dx * dx + dy * dy + dz * dz)

    print("\n========== SAFETY TARGET CHECK ==========")
    print("[SAFETY] current tool0:")
    print("  x:", cur.position.x)
    print("  y:", cur.position.y)
    print("  z:", cur.position.z)
    print("[SAFETY] target:")
    print("  x:", target_pose.position.x)
    print("  y:", target_pose.position.y)
    print("  z:", target_pose.position.z)
    print("[SAFETY] delta:", [dx, dy, dz])
    print("[SAFETY] dist_xy :", dist_xy)
    print("[SAFETY] dist_xyz:", dist_xyz)
    print("[SAFETY] max_xyz_distance:", max_xyz_distance)
    print("=========================================\n")

    if dist_xyz > float(max_xyz_distance):
        raise RuntimeError(
            "Target terlalu jauh dari current pose. "
            "Kemungkinan frame salah atau file target tidak sesuai capture."
        )


def execute_trajectory_with_retry(group, trajectory, label, execute_retry=3):
    ok = False

    for i in range(int(execute_retry)):
        print(f"[EXECUTE] Executing {label} attempt {i + 1}/{execute_retry} ...")

        group.stop()
        group.clear_pose_targets()
        rospy.sleep(0.5)

        ok = group.execute(trajectory, wait=True)

        group.stop()
        group.clear_pose_targets()
        rospy.sleep(1.0)

        print(f"[EXECUTE] {label} result attempt {i + 1}:", ok)

        if ok:
            break

        print("[WARN] Execute gagal/PREEMPTED. Retry dengan current state...")
        group.set_start_state_to_current_state()
        rospy.sleep(1.0)

    if not ok:
        raise RuntimeError(f"Execute {label} gagal setelah {execute_retry} percobaan.")

    return ok


# ============================================================
# IMPORTANT: IK JOINT TARGET PLANNING
# ============================================================

def set_ik_joint_target_or_fallback(group, pose, eef_link):
    """
    Ini perbedaan utama dari script lama.

    Lama:
        group.set_pose_target(pose)

    Baru:
        group.set_joint_value_target(pose, eef_link, True)

    Tujuannya agar IK solution lebih dekat ke current joint seed,
    sehingga wrist tidak mudah memilih branch flip.
    """

    try:
        ik_ok = group.set_joint_value_target(pose, eef_link, True)
        print("[IK] set_joint_value_target result:", ik_ok)
        return "joint_value_target"
    except Exception as e:
        print("[WARN] set_joint_value_target gagal:", e)
        print("[WARN] fallback ke set_pose_target")
        group.set_pose_target(pose, end_effector_link=eef_link)
        return "pose_target"


def plan_pose_with_current_ik_and_publish(
    group,
    robot,
    pose,
    label,
    eef_link="tool0",
    rviz_preview_wait=3.0,
    rviz_confirm=True,
    max_plan_tries=10,
    max_base_delta=1.80,
    max_wrist_delta=1.00,
):
    print(f"\n[PLAN] Planning {label} with current-seeded IK ...")
    print_pose(f"{label} TARGET POSE", pose)

    accepted_trajectory = None

    for attempt in range(1, int(max_plan_tries) + 1):
        print(f"\n[PLAN] Attempt {attempt}/{max_plan_tries}")

        group.stop()
        group.clear_pose_targets()
        group.set_start_state_to_current_state()
        rospy.sleep(0.3)

        start_joints = [float(v) for v in group.get_current_joint_values()]
        print("[PLAN] start joints:", start_joints)

        target_mode = set_ik_joint_target_or_fallback(
            group=group,
            pose=pose,
            eef_link=eef_link,
        )

        plan_result = group.plan()
        success, trajectory_raw = normalize_plan_result(plan_result)

        group.clear_pose_targets()

        print("[PLAN] target mode:", target_mode)
        print("[PLAN] success:", success)

        if not success:
            print("[WARN] Plan failed. Retry...")
            rospy.sleep(0.5)
            continue

        try:
            print("[PLAN] raw trajectory points:", len(trajectory_raw.joint_trajectory.points))
        except Exception:
            pass

        print_trajectory_joint_summary(
            group=group,
            start_joints=start_joints,
            trajectory=trajectory_raw,
            label=f"{label} RAW",
        )

        trajectory = unwrap_trajectory_to_nearest_start(
            trajectory=trajectory_raw,
            start_joints=start_joints,
        )

        print_trajectory_joint_summary(
            group=group,
            start_joints=start_joints,
            trajectory=trajectory,
            label=f"{label} UNWRAPPED",
        )

        final_joints = get_trajectory_final_positions(trajectory)

        if not joint_delta_ok(
            group=group,
            start_joints=start_joints,
            final_joints=final_joints,
            max_base_delta=max_base_delta,
            max_wrist_delta=max_wrist_delta,
        ):
            print("[WARN] Trajectory ditolak karena joint delta terlalu besar. Retry...")
            rospy.sleep(0.5)
            continue

        accepted_trajectory = trajectory
        print("[OK] Trajectory accepted.")
        break

    if accepted_trajectory is None:
        raise RuntimeError(
            f"Gagal mendapatkan trajectory aman untuk {label}. "
            "Coba --max_wrist_delta 1.20 atau cek RViz target."
        )

    publish_display_trajectory(
        robot,
        accepted_trajectory,
        label=label,
        wait_time=rviz_preview_wait,
    )

    print("\n[RVIZ CHECK]")
    print(f"Trajectory {label} sudah dipublish ke RViz.")
    print("Kalau wrist/camera masih muter ekstrem, tekan CTRL+C.")
    print("Kalau sudah benar, tekan ENTER.")
    print()

    if rviz_confirm:
        input(f"Tekan ENTER jika trajectory {label} di RViz sudah benar... ")

    return accepted_trajectory


def plan_and_execute_descend_pose(
    group,
    pose,
    label,
    eef_link="tool0",
    execute_retry=3,
    max_wrist_delta=1.00,
):
    print(f"\n[DESCEND] Planning {label} ...")
    print_pose(f"{label} TARGET POSE", pose)

    group.stop()
    group.clear_pose_targets()
    group.set_start_state_to_current_state()
    rospy.sleep(0.3)

    start_joints = [float(v) for v in group.get_current_joint_values()]
    print("[DESCEND] start joints:", start_joints)

    set_ik_joint_target_or_fallback(
        group=group,
        pose=pose,
        eef_link=eef_link,
    )

    plan_result = group.plan()
    success, trajectory_raw = normalize_plan_result(plan_result)

    group.clear_pose_targets()

    print("[DESCEND] plan success:", success)

    if not success:
        raise RuntimeError(f"Planning {label} gagal.")

    trajectory = unwrap_trajectory_to_nearest_start(
        trajectory=trajectory_raw,
        start_joints=start_joints,
    )

    print_trajectory_joint_summary(
        group=group,
        start_joints=start_joints,
        trajectory=trajectory,
        label=f"{label} UNWRAPPED",
    )

    execute_trajectory_with_retry(
        group=group,
        trajectory=trajectory,
        label=label,
        execute_retry=execute_retry,
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "grasp.py current IK version: target base_link, no observation by default, "
            "current orientation, set_joint_value_target IK seed."
        )
    )

    parser.add_argument("--target_json", default=None)
    parser.add_argument("--waypoints_json", default="configs/waypoints_ur5.json")

    # Default sekarang: tidak pakai observation.
    parser.add_argument(
        "--use_observation",
        action="store_true",
        help="Aktifkan hanya kalau capture memang dari OBSERVATION.",
    )

    # Supaya command lama dengan --skip_observation tidak error.
    parser.add_argument("--skip_observation", action="store_true")

    parser.add_argument("--group_name", default="manipulator")
    parser.add_argument("--reference_frame", default="base_link")
    parser.add_argument("--eef_link", default="tool0")
    parser.add_argument("--robot_ip", default="192.168.200.1")

    parser.add_argument("--descend_z", type=float, default=0.08)
    parser.add_argument("--pregrasp_wait", type=float, default=5.0)

    parser.add_argument(
        "--orientation_mode",
        choices=["current", "observation", "target"],
        default="current",
    )

    parser.add_argument("--velocity", type=float, default=0.05)
    parser.add_argument("--acceleration", type=float, default=0.05)
    parser.add_argument("--planning_time", type=float, default=30.0)
    parser.add_argument("--planning_attempts", type=int, default=30)
    parser.add_argument("--position_tolerance", type=float, default=0.015)
    parser.add_argument("--orientation_tolerance", type=float, default=0.30)

    parser.add_argument("--rviz_preview_wait", type=float, default=3.0)
    parser.add_argument("--no_rviz_confirm", action="store_true")

    parser.add_argument("--max_plan_tries", type=int, default=10)
    parser.add_argument("--max_base_delta", type=float, default=1.80)
    parser.add_argument("--max_wrist_delta", type=float, default=1.00)
    parser.add_argument("--max_xyz_distance", type=float, default=0.80)

    parser.add_argument("--execute_retry", type=int, default=3)

    parser.add_argument("--gripper_position", type=int, default=180)
    parser.add_argument("--gripper_speed", type=int, default=150)
    parser.add_argument("--gripper_force", type=int, default=80)

    parser.add_argument("--disable_gripper", action="store_true")
    parser.add_argument("--execute", action="store_true")

    args = parser.parse_args()

    target_json = (
        resolve_path(args.target_json)
        if args.target_json is not None
        else get_default_target_json()
    )

    waypoints = None
    if args.use_observation:
        waypoints = load_waypoints(args.waypoints_json)

    print("\n========== GRASP CURRENT-IK VERSION ==========")
    print("PROJECT_DIR:", PROJECT_DIR)

    if CFG is not None:
        print("capture_config: LOADED")
        print("capture_config TEST_NAME:", getattr(CFG, "TEST_NAME", "N/A"))
        print("capture_config VISION_OUTPUT_DIR:", getattr(CFG, "VISION_OUTPUT_DIR", "N/A"))
    else:
        print("capture_config: NOT LOADED")

    print("TARGET_JSON:", target_json)
    print("reference_frame:", args.reference_frame)
    print("orientation_mode:", args.orientation_mode)
    print("use_observation:", args.use_observation)
    print("descend_z:", args.descend_z)
    print("max_base_delta:", args.max_base_delta)
    print("max_wrist_delta:", args.max_wrist_delta)
    print("max_xyz_distance:", args.max_xyz_distance)
    print("rviz_confirm:", not args.no_rviz_confirm)
    print("disable_gripper:", args.disable_gripper)
    print("==============================================\n")

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("grasp_current_ik_seed", anonymous=True)

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
    print("[INFO] Pose reference frame:", args.reference_frame)
    print("[INFO] End effector link:", group.get_end_effector_link())

    print_current_pose(group, args.eef_link)

    pos, quat, data, loaded_path = load_target(target_json)

    print("\n========== TARGET JSON LOADED ==========")
    print("Path:", loaded_path)
    print("success:", data.get("success"))
    print("frame:", data.get("frame"))
    print("translation_tool0_pregrasp:", data.get("translation_tool0_pregrasp"))
    print("quaternion_tool0_xyzw:", data.get("quaternion_tool0_xyzw"))
    print("manual_nudge_base_m:", data.get("manual_nudge_base_m"))
    print("=======================================\n")

    if not args.execute:
        input("Tekan ENTER untuk mulai, atau CTRL+C untuk batal... ")

    gripper = None

    try:
        # ------------------------------------------------------------
        # Optional OBSERVATION
        # ------------------------------------------------------------

        if args.use_observation:
            print("\n[WAYPOINT] Move ke OBSERVATION karena --use_observation aktif.")

            go_joint(
                group=group,
                joints=waypoints["OBSERVATION"],
                label="OBSERVATION",
            )

            print_current_pose(group, args.eef_link)
        else:
            print("[INFO] Tidak lewat OBSERVATION. Planning dari current/capture pose.")

        # ------------------------------------------------------------
        # Build pregrasp pose
        # ------------------------------------------------------------

        pregrasp_pose = make_pose(pos, quat)

        pregrasp_pose = apply_orientation_mode(
            group=group,
            pose=pregrasp_pose,
            target_quat=quat,
            eef_link=args.eef_link,
            orientation_mode=args.orientation_mode,
        )

        print_pose("FINAL TOOL0 PREGRASP POSE USED", pregrasp_pose)

        safety_check_target_from_current(
            group=group,
            target_pose=pregrasp_pose,
            eef_link=args.eef_link,
            max_xyz_distance=args.max_xyz_distance,
        )

        descend_pose = copy.deepcopy(pregrasp_pose)
        descend_pose.position.z -= float(args.descend_z)

        print_pose("FINAL TOOL0 DESCEND / GRASP POSE USED", descend_pose)

        # ------------------------------------------------------------
        # Plan to pregrasp with current IK seed
        # ------------------------------------------------------------

        pregrasp_traj = plan_pose_with_current_ik_and_publish(
            group=group,
            robot=robot,
            pose=pregrasp_pose,
            label="CURRENT_TO_TOOL0_PREGRASP_IK_SEEDED",
            eef_link=args.eef_link,
            rviz_preview_wait=args.rviz_preview_wait,
            rviz_confirm=(not args.no_rviz_confirm),
            max_plan_tries=args.max_plan_tries,
            max_base_delta=args.max_base_delta,
            max_wrist_delta=args.max_wrist_delta,
        )

        execute_trajectory_with_retry(
            group=group,
            trajectory=pregrasp_traj,
            label="CURRENT_TO_TOOL0_PREGRASP_IK_SEEDED",
            execute_retry=args.execute_retry,
        )

        # ------------------------------------------------------------
        # Open gripper
        # ------------------------------------------------------------

        if not args.disable_gripper:
            gripper = RobotiqSocket(args.robot_ip)
            gripper.connect()
            gripper.activate()

            print("\n[GRIPPER] Opening gripper at pre-grasp...")
            gripper.open()

            if args.pregrasp_wait > 0:
                print(f"[WAIT] Diam {args.pregrasp_wait:.1f} detik sebelum descend...")
                time.sleep(float(args.pregrasp_wait))

            group.stop()
            group.clear_pose_targets()
            rospy.sleep(0.5)

        else:
            print("[GRIPPER] disabled, skip open.")

        # ------------------------------------------------------------
        # Descend
        # ------------------------------------------------------------

        plan_and_execute_descend_pose(
            group=group,
            pose=descend_pose,
            label="DESCEND_TO_GRASP_IK_SEEDED",
            eef_link=args.eef_link,
            execute_retry=args.execute_retry,
            max_wrist_delta=args.max_wrist_delta,
        )

        # ------------------------------------------------------------
        # Close gripper
        # ------------------------------------------------------------

        if not args.disable_gripper:
            gripper.close(
                position=args.gripper_position,
                speed=args.gripper_speed,
                force=args.gripper_force,
            )
        else:
            print("[GRIPPER] disabled, skip close.")

    finally:
        group.clear_pose_targets()

        if gripper is not None:
            gripper.close_socket()

    print("\n========== SELESAI ==========")
    print("Robot sudah:")
    print("1. Plan dari current/capture pose ke TOOL0_PREGRASP.")
    print("2. Execute setelah preview RViz.")
    print("3. Open gripper.")
    print(f"4. Wait {args.pregrasp_wait:.1f} detik.")
    print(f"5. Descend {args.descend_z:.3f} m.")
    print("6. Close gripper.")
    print("Belum lift/discard.")
    print("=============================\n")


if __name__ == "__main__":
    main()