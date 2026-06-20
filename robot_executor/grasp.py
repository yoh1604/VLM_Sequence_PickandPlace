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


PROJECT_DIR = Path(__file__).resolve().parent.parent

def resolve_path(path_like):
    p = Path(str(path_like)).expanduser()
    if not p.is_absolute():
        p = PROJECT_DIR / p
    return p.resolve()


def load_target(path):
    path = resolve_path(path)

    if not path.exists():
        raise FileNotFoundError(f"Target JSON tidak ditemukan: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if not data.get("success", False):
        raise RuntimeError("Target JSON success=False")

    if "translation_tool0_pregrasp" not in data:
        raise KeyError("Target JSON tidak punya translation_tool0_pregrasp")

    if "quaternion_tool0_xyzw" not in data:
        raise KeyError("Target JSON tidak punya quaternion_tool0_xyzw")

    pos = data["translation_tool0_pregrasp"]
    quat = data["quaternion_tool0_xyzw"]

    if len(pos) != 3:
        raise ValueError("translation_tool0_pregrasp harus 3 angka")

    if len(quat) != 4:
        raise ValueError("quaternion_tool0_xyzw harus 4 angka")

    return pos, quat, data, path


# ============================================================
# GRIPPER
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
            raise RuntimeError("Socket gripper belum connect")

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

    # def activate(self):
    #     print("[GRIPPER] Activate")
    #     self.set_var("ACT", 1)
    #     self.set_var("GTO", 1)
    #     self.set_var("SPE", 255)
    #     self.set_var("FOR", 150)
    #     time.sleep(0.5)

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
        return bool(plan_result[0]), plan_result[1]

    traj = plan_result
    try:
        ok = len(traj.joint_trajectory.points) > 0
    except Exception:
        ok = traj is not None

    return ok, traj


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
    print("====================================\n")


def publish_display_trajectory(robot, trajectory, label="trajectory", wait_time=1.0):
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


def execute_trajectory(group, trajectory, label, retry=3):
    for i in range(int(retry)):
        print(f"[EXECUTE] {label} attempt {i + 1}/{retry}")

        group.stop()
        group.clear_pose_targets()
        rospy.sleep(0.4)

        ok = group.execute(trajectory, wait=True)

        group.stop()
        group.clear_pose_targets()
        rospy.sleep(0.8)

        print(f"[EXECUTE] {label} result:", ok)

        if ok:
            return True

        print("[WARN] Execute gagal. Retry...")

    raise RuntimeError(f"Execute gagal untuk {label}")


# ============================================================
# JOINT CHECK
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

    for pt in new_traj.joint_trajectory.points:
        pos = [float(v) for v in pt.positions]
        new_pos = []

        for a, ref in zip(pos, prev):
            new_pos.append(wrap_to_nearest(a, ref))

        pt.positions = tuple(new_pos)
        prev = new_pos

    return new_traj


def get_final_joints(trajectory):
    if len(trajectory.joint_trajectory.points) == 0:
        return None
    return list(trajectory.joint_trajectory.points[-1].positions)


def print_joint_summary(group, start_joints, trajectory, label):
    names = group.get_active_joints()
    final = get_final_joints(trajectory)

    print(f"\n========== JOINT SUMMARY: {label} ==========")
    print("start:", start_joints)
    print("final:", final)

    if final is None:
        print("===========================================\n")
        return

    final_nearest = [
        wrap_to_nearest(f, s)
        for s, f in zip(start_joints, final)
    ]

    print("final nearest:", final_nearest)

    for name, s, f in zip(names, start_joints, final_nearest):
        delta = abs(float(f) - float(s))
        print(f"{name}: delta={delta:.4f} rad")

    print("===========================================\n")


def reject_if_large_joint_jump(
    group,
    start_joints,
    trajectory,
    max_base_delta=1.80,
    max_wrist_delta=1.20,
):
    names = group.get_active_joints()
    final = get_final_joints(trajectory)

    if final is None:
        raise RuntimeError("Trajectory tidak punya final joints")

    final_nearest = [
        wrap_to_nearest(f, s)
        for s, f in zip(start_joints, final)
    ]

    deltas = [
        abs(float(f) - float(s))
        for s, f in zip(start_joints, final_nearest)
    ]

    if len(deltas) >= 1 and deltas[0] > float(max_base_delta):
        raise RuntimeError(
            f"[REJECT] {names[0]} delta {deltas[0]:.4f} > {max_base_delta}"
        )

    for idx in [3, 4, 5]:
        if len(deltas) > idx and deltas[idx] > float(max_wrist_delta):
            raise RuntimeError(
                f"[REJECT] {names[idx]} delta {deltas[idx]:.4f} > {max_wrist_delta}"
            )


# ============================================================
# FRAME / TARGET CHECK
# ============================================================

def apply_orientation_mode(group, pose, quat, eef_link, orientation_mode):
    if orientation_mode == "target":
        print("[ORIENTATION] Pakai quaternion target JSON")
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])
        return pose

    if orientation_mode == "current":
        print("[ORIENTATION] Pakai orientasi current robot")
        cur = group.get_current_pose(end_effector_link=eef_link).pose
        pose.orientation = copy.deepcopy(cur.orientation)
        return pose

    raise ValueError(f"orientation_mode tidak dikenal: {orientation_mode}")


def safety_check_distance(group, target_pose, eef_link, max_xyz_distance):
    cur = group.get_current_pose(end_effector_link=eef_link).pose

    dx = target_pose.position.x - cur.position.x
    dy = target_pose.position.y - cur.position.y
    dz = target_pose.position.z - cur.position.z

    dist_xy = math.sqrt(dx * dx + dy * dy)
    dist_xyz = math.sqrt(dx * dx + dy * dy + dz * dz)

    print("\n========== TARGET DISTANCE CHECK ==========")
    print("current:", [cur.position.x, cur.position.y, cur.position.z])
    print("target :", [target_pose.position.x, target_pose.position.y, target_pose.position.z])
    print("delta  :", [dx, dy, dz])
    print("dist_xy :", dist_xy)
    print("dist_xyz:", dist_xyz)
    print("max_xyz_distance:", max_xyz_distance)
    print("==========================================\n")

    if dist_xyz > float(max_xyz_distance):
        raise RuntimeError(
            f"Target terlalu jauh: {dist_xyz:.3f} m > {max_xyz_distance:.3f} m"
        )


# ============================================================
# CARTESIAN PLANNING
# ============================================================

def compute_cartesian_single_target(
    group,
    target_pose,
    label,
    eef_link="tool0",
    eef_step=0.005,
    min_fraction=0.90,
):
    print(f"\n[CARTESIAN] Planning {label}")
    print("[CARTESIAN] eef_step:", eef_step)
    print("[CARTESIAN] min_fraction:", min_fraction)

    group.stop()
    group.clear_pose_targets()
    group.set_start_state_to_current_state()
    rospy.sleep(0.3)

    start_joints = [float(v) for v in group.get_current_joint_values()]
    start_pose = group.get_current_pose(end_effector_link=eef_link).pose

    print_pose(f"{label} START POSE", start_pose)
    print_pose(f"{label} TARGET POSE", target_pose)

    waypoints = [copy.deepcopy(target_pose)]

    # MoveIt Noetic pada sistem kamu:
    # compute_cartesian_path(waypoints, eef_step, avoid_collisions)
    # Argumen ketiga harus bool, bukan jump_threshold float.
    plan, fraction = group.compute_cartesian_path(
        waypoints,
        float(eef_step),
        True,
    )

    print(f"[CARTESIAN] {label} fraction:", fraction)

    if fraction < float(min_fraction):
        raise RuntimeError(
            f"Cartesian fraction {label} terlalu kecil: "
            f"{fraction:.3f} < {min_fraction:.3f}"
        )

    plan = unwrap_trajectory_to_nearest_start(plan, start_joints)

    print_joint_summary(
        group=group,
        start_joints=start_joints,
        trajectory=plan,
        label=label,
    )

    return plan, start_joints


def plan_cartesian_segment(
    group,
    robot,
    target_pose,
    label,
    eef_link="tool0",
    eef_step=0.005,
    min_fraction=0.90,
    rviz_preview_wait=3.0,
    rviz_confirm=True,
    max_base_delta=1.80,
    max_wrist_delta=1.20,
):
    plan, start_joints = compute_cartesian_single_target(
        group=group,
        target_pose=target_pose,
        label=label,
        eef_link=eef_link,
        eef_step=eef_step,
        min_fraction=min_fraction,
    )

    reject_if_large_joint_jump(
        group=group,
        start_joints=start_joints,
        trajectory=plan,
        max_base_delta=max_base_delta,
        max_wrist_delta=max_wrist_delta,
    )

    publish_display_trajectory(
        robot=robot,
        trajectory=plan,
        label=label,
        wait_time=rviz_preview_wait,
    )

    # print("\n[RVIZ CHECK]")
    # print(f"Trajectory {label} sudah dipublish ke RViz.")
    # print("Kalau aman, tekan ENTER.")
    # print("Kalau tidak aman, tekan CTRL+C.")
    # print()

    # if rviz_confirm:
    #     input(f"Tekan ENTER jika trajectory {label} sudah benar... ")

    return plan


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_json", required=True)
    parser.add_argument("--robot_ip", default="192.168.200.1")

    parser.add_argument("--group_name", default="manipulator")
    parser.add_argument("--reference_frame", default="base_link")
    parser.add_argument("--eef_link", default="tool0")

    parser.add_argument(
        "--orientation_mode",
        choices=["current", "target"],
        default="current",
    )

    # Supaya command lama tetap tidak error.
    parser.add_argument("--skip_observation", action="store_true")

    parser.add_argument("--safe_lift_z", type=float, default=0.10)
    parser.add_argument("--safe_lift_eef_step", type=float, default=0.005)
    parser.add_argument("--cartesian_min_fraction", type=float, default=0.90)

    parser.add_argument("--descend_z", type=float, default=0.08)
    parser.add_argument("--pregrasp_wait", type=float, default=5.0)

    parser.add_argument("--velocity", type=float, default=0.05)
    parser.add_argument("--acceleration", type=float, default=0.05)
    parser.add_argument("--planning_time", type=float, default=30.0)
    parser.add_argument("--planning_attempts", type=int, default=30)

    parser.add_argument("--max_xyz_distance", type=float, default=0.80)
    parser.add_argument("--max_base_delta", type=float, default=1.80)
    parser.add_argument("--max_wrist_delta", type=float, default=1.20)

    parser.add_argument("--rviz_preview_wait", type=float, default=3.0)
    parser.add_argument("--no_rviz_confirm", action="store_true")

    parser.add_argument("--execute_retry", type=int, default=3)

    parser.add_argument("--disable_gripper", action="store_true")
    parser.add_argument("--gripper_position", type=int, default=180)
    parser.add_argument("--gripper_speed", type=int, default=150)
    parser.add_argument("--gripper_force", type=int, default=80)

    parser.add_argument("--execute", action="store_true")

    args = parser.parse_args()

    target_json = resolve_path(args.target_json)

    print("\n========== GRASP CARTESIAN SAFE LIFT VERSION ==========")
    print("PROJECT_DIR:", PROJECT_DIR)
    print("target_json:", target_json)
    print("reference_frame:", args.reference_frame)
    print("orientation_mode:", args.orientation_mode)
    print("safe_lift_z:", args.safe_lift_z)
    print("safe_lift_eef_step:", args.safe_lift_eef_step)
    print("descend_z:", args.descend_z)
    print("max_wrist_delta:", args.max_wrist_delta)
    print("execute:", args.execute)
    print("=======================================================\n")

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("grasp_cartesian_safe_lift", anonymous=True)

    robot = moveit_commander.RobotCommander()
    group = moveit_commander.MoveGroupCommander(args.group_name)

    group.set_pose_reference_frame(args.reference_frame)
    group.set_end_effector_link(args.eef_link)

    group.set_max_velocity_scaling_factor(float(args.velocity))
    group.set_max_acceleration_scaling_factor(float(args.acceleration))
    group.set_planning_time(float(args.planning_time))
    group.set_num_planning_attempts(int(args.planning_attempts))

    print("[INFO] Planning frame:", group.get_planning_frame())
    print("[INFO] Reference frame:", args.reference_frame)
    print("[INFO] EEF link:", group.get_end_effector_link())

    current_pose = group.get_current_pose(end_effector_link=args.eef_link).pose
    print_pose("CURRENT TOOL0 POSE", current_pose)

    pos, quat, target_data, loaded_path = load_target(target_json)

    print("\n========== TARGET JSON ==========")
    print("path:", loaded_path)
    print("success:", target_data.get("success"))
    print("frame:", target_data.get("frame"))
    print("translation_tool0_pregrasp:", target_data.get("translation_tool0_pregrasp"))
    print("quaternion_tool0_xyzw:", target_data.get("quaternion_tool0_xyzw"))
    print("manual_nudge_base_m:", target_data.get("manual_nudge_base_m"))
    print("=================================\n")

    pregrasp_pose = make_pose(pos, quat)

    pregrasp_pose = apply_orientation_mode(
        group=group,
        pose=pregrasp_pose,
        quat=quat,
        eef_link=args.eef_link,
        orientation_mode=args.orientation_mode,
    )

    print_pose("FINAL PREGRASP POSE", pregrasp_pose)

    safety_check_distance(
        group=group,
        target_pose=pregrasp_pose,
        eef_link=args.eef_link,
        max_xyz_distance=args.max_xyz_distance,
    )

    # Pose safe lift: current naik vertikal.
    safe_lift_pose = copy.deepcopy(current_pose)
    safe_lift_pose.position.z += float(args.safe_lift_z)

    # Pose descend/grasp: dari pregrasp turun Z.
    descend_pose = copy.deepcopy(pregrasp_pose)
    descend_pose.position.z -= float(args.descend_z)

    print_pose("SAFE LIFT POSE", safe_lift_pose)
    print_pose("DESCEND / GRASP POSE", descend_pose)

    # if not args.execute:
    #     input("Tekan ENTER untuk mulai planning, atau CTRL+C untuk batal... ")

    rviz_confirm = not args.no_rviz_confirm

    gripper = None

    try:
        # ====================================================
        # SEGMENT 1: CURRENT / IDLE -> SAFE LIFT
        # Cartesian vertical lift.
        # ====================================================

        if abs(float(args.safe_lift_z)) > 1e-6:
            safe_traj = plan_cartesian_segment(
                group=group,
                robot=robot,
                target_pose=safe_lift_pose,
                label="CURRENT_TO_SAFE_LIFT_CARTESIAN",
                eef_link=args.eef_link,
                eef_step=args.safe_lift_eef_step,
                min_fraction=args.cartesian_min_fraction,
                rviz_preview_wait=args.rviz_preview_wait,
                rviz_confirm=rviz_confirm,
                max_base_delta=args.max_base_delta,
                max_wrist_delta=args.max_wrist_delta,
            )

            execute_trajectory(
                group=group,
                trajectory=safe_traj,
                label="CURRENT_TO_SAFE_LIFT_CARTESIAN",
                retry=args.execute_retry,
            )
        else:
            print("[SAFE LIFT] safe_lift_z=0, skip safe lift.")

        # ====================================================
        # SEGMENT 2: SAFE LIFT -> PREGRASP
        # INI BAGIAN YANG DIMINTA:
        # Tidak pakai IK pose target.
        # Pakai Cartesian path dari posisi setelah lift ke pregrasp.
        # ====================================================

        pregrasp_traj = plan_cartesian_segment(
            group=group,
            robot=robot,
            target_pose=pregrasp_pose,
            label="SAFE_TO_PREGRASP_CARTESIAN",
            eef_link=args.eef_link,
            eef_step=args.safe_lift_eef_step,
            min_fraction=args.cartesian_min_fraction,
            rviz_preview_wait=args.rviz_preview_wait,
            rviz_confirm=rviz_confirm,
            max_base_delta=args.max_base_delta,
            max_wrist_delta=args.max_wrist_delta,
        )

        execute_trajectory(
            group=group,
            trajectory=pregrasp_traj,
            label="SAFE_TO_PREGRASP_CARTESIAN",
            retry=args.execute_retry,
        )

        # ====================================================
        # GRIPPER OPEN
        # ====================================================

        if not args.disable_gripper:
            gripper = RobotiqSocket(args.robot_ip)
            gripper.connect()
            # gripper.activate()
            gripper.open()

            if args.pregrasp_wait > 0:
                print(f"[WAIT] pregrasp wait {args.pregrasp_wait:.1f} s")
                time.sleep(float(args.pregrasp_wait))
        else:
            print("[GRIPPER] Disabled, skip open.")

        # ====================================================
        # SEGMENT 3: PREGRASP -> DESCEND
        # Cartesian descend.
        # ====================================================

        descend_traj = plan_cartesian_segment(
            group=group,
            robot=robot,
            target_pose=descend_pose,
            label="PREGRASP_TO_DESCEND_CARTESIAN",
            eef_link=args.eef_link,
            eef_step=args.safe_lift_eef_step,
            min_fraction=args.cartesian_min_fraction,
            rviz_preview_wait=args.rviz_preview_wait,
            rviz_confirm=rviz_confirm,
            max_base_delta=args.max_base_delta,
            max_wrist_delta=args.max_wrist_delta,
        )

        execute_trajectory(
            group=group,
            trajectory=descend_traj,
            label="PREGRASP_TO_DESCEND_CARTESIAN",
            retry=args.execute_retry,
        )

        # ====================================================
        # GRIPPER CLOSE
        # ====================================================

        if not args.disable_gripper:
            gripper.close(
                position=args.gripper_position,
                speed=args.gripper_speed,
                force=args.gripper_force,
            )
        else:
            print("[GRIPPER] Disabled, skip close.")

    finally:
        group.stop()
        group.clear_pose_targets()

        if gripper is not None:
            gripper.close_socket()

    print("\n========== SELESAI ==========")
    print("Alur selesai:")
    print("1. current/IDLE -> safe lift Cartesian")
    print("2. safe lift -> pregrasp Cartesian")
    print("3. pregrasp -> descend Cartesian")
    print("4. close gripper")
    print("=============================\n")


if __name__ == "__main__":
    main()
