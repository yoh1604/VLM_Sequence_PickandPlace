#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
execute_tool0_grasp_pipeline_simple.py

Script execute sederhana untuk UR5 + Robotiq.

Alur:
1. Load waypoint IDLE dan OBSERVATION.
2. Load target JSON:
   - translation_tool0_pregrasp = posisi x y z
   - quaternion_tool0_xyzw      = orientation qx qy qz qw
3. Gerak:
   current -> IDLE -> OBSERVATION -> TOOL0_PREGRASP
4. Jika --execute:
   open gripper -> descend -> close -> lift

Catatan penting:
- Ini masih menggunakan hasil GraspNet pipeline kamu.
- Kalau robot muter balik, jalankan dengan:
    --orientation_mode observation
  supaya posisi tetap dari target_json, tapi orientasi gripper dipertahankan dari OBSERVATION.

Contoh plan-only:
python3 robot_executor/execute_tool0_grasp_pipeline_simple.py \
  --target_json outputs/test_grasp_2/vision_output/tool0_pregrasp_target.json \
  --robot_ip 192.168.200.1 \
  --orientation_mode observation

Contoh execute:
python3 robot_executor/execute_tool0_grasp_pipeline_simple.py \
  --target_json outputs/test_grasp_2/vision_output/tool0_pregrasp_target.json \
  --robot_ip 192.168.200.1 \
  --orientation_mode observation \
  --grasp_down 0.03 \
  --execute
"""

import argparse
import json
import socket
import sys
import time
from copy import deepcopy
from pathlib import Path

import rospy
import moveit_commander
from geometry_msgs.msg import Pose


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WAYPOINT_FILE = PROJECT_DIR / "configs" / "waypoints_ur5.json"
DEFAULT_TARGET_JSON = (
    PROJECT_DIR
    / "outputs"
    / "test_default"
    / "vision_output"
    / "tool0_pregrasp_target.json"
)


# ============================================================
# PATH / JSON
# ============================================================

def resolve_path(path_like):
    path = Path(str(path_like)).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


def load_json(path_like, label="JSON"):
    path = resolve_path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return data, path


def load_tool0_target(path_like):
    data, path = load_json(path_like, "TOOL0_PREGRASP_TARGET_JSON")

    if not data.get("success", False):
        raise RuntimeError(f"Target JSON success=False: {path}")

    if "translation_tool0_pregrasp" not in data:
        raise KeyError("Target JSON tidak punya key translation_tool0_pregrasp")

    if "quaternion_tool0_xyzw" not in data:
        raise KeyError("Target JSON tidak punya key quaternion_tool0_xyzw")

    pos = data["translation_tool0_pregrasp"]
    quat = data["quaternion_tool0_xyzw"]

    if len(pos) != 3:
        raise ValueError("translation_tool0_pregrasp harus 3 angka.")
    if len(quat) != 4:
        raise ValueError("quaternion_tool0_xyzw harus 4 angka.")

    return [float(v) for v in pos], [float(v) for v in quat], data, path


# ============================================================
# PRINT
# ============================================================

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


def print_moveit_info(group, robot, eef_link):
    print("\n========== MOVEIT INFO ==========")
    print("Planning frame:", group.get_planning_frame())
    print("End effector link:", group.get_end_effector_link())
    print("Available groups:", robot.get_group_names())
    print("Active joints:", group.get_active_joints())
    print("Current joints:", [round(x, 6) for x in group.get_current_joint_values()])
    try:
        current = group.get_current_pose(end_effector_link=eef_link).pose
        print_pose(f"CURRENT {eef_link} POSE", current)
    except Exception as e:
        print("[WARN] Gagal membaca current pose:", e)
    print("=================================\n")


# ============================================================
# ROBOTIQ
# ============================================================

class RobotiqSocket:
    def __init__(self, robot_ip, port=63352, timeout=3.0, enabled=True):
        self.robot_ip = robot_ip
        self.port = int(port)
        self.timeout = float(timeout)
        self.enabled = bool(enabled)
        self.sock = None

    def connect(self):
        if not self.enabled:
            print("[GRIPPER] Disabled. Tidak connect.")
            return

        if not self.robot_ip:
            raise ValueError("--robot_ip wajib diisi jika gripper aktif.")

        print(f"[GRIPPER] Connecting to {self.robot_ip}:{self.port}")
        self.sock = socket.create_connection(
            (self.robot_ip, self.port),
            timeout=self.timeout,
        )
        self.sock.settimeout(self.timeout)
        print("[GRIPPER] Connected")

    def close_socket(self):
        if self.sock:
            self.sock.close()
            self.sock = None
            print("[GRIPPER] Socket closed")

    def send_cmd(self, cmd):
        if not self.enabled:
            print(f"[GRIPPER DISABLED] skip: {cmd}")
            return ""

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
        self.send_cmd(f"SET {name} {int(value)}")

    def activate(self):
        print("[GRIPPER] Activating...")
        self.set_var("ACT", 1)
        self.set_var("GTO", 1)
        self.set_var("SPE", 255)
        self.set_var("FOR", 150)
        time.sleep(1.0)

    def open(self, speed=255, force=150, wait=1.0):
        print("[GRIPPER] Opening...")
        self.set_var("SPE", speed)
        self.set_var("FOR", force)
        self.set_var("POS", 0)
        self.set_var("GTO", 1)
        time.sleep(float(wait))

    def close(self, position=255, speed=255, force=150, wait=1.0):
        print("[GRIPPER] Closing...")
        self.set_var("SPE", speed)
        self.set_var("FOR", force)
        self.set_var("POS", position)
        self.set_var("GTO", 1)
        time.sleep(float(wait))


# ============================================================
# MOVEIT HELPERS
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


def make_pose_from_target(pos, quat):
    pose = Pose()
    pose.position.x = float(pos[0])
    pose.position.y = float(pos[1])
    pose.position.z = float(pos[2])
    pose.orientation.x = float(quat[0])
    pose.orientation.y = float(quat[1])
    pose.orientation.z = float(quat[2])
    pose.orientation.w = float(quat[3])
    return pose


def make_pose_position_with_observation_orientation(pos, observation_pose):
    pose = Pose()
    pose.position.x = float(pos[0])
    pose.position.y = float(pos[1])
    pose.position.z = float(pos[2])
    pose.orientation = deepcopy(observation_pose.orientation)
    return pose


def move_joint_waypoint(group, waypoint_name, joints, execute=False):
    if len(joints) != 6:
        raise ValueError(f"Waypoint {waypoint_name} harus punya 6 joint, sekarang {len(joints)}")

    print(f"\n========== JOINT TARGET: {waypoint_name} ==========")
    print("Target joints:", joints)
    print("Execute:", execute)
    print("=================================================\n")

    group.clear_pose_targets()

    if execute:
        rospy.loginfo(f"Moving to {waypoint_name}: {joints}")
        ok = group.go(joints, wait=True)
        group.stop()
        group.clear_pose_targets()
        print(f"[EXECUTE] {waypoint_name} result:", ok)

        if not ok:
            raise RuntimeError(f"Execute ke {waypoint_name} gagal.")

        rospy.sleep(0.5)
        return True

    group.set_joint_value_target(joints)
    plan_result = group.plan()
    ok, _ = normalize_plan_result(plan_result)
    print(f"[PLAN ONLY] {waypoint_name} success:", ok)

    if not ok:
        raise RuntimeError(f"Planning ke {waypoint_name} gagal.")

    return True


def plan_or_execute_pose(group, pose, eef_link, label, execute=False):
    print_pose(label, pose)

    group.clear_pose_targets()
    group.set_pose_target(pose, end_effector_link=eef_link)

    print(f"[PLAN] Planning to {label}...")
    plan_result = group.plan()
    ok, trajectory = normalize_plan_result(plan_result)
    print(f"[PLAN] {label} success:", ok)

    if not ok:
        group.clear_pose_targets()
        raise RuntimeError(f"Planning ke {label} gagal.")

    try:
        print(f"[PLAN] {label} trajectory points:", len(trajectory.joint_trajectory.points))
    except Exception:
        pass

    if execute:
        print(f"[EXECUTE] Moving to {label}...")
        ok_exec = group.execute(trajectory, wait=True)
        group.stop()
        group.clear_pose_targets()
        print(f"[EXECUTE] {label} result:", ok_exec)

        if not ok_exec:
            raise RuntimeError(f"Execute ke {label} gagal.")

        rospy.sleep(0.5)
    else:
        print(f"[PLAN ONLY] {label} tidak dieksekusi.")

    return True


def cartesian_z_motion(group, eef_link, dz, label, execute=False, min_fraction=0.20, eef_step=0.005):
    current_pose = group.get_current_pose(end_effector_link=eef_link).pose
    target_pose = deepcopy(current_pose)
    target_pose.position.z += float(dz)

    print_pose(f"{label} START", current_pose)
    print_pose(f"{label} TARGET", target_pose)

    waypoints = [target_pose]

    print(f"[CARTESIAN PLAN] {label}, dz={dz}, eef_step={eef_step}")

    try:
        plan, fraction = group.compute_cartesian_path(
            waypoints,
            float(eef_step),
            True,
        )
    except TypeError:
        plan, fraction = group.compute_cartesian_path(
            waypoints,
            float(eef_step),
            0.0,
        )

    print(f"[CARTESIAN PLAN] {label} fraction:", fraction)

    if fraction < float(min_fraction):
        raise RuntimeError(
            f"Cartesian path {label} kurang aman: fraction={fraction:.3f}, "
            f"minimal={min_fraction:.3f}. Robot tidak dieksekusi."
        )

    if execute:
        print(f"[EXECUTE] Cartesian {label}...")
        ok = group.execute(plan, wait=True)
        group.stop()
        group.clear_pose_targets()
        print(f"[EXECUTE] Cartesian {label} result:", ok)

        if not ok:
            raise RuntimeError(f"Execute Cartesian {label} gagal.")

        rospy.sleep(0.3)
    else:
        print(f"[PLAN ONLY] Cartesian {label} tidak dieksekusi.")

    return True


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--move_group", default="manipulator")
    parser.add_argument("--reference_frame", default="base")
    parser.add_argument("--eef_link", default="tool0")
    parser.add_argument("--waypoints", default=str(DEFAULT_WAYPOINT_FILE))
    parser.add_argument("--target_json", default=str(DEFAULT_TARGET_JSON))

    parser.add_argument("--idle_name", default="IDLE")
    parser.add_argument("--observation_name", default="OBSERVATION")
    parser.add_argument("--skip_idle", action="store_true")
    parser.add_argument("--skip_observation", action="store_true")

    parser.add_argument(
        "--orientation_mode",
        choices=["target", "observation"],
        default="observation",
        help=(
            "target = pakai quaternion_tool0_xyzw dari target_json. "
            "observation = posisi dari target_json, orientasi dari TCP saat OBSERVATION."
        ),
    )

    parser.add_argument("--velocity", type=float, default=0.05)
    parser.add_argument("--acceleration", type=float, default=0.05)
    parser.add_argument("--planning_time", type=float, default=15.0)
    parser.add_argument("--planning_attempts", type=int, default=20)
    parser.add_argument("--position_tolerance", type=float, default=0.015)
    parser.add_argument("--orientation_tolerance", type=float, default=0.35)

    parser.add_argument("--grasp_down", type=float, default=0.03)
    parser.add_argument("--lift_up", type=float, default=0.10)
    parser.add_argument("--cartesian_min_fraction", type=float, default=0.20)
    parser.add_argument("--cartesian_eef_step", type=float, default=0.005)

    parser.add_argument("--robot_ip", default="192.168.200.1")
    parser.add_argument("--gripper_port", type=int, default=63352)
    parser.add_argument("--disable_gripper", action="store_true")
    parser.add_argument("--open_speed", type=int, default=255)
    parser.add_argument("--open_force", type=int, default=150)
    parser.add_argument("--close_position", type=int, default=255)
    parser.add_argument("--close_speed", type=int, default=255)
    parser.add_argument("--close_force", type=int, default=150)
    parser.add_argument("--gripper_wait", type=float, default=1.0)

    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no_prompt", action="store_true")

    args = parser.parse_args()

    target_json = resolve_path(args.target_json)
    waypoint_file = resolve_path(args.waypoints)

    print("\n========== SIMPLE TOOL0 GRASP EXECUTE ==========")
    print("PROJECT_DIR:", PROJECT_DIR)
    print("WAYPOINT_FILE:", waypoint_file)
    print("TARGET_JSON:", target_json)
    print("orientation_mode:", args.orientation_mode)
    print("execute:", args.execute)
    print("grasp_down:", args.grasp_down)
    print("lift_up:", args.lift_up)
    print("robot_ip:", args.robot_ip)
    print("disable_gripper:", args.disable_gripper)
    print("================================================\n")

    if args.execute and not args.no_prompt:
        print("PERIKSA SEBELUM EXECUTE:")
        print("1. Robot ready dan speed slider > 0.")
        print("2. Jalur IDLE -> OBSERVATION -> TOOL0_PREGRASP aman.")
        print("3. Jika tidak mau wrist muter, gunakan --orientation_mode observation.")
        print("4. Mulai grasp_down kecil dulu, contoh 0.02 atau 0.03.")
        input("Tekan ENTER untuk mulai EXECUTE, atau CTRL+C untuk batal... ")

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("execute_tool0_grasp_pipeline_simple", anonymous=True)

    robot = moveit_commander.RobotCommander()
    group = moveit_commander.MoveGroupCommander(args.move_group)

    group.set_pose_reference_frame(args.reference_frame)
    group.set_end_effector_link(args.eef_link)
    group.set_max_velocity_scaling_factor(float(args.velocity))
    group.set_max_acceleration_scaling_factor(float(args.acceleration))
    group.set_planning_time(float(args.planning_time))
    group.set_num_planning_attempts(int(args.planning_attempts))
    group.allow_replanning(True)
    group.set_goal_position_tolerance(float(args.position_tolerance))
    group.set_goal_orientation_tolerance(float(args.orientation_tolerance))

    gripper = RobotiqSocket(
        robot_ip=args.robot_ip,
        port=args.gripper_port,
        enabled=(not args.disable_gripper),
    )

    try:
        print_moveit_info(group, robot, args.eef_link)

        waypoints, waypoint_path = load_json(waypoint_file, "WAYPOINTS_JSON")
        print("[INFO] Waypoint file loaded:", waypoint_path)

        if args.idle_name not in waypoints:
            raise KeyError(f"Waypoint {args.idle_name} tidak ditemukan.")
        if args.observation_name not in waypoints:
            raise KeyError(f"Waypoint {args.observation_name} tidak ditemukan.")

        pos, quat, target_data, target_path = load_tool0_target(target_json)
        print("[INFO] Target file loaded:", target_path)
        print("[INFO] translation_tool0_pregrasp:", pos)
        print("[INFO] quaternion_tool0_xyzw:", quat)

        if args.execute and not args.disable_gripper:
            gripper.connect()
            gripper.activate()
            gripper.open(
                speed=args.open_speed,
                force=args.open_force,
                wait=args.gripper_wait,
            )
        elif args.execute and args.disable_gripper:
            print("[INFO] Execute aktif, gripper disabled.")
        else:
            print("[PLAN ONLY] Skip gripper connect/open.")

        # 1. IDLE
        if not args.skip_idle:
            move_joint_waypoint(
                group=group,
                waypoint_name=args.idle_name,
                joints=waypoints[args.idle_name],
                execute=args.execute,
            )
        else:
            print("[INFO] Skip IDLE.")

        # 2. OBSERVATION
        if not args.skip_observation:
            move_joint_waypoint(
                group=group,
                waypoint_name=args.observation_name,
                joints=waypoints[args.observation_name],
                execute=args.execute,
            )
        else:
            print("[INFO] Skip OBSERVATION.")

        # 3. Build target pose
        rospy.sleep(0.5)
        observation_pose = group.get_current_pose(end_effector_link=args.eef_link).pose
        print_pose("OBSERVATION/CURRENT TCP POSE USED AS ORIENTATION SOURCE", observation_pose)

        if args.orientation_mode == "target":
            target_pose = make_pose_from_target(pos, quat)
            print("[INFO] Orientation mode: target. Pakai quaternion_tool0_xyzw dari JSON.")
        else:
            target_pose = make_pose_position_with_observation_orientation(pos, observation_pose)
            print("[INFO] Orientation mode: observation. Posisi dari JSON, orientasi dari OBSERVATION/current TCP.")
            print("[INFO] quaternion_tool0_xyzw diabaikan untuk mencegah gripper muter balik.")

        # 4. PREGRASP
        plan_or_execute_pose(
            group=group,
            pose=target_pose,
            eef_link=args.eef_link,
            label="TOOL0 PREGRASP",
            execute=args.execute,
        )

        if not args.execute:
            print("\n========== PLAN ONLY SELESAI ==========")
            print("Plan only selesai sampai TOOL0 PREGRASP.")
            print("Descend, close, dan lift dilewati karena --execute belum aktif.")
            print("======================================\n")
            return

        # 5. DESCEND
        cartesian_z_motion(
            group=group,
            eef_link=args.eef_link,
            dz=-abs(float(args.grasp_down)),
            label="DESCEND TO GRASP",
            execute=True,
            min_fraction=args.cartesian_min_fraction,
            eef_step=args.cartesian_eef_step,
        )

        # 6. CLOSE
        if not args.disable_gripper:
            gripper.close(
                position=args.close_position,
                speed=args.close_speed,
                force=args.close_force,
                wait=args.gripper_wait,
            )

        # 7. LIFT
        cartesian_z_motion(
            group=group,
            eef_link=args.eef_link,
            dz=abs(float(args.lift_up)),
            label="LIFT AFTER GRASP",
            execute=True,
            min_fraction=args.cartesian_min_fraction,
            eef_step=args.cartesian_eef_step,
        )

        print("\n========== SELESAI ==========")
        print("Robot selesai: IDLE -> OBSERVATION -> TOOL0_PREGRASP -> DESCEND -> CLOSE -> LIFT.")
        print("=============================\n")

    finally:
        gripper.close_socket()
        try:
            moveit_commander.roscpp_shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
