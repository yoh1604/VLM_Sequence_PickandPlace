#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
execute_tool0_grasp_pipeline.py

Script gabungan untuk UR5 + Robotiq 2F-85:
1. Load waypoint IDLE dan OBSERVATION dari configs/waypoints_ur5.json
2. Load target tool0 pre-grasp dari tool0_pregrasp_target.json
3. Gerakkan robot: current -> IDLE -> OBSERVATION -> TOOL0_PREGRASP
4. Buka gripper sebelum turun
5. Turun dari pre-grasp ke grasp dengan Cartesian path
6. Tutup gripper
7. Angkat objek dengan Cartesian path

Input utama:
- tool0_pregrasp_target.json hasil convert_gripper_tip_to_tool0_target.py
  Wajib punya:
    success: true
    translation_tool0_pregrasp: [x, y, z]
    quaternion_tool0_xyzw: [qx, qy, qz, qw]

Contoh plan-only:
python3 robot_executor/execute_tool0_grasp_pipeline.py \
  --target_json outputs/test_default/vision_output/tool0_pregrasp_target.json \
  --robot_ip 192.168.0.10

Contoh execute sungguhan:
python3 robot_executor/execute_tool0_grasp_pipeline.py \
  --target_json outputs/test_default/vision_output/tool0_pregrasp_target.json \
  --robot_ip 192.168.0.10 \
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


# ============================================================
# PROJECT PATH
# ============================================================

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
# OPTIONAL CAPTURE CONFIG
# ============================================================

def try_load_capture_config():
    """
    capture_config.py opsional.
    Jika gagal karena beda conda/env atau python-dotenv tidak ada,
    script tetap jalan menggunakan argumen eksplisit/default path.
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
    if CFG is not None:
        if hasattr(CFG, "TOOL0_PREGRASP_TARGET_JSON"):
            return resolve_path(CFG.TOOL0_PREGRASP_TARGET_JSON)
        if hasattr(CFG, "VISION_OUTPUT_DIR"):
            return resolve_path(Path(CFG.VISION_OUTPUT_DIR) / "tool0_pregrasp_target.json")
    return resolve_path(DEFAULT_TARGET_JSON)


# ============================================================
# JSON HELPERS
# ============================================================

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

    required = ["translation_tool0_pregrasp", "quaternion_tool0_xyzw"]
    for key in required:
        if key not in data:
            raise KeyError(
                f"Target JSON tidak punya key '{key}'. "
                "Pastikan file berasal dari convert_gripper_tip_to_tool0_target.py"
            )

    pos = data["translation_tool0_pregrasp"]
    quat = data["quaternion_tool0_xyzw"]

    if len(pos) != 3:
        raise ValueError("translation_tool0_pregrasp harus berisi 3 angka.")
    if len(quat) != 4:
        raise ValueError("quaternion_tool0_xyzw harus berisi 4 angka.")

    return [float(v) for v in pos], [float(v) for v in quat], data, path


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
# ROBOTIQ SOCKET CONTROL
# ============================================================

class RobotiqSocket:
    """
    Kontrol Robotiq gripper via URCap socket port 63352.

    Syarat:
    - Robotiq URCap aktif di teach pendant
    - Gripper dikenali oleh UR controller
    - Port 63352 terbuka
    """

    def __init__(self, robot_ip, port=63352, timeout=3.0, enabled=True):
        self.robot_ip = robot_ip
        self.port = int(port)
        self.timeout = float(timeout)
        self.enabled = bool(enabled)
        self.sock = None

    def connect(self):
        if not self.enabled:
            print("[GRIPPER] Disabled. Tidak connect ke gripper.")
            return
        if not self.robot_ip:
            raise ValueError("--robot_ip wajib diisi jika gripper enabled.")

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
        return self.send_cmd(f"SET {name} {int(value)}")

    def get_var(self, name):
        return self.send_cmd(f"GET {name}")

    # def activate(self):
    #     print("[GRIPPER] Activating...")
    #     self.set_var("ACT", 1)
    #     self.set_var("GTO", 1)
    #     self.set_var("SPE", 255)
    #     self.set_var("FOR", 150)
    #     time.sleep(1.0)

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

    def status(self):
        print("[GRIPPER] Status:")
        for key in ["ACT", "GTO", "STA", "OBJ", "POS", "PRE", "SPE", "FOR"]:
            self.get_var(key)


# ============================================================
# MOVEIT HELPERS
# ============================================================

def normalize_plan_result(plan_result):
    """
    Kompatibel dengan MoveIt Commander ROS Noetic:
    - RobotTrajectory
    - tuple(success, trajectory, planning_time, error_code)
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
        print(f"[WARN] Gagal membaca current pose {eef_link}:", e)
    print("=================================\n")


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
    success, _ = normalize_plan_result(plan_result)
    print(f"[PLAN ONLY] {waypoint_name} success:", success)
    if not success:
        raise RuntimeError(f"Planning ke {waypoint_name} gagal.")
    return True


def plan_or_execute_pose(group, pose, eef_link, label, execute=False):
    print_pose(label, pose)

    group.clear_pose_targets()
    group.set_pose_target(pose, end_effector_link=eef_link)

    print(f"[PLAN] Planning to {label}...")
    plan_result = group.plan()
    success, trajectory = normalize_plan_result(plan_result)
    print(f"[PLAN] {label} success:", success)

    if not success:
        group.clear_pose_targets()
        raise RuntimeError(f"Planning ke {label} gagal. Robot tidak dieksekusi.")

    try:
        print(f"[PLAN] {label} trajectory points:", len(trajectory.joint_trajectory.points))
    except Exception:
        pass

    if execute:
        print(f"[EXECUTE] Moving to {label}...")
        ok = group.execute(trajectory, wait=True)
        group.stop()
        group.clear_pose_targets()
        print(f"[EXECUTE] {label} result:", ok)
        if not ok:
            raise RuntimeError(f"Execute ke {label} gagal.")
        rospy.sleep(0.5)
    else:
        print(f"[PLAN ONLY] {label} tidak dieksekusi.")

    return True


def cartesian_z_motion(group, eef_link, dz, label, execute=False, min_fraction=0.90, eef_step=0.005):
    """
    Gerak Cartesian berdasarkan sumbu Z base/reference frame.
    dz negatif = turun, dz positif = naik.
    """
    current_pose = group.get_current_pose(end_effector_link=eef_link).pose
    target_pose = deepcopy(current_pose)
    target_pose.position.z += float(dz)

    print_pose(f"{label} START", current_pose)
    print_pose(f"{label} TARGET", target_pose)

    waypoints = [target_pose]

    print(f"[CARTESIAN PLAN] {label}, dz={dz}, eef_step={eef_step}")
    plan, fraction = group.compute_cartesian_path(
        waypoints,
        float(eef_step),
        True,
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
# MAIN PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gabungan MoveIt pre-grasp + descend grasp + Robotiq close + lift."
    )

    # MoveIt args
    parser.add_argument("--move_group", default="manipulator")
    parser.add_argument("--reference_frame", default="base")
    parser.add_argument("--eef_link", default="tool0")
    parser.add_argument("--waypoints", default=str(DEFAULT_WAYPOINT_FILE))
    parser.add_argument("--target_json", default=None)
    parser.add_argument("--idle_name", default="IDLE")
    parser.add_argument("--observation_name", default="OBSERVATION")
    parser.add_argument("--skip_idle", action="store_true")
    parser.add_argument("--skip_observation", action="store_true")

    # Motion safety args
    parser.add_argument("--velocity", type=float, default=0.05)
    parser.add_argument("--acceleration", type=float, default=0.05)
    parser.add_argument("--planning_time", type=float, default=15.0)
    parser.add_argument("--planning_attempts", type=int, default=20)
    parser.add_argument("--position_tolerance", type=float, default=0.01)
    parser.add_argument("--orientation_tolerance", type=float, default=0.08)
    parser.add_argument("--cartesian_min_fraction", type=float, default=0.90)
    parser.add_argument("--cartesian_eef_step", type=float, default=0.005)

    # Grasp args
    parser.add_argument(
        "--grasp_down",
        type=float,
        default=0.08,
        help="Jarak turun dari pre-grasp ke grasp dalam meter. Default 0.08."
    )
    parser.add_argument(
        "--lift_up",
        type=float,
        default=0.12,
        help="Jarak naik setelah close gripper dalam meter. Default 0.12."
    )

    # Gripper args
    parser.add_argument("--robot_ip", default=None, help="IP UR5 untuk Robotiq socket 63352.")
    parser.add_argument("--gripper_port", type=int, default=63352)
    parser.add_argument("--disable_gripper", action="store_true")
    parser.add_argument("--open_speed", type=int, default=255)
    parser.add_argument("--open_force", type=int, default=150)
    parser.add_argument("--close_position", type=int, default=255)
    parser.add_argument("--close_speed", type=int, default=255)
    parser.add_argument("--close_force", type=int, default=150)
    parser.add_argument("--gripper_wait", type=float, default=1.0)

    # Execution mode
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Jika tidak diberikan, script hanya melakukan planning/check tanpa execute."
    )
    parser.add_argument(
        "--no_prompt",
        action="store_true",
        help="Jangan minta ENTER sebelum execute. Hanya berlaku jika --execute aktif."
    )

    args = parser.parse_args()

    target_json = resolve_path(args.target_json) if args.target_json else get_default_target_json()
    waypoint_file = resolve_path(args.waypoints)

    print("\n========== TOOL0 GRASP PIPELINE ==========")
    print("PROJECT_DIR:", PROJECT_DIR)
    print("capture_config:", "LOADED" if CFG is not None else "NOT LOADED")
    if CFG is not None:
        print("capture_config TEST_NAME:", getattr(CFG, "TEST_NAME", "N/A"))
        print("capture_config VISION_OUTPUT_DIR:", getattr(CFG, "VISION_OUTPUT_DIR", "N/A"))
    print("WAYPOINT_FILE:", waypoint_file)
    print("TARGET_JSON:", target_json)
    print("move_group:", args.move_group)
    print("reference_frame:", args.reference_frame)
    print("eef_link:", args.eef_link)
    print("execute:", args.execute)
    print("grasp_down:", args.grasp_down)
    print("lift_up:", args.lift_up)
    print("disable_gripper:", args.disable_gripper)
    print("==========================================\n")

    if args.execute and not args.no_prompt:
        print("PERIKSA SEBELUM EXECUTE:")
        print("1. Robot sudah remote/control mode dan tidak protective stop.")
        print("2. Speed slider teach pendant > 0.")
        print("3. Jalur IDLE -> OBSERVATION -> PREGRASP aman.")
        print("4. Nilai --grasp_down tidak terlalu dalam menabrak meja.")
        print("5. Manusia dan objek lain aman dari workspace robot.")
        input("Tekan ENTER untuk mulai EXECUTE, atau CTRL+C untuk batal... ")

    gripper = RobotiqSocket(
        robot_ip=args.robot_ip,
        port=args.gripper_port,
        enabled=not args.disable_gripper,
    )

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("execute_tool0_grasp_pipeline", anonymous=True)

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

    try:
        print_moveit_info(group, robot, args.eef_link)

        waypoints, loaded_waypoint_path = load_json(waypoint_file, "WAYPOINTS_JSON")
        print("[INFO] Waypoint file loaded:", loaded_waypoint_path)

        if args.idle_name not in waypoints:
            raise KeyError(f"Waypoint {args.idle_name} tidak ditemukan di {loaded_waypoint_path}")
        if args.observation_name not in waypoints:
            raise KeyError(f"Waypoint {args.observation_name} tidak ditemukan di {loaded_waypoint_path}")

        pos, quat, target_data, loaded_target_path = load_tool0_target(target_json)
        print("[INFO] Target file loaded:", loaded_target_path)
        print("[INFO] translation_tool0_pregrasp:", pos)
        print("[INFO] quaternion_tool0_xyzw:", quat)

        target_pose = make_pose(pos, quat)

        # Gripper connect lebih awal supaya kalau gagal langsung ketahuan sebelum robot turun.
        gripper.connect()
        if not args.disable_gripper:
            # gripper.activate()
            gripper.open(
                speed=args.open_speed,
                force=args.open_force,
                wait=args.gripper_wait,
            )

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
            print("[WARN] Skip OBSERVATION.")

        # 3. OBSERVATION -> TOOL0 PREGRASP
        plan_or_execute_pose(
            group=group,
            pose=target_pose,
            eef_link=args.eef_link,
            label="TOOL0 PREGRASP",
            execute=args.execute,
        )

        # 4. PREGRASP -> GRASP turun base-Z
        cartesian_z_motion(
            group=group,
            eef_link=args.eef_link,
            dz=-abs(float(args.grasp_down)),
            label="DESCEND TO GRASP",
            execute=args.execute,
            min_fraction=args.cartesian_min_fraction,
            eef_step=args.cartesian_eef_step,
        )

        # 5. Close gripper
        if args.execute:
            gripper.close(
                position=args.close_position,
                speed=args.close_speed,
                force=args.close_force,
                wait=args.gripper_wait,
            )
        else:
            print("[PLAN ONLY] Skip close gripper karena --execute tidak aktif.")

        # 6. Lift object
        cartesian_z_motion(
            group=group,
            eef_link=args.eef_link,
            dz=abs(float(args.lift_up)),
            label="LIFT AFTER GRASP",
            execute=args.execute,
            min_fraction=args.cartesian_min_fraction,
            eef_step=args.cartesian_eef_step,
        )

        print("\n========== PIPELINE SELESAI ==========")
        if args.execute:
            print("Robot selesai: IDLE/OBSERVATION -> pregrasp -> turun -> close -> lift.")
        else:
            print("Plan-only selesai. Jika semua plan aman di RViz, jalankan ulang dengan --execute.")
        print("======================================\n")

    finally:
        gripper.close_socket()
        try:
            moveit_commander.roscpp_shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
