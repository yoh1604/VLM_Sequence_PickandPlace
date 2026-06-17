#!/usr/bin/env python3

import argparse
import copy
import json
import socket
import sys
import time
from pathlib import Path

import rospy
import moveit_commander


PROJECT_DIR = Path(__file__).resolve().parent.parent


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
        waypoints = json.load(f)

    required = [
        "DISCARD_INIT",
        "DISCARD_STEP_1",
        "DISCARD_STEP_2",
        "DISCARD",
        "DISCARD_TO_IDLE",
    ]

    for name in required:
        if name not in waypoints:
            raise KeyError(f"Waypoint '{name}' tidak ditemukan di {path}")
        if len(waypoints[name]) != 6:
            raise ValueError(f"Waypoint '{name}' harus berisi 6 joint value.")

    print("[OK] Loaded waypoints:", path)
    return waypoints


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


# ============================================================
# MOVEIT HELPERS
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


def execute_cartesian_lift(group, eef_link, lift_up, min_fraction=0.50):
    """
    Naik relatif dari posisi sekarang.
    Dipakai hanya untuk lift kecil 3 cm, supaya tidak memilih IK yang muter.
    """

    current_pose = group.get_current_pose(end_effector_link=eef_link).pose
    lift_pose = copy.deepcopy(current_pose)
    lift_pose.position.z += float(lift_up)

    print_pose("CURRENT POSE BEFORE LIFT", current_pose)
    print_pose("TARGET POSE AFTER LIFT", lift_pose)

    print(f"[LIFT] Cartesian lift {lift_up:.3f} m")

    try:
        plan, fraction = group.compute_cartesian_path(
            [lift_pose],
            0.003,
            True,
        )
    except TypeError:
        plan, fraction = group.compute_cartesian_path(
            [lift_pose],
            0.003,
            0.0,
        )

    print("[LIFT] fraction:", fraction)

    if fraction < min_fraction:
        raise RuntimeError(
            f"Cartesian lift gagal. fraction={fraction:.3f}, minimal={min_fraction:.3f}"
        )

    ok = group.execute(plan, wait=True)
    group.stop()
    group.clear_pose_targets()

    print("[LIFT] execute result:", ok)
    return ok

def go_joint_soft(group, joints, label, execute=True):
    try:
        return go_joint_direct(
            group,
            joints,
            label,
            execute=execute,
        )
    except Exception as e:
        print(f"[WARN] {label} reported failed:", e)
        print(f"[WARN] {label} dibuat non-fatal.")
        group.stop()
        group.clear_pose_targets()
        rospy.sleep(1.0)
        return False
    
def go_joint_direct(group, joints, label, execute=True):
    """
    Gerak waypoint seperti script go_to_idle.py:
    group.go(joints, wait=True)

    Ini lebih cocok untuk waypoint yang sudah kamu rekam.
    """

    print(f"\n[JOINT GO] Moving to {label}")
    print("[JOINT GO] target:", joints)
    print("[JOINT GO] current:", group.get_current_joint_values())

    if not execute:
        print(f"[JOINT GO] plan_only=True, skip execute {label}")
        return True

    ok = group.go(joints, wait=True)
    group.stop()
    group.clear_pose_targets()

    print(f"[JOINT GO] {label} result:", ok)

    if not ok:
        raise RuntimeError(f"Gerak ke {label} gagal.")

    return ok


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Setelah objek sudah tergenggam: naik 3 cm, lalu ikuti waypoint discard "
            "dengan group.go seperti go_to_idle.py."
        )
    )

    parser.add_argument("--robot_ip", default="192.168.200.1")
    parser.add_argument("--group_name", default="manipulator")
    parser.add_argument("--eef_link", default="tool0")

    parser.add_argument(
        "--waypoints_json",
        default="configs/waypoints_ur5.json",
        help="Path ke file waypoints_ur5.json.",
    )

    parser.add_argument(
        "--lift_up",
        type=float,
        default=0.03,
        help="Naik relatif dari posisi sekarang sebelum discard. Default 0.03 m.",
    )

    parser.add_argument("--velocity", type=float, default=0.05)
    parser.add_argument("--acceleration", type=float, default=0.05)

    parser.add_argument(
        "--disable_gripper",
        action="store_true",
        help="Tidak open gripper di posisi DISCARD.",
    )

    parser.add_argument(
        "--skip_lift",
        action="store_true",
        help="Langsung ke DISCARD_INIT tanpa naik 3 cm.",
    )

    parser.add_argument(
        "--plan_only",
        action="store_true",
        help="Tidak execute joint waypoint.",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Langsung execute tanpa prompt ENTER.",
    )

    args = parser.parse_args()

    waypoints = load_waypoints(args.waypoints_json)

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("lift_then_discard_direct_joint_go", anonymous=True)

    group = moveit_commander.MoveGroupCommander(args.group_name)

    group.set_max_velocity_scaling_factor(float(args.velocity))
    group.set_max_acceleration_scaling_factor(float(args.acceleration))
    group.set_planning_time(5.0)
    group.set_num_planning_attempts(5)

    print("\n========== LIFT 3CM -> DISCARD WAYPOINTS ==========")
    print("robot_ip:", args.robot_ip)
    print("group_name:", args.group_name)
    print("eef_link:", args.eef_link)
    print("waypoints_json:", resolve_path(args.waypoints_json))
    print("lift_up:", args.lift_up)
    print("velocity:", args.velocity)
    print("acceleration:", args.acceleration)
    print("disable_gripper:", args.disable_gripper)
    print("skip_lift:", args.skip_lift)
    print("plan_only:", args.plan_only)
    print("===================================================\n")

    print("[INFO] Current joints:")
    print(group.get_current_joint_values())

    print_pose(
        "CURRENT TOOL0 POSE",
        group.get_current_pose(end_effector_link=args.eef_link).pose,
    )

    print("URUTAN:")
    print("1. Objek diasumsikan SUDAH tergenggam.")
    print(f"2. Naik {args.lift_up:.3f} m dari posisi sekarang.")
    print("3. group.go(DISCARD_INIT).")
    print("4. group.go(DISCARD_STEP_1).")
    print("5. group.go(DISCARD_STEP_2).")
    print("6. group.go(DISCARD).")
    print("7. Open gripper.")
    print("8. group.go(DISCARD_TO_IDLE).")
    print()

    if not args.execute:
        input("Tekan ENTER untuk mulai, atau CTRL+C untuk batal... ")

    do_execute = not args.plan_only
    gripper = None

    try:
        # 1. Lift 3 cm dari posisi grasp sekarang
        if not args.skip_lift:
            execute_cartesian_lift(
                group,
                eef_link=args.eef_link,
                lift_up=args.lift_up,
                min_fraction=0.50,
            )
        else:
            print("[SKIP] Lift skipped.")

        # 2. Ikuti waypoint discard dengan group.go, seperti go_to_idle.py

        go_joint_direct(
            group,
            waypoints["DISCARD_STEP_1"],
            "DISCARD_STEP_1",
            execute=do_execute,
        )

        go_joint_direct(
            group,
            waypoints["DISCARD_STEP_2"],
            "DISCARD_STEP_2",
            execute=do_execute,
        )

        go_joint_direct(
            group,
            waypoints["UP_DISCARD"],
            "UP_DISCARD",
            execute=do_execute,
        )

        go_joint_direct(
            group,
            waypoints["DISCARD"],
            "DISCARD",
            execute=do_execute,
        )

        # 3. Open gripper di discard
        if not args.disable_gripper:
            gripper = RobotiqSocket(args.robot_ip)
            gripper.connect()
            gripper.activate()
            gripper.open()
        else:
            print("[GRIPPER] disabled, skip open.")

                # 4. Kembali ke idle / safe pose

        go_joint_soft(
            group,
            waypoints["UP_DISCARD"],
            "UP_DISCARD",
            execute=do_execute,
        )

        go_joint_soft(
            group,
            waypoints["DISCARD_STEP_2"],
            "DISCARD_STEP_2",
            execute=do_execute,
        )
        
        go_joint_soft(
            group,
            waypoints["DISCARD_TO_IDLE"],
            "DISCARD_TO_IDLE",
            execute=do_execute,
        )

        go_joint_soft(
            group,
            waypoints["IDLE"],
            "IDLE",
            execute=do_execute,
        )

    finally:
        if gripper is not None:
            gripper.close_socket()

    print("\n========== DONE ==========")
    print("Selesai: lift 3 cm -> discard waypoint -> open -> idle.")
    print("==========================\n")


if __name__ == "__main__":
    main()