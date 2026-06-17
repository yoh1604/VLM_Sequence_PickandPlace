#!/usr/bin/env python3

import argparse
import json
import math
import socket
import time
from pathlib import Path

import rospy
import moveit_commander


PROJECT_DIR = Path(__file__).resolve().parent.parent


def resolve_path(path_like):
    p = Path(str(path_like)).expanduser()
    if not p.is_absolute():
        p = PROJECT_DIR / p
    return p.resolve()


def load_waypoints(path):
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Waypoints JSON tidak ditemukan: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    required = ["DISCARD_STEP_2", "DISCARD", "DISCARD_TO_IDLE"]
    for k in required:
        if k not in data:
            raise KeyError(f"Waypoint {k} tidak ada di {path}")
        if len(data[k]) != 6:
            raise ValueError(f"Waypoint {k} harus punya 6 joint.")

    print("[OK] Loaded waypoints:", path)
    return data


def robotiq_open(robot_ip, port=63352):
    print("[GRIPPER] Opening gripper...")
    cmds = [
        "SET ACT 1",
        "SET GTO 1",
        "SET SPE 255",
        "SET FOR 150",
        "SET POS 0",
    ]

    with socket.create_connection((robot_ip, port), timeout=3.0) as sock:
        sock.settimeout(1.0)
        for cmd in cmds:
            sock.sendall((cmd + "\n").encode("utf-8"))
            try:
                resp = sock.recv(1024).decode("utf-8", errors="ignore").strip()
            except socket.timeout:
                resp = ""
            print(f"[GRIPPER] >> {cmd}")
            print(f"[GRIPPER] << {resp}")
            time.sleep(0.15)

    time.sleep(1.0)
    print("[GRIPPER] Open command sent.")


def go_joint_interpolated(group, target, label, execute=True, max_joint_step=0.20):
    current = group.get_current_joint_values()
    target = [float(x) for x in target]

    deltas = [target[i] - current[i] for i in range(6)]
    max_delta = max(abs(d) for d in deltas)

    n_steps = max(1, int(math.ceil(max_delta / float(max_joint_step))))

    print(f"\n[INTERP JOINT] {label}")
    print("[INTERP JOINT] current:", current)
    print("[INTERP JOINT] target :", target)
    print("[INTERP JOINT] max_delta_rad:", max_delta)
    print("[INTERP JOINT] n_steps:", n_steps)

    if not execute:
        print("[PLAN ONLY] Tidak execute.")
        return True

    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        q = [current[i] + alpha * deltas[i] for i in range(6)]

        print(f"\n[INTERP JOINT] {label} substep {step}/{n_steps}")
        print("[INTERP JOINT] q:", q)

        group.set_start_state_to_current_state()
        ok = group.go(q, wait=True)
        group.stop()
        group.clear_pose_targets()

        print(f"[INTERP JOINT] result substep {step}/{n_steps}:", ok)

        if not ok:
            raise RuntimeError(f"Gagal di {label} substep {step}/{n_steps}")

        rospy.sleep(0.3)

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", default="192.168.200.1")
    parser.add_argument("--waypoints_json", default="configs/waypoints_ur5.json")
    parser.add_argument("--group_name", default="manipulator")
    parser.add_argument("--velocity", type=float, default=0.02)
    parser.add_argument("--acceleration", type=float, default=0.02)
    parser.add_argument("--max_joint_step", type=float, default=0.20)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--disable_gripper", action="store_true")
    args = parser.parse_args()

    waypoints = load_waypoints(args.waypoints_json)

    rospy.init_node("recover_discard_from_step1", anonymous=True)
    moveit_commander.roscpp_initialize([])

    group = moveit_commander.MoveGroupCommander(args.group_name)
    group.set_max_velocity_scaling_factor(args.velocity)
    group.set_max_acceleration_scaling_factor(args.acceleration)
    group.set_planning_time(10.0)
    group.set_num_planning_attempts(10)

    print("\n========== RECOVER DISCARD FROM CURRENT ==========")
    print("robot_ip       :", args.robot_ip)
    print("velocity       :", args.velocity)
    print("acceleration   :", args.acceleration)
    print("max_joint_step :", args.max_joint_step)
    print("execute        :", args.execute)
    print("disable_gripper:", args.disable_gripper)
    print("current joints :", group.get_current_joint_values())
    print("==================================================\n")

    if not args.execute:
        input("Tekan ENTER untuk execute recovery, atau CTRL+C untuk batal... ")

    go_joint_interpolated(
        group,
        waypoints["DISCARD_STEP_2"],
        "DISCARD_STEP_2",
        execute=True,
        max_joint_step=args.max_joint_step,
    )

    go_joint_interpolated(
        group,
        waypoints["DISCARD"],
        "DISCARD",
        execute=True,
        max_joint_step=args.max_joint_step,
    )

    if not args.disable_gripper:
        robotiq_open(args.robot_ip)
    else:
        print("[GRIPPER] disabled, skip open.")

    go_joint_interpolated(
        group,
        waypoints["DISCARD_TO_IDLE"],
        "DISCARD_TO_IDLE",
        execute=True,
        max_joint_step=args.max_joint_step,
    )

    print("\n✅ Recovery discard selesai: STEP_2 -> DISCARD -> OPEN -> IDLE")


if __name__ == "__main__":
    main()
