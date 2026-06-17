#!/usr/bin/env python3

import argparse
import socket
import time


class RobotiqSocket:
    """
    Kontrol Robotiq gripper via URCap socket port 63352.

    Syarat:
    - Robotiq URCap aktif di teach pendant
    - Gripper dikenali oleh UR controller
    - Port 63352 terbuka
    """

    def __init__(self, robot_ip, port=63352, timeout=3.0):
        self.robot_ip = robot_ip
        self.port = port
        self.timeout = timeout
        self.sock = None

    def connect(self):
        print(f"[INFO] Connecting to Robotiq socket {self.robot_ip}:{self.port}")
        self.sock = socket.create_connection(
            (self.robot_ip, self.port),
            timeout=self.timeout
        )
        self.sock.settimeout(self.timeout)
        print("[OK] Connected")

    def close_socket(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def send_cmd(self, cmd):
        if self.sock is None:
            raise RuntimeError("Socket belum connect.")

        msg = cmd.strip() + "\n"
        self.sock.sendall(msg.encode("utf-8"))

        try:
            resp = self.sock.recv(1024).decode("utf-8", errors="ignore").strip()
        except socket.timeout:
            resp = ""

        print(f">> {cmd}")
        print(f"<< {resp}")
        return resp

    def set_var(self, name, value):
        return self.send_cmd(f"SET {name} {value}")

    def get_var(self, name):
        resp = self.send_cmd(f"GET {name}")
        return resp

    def activate(self):
        print("[INFO] Activating gripper...")
        self.set_var("ACT", 1)
        self.set_var("GTO", 1)
        self.set_var("SPE", 255)
        self.set_var("FOR", 150)
        time.sleep(1.0)

    def open(self, speed=255, force=150):
        print("[INFO] Opening gripper...")
        self.set_var("SPE", int(speed))
        self.set_var("FOR", int(force))
        self.set_var("POS", 0)
        self.set_var("GTO", 1)
        time.sleep(1.0)

    def close(self, position=255, speed=255, force=150):
        print("[INFO] Closing gripper...")
        self.set_var("SPE", int(speed))
        self.set_var("FOR", int(force))
        self.set_var("POS", int(position))
        self.set_var("GTO", 1)
        time.sleep(1.0)

    def half(self, speed=255, force=150):
        print("[INFO] Moving gripper half...")
        self.set_var("SPE", int(speed))
        self.set_var("FOR", int(force))
        self.set_var("POS", 128)
        self.set_var("GTO", 1)
        time.sleep(1.0)

    def status(self):
        print("[INFO] Reading gripper status...")
        for key in ["ACT", "GTO", "STA", "OBJ", "POS", "PRE", "SPE", "FOR"]:
            self.get_var(key)

    def cycle(self):
        self.activate()
        time.sleep(0.5)

        self.open()
        time.sleep(1.0)

        self.close()
        time.sleep(1.0)

        self.open()
        time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", required=True)
    parser.add_argument(
        "command",
        choices=["activate", "open", "close", "half", "status", "cycle"]
    )
    parser.add_argument("--position", type=int, default=255)
    parser.add_argument("--speed", type=int, default=255)
    parser.add_argument("--force", type=int, default=150)
    parser.add_argument("--port", type=int, default=63352)

    args = parser.parse_args()

    gripper = RobotiqSocket(args.robot_ip, port=args.port)

    try:
        gripper.connect()

        if args.command == "activate":
            gripper.activate()

        elif args.command == "open":
            gripper.activate()
            gripper.open(speed=args.speed, force=args.force)

        elif args.command == "close":
            gripper.activate()
            gripper.close(
                position=args.position,
                speed=args.speed,
                force=args.force
            )

        elif args.command == "half":
            gripper.activate()
            gripper.half(speed=args.speed, force=args.force)

        elif args.command == "status":
            gripper.status()

        elif args.command == "cycle":
            gripper.cycle()

    finally:
        gripper.close_socket()


if __name__ == "__main__":
    main()
