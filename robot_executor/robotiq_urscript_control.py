#!/usr/bin/env python3

import argparse
import socket
import time


def send_urscript(robot_ip, script, port=30002, timeout=3.0):
    """
    Mengirim URScript ke UR controller.
    Default port 30002 = secondary interface.
    """
    print(f"[INFO] Connecting to UR robot {robot_ip}:{port}")

    with socket.create_connection((robot_ip, port), timeout=timeout) as s:
        s.sendall(script.encode("utf-8"))
        time.sleep(0.2)

    print("[INFO] URScript sent.")


def make_gripper_script(command):
    """
    Command ini butuh Robotiq URCap function tersedia:
    - rq_activate()
    - rq_open()
    - rq_close()
    - rq_move_and_wait(pos)
    """

    if command == "activate":
        body = """
  rq_activate()
  sleep(1.0)
"""

    elif command == "open":
        body = """
  rq_activate()
  sleep(0.5)
  rq_open()
  sleep(1.0)
"""

    elif command == "close":
        body = """
  rq_activate()
  sleep(0.5)
  rq_close()
  sleep(1.0)
"""

    elif command == "half":
        body = """
  rq_activate()
  sleep(0.5)
  rq_move_and_wait(128)
  sleep(1.0)
"""

    elif command == "cycle":
        body = """
  rq_activate()
  sleep(1.0)

  rq_open()
  sleep(1.0)

  rq_close()
  sleep(1.0)

  rq_open()
  sleep(1.0)

  rq_close()
  sleep(1.0)

  rq_open()
  sleep(1.0)
"""

    else:
        raise ValueError(f"Unknown command: {command}")

    script = f"""
def robotiq_gripper_test():
{body}
end
"""
    return script


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", required=True, help="IP robot UR5")
    parser.add_argument(
        "command",
        choices=["activate", "open", "close", "half", "cycle"],
        help="Perintah gripper"
    )
    parser.add_argument("--port", type=int, default=30002)

    args = parser.parse_args()

    script = make_gripper_script(args.command)

    print("\n========== URSCRIPT ==========")
    print(script)
    print("==============================\n")

    send_urscript(args.robot_ip, script, port=args.port)


if __name__ == "__main__":
    main()
