import socket
import time

ROBOT_IP = "192.168.200.1"   # ganti IP UR5 kamu
PORT = 30002

def send_urscript(script: str):
    with socket.create_connection((ROBOT_IP, PORT), timeout=5) as s:
        s.sendall(script.encode("utf-8"))
    print("Script sent.")

script = """
def small_safe_move():
  popup("Small move test starting", title="UR5 Test", blocking=False)
  sleep(1.0)

  # Gerakan kecil relatif terhadap TCP saat ini.
  # p[dx, dy, dz, rx, ry, rz]
  # dz = 0.02 artinya naik 2 cm relatif base/tool context yang digunakan pose_trans.
  movel(pose_trans(get_actual_tcp_pose(), p[0, 0, 0.05, 0, 0, 0]), a=0.1, v=0.03)

  sleep(1.0)
  popup("Small move test done", title="UR5 Test", blocking=False)
end
"""

send_urscript(script)
