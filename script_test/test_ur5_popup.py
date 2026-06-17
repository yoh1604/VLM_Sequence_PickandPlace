import socket

ROBOT_IP = "192.168.200.1"   # ganti dengan IP UR5 kamu
PORT = 30002

script = """
def pc_connection_test():
  popup("Hello from PC", title="UR5 Connection Test", blocking=False)
end
"""

with socket.create_connection((ROBOT_IP, PORT), timeout=5) as s:
    s.sendall(script.encode("utf-8"))

print("Popup script sent.")
