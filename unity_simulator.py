import json
import socket

UNITY_IP = "10.125.4.65"
COMMAND_PORT = 5005
POSE_PORT = 5006

command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
pose_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
pose_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
pose_socket.bind(("", POSE_PORT))


def send_command(u1, u2):
    command = json.dumps({"u1": u1, "u2": u2}).encode("utf-8")
    command_socket.sendto(command, (UNITY_IP, COMMAND_PORT))


def receive_pose():
    data, _ = pose_socket.recvfrom(4096)
    pose = json.loads(data.decode("utf-8"))
    x = pose.get("x")
    y = pose.get("y")
    cos_value = pose.get("cos")
    sin_value = pose.get("sin")
    return x, y, cos_value, sin_value


if __name__ == "__main__":
    while True:  # Control loop
        send_command(0, 0)

        x, y, cos_value, sin_value = receive_pose()

        print(x, y, cos_value, sin_value)
