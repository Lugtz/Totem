import json, os, socket

class BlenderBridge:
    def __init__(self, mode="file", file_path="data/blender/jaw.json", udp_host="127.0.0.1", udp_port=9001):
        self.mode = mode; self.file_path = file_path; self.udp_host = udp_host; self.udp_port = udp_port
        if self.mode == "file":
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def _send(self, payload: dict):
        if self.mode == "file":
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        elif self.mode == "udp":
            msg = json.dumps(payload).encode("utf-8")
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.sendto(msg, (self.udp_host, self.udp_port))
            s.close()

    def set_jaw(self, val: float):
        self._send({"jaw": float(val)})

    def set_head(self, yaw: float, pitch: float):
        self._send({"head": {"yaw": float(yaw), "pitch": float(pitch)}})
