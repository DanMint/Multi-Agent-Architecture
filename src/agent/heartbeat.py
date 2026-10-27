import socket
import threading
import json

class Heartbeat():
    def __init__(self, agent_id: int, registry_host: str = "127.0.0.1", registry_port: int = 9000, interval: float = 10.0):
        self.agent_id = agent_id
        self.registry_address = (
            registry_host,
            registry_port
        )

        self.interval = interval
        self.stop_event = threading.Event()

    def send_heartbeat(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            while not self.stop_event.is_set():
                message = {
                    "type": "heartbeat",
                    "id": self.agent_id
                }

                sock.sendto(
                    json.dumps(message).encode("utf-8"),
                    self.registry_address
                )

                self.stop_event.wait(self.interval)
            

    def start(self):
        self.thread = threading.Thread(
            target=self.send_heartbeat,
            daemon=True
        )

        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()