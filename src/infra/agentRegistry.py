import threading
import time
import socket
import json
import secrets

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import socket
from dataclasses import asdict

from src.models import AgentInfo


"""
AgentRegistry: registers new agents via HTTP and keeps track of current agents via UDP heartbeat signals

    - agent_amount : int = amount of agents registred by the registry
    - agent_ids : list = stores agent ids
    - agents : dict = stores agent id with the corresponding agent
    - lock : threading.Lock() = mutex lock to access shared data structures
    - app : FastAPI() = HTTP server for the registration
    - server : str = server IPv4 address of the server
    - server_thread = keeps track of the server thread
    
    - setup_routes() = sets up HTTP requests handling
    - start() = the main function of the class. it starts the http server, UDP server and sends UDP heartbeats
"""

class AgentRegistry:
    def __init__(self) -> None:
        # information about agents
        self.agent_amount : int = 0
        self.agent_ids : list[int] = []
        self.agents : dict[AgentInfo] = {}

        # shared data protection
        self.lock = threading.Lock()

        # HTTP server
        self.app = FastAPI()

        # HTTP server refrences
        self.http_server = None
        self.http_server_thread = None

        # UDP server refrences
        self.UDP_server = None
        self.UDP_server_thread = None
        self.stop_event = threading.Event()

        # HTTP endpoint configuration
        self.setup_routes()

    # Tells FastAPI which HTTP endpoints exists and how to handle them
    def setup_routes(self):

        @self.app.post("/register")
        async def register(request: Request):
            data = await request.json()
            agent = AgentInfo(**data)

            with self.lock:
                if agent.agent_id in self.agents:
                    raise HTTPException(
                        status_code = 409,
                        detail = "Agent already exists"
                    )
                else:
                    self.agent_amount += 1
                    self.agent_ids.append(agent.agent_id)
                    self.agents[agent.agent_id] = agent

            return {"message": "Agent registered"}

        @self.app.get("/agent/id")
        async def get_agent(agent_id: int):
            agent_info = None
            with self.lock:
                if agent_id in self.agents:
                    agent_info = self.agents[agent_id]
                else:
                    raise HTTPException(
                        status_code=404,
                        detail="Agent does not exist"
                    )

            return{"agent": agent_info}

        @self.app.get("/agent_amount")
        def get_amount_of_agents():
            agent_amount:int
            with self.lock:
                agent_amount = self.agent_amount

            return{"agent_amount": agent_amount}

    # heartbeat server
    def udp_server(self):
        UDP_HOST = "127.0.0.1"
        UDP_PORT = 9000

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_server:
            udp_server.bind((UDP_HOST, UDP_PORT))
            udp_server.settimeout(0.5)

            print(f"Server listening on {UDP_HOST}:{UDP_PORT}")

            while not self.stop_event.is_set():
                try: 
                    data, address = udp_server.recvfrom(1024)
                    message = data.decode("utf-8")
                    print(f"Recieved  from {address}: {message}")
                except socket.timeout:
                    continue

        print("UDP server stopped")

    def start_udp_server_thread(self):
        # preventing starting another udp server
        if self.UDP_server_thread and self.UDP_server_thread.is_alive():
            return

        self.UDP_server_thread = threading.Thread(
            target=self.udp_server,
            daemon=True
        )

        self.UDP_server_thread.start()

    # registration server of agents
    def start_http_server(self):
        # preventing starting another server
        if self.http_server_thread and self.http_server_thread.is_alive():
            return

        config = uvicorn.Config(
            self.app,
            host="127.0.0.1",
            port=8000,
            log_level="info"
        )

        self.http_server = uvicorn.Server(config)

        # run the server in a thread
        self.http_server_thread = threading.Thread(
            target=self.http_server.run,
            daemon=True
        )

        self.http_server_thread.start()

    def start(self):
        self.start_http_server()
        self.start_udp_server_thread()

        print("Registry HTTP server starting")

    def stop(self):
        # stopping udp server
        self.stop_event.set()

        if self.http_server:
            self.http_server.should_exit = True

        if self.http_server_thread:
            self.http_server_thread.join()

        if self.UDP_server_thread:
            self.UDP_server_thread.join()

        

def main():
    agent_registry = AgentRegistry()
    agent_registry.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        agent_registry.stop()

if __name__ == "__main__":
    main()


