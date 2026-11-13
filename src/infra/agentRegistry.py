import threading
import time
import socket

import uvicorn
from fastapi import FastAPI, HTTPException
import socket

from src.models.agent_registration import AgentRegistration
from src.models.registry_agent_record import RegistryAgentRecord
from src.models.agent_discovery_info import AgentDiscoveryInfo

"""
AgentRegistry: Central registry responsible for agent registration, discovery,
and tracking registered agents within the multi-agent architecture.

    Uses FastAPI for HTTP communication and a separate UDP server for
    receiving heartbeat messages from registered agents.

    Attributes:
        - agent_amount: Total number of registered agents.
        - agent_ids: List of registered agent identifiers.
        - agents: Dictionary mapping agent IDs to RegistryAgentRecord objects.
        - lock: Threading mutex protecting shared registry data.
        - app: FastAPI application handling HTTP requests.
        - http_server: Uvicorn server handling HTTP communication.
        - http_server_thread: Background thread running the HTTP server.
        - UDP_server_thread: Background thread receiving UDP heartbeats.
        - stop_event: Threading event used to stop the UDP server.

    Methods:
        - __init__(): Initializes registry data structures, synchronization
          mechanisms, server configuration, and HTTP routes.

        - setup_routes(): Defines HTTP endpoints for agent registration,
          retrieval by ID, agent counts, and worker discovery.

        - udp_server(): Listens for incoming UDP heartbeat messages
          from registered agents.

        - start_udp_server_thread(): Starts the UDP listener in a daemon thread.

        - start_http_server(): Starts the FastAPI/Uvicorn HTTP server
          in a separate daemon thread.

        - start(): Starts both HTTP and UDP servers.

        - stop(): Signals both servers to stop and joins their threads.

        - main(): Creates and starts the Agent Registry, keeping it
          running until interrupted.

    HTTP Endpoints:
        - POST /register: Registers an agent and assigns a session ID.
        - GET /agent/id: Retrieves a registered agent by its ID.
        - GET /agent_amount: Returns the number of registered agents.
        - GET /get_all_agents: Returns discovery information for active agents.

    Note:
        UDP heartbeats are currently received and logged but do not yet
        update agent health status or detect unavailable agents.
"""

class AgentRegistry:
    def __init__(self) -> None:
        # information about agents
        self.agent_amount : int = 0
        self.agent_ids : list[int] = []
        self.agents : dict[int, RegistryAgentRecord] = {}

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

        @self.app.post("/register", status_code=201)
        async def register(registration: AgentRegistration):
            print(registration)
            with self.lock:
                if registration.agent_id in self.agents:
                    raise HTTPException(
                        status_code=409,
                        detail="Agent already exists"
                    )

                # create internal registry record
                record = RegistryAgentRecord(registration=registration)

                self.agents[registration.agent_id] = record
                self.agent_amount += 1
                self.agent_ids.append(registration.agent_id)

            print(f"REGISTRY: registered agent: {record.registration.agent_id}")

            return {
                    "message": "Agent registered",
                    "session_id": record.session_id,
                    "status": record.status
                }
        
        @self.app.get("/agent/id")
        def get_agent(agent_id: int):
            agent_info = None
            with self.lock:
                if agent_id not in self.agents:
                    raise HTTPException(
                            status_code=404,
                            detail="Agent does not exist"
                        )
                else:
                    agent_info = self.agents[agent_id]

            return{"agent": agent_info}

        @self.app.get("/agent_amount")
        def get_amount_of_agents():
            agent_amount = 0
            with self.lock:
                agent_amount = self.agent_amount

            return{"agent_amount": agent_amount}

        # this will be used mainly by the coordinator agent to get all of the avilable agents
        @self.app.get("/get_all_agents")
        def get_all_agents():
            with self.lock:
                avilable_agents = None

                avilable_agents = [
                    AgentDiscoveryInfo(
                        agent_id=agent.registration.agent_id,
                        agent_name=agent.registration.agent_name,
                        status = agent.status,
                        max_concurrent_tasks=None
                    ).model_dump(mode='json')

                    for agent in self.agents.values()
                    if agent.status == "active"
                ]

            return{"agents": avilable_agents}
                    
    # heartbeat server
    def udp_server(self):
        UDP_HOST = "0.0.0.0"
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
            host="0.0.0.0",
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
