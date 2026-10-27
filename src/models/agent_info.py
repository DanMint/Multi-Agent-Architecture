
from dataclasses import dataclass, field

"""
AgentInfo: store information about agent 
    - agent_id = the unique id of the agent
    - capabilities = what the agent can do. Types of capabilities:
        1. web_search
        2. document_analysis
        3. summarization
    - endpoint = A2A endpoint for communcation with other agents
    - udp_host = the IP address where agent listens for UDP heartbeat probes
    - udp_port = the port for UDP heartbeat listening
    - session_id = identifies a particular running instance of an agent
    - status = represents the agents or operational state
        1. registered = agent has registered but hasnt passed its first heartbeat
        2. healthy = agent is responding to heatbeat probes
        3. suspect = agent missed one or more probes
        4. unavailable = agent exceeded the missed heartbeat threshold
    - last_heartbeat = last sucessful heartbeat response
    - missed_heartbeat = amount of consecutive missed heartbeats
    - max_concurrent_tasks = the maximum number of tasks the agent is configured to execute at the same time

"""

@dataclass
class AgentInfo:
    # identification and information
    agent_id: str
    capabilities: list[str]
    llm_model: str
    llm_temperature: int
    agent_system_prompt: str

    # agent to agent communication
    endpoint: str

    # udp heartbeat listener
    udp_host: str
    udp_port: int

    # runtime metdata
    session_id: str
    status: str
    last_heartbeat: float | None
    missed_hearbeats: int

    # optimal capacity information
    max_concurrent_tasks: int