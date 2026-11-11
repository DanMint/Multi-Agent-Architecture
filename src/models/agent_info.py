
from pydantic import BaseModel, ValidationError

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

class AgentInfo(BaseModel):
    # identification and information
    agent_id: int
    capabilities: list[str] | None
    llm_model: str
    llm_temperature: int
    system_prompt: str
    agent_prompt: str

    # registry information
    registry_url: str
    register_agent_url: str

    # agent to agent communication
    endpoint: str | None

    # udp heartbeat listener
    udp_host: str | None
    udp_port: int | None

    # runtime metdata
    session_id: str | None
    status: str | None
    last_heartbeat: float | None
    missed_hearbeats: int | None

    # optimal capacity information
    max_concurrent_tasks: int | None