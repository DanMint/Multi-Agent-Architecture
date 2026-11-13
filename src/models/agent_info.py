
from pydantic import BaseModel

"""
AgentInfo: Stores information about an agent, including its identity,
configuration, registry endpoints, communication settings, runtime
state, and execution capacity.

    - agent_id = Unique integer identifier assigned to an agent
      within the multi-agent system.

    - capabilities = List of tasks or capabilities supported by
      the agent. Can be None if capabilities have not yet been
      configured. Examples:
        1. web_search
        2. document_analysis
        3. summarization

    - llm_model = Name of the LLM used by the agent.
      Example: llama3.1:8b

    - llm_temperature = Temperature parameter used when generating
      LLM responses. Controls the randomness of generated output.

    - system_prompt = System-level instructions defining the
      agent's role, responsibilities, and behavioral constraints.

    - agent_prompt = Additional instructions or prompts used
      to configure the agent's behavior.

    - registry_url = Base HTTP URL of the Agent Registry.
      Example: http://127.0.0.1:8000/

    - registry_register_agent_url = HTTP endpoint used by the
      agent to register itself with the Agent Registry.

    - registry_get_agent_by_id_url = HTTP endpoint used to
      retrieve information about a specific registered agent.

    - registry_get_agent_amount_url = HTTP endpoint used to
      retrieve the total number of registered agents.

    - registry_get_all_agents = HTTP endpoint used to retrieve
      information about all registered agents.

    - endpoint = A2A communication endpoint used by other agents
      to send tasks and exchange messages with this agent.
      Can be None if the endpoint has not yet been configured.

    - udp_host = IP address or hostname where the agent listens
      for UDP heartbeat probes from the Agent Registry.
      Can be None if the UDP listener has not been configured.

    - udp_port = UDP port on which the agent listens for
      heartbeat probes. Can be None if not yet configured.

    - session_id = Identifier representing a specific running
      instance of the agent. Used to distinguish agent restarts
      or different execution sessions.

    - status = Current operational state of the agent.
      Can be None before its state has been initialized.
      Possible states:
        1. registered = Agent has registered but has not yet
           passed its first heartbeat check.
        2. healthy = Agent is responding to heartbeat probes
           and is considered operational.
        3. suspect = Agent has missed one or more heartbeat
           probes but has not exceeded the failure threshold.
        4. unavailable = Agent has exceeded the configured
           missed-heartbeat threshold and is considered offline.

    - last_heartbeat = Timestamp of the last successful
      heartbeat response received from the agent.
      Stored as a float (typically Unix time in seconds).
      Can be None if no heartbeat has been received.

    - missed_hearbeats = Number of consecutive heartbeat
      checks the agent has failed to respond to.
      Can be None before heartbeat tracking is initialized.

    - max_concurrent_tasks = Maximum number of tasks the agent
      is configured to execute simultaneously.
      Can be None if the execution capacity has not yet
      been configured.

Note:
    AgentInfo currently contains both local agent configuration
    and runtime metadata.

    The Agent Registry maintains information about registered
    agents, including their health and availability.

    AgentRegistration and AgentDiscoveryInfo are separate models
    used to define registration requests and the information
    exposed to the Coordinator, respectively.
"""

class AgentInfo(BaseModel):
    # identification and information
    agent_id: int
    agent_name: str
    capabilities: list[str] | None
    llm_model: str
    llm_temperature: int
    system_prompt: str
    agent_prompt: str
    agent_card_url: str | None

    # registry information
    registry_url: str
    registry_register_agent_url: str
    registry_get_agent_by_id_url: str
    registry_get_agent_amount_url: str
    registry_get_all_agents: str

    # agent to agent communication
    endpoint: str | None

    # udp heartbeat listener
    udp_host: str | None
    udp_port: int | None

    # runtime metadata
    session_id: str | None
    status: str | None
    last_heartbeat: float | None
    missed_hearbeats: int | None

    # optimal capacity information
    max_concurrent_tasks: int | None
