from pydantic import BaseModel, Field, AnyHttpUrl

"""
AgentRegistration:	Worker -> Registry

AgentRegistration: Defines the information an agent must provide
to the Agent Registry when registering itself.

    - agent_id = Unique identifier of the agent within the system.

    - agent_card_url = HTTP URL where the agent's A2A Agent Card
      can be retrieved. The Agent Card describes the agent's
      identity, skills, communication interfaces, and supported
      protocol features.

    - udp_host = IP address or hostname where the agent listens
      for UDP heartbeat probes from the registry.

    - udp_port = UDP port on which the agent listens for
      heartbeat probes. Must be between 1 and 65535.

    - max_concurrent_tasks = Maximum number of tasks the agent
      is configured to execute simultaneously. Must be at
      least 1.

Note:
    Runtime information such as status, last_heartbeat,
    missed_heartbeats, and session_id is managed by the
    Agent Registry and is not supplied through this model.
"""

class AgentRegistration(BaseModel):
    # identity
    agent_id: int = Field(ge=1)

    # A2A discovery
    agent_card_url: AnyHttpUrl | None

    # capacity
    max_concurrent_tasks: int = Field(default=1, ge=1) | None

