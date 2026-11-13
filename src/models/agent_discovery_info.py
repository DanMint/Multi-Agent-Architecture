from pydantic import BaseModel

# src/models/agent_discovery_info.py

from typing import Literal
from pydantic import BaseModel, Field, AnyHttpUrl

"""
AgentDiscoveryInfo:	Registry -> Coordinator

AgentDiscoveryInfo: Defines the information returned by the Agent Registry to the Coordinator for agent discovery.

    - agent_id = Unique identifier of the agent within the system.

    - name = Human-readable name of the agent, retrieved from
      its A2A Agent Card. Examples:
        1. WebSearchAgent
        2. DocumentAgent
        3. CodeAgent

    - skills = List of skill identifiers describing what the
      agent can do. Retrieved from the A2A Agent Card.
      Examples:
        1. web_search
        2. document_analysis
        3. summarization

    - agent_card_url = HTTP URL where the coordinator can
      retrieve the agent's A2A Agent Card to discover its
      supported communication interfaces, protocol features,
      and authentication requirements.

    - status = Current operational state of the agent, as
      determined by the Agent Registry.

      This discovery model only returns healthy agents:
        1. healthy = Agent is responding to heartbeat probes
           and is considered operational by the registry.

    - max_concurrent_tasks = Maximum number of tasks the
      agent is configured to execute simultaneously.
      This represents configured capacity, not necessarily
      the number of currently available task slots.

Note:
    This model exposes only the information needed by the
    Coordinator for agent selection and discovery.

    Internal information such as system prompts, LLM settings,
    heartbeat configuration, session identifiers, and missed
    heartbeat counts is not exposed.
"""

class AgentDiscoveryInfo(BaseModel):
  # Identity
  agent_id: int
  agent_name: str | None

  # # A2A discovery
  # skills: list[str]
  # agent_card_url: AnyHttpUrl

  # Runtime availability
  status: Literal["active"]
  max_concurrent_tasks: int | None