import secrets

from typing import Literal
from pydantic import BaseModel, Field

from src.models.agent_registration import AgentRegistration

"""
RegistryAgentRecord: Internal registry record containing
the agent's registration information and runtime state.

    - registration = Information provided by the agent.

    - session_id = Unique identifier for the current
      registered instance of the agent.

    - status = Current operational state of the agent:
        1. registered
        2. healthy
        3. suspect
        4. unavailable

    - last_heartbeat = Timestamp of the most recently
      accepted heartbeat.

    - missed_heartbeats = Number of consecutive heartbeat
      checks the agent has missed.
"""

class RegistryAgentRecord(BaseModel):
    # Information submitted during registration
    registration: AgentRegistration

    # # Information retrieved from the Agent Card
    # name: str
    # skills: list[str]

    # Registry-managed runtime information
    session_id: str = Field(default_factory=lambda: secrets.token_urlsafe(16))

    status: Literal[
        "registered",
        "healthy",
        "suspect",
        "unavailable"
    ] = "registered"

    last_heartbeat: float | None = None
    missed_heartbeats: int = 0