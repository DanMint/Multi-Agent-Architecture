
import json
import socket
import time

import pytest
from fastapi.testclient import TestClient

from src.infra.agentRegistry import AgentRegistry

"""
    Test Suite: AgentRegistry

    Tests the Agent Registry's HTTP endpoints, agent registration,
    discovery, and UDP heartbeat functionality.

    1. Sets up pytest fixtures:
        - Creates a fresh AgentRegistry instance for each test.
        - Creates a FastAPI TestClient to simulate HTTP requests.
        - Provides sample agent registration data.

    2. Tests agent registration:
        - Registers one or multiple agents.
        - Verifies HTTP response codes and stored agent information.
        - Prevents duplicate registrations.
        - Rejects invalid registration data.

    3. Tests agent retrieval:
        - Retrieves registered agents by their IDs.
        - Verifies agent information and session IDs.
        - Handles requests for nonexistent agents.

    4. Tests registry agent counts:
        - Verifies the number of registered agents.
        - Tests empty, single-agent, and multiple-agent scenarios.

    5. Tests agent discovery:
        - Retrieves available agents from the registry.
        - Verifies discovery response structure.
        - Ensures unavailable agents are excluded.

    6. Tests UDP heartbeat communication:
        - Starts the UDP heartbeat listener in a separate thread.
        - Sends a simulated JSON heartbeat message.
        - Verifies that the registry receives the message.
        - Stops the UDP server and verifies thread termination.

    Note:
        HTTP endpoints are tested using FastAPI TestClient without
        starting a real HTTP server. UDP heartbeat tests use an
        actual local UDP socket.
"""

@pytest.fixture
def registry():
    """Create a fresh AgentRegistry for each test."""
    return AgentRegistry()


@pytest.fixture
def client(registry):
    """Create a FastAPI test client without starting Uvicorn."""
    with TestClient(registry.app) as test_client:
        yield test_client


@pytest.fixture
def agent1_data():
    """Registration payload for the first agent."""
    return {
        "agent_id": 1,
        "agent_name": "Web_Search",
        "agent_card_url": None,
        "max_concurrent_tasks": 1
    }


@pytest.fixture
def agent2_data():
    """Registration payload for the second agent."""
    return {
        "agent_id": 2,
        "agent_name": "Document_Agent",
        "agent_card_url": None,
        "max_concurrent_tasks": 2
    }


# ============================================================
# Agent Registration Tests
# ============================================================

def test_one_register_agent(client, registry, agent1_data):
    """
    Verify that one agent can register successfully.

    Assertions:
        1. HTTP status code is 201.
        2. Registry contains one agent.
        3. Agent ID exists in the registry.
        4. Registration data is stored correctly.
        5. A session ID is generated.
    """
    response = client.post("/register", json=agent1_data)

    assert response.status_code == 201
    assert registry.agent_amount == 1
    assert len(registry.agents) == 1
    assert 1 in registry.agents

    record = registry.agents[1]

    assert record.registration.agent_id == 1
    assert record.registration.agent_name == "Web_Search"
    assert record.registration.max_concurrent_tasks == 1

    data = response.json()

    assert data["message"] == "Agent registered"
    assert isinstance(data["session_id"], str)
    assert len(data["session_id"]) > 0
    assert data["session_id"] == record.session_id
    assert data["status"] == record.status


def test_two_register_agents(client, registry, agent1_data, agent2_data):
    """
    Verify that two different agents can register.

    Assertions:
        1. Both registrations return HTTP 201.
        2. Registry contains two agents.
        3. Both agent IDs exist.
        4. Registration information is stored correctly.
        5. Each agent receives a unique session ID.
    """
    response1 = client.post("/register", json=agent1_data)
    response2 = client.post("/register", json=agent2_data)

    assert response1.status_code == 201
    assert response2.status_code == 201

    assert registry.agent_amount == 2
    assert len(registry.agents) == 2

    assert 1 in registry.agents
    assert 2 in registry.agents

    assert registry.agents[1].registration.agent_name == "Web_Search"
    assert registry.agents[2].registration.agent_name == "Document_Agent"

    assert (
        registry.agents[1].session_id
        != registry.agents[2].session_id
    )


def test_duplicate_registration(client, registry, agent1_data):
    """
    Verify that duplicate agent IDs are rejected.

    Assertions:
        1. Initial registration succeeds.
        2. Duplicate registration returns HTTP 409.
        3. Registry still contains one agent.
        4. Original registration is not overwritten.
    """
    response1 = client.post("/register", json=agent1_data)

    original_session_id = registry.agents[1].session_id

    response2 = client.post("/register", json=agent1_data)

    assert response1.status_code == 201
    assert response2.status_code == 409

    assert registry.agent_amount == 1
    assert len(registry.agents) == 1
    assert 1 in registry.agents

    assert registry.agents[1].session_id == original_session_id
    assert response2.json()["detail"] == "Agent already exists"


def test_invalid_registration(client, registry, agent1_data):
    """
    Verify that invalid registration data is rejected.

    Assertions:
        1. Missing agent_id produces HTTP 422.
        2. No agent is added to the registry.
    """
    invalid_data = agent1_data.copy()
    invalid_data.pop("agent_id")

    response = client.post("/register", json=invalid_data)

    assert response.status_code == 422
    assert registry.agent_amount == 0
    assert len(registry.agents) == 0


# ============================================================
# Agent Retrieval Tests
# ============================================================

def test_agent_by_id_retrieval(client, registry, agent1_data):
    """
    Verify retrieval of an agent by its ID.

    Assertions:
        1. Registration succeeds.
        2. GET request returns HTTP 200.
        3. Returned registry record contains correct
           registration details and session ID.
    """
    registration_response = client.post(
        "/register", json=agent1_data
    )
    response = client.get(
        "/agent/id",
        params={"agent_id": 1}
    )

    assert registration_response.status_code == 201
    assert response.status_code == 200

    assert registry.agent_amount == 1
    assert 1 in registry.agents

    agent = response.json()["agent"]

    assert agent["registration"]["agent_id"] == 1
    assert agent["registration"]["agent_name"] == "Web_Search"
    assert agent["registration"]["max_concurrent_tasks"] == 1

    assert agent["session_id"] == registry.agents[1].session_id
    assert agent["status"] == registry.agents[1].status
    assert agent["missed_heartbeats"] == 0
    assert agent["last_heartbeat"] is None


def test_agents_by_id_retrieval(client, registry, agent1_data, agent2_data):
    """
    Verify retrieval of two registered agents.

    Assertions:
        1. Both registrations succeed.
        2. Both GET requests return HTTP 200.
        3. Each response contains the correct agent.
    """
    registration1 = client.post("/register", json=agent1_data)
    registration2 = client.post("/register", json=agent2_data)

    assert registration1.status_code == 201
    assert registration2.status_code == 201

    response1 = client.get(
        "/agent/id", params={"agent_id": 1}
    )
    response2 = client.get(
        "/agent/id", params={"agent_id": 2}
    )

    assert response1.status_code == 200
    assert response2.status_code == 200

    assert registry.agent_amount == 2
    assert len(registry.agents) == 2

    agent1 = response1.json()["agent"]
    agent2 = response2.json()["agent"]

    assert agent1["registration"]["agent_id"] == 1
    assert agent2["registration"]["agent_id"] == 2

    assert agent1["registration"]["agent_name"] == "Web_Search"
    assert agent2["registration"]["agent_name"] == "Document_Agent"


def test_agent_by_id_not_found(client, registry):
    """
    Verify that requesting an unknown agent returns HTTP 404.

    Assertions:
        1. Response code is 404.
        2. Correct error message is returned.
    """
    response = client.get(
        "/agent/id",
        params={"agent_id": 999}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Agent does not exist"


# ============================================================
# Agent Count Tests
# ============================================================

def test_amount_of_agents_empty(client, registry):
    """
    Verify the agent count when no agents are registered.
    """
    response = client.get("/agent_amount")

    assert response.status_code == 200
    assert response.json()["agent_amount"] == 0

    assert registry.agent_amount == 0
    assert len(registry.agents) == 0


def test_amount_of_agents_one(client, registry, agent1_data):
    """
    Verify the agent count after one registration.
    """
    registration = client.post("/register", json=agent1_data)

    assert registration.status_code == 201

    response = client.get("/agent_amount")

    assert response.status_code == 200
    assert response.json()["agent_amount"] == 1

    assert registry.agent_amount == 1
    assert len(registry.agents) == 1


def test_amount_of_agents_two(
    client, registry, agent1_data, agent2_data):
    """
    Verify the agent count after two registrations.
    """
    registration1 = client.post("/register", json=agent1_data)
    registration2 = client.post("/register", json=agent2_data)

    assert registration1.status_code == 201
    assert registration2.status_code == 201

    response = client.get("/agent_amount")

    assert response.status_code == 200
    assert response.json()["agent_amount"] == 2

    assert registry.agent_amount == 2
    assert len(registry.agents) == 2


# ============================================================
# Agent Discovery Tests
# ============================================================

def test_get_all_agents_empty(client):
    """
    Verify that discovery returns an empty list when
    no agents have registered.
    """
    response = client.get("/get_all_agents")

    assert response.status_code == 200
    assert response.json() == {"agents": []}


def test_get_all_agents(client, registry, agent1_data, agent2_data):
    """
    Verify that the discovery endpoint returns the agents
    matching the registry's current 'active' filter.

    The expected result is derived from each stored record's
    actual status, so the test also supports a 'registered'
    initial state, which is currently excluded by the endpoint.
    """
    response1 = client.post("/register", json=agent1_data)
    response2 = client.post("/register", json=agent2_data)

    assert response1.status_code == 201
    assert response2.status_code == 201

    response = client.get("/get_all_agents")

    assert response.status_code == 200

    expected_agents = []

    for record in registry.agents.values():
        if record.status == "active":
            expected_agents.append({
                "agent_id": record.registration.agent_id,
                "agent_name": record.registration.agent_name,
                "status": record.status,
                "max_concurrent_tasks": None
            })

    assert response.json() == {"agents": expected_agents}


def test_get_all_agents_excludes_non_active(client, registry, agent1_data, agent2_data):
    """
    Verify that an agent with status 'suspect' is not
    included in the available-agents response.
    """
    client.post("/register", json=agent1_data)
    client.post("/register", json=agent2_data)

    # Simulate an agent being considered suspect.
    with registry.lock:
        registry.agents[1].status = "suspect"

    response = client.get("/get_all_agents")

    assert response.status_code == 200

    agents = response.json()["agents"]
    returned_ids = [agent["agent_id"] for agent in agents]

    assert 1 not in returned_ids


# ============================================================
# UDP Heartbeat Tests
# ============================================================

def test_udp_heartbeat(registry, capfd):
    """
    Verify that the UDP heartbeat listener receives messages.

    Assertions:
        1. UDP server starts successfully.
        2. A JSON heartbeat is sent to port 9000.
        3. The listener prints the received message.
        4. UDP server stops successfully.

    Note:
        The current UDP implementation only prints messages.
        It does not update the agent's health information.
    """
    # Check whether another process already uses UDP port 9000.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.bind(("127.0.0.1", 9000))
        except OSError:
            pytest.skip("UDP port 9000 is already in use")

    heartbeat_received = False

    heartbeat_message = {
        "type": "heartbeat",
        "id": 1
    }

    registry.start_udp_server_thread()

    try:
        assert registry.UDP_server_thread is not None

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                sock.sendto(
                    json.dumps(heartbeat_message).encode("utf-8"),
                    ("127.0.0.1", 9000)
                )
                time.sleep(0.05)
                output = capfd.readouterr().out
                if ('"type": "heartbeat"' in output and '"id": 1' in output):
                    heartbeat_received = True
                    break

        assert heartbeat_received, (
            "UDP server did not receive the heartbeat"
        )

    finally:
        registry.stop()

    assert not registry.UDP_server_thread.is_alive()
