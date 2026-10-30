import pytest
from fastapi.testclient import TestClient
import socket 
import time

from src.infra.agentRegistry import AgentRegistry


@pytest.fixture
def registry():
    return AgentRegistry()

@pytest.fixture
def client(registry):
    with TestClient(registry.app) as test_client:
        yield test_client

@pytest.fixture
def agent1_data():
    return {
        "agent_id": 1,
        "capabilities": None,
        "llm_model": "llama3.1:8b",
        "llm_temperature": 0,
        "agent_system_prompt": "You are a helpful assistant",
        "endpoint": None,
        "udp_host": None,
        "udp_port": None,
        "session_id": None,
        "status": None,
        "last_heartbeat": None,
        "missed_hearbeats": None,
        "max_concurrent_tasks": None
    }

@pytest.fixture
def agent2_data():
    return {
        "agent_id": 2,
        "capabilities": None,
        "llm_model": "llama3.1:8b",
        "llm_temperature": 0,
        "agent_system_prompt": "You are a helpful assistant",
        "endpoint": None,
        "udp_host": None,
        "udp_port": None,
        "session_id": None,
        "status": None,
        "last_heartbeat": None,
        "missed_hearbeats": None,
        "max_concurrent_tasks": None
    }

"""
    This test tests registering one agent to the registery and asserts the following:
        1. response code is 200
        2. amount of agents in registry is 1
        3. amount of agents stored is 1
"""
def test_one_register_agent(client, registry, agent1_data):
    response = client.post("/register", json=agent1_data)

    assert response.status_code == 200
    assert registry.agent_amount == 1
    assert 1 in registry.agents

"""
    This test registers two agents and asserts the following:
        1. response code is 200 for both agents
        2. amount of agents in registry is 2
        3. amount of agents stored is 2
"""
def test_two_register_agent(client, registry, agent1_data, agent2_data):
    response1 = client.post("/register", json=agent1_data)
    response2 = client.post("/register", json=agent2_data)

    assert response1.status_code == 200
    assert response2.status_code == 200

    assert registry.agent_amount == 2
    assert 2 in registry.agents

"""
    This test makes sure that two agents with the same ids wont get registered and asserts the following
        1. response code is 409
        2. amount of agents in registry is 1
        3. amount of agents stored is 1
"""
def test_duplicate_registration(client, registry, agent1_data):
    client.post("/register", json=agent1_data)
    response = client.post("/register", json=agent1_data)

    assert response.status_code == 409
    assert registry.agent_amount == 1
    assert 1 in registry.agents

"""
    This test retrieves the agent from the registry using the agents id and asserts the following:
        1. response code is 200 when registering the agent
        2. amount of agents in registry is 1
        3. amount of agents stored is 1
        4. when receiving agent from registry the id of the agent is 1  
"""
def test_agent_by_id_retrieval(client, registry, agent1_data):
    client.post("/register", json=agent1_data)
    response = client.get("/agent/id", params={"agent_id": 1})

    assert response.status_code == 200
    assert registry.agent_amount == 1
    assert 1 in registry.agents
    assert response.json()["agent"]["agent_id"] == 1

"""
    This test retrieves two agents from the registry using thier ids and asserts the following:
        1. There are two response codes 200 when regisering the two agents
        2. amount of agents in registry is 2
        3. amount of agents stored is 2
        4. when receiving agent1 from registry the id of the agent is 1  
        5. when receiving agent2 from registry the id of the agent is 2
"""
def test_agents_by_id_retreivals(client, registry, agent1_data, agent2_data):
    client.post("/register", json=agent1_data)
    client.post("/register", json=agent2_data)

    response_agent1 = client.get("/agent/id", params={"agent_id": 1})
    response_agent2 = client.get("/agent/id", params={"agent_id": 2})

    assert response_agent1.status_code == 200
    assert response_agent2.status_code == 200

    assert registry.agent_amount == 2
    assert 2 in registry.agents

    assert response_agent1.json()["agent"]["agent_id"] == 1
    assert response_agent2.json()["agent"]["agent_id"] == 2

"""
    This test gets the amount of agents in the registry and asserts the following
        1. response has the status code of 200
        2. amount of agents in registry is 1
        3. amount of agents stored is 1
        4. response from the registry for agent_amount is 1
"""
def test_amount_of_agents(client, registry, agent1_data):
    client.post("/register", json=agent1_data)
    response = client.get("/agent_amount")

    assert response.status_code == 200
    assert registry.agent_amount == 1
    assert 1 in registry.agents
    assert response.json()["agent_amount"] == 1

"""
    This test gets the amount of agents in the registry and asserts the following
        1. response has the status code of 200
        2. amount of agents in registry is 2
        3. amount of agents stored is 2
        4. response from the registry for agent_amount is 2
"""
def test_amount_of_agents(client, registry, agent1_data, agent2_data):
    client.post("/register", json=agent1_data)
    client.post("/register", json=agent2_data)
    
    response = client.get("/agent_amount")

    assert response.status_code == 200
    assert registry.agent_amount == 2
    assert 2 in registry.agents
    assert response.json()["agent_amount"] == 2

"""
    This test verifies that the registry receives UDP heartbeat messages and asserts the following:
        1. UDP server thread starts successfully
        2. UDP heartbeat message is sent to the registry
        3. Registry successfully receives the heartbeat message
        4. Received heartbeat message appears in the server output
        5. UDP server thread stops successfully after the test
"""
def test_udp_heartbeat(registry, capfd):
    registry.start_udp_server_thread()
    heartbeat_recieved = False

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline + 2:
                # send heartbeat
                sock.sendto(
                    b"heartbeat-agent-1",
                    ("127.0.0.1", 9000)
                )

                time.sleep(0.05)

                # read the output printed by the UDP server
                output = capfd.readouterr().out

                if "heartbeat-agent-1" in output:
                    heartbeat_recieved = True
                    break

            assert heartbeat_recieved

    finally:
        # stop udp thread
        registry.stop()

    assert not registry.UDP_server_thread.is_alive()
    