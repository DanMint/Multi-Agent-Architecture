import pytest
import httpx
from unittest.mock import Mock

# access the module namespaces to patch dependencies inside it
import src.agent.base as base_module

# access the class directly when constructing test agents
from src.agent.base import BaseAgent

"""
    pytest.fixture that does the following:
        1. Mocks ollama model
            mock_model = Mock() -> model = ChatOllama(...)
            mock_chat_ollama = Mock(return_value=mock_model) -> represents the consructor
        2. Mocks langchain agent
        3. Mocks the registry HTTP responses
        4. Mocks heartbeat services
"""
@pytest.fixture
def mock_dependencies(monkeypatch):
    # Mock the LLM model
    mock_model = Mock()
    mock_chat_ollama = Mock(return_value=mock_model)

    # Mock the LangChain agent
    mock_agent = Mock()
    mock_agent.invoke.return_value = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
    }
    mock_create_agent = Mock(return_value=mock_agent)

    # Mock the registry HTTP response
    mock_response = Mock()
    mock_response.json.return_value = {"status": "registered"}
    mock_response.raise_for_status.return_value = None
    mock_http_post = Mock(return_value=mock_response)

    # Mock the heartbeat service
    mock_heartbeat = Mock()
    mock_heartbeat_class = Mock(return_value=mock_heartbeat)

    monkeypatch.setattr(base_module, "ChatOllama", mock_chat_ollama)
    monkeypatch.setattr(base_module, "create_agent", mock_create_agent)
    monkeypatch.setattr(base_module.httpx, "post", mock_http_post)
    monkeypatch.setattr(base_module, "Heartbeat", mock_heartbeat_class)

    return {
        "model": mock_model,
        "chat_ollama": mock_chat_ollama,
        "agent": mock_agent,
        "create_agent": mock_create_agent,
        "http_post": mock_http_post,
        "http_response": mock_response,
        "heartbeat": mock_heartbeat,
        "heartbeat_class": mock_heartbeat_class
    }

"""
    pytest.fixture agent that depends on mock_dependencies. It first makes sure that mock_dependencies is completed
    AgentInfo gets created with the values supplied from mock_dependencies fixture.
"""
@pytest.fixture
def agent(mock_dependencies):
    return BaseAgent(
        model="llama3.1:8b",
        temperature=0,
        system_prompt="You are a helpful assistant",
        agent_id=1
    )

"""
    This test tests the following:
        1. testing agent meta data
        2. testing ollama constructor call
        3. testing  
"""
def test_agent_initialization(agent, mock_dependencies):
    # testing agent metadata
    assert agent.agent_info.agent_id == 1
    assert agent.agent_info.llm_model == "llama3.1:8b"
    assert agent.agent_info.llm_temperature == 0
    assert agent.agent_info.agent_system_prompt == "You are a helpful assistant"

    # testing ollama constructor call
    mock_dependencies["chat_ollama"].assert_called_once_with(
        model="llama3.1:8b",
        temperature=0
    )

    # testing initialization of agent
    mock_dependencies["create_agent"].assert_called_once_with(
        model=mock_dependencies["model"],
        system_prompt="You are a helpful assistant"
    )

def test_agent_registration(agent, mock_dependencies):
    mock_post = mock_dependencies["http_post"]
    mock_post.assert_called_once()

    args, kwargs = mock_post.call_args
    assert args[0] == "http://127.0.0.1:8000/register"
    assert kwargs["json"]["agent_id"] == 1
    assert kwargs["json"]["llm_model"] == "llama3.1:8b"
    assert kwargs["timeout"] == 0.5

    mock_dependencies["http_response"].raise_for_status.assert_called_once()

"""
    this test verifies that base agent correctly initializes the heartbeat
"""
def test_agent_heartbeat(agent, mock_dependencies):
    mock_heartbeat_class = mock_dependencies["heartbeat_class"].assert_called_once_with(
        agent_id=1,
        registry_host="127.0.0.1",
        registry_port=9000,
        interval=1
    )

    mock_dependencies["heartbeat"].start.assert_called_once()

"""
    This test tests the communication between the user and the agent.
    More specifically it tests the communicate function in base agent
"""
def test_agent_communication(agent, mock_dependencies):
    response = agent.communicate("Hello")

    mock_dependencies["agent"].invoke.assert_called_once_with({
        "messages" : [
            {
                "role": "user",
                "content": "Hello"
            }
        ]
    })

    assert response["messages"][1]["content"] == "Hi there!"

"""
    This test assures no hearbeat is called when the agent did not successfuly register
"""
def test_registration_failure(mock_dependencies):
    mock_dependencies["http_response"].raise_for_status.side_effect = (
    httpx.HTTPStatusError(
        "Registry unavailable",
            request=httpx.Request(
                "POST", "http://127.0.0.1:8000/register"
            ),
            response=httpx.Response(500)
    )
)

    with pytest.raises(httpx.HTTPStatusError):
        BaseAgent(
            model="llama3.1:8b",
            temperature=0,
            system_prompt="Test",
            agent_id=2
        )

    # heartbeats should NOT start when registration fails
    mock_dependencies["heartbeat_class"].assert_not_called()