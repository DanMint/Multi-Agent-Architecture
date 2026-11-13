import asyncio

import httpx
import pytest

from unittest.mock import Mock, MagicMock, AsyncMock

# Access module namespace to patch dependencies.
import src.agent.base as base_module

# Import the class being tested.
from src.agent.base import BaseAgent


"""
    Test Suite: BaseAgent

    Tests the BaseAgent class, including initialization,
    registration, asynchronous communication, and heartbeat setup.

    1. Sets up pytest fixtures:
        - Mocks the Ollama model and ChatOllama constructor.
        - Mocks LangChain's create_agent() and ainvoke().
        - Mocks asynchronous HTTP requests to the Agent Registry.
        - Mocks the UDP heartbeat service.
        - Creates a BaseAgent instance for testing.

    2. Tests agent construction:
        - Verifies AgentInfo metadata and configuration.
        - Verifies Ollama and LangChain agent creation.
        - Verifies heartbeat configuration.
        - Ensures registration does not occur during __init__().

    3. Tests asynchronous initialization:
        - Registers the agent using HTTP POST.
        - Verifies registration payload and response handling.
        - Stores the session ID returned by the Registry.
        - Starts the heartbeat after successful registration.

    4. Tests the async factory method:
        - Calls BaseAgent.create().
        - Verifies object construction and initialization.
        - Ensures the initialized agent is returned.
        - Verifies subclass initialization overrides.

    5. Tests agent communication:
        - Sends messages using the asynchronous ainvoke() method.
        - Verifies the request format and returned response.

    6. Tests registration failure handling:
        - Simulates an HTTP registration failure.
        - Verifies the exception is propagated.
        - Ensures heartbeat transmission does not start.

    7. Tests Coordinator mode:
        - Verifies that coordinator=True skips creation
          of the default LangChain agent.

    Note:
        All external dependencies are mocked.
        Tests do not require Ollama, the Agent Registry,
        or the UDP heartbeat server to be running.
"""

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_dependencies(monkeypatch):
    """
    Mock external dependencies used by BaseAgent.

    1. Mock ChatOllama constructor.
    2. Mock LangChain create_agent() and ainvoke().
    3. Mock httpx.AsyncClient and asynchronous HTTP POST.
    4. Mock Registry response containing a session ID.
    5. Mock Heartbeat constructor and service methods.
    """
    # Mock Ollama model
    mock_model = Mock()

    mock_chat_ollama = Mock(
        return_value=mock_model
    )

    # Mock LangChain agent
    mock_agent = Mock()
    mock_agent.ainvoke = AsyncMock(
        return_value={
            "messages": [
                {
                    "role": "user",
                    "content": "Hello"
                },
                {
                    "role": "assistant",
                    "content": "Hi there!"
                }
            ]
        }
    )

    mock_create_agent = Mock(
        return_value=mock_agent
    )

    # Mock Registry HTTP response
    mock_http_response = Mock()

    mock_http_response.json.return_value = {
        "message": "Agent registered",
        "session_id": "test-session-123",
        "status": "active"
    }

    mock_http_response.raise_for_status.return_value = None

    # Mock the asynchronous HTTP POST operation.
    mock_http_client = Mock()

    mock_http_client.post = AsyncMock(
        return_value=mock_http_response
    )

    # Mock the async context manager:
    # async with httpx.AsyncClient(...) as client:
    mock_http_context = MagicMock()

    mock_http_context.__aenter__ = AsyncMock(
        return_value=mock_http_client
    )

    mock_http_context.__aexit__ = AsyncMock(
        return_value=False
    )

    mock_async_client = Mock(
        return_value=mock_http_context
    )

    # Mock heartbeat service
    mock_heartbeat = Mock()

    mock_heartbeat_class = Mock(
        return_value=mock_heartbeat
    )

    # Apply monkeypatches
    monkeypatch.setattr(
        base_module,
        "ChatOllama",
        mock_chat_ollama
    )

    monkeypatch.setattr(
        base_module,
        "create_agent",
        mock_create_agent
    )

    monkeypatch.setattr(
        base_module.httpx,
        "AsyncClient",
        mock_async_client
    )

    monkeypatch.setattr(
        base_module,
        "Heartbeat",
        mock_heartbeat_class
    )

    return {
        "model": mock_model,
        "chat_ollama": mock_chat_ollama,
        "agent": mock_agent,
        "create_agent": mock_create_agent,
        "async_client": mock_async_client,
        "http_client": mock_http_client,
        "http_response": mock_http_response,
        "heartbeat": mock_heartbeat,
        "heartbeat_class": mock_heartbeat_class
    }


@pytest.fixture
def agent(mock_dependencies):
    """
    Create a BaseAgent instance using mocked dependencies.

    Note:
        __init__() only constructs the agent.
        Async initialization is tested separately.
    """
    return BaseAgent(
        model="llama3.1:8b",
        temperature=0,
        system_prompt="You are a helpful assistant",
        agent_prompt="Test agent prompt",
        agent_id=1,
        agent_name="TestAgent"
    )


# ============================================================
# Agent Construction Tests
# ============================================================

def test_agent_initialization(agent, mock_dependencies):
    """
    Verify that BaseAgent.__init__() initializes:

        1. AgentInfo metadata.
        2. Ollama model.
        3. LangChain agent.
        4. Registry endpoint configuration.
        5. Heartbeat object.

    Also verify that registration and heartbeat transmission
    have not started during object construction.
    """
    # Verify agent metadata.
    assert agent.agent_info.agent_id == 1
    assert agent.agent_info.agent_name == "TestAgent"

    assert agent.agent_info.llm_model == "llama3.1:8b"
    assert agent.agent_info.llm_temperature == 0

    assert (
        agent.agent_info.system_prompt
        == "You are a helpful assistant"
    )

    assert (
        agent.agent_info.agent_prompt
        == "Test agent prompt"
    )

    assert agent.agent_info.session_id is None

    # Verify Registry endpoint configuration.
    assert (
        agent.agent_info.registry_register_agent_url
        == "http://127.0.0.1:8000/register"
    )

    assert (
        agent.agent_info.registry_get_all_agents
        == "http://127.0.0.1:8000/get_all_agents"
    )

    # Verify ChatOllama constructor.
    mock_dependencies["chat_ollama"].assert_called_once_with(
        model="llama3.1:8b",
        temperature=0
    )

    # Verify LangChain agent constructor.
    mock_dependencies["create_agent"].assert_called_once_with(
        model=mock_dependencies["model"],
        tools=[],
        system_prompt="You are a helpful assistant"
    )

    # Verify heartbeat constructor.
    mock_dependencies["heartbeat_class"].assert_called_once_with(
        agent_id=1,
        registry_host="127.0.0.1",
        registry_port=9000,
        interval=1
    )

    # Registration should not happen during __init__().
    mock_dependencies["async_client"].assert_not_called()

    # Heartbeat should not start during __init__().
    mock_dependencies["heartbeat"].start.assert_not_called()


# ============================================================
# Registration Tests
# ============================================================

def test_agent_registration(agent, mock_dependencies):
    """
    Verify asynchronous registration with the Agent Registry.

    Assertions:
        1. HTTP AsyncClient is created correctly.
        2. POST request targets the registration endpoint.
        3. Registration payload contains the expected fields.
        4. HTTP response is validated.
        5. Registry session ID is stored in AgentInfo.
    """
    # Execute the async registration method.
    asyncio.run(
        agent.register_agent(
            agent.agent_info.agent_card_url
        )
    )

    # Verify asynchronous HTTP client configuration.
    mock_dependencies["async_client"].assert_called_once_with(
        timeout=0.5
    )

    # Verify HTTP POST was awaited.
    mock_post = mock_dependencies["http_client"].post

    mock_post.assert_awaited_once()

    args, kwargs = mock_post.await_args

    # Verify endpoint.
    assert args[0] == "http://127.0.0.1:8000/register"

    # Verify registration payload.
    registration_data = kwargs["json"]

    assert registration_data["agent_id"] == 1
    assert registration_data["agent_name"] == "TestAgent"
    assert registration_data["agent_card_url"] is None
    assert registration_data["max_concurrent_tasks"] is None

    # Verify HTTP status validation.
    mock_dependencies[
        "http_response"
    ].raise_for_status.assert_called_once()

    # Verify session ID was stored.
    assert agent.agent_info.session_id == "test-session-123"

    # register_agent() does not start the heartbeat itself.
    mock_dependencies["heartbeat"].start.assert_not_called()


# ============================================================
# Asynchronous Initialization Tests
# ============================================================

def test_agent_async_initialize(agent, mock_dependencies):
    """
    Verify that initialize() performs asynchronous setup.

    Assertions:
        1. Agent registers with the Registry.
        2. Registry assigns a session ID.
        3. Heartbeat starts after successful registration.
    """
    asyncio.run(agent.initialize())

    # Verify registration HTTP request.
    mock_dependencies[
        "http_client"
    ].post.assert_awaited_once()

    # Verify session ID.
    assert agent.agent_info.session_id == "test-session-123"

    # Verify heartbeat started.
    mock_dependencies[
        "heartbeat"
    ].start.assert_called_once_with()


# ============================================================
# Async Factory Method Tests
# ============================================================

def test_agent_async_factory(mock_dependencies):
    """
    Verify that BaseAgent.create():

        1. Constructs a BaseAgent instance.
        2. Calls and awaits initialize().
        3. Registers the agent.
        4. Starts the heartbeat.
        5. Returns the initialized instance.
    """
    agent = asyncio.run(
        BaseAgent.create(
            model="llama3.1:8b",
            temperature=0,
            system_prompt="You are a helpful assistant",
            agent_prompt="Test agent prompt",
            agent_id=2,
            agent_name="FactoryAgent"
        )
    )

    # Verify returned object type.
    assert isinstance(agent, BaseAgent)

    # Verify agent metadata.
    assert agent.agent_info.agent_id == 2
    assert agent.agent_info.agent_name == "FactoryAgent"

    # Verify registration completed.
    assert agent.agent_info.session_id == "test-session-123"

    mock_dependencies[
        "http_client"
    ].post.assert_awaited_once()

    # Verify heartbeat started during initialization.
    mock_dependencies[
        "heartbeat"
    ].start.assert_called_once_with()


def test_async_factory_subclass_override(mock_dependencies):
    """
    Verify that the inherited BaseAgent.create() method
    calls a subclass's overridden initialize() method.
    """
    class TestWorkerAgent(BaseAgent):

        async def initialize(self):
            self.custom_initialized = True
            await super().initialize()

    agent = asyncio.run(
        TestWorkerAgent.create(
            model="llama3.1:8b",
            temperature=0,
            system_prompt="Test",
            agent_prompt="Test worker",
            agent_id=3,
            agent_name="WorkerAgent"
        )
    )

    # The factory must create the subclass, not BaseAgent.
    assert isinstance(agent, TestWorkerAgent)

    # Verify overridden initialization executed.
    assert agent.custom_initialized is True

    # Verify parent initialization also executed.
    assert agent.agent_info.session_id == "test-session-123"

    mock_dependencies["heartbeat"].start.assert_called_once()


# ============================================================
# Heartbeat Tests
# ============================================================

def test_agent_heartbeat(agent, mock_dependencies):
    """
    Verify heartbeat initialization and startup.

    Assertions:
        1. Heartbeat constructor receives correct configuration.
        2. Heartbeat is not started during __init__().
        3. Heartbeat starts during initialize().
    """
    mock_dependencies["heartbeat_class"].assert_called_once_with(
        agent_id=1,
        registry_host="127.0.0.1",
        registry_port=9000,
        interval=1
    )

    # Before async initialization.
    mock_dependencies[
        "heartbeat"
    ].start.assert_not_called()

    # Initialize the agent.
    asyncio.run(agent.initialize())

    # After async initialization.
    mock_dependencies[
        "heartbeat"
    ].start.assert_called_once_with()


# ============================================================
# Agent Communication Tests
# ============================================================

def test_agent_communication(agent, mock_dependencies):
    """
    Verify asynchronous communication with LangChain.

    Assertions:
        1. communicate() calls ainvoke().
        2. Correct user message is passed.
        3. The agent response is returned.
    """
    response = asyncio.run(agent.communicate("Hello"))

    # Verify asynchronous LangChain invocation.
    mock_dependencies["agent"].ainvoke.assert_awaited_once_with(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello"
                }
            ]
        }
    )

    # Verify response content.
    assert response["messages"][1]["content"] == "Hi there!"


# ============================================================
# Coordinator Mode Tests
# ============================================================

def test_coordinator_mode(mock_dependencies):
    """
    Verify that coordinator=True skips creation of the
    default LangChain agent.

    Assertions:
        1. AgentInfo is initialized.
        2. Ollama model is initialized.
        3. create_agent() is not called.
        4. Heartbeat object is created.
    """
    coordinator = BaseAgent(
        model="llama3.1:8b",
        temperature=0,
        system_prompt="You are a coordinator",
        agent_prompt="Coordinate worker agents",
        agent_id=5,
        agent_name="coordinator",
        coordinator=True
    )

    assert coordinator.agent_info.agent_id == 5
    assert coordinator.agent_info.agent_name == "coordinator"

    # LLM constructor is still called.
    mock_dependencies["chat_ollama"].assert_called_once()

    # Default LangChain agent must not be constructed.
    mock_dependencies["create_agent"].assert_not_called()

    # Heartbeat object should still exist.
    mock_dependencies["heartbeat_class"].assert_called_once()

# ============================================================
# Registration Failure Tests
# ============================================================

def test_registration_failure(agent, mock_dependencies):
    """
    Verify behavior when Registry registration fails.

    Assertions:
        1. HTTP error is propagated.
        2. Agent session ID remains unset.
        3. Heartbeat service does not start.
    """
    # Simulate Registry returning HTTP 500.
    mock_dependencies["http_response"].raise_for_status.side_effect = httpx.HTTPStatusError(
        "Registry unavailable",
        request=httpx.Request(
            "POST",
            "http://127.0.0.1:8000/register"
        ),
        response=httpx.Response(500)
    )

    # Initialization should raise an HTTP error.
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(agent.initialize())

    # Verify registration was attempted.
    mock_dependencies["http_client"].post.assert_awaited_once()

    # Session ID should remain unset.
    assert agent.agent_info.session_id is None

    # Heartbeat object was created during __init__(),
    # but must NOT be started if registration fails.
    mock_dependencies["heartbeat"].start.assert_not_called()

def test_async_factory_registration_failure(mock_dependencies):
    """
    Verify that BaseAgent.create() does not return an
    initialized agent when registration fails.

    Assertions:
        1. Factory propagates the registration exception.
        2. Heartbeat transmission never starts.
    """

    mock_dependencies["http_response"].raise_for_status.side_effect = httpx.HTTPStatusError(
        "Registration failed",
        request=httpx.Request(
            "POST",
            "http://127.0.0.1:8000/register"
        ),
        response=httpx.Response(500)
    )

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(
            BaseAgent.create(
                model="llama3.1:8b",
                temperature=0,
                system_prompt="Test",
                agent_prompt="Test prompt",
                agent_id=6,
                agent_name="FailedAgent"
            )
        )

    # Heartbeat constructor runs during __init__().
    mock_dependencies["heartbeat_class"].assert_called_once()

    # Heartbeat must not start after failed registration.
    mock_dependencies["heartbeat"].start.assert_not_called()