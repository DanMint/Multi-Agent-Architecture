
"""
    Test Suite: CoordinatorAgent

    Tests the Coordinator Agent's initialization, worker discovery,
    task planning, selection, execution, and LangGraph workflow.

    1. Sets up pytest fixtures:
        - Mocks the Ollama model and structured LLM planner.
        - Mocks asynchronous HTTP Registry communication.
        - Mocks heartbeat services.
        - Provides sample AgentDiscoveryInfo data.
        - Creates a CoordinatorAgent instance for testing.

    2. Tests Coordinator initialization:
        - Verifies agent configuration and LangGraph setup.
        - Discovers registered workers.
        - Registers the Coordinator and starts its heartbeat.
        - Verifies the inherited asynchronous factory method.

    3. Tests agent discovery:
        - Retrieves available agents from the Registry.
        - Converts JSON dictionaries into Pydantic models.
        - Excludes the Coordinator from the worker dictionary.
        - Handles invalid discovery data and HTTP errors.

    4. Tests task planning and worker selection:
        - Uses structured LLM output to assign tasks.
        - Verifies valid and invalid worker selections.
        - Tests conditional task routing.

    5. Tests worker execution:
        - Simulates successful worker communication.
        - Verifies execution errors are captured.
        - Tests the current AgentDiscoveryInfo limitation.

    6. Tests the LangGraph workflow:
        - Executes the complete planning and routing process.
        - Verifies successful and failed task execution.
        - Verifies final responses.

    Note:
        External services are mocked, but the real LangGraph
        workflow is used. Remote A2A communication is not yet
        implemented in the Coordinator.
"""

import asyncio

from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock, MagicMock, call

import httpx
import pytest
from pydantic import ValidationError

import src.agent.base as base_module
import src.agent.coordinatorAgent as coordinator_module

from src.agent.coordinatorAgent import (
    CoordinatorAgent,
    CoordinatorState,
    TaskAssignment
)

from src.models.agent_discovery_info import AgentDiscoveryInfo


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_dependencies(monkeypatch):
    """
    Mock external CoordinatorAgent dependencies.

    1. Mock ChatOllama and structured LLM output.
    2. Mock asynchronous Registry GET and POST requests.
    3. Mock the heartbeat service.
    4. Prevent creation of a real LangChain agent.
    """

    # --------------------------------------------------------
    # Mock Ollama model and planner
    # --------------------------------------------------------

    mock_planner = Mock()

    mock_planner.ainvoke = AsyncMock(
        return_value=TaskAssignment(
            worker_name="Web_Search",
            task="Research the best index funds"
        )
    )

    mock_llm = Mock()

    mock_llm.with_structured_output.return_value = mock_planner

    mock_chat_ollama = Mock(
        return_value=mock_llm
    )

    # BaseAgent and CoordinatorAgent import ChatOllama separately.
    monkeypatch.setattr(
        base_module,
        "ChatOllama",
        mock_chat_ollama
    )

    monkeypatch.setattr(
        coordinator_module,
        "ChatOllama",
        mock_chat_ollama
    )

    # BaseAgent must not create a default LangChain agent
    # when coordinator=True.
    mock_create_agent = Mock()

    monkeypatch.setattr(
        base_module,
        "create_agent",
        mock_create_agent
    )

    # --------------------------------------------------------
    # Mock Registry discovery response
    # --------------------------------------------------------

    registry_agents = [
        {
            "agent_id": 2,
            "agent_name": "Web_Search",
            "status": "active",
            "max_concurrent_tasks": 1
        },
        {
            "agent_id": 4,
            "agent_name": "Document_Agent",
            "status": "active",
            "max_concurrent_tasks": 2
        },
        {
            "agent_id": 3,
            "agent_name": "coordinator",
            "status": "active",
            "max_concurrent_tasks": None
        }
    ]

    mock_get_response = Mock()

    mock_get_response.json.return_value = {
        "agents": registry_agents
    }

    mock_get_response.raise_for_status.return_value = None

    # --------------------------------------------------------
    # Mock Registry registration response
    # --------------------------------------------------------

    mock_post_response = Mock()

    mock_post_response.json.return_value = {
        "message": "Agent registered",
        "session_id": "coordinator-session-123",
        "status": "active"
    }

    mock_post_response.raise_for_status.return_value = None

    # --------------------------------------------------------
    # Mock asynchronous HTTP client
    # --------------------------------------------------------

    mock_http_client = Mock()

    mock_http_client.get = AsyncMock(
        return_value=mock_get_response
    )

    mock_http_client.post = AsyncMock(
        return_value=mock_post_response
    )

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

    monkeypatch.setattr(
        coordinator_module.httpx,
        "AsyncClient",
        mock_async_client
    )

    # --------------------------------------------------------
    # Mock heartbeat
    # --------------------------------------------------------

    mock_heartbeat = Mock()

    mock_heartbeat_class = Mock(
        return_value=mock_heartbeat
    )

    monkeypatch.setattr(
        base_module,
        "Heartbeat",
        mock_heartbeat_class
    )

    return {
        "llm": mock_llm,
        "chat_ollama": mock_chat_ollama,
        "planner": mock_planner,
        "create_agent": mock_create_agent,
        "registry_agents": registry_agents,
        "get_response": mock_get_response,
        "post_response": mock_post_response,
        "http_client": mock_http_client,
        "async_client": mock_async_client,
        "heartbeat": mock_heartbeat,
        "heartbeat_class": mock_heartbeat_class
    }


@pytest.fixture
def coordinator(mock_dependencies):
    """
    Construct a CoordinatorAgent without asynchronous initialization.
    """

    return CoordinatorAgent(
        model="llama3.1:8b",
        temperature=0,
        system_prompt="You are a coordinator.",
        agent_prompt="Delegate tasks to worker agents.",
        agent_id=3,
        agent_name="coordinator"
    )


def create_state(**overrides) -> CoordinatorState:
    """
    Create a default LangGraph state for testing individual nodes.
    """

    state: CoordinatorState = {
        "user_request": "",
        "worker_name": "",
        "task": "",
        "worker_result": "",
        "error": "",
        "final_answer": ""
    }

    state.update(overrides)

    return state


# ============================================================
# Coordinator Construction Tests
# ============================================================

def test_coordinator_construction(coordinator, mock_dependencies):
    """
    Verify CoordinatorAgent.__init__() configuration.

    Assertions:
        1. AgentInfo is initialized.
        2. Worker dictionary starts empty.
        3. ChatOllama is configured.
        4. Structured planner is created.
        5. LangGraph is compiled.
        6. No registration or heartbeat starts yet.
    """

    assert coordinator.agent_info.agent_id == 3
    assert coordinator.agent_info.agent_name == "coordinator"

    assert coordinator.agent_info.llm_model == "llama3.1:8b"
    assert coordinator.agent_info.llm_temperature == 0

    assert (
        coordinator.agent_info.system_prompt
        == "You are a coordinator."
    )

    assert coordinator.workers == {}

    # BaseAgent and CoordinatorAgent each construct ChatOllama.
    assert mock_dependencies["chat_ollama"].call_count == 2

    mock_dependencies[
        "chat_ollama"
    ].assert_has_calls([
        call(model="llama3.1:8b", temperature=0),
        call(model="llama3.1:8b", temperature=0)
    ])

    mock_dependencies[
        "llm"
    ].with_structured_output.assert_called_once_with(
        TaskAssignment,
        method="json_schema"
    )

    assert coordinator.planner is mock_dependencies["planner"]
    assert coordinator.graph is not None

    # Coordinator skips BaseAgent's default LangChain agent.
    mock_dependencies["create_agent"].assert_not_called()

    # No network activity during construction.
    mock_dependencies["async_client"].assert_not_called()

    mock_dependencies["heartbeat"].start.assert_not_called()


# ============================================================
# Agent Discovery Tests
# ============================================================

def test_get_current_agents(coordinator, mock_dependencies):
    """
    Verify Registry discovery and Pydantic conversion.

    Assertions:
        1. Registry GET request is sent.
        2. Agent dictionaries are converted to Pydantic objects.
        3. Worker names are dictionary keys.
        4. Coordinator is excluded from discovered workers.
    """

    asyncio.run(coordinator.get_current_agents())

    mock_dependencies[
        "http_client"
    ].get.assert_awaited_once_with(
        "http://127.0.0.1:8000/get_all_agents"
    )

    mock_dependencies[
        "get_response"
    ].raise_for_status.assert_called_once()

    assert len(coordinator.workers) == 2

    assert "Web_Search" in coordinator.workers
    assert "Document_Agent" in coordinator.workers
    assert "coordinator" not in coordinator.workers

    web_agent = coordinator.workers["Web_Search"]

    assert isinstance(web_agent, AgentDiscoveryInfo)

    assert web_agent.agent_id == 2
    assert web_agent.agent_name == "Web_Search"
    assert web_agent.status == "active"
    assert web_agent.max_concurrent_tasks == 1

    document_agent = coordinator.workers["Document_Agent"]

    assert document_agent.agent_id == 4
    assert document_agent.max_concurrent_tasks == 2


def test_get_current_agents_empty(coordinator, mock_dependencies):
    """
    Verify discovery when the Registry has no agents.
    """

    mock_dependencies[
        "get_response"
    ].json.return_value = {
        "agents": []
    }

    asyncio.run(coordinator.get_current_agents())

    assert coordinator.workers == {}


def test_get_current_agents_invalid_data(
    coordinator, mock_dependencies
):
    """
    Verify invalid Registry data raises a Pydantic error.
    """

    mock_dependencies[
        "get_response"
    ].json.return_value = {
        "agents": [
            {
                "agent_id": "invalid",
                "agent_name": "Web_Search",
                "status": "active",
                "max_concurrent_tasks": 1
            }
        ]
    }

    with pytest.raises(ValidationError):
        asyncio.run(coordinator.get_current_agents())


def test_get_current_agents_http_failure(
    coordinator, mock_dependencies
):
    """
    Verify Registry discovery errors are propagated.
    """

    mock_dependencies[
        "get_response"
    ].raise_for_status.side_effect = httpx.HTTPStatusError(
        "Registry unavailable",
        request=httpx.Request(
            "GET",
            "http://127.0.0.1:8000/get_all_agents"
        ),
        response=httpx.Response(500)
    )

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(coordinator.get_current_agents())

    assert coordinator.workers == {}


# ============================================================
# Async Initialization and Factory Tests
# ============================================================

def test_coordinator_initialization(
    coordinator, mock_dependencies
):
    """
    Verify asynchronous Coordinator initialization.

    Assertions:
        1. Worker discovery executes.
        2. Coordinator registers with the Registry.
        3. Session ID is stored.
        4. Heartbeat starts after registration.
    """

    asyncio.run(coordinator.initialize())

    # Verify worker discovery.
    assert len(coordinator.workers) == 2

    mock_dependencies[
        "http_client"
    ].get.assert_awaited_once()

    # Verify coordinator registration.
    mock_post = mock_dependencies["http_client"].post

    mock_post.assert_awaited_once()

    args, kwargs = mock_post.await_args

    assert args[0] == "http://127.0.0.1:8000/register"

    registration = kwargs["json"]

    assert registration["agent_id"] == 3
    assert registration["agent_name"] == "coordinator"

    # Verify discovery happens before registration.
    assert (
        mock_dependencies["http_client"].method_calls[0][0]
        == "get"
    )

    assert (
        mock_dependencies["http_client"].method_calls[1][0]
        == "post"
    )

    assert (
        coordinator.agent_info.session_id
        == "coordinator-session-123"
    )

    mock_dependencies[
        "heartbeat"
    ].start.assert_called_once()


def test_coordinator_async_factory(mock_dependencies):
    """
    Verify the inherited asynchronous factory method.

    Assertions:
        1. CoordinatorAgent.create() constructs the subclass.
        2. Overridden initialize() executes.
        3. Workers are discovered.
        4. Coordinator registers successfully.
        5. Heartbeat starts.
    """

    coordinator = asyncio.run(
        CoordinatorAgent.create(
            model="llama3.1:8b",
            temperature=0,
            system_prompt="You are a coordinator.",
            agent_prompt="Delegate tasks.",
            agent_id=3,
            agent_name="coordinator"
        )
    )

    assert isinstance(coordinator, CoordinatorAgent)

    assert len(coordinator.workers) == 2

    assert (
        coordinator.agent_info.session_id
        == "coordinator-session-123"
    )

    mock_dependencies[
        "http_client"
    ].get.assert_awaited_once()

    mock_dependencies[
        "http_client"
    ].post.assert_awaited_once()

    mock_dependencies[
        "heartbeat"
    ].start.assert_called_once()


# ============================================================
# Task Planning Tests
# ============================================================

def test_plan_task(coordinator, mock_dependencies):
    """
    Verify structured LLM task planning.

    Assertions:
        1. Available workers are provided to the planner.
        2. User request is included.
        3. Structured TaskAssignment is returned.
    """

    asyncio.run(coordinator.get_current_agents())

    state = create_state(
        user_request="Find the best index funds"
    )

    result = asyncio.run(coordinator.plan_task(state))

    assert result["worker_name"] == "Web_Search"
    assert result["task"] == "Research the best index funds"

    mock_dependencies[
        "planner"
    ].ainvoke.assert_awaited_once()

    messages = mock_dependencies["planner"].ainvoke.await_args.args[0]

    system_message = messages[0][1]
    user_message = messages[1][1]

    assert "Web_Search" in system_message
    assert "Document_Agent" in system_message
    assert "Available workers:" in system_message

    assert user_message == "Find the best index funds"


# ============================================================
# Worker Selection Tests
# ============================================================

def test_select_worker_valid(coordinator, mock_dependencies):
    """
    Verify selection of a registered worker.
    """

    asyncio.run(coordinator.get_current_agents())

    state = create_state(
        worker_name="Web_Search"
    )

    result = coordinator.select_worker(state)

    assert result == {}


def test_select_worker_invalid(coordinator, mock_dependencies):
    """
    Verify that an unknown worker produces an error.
    """

    asyncio.run(coordinator.get_current_agents())

    state = create_state(
        worker_name="UnknownAgent"
    )

    result = coordinator.select_worker(state)

    assert result == {
        "error": "No available worker: UnknownAgent"
    }


# ============================================================
# Task Routing Tests
# ============================================================

def test_route_task(coordinator):
    """
    Verify conditional LangGraph routing.

    Assertions:
        1. No error routes to execute.
        2. Existing error routes to finish.
    """

    valid_state = create_state(
        worker_name="Web_Search",
        error=""
    )

    invalid_state = create_state(
        worker_name="UnknownAgent",
        error="No available worker"
    )

    assert coordinator.route_task(valid_state) == "execute"
    assert coordinator.route_task(invalid_state) == "finish"


# ============================================================
# Worker Execution Tests
# ============================================================

def test_execute_worker_current_limitation(
    coordinator, mock_dependencies
):
    """
    Verify current behavior when executing a discovered worker.

    AgentDiscoveryInfo contains metadata and does not implement
    communicate(). The Coordinator should capture the error.
    """

    asyncio.run(coordinator.get_current_agents())

    state = create_state(
        worker_name="Web_Search",
        task="Research index funds"
    )

    result = asyncio.run(coordinator.execute_worker(state))

    assert "error" in result

    assert "Worker failed:" in result["error"]
    assert "AgentDiscoveryInfo" in result["error"]
    assert "communicate" in result["error"]


def test_execute_worker_mock_success(coordinator):
    """
    Verify the successful execution branch with a mock worker.

    A temporary worker stub is used because remote HTTP/A2A
    communication has not yet been implemented.
    """

    mock_worker = SimpleNamespace(
        communicate=AsyncMock(
            return_value={
                "messages": [
                    SimpleNamespace(
                        content="Research completed successfully."
                    )
                ]
            }
        )
    )

    # Test-only executable worker stub.
    coordinator.workers["Web_Search"] = mock_worker

    state = create_state(
        worker_name="Web_Search",
        task="Research index funds"
    )

    result = asyncio.run(coordinator.execute_worker(state))

    mock_worker.communicate.assert_awaited_once_with(
        "Research index funds"
    )

    assert result == {
        "worker_result": "Research completed successfully."
    }


def test_execute_worker_failure(coordinator):
    """
    Verify that worker execution exceptions are captured.
    """

    mock_worker = SimpleNamespace(
        communicate=AsyncMock(
            side_effect=RuntimeError("Worker unavailable")
        )
    )

    coordinator.workers["Web_Search"] = mock_worker

    state = create_state(
        worker_name="Web_Search",
        task="Research index funds"
    )

    result = asyncio.run(coordinator.execute_worker(state))

    assert result == {
        "error": "Worker failed: Worker unavailable"
    }


# ============================================================
# Final Response Tests
# ============================================================

def test_finish_success(coordinator):
    """
    Verify successful final response generation.
    """

    state = create_state(
        worker_result="Task completed.",
        error=""
    )

    result = coordinator.finish(state)

    assert result == {
        "final_answer": "Task completed."
    }


def test_finish_failure(coordinator):
    """
    Verify that execution errors become final responses.
    """

    state = create_state(
        error="Worker unavailable"
    )

    result = coordinator.finish(state)

    assert result == {
        "final_answer": "Worker unavailable"
    }


# ============================================================
# LangGraph Integration Tests
# ============================================================

def test_langgraph_workflow_success(
    coordinator, mock_dependencies
):
    """
    Test the complete LangGraph workflow with a mock worker.

    Workflow:
        START -> plan -> select -> execute -> finish -> END

    Verifies the complete graph without a real remote agent.
    """

    mock_worker = SimpleNamespace(
        communicate=AsyncMock(
            return_value={
                "messages": [
                    SimpleNamespace(
                        content="Here are the research findings."
                    )
                ]
            }
        )
    )

    coordinator.workers["Web_Search"] = mock_worker

    response = asyncio.run(
        coordinator.communicate(
            "Find the best index funds"
        )
    )

    assert response == "Here are the research findings."

    mock_dependencies[
        "planner"
    ].ainvoke.assert_awaited_once()

    mock_worker.communicate.assert_awaited_once_with(
        "Research the best index funds"
    )


def test_langgraph_workflow_invalid_worker(
    coordinator, mock_dependencies
):
    """
    Verify the graph handles an invalid worker selection.

    Workflow:
        START -> plan -> select -> finish -> END

    The execute node should not run.
    """

    mock_dependencies[
        "planner"
    ].ainvoke.return_value = TaskAssignment(
        worker_name="UnknownAgent",
        task="Research index funds"
    )

    response = asyncio.run(
        coordinator.communicate(
            "Find the best index funds"
        )
    )

    assert response == "No available worker: UnknownAgent"


def test_langgraph_workflow_discovery_limitation(
    coordinator, mock_dependencies
):
    """
    Verify the complete graph handles the current limitation
    where discovered workers are AgentDiscoveryInfo objects.
    """

    asyncio.run(coordinator.get_current_agents())

    response = asyncio.run(
        coordinator.communicate(
            "Find the best index funds"
        )
    )

    assert response.startswith("Worker failed:")
    assert "AgentDiscoveryInfo" in response
    assert "communicate" in response


# ============================================================
# Registration Failure Tests
# ============================================================

def test_coordinator_registration_failure(
    coordinator, mock_dependencies
):
    """
    Verify behavior when Coordinator registration fails.

    Assertions:
        1. Worker discovery completes.
        2. HTTP registration raises an exception.
        3. Session ID remains unset.
        4. Heartbeat does not start.
    """

    mock_dependencies[
        "post_response"
    ].raise_for_status.side_effect = httpx.HTTPStatusError(
        "Registration failed",
        request=httpx.Request(
            "POST",
            "http://127.0.0.1:8000/register"
        ),
        response=httpx.Response(500)
    )

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(coordinator.initialize())

    assert len(coordinator.workers) == 2
    assert coordinator.agent_info.session_id is None

    mock_dependencies[
        "http_client"
    ].post.assert_awaited_once()

    mock_dependencies[
        "heartbeat"
    ].start.assert_not_called()


def test_coordinator_discovery_failure_during_initialize(
    coordinator, mock_dependencies
):
    """
    Verify that failed discovery prevents registration.

    The current initialize() method discovers workers before
    calling BaseAgent.initialize().
    """

    mock_dependencies[
        "get_response"
    ].raise_for_status.side_effect = httpx.HTTPStatusError(
        "Discovery failed",
        request=httpx.Request(
            "GET",
            "http://127.0.0.1:8000/get_all_agents"
        ),
        response=httpx.Response(500)
    )

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(coordinator.initialize())

    mock_dependencies[
        "http_client"
    ].post.assert_not_awaited()

    mock_dependencies[
        "heartbeat"
    ].start.assert_not_called()
