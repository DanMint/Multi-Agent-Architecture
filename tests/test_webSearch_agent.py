import httpx
import pytest

from unittest.mock import Mock, AsyncMock, MagicMock, call

from src.agent.base import BaseAgent
import src.agent.base as base_module

import src.agent.CustomAgents.webSearchAgent.webSearchAgent as webSearch_module

"""
    Test Suite: ResearchAgent (WebSearchAgent)

    Tests the ResearchAgent class, including MCP tool discovery,
    asynchronous initialization, communication, and error handling.

    1. Sets up pytest fixtures:
        - Mocks the Ollama model and ChatOllama constructor.
        - Mocks the MCPAdapter and MCP server tools.
        - Mocks LangChain's create_agent() and ainvoke().
        - Mocks asynchronous HTTP registration with the Registry.
        - Mocks the UDP heartbeat service.

    2. Tests agent construction:
        - Verifies agent metadata and configuration.
        - Verifies MCP server URL.
        - Verifies agent capabilities and task capacity.
        - Ensures registration does not occur during __init__().

    3. Tests MCP tool discovery:
        - Connects to the mocked MCP server.
        - Retrieves web_search and fetch_webpage tools.
        - Verifies that tools are stored correctly.

    4. Tests asynchronous initialization:
        - Retrieves MCP tools.
        - Creates the LangChain agent using discovered tools.
        - Registers the agent with the Registry.
        - Stores the Registry session ID.
        - Starts the heartbeat service.

    5. Tests the asynchronous factory method:
        - Calls ResearchAgent.create().
        - Verifies that the inherited BaseAgent.create()
          executes the overridden initialize() method.
        - Ensures the initialized agent is returned.

    6. Tests agent communication:
        - Sends user messages through ainvoke().
        - Verifies the message format and response.

    7. Tests failure handling:
        - Handles MCP connection failures.
        - Handles LLM invocation failures.
        - Handles HTTP registration failures.
        - Verifies heartbeat does not start on failed setup.

    8. Tests empty MCP tool discovery:
        - Simulates an MCP server returning no tools.
        - Verifies the agent is created with an empty tool list.

    Note:
        All external dependencies are mocked.
        Tests do not require Ollama, an MCP server,
        or the Agent Registry to be running.
"""

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_dependencies(monkeypatch):
    """
    Mock dependencies used by BaseAgent and ResearchAgent.

    1. Mock Ollama and LangChain.
    2. Mock MCPAdapter and available MCP tools.
    3. Mock the asynchronous Registry HTTP response.
    4. Mock the heartbeat service.
    """
    # Mock Ollama model
    mock_model = Mock()

    mock_chat_ollama = Mock(
        return_value=mock_model
    )

    monkeypatch.setattr(
        base_module,
        "ChatOllama",
        mock_chat_ollama
    )

    # Mock LangChain agent
    mock_agent = Mock()

    mock_agent.ainvoke = AsyncMock(
        return_value={
            "messages": [
                {
                    "role": "user",
                    "content": "Find 5 stocks to buy"
                },
                {
                    "role": "assistant",
                    "content": "Here are 5 stocks."
                }
            ]
        }
    )

    mock_create_agent = Mock(
        return_value=mock_agent
    )

    # BaseAgent creates a default agent during __init__().
    monkeypatch.setattr(
        base_module,
        "create_agent",
        mock_create_agent
    )

    # ResearchAgent recreates the agent after MCP discovery.
    monkeypatch.setattr(
        webSearch_module,
        "create_agent",
        mock_create_agent
    )

    # Mock MCP tools
    mock_web_search_tool = Mock()
    mock_web_search_tool.name = "web_search"

    mock_fetch_webpage_tool = Mock()
    mock_fetch_webpage_tool.name = "fetch_webpage"

    mock_tools = [
        mock_web_search_tool,
        mock_fetch_webpage_tool
    ]

    # Mock MCP client
    mock_mcp_client = Mock()

    mock_mcp_client.list_tools = AsyncMock(
        return_value=mock_tools
    )

    # Mock MCPAdapter async context manager.
    mock_mcp_context = MagicMock()

    mock_mcp_context.__aenter__ = AsyncMock(
        return_value=mock_mcp_client
    )

    mock_mcp_context.__aexit__ = AsyncMock(
        return_value=False
    )

    mock_mcp_adapter = Mock(
        return_value=mock_mcp_context
    )

    monkeypatch.setattr(
        webSearch_module,
        "MCPAdapter",
        mock_mcp_adapter
    )

    # Mock Registry HTTP response
    mock_http_response = Mock()

    mock_http_response.json.return_value = {
        "message": "Agent registered",
        "session_id": "web-search-session-123",
        "status": "active"
    }

    mock_http_response.raise_for_status.return_value = None

    mock_http_client = Mock()

    mock_http_client.post = AsyncMock(
        return_value=mock_http_response
    )

    # Mock httpx.AsyncClient async context manager.
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
        base_module.httpx,
        "AsyncClient",
        mock_async_client
    )

    # Mock heartbeat service
    mock_heartbeat = Mock()

    mock_heartbeat_class = Mock(
        return_value=mock_heartbeat
    )

    monkeypatch.setattr(
        base_module,
        "Heartbeat",
        mock_heartbeat_class
    )

    # Return dependencies for assertions
    return {
        "model": mock_model,
        "chat_ollama": mock_chat_ollama,
        "agent": mock_agent,
        "create_agent": mock_create_agent,
        "tools": mock_tools,
        "mcp_client": mock_mcp_client,
        "mcp_adapter": mock_mcp_adapter,
        "http_client": mock_http_client,
        "http_response": mock_http_response,
        "async_client": mock_async_client,
        "heartbeat": mock_heartbeat,
        "heartbeat_class": mock_heartbeat_class
    }


@pytest.fixture
def agent(mock_dependencies):
    """
    Create a ResearchAgent with mocked dependencies.

    Note:
        __init__() constructs the agent.
        initialize() is called separately by the tests.
    """
    return webSearch_module.ResearchAgent(
        model="llama3.1:8b",
        temperature=0,
        agent_id=2,
        agent_name="Web_Search",
        system_prompt="You are a web search agent.",
        agnet_prompt="Search the web for information."
    )

# ============================================================
# Agent Construction Tests
# ============================================================

def test_agent_construction(agent, mock_dependencies):
    """
    Verify ResearchAgent.__init__() configuration.

    Assertions:
        1. Agent metadata is initialized correctly.
        2. MCP server URL is configured.
        3. Capabilities and task capacity are assigned.
        4. Default LangChain agent is constructed.
        5. Registry registration and heartbeat have not started.
    """
    # Verify agent metadata.
    assert agent.agent_info.agent_id == 2
    assert agent.agent_info.agent_name == "Web_Search"

    assert agent.agent_info.llm_model == "llama3.1:8b"
    assert agent.agent_info.llm_temperature == 0

    assert (agent.agent_info.system_prompt == "You are a web search agent.")
    assert (agent.agent_info.agent_prompt == "Search the web for information.")

    # Verify MCP configuration.
    assert agent.MCP_server == "http://127.0.0.1:8001/mcp"
    assert agent.tools == []

    # Verify capabilities and capacity.
    assert agent.agent_info.capabilities == ["web_search"]
    assert agent.agent_info.max_concurrent_tasks == 1

    # BaseAgent creates a default LangChain agent.
    mock_dependencies["create_agent"].assert_called_once_with(
        model=mock_dependencies["model"],
        tools=[],
        system_prompt="You are a web search agent."
    )

    # No MCP connection during construction.
    mock_dependencies["mcp_adapter"].assert_not_called()

    # No registration during construction.
    mock_dependencies["async_client"].assert_not_called()

    # No heartbeat transmission during construction.
    mock_dependencies["heartbeat"].start.assert_not_called()

# ============================================================
# MCP Tool Discovery Tests
# ============================================================

@pytest.mark.asyncio
async def test_get_mcp_tools(agent, mock_dependencies):
    """
    Verify MCP tool discovery.

    Assertions:
        1. MCPAdapter connects to the correct URL.
        2. list_tools() is awaited.
        3. Discovered tools are stored in agent.tools.
    """
    await agent.get_mcp_tools()

    mock_dependencies["mcp_adapter"].assert_called_once_with(
        "http://127.0.0.1:8001/mcp"
    )

    mock_dependencies["mcp_client"].list_tools.assert_awaited_once()

    assert len(agent.tools) == 2

    assert agent.tools[0].name == "web_search"
    assert agent.tools[1].name == "fetch_webpage"

    assert agent.tools == mock_dependencies["tools"]

# ============================================================
# Asynchronous Initialization Tests
# ============================================================

@pytest.mark.asyncio
async def test_agent_initialization(agent, mock_dependencies):
    """
    Verify complete ResearchAgent initialization.

    Assertions:
        1. MCP tools are discovered.
        2. LangChain agent is recreated with MCP tools.
        3. Agent registers with the Registry.
        4. Session ID is stored.
        5. Heartbeat starts successfully.
    """
    await agent.initialize()

    # Verify MCP discovery.
    mock_dependencies["mcp_client"].list_tools.assert_awaited_once()

    assert agent.tools == mock_dependencies["tools"]

    # create_agent() is called twice:
    # 1. BaseAgent.__init__() with no tools.
    # 2. ResearchAgent.initialize() with MCP tools.
    mock_create_agent = mock_dependencies["create_agent"]

    assert mock_create_agent.call_count == 2

    assert mock_create_agent.call_args_list == [
        call(
            model=mock_dependencies["model"],
            tools=[],
            system_prompt="You are a web search agent."
        ),
        call(
            model=mock_dependencies["model"],
            tools=mock_dependencies["tools"],
            system_prompt="You are a web search agent."
        )
    ]

    # Verify final LangChain agent.
    assert agent.agent is mock_dependencies["agent"]

    # Verify Registry registration.
    mock_post = mock_dependencies["http_client"].post

    mock_post.assert_awaited_once()

    args, kwargs = mock_post.await_args

    assert args[0] == "http://127.0.0.1:8000/register"

    registration_data = kwargs["json"]

    assert registration_data["agent_id"] == 2
    assert registration_data["agent_name"] == "Web_Search"
    assert registration_data["agent_card_url"] is None
    assert registration_data["max_concurrent_tasks"] == 1

    mock_dependencies["http_response"].raise_for_status.assert_called_once()

    # Verify Registry session ID.
    assert (agent.agent_info.session_id == "web-search-session-123")

    # Verify heartbeat starts.
    mock_dependencies["heartbeat"].start.assert_called_once_with()

# ============================================================
# Async Factory Method Tests
# ============================================================

@pytest.mark.asyncio
async def test_agent_async_factory(mock_dependencies):
    """
    Verify the inherited BaseAgent.create() factory method.

    Assertions:
        1. A ResearchAgent instance is created.
        2. ResearchAgent.initialize() executes.
        3. MCP tools are discovered.
        4. Agent registers with the Registry.
        5. Heartbeat starts.
        6. Initialized ResearchAgent is returned.
    """
    agent = await webSearch_module.ResearchAgent.create(
        model="llama3.1:8b",
        temperature=0,
        agent_id=2,
        agent_name="Web_Search",
        system_prompt="You are a web search agent.",
        agnet_prompt="Search the web for information."
    )

    # Verify returned instance.
    assert isinstance(agent, webSearch_module.ResearchAgent)
    assert isinstance(agent, BaseAgent)

    # Verify agent configuration.
    assert agent.agent_info.agent_id == 2
    assert agent.agent_info.agent_name == "Web_Search"

    # Verify overridden initialization executed.
    assert agent.tools == mock_dependencies["tools"]

    assert mock_dependencies["create_agent"].call_count == 2

    # Verify Registry registration.
    mock_dependencies["http_client"].post.assert_awaited_once()

    assert (agent.agent_info.session_id == "web-search-session-123")

    # Verify heartbeat.
    mock_dependencies["heartbeat"].start.assert_called_once()

# ============================================================
# Agent Communication Tests
# ============================================================

@pytest.mark.asyncio
async def test_agent_communication(agent, mock_dependencies):
    """
    Verify asynchronous agent communication.

    Assertions:
        1. Agent completes initialization.
        2. communicate() invokes the LangChain agent.
        3. Correct user message is supplied.
        4. Response is returned correctly.
    """
    await agent.initialize()

    prompt = "Find 5 stocks to buy"
    response = await agent.communicate(prompt)

    mock_dependencies["agent"].ainvoke.assert_awaited_once_with(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    )

    assert response["messages"][-1]["content"] == (
        "Here are 5 stocks."
    )

# ============================================================
# MCP Failure Tests
# ============================================================

@pytest.mark.asyncio
async def test_mcp_connection_failure(agent, mock_dependencies):
    """
    Verify behavior when the MCP server is unavailable.

    Assertions:
        1. MCP discovery raises ConnectionError.
        2. ResearchAgent does not recreate the agent with tools.
        3. Registry registration is not attempted.
        4. Heartbeat does not start.
    """
    mock_dependencies["mcp_client"].list_tools.side_effect = ConnectionError(
        "MCP server unavailable"
    )

    with pytest.raises(ConnectionError, match="MCP server unavailable"):
        await agent.initialize()

    # BaseAgent.__init__() already created one agent.
    # ResearchAgent.initialize() should not create another.
    assert mock_dependencies["create_agent"].call_count == 1

    # Registration must not occur.
    mock_dependencies["http_client"].post.assert_not_awaited()

    # Heartbeat must not start.
    mock_dependencies["heartbeat"].start.assert_not_called()

# ============================================================
# LLM Failure Tests
# ============================================================

@pytest.mark.asyncio
async def test_agent_communication_failure(agent, mock_dependencies):
    """
    Verify that LLM invocation errors are propagated.

    Assertions:
        1. Agent initializes successfully.
        2. ainvoke() raises RuntimeError.
        3. communicate() propagates the exception.
    """
    mock_dependencies["agent"].ainvoke.side_effect = RuntimeError(
        "LLM execution failed"
    )

    await agent.initialize()

    with pytest.raises(RuntimeError, match="LLM execution failed"):
        await agent.communicate("Search the web")

    mock_dependencies["agent"].ainvoke.assert_awaited_once()

# ============================================================
# Empty MCP Tools Tests
# ============================================================

@pytest.mark.asyncio
async def test_agent_with_no_tools(agent, mock_dependencies):
    """
    Verify behavior when the MCP server returns no tools.

    Assertions:
        1. MCP discovery returns an empty list.
        2. ResearchAgent creates a LangChain agent with no tools.
        3. Registry registration still succeeds.
        4. Heartbeat starts successfully.
    """
    mock_dependencies["mcp_client"].list_tools.return_value = []

    await agent.initialize()

    assert agent.tools == []

    # BaseAgent and ResearchAgent each call create_agent().
    assert mock_dependencies["create_agent"].call_count == 2

    # Verify final agent was created with no tools.
    final_call = mock_dependencies["create_agent"].call_args

    assert final_call.kwargs["tools"] == []
    assert final_call.kwargs["model"] is mock_dependencies["model"]

    # Verify registration and heartbeat.
    mock_dependencies["http_client"].post.assert_awaited_once()

    mock_dependencies["heartbeat"].start.assert_called_once()

# ============================================================
# Registry Registration Failure Tests
# ============================================================

@pytest.mark.asyncio
async def test_registration_failure(agent, mock_dependencies):
    """
    Verify behavior when Registry registration fails.

    Assertions:
        1. MCP tools are discovered successfully.
        2. LangChain agent is created with MCP tools.
        3. HTTP registration raises an error.
        4. Session ID remains unset.
        5. Heartbeat does not start.
    """
    # Simulate HTTP 500 from the Registry.
    mock_dependencies["http_response"].raise_for_status.side_effect = httpx.HTTPStatusError(
        "Registry unavailable",
        request=httpx.Request(
            "POST",
            "http://127.0.0.1:8000/register"
        ),
        response=httpx.Response(500)
    )

    with pytest.raises(httpx.HTTPStatusError):
        await agent.initialize()

    # MCP discovery should succeed before registration.
    mock_dependencies["mcp_client"].list_tools.assert_awaited_once()

    # LangChain agent should be created with MCP tools.
    assert mock_dependencies["create_agent"].call_count == 2

    # Verify registration was attempted.
    mock_dependencies["http_client"].post.assert_awaited_once()

    # Session ID should remain unset.
    assert agent.agent_info.session_id is None

    # Heartbeat must not start after failed registration.
    mock_dependencies["heartbeat"].start.assert_not_called()
