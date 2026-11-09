import pytest
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock, MagicMock

from src.agent.base import BaseAgent
import src.agent.CustomAgents.webSearchAgent.webSearchAgent as webSearch_module

@pytest.fixture
def mock_dependencies(monkeypatch):
    # 1. Mock the BaseAgent initialization
    mock_model = Mock()

    def mock_base_init(self, model, temperature, system_prompt, agent_id):
        self.llm = mock_model
        self.agent_info = SimpleNamespace(agent_system_prompt=system_prompt)
        self.agent = None

    monkeypatch.setattr(BaseAgent, "__init__", mock_base_init)

    # 2. Mock the MCP tools
    mock_web_search_tool = Mock()
    mock_web_search_tool.name = "web_search"

    mock_fetch_webpage_tool = Mock()
    mock_fetch_webpage_tool.name = "fetch_webpage"

    mock_tools = [
        mock_web_search_tool,
        mock_fetch_webpage_tool
    ]

    # 3. Mock the MCP client
    mock_mcp_client = Mock()
    mock_mcp_client.list_tools = AsyncMock(return_value=mock_tools)

    # MCPAdapter uses an async context manager
    mock_mcp_context = MagicMock()
    mock_mcp_context.__aenter__ = AsyncMock(return_value=mock_mcp_client)
    mock_mcp_context.__aexit__ = AsyncMock(return_value=False)

    mock_mcp_adapter = Mock(return_value=mock_mcp_context)

    monkeypatch.setattr(
        webSearch_module,
        "MCPAdapter",
        mock_mcp_adapter
    )

    # 4. Mock the LangChain agent
    mock_agent = Mock()

    mock_agent.ainvoke = AsyncMock(
        return_value={
            "messages": [
                {"role": "user", "content": "Find 5 stocks to buy"},
                {"role": "assistant", "content": "Here are 5 stocks."}
            ]
        }
    )

    mock_create_agent = Mock(
        return_value=mock_agent
    )

    monkeypatch.setattr(webSearch_module, "create_agent", mock_create_agent)

    # Return all dependencies for assertions
    return {
        "model": mock_model,
        "tools": mock_tools,
        "mcp_client": mock_mcp_client,
        "mcp_adapter": mock_mcp_adapter,
        "create_agent": mock_create_agent,
        "agent": mock_agent
    }


# TEST 1: Agent initialization
@pytest.mark.asyncio
async def test_agent_initialization(mock_dependencies):

    agent = webSearch_module.ResearchAgent(
        model="llama3.1:8b",
        temperature=0,
        agent_id=1,
        system_prompt="You are a web search agent."
    )

    await agent.initialize()

    assert agent.MCP_server == ("http://127.0.0.1:8001/mcp")

    assert agent.tools == mock_dependencies["tools"]

    mock_dependencies["create_agent"].assert_called_once_with(
        model=mock_dependencies["model"],
        tools=mock_dependencies["tools"],
        system_prompt="You are a web search agent."
    )

    assert agent.agent is mock_dependencies["agent"]


# TEST 2: MCP tool discovery
@pytest.mark.asyncio
async def test_get_mcp_tools(mock_dependencies):

    agent = webSearch_module.ResearchAgent(
        model="llama3.1:8b",
        temperature=0,
        agent_id=1
    )

    await agent.get_mcp_tools()

    mock_dependencies["mcp_adapter"].assert_called_once_with("http://127.0.0.1:8001/mcp")

    mock_dependencies["mcp_client"].list_tools.assert_awaited_once()

    assert len(agent.tools) == 2

    assert agent.tools[0].name == "web_search"
    assert agent.tools[1].name == "fetch_webpage"


# TEST 3: Agent communication
@pytest.mark.asyncio
async def test_agent_communication(mock_dependencies):

    agent = webSearch_module.ResearchAgent(
        model="llama3.1:8b",
        temperature=0,
        agent_id=1
    )

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


# TEST 4: MCP discovery failure
@pytest.mark.asyncio
async def test_mcp_connection_failure(mock_dependencies):

    mock_dependencies["mcp_client"].list_tools.side_effect = (
        ConnectionError("MCP server unavailable")
    )

    agent = webSearch_module.ResearchAgent(
        model="llama3.1:8b",
        temperature=0,
        agent_id=1
    )

    with pytest.raises(
        ConnectionError,
        match="MCP server unavailable"
    ):
        await agent.initialize()

    mock_dependencies["create_agent"].assert_not_called()


# TEST 5: LLM invocation failure
@pytest.mark.asyncio
async def test_agent_communication_failure(mock_dependencies):

    mock_dependencies["agent"].ainvoke.side_effect = (
        RuntimeError("LLM execution failed")
    )

    agent = webSearch_module.ResearchAgent(
        model="llama3.1:8b",
        temperature=0,
        agent_id=1
    )

    await agent.initialize()

    with pytest.raises(
        RuntimeError,
        match="LLM execution failed"
    ):
        await agent.communicate("Search the web")


# TEST 6: No MCP tools available
@pytest.mark.asyncio
async def test_agent_with_no_tools(mock_dependencies):

    mock_dependencies["mcp_client"].list_tools.return_value = []

    agent = webSearch_module.ResearchAgent(
        model="llama3.1:8b",
        temperature=0,
        agent_id=1
    )

    await agent.initialize()

    assert agent.tools == []

    assert (mock_dependencies["create_agent"].call_args.kwargs["tools"]== [])
