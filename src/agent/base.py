import httpx
from langchain.agents import create_agent
from langchain_ollama import ChatOllama, OllamaEmbeddings
import httpx
import asyncio

from src.models.agent_info import AgentInfo
from src.models.agent_registration import AgentRegistration
from .heartbeat import Heartbeat
from .A2A_server import A2AServer
from .A2A_client import A2AClient


"""
BaseAgent: Base class defining the common structure and functionality of agents
within the multi-agent architecture. Intended to be inherited by specialized agents.

    Provides shared functionality for LLM configuration, agent registration,
    heartbeat management, and Agent-to-Agent (A2A) server initialization.

    Parameters:
        - model (str): Name of the Ollama LLM model (e.g., llama3.1:8b).
        - temperature (float): Controls the LLM's response randomness.
        - system_prompt (str): System instructions defining agent behavior.
        - agent_prompt (str): Agent-specific instructions stored in AgentInfo.
        - agent_id (int): Unique identifier assigned to the agent.
        - agent_name (str): Name used to identify the agent.
        - description (str): Description of the agent's responsibilities.
        - registry_url (str): HTTP URL of the Agent Registry.
        - coordinator (bool): Identifies whether the agent is a Coordinator.
          Defaults to False.

    Attributes:
        - agent_info: AgentInfo model containing identification, LLM settings,
          Registry endpoints, A2A information, and runtime metadata.
        - llm: ChatOllama instance configured with the selected model.
        - agent: LangChain agent used for task execution.
          Created only when coordinator=False.
        - heartbeat: UDP heartbeat service used to notify the Registry
          that the agent is active.
        - coordinator: Determines whether the agent acts as a Coordinator.
        - registry_url: Base HTTP address of the Agent Registry.
        - aegnt_A2AServer: A2AServer instance exposing the worker's
          A2A communication interface. Created for non-coordinator agents.

    Methods:
        - __init__(): Initializes agent configuration, LLM, heartbeat,
          and the default LangChain agent when applicable.

        - register_agent(agent_card_url): Registers the agent with the
          Registry using HTTP POST and stores its assigned session ID.

        - communicate(prompt): Asynchronously invokes the LangChain agent
          to process a prompt and returns the execution result.

        - initialize(): Starts the A2A server for non-coordinator agents,
          configures Agent Card and endpoint URLs, registers the agent
          with the Registry, and starts heartbeat transmission.

        - create(): Asynchronous factory method that creates an agent,
          awaits initialization, and returns the initialized instance.

        - __str__(): Returns a string representation of agent configuration.

        - main(): Creates a sample BaseAgent and provides an interactive
          command-line interface for testing.

    Note:
        Worker agents expose A2A endpoints for remote task execution,
        while Coordinator agents manage their own orchestration logic.

        Agent discovery and registration are handled by the Registry,
        while A2A communication is handled by the A2AServer.

        Specialized agents can override initialize() and communicate()
        to provide additional capabilities while reusing BaseAgent functionality.
"""

class BaseAgent:
    def __init__(self, model: str, temperature: int, system_prompt: str, agent_prompt:str ,agent_id: int, agent_name: str, description: str, registry_url: str = "http://127.0.0.1:8000/", coordinator: bool = False) -> None:
        self.coordinator = coordinator
        self.registry_url = registry_url
        self.agent_info = AgentInfo(
            # identification and information
            agent_id=agent_id,
            agent_name=agent_name,
            description=description,
            version="1.0.0",
            capabilities=None,
            llm_model=model,
            llm_temperature=temperature,
            system_prompt=system_prompt,
            agent_prompt=agent_prompt,

            # registry information
            registry_url=registry_url,
            registry_register_agent_url=registry_url + "register",
            registry_get_agent_by_id_url=registry_url + "agent/id",
            registry_get_agent_amount_url=registry_url + "agent_amount",
            registry_get_all_agents=registry_url + "get_all_agents",

            # A2A agent card url
            agent_card_url=None,
            agent_A2A_endpoint=None,

            # runtime metdata
            session_id=None,
            status=None,
            last_heartbeat=None,
            missed_hearbeats=None,

            # optimal capacity information
            max_concurrent_tasks=None
        )

        self.llm = ChatOllama(
            model = self.agent_info.llm_model,
            temperature = self.agent_info.llm_temperature
        )

        if not self.coordinator:
            self.agent = create_agent(
                model = self.llm,
                tools = [],
                system_prompt = self.agent_info.system_prompt
            )

        # setting up heartbeat config
        self.heartbeat = Heartbeat(
            agent_id = self.agent_info.agent_id, 
            registry_host = "127.0.0.1", 
            registry_port = 9000, 
            interval = 1
        )

    def __str__(self) -> str:
        return f"This is the base agent.\n Model: {self.agent_info.llm_model}\nTemperature: {self.agent_info.llm_temperature} \nSystem prompt: "

    async def register_agent(self, agent_card_url: str) -> None:
        registration = AgentRegistration(
            agent_id=self.agent_info.agent_id,
            agent_name=self.agent_info.agent_name,
            agent_card_url=self.agent_info.agent_card_url,
            max_concurrent_tasks=self.agent_info.max_concurrent_tasks
        )

        # sending registration events
        async with httpx.AsyncClient(timeout=0.5) as client:
            response = await client.post(
                self.agent_info.registry_register_agent_url,
                json=registration.model_dump(mode='json')
            )

            response.raise_for_status()

        # storing session that is assingend by the registry
        result = response.json()
        self.agent_info.session_id = result["session_id"]

        print(f"Agent {self.agent_info.agent_id} registered successfully")

    async def communicate(self, prompt: str) -> str:
        response = await self.agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        )

        return response

    async def initialize(self):
        """
        Perform asynchronous initialization.
        Subclasses override this method.
        """
        port = 8008
        # start A2A server if NOT coordinator
        if not self.coordinator:
            self.aegnt_A2AServer = A2AServer(agent=self, port=8008)
            await self.aegnt_A2AServer.start()
            self.agent_info.agent_card_url = f"http://127.0.0.1:{port}/.well-known/agent-card.json"
            self.agent_info.agent_A2A_endpoint = f"http://127.0.0.1:{port}"
            
        # register agent with registry
        await self.register_agent(self.agent_info.agent_card_url)

        # start heartbeat service with registry
        self.heartbeat.start()

    @classmethod
    async def create(cls, *args, **kwargs):
        agent = cls(*args, **kwargs)
        await agent.initialize()
        return agent


async def main():
    a1 = await BaseAgent.create(model="llama3.1:8b", temperature=0, system_prompt="You are a helpful assistant", agent_prompt="...", agent_id=2, agent_name = "base", description="You are the base agent")

    print(a1)

    user_input = ""

    while user_input != "exit":
        user_input = input("Input a message: ")

        response = await a1.communicate(user_input)
        print(response.get('messages')[1].content)

        if user_input == "exit":
            a1.heartbeat.stop()

if __name__ == "__main__":
    asyncio.run(main())
    