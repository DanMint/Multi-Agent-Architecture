import httpx
from langchain.agents import create_agent
from langchain_ollama import ChatOllama, OllamaEmbeddings
import httpx
import asyncio

from src.models.agent_info import AgentInfo
from src.models.agent_registration import AgentRegistration
from .heartbeat import Heartbeat

"""
BaseAgent: Base class defining the common structure and functionality of agents
in the multi-agent architecture. Intended to be inherited by specialized agents.

Parameters:
    model (str): Name of the LLM model used by the agent (e.g., llama3.1:8b).
    temperature (int): Temperature setting controlling the LLM's randomness.
    system_prompt (str): System instructions defining the agent's behavior.
    agent_prompt (str): Agent-specific prompt stored in AgentInfo.
    agent_id (int): Unique identifier assigned to the agent.
    registry_url (str): HTTP URL of the agent registry server.
    coordinator (bool): If True, skips creation of the default LangChain agent.
                        Defaults to False.

Attributes:
    agent_info (AgentInfo): Stores agent identification, configuration, registry
                            endpoints, communication details, and runtime metadata.
    llm (ChatOllama): Local Ollama language model instance.
    agent: LangChain agent created using the configured LLM and system prompt.
           Only initialized when coordinator=False.
    heartbeat (Heartbeat): UDP heartbeat service used to notify the registry
                           that the agent is active.

Methods:
    __init__(): Initializes agent information, LLM, LangChain agent, and heartbeat.

    initialize(): Performs asynchronous initialization by registering the agent
                  with the registry and starting the heartbeat service.
                  Can be overridden by subclasses.

    create(): Asynchronous factory method that creates an agent instance,
              awaits initialization, and returns the initialized agent.

    register_agent(agent_card_url: str): Asynchronously registers the agent with
              the registry using HTTP POST and stores the assigned session ID.

    communicate(prompt: str): Sends a user prompt to the LangChain agent
              for asynchronous processing and returns the agent response.

    __str__(): Returns a string representation of the agent configuration.
"""

class BaseAgent:
    def __init__(self, model: str, temperature: int, system_prompt: str, agent_prompt:str ,agent_id: int, registry_url: str = "http://127.0.0.1:8000/", coordinator: bool = False) -> None:
        self.registry_url = registry_url
        self.agent_info = AgentInfo(
            # identification and information
            agent_id=agent_id,
            capabilities=None,
            llm_model=model,
            llm_temperature=temperature,
            system_prompt=system_prompt,
            agent_prompt=agent_prompt,
            agent_card_url=None,

            # registry information
            registry_url=registry_url,
            registry_register_agent_url=registry_url + "register",
            registry_get_agent_by_id_url=registry_url + "agent/id",
            registry_get_agent_amount_url=registry_url + "agent_amount",
            registry_get_all_agents=registry_url + "get_all_agents",

            # agent to agent communication
            endpoint=None,

            # udp heartbeat listener
            udp_host=None,
            udp_port=None,

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

        if not coordinator:
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

    async def initialize(self):
        """
        Perform asynchronous initialization.
        Subclasses can override this method.
        """

        # register agent with registry
        await self.register_agent(self.agent_info.agent_card_url)

        # start heartbeat service with registry
        self.heartbeat.start()

    @classmethod
    async def create(cls, *args, **kwargs):
        agent = cls(*args, **kwargs)
        await agent.initialize()
        return agent

    async def register_agent(self, agent_card_url: str) -> None:
        registration = AgentRegistration(
            agent_id=self.agent_info.agent_id,
            agent_card_url=agent_card_url,
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


async def main():
    a1 = await BaseAgent.create(model="llama3.1:8b", temperature=0, system_prompt="You are a helpful assistant", agent_prompt="...", agent_id=2)

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
    