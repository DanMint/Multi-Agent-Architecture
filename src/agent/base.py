import httpx
from langchain.agents import create_agent
from langchain_ollama import ChatOllama, OllamaEmbeddings
import httpx
import asyncio

from src.models.agent_info import AgentInfo
from src.models.agent_registration import AgentRegistration
from .heartbeat import Heartbeat

"""
BaseAgent: base agent class that difnies the base structure of agents. It is meant to be used as a parent class
    passed parameters:
        model: str =  the type of LLM model that the agent will use
        temperature: str = the temperature 
        system_prompt: str = the prompt that is given to the model
        agent_id: int = unique id for each agent
        registry_url: str = http url where the registry http server lives

    - register_agent(): sends an http message to the register server sending agents information
    - communicate(prompt: str): communication handeling with the agent
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
        response = self.agent.ainvoke(
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
    