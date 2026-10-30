import httpx
from langchain.agents import create_agent
# from langchain.tools import tools
from langchain_ollama import ChatOllama, OllamaEmbeddings
from src.models import AgentInfo
from dataclasses import asdict
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
    def __init__(self, model: str, temperature: int, system_prompt: str, agent_id: int, registry_url: str = "http://127.0.0.1:8000/register") -> None:
        self.registry_url = registry_url
        self.agent_info = AgentInfo(
            agent_id=agent_id,
            capabilities=None,
            llm_model=model,
            llm_temperature=temperature,
            agent_system_prompt=system_prompt,
            endpoint=None,
            udp_host=None,
            udp_port=None,
            session_id=None,
            status=None,
            last_heartbeat=None,
            missed_hearbeats=None,
            max_concurrent_tasks=None
        )

        model = ChatOllama(
            model = self.agent_info.llm_model,
            temperature = self.agent_info.llm_temperature
        )

        self.agent = create_agent(
            model = model,
            system_prompt = self.agent_info.agent_system_prompt
        )

        self.register_agent()

        # call heartbeats
        self.heartbeat = Heartbeat(agent_id = self.agent_info.agent_id, registry_host = "127.0.0.1", registry_port = 9000, interval = 1)
        self.heartbeat.start()

    def __str__(self):
        return f"This is the base agent.\n Model: {self.agent_info.llm_model}\nTemperature: {self.agent_info.llm_temperature} \nSystem prompt: "

    def register_agent(self):
        payload = asdict(self.agent_info)

        print("Sending:", payload)
            
        r = httpx.post(self.registry_url, 
                    json=payload,
                    timeout=0.5)

        r.raise_for_status()

        return r.json()

    def udp_server(self):
        pass

    def communicate(self, prompt: str) -> str:
        response = self.agent.invoke(
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


def main():
    a1 = BaseAgent(model="llama3.1:8b", temperature=0, system_prompt="You are a helpful assistant", agent_id=3)

    print(a1)

    user_input = ""

    while user_input != "exit":
        user_input = input("Input a message: ")

        print(a1.communicate(user_input).get('messages')[1].content)

if __name__ == "__main__":
    main()

