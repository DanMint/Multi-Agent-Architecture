from langchain.agents import create_agent
# from langchain.tools import tools
from langchain_ollama import ChatOllama, OllamaEmbeddings

class Agent:
    def __init__(self, model: str, temperature: int, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        self.model_name = model
        self.temperature = temperature

        self.model = ChatOllama(
            model = self.model_name,
            temperature = self.temperature
        )

        self.agent = create_agent(
            model = self.model,
            system_prompt = self.system_prompt
        )

    def __str__(self):
        return f"This is the base agent.\n Model: {self.model_name}\nTemperature: {self.temperature} \nSystem prompt: {self.system_prompt}"

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
    a1 = Agent(model="llama3.1:8b", temperature=0, system_prompt="You are a helpful assistant")

    print(a1)

    user_input = ""

    while user_input != "exit":
        user_input = input("Input a message: ")

        print(a1.communicate(user_input).get('messages')[1].content)

if __name__ == "__main__":
    main()