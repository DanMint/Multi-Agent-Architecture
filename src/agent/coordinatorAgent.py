from .base import BaseAgent

"""
    This is the coordinator agent. The coordinator agent 
"""

class CoordinatorAgent(BaseAgent):
    def __init__(self, model: str, temperature: int, system_prompt : str = "You are a coordinator agent"):
        super().__init__(model=model, temperature=temperature, system_prompt=system_prompt)



def main():
    ca1 = CoordinatorAgent(model="llama3.1:8b", temperature=0)

    print(ca1)
    print(ca1.communicate("HELLO").get('messages')[1].content)

if __name__ == "__main__":
    main()