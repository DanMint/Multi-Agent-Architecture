from .base import BaseAgent

class WorkerAgent(BaseAgent):
    def __init__(self, model: str, temperature: str, prompt : str ="You are a worker agent. Listen to the coordinator"):
        super().__init__(model=model, temperature=temperature, prompt = prompt)
        