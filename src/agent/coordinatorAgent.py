
from typing import Any, TypedDict
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing import override
import asyncio

import httpx
from src.agent.base import BaseAgent
from src.models.agent_info import AgentInfo

class CoordinatorState(TypedDict):
    user_request: str
    worker_name: str
    task: str
    worker_result: str
    error: str
    final_answer: str

class TaskAssignment(BaseModel):
    worker_name: str = Field(
        description="Name of the worker to execute the task, or 'none'"
    )

    task: str = Field(
        description="Instructions to send to the selected worker"
    )

class CoordinatorAgent(BaseAgent):
    def __init__(self, model: str, system_prompt: str, agent_prompt: str, agent_id: int, agent_name: str, temperature: float = 0.0, workers: dict[str, Any] | None = None):
        print(f"Agent id: {agent_id}")
        super().__init__(model=model, temperature=temperature, system_prompt=system_prompt, agent_prompt=agent_prompt, agent_id=agent_id, coordinator=True, agent_name=agent_name)
        # Available worker agents
        self.workers = workers or {}

        # Coordinator's LLM
        self.llm = ChatOllama(
            model=self.agent_info.llm_model,
            temperature=self.agent_info.llm_temperature
        )

        # Force structured planning output
        self.planner = self.llm.with_structured_output(
            TaskAssignment,
            method="json_schema"
        )

        # Create LangGraph
        workflow = StateGraph(CoordinatorState)

        # Register nodes
        workflow.add_node("plan", self.plan_task)
        workflow.add_node("select", self.select_worker)
        workflow.add_node("execute", self.execute_worker)
        workflow.add_node("finish", self.finish)

        # Connect nodes
        workflow.add_edge(START, "plan")
        workflow.add_edge("plan", "select")

        # Conditional routing
        workflow.add_conditional_edges(
            "select",
            self.route_task,
            {
                "execute": "execute",
                "finish": "finish"
            }
        )

        workflow.add_edge("execute", "finish")
        workflow.add_edge("finish", END)

        # Compile the workflow
        self.graph = workflow.compile()

    # @classmethod
    # async def create(cls, model, temperature, system_prompt, agent_prompt, agent_id):
    #     coordinator = cls(model=model, temperature=temperature, system_prompt=system_prompt, agent_prompt=agent_prompt, agent_id=agent_id)
    #     await coordinator.get_current_agents()
    #     return coordinator
    @override
    async def initialize(self) -> None:
        print("---getting all avilabe agents---\n")
        await self.get_current_agents()
        print("---setting up---n")
        await super().initialize()

    async def get_current_agents(self):
        r = httpx.get(self.agent_info.registry_get_all_agents)
        print('---------------------------------\n\n')
        print(r.text)
    
    async def plan_task(self, state: CoordinatorState):
        print("planning")
        available_agents = ", ".join(self.workers.keys())

        coordinator_system_prompt = (
            f"{self.agent_info.system_prompt}\n\n"
            f"{self.agent_info.agent_prompt}\n\n"
            f"Available workers: {available_agents}\n"
            "Select exactly one available worker. "
            "If none is suitable, return 'none'. "
            "Do not execute the task yourself."
        )

        # ask the LLM to create a task assignment
        response = await self.planner.ainvoke([
            ("system", coordinator_system_prompt),
            ("human", state["user_request"])
        ])

        return {
            "worker_name": response.worker_name,
            "task": response.task
        }

    def select_worker(self, state: CoordinatorState):
        print("selecting")
        worker_name = state["worker_name"]

        if worker_name not in self.workers:
            return {
                "error": f"No available worker: {worker_name}"
            }

        return {}

    def route_task(self, state: CoordinatorState):
        print("routing")
        if state["error"]:
            return "finish"

        return "execute"

    async def execute_worker(self, state: CoordinatorState):
        print("executing")
        worker = self.workers[state["worker_name"]]

        try:
            response = await worker.communicate(
                state["task"]
            )

            result = response["messages"][-1].content

            return {"worker_result": str(result)}

        except Exception as exc:
            return {
                "error": f"Worker failed: {exc}"
            }

    def finish(self, state: CoordinatorState):
        print("finishing")
        if state["error"]:
            return {"final_answer": state["error"]}

        return {
            "final_answer": state["worker_result"]
        }

    async def communicate(self, prompt: str):
        result = await self.graph.ainvoke({
            "user_request": prompt,
            "worker_name": "",
            "task": "",
            "worker_result": "",
            "error": "",
            "final_answer": ""
        })

        return result["final_answer"]


async def main():
    # c1 = CoordinatorAgent(model="llama3.1:8b", temperature=0, workers={}, system_prompt='AA', agent_prompt="", agent_id=1)
    # response = await c1.communicate(prompt="wubba lubba")
    # print(response)
    c1 = await CoordinatorAgent.create(model="llama3.1:8b", temperature=0, system_prompt='AA', agent_prompt="", agent_id=3, agent_name="coordinator")
    ans = await c1.communicate(prompt="Find me the best stocks to buy")
    print(ans)

if __name__ == "__main__":
    asyncio.run(main())     