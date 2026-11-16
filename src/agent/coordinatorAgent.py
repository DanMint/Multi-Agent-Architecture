
from typing import Any, TypedDict
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing import override
import asyncio
import httpx

from src.agent.base import BaseAgent
from src.agent.A2A_client import A2AClient

from src.models.agent_discovery_info import AgentDiscoveryInfo



"""
CoordinatorAgent: Central orchestration agent responsible for discovering,
selecting, and delegating tasks to specialized worker agents using A2A communication.

    Inherits from BaseAgent for agent registration, heartbeat management,
    LLM configuration, and asynchronous initialization.

    Uses LangGraph to manage task planning, worker selection, remote execution,
    and response handling.

    Attributes:
        - workers: Dictionary of available worker metadata retrieved from
          the Agent Registry using AgentDiscoveryInfo models.
        - a2a_client: A2AClient instance responsible for communicating with
          remote worker agents.
        - llm: Ollama model used for task planning and worker selection.
        - planner: Structured LLM planner that generates TaskAssignment objects.
        - graph: LangGraph workflow managing task planning and execution.

    Data Models:
        - CoordinatorState: Stores the user request, selected worker,
          assigned task, execution results, errors, and final response.
        - TaskAssignment: Defines the selected worker and assigned task
          using Pydantic for structured LLM output.

    Methods:
        - __init__(): Initializes the Coordinator, A2A client, LLM planner,
          and LangGraph workflow.

        - get_current_agents(): Asynchronously retrieves available workers
          from the Agent Registry and populates the workers dictionary.

        - plan_task(state): Uses the LLM to select an available worker
          and generate task instructions.

        - select_worker(state): Verifies that the selected worker exists
          in the discovered workers dictionary.

        - route_task(state): Routes execution to the worker or finish node
          based on the current error state.

        - execute_worker(state): Sends the assigned task to the selected
          remote worker through A2A and captures its response or errors.

        - finish(state): Produces the final response or execution error.

        - communicate(prompt): Executes the LangGraph workflow for a
          user request and returns the final response.

        - initialize(): Discovers available workers, registers the
          Coordinator with the Registry, and starts heartbeat transmission.

        - main(): Creates the Coordinator using the inherited asynchronous
          factory method and executes a sample task.

    Note:
        Worker discovery is handled by the Agent Registry, while remote
        communication is performed through the A2A protocol using each
        worker's Agent Card URL.

        Workers are represented by AgentDiscoveryInfo metadata objects,
        rather than local agent instances.

        Currently supports single-worker task delegation with direct
        text responses. Multi-worker orchestration and task-based A2A
        responses are not yet implemented.
"""

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
    def __init__(self, model: str, system_prompt: str, agent_prompt: str, agent_id: int, agent_name: str, description: str, temperature: float = 0.0, workers: dict[str, AgentDiscoveryInfo] | None = None):
        print(f"Agent id: {agent_id}")
        super().__init__(model=model, temperature=temperature, system_prompt=system_prompt, agent_prompt=agent_prompt, agent_id=agent_id, coordinator=True, agent_name=agent_name, description=description)
        # Available worker agents
        self.workers =  workers or {}

        # Set up A2A client
        self.a2a_client = A2AClient()

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

    async def get_current_agents(self) -> None:
        print("getting them boy!!!")
        async with httpx.AsyncClient(timeout=0.5) as client:
            response = await client.get(self.agent_info.registry_get_all_agents) 
            response.raise_for_status()

        data = response.json()

        for agent in data["agents"]:
            if agent.get("agent_name") != "coordinator":
                print(agent.get("agent_name"))
                self.workers[agent.get("agent_name")] = AgentDiscoveryInfo.model_validate(agent)
            
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
            result = await self.a2a_client.send_message(
                agent_card_url=str(worker.agent_card_url),
                prompt=state["task"]
            )

            return {"worker_result": result}

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

    async def initialize(self) -> None:
        # get all worker agents
        await self.get_current_agents()

        # register coordinator agent to registry and set up heartbeat messages
        await super().initialize()

async def main():
    c1 = await CoordinatorAgent.create(model="llama3.1:8b", temperature=0, system_prompt='AA', agent_prompt="", agent_id=1, agent_name="coordinator", description="Coordinator agent responsible for selecting appropriate agents for tasks, and orchestrating task execution through A2A communication")
    ans = await c1.communicate(prompt="Find me the best stocks to buy")
    print(ans)

if __name__ == "__main__":
    asyncio.run(main())     