from fastapi import FastAPI

from a2a.types import (
    AgentCard, AgentSkill, AgentCapabilities,
    AgentInterface, Role
)
from a2a.helpers import get_message_text, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes
)

class WorkerExecutor(AgentExecutor):
    def __init__(self, agent):
        self.agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        # get coordinators message
        message = get_message_text(context.message)

        # let worker process it
        result = await self.agent.communicate(message)

        # extract final answer
        if isinstance(result, dict):
            result = result["messages"][-1].content

        # send answer back through A2A
        response = new_text_message(
            str(result),
            role=Role.ROLE_AGENT
        )

        await event_queue.enqeue_event(response)

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise NotImplementedError("Cancellation not supported")

def create_a2a_app(agent, port: int) -> FastAPI:
    card = AgentCard(
        name=agent.agent_info.agent_name,
        description=agent.agent_info.description,
        version=agent.agent_info.version,
        supported_interfaces=[
            AgentInterface(
                url=f"http://127.0.0.1:{port}",
                protocol_bindings="JSONRPC",
                protocol_version="1.0"
            )
        ],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        defualt_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="task-task",
                name="Text Task",
                description="Process text requests",
                tags=["text", "worker"]
            )
        ]
    )

    handler = DefaultRequestHandler(
        agent_executor=WorkerExecutor(agent),
        task_store=InMemoryTaskStore(),
        agent_card=card
    )

    routes = [
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(handler, "/")
    ]

    return FastAPI(routes=routes)