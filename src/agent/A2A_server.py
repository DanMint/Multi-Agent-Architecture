from fastapi import FastAPI
import asyncio
import uvicorn

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


"""
A2AServer: Handles incoming Agent-to-Agent (A2A) requests and enables
remote communication with agents within the multi-agent architecture.

    Inherits from AgentExecutor to process A2A requests and execute
    tasks using the associated agent's communication interface.

    Attributes:
        - agent: Local agent responsible for executing incoming tasks.
        - port: HTTP port on which the A2A server operates.
        - card: Agent Card containing identity, capabilities, and endpoints.
        - handler: A2A request handler responsible for processing requests.
        - app: FastAPI application exposing Agent Card and JSON-RPC routes.
        - server: Uvicorn server instance.
        - server_task: Asynchronous task managing the server lifecycle.

    Methods:
        - __init__(agent, port): Initializes the Agent Card, A2A request
          handler, FastAPI routes, and server configuration.

        - execute(context, event_queue): Receives an A2A message, delegates
          execution to the local agent, and returns the response through
          the event queue.

        - cancel(context, event_queue): Handles task cancellation.
          Currently not implemented.

        - start(): Starts the Uvicorn server asynchronously and waits
          until it is ready to accept incoming requests.

        - stop(): Shuts down the Uvicorn server and releases resources
          associated with the A2A request handler.

    Note:
        Supports JSON-RPC communication with text-based messages.
        Streaming and task cancellation are not currently supported.
        Uses an in-memory task store without persistent storage.
"""

class A2AServer(AgentExecutor):
    def __init__(self, agent, port):
        self.agent = agent
        self.port = port

        # create agent card
        self.card = AgentCard(
                name=agent.agent_info.agent_name,
                description=agent.agent_info.description,
                version=agent.agent_info.version,
                supported_interfaces=[
                    AgentInterface(
                        url=f"http://127.0.0.1:{port}",
                        protocol_binding="JSONRPC",
                        protocol_version="1.0"
                    )
                ],
                capabilities=AgentCapabilities(streaming=False),
                default_input_modes=["text/plain"],
                default_output_modes=["text/plain"],
                skills=[
                    AgentSkill(
                        id="task-task",
                        name="Text Task",
                        description="Process text requests",
                        tags=["text", "worker"]
                    )
                ]
            )

        # create A2A request handler
        self.handler = DefaultRequestHandler(
            agent_executor=self,
            task_store=InMemoryTaskStore(),
            agent_card=self.card
        )

        # create routes and FastAPI app
        routes = [
            *create_agent_card_routes(self.card),
            *create_jsonrpc_routes(self.handler, "/")
        ]

        self.app = FastAPI(routes=routes)

        # server lifecycle
        self.server = None
        self.server_task = None

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

        await event_queue.enqueue_event(response)

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise NotImplementedError("Cancellation not supported")

    async def start(self):
        # configure Uvicorn
        config = uvicorn.Config(
            app=self.app,
            host="127.0.0.1",
            port=self.port
        )

        self.server = uvicorn.Server(config)

        # start server asynchronously
        self.server_task = asyncio.create_task(
            self.server.serve()
        )

        # wait until the server starts listening
        for _ in range(100):
            if self.server.started:
                return

            if self.server_task.done():
                await self.server_task
                raise RuntimeError("A2A server failed to start")

            await asyncio.sleep(0.05)

        # startup timed out
        await self.stop()
        raise RuntimeError("A2A server startup timed out")

    async def stop(self):
        if self.server is not None:
            self.server.server_should_exit = True

        if self.server_task is not None:
            await self.server_task

        await self.handler.aclose()