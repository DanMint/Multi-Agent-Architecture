import httpx

from a2a.client import ClientConfig, create_client
from a2a.helpers import get_message_text, new_text_message
from a2a.types import Role, SendMessageRequest

"""
A2AClient: Handles asynchronous Agent-to-Agent (A2A) communication
between the CoordinatorAgent and remote worker agents.

    Uses the A2A SDK for agent discovery, message transmission, and
    response handling over HTTP.

    Features:
        - Agent Discovery: Retrieves worker information using the Agent Card URL.
        - HTTP Configuration: Sets connection, read, write, and pool timeouts.
        - Message Exchange: Sends requests and receives responses using A2A.
        - Resource Management: Ensures HTTP and A2A clients are properly closed.

    Methods:
        - send_message(agent_card_url, prompt): Discovers the remote agent,
          establishes an A2A connection, sends the provided prompt, and
          returns the worker's response as a string.

    Note:
        Currently supports direct Message responses only.
        Task-based responses and streaming are not yet supported.
"""

class A2AClient:
    async def send_message(self, agent_card_url: str, prompt: str) -> str:
        # exrecting base url
        base_url = agent_card_url.removesuffix(
            "/.well-known/agent-card.json"
        )

        # configuring http timeout. diffrent type of operatons require more/less time
        timeout = httpx.Timeout(
            connect=5.0,
            read=120.0,
            write=15.0,
            pool=5.0
        )

        async with httpx.AsyncClient(timeout=timeout) as http_client:
            print("Creating A2A client")

            client = await create_client(
                agent=base_url,
                client_config=ClientConfig(
                    streaming=False,
                    httpx_client=http_client
                )
            )

            try:
                print("A2A client created")

                message = new_text_message(
                    prompt,
                    role=Role.ROLE_USER
                )

                request = SendMessageRequest(message=message)

                print("Sending task to:", base_url)

                async for response in client.send_message(request):
                    print("A2A response received")

                    if response.HasField("message"):
                        return get_message_text(response.message)

                    if response.HasField("task"):
                        raise NotImplementedError(
                            "Task responses are not supported yet"
                        )

                raise RuntimeError("No responses received")

            finally:
                await client.close()
