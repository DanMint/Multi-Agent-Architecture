from a2a.client import ClientConfig, create_client
from a2a.helpers import get_message_text, new_text_message
from a2a.types import Message, Role, SendMessageRequest

class A2AClient:
    async def send_message(self, agent_card_url: str, prompt: str) -> str:
        # get workers base url
        base_url = agent_card_url.removesuffix(
            "/.well-known/agent-card.json"
        )

        # discover the worker and create A2A client
        client = await create_client(
            agent=base_url,
            client_config=ClientConfig(streaming=False)
        )

        try:
            # construct request
            message = new_text_message(prompt, role=Role.ROLE_USER)
            request = SendMessageRequest(message=message)
            
            # send the request and receive a response
            async for response in client.send_message(request):
                if isinstance(response, Message):

                    if isinstance(response, Message):
                        return get_message_text(response)

                    raise NotImplementedError("Task reponses are not supported yet")

            raise RuntimeError("No responses received")
                        

        finally:
            await client.close()