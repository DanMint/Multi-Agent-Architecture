FROM python:3.12-slim

WORKDIR /app

# installing registry dependencies
COPY docker/requirements/agent-registry.txt ./agent-registry.txt
RUN pip install --no-cache-dir -r ./agent-registry.txt

# copying agentRegistry
COPY src/infra/agentRegistry.py ./src/infra/agentRegistry.py

# copying agent_info.py
COPY src/models/agent_info.py ./src/models/agent_info.py

# documenting registry ports
EXPOSE 8000/tcp
EXPOSE 9000/udp

CMD ["python", "-m", "src.infra.agentRegistry"]