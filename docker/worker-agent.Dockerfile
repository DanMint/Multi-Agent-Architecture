FROM python:3.12-slim

WORKDIR /app 

# get the needed libraries
COPY docker/requirements/worker-agent.txt ./worker-agent.txt
RUN pip install --no-cache-dir -r ./worker-agent.txt

# get the base agent 
COPY src/agent/base.py ./src/agent/base.py

# get the worker agent 
COPY src/agent/workerAgent.py ./src/agent/workerAgent.py

# get the models agent 
COPY src/models/agent_info.py ./src/models/agent_info.py

# get the heartbeat class
COPY src/agent/heartbeat.py ./src/agent/heartbeat.py

CMD ["python", "-m", "src.agent.workerAgent"]