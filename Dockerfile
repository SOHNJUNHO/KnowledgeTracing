FROM python:3.11-slim

WORKDIR /app

COPY requirements/orchestrator.txt requirements/orchestrator.txt
RUN pip install --no-cache-dir -r requirements/orchestrator.txt

COPY . .
RUN pip install --no-cache-dir -e . --no-deps

CMD ["ai-tutor-api"]
