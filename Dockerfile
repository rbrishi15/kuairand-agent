FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# CPU-only by design (see CLAUDE.md §3) — GPU is an optional speedup,
# never a requirement for reproducing a teammate's result.
CMD ["python", "agent/orchestrator.py", "--config", "configs/kuairand_pure.yaml"]
