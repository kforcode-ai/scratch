# MiniAgent Framework

MiniAgent keeps agent development small and sharp: a single Agent loop, pluggable tools, optional Redis sessions, and observability baked in so you can ship assistants without wrestling a full platform.

## Core Ideas
- **Agent + Thread** – `Agent.run(...)` handles each user turn; `Thread` keeps history, plan state, and telemetry.
- **Tools** – Register functions with lightweight decorators and let the model invoke them.
- **Multi‑LLM** – Switch between OpenAI, Gemini, or Anthropic via `AgentConfig` or environment variables.
- **Sessions** – Drop-in Redis support keeps conversations alive across processes.
- **Observability** – Every event carries `session_id`, `request_id`, and `event_id` so logs and dashboards stay in sync.

## Quick Start
```bash
pip install -r requirements.txt          # brings OpenAI, Gemini, Anthropic clients
cp .env.example .env && edit your keys   # OPENAI_API_KEY / GOOGLE_API_KEY / ANTHROPIC_API_KEY
python demos/demo_miniagent.py           # baseline agent loop
```