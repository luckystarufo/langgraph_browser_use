# Examples

This folder contains example usage scripts.

## Environment setup

1. Copy `.env.template` at the project root to `.env`.
2. Edit `.env` and set at least your `OPENAI_API_KEY`.

```bash
cp .env.template .env
$EDITOR .env
```

3. Activate your virtual environment and run an example:

```bash
source .venv/bin/activate
python examples/run_browser_agent.py
```

## Available Examples

### `run_browser_agent.py`
Complete working example that demonstrates:
- Creating a browser-use Agent with OpenAI
- Wrapping it with LangGraphBrowserAgent
- Running a complete workflow that searches Amazon for high-end CPUs


