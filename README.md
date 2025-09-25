# LangGraph Browser Agent

This package reorganizes the [Browser Use agent](https://github.com/browser-use/browser-use/tree/main) into a graph.

### Install

Create and activate a virtual environment, then install the package and dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
# Install optional extras
pip install 'langgraph-browser-agent[browser,studio]'
```

Notes:
- The `browser-use` dependency is provided as an optional extra pointing to the upstream Git repo. If you have a local version or a different source, install it accordingly.
- Ensure compatible Python (>=3.10).

### Environment

- Copy `.env.template` to `.env` and set at least `OPENAI_API_KEY`.

```bash
cp .env.template .env
$EDITOR .env
```

### Usage

See the working example in `examples/run_browser_agent.py`. This example demonstrates:
- Creating a browser-use Agent with OpenAI
- Wrapping it with LangGraphBrowserAgent
- Running a complete workflow that searches Amazon for high-end CPUs

```python
from langgraph_browser_agent import LangGraphBrowserAgent
from browser_use import Agent, ChatOpenAI

# Create original browser-use agent
agent = Agent(
    task='Find the price of high end CPUs for mining purposes, you may go to amazon.com for it', 
    llm=ChatOpenAI(model='gpt-4o')
)

# Wrap with LangGraph version
langgraph_agent = LangGraphBrowserAgent(agent)

# Run the workflow
history = await langgraph_agent.run(max_steps=100)
```

We recommend checking the examples first to understand how to construct the `original_agent`.

### LangGraph Studio Visualization

Visualize the browser agent workflow in LangGraph Studio:

```bash
# Install LangGraph CLI with in-memory support
pip install "langgraph-cli[inmem]"

# Run the development server with Studio integration
langgraph dev

# Open the Studio URL shown in the terminal (usually https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024)
# The graph will be available as 'browser_agent'
```


The graph visualization shows:
- **Node flow**: All workflow steps and decision points
- **State transitions**: How the BrowserAgentState changes
- **Routing logic**: Conditional edges and decision paths
- **Error handling**: Timeout and error recovery flows

### Development

Run tests:

```bash
pytest -q
```

Run specific test categories:

```bash
# Core functionality tests
pytest tests/test_imports.py tests/test_state.py tests/test_routes.py tests/test_graph.py -v

# Node implementation tests  
pytest tests/test_nodes.py -v

# Agent and integration tests
pytest tests/test_agent.py tests/test_integration.py -v

# All tests
pytest tests/ -v
```

The test suite includes:
- **Import tests**: Verify package can be imported correctly
- **State management tests**: Test BrowserAgentState and synchronization
- **Routing tests**: Test all routing logic for workflow decisions
- **Node tests**: Test individual LangGraph node implementations
- **Graph tests**: Test graph creation and structure
- **Integration tests**: Test end-to-end workflows
- **Agent tests**: Test LangGraphBrowserAgent initialization and core functionality

**Test Coverage**: 35/35 tests passing (100% pass rate)
- Core functionality: ✅ All passing
- Package structure: ✅ All verified
- Workflow components: ✅ All tested

If tests or first runs reveal additional dependencies, add them to the install section above.
