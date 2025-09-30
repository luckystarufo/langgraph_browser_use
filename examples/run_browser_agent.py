import asyncio
from langgraph_browser_agent import LangGraphBrowserAgent
from browser_use import Agent, ChatOpenAI

async def test_complete_workflow():
    print('🧪 Testing complete LangGraph workflow with action execution...')
    
    # Create a simple agent with browser session
    agent = Agent(
        task="Go to amazon.com to check the price of high end CPUs for mining purposes",
        llm=ChatOpenAI(model='gpt-4o')
    )
    
    # Create LangGraph version BEFORE starting browser session
    langgraph_agent = LangGraphBrowserAgent(agent)
    
    # Test running the workflow (this will start browser session and execute initial actions)
    try:
        final_state = await langgraph_agent.run()
    except Exception as e:
        print(f'❌ Workflow failed: {e}')
        import traceback
        traceback.print_exc()

# Run the test
asyncio.run(test_complete_workflow())
