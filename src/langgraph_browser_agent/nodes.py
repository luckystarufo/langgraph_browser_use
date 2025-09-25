import time
import inspect

from .state import BrowserAgentState
from browser_use.agent.views import AgentStepInfo, ActionResult


def check_paused_node(state: BrowserAgentState, agent_instance) -> BrowserAgentState:
    return state


def check_consecutive_failures_node(state: BrowserAgentState, agent_instance) -> BrowserAgentState:
    return state


def check_stopped_node(state: BrowserAgentState, agent_instance) -> BrowserAgentState:
    return state


async def on_step_start_node(state: BrowserAgentState, agent_instance) -> BrowserAgentState:
    if hasattr(agent_instance, 'on_step_start') and agent_instance.on_step_start is not None:
        await agent_instance.on_step_start(agent_instance.original_agent)
    return state


async def on_step_end_node(state: BrowserAgentState, agent_instance) -> BrowserAgentState:
    if hasattr(agent_instance, 'on_step_end') and agent_instance.on_step_end is not None:
        await agent_instance.on_step_end(agent_instance.original_agent)
    return state


async def paused_state_actions_node(state: BrowserAgentState, agent_instance) -> BrowserAgentState:
    agent = agent_instance.original_agent
    agent.logger.debug(f'⏸️ Step {agent_instance.current_step}: Agent paused, waiting to resume...')
    await agent._external_pause_event.wait()
    agent_instance.signal_handler.reset()
    return state


def consecutive_failure_actions_node(state: BrowserAgentState, agent_instance) -> BrowserAgentState:
    agent = agent_instance.original_agent
    agent.logger.error(f'❌ Stopping due to {agent.settings.max_failures} consecutive failures')
    agent_instance.ended_due_to_break = True
    return state


def stopped_state_actions_node(state: BrowserAgentState, agent_instance) -> BrowserAgentState:
    agent = agent_instance.original_agent
    agent.logger.info('🛑 Agent stopped')
    agent_instance.ended_due_to_break = True
    return state


async def history_is_done_actions_node(state: BrowserAgentState, agent_instance) -> BrowserAgentState:
    agent = agent_instance.original_agent
    agent.logger.debug(f'🎯 Task completed after {agent_instance.current_step + 1} steps!')
    await agent.log_completion()
    if agent.register_done_callback:
        if inspect.iscoroutinefunction(agent.register_done_callback):
            await agent.register_done_callback(agent.history)
        else:
            agent.register_done_callback(agent.history)
    agent_instance.ended_due_to_break = True
    return state


def check_step_timeout(state: BrowserAgentState, agent) -> bool:
    elapsed_time = time.time() - agent.original_agent.step_start_time
    if elapsed_time > agent.original_agent.settings.step_timeout:
        error_msg = f'Step {agent.current_step + 1} timed out after {agent.original_agent.settings.step_timeout} seconds'
        agent.original_agent.logger.error(f'⏰ {error_msg}')
        agent.original_agent.state.consecutive_failures += 1
        agent.original_agent.state.last_result = [ActionResult(error=error_msg)]
        agent.step_timed_out = True
        return True
    return False


async def prepare_context_node(state: BrowserAgentState, agent) -> BrowserAgentState:
    agent.original_agent.logger.debug(f'🚶 Starting step {agent.current_step + 1}/{agent.max_steps}...')
    agent.original_agent.step_start_time = time.time()
    print(f"🔄 Step {agent.current_step}: Preparing context...")
    try:
        step_info = AgentStepInfo(
            step_number=agent.current_step,
            max_steps=agent.max_steps
        )
        browser_state_summary = await agent.original_agent._prepare_context(step_info)
        state['browser_state_summary'] = browser_state_summary
        agent.step_info = step_info
        agent.last_error = None
        print(f"✅ Context prepared for step {agent.current_step}")
    except Exception as e:
        agent.last_error = str(e)
        print(f"❌ Error in prepare_context for step {agent.current_step}: {e}")
    if check_step_timeout(state, agent):
        print(f"⏰ Step {agent.current_step} timed out in prepare_context")
    return state


async def get_next_action_node(state: BrowserAgentState, agent) -> BrowserAgentState:
    print(f"🤖 Step {agent.current_step}: Getting next action from LLM...")
    try:
        await agent.original_agent._get_next_action(state['browser_state_summary'])
        agent.last_error = None
        state['last_model_output'] = agent.original_agent.state.last_model_output
        print(f"✅ LLM response received for step {agent.current_step}")
    except Exception as e:
        agent.last_error = str(e)
        print(f"❌ Error in get_next_action for step {agent.current_step}: {e}")
    if check_step_timeout(state, agent):
        print(f"⏰ Step {agent.current_step} timed out in get_next_action")
    return state


async def execute_actions_node(state: BrowserAgentState, agent) -> BrowserAgentState:
    print(f"⚡ Step {agent.current_step}: Executing actions...")
    try:
        await agent.original_agent._execute_actions()
        agent.last_error = None
        state['last_result'] = agent.original_agent.state.last_result
        print(f"✅ Actions executed for step {agent.current_step}")
    except Exception as e:
        agent.last_error = str(e)
        print(f"❌ Error in execute_actions for step {agent.current_step}: {e}")
    if check_step_timeout(state, agent):
        print(f"⏰ Step {agent.current_step} timed out in execute_actions")
    return state


async def evaluate_result_node(state: BrowserAgentState, agent) -> BrowserAgentState:
    print(f"📊 Step {agent.current_step}: Evaluating result...")
    try:
        await agent.original_agent._post_process()
        agent.last_error = None
        print(f"✅ Result evaluated for step {agent.current_step}")
    except Exception as e:
        agent.last_error = str(e)
        print(f"❌ Error in evaluate_result for step {agent.current_step}: {e}")
    if check_step_timeout(state, agent):
        print(f"⏰ Step {agent.current_step} timed out in evaluate_result")
    return state


async def finalize_step_node(state: BrowserAgentState, agent) -> BrowserAgentState:
    print(f"🔚 Step {agent.current_step}: Finalizing step...")
    await agent.original_agent._finalize(state['browser_state_summary'])
    agent.current_step += 1
    print(f"✅ Step {agent.current_step - 1} finalized, next step will be {agent.current_step}")
    if check_step_timeout(state, agent):
        print(f"⏰ Step {agent.current_step - 1} timed out in finalize_step")
    return state


async def handle_error_node(state: BrowserAgentState, agent) -> BrowserAgentState:
    print(f"❌ Step {agent.current_step}: Handling error...")
    try:
        error = Exception(agent.last_error) if agent.last_error else Exception("Unknown error")
        await agent.original_agent._handle_step_error(error)
        print(f"✅ Error handled for step {agent.current_step}")
    except Exception as e:
        print(f"❌ Error in handle_error_node for step {agent.current_step}: {e}")
    return state