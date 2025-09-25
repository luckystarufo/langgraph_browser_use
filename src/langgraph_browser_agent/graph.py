from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import BrowserAgentState
from .nodes import (
    check_paused_node,
    check_consecutive_failures_node,
    check_stopped_node,
    paused_state_actions_node,
    consecutive_failure_actions_node,
    stopped_state_actions_node,
    history_is_done_actions_node,
    on_step_start_node,
    on_step_end_node,
    prepare_context_node,
    get_next_action_node,
    execute_actions_node,
    evaluate_result_node,
    finalize_step_node,
    handle_error_node,
)
from .routes import (
    route_paused,
    route_consecutive_failures,
    route_stopped,
    route_completion,
    route_on_timeout_or_error,
)


def create_browser_agent_graph(agent_instance):
    """Create LangGraph workflow that mirrors the original Agent's for loop logic"""
    workflow = StateGraph(BrowserAgentState)

    # Create wrapper functions for all nodes
    def check_paused_node_with_agent(state: BrowserAgentState) -> BrowserAgentState:
        return check_paused_node(state, agent_instance)

    def check_consecutive_failures_node_with_agent(state: BrowserAgentState) -> BrowserAgentState:
        return check_consecutive_failures_node(state, agent_instance)

    def check_stopped_node_with_agent(state: BrowserAgentState) -> BrowserAgentState:
        return check_stopped_node(state, agent_instance)

    async def paused_state_actions_node_with_agent(state: BrowserAgentState) -> BrowserAgentState:
        return await paused_state_actions_node(state, agent_instance)

    def consecutive_failure_actions_node_with_agent(state: BrowserAgentState) -> BrowserAgentState:
        return consecutive_failure_actions_node(state, agent_instance)

    def stopped_state_actions_node_with_agent(state: BrowserAgentState) -> BrowserAgentState:
        return stopped_state_actions_node(state, agent_instance)

    async def history_is_done_actions_node_with_agent(state: BrowserAgentState) -> BrowserAgentState:
        return await history_is_done_actions_node(state, agent_instance)

    async def on_step_start_node_with_agent(state: BrowserAgentState) -> BrowserAgentState:
        return await on_step_start_node(state, agent_instance)

    # Create wrapper functions for routes
    def route_paused_with_agent(state: BrowserAgentState) -> str:
        return route_paused(state, agent_instance)

    def route_stopped_with_agent(state: BrowserAgentState) -> str:
        return route_stopped(state, agent_instance)

    async def on_step_end_node_with_agent(state: BrowserAgentState) -> BrowserAgentState:
        return await on_step_end_node(state, agent_instance)

    async def prepare_context_node_with_agent(state: BrowserAgentState) -> BrowserAgentState:
        return await prepare_context_node(state, agent_instance)

    async def get_next_action_node_with_agent(state: BrowserAgentState) -> BrowserAgentState:
        return await get_next_action_node(state, agent_instance)

    async def execute_actions_node_with_agent(state: BrowserAgentState) -> BrowserAgentState:
        return await execute_actions_node(state, agent_instance)

    async def evaluate_result_node_with_agent(state: BrowserAgentState) -> BrowserAgentState:
        return await evaluate_result_node(state, agent_instance)

    async def finalize_step_node_with_agent(state: BrowserAgentState) -> BrowserAgentState:
        return await finalize_step_node(state, agent_instance)

    async def handle_error_node_with_agent(state: BrowserAgentState) -> BrowserAgentState:
        return await handle_error_node(state, agent_instance)

    # Add all nodes to the workflow
    workflow.add_node("check_paused", check_paused_node_with_agent)
    workflow.add_node("check_consecutive_failures", check_consecutive_failures_node_with_agent)
    workflow.add_node("check_stopped", check_stopped_node_with_agent)
    workflow.add_node("paused_state_actions", paused_state_actions_node_with_agent)
    workflow.add_node("consecutive_failure_actions", consecutive_failure_actions_node_with_agent)
    workflow.add_node("stopped_state_actions", stopped_state_actions_node_with_agent)
    workflow.add_node("history_is_done_actions", history_is_done_actions_node_with_agent)
    workflow.add_node("on_step_start", on_step_start_node_with_agent)
    workflow.add_node("on_step_end", on_step_end_node_with_agent)
    workflow.add_node("prepare_context", prepare_context_node_with_agent)
    workflow.add_node("get_next_action", get_next_action_node_with_agent)
    workflow.add_node("execute_actions", execute_actions_node_with_agent)
    workflow.add_node("evaluate_result", evaluate_result_node_with_agent)
    workflow.add_node("finalize_step", finalize_step_node_with_agent)
    workflow.add_node("handle_error", handle_error_node_with_agent)

    # Set entry point to start with the first check
    workflow.set_entry_point("check_paused")

    # Create wrapper functions for routing with agent access
    def route_consecutive_failures_with_agent(state: BrowserAgentState) -> str:
        return route_consecutive_failures(state, agent_instance)

    def route_completion_with_agent(state: BrowserAgentState) -> str:
        return route_completion(state, agent_instance)

    def route_on_timeout_or_error_with_agent(state: BrowserAgentState) -> str:
        return route_on_timeout_or_error(state, agent_instance)

    # pre-processing edges
    workflow.add_conditional_edges(
        "check_paused",
        route_paused_with_agent,
        {
            "paused": "paused_state_actions",
            "not_paused": "check_consecutive_failures"
        }
    )

    workflow.add_edge("paused_state_actions", "check_consecutive_failures")

    workflow.add_conditional_edges(
        "check_consecutive_failures",
        route_consecutive_failures_with_agent,
        {
            "too_many_failures": "consecutive_failure_actions",
            "ok": "check_stopped"
        }
    )

    workflow.add_edge("consecutive_failure_actions", END)

    workflow.add_conditional_edges(
        "check_stopped",
        route_stopped_with_agent,
        {
            "stopped": "stopped_state_actions",
            "not_stopped": "on_step_start"
        }
    )

    workflow.add_edge("stopped_state_actions", END)

    workflow.add_edge("on_step_start", "prepare_context")

    workflow.add_conditional_edges(
        "prepare_context",
        route_on_timeout_or_error_with_agent,
        {
            "timeout": "on_step_end",
            "error": "handle_error",
            "continue": "get_next_action"
        }
    )

    workflow.add_conditional_edges(
        "get_next_action",
        route_on_timeout_or_error_with_agent,
        {
            "timeout": "on_step_end",
            "error": "handle_error",
            "continue": "execute_actions"
        }
    )

    workflow.add_conditional_edges(
        "execute_actions",
        route_on_timeout_or_error_with_agent,
        {
            "timeout": "on_step_end",
            "error": "handle_error",
            "continue": "evaluate_result"
        }
    )

    workflow.add_conditional_edges(
        "evaluate_result",
        route_on_timeout_or_error_with_agent,
        {
            "timeout": "on_step_end",
            "error": "handle_error",
            "continue": "finalize_step"
        }
    )

    workflow.add_edge("finalize_step", "on_step_end")
    workflow.add_edge("handle_error", "finalize_step")

    workflow.add_conditional_edges(
        "on_step_end",
        route_completion_with_agent,
        {
            "done": "history_is_done_actions",
            "continue": "check_paused"
        }
    )

    workflow.add_edge("history_is_done_actions", END)

    # Compile without checkpointer to avoid serialization issues
    # For Studio visualization, use create_standalone_graph() instead
    return workflow.compile()


def create_standalone_graph():
    """
    Create a standalone graph for LangGraph Studio visualization.
    This creates a mock agent instance for visualization purposes.
    """
    from unittest.mock import Mock
    
    # Create a mock agent instance for visualization
    mock_agent = Mock()
    mock_agent.original_agent = Mock()
    mock_agent.original_agent.settings = Mock()
    mock_agent.original_agent.settings.max_failures = 3
    mock_agent.original_agent.settings.final_response_after_failure = False
    
    return create_browser_agent_graph(mock_agent)


