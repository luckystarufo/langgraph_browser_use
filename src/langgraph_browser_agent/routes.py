from .state import BrowserAgentState


def route_paused(state: BrowserAgentState, agent_instance) -> str:
    agent = agent_instance.original_agent
    if agent.state.paused:
        return "paused"
    else:
        return "not_paused"


def route_consecutive_failures(state: BrowserAgentState, agent_instance) -> str:
    agent = agent_instance.original_agent
    if agent.state.consecutive_failures >= agent.settings.max_failures + int(agent.settings.final_response_after_failure):
        return "too_many_failures"
    else:
        return "ok"


def route_stopped(state: BrowserAgentState, agent_instance) -> str:
    agent = agent_instance.original_agent
    if agent.state.stopped:
        return "stopped"
    else:
        return "not_stopped"


def route_completion(state: BrowserAgentState, agent_instance) -> str:
    agent = agent_instance.original_agent
    if agent.history.is_done():
        return "done"
    else:
        return "continue"


def route_on_timeout_or_error(state: BrowserAgentState, agent_instance) -> str:
    if agent_instance.step_timed_out:
        return "timeout"
    elif agent_instance.last_error is not None:
        return "error"
    else:
        return "continue"


