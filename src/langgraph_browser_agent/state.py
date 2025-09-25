from typing import TypedDict, Optional, List

from browser_use.agent.views import ActionResult, AgentOutput
from browser_use.browser.views import BrowserStateSummary


class BrowserAgentState(TypedDict):
    """State schema for LangGraph browser agent - minimal fields for LangGraph Studio"""
    # Task information (required for LangGraph Studio input)
    task: str
    
    # Browser state summary (passed between nodes)
    browser_state_summary: Optional[BrowserStateSummary]
    last_model_output: Optional[AgentOutput]
    last_result: Optional[List[ActionResult]]
