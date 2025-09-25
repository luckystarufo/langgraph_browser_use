"""Tests for state management."""
import pytest
from unittest.mock import Mock

from langgraph_browser_agent.state import BrowserAgentState


class TestBrowserAgentState:
    """Test BrowserAgentState TypedDict."""
    
    def test_state_schema(self):
        """Test that BrowserAgentState has all required fields."""
        state: BrowserAgentState = {
            'task': 'test task',
            'browser_state_summary': None,
            'last_model_output': None,
            'last_result': None
        }
        
        assert state['task'] == 'test task'
        assert state['browser_state_summary'] is None
        assert state['last_model_output'] is None
        assert state['last_result'] is None
    
    def test_state_with_optional_fields(self):
        """Test that BrowserAgentState can have optional fields set."""
        from browser_use.agent.views import AgentOutput, ActionResult
        
        # Mock AgentOutput and ActionResult
        mock_output = Mock(spec=AgentOutput)
        mock_result = [Mock(spec=ActionResult)]
        
        state: BrowserAgentState = {
            'task': 'test task',
            'browser_state_summary': None,
            'last_model_output': mock_output,
            'last_result': mock_result
        }
        
        assert state['task'] == 'test task'
        assert state['last_model_output'] == mock_output
        assert state['last_result'] == mock_result
