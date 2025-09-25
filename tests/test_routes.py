"""Tests for routing functions."""
import pytest
from unittest.mock import Mock

from langgraph_browser_agent.state import BrowserAgentState
from langgraph_browser_agent.routes import (
    route_paused,
    route_consecutive_failures,
    route_stopped,
    route_completion,
    route_on_timeout_or_error,
)


class TestRoutePaused:
    """Test route_paused function."""
    
    def test_route_paused_when_paused(self):
        """Test routing when agent is paused."""
        # Mock agent instance
        mock_agent_instance = Mock()
        mock_agent_instance.original_agent.state.paused = True
        
        state: BrowserAgentState = {
            'task': 'test',
            'browser_state_summary': None,
            'last_model_output': None,
            'last_result': None
        }
        
        result = route_paused(state, mock_agent_instance)
        assert result == "paused"
    
    def test_route_paused_when_not_paused(self):
        """Test routing when agent is not paused."""
        # Mock agent instance
        mock_agent_instance = Mock()
        mock_agent_instance.original_agent.state.paused = False
        
        state: BrowserAgentState = {
            'task': 'test',
            'browser_state_summary': None,
            'last_model_output': None,
            'last_result': None
        }
        
        result = route_paused(state, mock_agent_instance)
        assert result == "not_paused"


class TestRouteConsecutiveFailures:
    """Test route_consecutive_failures function."""
    
    def test_route_too_many_failures(self):
        """Test routing when too many consecutive failures."""
        # Mock agent instance
        mock_agent_instance = Mock()
        mock_agent_instance.original_agent.state.consecutive_failures = 4  # >= max_failures + final_response_after_failure
        mock_agent_instance.original_agent.settings.max_failures = 3
        mock_agent_instance.original_agent.settings.final_response_after_failure = False
        
        state: BrowserAgentState = {
            'task': 'test',
            'browser_state_summary': None,
            'last_model_output': None,
            'last_result': None
        }
        
        result = route_consecutive_failures(state, mock_agent_instance)
        assert result == "too_many_failures"
    
    def test_route_ok_failures(self):
        """Test routing when failures are within limit."""
        # Mock agent instance
        mock_agent_instance = Mock()
        mock_agent_instance.original_agent.state.consecutive_failures = 2  # < max_failures + final_response_after_failure
        mock_agent_instance.original_agent.settings.max_failures = 3
        mock_agent_instance.original_agent.settings.final_response_after_failure = False
        
        state: BrowserAgentState = {
            'task': 'test',
            'browser_state_summary': None,
            'last_model_output': None,
            'last_result': None
        }
        
        result = route_consecutive_failures(state, mock_agent_instance)
        assert result == "ok"


class TestRouteStopped:
    """Test route_stopped function."""
    
    def test_route_stopped_when_stopped(self):
        """Test routing when agent is stopped."""
        # Mock agent instance
        mock_agent_instance = Mock()
        mock_agent_instance.original_agent.state.stopped = True
        
        state: BrowserAgentState = {
            'task': 'test',
            'browser_state_summary': None,
            'last_model_output': None,
            'last_result': None
        }
        
        result = route_stopped(state, mock_agent_instance)
        assert result == "stopped"
    
    def test_route_not_stopped(self):
        """Test routing when agent is not stopped."""
        # Mock agent instance
        mock_agent_instance = Mock()
        mock_agent_instance.original_agent.state.stopped = False
        
        state: BrowserAgentState = {
            'task': 'test',
            'browser_state_summary': None,
            'last_model_output': None,
            'last_result': None
        }
        
        result = route_stopped(state, mock_agent_instance)
        assert result == "not_stopped"


class TestRouteCompletion:
    """Test route_completion function."""
    
    def test_route_done_when_done(self):
        """Test routing when task is done."""
        # Mock agent instance
        mock_agent_instance = Mock()
        mock_history = Mock()
        mock_history.is_done.return_value = True
        mock_agent_instance.original_agent.history = mock_history
        
        state: BrowserAgentState = {
            'task': 'test',
            'browser_state_summary': None,
            'last_model_output': None,
            'last_result': None
        }
        
        result = route_completion(state, mock_agent_instance)
        assert result == "done"
        mock_history.is_done.assert_called_once()
    
    def test_route_continue_when_not_done(self):
        """Test routing when task is not done."""
        # Mock agent instance
        mock_agent_instance = Mock()
        mock_history = Mock()
        mock_history.is_done.return_value = False
        mock_agent_instance.original_agent.history = mock_history
        
        state: BrowserAgentState = {
            'task': 'test',
            'browser_state_summary': None,
            'last_model_output': None,
            'last_result': None
        }
        
        result = route_completion(state, mock_agent_instance)
        assert result == "continue"
        mock_history.is_done.assert_called_once()


class TestRouteOnTimeoutOrError:
    """Test route_on_timeout_or_error function."""
    
    def test_route_timeout(self):
        """Test routing when step timed out."""
        # Mock agent instance
        mock_agent_instance = Mock()
        mock_agent_instance.step_timed_out = True
        mock_agent_instance.last_error = 'some error'
        
        state: BrowserAgentState = {
            'task': 'test',
            'browser_state_summary': None,
            'last_model_output': None,
            'last_result': None
        }
        
        result = route_on_timeout_or_error(state, mock_agent_instance)
        assert result == "timeout"
    
    def test_route_error(self):
        """Test routing when there's an error."""
        # Mock agent instance
        mock_agent_instance = Mock()
        mock_agent_instance.step_timed_out = False
        mock_agent_instance.last_error = 'some error'
        
        state: BrowserAgentState = {
            'task': 'test',
            'browser_state_summary': None,
            'last_model_output': None,
            'last_result': None
        }
        
        result = route_on_timeout_or_error(state, mock_agent_instance)
        assert result == "error"
    
    def test_route_continue(self):
        """Test routing when no timeout or error."""
        # Mock agent instance
        mock_agent_instance = Mock()
        mock_agent_instance.step_timed_out = False
        mock_agent_instance.last_error = None
        
        state: BrowserAgentState = {
            'task': 'test',
            'browser_state_summary': None,
            'last_model_output': None,
            'last_result': None
        }
        
        result = route_on_timeout_or_error(state, mock_agent_instance)
        assert result == "continue"