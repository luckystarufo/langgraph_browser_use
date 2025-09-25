"""Tests for node implementations."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import time

from langgraph_browser_agent.state import BrowserAgentState
from langgraph_browser_agent.nodes import (
    check_paused_node,
    check_consecutive_failures_node,
    check_stopped_node,
    on_step_start_node,
    on_step_end_node,
    paused_state_actions_node,
    consecutive_failure_actions_node,
    stopped_state_actions_node,
    history_is_done_actions_node,
    check_step_timeout,
    prepare_context_node,
    get_next_action_node,
    execute_actions_node,
    evaluate_result_node,
    finalize_step_node,
    handle_error_node,
)


class TestCheckNodes:
    """Test check nodes."""
    
    def test_check_paused_node(self):
        """Test check_paused_node."""
        # Mock agent instance
        mock_agent_instance = Mock()
        
        state: BrowserAgentState = {
            'task': 'test',
            'browser_state_summary': None,
            'last_model_output': None,
            'last_result': None
        }
        
        result = check_paused_node(state, mock_agent_instance)
        
        # Verify state was returned unchanged
        assert result == state
    
    def test_check_consecutive_failures_node(self):
        """Test check_consecutive_failures_node."""
        # Mock agent instance
        mock_agent_instance = Mock()
        
        state: BrowserAgentState = {
            'task': 'test',
            'browser_state_summary': None,
            'last_model_output': None,
            'last_result': None
        }
        
        result = check_consecutive_failures_node(state, mock_agent_instance)
        
        # Verify state was returned unchanged
        assert result == state
    
    def test_check_stopped_node(self):
        """Test check_stopped_node."""
        # Mock agent instance
        mock_agent_instance = Mock()
        
        state: BrowserAgentState = {
            'task': 'test',
            'browser_state_summary': None,
            'last_model_output': None,
            'last_result': None
        }
        
        result = check_stopped_node(state, mock_agent_instance)
        
        # Verify state was returned unchanged
        assert result == state


class TestActionNodes:
    """Test action nodes."""
    
    @pytest.mark.asyncio
    async def test_paused_state_actions_node(self):
        """Test paused_state_actions_node."""
        # Mock agent instance
        mock_agent_instance = Mock()
        mock_agent_instance.current_step = 0
        mock_agent_instance.original_agent.logger = Mock()
        mock_agent_instance.original_agent._external_pause_event = AsyncMock()
        mock_agent_instance.signal_handler = Mock()
        mock_agent_instance.signal_handler.reset = Mock()
        
        state: BrowserAgentState = {
            'task': 'test',
            'browser_state_summary': None,
            'last_model_output': None,
            'last_result': None
        }
        
        result = await paused_state_actions_node(state, mock_agent_instance)
        
        # Verify pause event was waited for
        mock_agent_instance.original_agent._external_pause_event.wait.assert_called_once()
        # Verify signal handler was reset
        mock_agent_instance.signal_handler.reset.assert_called_once()
    
    def test_consecutive_failure_actions_node(self):
        """Test consecutive_failure_actions_node."""
        # Mock agent instance
        mock_agent_instance = Mock()
        mock_agent_instance.original_agent.logger = Mock()
        mock_agent_instance.original_agent.settings.max_failures = 3
        
        state: BrowserAgentState = {
            'task': 'test',
            'browser_state_summary': None,
            'last_model_output': None,
            'last_result': None
        }
        
        result = consecutive_failure_actions_node(state, mock_agent_instance)
        
        # Verify ended_due_to_break is set on agent
        assert mock_agent_instance.ended_due_to_break is True
    
    def test_stopped_state_actions_node(self):
        """Test stopped_state_actions_node."""
        # Mock agent instance
        mock_agent_instance = Mock()
        mock_agent_instance.original_agent.logger = Mock()
        
        state: BrowserAgentState = {
            'task': 'test',
            'browser_state_summary': None,
            'last_model_output': None,
            'last_result': None
        }
        
        result = stopped_state_actions_node(state, mock_agent_instance)
        
        # Verify ended_due_to_break is set on agent
        assert mock_agent_instance.ended_due_to_break is True


class TestStepTimeout:
    """Test step timeout functionality."""
    
    def test_check_step_timeout_no_timeout(self):
        """Test check_step_timeout when no timeout occurs."""
        # Mock agent
        mock_agent = Mock()
        mock_agent.current_step = 0
        mock_agent.step_timed_out = False
        mock_agent.original_agent.step_start_time = time.time() - 10  # 10 seconds ago
        mock_agent.original_agent.settings.step_timeout = 30  # 30 second timeout
        
        state: BrowserAgentState = {
            'task': 'test',
            'browser_state_summary': None,
            'last_model_output': None,
            'last_result': None
        }
        
        result = check_step_timeout(state, mock_agent)
        
        # Should not timeout
        assert result is False
        assert mock_agent.step_timed_out is False
    
    def test_check_step_timeout_with_timeout(self):
        """Test check_step_timeout when timeout occurs."""
        # Mock agent
        mock_agent = Mock()
        mock_agent.current_step = 0
        mock_agent.step_timed_out = False
        mock_agent.original_agent.step_start_time = time.time() - 40  # 40 seconds ago
        mock_agent.original_agent.settings.step_timeout = 30  # 30 second timeout
        mock_agent.original_agent.logger = Mock()
        mock_agent.original_agent.state.consecutive_failures = 0
        
        state: BrowserAgentState = {
            'task': 'test',
            'browser_state_summary': None,
            'last_model_output': None,
            'last_result': None
        }
        
        result = check_step_timeout(state, mock_agent)
        
        # Should timeout
        assert result is True
        assert mock_agent.step_timed_out is True
        assert mock_agent.original_agent.state.consecutive_failures == 1