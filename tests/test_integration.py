"""Integration tests for the complete package."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from langgraph_browser_agent import LangGraphBrowserAgent


class TestPackageIntegration:
    """Integration tests for the complete package."""
    
    def test_package_imports(self):
        """Test that all package components can be imported."""
        from langgraph_browser_agent import (
            LangGraphBrowserAgent,
            BrowserAgentState,
            create_browser_agent_graph,
        )
        
        # Verify imports work
        assert LangGraphBrowserAgent is not None
        assert BrowserAgentState is not None
        assert create_browser_agent_graph is not None
    
    def test_package_structure(self):
        """Test that package has correct structure."""
        import langgraph_browser_agent
        
        # Check that package has expected attributes
        assert hasattr(langgraph_browser_agent, 'LangGraphBrowserAgent')
        assert hasattr(langgraph_browser_agent, 'BrowserAgentState')
        assert hasattr(langgraph_browser_agent, 'create_browser_agent_graph')
    
    def test_end_to_end_workflow_initialization(self):
        """Test end-to-end workflow initialization without complex async execution."""
        # Mock original agent with all required attributes
        mock_original_agent = Mock()
        mock_original_agent.session_id = 'test-session-id-1234'
        mock_original_agent.task_id = 'test-task-id-5678'
        mock_original_agent.browser_session = Mock()
        mock_original_agent.browser_session.id = 'test-browser-id-9012'
        mock_original_agent.browser_session.cdp_url = None
        mock_original_agent.tools = Mock()
        mock_original_agent.llm = Mock()
        mock_original_agent._message_manager = Mock()
        mock_original_agent.settings = Mock()
        mock_original_agent.logger = Mock()
        mock_original_agent.state = Mock()
        mock_original_agent.task = 'test task'
        mock_original_agent.id = 'test-id'
        mock_original_agent.history = Mock()
        mock_original_agent.history.usage = None
        mock_original_agent.history._output_model_schema = None
        mock_original_agent.output_model_schema = None
        mock_original_agent.enable_cloud_sync = False
        mock_original_agent.settings.generate_gif = False
        
        # Create LangGraph agent
        agent = LangGraphBrowserAgent(mock_original_agent)
        
        # Verify agent was created successfully
        assert agent.original_agent == mock_original_agent
        assert agent.graph is not None
        assert agent.signal_handler is None  # Not set until run() is called
        
        # Verify all attributes were copied correctly
        assert agent.browser_session == mock_original_agent.browser_session
        assert agent.tools == mock_original_agent.tools
        assert agent.llm == mock_original_agent.llm
        assert agent.settings == mock_original_agent.settings
        assert agent.logger == mock_original_agent.logger
    
    def test_agent_initialization_with_minimal_mock(self):
        """Test agent initialization with minimal mocking."""
        # Minimal mock original agent
        mock_original_agent = Mock()
        mock_original_agent.browser_session = Mock()
        mock_original_agent.tools = Mock()
        mock_original_agent.llm = Mock()
        mock_original_agent._message_manager = Mock()
        mock_original_agent.settings = Mock()
        mock_original_agent.logger = Mock()
        
        # Create LangGraph agent
        agent = LangGraphBrowserAgent(mock_original_agent)
        
        # Verify basic attributes
        assert agent.original_agent == mock_original_agent
        assert agent.browser_session == mock_original_agent.browser_session
        assert agent.tools == mock_original_agent.tools
        assert agent.llm == mock_original_agent.llm
        assert agent.graph is not None
