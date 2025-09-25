"""Tests for LangGraphBrowserAgent class."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from langgraph_browser_agent import LangGraphBrowserAgent


class TestLangGraphBrowserAgent:
    """Test LangGraphBrowserAgent class."""
    
    def test_init(self):
        """Test agent initialization."""
        # Mock original agent
        mock_original_agent = Mock()
        mock_original_agent.browser_session = Mock()
        mock_original_agent.tools = Mock()
        mock_original_agent.llm = Mock()
        mock_original_agent._message_manager = Mock()
        mock_original_agent.settings = Mock()
        mock_original_agent.logger = Mock()
        
        # Create LangGraph agent
        agent = LangGraphBrowserAgent(mock_original_agent)
        
        # Verify attributes are copied
        assert agent.original_agent == mock_original_agent
        assert agent.browser_session == mock_original_agent.browser_session
        assert agent.tools == mock_original_agent.tools
        assert agent.llm == mock_original_agent.llm
        assert agent._message_manager == mock_original_agent._message_manager
        assert agent.settings == mock_original_agent.settings
        assert agent.logger == mock_original_agent.logger
        
        # Verify graph is created
        assert agent.graph is not None
        assert agent.signal_handler is None
    
    def test_init_with_optional_attributes(self):
        """Test initialization with optional attributes."""
        # Mock original agent with optional attributes
        mock_original_agent = Mock()
        mock_original_agent.browser_session = Mock()
        mock_original_agent.tools = Mock()
        mock_original_agent.llm = Mock()
        mock_original_agent._message_manager = Mock()
        mock_original_agent.settings = Mock()
        mock_original_agent.logger = Mock()
        mock_original_agent.include_recent_events = True
        mock_original_agent.sensitive_data = "test data"
        mock_original_agent.available_file_paths = ["path1", "path2"]
        
        # Create LangGraph agent
        agent = LangGraphBrowserAgent(mock_original_agent)
        
        # Verify optional attributes are copied
        assert agent.include_recent_events is True
        assert agent.sensitive_data == "test data"
        assert agent.available_file_paths == ["path1", "path2"]
    
    def test_init_without_optional_attributes(self):
        """Test initialization without optional attributes."""
        # Mock original agent without optional attributes
        mock_original_agent = Mock()
        mock_original_agent.browser_session = Mock()
        mock_original_agent.tools = Mock()
        mock_original_agent.llm = Mock()
        mock_original_agent._message_manager = Mock()
        mock_original_agent.settings = Mock()
        mock_original_agent.logger = Mock()
        
        # Remove optional attributes to test getattr behavior
        del mock_original_agent.include_recent_events
        del mock_original_agent.sensitive_data
        del mock_original_agent.available_file_paths
        
        # Create LangGraph agent
        agent = LangGraphBrowserAgent(mock_original_agent)
        
        # Verify optional attributes have default values
        assert agent.include_recent_events is False
        assert agent.sensitive_data is None
        assert agent.available_file_paths is None


class TestLangGraphBrowserAgentRun:
    """Test LangGraphBrowserAgent.run method."""
    
    def test_run_initialization_only(self):
        """Test agent initialization without running complex async workflow."""
        # Mock original agent
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
