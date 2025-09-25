import time
import asyncio
from pathlib import Path

from browser_use.agent.views import ActionResult, AgentHistoryList, AgentHistory, BrowserStateHistory

from .state import BrowserAgentState
from .graph import create_browser_agent_graph


class LangGraphBrowserAgent:
    """LangGraph version of the browser-use Agent"""

    def __init__(self, original_agent):
        self.original_agent = original_agent
        self.browser_session = original_agent.browser_session
        self.tools = original_agent.tools
        self.llm = original_agent.llm
        self._message_manager = original_agent._message_manager
        self.settings = original_agent.settings
        self.logger = original_agent.logger

        # LangGraph workflow state attributes
        self.current_step = 0
        self.max_steps = 0
        self.step_info = None
        self.last_error = None
        self.ended_due_to_break = False
        self.step_timed_out = False

        self.signal_handler = None

        self.graph = create_browser_agent_graph(self)

    async def run(
        self,
        max_steps: int = 100,
        step_timeout: int = 30,
        on_step_start=None,
        on_step_end=None,
    ) -> AgentHistoryList:

        self.original_agent.settings.step_timeout = step_timeout
    
        loop = asyncio.get_event_loop()
        agent_run_error: str | None = None
        self.original_agent._force_exit_telemetry_logged = False

        from browser_use.utils import SignalHandler

        def on_force_exit_log_telemetry():
            if hasattr(self.original_agent, '_log_agent_event'):
                self.original_agent._log_agent_event(max_steps=max_steps, agent_run_error='SIGINT: Cancelled by user')
            if hasattr(self.original_agent, 'telemetry') and self.original_agent.telemetry:
                self.original_agent.telemetry.flush()
            self.original_agent._force_exit_telemetry_logged = True

        self.signal_handler = SignalHandler(
            loop=loop,
            pause_callback=self.original_agent.pause,
            resume_callback=self.original_agent.resume,
            custom_exit_callback=on_force_exit_log_telemetry,
            exit_on_second_int=True,
        )
        self.signal_handler.register()

        try:
            await self.original_agent._log_agent_run()

            self.original_agent.logger.debug(
                f'🔧 Agent setup: Agent Session ID {self.original_agent.session_id[-4:]}, Task ID {self.original_agent.task_id[-4:]}, Browser Session ID {self.original_agent.browser_session.id[-4:] if self.original_agent.browser_session else "None"} {"(connecting via CDP)" if (self.original_agent.browser_session and self.original_agent.browser_session.cdp_url) else "(launching local browser)"}'
            )

            self.original_agent._session_start_time = time.time()
            self.original_agent._task_start_time = self.original_agent._session_start_time

            if not self.original_agent.state.session_initialized:
                if self.original_agent.enable_cloud_sync:
                    self.original_agent.logger.debug('📡 Dispatching CreateAgentSessionEvent...')
                    from browser_use.agent.cloud_events import CreateAgentSessionEvent
                    self.original_agent.eventbus.dispatch(CreateAgentSessionEvent.from_agent(self.original_agent))
                    await asyncio.sleep(0.2)
                self.original_agent.state.session_initialized = True

            if self.original_agent.enable_cloud_sync:
                self.original_agent.logger.debug('📡 Dispatching CreateAgentTaskEvent...')
                from browser_use.agent.cloud_events import CreateAgentTaskEvent
                self.original_agent.eventbus.dispatch(CreateAgentTaskEvent.from_agent(self.original_agent))

            await self.original_agent.browser_session.start()
            await self.original_agent._execute_initial_actions()
            self.original_agent._log_first_step_startup()
            self.original_agent.logger.debug(f'🔄 Starting main execution loop with max {max_steps} steps...')

            # Initialize agent attributes for this run
            self.current_step = 0
            self.max_steps = max_steps
            self.step_info = None
            self.last_error = None
            self.ended_due_to_break = False
            self.step_timed_out = False

            initial_state: BrowserAgentState = {
                'task': self.original_agent.task,
                'browser_state_summary': None,
                'last_model_output': None,
                'last_result': None
            }

            # Create config for graph execution
            config = {"recursion_limit": max_steps * 15}
            final_state = await self.graph.ainvoke(initial_state, config)

            if self.ended_due_to_break:
                pass
            else:
                agent_run_error = 'Failed to complete task in maximum steps'
                self.original_agent.history.add_item(
                    AgentHistory(
                        model_output=None,
                        result=[ActionResult(error=agent_run_error, include_in_memory=True)],
                        state=BrowserStateHistory(
                            url='',
                            title='',
                            tabs=[],
                            interacted_element=[],
                            screenshot_path=None,
                        ),
                        metadata=None,
                    )
                )
                self.original_agent.logger.info(f'❌ {agent_run_error}')

            self.original_agent.logger.debug('📊 Collecting usage summary...')
            self.original_agent.history.usage = await self.original_agent.token_cost_service.get_usage_summary()

            if self.original_agent.history._output_model_schema is None and self.original_agent.output_model_schema is not None:
                self.original_agent.history._output_model_schema = self.original_agent.output_model_schema

            self.original_agent.logger.debug('🏁 Agent.run() completed successfully')
            return self.original_agent.history

        except KeyboardInterrupt:
            self.original_agent.logger.debug('Got KeyboardInterrupt during execution, returning current history')
            agent_run_error = 'KeyboardInterrupt'
            self.original_agent.history.usage = await self.original_agent.token_cost_service.get_usage_summary()
            return self.original_agent.history

        except Exception as e:
            self.original_agent.logger.error(f'LangGraph agent run failed with exception: {e}', exc_info=True)
            agent_run_error = str(e)
            raise e

        finally:
            await self.original_agent.token_cost_service.log_usage_summary()
            self.signal_handler.unregister()
            if not self.original_agent._force_exit_telemetry_logged:
                try:
                    self.original_agent._log_agent_event(max_steps=max_steps, agent_run_error=agent_run_error)
                except Exception as log_e:
                    self.original_agent.logger.error(f'Failed to log telemetry event: {log_e}', exc_info=True)
            else:
                self.original_agent.logger.debug('Telemetry for force exit (SIGINT) was logged by custom exit callback.')

            if self.original_agent.enable_cloud_sync:
                from browser_use.agent.cloud_events import UpdateAgentTaskEvent
                self.original_agent.eventbus.dispatch(UpdateAgentTaskEvent.from_agent(self.original_agent))

            if self.original_agent.settings.generate_gif:
                output_path: str = 'agent_history.gif'
                if isinstance(self.original_agent.settings.generate_gif, str):
                    output_path = self.original_agent.settings.generate_gif
                from browser_use.agent.gif import create_history_gif
                create_history_gif(task=self.original_agent.task, history=self.original_agent.history, output_path=output_path)
                if Path(output_path).exists():
                    from browser_use.agent.cloud_events import CreateAgentOutputFileEvent
                    output_event = await CreateAgentOutputFileEvent.from_agent_and_file(self.original_agent, output_path)
                    self.original_agent.eventbus.dispatch(output_event)

            if self.original_agent.enable_cloud_sync and hasattr(self.original_agent, 'cloud_sync') and self.original_agent.cloud_sync is not None:
                if self.original_agent.cloud_sync.auth_task and not self.original_agent.cloud_sync.auth_task.done():
                    try:
                        await asyncio.wait_for(self.original_agent.cloud_sync.auth_task, timeout=1.0)
                    except TimeoutError:
                        self.original_agent.logger.debug('Cloud authentication started - continuing in background')
                    except Exception as e:
                        self.original_agent.logger.debug(f'Cloud authentication error: {e}')

            await self.original_agent.eventbus.stop(timeout=3.0)
            await self.original_agent.close()


