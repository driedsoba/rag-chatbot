import sys
from unittest.mock import MagicMock

# Mock chainlit and its dependencies to avoid configuration issues

# Create mock for chainlit
mock_chainlit = MagicMock()
mock_chainlit.on_chat_start = lambda: lambda f: f
mock_chainlit.on_message = lambda: lambda f: f
mock_chainlit.Message = MagicMock()

# Add necessary mocks for all chainlit modules
sys.modules['chainlit'] = mock_chainlit
sys.modules['chainlit.message'] = MagicMock()
sys.modules['chainlit.config'] = MagicMock()
sys.modules['chainlit.action'] = MagicMock()
sys.modules['chainlit.telemetry'] = MagicMock()