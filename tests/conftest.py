import sys, os
from unittest.mock import MagicMock

# ensure project root is on PYTHONPATH for `import app`
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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