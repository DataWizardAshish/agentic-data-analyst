import threading

import dspy

from config import OPENAI_API_KEY

_lock = threading.Lock()
_configured = False


def get_configured_lm():
    global _configured
    if not _configured:
        with _lock:
            if not _configured:
                dspy.configure(lm=dspy.LM("openai/gpt-4.1", api_key=OPENAI_API_KEY))
                _configured = True
    return dspy.settings.lm
