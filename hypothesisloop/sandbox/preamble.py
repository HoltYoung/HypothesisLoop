"""Trusted preamble prepended to every sandboxed script.

The AST denylist runs on the user code only — *not* on this preamble. The
preamble can therefore use ``os`` and ``pathlib`` to wire up reproducibility,
the ``hl_emit`` helper, and a sandbox env self-check.
"""

from __future__ import annotations


# Marker the runner uses to delimit preamble vs. user code in exp.py.
USER_CODE_MARKER = "# ===== USER CODE BELOW ====="


_PREAMBLE_TEMPLATE = '''\
# Auto-generated sandbox preamble — do not edit. (See hypothesisloop/sandbox/preamble.py)
import json as _hl_json
import os as _hl_os
import random as _hl_random
from pathlib import Path as _HlPath

import numpy as _hl_np

SEED = {seed}
_hl_random.seed(SEED)
_hl_np.random.seed(SEED)

# Re-export under public names the user code may reference. We deliberately
# keep these names short; the AST check ignores the preamble entirely.
import json
import random
from pathlib import Path
import numpy as np

_HL_METRICS_PATH = _HlPath("metrics.json")


def hl_emit(key, value):
    """Append/overwrite ``key`` in metrics.json (cwd). Idempotent."""
    data = {{}}
    if _HL_METRICS_PATH.exists():
        try:
            data = _hl_json.loads(_HL_METRICS_PATH.read_text())
        except Exception:
            data = {{}}
    data[key] = value
    _HL_METRICS_PATH.write_text(_hl_json.dumps(data, default=str))


# Sandbox env self-check. Written to a separate file so it cannot collide
# with user metrics. The runner does not surface this on SandboxResult; it
# exists so integration tests can confirm the env was scrubbed.
try:
    _HlPath(".sandbox_env.json").write_text(
        _hl_json.dumps(
            {{
                "openai_api_key_present": "OPENAI_API_KEY" in _hl_os.environ,
                "kimi_api_key_present": "KIMI_API_KEY" in _hl_os.environ,
                "moonshot_api_key_present": "MOONSHOT_API_KEY" in _hl_os.environ,
                "langfuse_keys_present": any(
                    k.startswith("LANGFUSE_") for k in _hl_os.environ
                ),
            }}
        )
    )
except Exception:
    pass

{user_marker}
'''


def render_preamble(seed: int) -> str:
    """Return the preamble source with ``seed`` baked in.

    The string ends with ``USER_CODE_MARKER`` on its own line; the runner
    appends user code immediately after.
    """
    return _PREAMBLE_TEMPLATE.format(seed=int(seed), user_marker=USER_CODE_MARKER)
