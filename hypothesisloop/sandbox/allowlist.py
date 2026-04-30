"""Static allow/deny lists for the sandbox AST check.

The runner enforces these at parse time (before any code is executed). The
threat model is "the LLM writes a buggy or wasteful script" — not a determined
attacker. Class-scope limitations to be aware of:

- We allow ``numpy`` at the top level. A determined adversary could still pull
  in ``ctypes``-adjacent surface via ``from numpy import ctypeslib``; the
  ``import`` statement passes the top-level allowlist. Acceptable for the
  course context; would not be acceptable for an untrusted-user product.
- We block ``open`` outright (no read-mode loophole). Generated code that
  needs a CSV must accept the path via ``pandas.read_csv``, which is fine.
- We block bare references to ``__builtins__`` to short-circuit
  ``__builtins__["__import__"]("os")``-style bypasses; ``__import__`` as a
  Call is also denied.
"""

from __future__ import annotations


ALLOWED_IMPORTS = frozenset(
    {
        "pandas", "numpy", "scipy", "sklearn", "matplotlib", "seaborn",
        "math", "statistics", "json", "datetime", "decimal",
        "collections", "itertools", "functools", "re",
    }
)

DENIED_IMPORTS = frozenset(
    {
        "os", "subprocess", "socket", "shutil", "sys", "requests",
        "urllib", "ctypes", "multiprocessing", "threading", "pathlib",
        "tempfile", "io", "pickle", "marshal", "importlib",
    }
)

DENIED_BUILTIN_CALLS = frozenset(
    {
        "eval", "exec", "compile", "__import__", "open",
        "globals", "locals", "vars", "getattr", "setattr", "delattr",
    }
)

DENIED_ATTRIBUTE_PREFIXES = frozenset(
    {"__class__", "__bases__", "__subclasses__", "__mro__", "__globals__", "__builtins__"}
)

# Bare-name references that would let user code reach into the interpreter
# despite our Call/Attribute checks (e.g. ``__builtins__["__import__"]("os")``).
DENIED_NAMES = frozenset({"__builtins__", "__import__"})
