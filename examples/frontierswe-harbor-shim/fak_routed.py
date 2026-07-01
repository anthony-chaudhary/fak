#!/usr/bin/env python3
"""harbor_ext.fak_routed — route a FrontierSWE agent's model traffic through a fak gateway.

FrontierSWE (Proximal Labs) drives each trial with a pluggable agent named in ``job.yaml``
by ``import_path`` — e.g.::

    agents:
      - name: claude-code-api-key-no-search
        import_path: harbor_ext.claude_code:ClaudeCodeApiKeyNoSearch
        model_name: anthropic/claude-opus-4-6
        override_timeout_sec: 72000
        kwargs: { effort_level: max }

Every ``harbor_ext`` agent class wraps a CLI coding harness (claude-code, codex,
gemini-cli, qwen-code, …) pointed at an OpenAI-compatible model endpoint through a
``base_url``. For fak's value stack to bite on a 20-hour trial — KV persistence +
RadixAttention prefix reuse so turn *k* doesn't re-prefill turns ``1..k-1``, in-kernel
adjudication instead of a per-tool-call hook spawn, vDSO call-elimination — that
``base_url`` must point at a ``fak serve`` / ``fak guard --serve`` gateway rather than the
model vendor. This shim is the seam that makes that happen.

`FakRoutedAgent` does **exactly one thing**: it constructs the real agent unchanged and
overrides its model ``base_url`` to the fak gateway. ``model_name``,
``override_timeout_sec``, and every other kwarg pass through untouched, so the trial runs
the SAME model with the SAME prompts and the SAME budget — only the serving path changes.
That single-dial property is what keeps the raw-vs-fak comparison honest (epic #1706, C11's
score-parity gate): identical agent, identical work, one changed thing.

Registration recipe: see this directory's README.md. In short, make this module importable
as ``harbor_ext.fak_routed`` (a namespace-package drop-in on your FrontierSWE checkout's
PYTHONPATH), then in ``job.yaml`` point the agent at
``harbor_ext.fak_routed:FakRoutedAgent`` with a ``wrapped`` kwarg naming the real agent.

No fak or harbor_ext import at module load: the wrapped agent is resolved lazily by
``import_path`` at construction time, so this file is importable — and unit-testable —
with neither package installed.
"""

from __future__ import annotations

import importlib
import ipaddress
import os
import urllib.parse
from typing import Any

# The default in-sandbox gateway. FrontierSWE tasks run with allow_internet=false, so the
# gateway is co-resident in the sandbox on loopback; 8080 is fak serve's default port and
# /v1 is the OpenAI-compatible base the harnesses post to.
DEFAULT_FAK_BASE_URL = os.environ.get("FAK_GATEWAY_URL", "http://127.0.0.1:8080/v1")

# The environment variables the various CLI harnesses read a model base URL from. The shim
# sets ALL of them for the wrapped agent's child harness process, so whichever one a given
# harness honors picks up the gateway. (Set, not appended — the point is to REPLACE the
# vendor endpoint.)
_BASE_URL_ENV_KEYS = (
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "ANTHROPIC_BASE_URL",
    "OPENROUTER_BASE_URL",
    "QWEN_BASE_URL",
)

# The instance/constructor attribute names those same harnesses expose a base URL under.
# After building the wrapped agent, any of these present on the instance is overridden.
_BASE_URL_ATTRS = (
    "base_url",
    "openai_base_url",
    "openrouter_base_url",
    "qwen_base_url",
    "api_base",
)

# Control kwargs the shim consumes itself — everything else flows to the wrapped agent.
_CONTROL_KWARGS = ("wrapped", "fak_base_url", "allow_internet")


def _resolve_wrapped(spec: Any) -> type:
    """Resolve the wrapped agent class from an import_path string or an already-resolved
    class. Accepts both ``module:Class`` (FrontierSWE's convention) and ``module.Class``.
    A class object is returned as-is (the programmatic / test path)."""
    if isinstance(spec, type):
        return spec
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(
            "FakRoutedAgent needs a `wrapped` kwarg: the import_path of the real agent to "
            "route (e.g. 'harbor_ext.claude_code:ClaudeCodeApiKeyNoSearch'), or a class"
        )
    mod_name, _, attr = spec.partition(":")
    if not attr:  # fall back to the dotted 'module.Class' form
        mod_name, _, attr = spec.rpartition(".")
    if not mod_name or not attr:
        raise ValueError(f"FakRoutedAgent: malformed wrapped import_path {spec!r}")
    module = importlib.import_module(mod_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(f"FakRoutedAgent: {spec!r} has no attribute {attr!r}") from exc


def _is_in_sandbox(base_url: str) -> bool:
    """A gateway URL is in-sandbox when its host is loopback or a private/link-local
    address (or the bare hostname 'localhost'). This is the check that enforces
    allow_internet=false: the gateway must be co-resident, never an external call."""
    host = urllib.parse.urlparse(base_url).hostname or ""
    if host in ("localhost", ""):
        return host == "localhost"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # A non-IP hostname under allow_internet=false must be an explicitly pinned host,
        # which the caller resolves upstream; treat an unpinned name as NOT in-sandbox.
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


class FakRoutedAgent:
    """A harbor_ext-registerable agent that wraps another agent and reroutes its model
    ``base_url`` to a fak gateway, changing nothing else.

    Constructor mirrors how FrontierSWE instantiates an agent from ``job.yaml``: the harness
    calls it with ``model_name`` / ``override_timeout_sec`` positionally-or-by-keyword and
    the ``kwargs`` block splatted in. The shim-control keys (``wrapped``, ``fak_base_url``,
    ``allow_internet``) are pulled out of that kwargs block; the rest reach the wrapped
    agent verbatim.
    """

    def __init__(
        self,
        model_name: str | None = None,
        override_timeout_sec: int | None = None,
        *,
        wrapped: Any = None,
        fak_base_url: str | None = None,
        allow_internet: bool = False,
        **passthrough: Any,
    ) -> None:
        cls = _resolve_wrapped(wrapped)
        base_url = (fak_base_url or DEFAULT_FAK_BASE_URL).strip()
        if not base_url:
            raise ValueError("FakRoutedAgent: empty fak_base_url")

        # allow_internet=false ⇒ the gateway must be in-sandbox (loopback/pinned), never an
        # external endpoint. Refuse a misconfiguration loudly rather than silently leaking
        # the trial's traffic off the sandbox.
        if not allow_internet and not _is_in_sandbox(base_url):
            raise ValueError(
                f"FakRoutedAgent: allow_internet=false but fak_base_url={base_url!r} is not "
                "an in-sandbox (loopback/private/pinned) host — the gateway must be "
                "co-resident with the trial, not an external call"
            )

        self._fak_base_url = base_url
        self._allow_internet = allow_internet

        # Route the child harness process's model traffic through the gateway. Snapshot the
        # prior values so restore() can put the environment back (tests, repeated runs).
        self._env_backup: dict[str, str | None] = {}
        for key in _BASE_URL_ENV_KEYS:
            self._env_backup[key] = os.environ.get(key)
            os.environ[key] = base_url

        # Build the real agent with model_name / override_timeout_sec / kwargs UNCHANGED —
        # this is the "only delta is the base URL" invariant the parity gate depends on.
        self._wrapped = cls(model_name, override_timeout_sec, **passthrough)

        # Override whichever base-url attribute the wrapped agent exposes.
        for attr in _BASE_URL_ATTRS:
            if hasattr(self._wrapped, attr):
                setattr(self._wrapped, attr, base_url)

    @property
    def wrapped(self) -> Any:
        """The underlying agent instance (base_url already rerouted)."""
        return self._wrapped

    @property
    def fak_base_url(self) -> str:
        """The gateway URL every base-url seam was pointed at."""
        return self._fak_base_url

    def restore_env(self) -> None:
        """Restore the base-url environment variables to their pre-shim values."""
        for key, prev in self._env_backup.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev

    def __getattr__(self, name: str) -> Any:
        # Anything not defined on the shim (run, __call__, solve, …) delegates to the
        # wrapped agent, so the harness drives it exactly as it would the un-routed agent.
        # Guard on _wrapped via __dict__ so an attribute access before construction finishes
        # raises AttributeError (not KeyError / infinite recursion).
        wrapped = self.__dict__.get("_wrapped")
        if wrapped is None:
            raise AttributeError(name)
        return getattr(wrapped, name)
