"""Tier 0 — Error class and retriability tests.

Validates CamaServerOverloadError, CamaOOMError, and is_retriable() behavior.
Pure Python, runs anywhere.
"""

import pytest

from l3_client.errors import CamaOOMError, CamaServerOverloadError
from l3_client.reconnect import is_retriable


class TestCamaServerOverloadError:
    """CamaServerOverloadError is a RuntimeError that is NOT retriable."""

    def test_is_runtime_error(self):
        exc = CamaServerOverloadError("dispatch queue full")
        assert isinstance(exc, RuntimeError)

    def test_message_contains_server_text(self):
        exc = CamaServerOverloadError("dispatch queue full")
        assert "dispatch queue full" in str(exc)
        assert "CAMA server overloaded" in str(exc)

    def test_server_message_attribute(self):
        exc = CamaServerOverloadError("dispatch queue full")
        assert exc.server_message == "dispatch queue full"

    def test_not_retriable(self):
        """CamaServerOverloadError must NOT trigger reconnection."""
        exc = CamaServerOverloadError("server overloaded: dispatch queue full")
        assert is_retriable(exc) is False

    def test_catchable_separately_from_generic_runtime_error(self):
        """Connector can catch CamaServerOverloadError without catching
        all RuntimeErrors."""
        with pytest.raises(CamaServerOverloadError):
            raise CamaServerOverloadError("test")


class TestIsRetriable:
    """Verify is_retriable() classifies exceptions correctly."""

    def test_cama_error_prefix_not_retriable(self):
        exc = RuntimeError("CAMA error: some server error")
        assert is_retriable(exc) is False

    def test_pool_rebuilding_is_retriable(self):
        exc = RuntimeError("pool rebuilding in progress")
        assert is_retriable(exc) is True

    def test_overload_error_not_retriable(self):
        exc = CamaServerOverloadError("server overloaded: dispatch queue full")
        assert is_retriable(exc) is False

    def test_oom_error_not_retriable(self):
        exc = CamaOOMError(95, 1000, 1024, "shard full")
        # OOM is a RuntimeError — but has "CAMA OOM:" prefix, not
        # "CAMA error:", so it goes through the generic RuntimeError path.
        # It should NOT be retriable (no reconnect will help).
        assert is_retriable(exc) is False

    def test_broken_pipe_is_retriable(self):
        exc = BrokenPipeError("broken pipe")
        assert is_retriable(exc) is True

    def test_connection_reset_is_retriable(self):
        exc = ConnectionResetError("reset")
        assert is_retriable(exc) is True

    def test_value_error_not_retriable(self):
        assert is_retriable(ValueError("bad")) is False

    def test_assertion_error_not_retriable(self):
        assert is_retriable(AssertionError("bad")) is False
