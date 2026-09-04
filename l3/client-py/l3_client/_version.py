"""Centralized version constants for cama-client."""

__version__ = "1.2.0"

PROTOCOL_VERSION = 0x01
API_VERSION = 1
MIN_SERVER_VERSION = "0.2.0"
MIN_SGLANG_VERSION = "0.5.9"


def compare_versions(a: str, b: str) -> int:
    """Compare two 'X.Y.Z' semver strings numerically.

    Returns -1 if a < b, 0 if a == b, +1 if a > b.
    Non-numeric or malformed segments are treated as 0.
    """
    a_parts = a.split(".")
    b_parts = b.split(".")
    max_len = max(len(a_parts), len(b_parts))
    for i in range(max_len):
        av = int(a_parts[i]) if i < len(a_parts) and a_parts[i].isdigit() else 0
        bv = int(b_parts[i]) if i < len(b_parts) and b_parts[i].isdigit() else 0
        if av < bv:
            return -1
        if av > bv:
            return 1
    return 0


def check_sglang_compatibility() -> bool:
    """Check if installed SGLang version meets minimum compatibility requirements."""
    try:
        import sglang
        v = getattr(sglang, "__version__", "")
        return compare_versions(v, MIN_SGLANG_VERSION) >= 0
    except ImportError:
        return True
