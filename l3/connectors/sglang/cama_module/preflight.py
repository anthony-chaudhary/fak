"""Early fail-fast connectivity check for the CAMA/PrisKV storage backend."""

import json
import logging
import os
from typing import Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def check_cama_preflight(extra_config_json: Optional[str] = None) -> None:
    """Validate that cama-client or priskv is importable and the server is reachable.

    Called from ``ServerArgs.check_server_args()`` so misconfigurations
    surface before model loading begins.

    Args:
        extra_config_json: The raw JSON string from
            ``--hicache-storage-backend-extra-config``, if provided.
    """
    # 1. Import guard — check cama-client or priskv is installed
    try:
        try:
            from l3_client import PriskvClient
            import l3_client as _client_mod
        except ImportError:
            from cama_client import PriskvClient
            import cama_client as _client_mod
        client_version = getattr(_client_mod, "__version__", "unknown")
        logger.debug("Cama preflight: cama-client version %s", client_version)
    except ImportError:
        try:
            from priskv.priskv_client import PriskvClient
        except ImportError:
            raise ImportError(
                "CAMA storage backend requires either 'cama-client' or 'priskv'. "
                "Install with: pip install cama-client (or pip install priskv)"
            )

    # 2. Resolve config (same triple-source priority as CamaStorage.__init__)
    extra_config = None
    if extra_config_json:
        try:
            extra_config = json.loads(extra_config_json)
        except json.JSONDecodeError:
            pass

    try:
        try:
            from sglang.srt.mem_cache.storage.cama.cama_storage import CamaConfig
        except ImportError:
            from cama_module.cama_storage import CamaConfig
        cfg = CamaConfig.resolve(extra_config)
    except Exception:
        cfg = None

    if cfg is not None:
        remote_addr = cfg.remote_addr
        remote_port = cfg.remote_port
        password = cfg.password
        check_server = cfg.check_server
        pool_size = cfg.pool_size
        send_buf_size = cfg.send_buf_size
    else:
        # Minimal fallback — CamaConfig import can fail in standalone mode
        def _env_get(attr, default):
            field = getattr(envs, attr, None)
            return field.get() if field is not None else default
        remote_addr = _env_get("SGLANG_CAMA_REMOTE_ADDR", "127.0.0.1")
        remote_port = int(_env_get("SGLANG_CAMA_REMOTE_PORT", 18001))
        password = _env_get("SGLANG_CAMA_PASSWORD", "")
        check_server = _env_get("SGLANG_CAMA_CHECK_SERVER", False)
        pool_size = int(_env_get("SGLANG_CAMA_POOL_SIZE", 8))
        send_buf_size = int(_env_get("SGLANG_CAMA_SEND_BUF_SIZE", 0))

    # 3. Skip probe if check_server is enabled (user expects PrisKV to start later)
    if check_server:
        logger.debug(
            "Cama preflight: check_server=true, skipping early connectivity probe "
            "(will poll during backend init)."
        )
        return

    # 4. Lightweight connectivity probe
    conn = None
    try:
        conn = PriskvClient(remote_addr, remote_port, password)
        conn.exists("__cama_preflight__")
        logger.debug(
            "Cama preflight: PrisKV reachable at %s:%d", remote_addr, remote_port
        )
    except Exception as e:
        raise RuntimeError(
            f"Cama preflight check failed: PrisKV server at {remote_addr}:{remote_port} "
            f"is not reachable. Error: {e}\n"
            f"Ensure the PrisKV server is running before starting SGLang, "
            f"or set check_server=true to poll until it's ready."
        ) from e
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    # 5. Signal child processes AND start prewarm in this process.
    #    SGLang uses mp.set_start_method("spawn") — children get fresh
    #    interpreters and cannot see the parent's PrewarmRegistry, so we
    #    serialize config to an env var that children read at import time
    #    (environ.py).  We also start prewarm directly here because
    #    environ.py's module-level hook ran before this env var was set.
    try:
        nic_striping = cfg.nic_striping if cfg is not None else True
        model_page_bytes = cfg.model_page_bytes if cfg is not None else 0
        signal = json.dumps({
            "addr": remote_addr,
            "port": remote_port,
            "password": password,
            "pool_size": pool_size,
            "send_buf_size": send_buf_size,
            "nic_striping": nic_striping,
            "model_page_bytes": model_page_bytes,
        })
        os.environ["_SGLANG_CAMA_PREWARM_SIGNAL"] = signal
        logger.debug(
            "Cama preflight: prewarm signaled for child processes "
            "(addr=%s:%d, pool_size=%d, buf=%s, nic_striping=%s)",
            remote_addr, remote_port, pool_size,
            f"{send_buf_size / (1 << 20):.0f} MB" if send_buf_size > 0 else "16 MB (default)",
            nic_striping,
        )
    except Exception as exc:
        logger.warning("Cama preflight: failed to set prewarm signal: %s", exc)

    # Also start prewarm directly in this process.  environ.py's
    # module-level _maybe_start_rank_prewarm() already ran (before
    # preflight), so the env var wasn't set yet — the parent process
    # never fires prewarm from environ.py.  Starting here covers
    # single-GPU / same-process mode.  In multi-GPU mode this goes
    # unused (daemon thread, harmless).
    try:
        try:
            from sglang.srt.mem_cache.storage.cama.prewarm import start_cama_prewarm
        except ImportError:
            from cama_module.prewarm import start_cama_prewarm
        start_cama_prewarm(
            remote_addr, remote_port, password,
            pool_size=pool_size,
            send_buf_size=send_buf_size,
            nic_striping=nic_striping,
            model_page_bytes=model_page_bytes,
        )
    except Exception as exc:
        logger.warning("Cama preflight: failed to start direct prewarm: %s", exc)
