# Installation

## Transport Overview

cama-client supports three transports. **Only TCP is required** — RDMA and CXL are optional.

| Transport | Platform | Native deps | Install command | Use case |
|---|---|---|---|---|
| **TCP** | Any (Linux, macOS, Windows) | None | `pip install .` | Development, testing, any platform |
| **RDMA** | Linux only | libibverbs-dev, librdmacm-dev, g++/clang++ | `pip install ".[rdma]"` | Production with ConnectX NICs |
| **CXL** | Linux only | g++/clang++ | `pip install ".[cxl]"` | Experimental CXL 2.0 devdax |

The client auto-selects the best available transport at import time. If RDMA is installed but no devices are detected, it falls back to TCP with a `RuntimeWarning`. No code changes needed — the `PriskvClient` API is transport-agnostic.

## Prerequisites

| Requirement | TCP-only | RDMA | CXL |
|---|---|---|---|
| Python | >= 3.10 | >= 3.10 | >= 3.10 |
| Platform | Any (Linux, macOS, Windows) | Linux only | Linux only |
| C++ compiler | Not needed | Required (g++ or clang++) | Required (g++ or clang++) |
| pybind11 | Not needed | >= 2.11 | >= 2.11 |
| libibverbs | Not needed | Required (`libibverbs-dev`) | Not needed |
| librdmacm | Not needed | Required (`librdmacm-dev`) | Not needed |
| RDMA devices | Not needed | At least one (mlx5, rxe, etc.) | Not needed |
| CXL devdax | Not needed | Not needed | `/dev/dax*` device |

## TCP-Only Install (Simplest)

Pure Python, zero native dependencies, works on any platform:

```bash
cd cama-client
pip install .
```

This installs the `cama_client` package with only the TCP transport. No C++ extensions are compiled.

> **No git required.** The version is read from `cama_client/_version.py`, not git tags. Works from a tarball, zip, or any directory.

## RDMA Install (Linux Only)

**Step 1 — Install system RDMA libraries:**

Ubuntu / Debian:
```bash
sudo apt-get install -y libibverbs-dev librdmacm-dev
```

RHEL / CentOS / Rocky:
```bash
sudo dnf install -y rdma-core-devel
```

**Step 2 — Install with RDMA extras:**

```bash
cd cama-client
pip install ".[rdma]"
```

The `[rdma]` extra pulls in `pybind11>=2.11` as a build dependency. The `setup.py` build script detects Linux + pybind11 and automatically compiles `cama_client/csrc/rdma_transport.cpp` into `_cama_rdma.so`, linking against `-lrdmacm` and `-libverbs`.

**Step 3 — Verify the extension built:**

```python
from cama_client._cama_rdma import is_available
print(is_available())  # True if RDMA devices exist
```

## CXL Install (Linux Only — Experimental)

CXL transport supports CXL 2.0 Type 3 memory devices exposed as devdax (`/dev/dax*`).

```bash
cd cama-client
pip install ".[cxl]"
```

This compiles `cama_client/csrc/cxl_transport.cpp` into `_cama_cxl.so`. No system libraries beyond a C++ compiler are required — CXL uses `mmap` on devdax devices.

The server must also be built with CXL support: `go build -tags cxl -o cama-server .`

## Upgrading / Rebuilding After Code Changes

The installed package is a **snapshot** — changing source files does NOT affect the installed version. After receiving updated files (new release, code fix, etc.), reinstall:

```bash
# Reinstall with RDMA (rebuilds C++ extension):
pip install ".[rdma]" --force-reinstall --no-deps

# Reinstall TCP-only:
pip install . --force-reinstall --no-deps
```

If Python and C++ versions get out of sync, `import cama_client` will warn you:

```
UserWarning: RDMA extension version mismatch: _cama_rdma.so was compiled for '0.2.3'
but cama-client is '0.2.4'. Rebuild with: pip install --force-reinstall ".[rdma]"
```

You can also check manually:

```python
import cama_client._cama_rdma as m
print(m.__version__)  # should match cama_client.__version__
```

## Production Install (No Git Repository)

When installing from a tarball, zip archive, or versioned release directory (no `.git`):

```bash
# Everything works without git — version comes from _version.py
cd cama-client
pip install .               # TCP-only
pip install ".[rdma]"       # with RDMA
pip install ".[rdma,cxl]"   # with RDMA + CXL
```

No `setuptools-scm`, no git tags, no `.git` directory needed. The version is statically defined in `cama_client/_version.py`.

> **Note:** SGLang (not cama-client) is the component that requires `SETUPTOOLS_SCM_PRETEND_VERSION` when installed outside a git repo. See [cama-connector/INSTALL.md](../../cama-connector/INSTALL.md) for details.

## Verification

**Check TCP client:**

```python
from cama_client.client import CamaClient
client = CamaClient("127.0.0.1", 18000)
client.setstr("test", "hello")
print(client.getstr("test"))  # "hello"
client.close()
```

**Check RDMA availability:**

```python
from cama_client._cama_rdma import is_available
print(f"RDMA available: {is_available()}")
```

**Check CXL availability:**

```python
from cama_client._cama_cxl import CxlTransport
print("CXL extension loaded")
```

**Check auto-selection:**

```python
from cama_client import PriskvClient
print(PriskvClient)
# <class 'cama_client.rdma_client.RDMAClient'>  — if RDMA is available
# <class 'cama_client.client.CamaClient'>        — otherwise (TCP fallback)
```

## Build System Details

`setup.py` conditionally compiles two C++ extensions:

```python
# Simplified from setup.py
if sys.platform == "linux":
    try:
        from pybind11.setup_helpers import Pybind11Extension
        ext_modules = [
            # RDMA extension — requires libibverbs + librdmacm
            Pybind11Extension(
                "cama_client._cama_rdma",
                ["cama_client/csrc/rdma_transport.cpp"],
                extra_link_args=["-lrdmacm", "-libverbs"],
            ),
            # CXL extension — no system libs (mmap on devdax)
            Pybind11Extension(
                "cama_client._cama_cxl",
                ["cama_client/csrc/cxl_transport.cpp"],
            ),
        ]
    except ImportError:
        pass  # pybind11 not available — pure-Python TCP-only
```

On non-Linux platforms or when pybind11 is not installed, `ext_modules` stays empty and the package installs as pure Python. The `__init__.py` transport selection handles this gracefully — if `import _cama_rdma` raises `ImportError`, the client falls back to TCP.
