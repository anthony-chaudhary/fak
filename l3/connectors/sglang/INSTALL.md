# CAMA Installation Guide

Manual installation steps for the CAMA storage backend with SGLang HiCache.

For one-command automated setup (venv + SGLang + CAMA + client), see:
```bash
python deploy.py /path/to/sglang --setup
```

> **No git required for CAMA components.** The CAMA server, client, and connector all work without a `.git` directory. Only SGLang requires `SETUPTOOLS_SCM_PRETEND_VERSION` when installed from a non-git tree (see [Troubleshooting](#troubleshooting) below).

---

## What Gets Installed

CAMA integration involves three layers. All three must be in place.

```
+-----------------------------------------------------------+
|  Layer 3: KV cache client (cama-client or priskv)         |
|           Python package -- talks to the storage server    |
+-----------------------------------------------------------+
|  Layer 2: CAMA module (4 files)                           |
|           CamaStorage class -- the SGLang storage backend  |
+-----------------------------------------------------------+
|  Layer 1: SGLang + patched integration files (4 files)    |
|           Wires CAMA into SGLang's backend system          |
+-----------------------------------------------------------+
|  Foundation: Python venv with torch, sgl-kernel, etc.     |
+-----------------------------------------------------------+
```

### Layer 2 -- Module files (copied into SGLang)

| File | Target in SGLang | Purpose |
|------|-----------------|---------|
| `cama_storage.py` | `.../storage/cama/` | CamaStorage class -- zero-copy RDMA via PrisKV |
| `preflight.py` | `.../storage/cama/` | Fail-fast server connectivity check |
| `profiling.py` | `.../storage/cama/` | Pyroscope + NVTX profiling hooks |
| `__init__.py` | `.../storage/cama/` | Package marker |

### Layer 1 -- Patched SGLang files (replaced in-place)

| Patch file | Replaces in SGLang | What it adds |
|------------|-------------------|-------------|
| `environ.py` | `python/sglang/srt/environ.py` | 11 `SGLANG_CAMA_*` env vars |
| `server_args.py` | `python/sglang/srt/server_args.py` | `"cama"` in `--hicache-storage-backend` + preflight |
| `backend_factory.py` | `.../storage/backend_factory.py` | CamaStorage factory registration |
| `cache_controller.py` | `.../managers/cache_controller.py` | `"cama"` in zero-copy list + ThreadPool prefetch |

---

## Choose Your Path

| Path | Starting point | Example |
|------|---------------|---------|
| **A** | Fresh SGLang clone | You don't have SGLang yet |
| **B** | Pre-packaged SGLang tree | `sglang-v0.5.7/` or a zip you received |

Both end at the same place: a venv with everything linked and working.

---

## Path A -- Fresh SGLang from Source

### A1. Create a virtual environment

```bash
# Pick where your SGLang will live
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Create a dedicated venv inside the SGLang tree
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

Everything below runs inside this activated venv.

### A2. Install SGLang

```bash
pip install "python/" \
    --find-links https://flashinfer.ai/whl/cu124/torch2.5/
```

This installs SGLang, torch, sgl-kernel, flashinfer, and all other deps
into the venv. The package is copied into `site-packages` as a snapshot.

### A3. Verify SGLang imports

```bash
python -c "
import sglang;     print('sglang:',     sglang.__file__)
import sgl_kernel;  print('sgl-kernel:', sgl_kernel.__version__)
import flashinfer;  print('flashinfer:', flashinfer.__version__)
"
```

If `sgl-kernel` or `flashinfer` fail:

```bash
pip install sgl-kernel
pip install flashinfer_python flashinfer_cubin \
    --find-links https://flashinfer.ai/whl/cu124/torch2.5/
```

### A4. Deploy CAMA into the SGLang tree

**Automated:**
```bash
cd /path/to/cama-connector
python deploy.py /path/to/sglang
```

**Manual:**
```bash
SGLANG=/path/to/sglang
CONNECTOR=/path/to/cama-connector

# -- Module: copy the CAMA backend --
mkdir -p "$SGLANG/python/sglang/srt/mem_cache/storage/cama/"
cp "$CONNECTOR"/cama_module/{cama_storage,preflight,profiling,__init__}.py \
   "$SGLANG/python/sglang/srt/mem_cache/storage/cama/"

# -- Patches: back up originals, then overwrite --
cp "$SGLANG/python/sglang/srt/environ.py"                            "$SGLANG/python/sglang/srt/environ.py.orig"
cp "$SGLANG/python/sglang/srt/server_args.py"                        "$SGLANG/python/sglang/srt/server_args.py.orig"
cp "$SGLANG/python/sglang/srt/mem_cache/storage/backend_factory.py"  "$SGLANG/python/sglang/srt/mem_cache/storage/backend_factory.py.orig"
cp "$SGLANG/python/sglang/srt/managers/cache_controller.py"          "$SGLANG/python/sglang/srt/managers/cache_controller.py.orig"

cp "$CONNECTOR"/patches/environ.py          "$SGLANG/python/sglang/srt/environ.py"
cp "$CONNECTOR"/patches/server_args.py      "$SGLANG/python/sglang/srt/server_args.py"
cp "$CONNECTOR"/patches/backend_factory.py  "$SGLANG/python/sglang/srt/mem_cache/storage/backend_factory.py"
cp "$CONNECTOR"/patches/cache_controller.py "$SGLANG/python/sglang/srt/managers/cache_controller.py"
```

After deploying, reinstall SGLang to pick up the changes:

```bash
pip install "python/" --force-reinstall --no-deps \
    --find-links https://flashinfer.ai/whl/cu124/torch2.5/
```

### A5. Install the KV cache client

```bash
# Option A: cama-client — TCP-only (any platform, zero native deps)
pip install cama-client

# Option A+: cama-client with RDMA (Linux, requires libibverbs-dev + librdmacm-dev)
pip install "cama-client[rdma]"

# Option B: priskv (for PrisKV server, port 6379)
pip install priskv
```

> No git required — cama-client version is in `_version.py`, not git tags.

### A6. Verify everything

```bash
python -c "
import sglang;     print('sglang:',     sglang.__file__)
import sgl_kernel;  print('sgl-kernel:', sgl_kernel.__version__)
import flashinfer;  print('flashinfer:', flashinfer.__version__)
from sglang.srt.mem_cache.storage.cama.cama_storage import CamaStorage
print('CamaStorage: OK')

# Uncomment the one you installed:
from cama_client import PriskvClient, SGL; print('cama-client: OK')
# from priskv.priskv_client import PriskvClient; print('priskv: OK')
"
```

### A7. Continue to [Start the Server](#start-the-server)

---

## Path B -- Pre-Built SGLang Tree

The `sglang-with-cama-connector/` directory is a complete SGLang v0.5.7
source tree with the CAMA connector already deployed: the 4 module files and
4 integration patches are pre-applied. **No deploy step is needed.**

### What's in the tree

```
sglang-with-cama-connector/
├── python/                    # SGLang source + CAMA module + patches (pre-applied)
├── wheels/                    # Bundled flashinfer wheels for offline install
│   ├── flashinfer_python-0.5.3-py3-none-any.whl   (6.7 MB)
│   └── flashinfer_cubin-0.5.3-py3-none-any.whl    (98.7 MB)
└── ...
```

### What needs network access

| Package | Source | Notes |
|---------|--------|-------|
| `flashinfer_python`, `flashinfer_cubin` | **Local** (`wheels/`) | Bundled — no network needed |
| `torch`, `torchvision`, `torchaudio` | PyPI | ~2.5 GB — largest downloads |
| `sgl-kernel` | PyPI | Pre-compiled CUDA kernels |
| All other SGLang deps | PyPI | `transformers`, `numpy`, etc. |
| `cama-client` | **Local** (`../cama-client/`) | From the monorepo — not on PyPI |

`wheels/` solves the flashinfer problem (custom wheel index at `flashinfer.ai`
that may be unreachable from isolated networks). Everything else installs
from standard PyPI.

For a **fully air-gapped** machine, pre-download all PyPI packages on a
networked machine (`pip download -r <requirements>`) and add them to `wheels/`.

### B1. Fresh install

```bash
cd sglang-with-cama-connector

# ── Step 1: Create isolated Python environment ──────────────────────
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# ── Step 2: Install SGLang + all dependencies ───────────────────────
#   • SETUPTOOLS_SCM_PRETEND_VERSION — required because the tree has
#     no .git metadata; without it, setuptools-scm fails with
#     "unable to detect version"
#   • --find-links wheels/ — resolves flashinfer from the bundled
#     wheels (offline; no network needed for flashinfer).
#     All other deps (torch, sgl-kernel, etc.) install from PyPI.
#   • NOT using -e (editable): installs a snapshot into site-packages.
#     This ensures the installed state matches the source tree exactly.
#     On update, reinstall explicitly (see B2 below).
SETUPTOOLS_SCM_PRETEND_VERSION=0.5.7 pip install "python/" \
    --find-links wheels/

# ── Step 3: Install the CAMA client ─────────────────────────────────
#   • ".[rdma]" builds the C++ RDMA extension (_cama_rdma.so).
#     Requires: libibverbs-dev, librdmacm-dev
#   • For TCP-only (no RDMA): pip install ../cama-client
cd ../cama-client
pip install ".[rdma]"
cd ../sglang-with-cama-connector

# ── Step 4: Verify all three layers ─────────────────────────────────
python -c "
import sglang;     print('sglang:',     sglang.__version__)
import sgl_kernel;  print('sgl-kernel:', sgl_kernel.__version__)
import flashinfer;  print('flashinfer:', flashinfer.__version__)
from sglang.srt.mem_cache.storage.cama.cama_storage import CamaStorage
print('CamaStorage: OK')
from cama_client import PriskvClient, SGL; print('cama-client: OK')
"
```

Expected output:
```
sglang:     0.5.7
sgl-kernel: 0.3.20
flashinfer: 0.5.3
CamaStorage: OK
cama-client: OK
```

If any import fails, see [Troubleshooting](#troubleshooting) below.

### B2. Update an existing installation

When you receive an updated source tree or updated files (new CAMA release,
connector update, client fix), you must reinstall to pick up the changes.
**There are no editable installs — the installed state is a snapshot.
Changing files in the source tree does NOT affect the running installation
until you reinstall.**

```bash
cd sglang-with-cama-connector
source .venv/bin/activate

# ── Reinstall SGLang (picks up updated CAMA module + patches) ───────
#   --force-reinstall  — forces pip to reinstall even if the version
#                        string hasn't changed (e.g., still "0.5.7")
#   --no-deps          — skips dependency resolution. Safe when only
#                        CAMA files changed (torch/sgl-kernel/etc.
#                        are unchanged). Drop --no-deps if the SGLang
#                        tree itself was upgraded to a new version.
SETUPTOOLS_SCM_PRETEND_VERSION=0.5.7 pip install "python/" \
    --find-links wheels/ \
    --force-reinstall --no-deps

# ── Reinstall the CAMA client (rebuilds C++ RDMA extension) ─────────
#   Always reinstall when updating — Python code AND the C++ extension
#   both live in this package, and pip cannot detect C++ source changes
#   without --force-reinstall.
cd ../cama-client
pip install ".[rdma]" --force-reinstall --no-deps
cd ../sglang-with-cama-connector

# ── Verify ──────────────────────────────────────────────────────────
python -c "
from sglang.srt.mem_cache.storage.cama.cama_storage import CamaStorage
print('CamaStorage: OK')
from cama_client import PriskvClient, SGL; print('cama-client: OK')
"
```

**When to reinstall what:**

| What changed | Command |
|---|---|
| CAMA module files (`cama_storage.py`, etc.) | `pip install "python/" --force-reinstall --no-deps` |
| Patched SGLang files (`cache_controller.py`, etc.) | `pip install "python/" --force-reinstall --no-deps` |
| cama-client Python or C++ code | `pip install ".[rdma]" --force-reinstall --no-deps` |
| SGLang version upgrade (new tree) | `pip install "python/" --force-reinstall` (without `--no-deps`) |
| Everything | Run both reinstall commands above |

### B3. Continue to [Start the Server](#start-the-server)

---

## Automated Setup (deploy.py --setup)

If you prefer a single command that does everything above:

```bash
# Full setup with cama-client (default)
python deploy.py /path/to/sglang --setup

# Full setup with priskv instead
python deploy.py /path/to/sglang --setup --client priskv

# Custom venv location
python deploy.py /path/to/sglang --setup --venv-dir /opt/sglang-venv

# Recreate venv from scratch
python deploy.py /path/to/sglang --setup --fresh
```

This creates a venv at `<sglang>/.venv`, installs SGLang + all
dependencies, deploys CAMA module + patches, installs the KV cache
client, and verifies all imports.

---

## Install CUDA Toolkit (Quantized Models Only)

Only needed for AWQ/GPTQ Marlin quantized models. SGLang JIT-compiles
CUDA kernels at runtime via `tvm_ffi`, which requires `nvcc`.

Non-quantized models can skip this.

```bash
nvidia-smi | head -5                            # check driver CUDA version
apt update && apt install -y cuda-toolkit-12-4   # match your driver
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
nvcc --version                                   # verify
```

---

## Start the Server

Make sure your venv is activated first:
```bash
source /path/to/sglang/.venv/bin/activate
```

### 1. Start the KV cache server

**CAMA server (TCP + RDMA):**
```bash
cd cama-complete/cama-server
CGO_ENABLED=1 go build -o cama-server .           # RDMA (Linux, requires libibverbs-dev)
# CGO_ENABLED=0 go build -o cama-server .         # TCP-only (any platform)
./cama-server server --listen 0.0.0.0:18000
```

> No git required — `go build` works from any source directory.

**PrisKV server (alternative):**
```bash
priskv-server --port 6379
```

### 2. Launch SGLang with CAMA

```bash
SGLANG_CAMA_REMOTE_ADDR=127.0.0.1 \
SGLANG_CAMA_REMOTE_PORT=18001 \
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama \
    --hicache-write-policy write_through
```

> **Port 18001** is the default RDMA port. For TCP-only setups, use `SGLANG_CAMA_REMOTE_PORT=18000`. For PrisKV, use port 6379.

### 3. Verify cache hits

```bash
# First request (cache miss -- writes to storage)
curl -s http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "default", "prompt": "The meaning of life is", "max_tokens": 32}'

# Second identical request (cache hit -- faster TTFT)
curl -s http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "default", "prompt": "The meaning of life is", "max_tokens": 32}'
```

---

## Quick Reference

| Task | Command |
|------|---------|
| **Full automated setup** | `python deploy.py /path/to/sglang --setup` |
| Deploy module + patches | `python deploy.py /path/to/sglang` |
| Module only | `python deploy.py /path/to/sglang --module` |
| Patches only | `python deploy.py /path/to/sglang --patch` |
| Check what differs | `python deploy.py /path/to/sglang --diff` |
| Deploy + zip archive | `python deploy.py /path/to/sglang --zip` |

## How Everything Links Together

```
<sglang>/
  .venv/                          <-- isolated Python environment
    lib/python3.x/site-packages/
      sglang/                     <-- installed copy (snapshot of python/sglang/)
      torch/                      <-- installed by pip from PyPI
      sgl_kernel/                 <-- installed by pip from PyPI
      flashinfer/                 <-- installed by pip (from wheels/ or PyPI)
      cama_client/                <-- installed by pip from ../cama-client/

  python/                         <-- source tree (what pip install copies FROM)
    sglang/
      srt/
        environ.py                <-- patched: has SGLANG_CAMA_* env vars
        server_args.py            <-- patched: "cama" backend choice
        managers/
          cache_controller.py     <-- patched: cama zero-copy + ThreadPool
        mem_cache/
          storage/
            backend_factory.py    <-- patched: CamaStorage registration
            cama/
              __init__.py         <-- module: package marker
              cama_storage.py     <-- module: CamaStorage class
              preflight.py        <-- module: server connectivity check
              profiling.py        <-- module: Pyroscope + NVTX hooks

  wheels/                         <-- bundled flashinfer for offline install
    flashinfer_python-0.5.3-py3-none-any.whl
    flashinfer_cubin-0.5.3-py3-none-any.whl
```

`pip install "python/"` copies the source tree into `site-packages/`.
The installed copy is a **snapshot** -- editing files in the source tree
does NOT affect the installed version. To pick up changes, reinstall
with `pip install "python/" --force-reinstall --no-deps`.

---

## Troubleshooting

### `setuptools-scm was unable to detect version`

**Full error:**
```
LookupError: setuptools-scm was unable to detect version for /path/to/sglang
```

**Cause:** SGLang uses `setuptools-scm` to get the version from git tags.
This fails when the tree is a zip, a copy, a renamed directory, or a shallow
clone without tags. Older releases (v0.5.7 and earlier) lack a
`fallback_version` in `pyproject.toml`, making this fatal.

**Fix:**
```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.5.7 pip install "python/" \
    --find-links wheels/
```

Set the version to match the SGLang release. The exact value only affects
`sglang.__version__` -- it does not change runtime behavior.

**Note:** `deploy.py --setup` detects this automatically and sets the
environment variable for you.

### `sgl-kernel` or `flashinfer` import fails

```bash
pip install sgl-kernel
pip install flashinfer_python flashinfer_cubin \
    --find-links https://flashinfer.ai/whl/cu124/torch2.5/
```

### SGLang server hangs at startup

See the full troubleshooting guide in `README.md` under "Troubleshooting".
Common causes: missing `nvcc` for quantized models, stale JIT lock files,
tokenizer deadlocks.
