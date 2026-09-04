# Development

## File Structure

```
cama-client/
├── cama_client/
│   ├── __init__.py          # Transport auto-selection, PriskvClient alias
│   ├── client.py            # CamaClient — TCP transport (186 lines)
│   ├── rdma_client.py       # RDMAClient — RDMA transport (234 lines)
│   ├── protocol.py          # Wire protocol encoder/decoder (233 lines)
│   ├── sgl.py               # SGL scatter-gather list shim (31 lines)
│   └── csrc/
│       └── rdma_transport.cpp  # pybind11 RDMA extension (485 lines)
├── setup.py                 # Conditional C++ extension build
├── pyproject.toml           # Package metadata and deps
├── docs/                    # Documentation (you are here)
└── README.md                # Overview and links
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Building for Development

```bash
# TCP-only (fast, no compiler needed)
pip install -e .

# With RDMA extension (requires Linux + libibverbs + pybind11)
pip install -e ".[rdma]" --no-build-isolation

# Force rebuild of the extension
pip install -e ".[rdma]" --no-build-isolation --force-reinstall
```
