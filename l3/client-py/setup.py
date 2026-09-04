"""Build script for L3 Python client.

On Linux with pybind11 installed, builds the _l3_rdma C++ extension
wrapping libibverbs/librdmacm and the _l3_cxl C++ extension for
devdax mmap. On other platforms, installs as pure-Python TCP-only client.
"""

import re
import sys
from pathlib import Path
from setuptools import setup

ext_modules = []


def _read_version() -> str:
    """Read __version__ from _version.py without importing."""
    text = (Path(__file__).parent / "l3_client" / "_version.py").read_text()
    m = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not m:
        raise RuntimeError("cannot find __version__ in _version.py")
    return m.group(1)


if sys.platform == "linux":
    try:
        from pybind11.setup_helpers import Pybind11Extension

        version = _read_version()
        ext_modules = [
            Pybind11Extension(
                "l3_client._l3_rdma",
                ["l3_client/csrc/rdma_transport.cpp"],
                extra_link_args=["-lrdmacm", "-libverbs"],
                define_macros=[("L3_VERSION", f'"{version}"'), ("CAMA_VERSION", f'"{version}"')],
            ),
            Pybind11Extension(
                "l3_client._l3_cxl",
                ["l3_client/csrc/cxl_transport.cpp"],
                define_macros=[("L3_VERSION", f'"{version}"'), ("CAMA_VERSION", f'"{version}"')],
            ),
        ]
    except ImportError:
        # pybind11 not available — install as pure-Python TCP-only
        pass

setup(ext_modules=ext_modules)
