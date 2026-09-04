from pathlib import Path

# Deployed layout: VERSION sits next to __init__.py in storage/cama/
# Source layout:   VERSION sits one level above cama_module/
_here = Path(__file__).resolve().parent
_version_file = _here / "VERSION"
if not _version_file.exists():
    _version_file = _here.parent / "VERSION"
__version__ = _version_file.read_text().strip() if _version_file.exists() else "0.0.0"
