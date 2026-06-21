#!/usr/bin/env python3
"""Build a small Qwen3.6 node bring-up packet.

The packet is for Mac, Linux/NVIDIA, Windows/NVIDIA, and Windows/Vulkan test
beds that may not have this repo checked out. It contains qwen36_node_server.py
plus launch wrappers and a driver-side smoke command. Generated packet
directories live under tools/_registry/ by default, which is ignored by git.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any


SCHEMA = "fak.qwen36-node-packet.v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M"
DEFAULT_SERVE_PORT = 8131
AUTO_REPORT_TARGET = "auto"


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def validate_report_target(value: str) -> str:
    target = value.strip()
    if not target:
        return ""
    if target == AUTO_REPORT_TARGET:
        return target
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._@-")
    if any(ch not in allowed for ch in target):
        raise ValueError("report target may contain only letters, numbers, dot, underscore, at-sign, or hyphen")
    return target


def write_text(path: Path, body: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")
    if executable and os.name != "nt":
        path.chmod(path.stat().st_mode | 0o111)


def copy_node_server(root: Path, payload_dir: Path) -> None:
    src = root / "tools" / "qwen36_node_server.py"
    if not src.exists():
        raise FileNotFoundError(f"missing node helper: {src}")
    shutil.copy2(src, payload_dir / "qwen36_node_server.py")


def mac_launcher(model: str, port: int) -> str:
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'cd "$(dirname "$0")"',
        "python3 qwen36_node_server.py \\",
        "  --profile mac \\",
        "  --bind tailnet \\",
        f"  --port {port} \\",
        f"  --model {sh_quote(model)} \\",
        '  "$@"',
        "",
    ])


def linux_nvidia_launcher(model: str, port: int) -> str:
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'cd "$(dirname "$0")"',
        "python3 qwen36_node_server.py \\",
        "  --profile nvidia \\",
        "  --bind tailnet \\",
        f"  --port {port} \\",
        f"  --model {sh_quote(model)} \\",
        '  "$@"',
        "",
    ])


def nvidia_launcher(model: str, port: int) -> str:
    return windows_profile_launcher("nvidia", model, port)


def vulkan_launcher(model: str, port: int) -> str:
    return windows_profile_launcher("vulkan", model, port)


def windows_profile_launcher(profile: str, model: str, port: int) -> str:
    return "\n".join([
        "$ErrorActionPreference = 'Stop'",
        "Set-Location $PSScriptRoot",
        "python .\\qwen36_node_server.py `",
        f"  --profile {profile} `",
        "  --bind tailnet `",
        f"  --port {port} `",
        f"  --model {ps_quote(model)} `",
        "  @args",
        "",
    ])


def mac_start_launcher(report_target: str = "") -> str:
    quoted_target = sh_quote(report_target)
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'cd "$(dirname "$0")"',
        f"report_target={quoted_target}",
        "send_reports() {",
        '  if [ -n "$report_target" ]; then',
        '    bash SEND-REPORTS-MAC.sh "$report_target" || true',
        "  fi",
        "}",
        "mkdir -p qwen36-reports",
        'stamp="$(date -u +%Y%m%d-%H%M%S)"',
        'preflight_report="qwen36-reports/preflight-mac-${stamp}.json"',
        'server_log="qwen36-reports/server-mac-${stamp}.log"',
        'echo "Qwen3.6 Mac preflight"',
        "if ! bash RUN-MAC.sh --preflight >\"$preflight_report\"; then",
        '  cat "$preflight_report"',
        '  echo "Preflight failed; fix the reported prerequisite before launch." >&2',
        "  send_reports",
        "  exit 1",
        "fi",
        'cat "$preflight_report"',
        'echo "Preflight report: $preflight_report"',
        "send_reports",
        'echo "Starting Qwen3.6 Mac server"',
        'echo "Server log: $server_log"',
        "set +e",
        'bash RUN-MAC.sh 2>&1 | tee "$server_log"',
        'server_rc="${PIPESTATUS[0]}"',
        "set -e",
        "send_reports",
        'exit "$server_rc"',
        "",
    ])


def linux_nvidia_start_launcher(report_target: str = "") -> str:
    quoted_target = sh_quote(report_target)
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'cd "$(dirname "$0")"',
        f"report_target={quoted_target}",
        "send_reports() {",
        '  if [ -n "$report_target" ]; then',
        '    bash SEND-REPORTS-LINUX-NVIDIA.sh "$report_target" || true',
        "  fi",
        "}",
        "mkdir -p qwen36-reports",
        'stamp="$(date -u +%Y%m%d-%H%M%S)"',
        'preflight_report="qwen36-reports/preflight-linux-nvidia-${stamp}.json"',
        'server_log="qwen36-reports/server-linux-nvidia-${stamp}.log"',
        'echo "Qwen3.6 Linux/NVIDIA preflight"',
        "if ! bash RUN-LINUX-NVIDIA.sh --preflight >\"$preflight_report\"; then",
        '  cat "$preflight_report"',
        '  echo "Preflight failed; fix the reported prerequisite before launch." >&2',
        "  send_reports",
        "  exit 1",
        "fi",
        'cat "$preflight_report"',
        'echo "Preflight report: $preflight_report"',
        "send_reports",
        'echo "Starting Qwen3.6 Linux/NVIDIA server"',
        'echo "Server log: $server_log"',
        "set +e",
        'bash RUN-LINUX-NVIDIA.sh 2>&1 | tee "$server_log"',
        'server_rc="${PIPESTATUS[0]}"',
        "set -e",
        "send_reports",
        'exit "$server_rc"',
        "",
    ])


def mac_install_launcher() -> str:
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'cd "$(dirname "$0")"',
        "if ! command -v llama-server >/dev/null 2>&1; then",
        "  if ! command -v brew >/dev/null 2>&1; then",
        '    echo "Homebrew is required to install llama.cpp automatically." >&2',
        '    echo "Install Homebrew or install llama.cpp manually, then rerun START-MAC.command." >&2',
        "    exit 1",
        "  fi",
        '  echo "Installing llama.cpp with Homebrew"',
        "  brew install llama.cpp",
        "fi",
        "exec bash START-MAC.command",
        "",
    ])


def linux_nvidia_install_launcher() -> str:
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'cd "$(dirname "$0")"',
        "if ! command -v llama-server >/dev/null 2>&1; then",
        '  echo "llama-server is required on Linux/NVIDIA test benches." >&2',
        '  echo "Install a current llama.cpp build with CUDA support, put llama-server on PATH, then rerun START-LINUX-NVIDIA.sh." >&2',
        "  exit 1",
        "fi",
        "exec bash START-LINUX-NVIDIA.sh",
        "",
    ])


def mac_report_sender() -> str:
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'cd "$(dirname "$0")"',
        'target="${1:-}"',
        'if [ -z "$target" ]; then',
        '  echo "usage: bash SEND-REPORTS-MAC.sh <driver-tailnet-name>" >&2',
        "  exit 2",
        "fi",
        "if ! command -v tailscale >/dev/null 2>&1; then",
        '  echo "tailscale CLI is required to send reports." >&2',
        "  exit 1",
        "fi",
        "mkdir -p qwen36-reports",
        'stamp="$(date -u +%Y%m%d-%H%M%S)"',
        'archive="qwen36-node-reports-mac-${stamp}.zip"',
        "if command -v zip >/dev/null 2>&1; then",
        '  zip -qr "$archive" qwen36-reports',
        "else",
        '  python3 - "$archive" <<\'PY\'',
        "import pathlib, sys, zipfile",
        "out = pathlib.Path(sys.argv[1])",
        "with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:",
        "    for path in pathlib.Path('qwen36-reports').rglob('*'):",
        "        if path.is_file() and path != out:",
        "            zf.write(path, path.as_posix())",
        "PY",
        "fi",
        'tailscale file cp "$archive" "${target}:"',
        'echo "Sent $archive to $target"',
        "",
    ])


def linux_nvidia_report_sender() -> str:
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'cd "$(dirname "$0")"',
        'target="${1:-}"',
        'if [ -z "$target" ]; then',
        '  echo "usage: bash SEND-REPORTS-LINUX-NVIDIA.sh <driver-tailnet-name>" >&2',
        "  exit 2",
        "fi",
        "if ! command -v tailscale >/dev/null 2>&1; then",
        '  echo "tailscale CLI is required to send reports." >&2',
        "  exit 1",
        "fi",
        "mkdir -p qwen36-reports",
        'stamp="$(date -u +%Y%m%d-%H%M%S)"',
        'archive="qwen36-node-reports-linux-nvidia-${stamp}.zip"',
        "if command -v zip >/dev/null 2>&1; then",
        '  zip -qr "$archive" qwen36-reports',
        "else",
        "  python3 - \"$archive\" <<'PY'",
        "import pathlib, sys, zipfile",
        "out = pathlib.Path(sys.argv[1])",
        "with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:",
        "    for path in pathlib.Path('qwen36-reports').rglob('*'):",
        "        if path.is_file() and path != out:",
        "            zf.write(path, path.as_posix())",
        "PY",
        "fi",
        'tailscale file cp "$archive" "${target}:"',
        'echo "Sent $archive to $target"',
        "",
    ])


def nvidia_cmd_launcher(report_target: str = "") -> str:
    return windows_cmd_launcher("nvidia", report_target)


def vulkan_cmd_launcher(report_target: str = "") -> str:
    return windows_cmd_launcher("vulkan", report_target)


def windows_cmd_launcher(profile: str, report_target: str = "") -> str:
    target = validate_report_target(report_target)
    upper = profile.upper()
    return "\n".join([
        "@echo off",
        "setlocal",
        "cd /d %~dp0",
        f"set \"REPORT_TARGET={target}\"",
        "if not exist qwen36-reports mkdir qwen36-reports",
        "for /f %%i in ('powershell -NoProfile -Command \"Get-Date -AsUTC -Format yyyyMMdd-HHmmss\"') do set STAMP=%%i",
        f"set PREFLIGHT_REPORT=qwen36-reports\\preflight-{profile}-%STAMP%.json",
        f"set SERVER_LOG=qwen36-reports\\server-{profile}-%STAMP%.log",
        f"echo Qwen3.6 {upper} preflight",
        f"powershell -NoProfile -ExecutionPolicy Bypass -File \"%~dp0RUN-{upper}.ps1\" --preflight > \"%PREFLIGHT_REPORT%\"",
        "if errorlevel 1 (",
        "  type \"%PREFLIGHT_REPORT%\"",
        "  echo Preflight failed; fix the reported prerequisite before launch.",
        f"  if not \"%REPORT_TARGET%\"==\"\" powershell -NoProfile -ExecutionPolicy Bypass -File \"%~dp0SEND-REPORTS-{upper}.ps1\" -Target \"%REPORT_TARGET%\"",
        "  pause",
        "  exit /b 1",
        ")",
        "type \"%PREFLIGHT_REPORT%\"",
        "echo Preflight report: %PREFLIGHT_REPORT%",
        f"if not \"%REPORT_TARGET%\"==\"\" powershell -NoProfile -ExecutionPolicy Bypass -File \"%~dp0SEND-REPORTS-{upper}.ps1\" -Target \"%REPORT_TARGET%\"",
        f"echo Starting Qwen3.6 {upper} server",
        "echo Server log: %SERVER_LOG%",
        f"powershell -NoProfile -Command \"& '%~dp0RUN-{upper}.ps1' 2>&1 | Tee-Object -FilePath '%SERVER_LOG%'; exit $LASTEXITCODE\"",
        "set SERVER_RC=%ERRORLEVEL%",
        f"if not \"%REPORT_TARGET%\"==\"\" powershell -NoProfile -ExecutionPolicy Bypass -File \"%~dp0SEND-REPORTS-{upper}.ps1\" -Target \"%REPORT_TARGET%\"",
        "exit /b %SERVER_RC%",
        "",
    ])


def nvidia_report_sender() -> str:
    return windows_report_sender("nvidia")


def vulkan_report_sender() -> str:
    return windows_report_sender("vulkan")


def windows_report_sender(profile: str) -> str:
    return "\n".join([
        "param(",
        "  [Parameter(Mandatory=$true)][string]$Target",
        ")",
        "$ErrorActionPreference = 'Stop'",
        "Set-Location $PSScriptRoot",
        "New-Item -ItemType Directory -Force .\\qwen36-reports | Out-Null",
        "if (-not (Get-ChildItem .\\qwen36-reports -Force | Select-Object -First 1)) {",
        "  'no qwen36 reports were present when this bundle was created' | Set-Content .\\qwen36-reports\\EMPTY.txt",
        "}",
        "$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')",
        f"$archive = Join-Path $PSScriptRoot \"qwen36-node-reports-{profile}-$stamp.zip\"",
        "if (Test-Path $archive) { Remove-Item -Force $archive }",
        "Compress-Archive -Force -Path .\\qwen36-reports\\* -DestinationPath $archive",
        "tailscale file cp $archive \"${Target}:\"",
        "Write-Host \"Sent $archive to $Target\"",
        "",
    ])


def nvidia_install_launcher() -> str:
    return windows_install_launcher("nvidia")


def vulkan_install_launcher() -> str:
    return windows_install_launcher("vulkan")


def windows_install_launcher(profile: str) -> str:
    upper = profile.upper()
    start_cmd = f"START-{upper}.cmd"
    return "\n".join([
        "@echo off",
        "setlocal",
        "cd /d %~dp0",
        "where llama-server >nul 2>nul",
        "if errorlevel 1 where llama-server.exe >nul 2>nul",
        "if errorlevel 1 (",
        "  where winget >nul 2>nul",
        "  if errorlevel 1 (",
        "    echo winget is required to install llama.cpp automatically.",
        f"    echo Install llama.cpp manually, then rerun {start_cmd}.",
        "    pause",
        "    exit /b 1",
        "  )",
        "  echo Installing llama.cpp with winget",
        "  winget install llama.cpp --accept-source-agreements --accept-package-agreements",
        "  if errorlevel 1 (",
        f"    echo winget install failed; install llama.cpp manually, then rerun {start_cmd}.",
        "    pause",
        "    exit /b 1",
        "  )",
        ")",
        f"call \"%~dp0START-{upper}.cmd\"",
        "",
    ])


def mac_bootstrap_launcher() -> str:
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'here="$(cd "$(dirname "$0")" && pwd)"',
        'work="${HOME}/qwen36-node-packet"',
        'mkdir -p "$work"',
        "find_archive() {",
        '  for dir in "$here" "${HOME}/Downloads" "$work"; do',
        '    if [ -d "$dir" ]; then',
        '      found="$(ls -t "$dir"/qwen36-node-packet-*.zip 2>/dev/null | head -n 1 || true)"',
        '      if [ -n "$found" ]; then',
        '        printf "%s\\n" "$found"',
        "        return 0",
        "      fi",
        "    fi",
        "  done",
        "  return 1",
        "}",
        'archive="$(find_archive)" || {',
        '  echo "No qwen36-node-packet-*.zip found next to this launcher, in Downloads, or in $work." >&2',
        "  exit 1",
        "}",
        'cp "$archive" "$work/"',
        'cd "$work"',
        "rm -rf qwen36-node-packet",
        'unzip -o "$(basename "$archive")"',
        "cd qwen36-node-packet",
        "exec bash INSTALL-MAC.command",
        "",
    ])


def linux_nvidia_bootstrap_launcher() -> str:
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'here="$(cd "$(dirname "$0")" && pwd)"',
        'work="${HOME}/qwen36-node-packet"',
        'mkdir -p "$work"',
        "find_archive() {",
        '  for dir in "$here" "${HOME}/Downloads" "$work"; do',
        '    if [ -d "$dir" ]; then',
        '      found="$(ls -t "$dir"/qwen36-node-packet-*.zip 2>/dev/null | head -n 1 || true)"',
        '      if [ -n "$found" ]; then',
        '        printf "%s\\n" "$found"',
        "        return 0",
        "      fi",
        "    fi",
        "  done",
        "  return 1",
        "}",
        'archive="$(find_archive)" || {',
        '  echo "No qwen36-node-packet-*.zip found next to this launcher, in Downloads, or in $work." >&2',
        "  exit 1",
        "}",
        'cp "$archive" "$work/"',
        'cd "$work"',
        "rm -rf qwen36-node-packet",
        'unzip -o "$(basename "$archive")"',
        "cd qwen36-node-packet",
        "exec bash INSTALL-LINUX-NVIDIA.sh",
        "",
    ])


def nvidia_bootstrap_cmd() -> str:
    return windows_bootstrap_cmd("nvidia")


def vulkan_bootstrap_cmd() -> str:
    return windows_bootstrap_cmd("vulkan")


def windows_bootstrap_cmd(profile: str) -> str:
    upper = profile.upper()
    return "\n".join([
        "@echo off",
        "setlocal",
        "cd /d %~dp0",
        f"powershell -NoProfile -ExecutionPolicy Bypass -File \"%~dp0START-QWEN36-{upper}.ps1\"",
        "exit /b %ERRORLEVEL%",
        "",
    ])


def nvidia_bootstrap_ps1() -> str:
    return windows_bootstrap_ps1("nvidia")


def vulkan_bootstrap_ps1() -> str:
    return windows_bootstrap_ps1("vulkan")


def windows_bootstrap_ps1(profile: str) -> str:
    upper = profile.upper()
    return "\n".join([
        "$ErrorActionPreference = 'Stop'",
        "$work = Join-Path $env:USERPROFILE 'qwen36-node-packet'",
        "$search = @($PSScriptRoot, (Join-Path $env:USERPROFILE 'Downloads'), $work) | Where-Object { Test-Path $_ }",
        "$archive = Get-ChildItem -Path $search -Filter 'qwen36-node-packet-*.zip' -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1",
        "if (-not $archive) { throw \"No qwen36-node-packet-*.zip found next to this launcher, in Downloads, or in $work.\" }",
        "New-Item -ItemType Directory -Force $work | Out-Null",
        "$localArchive = Join-Path $work $archive.Name",
        "Copy-Item -Force $archive.FullName $localArchive",
        "$packetDir = Join-Path $work 'qwen36-node-packet'",
        "if (Test-Path $packetDir) { Remove-Item -Recurse -Force $packetDir }",
        "Expand-Archive -Force -Path $localArchive -DestinationPath $work",
        "Set-Location $packetDir",
        f"& .\\INSTALL-{upper}.cmd",
        "exit $LASTEXITCODE",
        "",
    ])


def driver_smoke(model: str, port: int) -> str:
    return "\n".join([
        "param(",
        "  [Parameter(Mandatory=$true)][string]$Node,",
        "  [string]$FleetDir = 'C:\\work\\fleet',",
        "  [string]$NodeName = 'qwen36-testbed'",
        ")",
        "$ErrorActionPreference = 'Stop'",
        "Set-Location $FleetDir",
        "python tools\\qwen36_surface_smoke.py `",
        "  --tailscale-node $Node `",
        f"  --serve-port {port} `",
        f"  --model {ps_quote(model)} `",
        "  --node-name $NodeName `",
        "  --gateway-chat",
        "",
    ])


def receive_instructions() -> dict[str, list[str]]:
    return {
        "mac": [
            "mkdir -p ~/qwen36-node-packet",
            "cd ~/qwen36-node-packet",
            "tailscale file get .",
            "unzip -o qwen36-node-packet-*.zip",
            "cd qwen36-node-packet",
            "bash RUN-MAC.sh --preflight",
            "bash INSTALL-MAC.command",
        ],
        "linux-nvidia": [
            "mkdir -p ~/qwen36-node-packet",
            "cd ~/qwen36-node-packet",
            "tailscale file get .",
            "unzip -o qwen36-node-packet-*.zip",
            "cd qwen36-node-packet",
            "bash RUN-LINUX-NVIDIA.sh --preflight",
            "bash INSTALL-LINUX-NVIDIA.sh",
        ],
        "nvidia": [
            "New-Item -ItemType Directory -Force $HOME\\qwen36-node-packet | Out-Null",
            "Set-Location $HOME\\qwen36-node-packet",
            "tailscale file get .",
            "Expand-Archive -Force .\\qwen36-node-packet-*.zip .",
            "Set-Location .\\qwen36-node-packet",
            ".\\RUN-NVIDIA.ps1 --preflight",
            ".\\INSTALL-NVIDIA.cmd",
        ],
        "vulkan": [
            "New-Item -ItemType Directory -Force $HOME\\qwen36-node-packet | Out-Null",
            "Set-Location $HOME\\qwen36-node-packet",
            "tailscale file get .",
            "Expand-Archive -Force .\\qwen36-node-packet-*.zip .",
            "Set-Location .\\qwen36-node-packet",
            ".\\RUN-VULKAN.ps1 --preflight",
            ".\\INSTALL-VULKAN.cmd",
        ],
    }


def readme(model: str, port: int, profiles: list[str], report_target: str = "") -> str:
    profile_list = ", ".join(profiles)
    auto_report = ""
    if report_target:
        auto_report = f"""
This packet was generated with driver report target `{report_target}`. The `START-*` and
`INSTALL-*` wrappers automatically Taildrop the latest preflight report before launching the
server, and send the server log again if the process exits.
"""
    return f"""# Qwen3.6 Node Packet

This packet starts a tailnet-only OpenAI-compatible `llama-server` for
`{model}` on Mac, Linux/NVIDIA, Windows/NVIDIA, or Windows/Vulkan test beds.

Included profiles: {profile_list}
{auto_report}

Prerequisites on the node:
- Python 3
- Tailscale connected
- `llama-server` on PATH
- NVIDIA profiles: `nvidia-smi` must report at least one GPU

Run on Mac:

```bash
mkdir -p ~/qwen36-node-packet
cd ~/qwen36-node-packet
tailscale file get .
unzip -o qwen36-node-packet-*.zip
cd qwen36-node-packet
bash RUN-MAC.sh --preflight
bash INSTALL-MAC.command
```

Run on Linux/NVIDIA or DGX-adjacent standalone benches:

```bash
mkdir -p ~/qwen36-node-packet
cd ~/qwen36-node-packet
tailscale file get .
unzip -o qwen36-node-packet-*.zip
cd qwen36-node-packet
bash RUN-LINUX-NVIDIA.sh --preflight
bash INSTALL-LINUX-NVIDIA.sh
```

Run on NVIDIA/Windows:

```powershell
New-Item -ItemType Directory -Force $HOME\\qwen36-node-packet | Out-Null
Set-Location $HOME\\qwen36-node-packet
tailscale file get .
Expand-Archive -Force .\\qwen36-node-packet-*.zip .
Set-Location .\\qwen36-node-packet
.\\RUN-NVIDIA.ps1 --preflight
.\\INSTALL-NVIDIA.cmd
```

Run on AMD/Vulkan Windows:

```powershell
New-Item -ItemType Directory -Force $HOME\\qwen36-node-packet | Out-Null
Set-Location $HOME\\qwen36-node-packet
tailscale file get .
Expand-Archive -Force .\\qwen36-node-packet-*.zip .
Set-Location .\\qwen36-node-packet
.\\RUN-VULKAN.ps1 --preflight
.\\INSTALL-VULKAN.cmd
```

The preflight command prints JSON and exits before launch. It must report
`"ok": true`; otherwise fix the listed prerequisite, usually Tailscale reachability
or `llama-server` missing from PATH. NVIDIA profiles also require `nvidia-smi`
to report a usable GPU, and the preflight JSON records that GPU identity. The
`START-*` wrappers rerun preflight and only start the server after it passes.
The `INSTALL-*` wrappers install llama.cpp with Homebrew on macOS or winget on
Windows when `llama-server` is missing, then hand off to the same preflight-first
start wrapper.

Wrapper reports and server logs are written under `qwen36-reports/` in the packet
directory. Keep that folder if a node still fails to serve; it records the last
preflight JSON and `llama-server` output.

To send the report folder back to a driver machine over Taildrop:

```bash
bash SEND-REPORTS-MAC.sh <driver-tailnet-name>
```

```bash
bash SEND-REPORTS-LINUX-NVIDIA.sh <driver-tailnet-name>
```

```powershell
.\\SEND-REPORTS-NVIDIA.ps1 -Target <driver-tailnet-name>
```

```powershell
.\\SEND-REPORTS-VULKAN.ps1 -Target <driver-tailnet-name>
```

The server binds to the node's Tailscale IP on port `{port}`. From the driver,
run:

```powershell
.\\SMOKE-FROM-DRIVER.ps1 -Node <tailscale-node-name> -NodeName qwen36-testbed
```

If `llama-server` is not installed, install llama.cpp first. The server command
is intentionally tailnet-only by default; do not bind this model endpoint to
`0.0.0.0` on a personal machine.
"""


def write_packet(
    root: Path,
    out_dir: Path,
    profiles: list[str],
    model: str,
    port: int,
    report_target: str = "",
) -> dict[str, Any]:
    report_target = resolve_report_target(report_target)
    payload_dir = out_dir / "qwen36-node-packet"
    if payload_dir.exists():
        shutil.rmtree(payload_dir)
    payload_dir.mkdir(parents=True, exist_ok=True)
    for stale in (
        "START-QWEN36-MAC.command",
        "START-QWEN36-LINUX-NVIDIA.sh",
        "START-QWEN36-NVIDIA.cmd",
        "START-QWEN36-NVIDIA.ps1",
        "START-QWEN36-VULKAN.cmd",
        "START-QWEN36-VULKAN.ps1",
    ):
        stale_path = out_dir / stale
        if stale_path.exists():
            stale_path.unlink()
    copy_node_server(root, payload_dir)

    files = ["qwen36_node_server.py"]
    if "mac" in profiles:
        write_text(payload_dir / "RUN-MAC.sh", mac_launcher(model, port), executable=True)
        write_text(payload_dir / "START-MAC.command", mac_start_launcher(report_target), executable=True)
        write_text(payload_dir / "INSTALL-MAC.command", mac_install_launcher(), executable=True)
        write_text(payload_dir / "SEND-REPORTS-MAC.sh", mac_report_sender(), executable=True)
        files.extend(["RUN-MAC.sh", "START-MAC.command", "INSTALL-MAC.command", "SEND-REPORTS-MAC.sh"])
    if "linux-nvidia" in profiles:
        write_text(payload_dir / "RUN-LINUX-NVIDIA.sh", linux_nvidia_launcher(model, port), executable=True)
        write_text(payload_dir / "START-LINUX-NVIDIA.sh", linux_nvidia_start_launcher(report_target), executable=True)
        write_text(payload_dir / "INSTALL-LINUX-NVIDIA.sh", linux_nvidia_install_launcher(), executable=True)
        write_text(payload_dir / "SEND-REPORTS-LINUX-NVIDIA.sh", linux_nvidia_report_sender(), executable=True)
        files.extend([
            "RUN-LINUX-NVIDIA.sh",
            "START-LINUX-NVIDIA.sh",
            "INSTALL-LINUX-NVIDIA.sh",
            "SEND-REPORTS-LINUX-NVIDIA.sh",
        ])
    if "nvidia" in profiles:
        write_text(payload_dir / "RUN-NVIDIA.ps1", nvidia_launcher(model, port))
        write_text(payload_dir / "START-NVIDIA.cmd", nvidia_cmd_launcher(report_target))
        write_text(payload_dir / "INSTALL-NVIDIA.cmd", nvidia_install_launcher())
        write_text(payload_dir / "SEND-REPORTS-NVIDIA.ps1", nvidia_report_sender())
        files.extend(["RUN-NVIDIA.ps1", "START-NVIDIA.cmd", "INSTALL-NVIDIA.cmd", "SEND-REPORTS-NVIDIA.ps1"])
    if "vulkan" in profiles:
        write_text(payload_dir / "RUN-VULKAN.ps1", vulkan_launcher(model, port))
        write_text(payload_dir / "START-VULKAN.cmd", vulkan_cmd_launcher(report_target))
        write_text(payload_dir / "INSTALL-VULKAN.cmd", vulkan_install_launcher())
        write_text(payload_dir / "SEND-REPORTS-VULKAN.ps1", vulkan_report_sender())
        files.extend(["RUN-VULKAN.ps1", "START-VULKAN.cmd", "INSTALL-VULKAN.cmd", "SEND-REPORTS-VULKAN.ps1"])
    bootstrap_files: list[str] = []
    if "mac" in profiles:
        write_text(out_dir / "START-QWEN36-MAC.command", mac_bootstrap_launcher(), executable=True)
        bootstrap_files.append("START-QWEN36-MAC.command")
    if "linux-nvidia" in profiles:
        write_text(out_dir / "START-QWEN36-LINUX-NVIDIA.sh", linux_nvidia_bootstrap_launcher(), executable=True)
        bootstrap_files.append("START-QWEN36-LINUX-NVIDIA.sh")
    if "nvidia" in profiles:
        write_text(out_dir / "START-QWEN36-NVIDIA.cmd", nvidia_bootstrap_cmd())
        write_text(out_dir / "START-QWEN36-NVIDIA.ps1", nvidia_bootstrap_ps1())
        bootstrap_files.extend(["START-QWEN36-NVIDIA.cmd", "START-QWEN36-NVIDIA.ps1"])
    if "vulkan" in profiles:
        write_text(out_dir / "START-QWEN36-VULKAN.cmd", vulkan_bootstrap_cmd())
        write_text(out_dir / "START-QWEN36-VULKAN.ps1", vulkan_bootstrap_ps1())
        bootstrap_files.extend(["START-QWEN36-VULKAN.cmd", "START-QWEN36-VULKAN.ps1"])
    write_text(payload_dir / "SMOKE-FROM-DRIVER.ps1", driver_smoke(model, port))
    write_text(payload_dir / "README.md", readme(model, port, profiles, report_target=report_target))
    files.extend(["SMOKE-FROM-DRIVER.ps1", "README.md"])
    files.append("manifest.json")

    manifest = {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": model,
        "serve_port": port,
        "profiles": profiles,
        "report_target": report_target,
        "files": files,
        "bootstrap_files": bootstrap_files,
        "node_command": {
            "mac": "bash RUN-MAC.sh" if "mac" in profiles else "",
            "linux-nvidia": "bash RUN-LINUX-NVIDIA.sh" if "linux-nvidia" in profiles else "",
            "nvidia": ".\\RUN-NVIDIA.ps1" if "nvidia" in profiles else "",
            "vulkan": ".\\RUN-VULKAN.ps1" if "vulkan" in profiles else "",
        },
        "preflight_command": {
            "mac": "bash RUN-MAC.sh --preflight" if "mac" in profiles else "",
            "linux-nvidia": "bash RUN-LINUX-NVIDIA.sh --preflight" if "linux-nvidia" in profiles else "",
            "nvidia": ".\\RUN-NVIDIA.ps1 --preflight" if "nvidia" in profiles else "",
            "vulkan": ".\\RUN-VULKAN.ps1 --preflight" if "vulkan" in profiles else "",
        },
        "start_command": {
            "mac": "bash START-MAC.command" if "mac" in profiles else "",
            "linux-nvidia": "bash START-LINUX-NVIDIA.sh" if "linux-nvidia" in profiles else "",
            "nvidia": ".\\START-NVIDIA.cmd" if "nvidia" in profiles else "",
            "vulkan": ".\\START-VULKAN.cmd" if "vulkan" in profiles else "",
        },
        "install_command": {
            "mac": "bash INSTALL-MAC.command" if "mac" in profiles else "",
            "linux-nvidia": "bash INSTALL-LINUX-NVIDIA.sh" if "linux-nvidia" in profiles else "",
            "nvidia": ".\\INSTALL-NVIDIA.cmd" if "nvidia" in profiles else "",
            "vulkan": ".\\INSTALL-VULKAN.cmd" if "vulkan" in profiles else "",
        },
        "report_command": {
            "mac": "bash SEND-REPORTS-MAC.sh <driver-tailnet-name>" if "mac" in profiles else "",
            "linux-nvidia": "bash SEND-REPORTS-LINUX-NVIDIA.sh <driver-tailnet-name>" if "linux-nvidia" in profiles else "",
            "nvidia": ".\\SEND-REPORTS-NVIDIA.ps1 -Target <driver-tailnet-name>" if "nvidia" in profiles else "",
            "vulkan": ".\\SEND-REPORTS-VULKAN.ps1 -Target <driver-tailnet-name>" if "vulkan" in profiles else "",
        },
        "bootstrap_command": {
            "mac": "bash START-QWEN36-MAC.command" if "mac" in profiles else "",
            "linux-nvidia": "bash START-QWEN36-LINUX-NVIDIA.sh" if "linux-nvidia" in profiles else "",
            "nvidia": ".\\START-QWEN36-NVIDIA.cmd" if "nvidia" in profiles else "",
            "vulkan": ".\\START-QWEN36-VULKAN.cmd" if "vulkan" in profiles else "",
        },
        "receive": receive_instructions(),
        "driver_command": ".\\SMOKE-FROM-DRIVER.ps1 -Node <tailscale-node-name> -NodeName qwen36-testbed",
    }
    write_text(payload_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    manifest["payload_dir"] = str(payload_dir)
    return manifest


def archive_packet(payload_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(payload_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(payload_dir.parent).as_posix())


def find_tailscale() -> str:
    exe = shutil.which("tailscale")
    if exe:
        return exe
    if os.name == "nt":
        candidate = r"C:\Program Files\Tailscale\tailscale.exe"
        if os.path.exists(candidate):
            return candidate
    return ""


def detect_report_target(timeout_s: float = 10.0) -> str:
    exe = find_tailscale()
    if not exe:
        return ""
    try:
        proc = subprocess.run(
            [exe, "status", "--json"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return ""
    self_node = data.get("Self") if isinstance(data, dict) else {}
    if not isinstance(self_node, dict):
        return ""
    for key in ("HostName", "DNSName"):
        raw = str(self_node.get(key) or "").strip().rstrip(".")
        if not raw:
            continue
        try:
            return validate_report_target(raw)
        except ValueError:
            continue
    return ""


def resolve_report_target(value: str) -> str:
    target = validate_report_target(value)
    if target != AUTO_REPORT_TARGET:
        return target
    detected = detect_report_target()
    if not detected:
        raise ValueError("could not auto-detect local Tailscale report target")
    return detected


def run_taildrop_command(command: list[str], timeout_s: float = 60.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(1.0, timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        detail = f"taildrop command timed out after {max(1.0, timeout_s):g}s"
        stderr = f"{stderr}\n{detail}".strip()
        return subprocess.CompletedProcess(command, 124, stdout=stdout, stderr=stderr)


def taildrop(
    archive: Path,
    target: str,
    dry_run: bool,
    retries: int = 1,
    retry_delay_s: float = 5.0,
    attempt_timeout_s: float = 60.0,
) -> dict[str, Any]:
    exe = find_tailscale()
    command = [exe or "tailscale", "file", "cp", str(archive), f"{target}:"]
    retries = max(1, retries)
    if dry_run:
        return {
            "target": target,
            "command": command,
            "sent": False,
            "dry_run": True,
            "attempt_count": 1,
            "attempts": [{"attempt": 1, "command": command, "sent": False, "dry_run": True}],
        }
    if not exe:
        return {"target": target, "command": command, "sent": False, "error": "tailscale CLI not found"}

    attempts: list[dict[str, Any]] = []
    last: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, retries + 1):
        proc = run_taildrop_command(command, timeout_s=attempt_timeout_s)
        last = proc
        attempts.append({
            "attempt": attempt,
            "command": command,
            "sent": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-1000:],
            "stderr": proc.stderr[-1000:],
        })
        if proc.returncode == 0:
            break
        if attempt < retries and retry_delay_s > 0:
            time.sleep(retry_delay_s)

    sent = bool(attempts and attempts[-1].get("sent"))
    result = {
        "target": target,
        "command": command,
        "sent": sent,
        "attempt_count": len(attempts),
        "attempts": attempts,
    }
    if last is not None:
        result.update({
            "exit_code": last.returncode,
            "stdout": last.stdout[-1000:],
            "stderr": last.stderr[-1000:],
        })
    return result


def taildrop_files(
    paths: list[Path],
    target: str,
    dry_run: bool,
    retries: int = 1,
    retry_delay_s: float = 5.0,
    attempt_timeout_s: float = 60.0,
) -> dict[str, Any]:
    sends: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        if sends and not (sends[-1].get("sent") or sends[-1].get("dry_run")):
            sends.extend({
                "target": target,
                "command": [],
                "sent": False,
                "skipped": True,
                "path": str(skipped_path),
                "reason": f"skipped after {paths[index - 1].name} failed",
            } for skipped_path in paths[index:])
            break
        sends.append(taildrop(
            path,
            target,
            dry_run,
            retries=retries,
            retry_delay_s=retry_delay_s,
            attempt_timeout_s=attempt_timeout_s,
        ))
    return {
        "target": target,
        "sent": bool(sends) and all(row.get("sent") for row in sends),
        "dry_run": dry_run,
        "file_count": len(paths),
        "files": sends,
    }


def taildrop_send_paths(archive: Path, out_dir: Path, manifest: dict[str, Any], include_bootstrap: bool) -> list[Path]:
    paths = [archive]
    if include_bootstrap:
        paths.extend(out_dir / str(name) for name in manifest.get("bootstrap_files", []))
    return paths


def parse_profiles(value: str) -> list[str]:
    if value == "all":
        return ["mac", "linux-nvidia", "nvidia", "vulkan"]
    if value == "both":
        return ["mac", "nvidia"]
    if value == "windows":
        return ["nvidia", "vulkan"]
    if value in {"dgx", "linux", "linux-nvidia"}:
        return ["linux-nvidia"]
    if value in {"mac", "nvidia", "vulkan"}:
        return [value]
    raise ValueError(f"unknown profile: {value}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a Qwen3.6 node bring-up packet")
    ap.add_argument("--profile", choices=["mac", "linux", "linux-nvidia", "dgx", "nvidia", "vulkan", "windows", "both", "all"], default="both")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--serve-port", type=int, default=DEFAULT_SERVE_PORT)
    ap.add_argument("--out-dir", default="", help="output directory; default tools/_registry/qwen36-node-packet/<stamp>")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--taildrop-target", action="append", default=[], help="optional Tailscale file target")
    ap.add_argument("--taildrop-retries", type=int, default=1, help="Taildrop attempts per target")
    ap.add_argument("--taildrop-retry-delay-s", type=float, default=5.0, help="seconds between Taildrop attempts")
    ap.add_argument("--taildrop-timeout-s", type=float, default=60.0, help="seconds before one Taildrop attempt is marked timed out")
    ap.add_argument("--taildrop-bootstrap", dest="taildrop_bootstrap", action="store_true", default=True, help="Taildrop the top-level START-QWEN36 launchers with the packet zip (default)")
    ap.add_argument("--no-taildrop-bootstrap", dest="taildrop_bootstrap", action="store_false", help="Taildrop only the packet zip")
    ap.add_argument("--report-target", default="", help="driver Taildrop target baked into node start wrappers; use 'auto' for this Tailscale node")
    ap.add_argument("--dry-run", action="store_true", help="build packet but do not Taildrop")
    args = ap.parse_args(argv)
    try:
        report_target = resolve_report_target(args.report_target)
    except ValueError as exc:
        ap.error(str(exc))

    stamp = utc_stamp()
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "tools" / "_registry" / "qwen36-node-packet" / stamp
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    profiles = parse_profiles(args.profile)
    manifest = write_packet(ROOT, out_dir, profiles, args.model, args.serve_port, report_target=report_target)

    archive = None
    if not args.no_zip:
        archive = out_dir / f"qwen36-node-packet-{stamp}.zip"
        archive_packet(Path(manifest["payload_dir"]), archive)
        manifest["archive"] = str(archive)
    sends = []
    if archive and args.taildrop_target:
        send_paths = taildrop_send_paths(archive, out_dir, manifest, include_bootstrap=args.taildrop_bootstrap)
        sends = [
            taildrop_files(
                send_paths,
                target,
                args.dry_run,
                retries=args.taildrop_retries,
                retry_delay_s=args.taildrop_retry_delay_s,
                attempt_timeout_s=args.taildrop_timeout_s,
            ) if args.taildrop_bootstrap else taildrop(
                archive,
                target,
                args.dry_run,
                retries=args.taildrop_retries,
                retry_delay_s=args.taildrop_retry_delay_s,
                attempt_timeout_s=args.taildrop_timeout_s,
            )
            for target in args.taildrop_target
        ]
    manifest["taildrop"] = sends
    print(json.dumps(manifest, indent=2))
    return 0 if all(row.get("sent") or row.get("dry_run") for row in sends) else (1 if sends else 0)


if __name__ == "__main__":
    raise SystemExit(main())
