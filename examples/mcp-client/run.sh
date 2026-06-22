#!/usr/bin/env bash
# Drive fak's MCP surface with a stdlib-only third-party client.
#
# client.py spawns `fak serve --stdio` itself over subprocess pipes, does the
# initialize handshake, and calls each of the six tools — so this wrapper just
# locates a Python and runs it. (The client also speaks HTTP: start a server with
# `fak serve --addr 127.0.0.1:8080` and run `client.py --http http://127.0.0.1:8080/mcp`.)
#
# Usage:
#   examples/mcp-client/run.sh                 # spawns fak serve --stdio
#   examples/mcp-client/run.sh --fak ./fak     # use an explicit fak binary
#   examples/mcp-client/run.sh --http http://127.0.0.1:8080/mcp
#
# Needs only Python 3 (stdlib) and the fak binary (or a Go toolchain to build it).
set -euo pipefail

# Run from the repo root so the default policy path (examples/dev-agent-policy.json)
# and the fak binary resolve, regardless of where this script is invoked from.
cd "$(dirname "$0")/../.."

PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
  echo "python3 not found on PATH" >&2
  exit 1
fi

exec "$PY" examples/mcp-client/client.py "$@"
