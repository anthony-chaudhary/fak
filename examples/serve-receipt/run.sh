#!/usr/bin/env sh
# Regenerates the serving receipt for the Kimi K3.1 Pro NVFP4 facts file.
# Prints the markdown receipt to stdout; pass extra flags to override the layout.
set -eu
cd "$(dirname "$0")/../.."
go run ./cmd/fak serve-receipt plan \
  --facts examples/serve-receipt/kimi-k31-pro-nvfp4.facts.json \
  --gpu rtx-pro-6000-blackwell-96gb --tp 8 --pp 3 "$@"
