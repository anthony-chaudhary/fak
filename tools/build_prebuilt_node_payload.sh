#!/usr/bin/env bash
# Build a no-Go/no-repo benchmark payload for a private benchmark node.
#
# Output:
#   tools/_registry/prebuilt-node/<stamp>/fak-node-payload-<stamp>.tgz
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
BASE="${FAK_PREBUILT_OUT:-$ROOT/tools/_registry/prebuilt-node/$STAMP}"
PAYLOAD_NAME="${FAK_PREBUILT_PAYLOAD_NAME:-fak-node-payload}"
PAYLOAD="$BASE/$PAYLOAD_NAME"
ARCHIVE="$BASE/$PAYLOAD_NAME-$STAMP.tgz"

mkdir -p "$PAYLOAD/bin" \
  "$PAYLOAD/fak/internal/model/.cache" \
  "$PAYLOAD/fak/experiments/agent-live"

if [[ ! -d "$ROOT/fak/internal/model/.cache/smollm2-135m" ]]; then
  echo "missing model cache: fak/internal/model/.cache/smollm2-135m" >&2
  exit 1
fi
if [[ ! -f "$ROOT/fak/experiments/agent-live/production-workload.json" ]]; then
  echo "missing workload: fak/experiments/agent-live/production-workload.json" >&2
  exit 1
fi

echo "[payload] building darwin/arm64 benchmark binaries ..."
for b in q8kernel modelprof modelbench batchbench fleetbench fleetserve; do
  GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go -C "$ROOT/fak" build -o "$PAYLOAD/bin/$b" "./cmd/$b"
  chmod +x "$PAYLOAD/bin/$b"
  echo "  built $b"
done

cp "$ROOT/tools/run_prebuilt_node_bench.sh" "$PAYLOAD/run_node_bench.sh"
chmod +x "$PAYLOAD/run_node_bench.sh"
cp "$ROOT/tools/run_prebuilt_turn_agent_sweep.sh" "$PAYLOAD/run_turn_agent_sweep.sh"
chmod +x "$PAYLOAD/run_turn_agent_sweep.sh"
cp "$ROOT/fak/experiments/agent-live/production-workload.json" \
  "$PAYLOAD/fak/experiments/agent-live/production-workload.json"
cp -R "$ROOT/fak/internal/model/.cache/smollm2-135m" \
  "$PAYLOAD/fak/internal/model/.cache/smollm2-135m"

git -C "$ROOT" rev-parse --short HEAD > "$PAYLOAD/BUILD-GIT.txt"
go version > "$PAYLOAD/BUILD-GO.txt"
cat > "$PAYLOAD/RUN-ON-NODE.txt" <<'EOF'
Run this on the private benchmark node:

  mkdir -p ~/fak-node
  cd ~/fak-node
  tar -xzf fak-node-payload-*.tgz
  cd fak-node-payload
  xattr -dr com.apple.quarantine . 2>/dev/null || true
  bash run_turn_agent_sweep.sh --host=remote-node --profile=long

The turn-agent script writes results/<host>/turn-agent-<profile>/ and makes a
<host>-turn-agent-<profile>-*.tgz result bundle. Set FAK_NODE_SEND_TO or pass
--send-to when Taildrop is available; otherwise return the archive manually.

Optional broader node baseline:

  bash run_node_bench.sh --host=remote-node
EOF

echo "[payload] archiving ..."
tar -czf "$ARCHIVE" -C "$BASE" "$PAYLOAD_NAME"
echo "$ARCHIVE"
