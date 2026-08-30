#!/usr/bin/env bash
set -euo pipefail

readonly BASE_REVISION="8fbba932b8128700aef41dd52ab548664a919003"
readonly MIXER_REVISION="46fdd8a52fd70b3e29345cd311be3cc89443e8fc"
readonly BLOCK_REVISION="99ea660ae222dd6a75dd661c54778f470904f9e7"
readonly EXPECTED_FILE="Qwen3.8-27B-Q4_K_M.gguf"
readonly EXPECTED_BYTES="17106775008"
readonly EXPECTED_SHA256="7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
readonly MINIMUM_MEMORY_BYTES="$((64 * 1024 * 1024 * 1024))"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

fail() {
  printf 'HOLD: %s\n' "$1" >&2
  exit 2
}

: "${FAK_SANCTIONED_APPLE_NODE:?set FAK_SANCTIONED_APPLE_NODE=YES only on an explicitly sanctioned Apple node}"
: "${GGUF_PATH:?set GGUF_PATH to the exact artifact without publishing the path}"
: "${OUT_DIR:?set OUT_DIR to a private absolute output directory}"

[[ "$FAK_SANCTIONED_APPLE_NODE" == "YES" ]] || fail "explicit sanctioned-node attestation must equal YES"
[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] || fail "route requires darwin/arm64"
cpu_brand="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
[[ "$cpu_brand" == Apple* ]] || fail "route requires Apple silicon"
memory_bytes="$(sysctl -n hw.memsize 2>/dev/null || true)"
[[ "$memory_bytes" =~ ^[0-9]+$ ]] || fail "physical memory could not be read"
(( memory_bytes >= MINIMUM_MEMORY_BYTES )) || fail "Apple node has less than the required 64 GiB"

case "$GGUF_PATH" in
  /*) ;;
  *) fail "GGUF_PATH must be absolute" ;;
esac
[[ -f "$GGUF_PATH" ]] || fail "exact artifact is not readable"
[[ "$(basename "$GGUF_PATH")" == "$EXPECTED_FILE" ]] || fail "artifact filename does not match $EXPECTED_FILE"
actual_bytes="$(stat -f '%z' "$GGUF_PATH")"
[[ "$actual_bytes" == "$EXPECTED_BYTES" ]] || fail "artifact byte length does not match the frozen receipt"
if command -v shasum >/dev/null 2>&1; then
  actual_sha256="$(shasum -a 256 "$GGUF_PATH" | awk '{print $1}')"
elif command -v sha256sum >/dev/null 2>&1; then
  actual_sha256="$(sha256sum "$GGUF_PATH" | awk '{print $1}')"
else
  fail "shasum or sha256sum is required"
fi
[[ "$actual_sha256" == "$EXPECTED_SHA256" ]] || fail "artifact SHA-256 does not match the frozen receipt"

case "$OUT_DIR" in
  /*) ;;
  *) fail "OUT_DIR must be absolute" ;;
esac
mkdir -p "$OUT_DIR"
[[ -d "$OUT_DIR" && -w "$OUT_DIR" ]] || fail "OUT_DIR is not writable"

cd "$REPO_ROOT"
git merge-base --is-ancestor "$BASE_REVISION" HEAD || fail "repository does not descend from the frozen base"
git merge-base --is-ancestor "$MIXER_REVISION" HEAD || fail "issue 9486 mechanism commit is absent"
git merge-base --is-ancestor "$BLOCK_REVISION" HEAD || fail "issue 9488 mechanism commit is absent"
git diff --quiet --ignore-submodules -- || fail "tracked working tree must be clean for VCS-bound capture"

printf '%s\n' 'Running focused mechanism and receipt-validator tests.'
CGO_ENABLED=1 go test ./internal/model \
  -run '^(TestQwen35MetalDecodeMixerFourSeededStepsParityAndAccounting|TestQwen35MetalDecodeMixerIsolationDeclineAndPostSubmitFailure|TestQwen35MetalDecodeBlockFourStepsParityAccountingAndGreedyToken|TestQwen35MetalDecodeBlockIsolationDeclineAndPostSubmitFailure|TestQwen35DecodeHandoffGradedModesRequireSequenceAndValidateCounts)$' \
  -count=1
CGO_ENABLED=1 go test ./cmd/modelbench \
  -run '^(TestNativeProfileControlsRefuseBeforeRun|TestNativeProfileReceiptBindsAllEvidence|TestNativeProfileReadbackRecomputesCompanion|TestNativeProfileRefusesAnyPromisedMetalFallback|TestNativeProfileComparisonM3DecodeHandoffRequiresExactRoutes|TestNativeProfileComparisonSelectsDecodeEndToEnd|TestNativeProfileComparisonDecodeEndToEndRequiresContiguousCanonicalWall)$' \
  -count=1

CGO_ENABLED=1 go build -trimpath -buildvcs=true -o "$OUT_DIR/modelbench" ./cmd/modelbench

cat <<'COMMANDS'
CURRENT EXACT P=32/T=64 SIX-ARM COMMANDS (3 CONTROL, THEN 3 MIXER):
for i in 1 2 3; do
  env -i PATH="$PATH" HOME="$HOME" TMPDIR="${TMPDIR:-/tmp}" \
    FAK_METAL_STREAM_Q4K=1 FAK_Q4K=1 FAK_GGUF_MMAP=1 \
    FAK_INKERNEL_QWEN35_METAL_GDN_SEQUENCE=ON \
    "$OUT_DIR/modelbench" -gguf "$GGUF_PATH" -q4k -metal \
    -name qwen38:27b -decode-prompt=32 -decode-steps=64 \
    -native-performance-qwen35-decode-handoff=CONTROL \
    -native-performance-profile="$OUT_DIR/control-$i.json"
done
for i in 1 2 3; do
  env -i PATH="$PATH" HOME="$HOME" TMPDIR="${TMPDIR:-/tmp}" \
    FAK_METAL_STREAM_Q4K=1 FAK_Q4K=1 FAK_GGUF_MMAP=1 \
    FAK_INKERNEL_QWEN35_METAL_GDN_SEQUENCE=ON \
    "$OUT_DIR/modelbench" -gguf "$GGUF_PATH" -q4k -metal \
    -name qwen38:27b -decode-prompt=32 -decode-steps=64 \
    -native-performance-qwen35-decode-handoff=MIXER \
    -native-performance-profile="$OUT_DIR/mixer-$i.json"
done

CURRENT EXACT READBACK COMMANDS:
for p in "$OUT_DIR"/control-{1,2,3}.json "$OUT_DIR"/mixer-{1,2,3}.json; do
  "$OUT_DIR/modelbench" -native-performance-readback="$p"
done

CURRENT EXACT STEADY-DECODE SIX-ARM COMPARISON:
"$OUT_DIR/modelbench" \
  -native-performance-compare="$OUT_DIR/control-1.json,$OUT_DIR/control-2.json,$OUT_DIR/control-3.json,$OUT_DIR/mixer-1.json,$OUT_DIR/mixer-2.json,$OUT_DIR/mixer-3.json" \
  -native-performance-compare-phase=steady-decode \
  -native-performance-compare-axis=m3-decode-handoff

CURRENT EXACT END-TO-END SIX-ARM COMPARISON:
"$OUT_DIR/modelbench" \
  -native-performance-compare="$OUT_DIR/control-1.json,$OUT_DIR/control-2.json,$OUT_DIR/control-3.json,$OUT_DIR/mixer-1.json,$OUT_DIR/mixer-2.json,$OUT_DIR/mixer-3.json" \
  -native-performance-compare-phase=end-to-end \
  -native-performance-compare-axis=m3-decode-handoff
COMMANDS

printf '%s\n' \
  'HOLD: commands are printed, not accepted as a performance route: the current modelbench envelope is pinned to the 36 GiB M3 Pro geometry and cannot issue a qualifying >=64 GiB receipt.' \
  'HOLD: current receipt topology cannot prove periodic full-attention whole-token residency or exact multi-token cosine >=0.9999 and matching greedy-token parity.' \
  'HOLD: no performance, quality, unsupported-geometry, or default claim is permitted; inclusive setup/recovery/prefill/first-token/steady-decode/verification/teardown and matched 2.9 versus 0.4-1.3 tok/s baselines remain required.' >&2
exit 2
