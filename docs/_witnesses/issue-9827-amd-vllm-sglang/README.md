---
title: "Issue #9859 — matched MI300X direct-engine operator packet"
---

# Matched AMD MI300X vLLM/SGLang operator packet

This directory contains a fail-closed operator packet for direct **vLLM 0.28.0** and
**SGLang 0.5.18** on one AMD Instinct MI300X. It authors the procedure; it does not
claim that a benchmark was run in this worktree. Runtime evidence belongs in the
operator-selected output directory and must not be committed without review.

## Immutable comparison contract

`mi300x-packet.json` is the machine-readable contract. Both engines use exactly:

- public, ungated `Qwen/Qwen2.5-0.5B-Instruct` at revision
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- one `gfx942` MI300X, tensor parallel 1, FP16, context length 2048;
- repository `go run ./cmd/loadgen` against `/v1/chat/completions`;
- concurrency 16, 64 requests, maximum 128 output tokens, the same literal prompt,
  and zero permitted request errors;
- fixed order: vLLM cold, vLLM warm, SGLang cold, SGLang warm.

The source pins recorded by the packet are vLLM tag `v0.28.0` commit
`2cf0a6915ce544dc493a0990f2ea38d81601128a` and SGLang tag `v0.5.18` commit
`71de97b264b04dcd514cf904003028aefe9775c8`. The runner additionally verifies the
installed engine versions. It pulls the named ROCm image tags, records each local
image ID and resolved repository digest, and launches the immutable local ID rather
than reusing the mutable tag.

## Safe validation (no accelerator or network)

From the repository root:

```bash
bash -n docs/_witnesses/issue-9827-amd-vllm-sglang/run-mi300x-baselines.sh
python3 -m json.tool docs/_witnesses/issue-9827-amd-vllm-sglang/mi300x-packet.json >/dev/null
python3 -m json.tool docs/_witnesses/issue-9827-amd-vllm-sglang/receipt.json >/dev/null
bash docs/_witnesses/issue-9827-amd-vllm-sglang/run-mi300x-baselines.sh --dry-run
git diff --check -- docs/_witnesses/issue-9827-amd-vllm-sglang/README.md \
  docs/_witnesses/issue-9827-amd-vllm-sglang/run-mi300x-baselines.sh \
  docs/_witnesses/issue-9827-amd-vllm-sglang/mi300x-packet.json \
  docs/_witnesses/issue-9827-amd-vllm-sglang/receipt.json
```

`--dry-run` is intentionally declarative: it does not inspect a GPU, invoke Docker,
contact a network, download a model, or create output files.

## Operator admission and execution

Run only on an operator-approved Linux host with Docker, Go, ROCm tools, sufficient
local storage, and exclusive access to an MI300X. From the repository root:

```bash
MI300X_PACKET_OUT="$PWD/mi300x-results" \
  bash docs/_witnesses/issue-9827-amd-vllm-sglang/run-mi300x-baselines.sh
```

The runner stops on the first violated condition. Host admission requires character
device `/dev/kfd`, `/dev/dri`, host ROCm >= 6.3, successful `rocminfo` and `amd-smi`, `gfx942`, and an
MI300X product name. Container admission independently requires `/dev/kfd`,
`/dev/dri`, successful `rocminfo`, and `gfx942`. `HIP_VISIBLE_DEVICES=0` and an empty
`CUDA_VISIBLE_DEVICES` constrain each server to the admitted AMD device.

For each engine the runner captures server output, readiness errors, in-container
ROCm output, one-second `amd-smi metric` telemetry, and cold/warm loadgen JSON plus
stderr. Global evidence includes host admission, image references with IDs and
RepoDigests, exact engine versions, stop results, and `errors.log`.

### Explicit no-fallback gate

A row is invalid unless all checks pass. The server log must contain the exact model
revision. The runner rejects fallback language, CPU fallback, CUDA/NVIDIA markers,
missing GPU markers, wrong engine versions, missing image digests, failed container
ROCm admission, readiness failure, server exit, and any loadgen error. Do not delete
errors, edit logs, retry only a favorable phase, reorder phases, substitute an image
or model, or report partial output as a matched result.

## Cloud provisioning and teardown (documentation only)

This repository does **not** provision or destroy cloud resources. Before using a
provider, a human operator must explicitly confirm all of the following in their own
approved control plane:

1. the public MI300X offering and region are permitted by organizational policy;
2. a hard spend cap and expected maximum runtime are set and accepted;
3. provisioning is confirmed immediately before creation;
4. destructive teardown is separately confirmed after evidence has been copied and
   reviewed; and
5. billing shows that all resources, disks, snapshots, IPs, and reservations are
   gone or intentionally retained under the cap.

Never place credentials, tokens, account identifiers, private hostnames, IP
addresses, tenancy/project coordinates, SSH material, or provider console links in
this packet, shell history, logs, commits, or issue comments. The script contains no
cloud API calls and no teardown commands.

## Review checklist

- Confirm the four loadgen JSON files have the exact packet geometry and zero errors.
- Confirm cold precedes warm for each engine and vLLM precedes SGLang.
- Confirm both server logs identify the exact model revision and contain no rejected
  fallback marker.
- Confirm telemetry spans both phases and host/container admission says MI300X,
  `gfx942`, and ROCm >= 6.3.
- Confirm `receipt.json` matches the recorded environment and execution state.
- Confirm `images.jsonl` has a nonempty ID and RepoDigest for each image and the
  version files say exactly 0.28.0 and 0.5.18.
- Preserve all errors. Treat missing, truncated, edited, or mixed-run evidence as a
  failed packet, not as a performance result.
