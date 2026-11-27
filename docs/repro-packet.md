---
title: "fak Repro Packet — Reproduce the Allow/Deny Boundary Offline"
description: "A no-credential, offline reproduction of fak's tool allow/deny policy: validate a policy manifest, deny a dangerous action, and allow a permitted action."
---

# Reproduce the offline allow/deny boundary

**Primary audience:** an evaluator deciding whether fak's tool-call boundary is worth deeper integration testing.

## Result and scope

The canonical offline proof demonstrates two current behaviors in one deterministic fixture:

| Check | Expected result | Scoped meaning |
|---|---|---|
| Dangerous `refund_payment` proposal | `DENY (POLICY_BLOCK)` | A reviewable manifest blocks the named action before tool execution. |
| Benign `search_kb` proposal | `ALLOW` | The same policy preserves its useful read/search path. |

This is a **current-generation, maintained deterministic witness**. It covers the checked-in customer-support fixture on any supported Go build host. It does not establish live-model behavior, detector recall, production readiness, external endorsement, or fleet-scale performance. Those claims require their own authorities and evidence.

**Default:** use the offline source-checkout route below. It needs no credential, live model, network service, or accelerator.

**Next action:** from a clean checkout with Go 1.26+, validate the manifest, then compare the two preflight results with the table above.

```bash
go run ./cmd/fak policy --check examples/customer-support-readonly-policy.json
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb --args "{}"
```

## Witness 1: Policy Manifest Validates

Command:

```bash
go run ./cmd/fak policy --check examples/customer-support-readonly-policy.json
```

Key output:

```text
OK  examples/customer-support-readonly-policy.json  (manifest valid; every deny cites a closed-vocabulary reason)

posture            : fail_closed
allow (exact)      : 4 tool(s)
allow (prefix)     : read_, get_, search_, list_, lookup_, find_
deny (explicit)    : 6 tool(s)
                     delete_account -> POLICY_BLOCK
                     export_customer_data -> SECRET_EXFIL
                     refund_payment -> POLICY_BLOCK
                     rotate_credentials -> POLICY_BLOCK
                     send_customer_email -> POLICY_BLOCK
                     transfer_funds -> POLICY_BLOCK
```

What this proves: the starter customer-support manifest parses, is fail-closed,
and its dangerous actions cite closed-vocabulary refusal reasons.

## Witness 2: Dangerous Action Denied

Command:

```bash
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"
```

Output:

```text
verdict=DENY reason=POLICY_BLOCK by=monitor
fak: loaded capability floor from examples/customer-support-readonly-policy.json
```

What this proves: a destructive support action is denied before any tool
execution. This is the smallest useful demo for a security lead: edit a manifest,
run one command, see a closed reason code.

## Witness 3: Benign Call Still Allowed

Command:

```bash
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb --args "{}"
```

Output:

```text
verdict=ALLOW reason=NONE by=monitor
fak: loaded capability floor from examples/customer-support-readonly-policy.json
```

What this proves: the policy is not a blanket block. It preserves the useful
read/search path while denying dangerous writes.

## Inkling-2-Mini reproducer (CLI-only)

Reproduce Inkling-2-Mini locally with the `fak` CLI alone — no Go test binaries
and no lab-specific file paths. The `inkling2-mini` alias fetches and
SHA-256-verifies `Inkling-2-Mini-UD-Q3_K_XL.gguf` (~13.6 GB) into your local
model cache on first run.

### 1. One-shot check (`fak run`)

```sh
fak run inkling2-mini --max-tokens 256 "What is 2 + 2? Reply with the number only."
```

On an Apple Silicon Mac (24 GB+ unified memory), `fak run inkling2-mini` runs
on Metal directly. On an NVIDIA Blackwell host (RTX 50-series), install or
build with `sm_120f` so native NVFP4 W4A4 prefill is compiled in (see
[INSTALL.md](../INSTALL.md#nvidia-cuda-from-source)):

```sh
# Linux NVIDIA without Docker (auto-selects sm_120f on CC 12.0/12.1):
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh -s -- --variant cuda
```

### 2. Serve over OpenAI-compatible HTTP (`fak serve`)

Start the server against the verified alias (or replace `inkling2-mini` with
your local GGUF path):

```sh
# Apple Silicon Metal:
fak serve --gguf inkling2-mini --addr 127.0.0.1:8080

# NVIDIA CUDA (sm_120f build with NVFP4 W4A4 default prefill):
fak serve --engine inkernel --backend cuda --gguf inkling2-mini --addr 127.0.0.1:8080
```

To reproduce the RTX 5080 concurrent-prefill profile from a checkout root, load
[`examples/inkling2-cuda-prefill.env.example`](../examples/inkling2-cuda-prefill.env.example)
before starting `fak serve`:

```sh
(
  set -ae
  . ./examples/inkling2-cuda-prefill.env.example
  set +a
  fak serve --engine inkernel --backend cuda \
    --gguf inkling2-mini \
    --session-state off --session-registry off --addr 127.0.0.1:8080
)
```

### 3. Measure prompt and decode rates (`fak bench`)

With `fak serve` listening on `127.0.0.1:8080`, run the built-in HTTP benchmark
against it:

```sh
fak bench h2h run \
  --arm fak=http://127.0.0.1:8080/v1 \
  --model inkling2-mini \
  --cold 512 --decode 256 --turns 0 --reps 3 \
  --speculative server-default \
  --ledger off
```

To compare against a second local OpenAI-compatible server (for example
`llama-server` on port `8091` serving the same GGUF), add another `--arm`:

```sh
fak bench h2h run \
  --arm fak=http://127.0.0.1:8080/v1 \
  --arm llama=http://127.0.0.1:8091/v1 \
  --model inkling2-mini \
  --cold 512 --decode 256 --turns 0 --reps 3 \
  --speculative server-default \
  --ledger off
```

## Fixture requests

The packet is an evaluation fixture rather than a production benchmark. To propose a scrubbed or synthetic boundary failure, use the [agent-tool boundary fixture issue form](https://github.com/anthony-chaudhary/fak/blob/main/.github/ISSUE_TEMPLATE/agent-tool-boundary-fixture.yml). For a framework or host adapter, use the [adapter fixture issue form](https://github.com/anthony-chaudhary/fak/blob/main/.github/ISSUE_TEMPLATE/framework-adapter-fixture.yml).

## Non-Claims

- This is an offline deterministic harness, not a live-model benchmark.
- The detector remains heuristic; this packet demonstrates the boundary behavior
  for this fixture, not broad prompt-injection recall.
- The production-readiness gates in
  [`docs/production-readiness.md`](production-benchmark-methodology.md) still matter.
- No vendor, government, or standards-body endorsement is implied.
