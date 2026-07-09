---
title: "Backend Conformance Profile (BCK)"
description: "A vendor-neutral profile for certifying a compute backend against the fak correctness contract: the conformance ladder (op-exact → forward-cosine → caps-honored → collective-identity), the `fak backend conformance` verb, the signed conformance artifact, and the `fak-certified backend` mark — with fak as one reference implementation."
---

# Backend Conformance Profile (BCK)

**Status:** draft profile, 2026-07-03. Generation stream `gen/next` — the
contract and its harness are shipped; the runnable vendor-facing verb, the
fixtures, and the badge renderer are named here but **not yet landed** (see the
[honest fence](#honest-fence)).

**Goal:** make backend correctness *standard-shaped*: any neo-silicon vendor —
GPU, NPU, XPU, dataflow chip, host-CPU extension — can point one command at their
`compute.Backend` implementation, get a deterministic pass/fail on a host with
their device, and on pass carry a **`fak-certified backend`** mark, without
reading fak's internals. This is the correctness-side companion to the
[Agent Tool Governance Gateway profile](agent-tool-governance-gateway.md): fak is
one reference implementation of the profile, not the profile itself.

## Goals

- Define the minimum evidence a `compute.Backend` must produce to be called
  correct against the CPU reference floor (`cpu-ref`).
- Make the pass/fail deterministic and re-derivable: same backend, same host,
  same verdict — no self-report, no author-only witness (the
  [support-maturity honesty fence](support-maturity-honesty-fence.md) rule).
- Let a vendor advertise *only what they honor*: a `Caps{}` bit is a promise the
  harness checks, not a marketing label.
- Keep the profile implementable by a vendor who forks nothing — the contract is
  the small [`compute.Backend`](https://github.com/anthony-chaudhary/fak/blob/main/internal/compute/compute.go)
  interface, not fak's forward loop.

## Non-goals

- This does not standardize a kernel API, a device ABI, or a tensor layout. The
  vendor owns kernels; the profile owns the *correctness evidence*.
- This does not require adopting fak. `cpu-ref` is one reference implementation
  of the `Reference` class; a vendor may implement the profile against any
  reference that satisfies the same rungs.
- This is not a performance benchmark. Conformance grades *correctness and
  honesty*, never speed. Perf claims live under
  [net-true-value](net-true-value.md) and [observer-effect](observer-effect.md).
- A `PASS` is not a support-rung promotion by itself — it is one non-author
  witness the [support-maturity matrix](support-maturity-honesty-fence.md) may
  read to retire a cell honestly.

## Actors

| Actor | Role |
|---|---|
| Vendor / backend author | Implements `compute.Backend` and advertises a `Caps{}`. Runs the verb. |
| Conformance harness | Registers the backend, runs the ladder against `cpu-ref`, emits the verdict + artifact. fak is one implementation. |
| Reference backend | The `Reference`-class floor (`cpu-ref`) every rung compares against. |
| Certification consumer | Reads the conformance artifact and the `fak-certified` mark — a fleet scheduler, a support matrix, a procurement reviewer. |

## The contract already enforced

Nothing in the numeric contract is new. The profile only *externalizes* it:

- [`internal/compute/compute.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/compute/compute.go) —
  `Backend.Class() CorrectnessClass` is `Reference` or `Approx`, and
  `RequireReference` mechanically scopes the exact rungs so only a `Reference`
  backend is held to bit identity.
- The parity discipline is shipped and witnessed: argmax **EXACT**, every other
  tensor **cosine ≥ threshold** plus a small `max|Δ|` bound (`cuda_test.go`,
  `vulkan_test.go`, the C-series `*-NOTES.md`).
- `cmd/modelbench -backend <name> -require-non-reference` already fails closed
  (`exit 2`) when the selected backend is `Reference` — the "did the host
  silently fall back to CPU?" gate (`cmd/modelbench/main.go`).

What is missing is the *outward packaging*: a single verb that runs the whole
ladder and emits a portable artifact. This profile is that verb's contract.

## The conformance ladder

A backend is certified against ordered rungs. A rung only runs when the one below
it passes; the verdict names the highest rung reached.

| Rung | Name | What it checks | Applies to |
|---|---|---|---|
| **R1** | op-exact / op-cosine | Per-op parity vs `cpu-ref` on fixed inputs. `Argmax` is EXACT for every class; the rest are `max|Δ|=0` for a `Reference` backend, else `cosine ≥ threshold` + bounded `max|Δ|`. | all backends |
| **R2** | forward-cosine | A full model forward pass vs `cpu-ref` on a fixed prompt. Final-logits `Argmax` EXACT; hidden-state cosine within bound. | all backends |
| **R3** | caps-honored | Every `Caps{}` bit the backend advertises is actually honored (below). A bit set but not honored is a **lie** and fails closed. | all backends |
| **R4** | collective-identity | `AllReduceSum` / `AllGather` / `ReduceScatter` are numerically identity vs the single-device reference reduction. | only if `Caps.Collective` |

### R3 — the caps-honored rung in detail

`Caps{}` is the honesty surface. The harness holds each advertised bit to its
declared contract:

| Cap advertised | Harness assertion |
|---|---|
| `FusedAttn` | `Attention` lowers to one fused primitive — it must **not** be the reference decomposition. |
| `DeviceMemory` | `Host(t)` returns `(nil, false)` for a resident tensor — tensors are genuinely not host-addressable. |
| `UploadDtype` | `Upload(t, as)` actually narrows to `as` — a device that ignores `as` may not advertise this. |
| `CapacityProbe` / `HostCapacityProbe` | `DeviceCapacity` / `HostCapacity` return a real reported size, not the fail-open "unknown". |
| `Collective` | The backend implements `CollectiveBackend`; enables R4. |
| `Async`, `FusedFFN`, `GraphCompile` | The advertised fused/async/graph path is exercised, not silently delegated to the synchronous core. |

The asymmetry is the point: leaving a cap **false** is always safe (the core
type-asserts and falls back), so a backend is never penalized for honesty. Only a
cap set to **true** and not honored fails the rung. This mirrors the
[support-maturity fence](support-maturity-honesty-fence.md): a rung is an
envelope with a witness, never a promise.

## The `fak backend conformance` verb

The vendor entry point — one command, no fak-internal knowledge:

```bash
fak backend conformance --backend <name> [--out conformance.json]
```

- Registers `<name>` from the compute registry (the vendor's `init()` registered
  it, exactly as `cpu-ref` and the device backends do).
- Runs the ladder R1→R4 against `cpu-ref` on the current host and device.
- Writes a conformance artifact (below) and prints the verdict.

Verdict vocabulary (closed, machine-readable):

| Verdict | Meaning |
|---|---|
| `PASS` | Every applicable rung green. The backend earns the `fak-certified` mark for this `(backend, host, model-shape)` triple. |
| `FAIL` | A rung failed — the artifact names the rung and the first divergence (op, `max|Δ|`, or the dishonored cap). Fails **closed**. |
| `ABSTAIN` | A rung could not run on this host (no device present, driver missing). Not a pass; the artifact names the missing capability. |

The CPU-reference path is `PASS` **by construction** — `cpu-ref` *is* the
reference, so its op-exact and forward-cosine deltas are `0`. A backend that lies
about a `Caps{}` bit is `FAIL` by construction. Those two invariants are the
profile's self-check.

## The conformance artifact

A `PASS`/`FAIL`/`ABSTAIN` emits one JSON object — the portable, re-derivable
evidence a certification consumer reads without trusting the vendor's word:

```json
{
  "profile": "backend-conformance/1",
  "backend": "acme-npu",
  "class": "Approx",
  "host": { "os": "linux", "arch": "arm64", "device_tier": "acme-v2" },
  "model_shape": "llama-3.2-1b",
  "verdict": "PASS",
  "rungs": {
    "op":         { "verdict": "PASS", "argmax_exact": true, "max_abs_delta": 3.1e-4, "cosine": 0.99998 },
    "forward":    { "verdict": "PASS", "argmax_exact": true, "cosine": 0.99991 },
    "caps":       { "verdict": "PASS", "advertised": ["FusedAttn","DeviceMemory"], "honored": ["FusedAttn","DeviceMemory"] },
    "collective": { "verdict": "ABSTAIN", "reason": "single device; Caps.Collective false" }
  },
  "reference": "cpu-ref",
  "witness": "git:<sha>"
}
```

The `witness` binds the artifact to the reference commit that defined the rungs,
so a stale artifact cannot masquerade as a fresh pass. The artifact carries **no
field the vendor can raise by narrating it** — every number is a measured delta
against `cpu-ref`.

## The `fak-certified backend` mark

On `PASS`, the backend earns the mark for that `(backend, host, model-shape)`
triple. The mark is **scoped and witnessed**, never a global badge:

- It names the triple it was earned on — a pass on `llama-3.2-1b / arm64` does
  not certify `mixtral / x86_64`.
- It is **non-author**: the verdict is derived by the harness from measured
  deltas, not asserted by the vendor (support-maturity Rule 1).
- It **can drop**: the [support-maturity matrix](support-maturity-honesty-fence.md)
  re-derives cells each run, so a regressed or stale artifact demotes the cell
  with no latch (Rule 2). A retired witness retires the mark.

Where it renders: the support-maturity matrix already carries a
`SUPPORTED / PROOF-PATH-ONLY / FENCED / UNDEFINED` rung per `model × backend`
cell. A `PASS` artifact is the witness that lets a cell retire from
`PROOF-PATH-ONLY` to `SUPPORTED` honestly, and a lost artifact drops it back.

## Conformance fixtures

Mirroring [`docs/standards/fixtures/`](fixtures/), the kit ships three reference
inputs, each with an expected verdict, so an implementer can self-check the
harness before pointing it at real silicon:

| Fixture | Shape | Expected verdict |
|---|---|---|
| passing backend | `cpu-ref` (or a byte-identical shim) | `PASS` — the reference passes by construction. |
| caps-liar | a shim that sets `Caps.DeviceMemory` but returns a host-addressable `Host(t)` | `FAIL` at rung R3 — advertised, not honored. |
| numeric-diverger | a shim whose `MatMul` adds bounded noise beyond the cosine threshold | `FAIL` at rung R1 — cosine below bound. |

These are the honest-fence equivalent of the governance profile's
`allow-read-only.json` / `deny-policy-block.json` pair: a pass, a lie, and a
divergence, each with the verdict the harness must return.

## Honest fence

`gen/next` requires naming what moves this forward, what would retire it, and one
assumption that could invalidate it.

- **Promotion evidence (what moves it toward `now`):** a landed
  `fak backend conformance` verb (`cmd/**`) that runs R1→R4 and emits the
  artifact above; the three fixtures under `docs/standards/fixtures/` with a test
  asserting each expected verdict; and **one real non-`cpu-ref` backend** (CUDA
  or Vulkan on device) producing a `PASS` artifact a third party can re-run.
  Until a vendor other than fak runs it on their silicon, this is a *path*, not a
  certification program.
- **Demotion / retirement evidence:** if the `compute.Backend` interface or the
  `Caps{}` set changes shape such that the rung table here no longer matches the
  harness, this profile is stale and must be re-derived or retired — it is a
  standard *over* the contract, not the contract. A drift-sentinel that fails
  when `compute.go`'s `Caps` fields diverge from the R3 table would keep this
  page honest.
- **Invalidating assumption:** the profile assumes the cosine + `max|Δ|` ladder
  is a sufficient correctness oracle for *any* future device class. A backend
  whose failure mode is *distributional* rather than *pointwise* (e.g. a
  low-precision path that is cosine-close per-op but drifts argmax over long
  generations) could pass every rung and still be wrong in production. If that
  failure class shows up, R2 needs a multi-step generation-identity rung, not
  just single-forward cosine.

## Status summary

| Piece | State |
|---|---|
| `compute.Backend` contract + `CorrectnessClass` | **shipped** — `internal/compute/compute.go`, harness-enforced. |
| Parity discipline (argmax exact, cosine + `max|Δ|`) | **shipped** — `cuda_test.go`, `vulkan_test.go`, C-series notes. |
| `-require-non-reference` fail-closed gate | **shipped** — `cmd/modelbench/main.go`. |
| This vendor-neutral profile | **shipped** (this page). |
| `fak backend conformance` verb | **not yet** — the promotion artifact for `#1684`. |
| Conformance fixtures + badge renderer | **not yet** — named above; land with the verb. |

## Related

- [Neo-silicon onboarding](https://github.com/anthony-chaudhary/fak/blob/main/docs/vendor/neo-silicon-onboarding.md) — the vendor path
  this profile is the certification step of.
- [Hardware portability](../explainers/hardware-portability.md) — the
  `-require-non-reference` gate and the reference floor.
- [Support-maturity honesty fence](support-maturity-honesty-fence.md) — the rule
  that keeps the `fak-certified` mark witnessed, not wished.
- [Agent Tool Governance Gateway profile](agent-tool-governance-gateway.md) — the
  governance-side sibling profile this mirrors.
