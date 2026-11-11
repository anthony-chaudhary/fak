---
title: "Router tool-guard defaults: raw pass-through by default, the guard is opt-in"
description: "fak serve is a raw inference pipe by default — no tool-call adjudication, no tool-result screening, zero interception processing. The guard (PII/secret masking, quarantine, deny rungs) is armed explicitly with --guard / FAK_GUARD=on or by the harness products."
---

# Router tool-guard defaults: raw pass-through by default, the guard is opt-in

[← Architecture](../architecture.md) · Related: [when to use fak guard for harnesses](when-to-use-fak-guard-for-harnesses.md) · Issue [#14327](https://github.com/anthony-chaudhary/fak/issues/14327)

## The default is NOTHING

`fak serve` is a **raw inference router**. Its default posture for the tool plane is
**off**: the router moves bytes between the client and the inference engine and
intercepts nothing.

Concretely, with the guard disarmed (the `fak serve` default):

- **Proposed tool calls** the model emits are forwarded to the client **unadjudicated** —
  no deny rungs, no grammar repair, no inline-serve (vDSO / reuse) probes, no kernel
  `Decide` fold. Zero interception processing: the seams are not even entered.
- **Inbound tool results** the client sends back are forwarded to the model
  **byte-identical** — no secret masking, no PII masking, no injection quarantine, no
  taint stamping, no admit-once ledger. The response carries no `fak` extension and no
  `[fak]` note.

This is the deliberate flip of the earlier default, where every served turn ran the
kernel result-admission chain and a credential-shaped span in a tool result was masked
in place with a note like:

```
[fak] masked 3 credential-shaped spans in a tool result (SECRET_REDACTED) — the rest of
the output is intact and in context. Warn-first default: ...
```

That interception is powerful — and it is exactly why it must be **armed by name**, not
silent. A router that rewrites tool output without being asked is a bad default: it
spends hot-path time, it changes what the model sees, and it surprises the operator.
The masking engine itself is unchanged and stays available; what changed is WHO turns
it on.

## Arming the guard

| Surface | Posture | How |
| --- | --- | --- |
| `fak serve` (raw router) | **off** (raw pass-through) | default |
| `fak serve --guard` | on | explicit flag |
| `FAK_GUARD=on` (+ `fak serve`) | on | env (flag wins when explicitly passed) |
| `fak guard` (harness product) | **on, pinned** | not env-disarmable |
| library callers (`gateway.Config{}`) | on | empty `ToolGuard` resolves ON |

`gateway.Config.ToolGuard` accepts `on|off` (plus the usual bool spellings: `1/true/yes/
enabled`, `0/false/no/disabled/raw`). An explicit unparsable value fails gateway boot
loud — a misconfigured security boundary never silently resolves. The resolved posture
is stated in the serve startup messages (`tool guard: off — raw pass-through` /
`tool guard: on — ...`).

`fak guard` pins `ToolGuard: "on"` in its own gateway config regardless of `FAK_GUARD`:
the harness product exists to enforce the floor, so an operator env cannot silently
disarm it.

## What the armed guard does

When armed, the served path runs the historical floor, unchanged:

1. **Proposed tool calls** (`adjudicateProposedServed`): the kernel `Decide` fold (policy
   floor, deny rungs, subagent depth, livelock fuse), grammar repair, the read-redirect,
   and the inline-serve probes (vDSO fast path, toolproc reuse) that answer a fresh
   read-only call locally instead of round-tripping the engine.
2. **Inbound tool results** (`admitInboundResults`): the kernel result-admission chain —
   the context-MMU quarantine (injection-shaped and secret-shaped results are paged out),
   normgate's normalize-and-rescan with warn-first masking (credential spans masked
   in place, `SECRET_REDACTED`; general-PII spans masked, `PII_REDACTED`), and the IFC
   source-stamp that raises the per-trace taint high-water mark the sink gate reads.
3. **The Responses restore continuation** (buffered `/v1/responses`): the kernel folds on
   the restore call and on the restored bytes are guard processing too — skipped raw
   (`ALLOW/RAW_PASSTHROUGH` recorded instead of a kernel verdict). The restore itself is
   fak-owned CAS serving and still runs under its own byte/page bounds in both postures.

The explicit verbs (`POST /v1/fak/admit`, the `fak_adjudicate`/`fak_syscall` MCP tools)
keep working in both postures: a caller that asks for a verdict is opting in per call.

## The demo: PII filtering end to end

`cmd/piidemo` is a self-contained HTML demo of the same gateway seams. It serves one
page; a click replays the SAME PII-bearing tool result through a real gateway server
twice — raw (default) and guarded — and renders both wire views side by side:
the raw arm returns the content byte-identical with no admissions and no note; the
guarded arm returns the masked content, the `PII_REDACTED`/`SECRET_REDACTED` verdicts,
and the one-line `[fak]` note.

```bash
go run ./cmd/piidemo -addr 127.0.0.1:8163   # open the printed URL
go run ./cmd/piidemo -selfcheck             # browserless: asserts both arms in-process
```

## Design notes

- **Why the library default stays ON.** `gateway.Config{}` with no `ToolGuard` resolves
  ON so `fak guard`, the demos, and every existing test keep the armed floor
  byte-for-byte. The product default that flipped is the ROUTER's (`fak serve`), via its
  flag/env resolution — the narrow, honest place for a default to change.
- **Zero overhead means zero calls.** Raw mode does not "run the guard with a fast
  bypass": the interception functions return before any digest, scan, resolver Put, or
  ledger probe. The only work is the branch itself.
- **Scope: the guard, not the token savers.** This switch governs tool-plane
  INTERCEPTION only. The token-saver stack (history compaction, result elision, prefix
  guard — the `fak serve` serving-efficiency defaults) is a separate axis with its own
  documented defaults and opt-outs; raw pass-through here does not turn those off.
  (Later the same day the context plane got its own flip:
  [serving-router context-plane defaults](serve-context-plane-defaults.md) — the savers
  now default OFF on `fak serve` too, each re-armed by name.)
- **Raw mode is raw.** The client harness owns tool safety in raw mode (it always had
  the option to route through `fak_adjudicate`/`fak_admit` explicitly). The router does
  not silently keep a partial floor under a posture named "off".

## Witness

```text
go test ./internal/gateway/ -run 'ToolGuard' -count=1
go test ./cmd/fak/ -run 'ToolGuard' -count=1
go run ./cmd/piidemo -selfcheck
```
