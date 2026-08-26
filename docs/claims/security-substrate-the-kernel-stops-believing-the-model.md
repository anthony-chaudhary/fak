---
title: "fak security substrate: kernel-enforced trust boundaries"
description: "fak enforces provenance, information-flow, secret, and capability gates before model-proposed actions; the page states measured ceilings and caveats."
---

# Security substrate (the kernel stops believing the model)

[← Claims index](../../CLAIMS.md)


- [SHIPPED] Information-flow control: `Ref.Taint` is source-stamped and a tainted→sink flow is sink-gated at adjudication time (rank-30, pre-call). Witness: `go test ./internal/ifc`. Prior art: FIDES/CaMeL IFC; the de-obfuscating canonicalization leaf (`internal/canon`) is shared with the recall re-screen.
- [SHIPPED] Kernel-authored trust/provenance: a classifier takes authorship of trust away from the model, with a hardened sink classifier (3 red-team fixes). Witness: `go test ./internal/provenance`.
- [SHIPPED] plan-CFI: a plan control-flow-integrity adjudicator with a `RequireApproval` verdict; `internal/harvest` folds the verdict stream into a frozen `LabelRow` corpus (the syscall-model training target). Witness: `go test ./internal/plancfi`, `./internal/harvest`.
- [SHIPPED] Effect-verifying witness gate: an in-process `dos_verify` effect-verify backs a `require-witness` verdict that fails closed when unwitnessed — a claim must be corroborated, not asserted. Witness: `go test ./internal/witness`.
- [SHIPPED] Dynamic attack battery: `internal/agentdojo` is an ASR-gated AgentDojo-style red-team that replaces the static poison fixture; the compiled defender loop (red-team → adjudicate → harvest → keep/revert) has 3 of 4 arrows shipped — the RL red-team generator is a documented seam. Witness: `go test ./internal/agentdojo`; `examples/agentdojo-redteam/README.md`.
- [SHIPPED] `normgate` (rank-5 canonicalize-and-decode ResultAdmitter, in front of ctxmmu) lifts agent-evasion catch 0→20/24 and cuts private real-transcript false positives 14.3%→7.1% with 0 new FPs / 0 leaks; one blank-import to enable. Witness: `go test ./internal/normgate` (6); `cmd/ctxbench -chain`.
- [SHIPPED] Default dev-agent floor + the CICD pillars on the **real** decision path: `adjudicator.DevAgentPolicy()` denies the shared-history git mutations (push/merge/tag), bounds writes off the kernel/policy spine (a spine write is SELF_MODIFY→ESCALATE), and allows one witness-gated `ship_release`; a registered `shipgate` adjudicator (rank 40) lifts a ship call to `require-witness` so an **unwitnessed ship is refused** and a **git-corroborated ship is allowed**; `witness` gains a `clean:` (green-tree) claim. Deployable as `examples/dev-agent-policy.json` (round-trips through the manifest loader). Witness: `go test ./internal/shipgate ./internal/adjudicator ./internal/witness` — `TestDevAgentDefaultPath` drives the real defconfig chain (self-modify denied ESCALATE, unwitnessed ship refused, corroborated ship allowed, RequireApproval emitted) (issue #11).

Honest ceiling, surfaced not hidden: the *detector* these drivers feed is ~100% evadable on a SOTA evasion battery and FP-prone on private real-transcript corpora. Detection is **deliberately non-load-bearing** — the structural guarantee is the capability floor + containment, which never run the detector; improving detection is additive, not the moat.
