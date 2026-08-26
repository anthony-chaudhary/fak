---
title: "fak adjudication: in-process default-deny reference monitor"
description: "How fak folds adjudicators into Deny, Defer, or Transform verdicts with closed refusal reasons, runtime policy manifests, and a default-deny empty policy."
---

# Adjudication (the in-process DOS reference monitor)

[← Claims index](../../CLAIMS.md)


- [SHIPPED] Provable refusal ⇒ `Deny`, unprovable ⇒ `Defer` (mirrors dos-preflake `decide.go`); default-deny on empty policy. Witness: `TestFoldDefaultDenyEmptyPolicy` (unit 15).
- [SHIPPED] Structured refusal from a closed 17-reason vocabulary + a bounded-disclosure witness (SELF_MODIFY returns only the offending glob). Witness: adjudicator tests (units 19, 20). Prior art: DOS `dos_refuse_reasons`; SMT unsat-core.
- [SHIPPED] Deny-as-value: a refusal carries a derived disposition (RETRYABLE/WAIT/ESCALATE/TERMINAL) the loop consumes (unit 74). Prior art: eBPF verdict, deny-loopback design.
- [SHIPPED] Batch adjudication (set shape) equals serial, in one pass (unit 75). Prior art: `dos-plan-price` generalized; speculative-decoding inverted.
- [SHIPPED] Deployable capability floor: the policy is a declarative, version-tagged JSON **manifest** loaded at runtime (`--policy FILE`), not a compiled-in Go literal — so an adopter configures WHICH tools the agent may call by editing a reviewable file, never by forking the kernel. Every `deny` reason is validated against the closed 17-reason vocabulary; unknown fields/reasons/versions are a fatal load error (fail-loud, never silently more-permissive); `--dump`↔`--check` round-trips exactly. `fak policy --dump|--check` authors+validates it; `fak preflight --policy` is the per-call oracle. Witness: `internal/policy` tests (9, incl. `TestRoundTrip`, `TestLoadedPolicyIsLoadBearing`, `TestUnknownDenyReasonRejected`); see `POLICY.md`, `examples/policy.example.json`. This is the deployable form of the "permissions as the floor" thesis.
- [SHIPPED] Git-shape prefilter: a registered adjudicator rung (`internal/gitgate`, rank 35) refuses the argv-decidable git hazards in a shell command — force-push, `commit --amend`, `add -A`, `--no-verify`, `tag -f`, `rebase -i` — at the call boundary, the in-kernel dual of `tools/githooks/*`. It Defers on non-git calls and on the state-dependent laws a stateless prefilter cannot honestly decide (OFF_TRUNK, sweep-a-peer, MERGE_HEAD — see `docs/notes/RESEARCH-git-in-kernel-prefilters-2026-06-22.md`); an operator whose git policy differs opts out with `FAK_GITGATE=off`. Witness: `go test ./internal/gitgate` (`TestClassify`, `TestAdjudicate`); `fak preflight --tool Bash --args '{"command":"git push --force"}'` ⇒ `DENY/POLICY_BLOCK/by=gitgate`.
