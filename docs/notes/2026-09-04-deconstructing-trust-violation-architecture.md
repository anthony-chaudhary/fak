# Deconstructing TRUST_VIOLATION: Zero-Trust Capability Boundaries

**Date**: 2026-09-04  
**Issue**: #11392  
**Status**: Shipped / Active Architecture Note  
**Centrality**: Core / Enabling (Tier 1-3 Agent Kernel Foundation)

---

## 1. Problem: Monolithic Conflation and Anthropomorphic Framing

Historically, the refusal token `TRUST_VIOLATION` functioned as a coarse, anthropomorphic catch-all across multiple orthogonal subsystems in the fak kernel. This created three severe architectural defects:

1. **Inverted Causality**:
   In a zero-trust, default-deny capability kernel, no unearned trust is ever extended. Calling an inbound untrusted tool result or an adversarial prompt injection (which the Context-MMU sanitizes or pages out) a "trust violation" inverts causality and blames the agent for external content. Similarly, calling an empirical contradiction in git/filesystem ground truth a "trust violation" or a "lie" injected moral drama into a deterministic verification oracle.
2. **Tail-Wagging Dispositions**:
   `kernel.Disposition()` historically hardcoded `TRUST_VIOLATION` to `ESCALATE`. Subsystems like Plan-CFI, residency gates, and A2A messaging were forced to emit `TRUST_VIOLATION` purely to borrow the loopback escalation disposition (`// Reason TRUST_VIOLATION so the deny-loopback disposition is ESCALATE`). This secondary effect wagged primary classification and disrupted autonomous agent loops (e.g., issues #10640, #11330).
3. **Monolithic Conflation**:
   Seven distinct subsystems were collapsed into a single token:
   - Inbound prompt-injection quarantine (Context-MMU / Normgate)
   - Information Flow Control (IFC) taint-sink floor
   - Engine residency boundaries
   - Scope ceilings (Agent, Fleet, Tenant)
   - Empirical witness refutation
   - Plan Control Flow Integrity (Plan-CFI)
   - Cryptographic organizational policy envelope verification

---

## 2. The Deconstruction: 4 Fine-Grained Physical Sub-Cases

Commit `42b75eb76` and follow-on work cleanly disaggregate `TRUST_VIOLATION` into four concrete physical boundaries in `internal/abi/reasons.go` and `dos.toml`:

### A. `TAINT_EGRESS`
- **Definition**: A tool call attempted external network egress or sensitive sink invocation carrying tainted session data, or a tool result exceeded the declared taint ceiling.
- **Owning Floor**: `internal/ifc/ifc.go`
- **Disposition**: `RETRYABLE`
- **Actionable Recovery**:
  - Sanitize or untaint data flow before dispatching to external network endpoints.
  - Use internal IPC channels (`send_input`, `multi_agent_v1.send_input`, `a2a_send`) for worker coordination within the workspace.
  - Supply an explicit, auditable `override_reason` / `justification` in metadata or tool arguments when authorized.

### B. `SCOPE_CROSSING`
- **Definition**: A tool call, message, or payload crossed its declared isolation, tenant, or compute engine residency boundary.
- **Owning Floor**: `internal/ifc/scope_ceiling.go`, `internal/engine/engine.go`
- **Disposition**: `ESCALATE` / Reroute
- **Actionable Recovery**:
  - Confine payload routing to its declared scope boundary (`ScopeAgent`, `ScopeFleet`, or `ScopeTenant`).
  - Route tenant-isolated compute requests only to authorized on-box or dedicated fleet compute nodes.

### C. `PROMPT_INJECTION`
- **Definition**: Inbound tool results or external content matched adversarial prompt injection or instruction override markers.
- **Owning Floor**: `internal/ctxmmu/mmu.go`, `internal/normgate/normgate.go`, `internal/wirescreen/wirescreen.go`
- **Disposition**: `RETRYABLE`
- **Actionable Recovery**:
  - Inspect untrusted content and strip jailbreak/override markers.
  - Page out poisoned context or read sanitized stubs via Context-MMU quarantine rather than raw instruction ingestion.

### D. `INTEGRITY_REFUTED`
- **Definition**: An external witness or deterministic verification check refuted the claimed state or artifact existence.
- **Owning Floor**: `internal/kernel/kernel.go`, `internal/witness/witness.go`
- **Disposition**: `ESCALATE`
- **Actionable Recovery**:
  - Verify claims against empirical ground-truth git history, test runs, or external reporters.
  - Replace self-reporting with checkable artifacts (`dos verify`, git ancestry, tracked paths).

---

## 3. Legacy Umbrella Role of `TRUST_VIOLATION`

To maintain absolute backward compatibility with external clients, existing `dos.toml` policies, and recorded decision journals:
- `TRUST_VIOLATION` is preserved in `internal/abi/reasons.go` and `dos.toml [reasons.TRUST_VIOLATION]`.
- `fak recover TRUST_VIOLATION` in `cmd/fak/recover.go` explicitly routes callers to the four physical sub-cases.
- Existing policy manifests that reference `TRUST_VIOLATION` continue to evaluate correctly, while new kernel producers emit fine-grained codes.

---

## 4. Elimination of Anthropomorphic "Lie" Framing

The kernel witnesses physical effects, not intent. Comments, error messages, and documentation across `internal/witness` and proof documents (`docs/proofs/witness.md`) have been updated from anthropomorphic moral language ("lying agent", "provable lie") to empirical ground-truth contradiction ("ungrounded claim", "refuted effect").

---

## 5. Verification Matrix

| Verification Target | Command | Status |
|---|---|---|
| ABI Golden Freeze | `go test ./internal/abi/...` | PASS |
| Recovery Catalog Coverage | `go test -run 'TestRecover.*' ./cmd/fak/` | PASS |
| Plan-CFI Conformance | `go test ./internal/plancfi/...` | PASS |
| Reason Vocabulary Emitters | `go test -run TestEveryDeclaredReasonHasAnEmitter ./internal/architest/` | PASS |
| IFC Sink & Taint Egress | `go test ./internal/ifc/... ./internal/policy/...` | PASS |
| Witness Soundness Proof | `docs/proofs/witness.md` | PASS |
