// Package wirescreen — ROADMAP for the "local model on the wire" proposer spine.
//
// This doc.go is the extension contract. The spine is the witnessed-lossy-proposer
// PATTERN plus the abi.SemanticScreen seam (internal/abi/semscreen.go); each rung below
// is a sibling proposer added without changing the spine. The full rationale and the
// when-it-makes-sense decision framework live in
// docs/notes/RESEARCH-local-model-on-the-wire-2026-06-23.md.
//
// THE PATTERN (every rung obeys it):
//
// A small LOCAL model is a LOSSY PROPOSER, never the load-bearing answer; it emits a
// routing bit / rank / digest, not a decision the system trusts. It is bounded by a
// WITNESS: the original bytes stay pinned in the CAS and a gated PageIn (after a witness
// Clear) restores them byte-exact, so a wrong proposal costs one demand-page fault, never
// a lost fact. It is strictly ADDITIVE and one-sided: a proposer may only make the system
// MORE careful (quarantine, demote, redact), never weaker than a deterministic floor. It
// is DEFAULT-INERT and gated (build tag or env), because the real model needs weights and
// a measured latency number before it can default on, so the pure-Go binary is unchanged
// until an operator opts in. The local-triage ENVELOPE on this box is the native CPU
// Q8/Q4_K path in internal/model (cmd/fakchat -gguf), 1-3B sweet spot; the compute HAL is
// f32-only (a dead end for quant) and the RX 7600 is slower than CPU at this size, so
// measure end-to-end latency before defaulting any rung on.
//
// RUNG 1 — SEMANTIC POISON SCREEN (this package; SHIPPED as the spine). Seam:
// abi.SemanticScreen, consulted by ctxmmu.MMU.Admit AFTER ScreenBytes; a hit routes
// through ctxmmu.quarantineResult and inherits the CAS-pin + PageIn witness. A registered
// Screener flags injection-shaped results the literal-marker regex floor misses. Reference
// impl: heuristicScreener (deterministic). NEXT: a model-backed Screener registered under
// "model" via cmd/fakchat's native CPU path, gated behind a build tag, with the
// end-to-end admit-latency measurement that lets it default on. Honest scope: on the
// flagship `fak guard -- claude` passthrough the byte-rewrite is dead (the model reads
// req.Raw verbatim); the live value is taint-gate hardening (a quarantine raises the IFC
// high-water mark adjudicateProposed reads). It removes bytes only on the non-passthrough
// re-marshal wire.
//
// RUNG 2 — USEFUL PAGE-OUT (digest; SHIPPED as the reference floor — issue #570). Seam:
// ScreenDigest in ScreenAdvice (wired in the interface) plus ctxmmu's oversize Transform
// branch (digestToPointer, mmu.go). Today an oversize-benign result pages out to an
// OPAQUE {_paged,ref,len} pointer; when a Digester (digester.go, selected by
// FAK_WIRE_SCREEN) authors a ~200-token digest, the stub carries the digest instead and
// the original is pinned in CAS under the held ledger so a witness Clear + PageIn
// restores it byte-exact. The reference heuristicDigester is the zero-model floor; the
// model-backed Digester is the gated follow-on (needs weights + a measured digest
// latency before default-on). It only reaches the wire on the non-passthrough re-marshal
// path; on the passthrough it is dead until #555 lands (see the outbound blocker below).
//
// RUNG 3 — MULTI-MODAL SCREENSHOT TRIAGE. Seam: the same ctxmmu Transform branch, but the
// body is a base64 image block. Reversible collapses: perceptual-hash dedup of an
// unchanged frame (ZERO model, buildable now), OCR/VLM collapse-to-text, crop-to-ROI.
// BLOCKER: no vision/OCR path exists (internal/model is text-only, no vision encoder), so
// only the phash arm is buildable on this stack today. Ship phash first; the vision arms
// wait on an encoder.
//
// RUNG 4 — MODEL-AUTHORED RELEVANCE FORECAST. Seam: ctxplan.Forecast.Intents
// (internal/ctxplan/forecast.go), a DIFFERENT call site (the context planner, not the
// MMU). A small model authors the predicted reference strings the next turns will touch;
// the planner keeps the right cold spans resident and demotes the rest, and a miss costs
// one demand-page fault. This is ctxplan #556. BLOCKER: needs the outbound transform seam
// below to affect the flagship wire.
//
// RUNG 5 — PRE-SEND PII/SECRET REDACTION (SHIPPED as the deterministic floor;
// issue #572). Seam: this leaf's Redactor proposer + Apply/Restore (redactor.go),
// the redaction peer of rung 1's Screener. A Redactor proposes [start,end) byte
// spans to redact; Apply replaces each with a "[REDACTED:<kind>]" placeholder and
// pins the UNREDACTED original in the shared CAS so an authorized Restore returns it
// byte-exact (the same pageOut + PinResolved witness the MMU's quarantine uses). The
// reference piiRedactor is a zero-model, high-precision regex + Luhn compliance floor
// (credit cards, SSNs, AWS/GitHub/Slack/Stripe/Google keys, emails, bearer tokens,
// PEM private keys). It is DEFAULT-INERT (FAK_WIRE_REDACT) and touches no ABI seam.
//
// Honest scope — this is a compliance floor, NOT a token saver. It is the floor
// READY to be wired, not yet on the live wire: the flagship `fak guard -- claude`
// Anthropic passthrough sends req.Raw VERBATIM, so the redaction cannot reach the
// model there until the cache-prefix-preserving req.Raw transform (#555,
// ctxplan-owned) lands; until then the redaction is reachable on the NON-passthrough
// re-marshal path (QuarantineOutboundMessages-style). The model-backed Redactor is
// the gated follow-on (needs weights + a measured span latency before default-on) —
// the same fence the ctxplan forecast AUTHOR (CLAIMS.md) shipped the deterministic
// seed under. This is the floor for the outbound surface, not a duplicate of ctxmmu's
// inbound ScreenBytes quarantine (which removes a whole secret-bearing RESULT).
//
// THE OUTBOUND BLOCKER (gates rungs 2/4/5 on the flagship route): on the
// `fak guard -- claude` Anthropic passthrough the upstream gets req.Raw VERBATIM (gateway
// messages.go, WithRawRequestBody, to preserve the cache_control prefix). The kernel's
// inbound rewrite targets req.Messages, which the passthrough never serializes, so any
// "shrink/rewrite the outbound prompt" rung changes nothing the model reads on the live
// route. Building a req.Raw transform that preserves the cache-prefix is the single seam
// that unblocks the digest (rung 2), the forecaster (rung 4/#556), and the redactor
// (rung 5/#572) on the flagship wire. Until it lands those rungs are non-passthrough-only
// (and rung 5's deterministic floor ships ready-to-wire). This is the same blocker the
// ctxwin program hit (ctxplan #555).
package wirescreen
