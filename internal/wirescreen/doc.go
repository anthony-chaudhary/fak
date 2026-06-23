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
// RUNG 2 — USEFUL PAGE-OUT (digest). Seam: ScreenDigest in ScreenAdvice (already in the
// interface) plus ctxmmu's oversize Transform branch (pageToPointer, mmu.go). Today an
// oversize-benign result pages out to an OPAQUE {_paged,ref,len} pointer; a model authors
// a ~200-token digest to put in the stub instead, with the original retained in CAS.
// ctxmmu maps ScreenDigest onto its existing Transform. It only reaches the wire on the
// non-passthrough re-marshal path; on the passthrough it is dead (see the outbound blocker
// below).
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
// RUNG 5 — PRE-SEND PII/SECRET REDACTION. Seam: a new outbound req.Raw transform that does
// NOT exist yet. A model proposes spans to redact in place (replace with a placeholder)
// before bytes leave the box. A compliance floor, not a token saver.
//
// THE OUTBOUND BLOCKER (gates rungs 2/4/5 on the flagship route): on the
// `fak guard -- claude` Anthropic passthrough the upstream gets req.Raw VERBATIM (gateway
// messages.go, WithRawRequestBody, to preserve the cache_control prefix). The kernel's
// inbound rewrite targets req.Messages, which the passthrough never serializes, so any
// "shrink/rewrite the outbound prompt" rung changes nothing the model reads on the live
// route. Building a req.Raw transform that preserves the cache-prefix is the single seam
// that unblocks the compressor and the forecaster. This is the same blocker the ctxwin
// program hit (ctxplan #555).
package wirescreen
