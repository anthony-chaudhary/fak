# TICKET-13435 — Incremental gated-FFN composition

Tracking: [fak#13435](https://github.com/anthony-chaudhary/fak/issues/13435)

The problem is three repeated gated feed-forward tails inside a large model package. The deliverable is one independently testable component, wired through the dense, resident-expert and streamed-expert adapters while preserving their projections, activation, bias, error and fused-device policies.

The contract, architecture, production consumers and validation ladder live in [the FFN guide](../../../internal/model/ffn/README.md). The GitHub issue owns acceptance and closure state; this ticket supplies a stable local entry point without duplicating a mutable checklist.

Scope: internal/model/ffn, internal/model/moe.go, internal/model/v41_forward.go, and one independent adapter parity test. No backend selection, session cache, decode activation, SDK or GPU kernel changes. Coordinate shared V4.1 source edits with #13325; physical performance stays with #13294.

Witnesses: nonempty component tests, TestFFNGatedAdaptersBitExact, existing dense/V4.1 tests, allocation and matched callback benchmarks, then owning-package, review, boundary, leak and provenance gates. Source closure and three-to-one repeated-tail removal are structural proxies; no developer-productivity or inference-speed ratio is claimed from them.
