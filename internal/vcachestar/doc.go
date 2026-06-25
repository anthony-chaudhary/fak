// Package vcachestar is the vCache M2 star-anchor decision layer.
//
// It is deliberately off the live provider path: callers hand it already-shaped
// prompt parts and provider telemetry, and it returns auditable decisions. It
// does not issue network calls, store payloads, or claim a provider cache hit
// before telemetry confirms one.
//
// The package covers issue #717's M2 rules:
//   - apply cachemeta.RecommendLayout before keying a warm candidate;
//   - key anchors by exact serialized prefix bytes, scoped by model, tokenizer,
//     tool set, breakpoint layout, ttl, and provider surface;
//   - model the shared anchor, not a tiny sibling unit, as the cache unit;
//   - use the first natural request as the warm write, with no dedicated warm;
//   - demote believed-warm entries on a zero cache_read and report divergence;
//   - book calls at uncached cost and rebate only telemetry-confirmed reads.
//
// Tier: mechanism (2) - see internal/architest. This package imports only
// cachemeta (tier 1) and the standard library. It is not registered into the
// kernel.
package vcachestar
