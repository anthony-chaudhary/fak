// Package billing provides fak's neutral per-request usage record for
// multi-tenant token metering: one versioned, JSON-serializable value type
// that a serving engine (the native fak engine or an OpenAI-compatible
// upstream such as SGLang/vLLM) can emit and a separate billing authority
// can consume, without either side importing the other's internals.
//
// It is a PUBLIC package (github.com/anthony-chaudhary/fak/pkg/billing) on
// purpose. Placement follows two design sources, cited by reference only:
//
//   - BOUNDARY.md in the companion private repository lists "neutral usage
//     receipts" as a public mechanism (keep-public column of the Gate 3 /
//     Gate 4 serving-placement rule); the money side of serving stays
//     private.
//   - Plan C ("off-the-shelf public spine"), billing-modularization lane,
//     Phase 1 item 3 (companion private repository,
//     docs/tickets/billing-modularization/PLAN-C-off-the-shelf-public-spine.md):
//     UsageRecord v1, the neutral per-request usage record.
//
// This package is pure MECHANISM. It contains NO pricing and NO policy, and
// no default may encode a billing choice:
//
//   - No rates, markups, or amounts. A record points at the price book in
//     force when it was produced (PriceBookVersion, a version stamp only)
//     and never inlines a cost.
//   - No admission, settlement, or balance semantics. Whether an attempt
//     was admitted, what it costs, and how unknown usage is treated are
//     decisions of the consuming authority, not of this package.
//   - No tenant economics. TenantID and KeyID are opaque registry
//     identifiers: this package neither mints nor interprets them.
//
// Structural privacy is enforced BY CONSTRUCTION, not by convention:
// UsageRecord has no field that can hold a prompt/response body or a
// credential, and DecodeUsageRecord decodes strictly (unknown fields are
// refused), so a payload that tries to carry a body or a key alongside a
// record cannot be decoded into one.
//
// The closed usage classification (recorded / missing / zero_known /
// not_applicable) keeps "unknown" distinct from "known zero": a consumer
// must never treat missing usage as known usage.
//
// The package has no external dependencies and holds no global state: a
// Seen dedup index is constructed by its owner and lives as long as that
// owner wants it.
package billing
