// Package pricebook is fak's neutral price-book mechanism for multi-tenant
// token metering: the schema of per-model price cards, a strict loader for
// the operator-supplied JSON file, the UNPRICED-never-$0 typing that keeps
// an unknown rate distinct from a zero rate, and a deterministic version
// stamp a usage record can point at.
//
// It is a PUBLIC package (github.com/anthony-chaudhary/fak/pkg/billing/pricebook)
// on purpose, beside the neutral usage record in pkg/billing. Placement
// follows Plan C ("off-the-shelf public spine"), billing-modularization
// lane, Phase 1 item 5 (companion private repository,
// docs/tickets/billing-modularization/PLAN-C-off-the-shelf-public-spine.md):
// price-book schema + operator-file loader + UNPRICED rule + version stamp.
//
// Like pkg/billing, this package is pure MECHANISM. It contains NO pricing
// and NO policy, and no default may encode a commercial choice:
//
//   - No rates, markups, or amounts in any default. Every rate in a book
//     comes from the operator file; examples and tests use obviously fake
//     values and generic model names.
//   - No settlement semantics. Valuate produces an analytical figure, not
//     a charge: what to do with an unknown cost is a decision of the
//     consuming authority, not of this package.
//   - The scope fence is token pricing plus the cached-token discount:
//     fresh input, cache read, cache write and output are the four billing
//     lines, each with its own separate rate. Other tariff shapes are out
//     of scope.
//
// The UNPRICED-never-$0 rule is structural, mirroring the closed
// classification lineage in pkg/billing: a missing or unknown rate is typed
// RateUnpriced and is never coerced to zero; a declared zero count costs a
// known zero even against an unpriced rate, while a nonzero count against an
// unpriced rate leaves the total cost unknown. "Unknown" is never treated as
// "known".
//
// A built book stamps itself: VersionStamp is the SHA-256 digest of the
// table's deterministic serialization, and it is the value a usage record's
// PriceBookVersion field points at.
//
// The package has no external dependencies and holds no global state.
package pricebook
