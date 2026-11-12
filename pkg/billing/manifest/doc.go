// Package manifest is fak's operator-facing billing manifest: the schema of
// the single file an operator points a serving deployment at, plus a strict
// loader for it. One manifest declares the served models, the
// OpenAI-compatible upstreams that serve them (stock SGLang/vLLM or any
// other compatible endpoint), the neutral caps that apply, the path to the
// price-book file, and the state directory where the ledger, key store, and
// reports live.
//
// It is a PUBLIC package
// (github.com/anthony-chaudhary/fak/pkg/billing/manifest) on purpose, beside
// the neutral usage record in pkg/billing and the price book in
// pkg/billing/pricebook. Placement follows Plan C ("off-the-shelf public
// spine"), billing-modularization lane, Phase 3 item 10 (companion private
// repository,
// docs/tickets/billing-modularization/PLAN-C-off-the-shelf-public-spine.md):
// the billing manifest schema + operator-file loader.
//
// The manifest is written by the OPERATOR, not by code: it is the
// deployment's declared intent, in a form a human can read and diff. The
// consuming command (fak serve --billing <file>, wired separately) loads it
// and acts on it; this package only declares the closed v1 shape and checks
// it. Nothing here routes, prices, enforces, or persists.
//
// Like its siblings, this package is pure MECHANISM. It contains NO pricing
// and NO policy, and no default may encode a billing choice:
//
//   - No rates or amounts anywhere: prices live in the price-book file that
//     PricesPath references, loaded separately via pkg/billing/pricebook.
//     This package carries the path and nothing else.
//   - No enforcement: Limits declares caps; whether a cap is enforced, how,
//     and what happens at the boundary are decisions of the consuming
//     command. A zero cap is uncapped, and zero is the default.
//   - No routing: ModelRef declares which upstream serves which model under
//     which upstream model id. Nothing here selects, weighs, or fails over
//     between upstreams.
//
// Strict decoding is the wire-side arm of the closed schema: Load refuses
// unknown fields, trailing data, and unparseable bytes (ErrMalformedPayload),
// and every field that parses but violates the closed contract reports a
// specific sentinel matchable up the chain to ErrInvalidManifest.
//
// The package has no external dependencies and holds no global state.
package manifest
