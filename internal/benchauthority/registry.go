package benchauthority

// registry is the literal source of truth for the primary benchmark NUMBERS — one
// typed Claim per row, the readable index Block() renders and Validate() cross-checks.
// Keep it stable-ordered (never renumber an ID; other docs deep-link the anchors).
//
// SEED — DELIBERATELY EMPTY. This declaration exists to make the package compile; the
// claims themselves are NOT yet transcribed. The authoritative numbers still live,
// hand-authored, in BENCHMARK-AUTHORITY.md (the "Quick Reference: Primary Numbers"
// table and the per-axis tables below it). Populating this slice — one Claim literal
// per committed number, honest Status, real on-disk Artifact path — plus landing the
// `cmd/authorityledger -write` generator and wiring the BEGIN/END markers into
// BENCHMARK-AUTHORITY.md is the additive-leaf work that finishes this leaf.
//
// Until that lands, treat the emptiness as LOUD, not silent: Block() renders an empty
// ledger, so do NOT wire Splice() into BENCHMARK-AUTHORITY.md yet — an empty splice
// would erase the hand-authored numbers. There is no marker in the doc today and no
// caller of Splice(), so nothing erases anything; keep it that way until the slice is
// populated.
var registry = []Claim{}
