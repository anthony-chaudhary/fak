// Package toon is a general JSON<->TOON (Token-Oriented Object Notation) codec whose
// correctness spine is a lossless, type-preserving round-trip: Decode(Encode(v)) deep-
// equals v for every value in the supported domain.
//
// TOON is a compact tabular serialization: a uniform array of flat objects collapses to
// a single header declaring the row count and field names ONCE (`name[N]{f1,f2,...}:`)
// followed by one delimiter-joined line per row, so field names are not repeated per row
// the way JSON repeats them. On uniform-array shapes that is a large token saving; on
// nested or non-uniform shapes it is near-zero or negative, which is exactly why the fire
// decision (issue #3066's GATE) lives elsewhere and consumes TabularEligibility from here.
//
// # Relationship to internal/memview
//
// internal/memview/format.go already ships a correct, tested flat-subset TOON *encoder*
// over a memview.Surface (a pre-stringified title + fixed fields + uniform string rows).
// It has no decoder, so no round-trip witness, and it only speaks Surface. This package is
// the generalization: arbitrary JSON `any`, a decoder, and the round-trip witness. The two
// agree byte-for-byte on the flat-Surface case (asserted by a test in memview), and this
// package reuses memview's quoting discipline (quote a cell only when a bare rendering
// would be ambiguous). memview's encoder stays standalone as the documented flat-subset
// special case.
//
// # Supported domain (this PR)
//
// The round-trip guarantee holds for JSON values as produced by encoding/json — i.e.
// containers are map[string]any and []any, numbers are float64, plus string/bool/nil.
// Encode also accepts the Go integer/float kinds and json.Number for convenience, but
// Decode always yields encoding/json's native types (numbers as float64), so a caller who
// wants an exact round-trip should pass values in that domain.
//
// Shapes handled losslessly:
//
//   - Uniform array of flat objects -> tabular header + delimiter-joined rows (the win case).
//   - Object (flat or nested) -> `key: value` lines, nested objects as an indented block.
//   - Scalars: string, number, bool, null. A string that merely LOOKS typed ("123", "true")
//     round-trips back as the original STRING, never coerced — it is quoted on encode so the
//     decoder does not misread it as a scalar.
//   - Ragged / non-uniform arrays, scalar arrays, and arrays whose elements carry nested
//     objects/arrays -> a safe per-item list fallback (`name[N]:` + `- <json>` lines), never
//     a corrupt tabular header with a wrong field count.
//
// # Out of scope for this PR (follow-on)
//
// Deeply nested objects/arrays *inside a tabular cell* are deliberately NOT collapsed into
// the tabular form: an array element that is not a flat object demotes the whole array to
// the per-item list fallback (detected, not corrupted). A token-optimal encoding of scalar
// arrays and nested-in-cell objects, plus key-folding (collapsing single-key wrapper chains
// to dotted paths, spec v1.5), are follow-on work — see issue #3065's non-goals. Options
// intentionally omits KeyFolding here.
//
// Tier: foundation (stdlib-only, imports nothing internal, off the hot path).
package toon
