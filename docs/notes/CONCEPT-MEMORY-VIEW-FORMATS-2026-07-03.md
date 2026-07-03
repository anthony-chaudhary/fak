# Memory-View Formats — a third, orthogonal axis (surface + ablate)

> Extends [`MEMORY-VIEW-CONTRACT-2026-06-26.md`](MEMORY-VIEW-CONTRACT-2026-06-26.md)
> (#904). That contract fixes two axes over a memory view: **Kind** (what was
> derived — snippet/summary/qa/fact) and the **taint gate** (whether it may enter
> context at all). This note adds a third, deliberately orthogonal axis: **Format**
> — how an already-admitted, already-rendered set of records is *serialized* for a
> consumer, and how that choice is *ablated* (measured, not assumed) rather than
> hardcoded to prose.

## The stance (one sentence)

Once the trust gate has already decided a view may render, *how it is spelled*
is a separate, swappable, measurable decision — markdown for a human-facing
loop-turn block, JSON for a lossless machine-parseable envelope, or a compact
tabular encoding (TOON-shaped) for token-constrained surfacing — and the choice
should be justified by a measured byte/token delta, not a hunch.

## Why this needed its own axis, not a Kind variant

`ViewKind` (memview.go) answers "what is this content" (a snippet vs a summary vs
a fact). Format answers "what bytes do we emit for a fixed set of already-typed,
already-admitted records." Conflating them would mean a `KindSummary` view
sometimes means "prose paragraph" and sometimes means "TOON row" depending on who
asked — which breaks every consumer that pattern-matches on Kind. Keeping them
separate means: change Format, and Kind/Source/Taint/Body are untouched; the
Format layer is a pure re-encoding of bytes that already cleared admission.

## What shipped

- **`internal/memview/format.go`** (tier 2, stdlib-only — confirmed by
  `internal/architest`'s layering gate): a `Format` type, an open `Register`
  seam, `KnownFormats()`, a generic `Surface`/`Row` shape (the same
  translate-in pattern `timeline.go` uses for `ProvenanceEvent` — a tier-3/4
  caller builds ITS OWN typed result into a `Surface` once, and every
  registered format is then free), `Encode(format, surface)`, and
  `SweepFormats(surface, formats)` — the ablation call: render the SAME
  content under every named (or every registered) format and report each
  one's byte/estimated-token cost.
- Three shipped formats: `markdown` (GFM table, the human-prose default),
  `json` (canonical order-preserving array-of-objects), `toon` (a fak-local
  flat subset of [TOON](https://github.com/toon-format/spec) — one header
  line declaring row count + field names once, then indented comma-joined
  rows; quoting only when a cell is ambiguous). All three are deterministic
  (`TestSweepFormatsIsDeterministic`) and total (never panic).
- **`fak memory recall`** (`cmd/fak/memory_recall.go`) is the first consumer:
  - `--list-formats` — discovery: what can this be surfaced as.
  - `--format toon|json|markdown` — surface the SAME rendered/withheld note
    set under a chosen encoding at read time (`recallSurface` translates the
    envelope's rows in; a withheld note's body still never crosses into the
    surface under ANY format — only its Verdict + evidence Detail does, same
    as the existing markdown withheld line).
  - `--ablate-formats <list>|all` — instead of rendering, measure this note
    set's cost under every named format and print a bytes/~tokens table —
    the concrete "which format should I ask for" answer, read off real
    numbers instead of taken on faith.
- Fail-closed throughout: an unknown `--format` or an unknown name inside
  `--ablate-formats` refuses the whole call, naming the bad token and the
  known set (mirrors `internal/ablate`'s unknown-feature-token refusal).

## The measured claim (not asserted — witnessed)

`TestTOONIsCheaperThanJSONForUniformRows` (`internal/memview/format_test.go`)
renders 20 uniform 4-field rows under both encodings and asserts TOON is
strictly smaller in both bytes and estimated tokens — because JSON repeats
every field name once per row while TOON declares the field list exactly
once. This is the load-bearing ablation witness for "TOON can be used at the
right time": a caller (or `--ablate-formats`) can read the delta off real
encoder output for its own actual content shape, rather than trusting a
general claim about the format.

## What this does not claim

- Not full TOON-spec compliance: no nested object/array cells, no alternate
  delimiter option — a flat subset, because every `Surface` this codebase
  builds is already a flat, uniform record set. Documented in `format.go`'s
  package comment, not silently passed off as spec-complete.
- Not a new admission mechanism: `Surface` is built only from records a
  caller already ran through `VerdictFor`/`pageIn`; Format never widens what
  may render, only how already-admitted bytes are spelled.
- Not wired into `memq`'s driver-level renderer yet — `fak memory recall` is
  the first consumer; a `memq.Op`-level format op (so any driver's
  `RenderItem` list gets the same axis) is a natural, still-open follow-on.

## Witnesses

- `internal/memview/format_test.go` — 14 tests: ragged-row rejection,
  unknown-format fail-closed, sorted `KnownFormats()`, golden markdown/TOON
  output, markdown cell escaping, JSON round-trip, TOON ambiguous-cell
  quoting, TOON empty-rows header, `SweepFormats` determinism +
  full-registry-default + fail-closed-on-typo, the TOON-vs-JSON cost witness,
  and the open `Register` extension seam.
- `cmd/fak/memory_recall_test.go` — `TestMemoryRecall_listFormats`,
  `TestMemoryRecall_formatTOON` (surfaces all three read-time verdicts under
  TOON, confirms a withheld note's prose body still never crosses),
  `TestMemoryRecall_formatUnknownFailsClosed`, `TestMemoryRecall_ablateFormats`
  (measures a real fixture store under `json,toon` and confirms the table is
  scoped to exactly the requested formats).
- Layering: `internal/architest` continues to pin `memview` at tier 2
  (imports only abi(0)+stdlib) — `format.go` adds no new imports beyond
  `encoding/json`, `sort`, `strconv`, `strings`.

Run: `go test ./internal/memview/... ./cmd/fak/... -run 'Format|MemoryRecall'`.
