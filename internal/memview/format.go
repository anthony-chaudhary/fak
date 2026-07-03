package memview

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// format.go — the OUTPUT-FORMAT axis over a rendered set of views (the "views can be
// consumed/surfaced in a format like TOON" concept). This is deliberately a THIRD axis,
// orthogonal to the two the package already has:
//
//   - Kind      says WHAT was derived (snippet/summary/qa/fact) — memview.go.
//   - The taint/digest gate says WHETHER a view may enter context at all.
//   - Format    (this file) says HOW an already-admitted, already-rendered set of
//     records is SERIALIZED for a consumer — markdown prose, JSON, or a compact
//     tabular encoding (TOON-shaped). Changing Format never touches Kind, Source,
//     Taint, or Body; it is a pure re-encoding of already-admissible bytes.
//
// A Format is chosen (or ablated) AT SURFACE TIME — after the trust gate has already
// decided what may render, never before. This file supplies no new admission logic; a
// caller builds a Surface only from records that already cleared VerdictFor/pageIn.
//
// Surface is a generic, uniform table (a fixed field set + string rows) rather than
// []MemoryViewRecord directly, so a tier-3/4 caller (memq, cmd/fak) can render ITS OWN
// shaped rows (a recall envelope, a driver's RenderItem list, a MemoryViewRecord slice)
// under the same format registry without memview importing that caller's types — the
// same translate-in discipline timeline.go already uses for ProvenanceEvent.

// Format names a serialization a Surface can be rendered under. The set is OPEN —
// Register lets a caller add its own — but the three below are the shipped floor.
type Format string

const (
	// FormatMarkdown renders a human-readable table (the default; matches the prose
	// shape `fak memory recall`'s text mode already emits).
	FormatMarkdown Format = "markdown"
	// FormatJSON renders a canonical array-of-objects JSON encoding — lossless,
	// machine-parseable, the most verbose of the three (repeats every field name
	// once per row).
	FormatJSON Format = "json"
	// FormatTOON renders a TOON-shaped (Token-Oriented Object Notation) tabular
	// encoding: a header row declaring the row count + field names ONCE, then one
	// comma-joined line per row — no per-row key repetition, no braces/quotes for
	// plain scalars. Modeled on the toon-format/spec tabular-array rule (uniform
	// arrays of objects collapse to `name[N]{f1,f2,...}:` + indented rows); this is
	// a fak-local subset (flat records only, no nested-object/array cells) good
	// enough for the flat, uniform rows every Surface in this codebase produces.
	FormatTOON Format = "toon"
)

// Encoder renders a Surface to bytes under one Format. It must be total (never panic)
// and deterministic (same Surface -> byte-identical output) — SweepFormats depends on
// determinism to make cross-format token deltas meaningful rather than noise.
type Encoder func(Surface) []byte

var formatRegistry = map[Format]Encoder{
	FormatMarkdown: EncodeMarkdown,
	FormatJSON:     EncodeJSON,
	FormatTOON:     EncodeTOON,
}

// Register adds (or replaces) a Format's encoder — the open extension seam, mirroring
// memq.Register's "a plugin adds a strategy, not a kernel edit" posture applied to the
// format axis instead of the query axis.
func Register(f Format, enc Encoder) { formatRegistry[f] = enc }

// KnownFormats returns every registered format name, sorted (deterministic listing for
// `--list-formats` and error messages).
func KnownFormats() []string {
	out := make([]string, 0, len(formatRegistry))
	for f := range formatRegistry {
		out = append(out, string(f))
	}
	sort.Strings(out)
	return out
}

// Row is one record of a Surface: a value per Surface.Fields, same order, same length.
// Values are pre-stringified by the caller (a translate-in step, like ProvenanceEvent)
// so every encoder stays total: no encoder needs to special-case numbers/nils/nested
// structures, and a value that looks numeric renders unquoted in TOON without the
// encoder having to guess a type from an any.
type Row []string

// Surface is the one render-ready shape every Format encoder consumes: a bounded
// title, a fixed field set, and uniform rows. It is the format axis's analogue of
// ProvenanceEvent — a caller (memq, cmd/fak) translates its own typed result into a
// Surface once, and every registered Format is then available for free.
type Surface struct {
	Title  string
	Fields []string
	Rows   []Row
}

// NewSurface validates that every row has the same field count as Fields and returns
// the Surface, or an error naming the offending row — fail-closed rather than a
// silently ragged table that would corrupt the TOON header's declared field count.
func NewSurface(title string, fields []string, rows []Row) (Surface, error) {
	for i, r := range rows {
		if len(r) != len(fields) {
			return Surface{}, fmt.Errorf("memview: row %d has %d value(s), want %d (fields %v)", i, len(r), len(fields), fields)
		}
	}
	return Surface{Title: title, Fields: append([]string(nil), fields...), Rows: rows}, nil
}

// Encode renders s under the named format, or an error naming the unknown format and
// the known set — the same fail-closed-with-the-full-list posture ablate.BuildSweep
// uses for an unknown feature token.
func Encode(f Format, s Surface) ([]byte, error) {
	enc, ok := formatRegistry[f]
	if !ok {
		return nil, fmt.Errorf("memview: unknown format %q (known: %s)", f, strings.Join(KnownFormats(), ", "))
	}
	return enc(s), nil
}

// EncodeMarkdown renders a GitHub-flavored markdown table: a title line, a header
// row, a separator, then one row per record. Empty Surfaces render a title plus a
// "(no rows)" line rather than a headerless/bodyless table.
func EncodeMarkdown(s Surface) []byte {
	var b strings.Builder
	if s.Title != "" {
		fmt.Fprintf(&b, "## %s\n\n", s.Title)
	}
	if len(s.Fields) == 0 {
		b.WriteString("(no fields)\n")
		return []byte(b.String())
	}
	b.WriteString("| " + strings.Join(s.Fields, " | ") + " |\n")
	b.WriteString("|" + strings.Repeat(" --- |", len(s.Fields)) + "\n")
	if len(s.Rows) == 0 {
		b.WriteString("| " + strings.Join(makeDashes(len(s.Fields)), " | ") + " |\n")
	}
	for _, r := range s.Rows {
		cells := make([]string, len(r))
		for i, v := range r {
			cells[i] = escapeMarkdownCell(v)
		}
		b.WriteString("| " + strings.Join(cells, " | ") + " |\n")
	}
	return []byte(b.String())
}

func makeDashes(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = "-"
	}
	return out
}

func escapeMarkdownCell(v string) string {
	v = strings.ReplaceAll(v, "\\", "\\\\")
	v = strings.ReplaceAll(v, "|", "\\|")
	v = strings.ReplaceAll(v, "\n", " ")
	return v
}

// surfaceJSON is the wire shape EncodeJSON emits: an array of {field: value} objects
// preserving Surface.Fields' order via an ordered-key encode (encoding/json sorts map
// keys alphabetically, which would silently reorder columns, so this walks Fields
// explicitly instead of json.Marshal'ing a map).
func EncodeJSON(s Surface) []byte {
	var b strings.Builder
	b.WriteByte('[')
	for i, r := range s.Rows {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteByte('{')
		for j, field := range s.Fields {
			if j > 0 {
				b.WriteByte(',')
			}
			key, _ := json.Marshal(field)
			val, _ := json.Marshal(r[j])
			b.Write(key)
			b.WriteByte(':')
			b.Write(val)
		}
		b.WriteByte('}')
	}
	b.WriteByte(']')
	b.WriteByte('\n')
	return []byte(b.String())
}

// EncodeTOON renders s as a TOON-shaped tabular array: one header line declaring the
// row count and field names ONCE (`rows[N]{f1,f2,...}:`), then one two-space-indented,
// comma-joined line per row. A bare value is emitted unquoted when it needs no
// escaping (no comma/quote/newline and non-empty); otherwise it is JSON-string-quoted
// — the same "quote only when required" rule the TOON spec uses to keep plain scalars
// token-cheap while staying unambiguous. This is a flat subset of the real spec (no
// nested objects/arrays per cell): every Surface in this codebase is already a flat
// record set, so the subset costs nothing here and keeps the encoder total.
func EncodeTOON(s Surface) []byte {
	var b strings.Builder
	name := "rows"
	if s.Title != "" {
		name = toonIdent(s.Title)
	}
	fmt.Fprintf(&b, "%s[%d]{%s}:\n", name, len(s.Rows), strings.Join(s.Fields, ","))
	for _, r := range s.Rows {
		cells := make([]string, len(r))
		for i, v := range r {
			cells[i] = toonCell(v)
		}
		b.WriteString("  " + strings.Join(cells, ",") + "\n")
	}
	return []byte(b.String())
}

// toonIdent lowercases and underscores a title into a bare TOON array name.
func toonIdent(title string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(title) {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	out := strings.Trim(b.String(), "_")
	if out == "" {
		return "rows"
	}
	return out
}

// toonCell renders one value bare when safe, else JSON-string-quoted. A value needs
// quoting if it contains the row/field delimiter (comma), a quote, a newline, is
// empty, or parses as a number/bool/"null" (which would otherwise be misread as a
// typed scalar rather than the literal string) — the same ambiguity TOON's own
// quoting rule exists to avoid.
func toonCell(v string) string {
	if v == "" || strings.ContainsAny(v, ",\"\n") || looksTyped(v) {
		q, _ := json.Marshal(v)
		return string(q)
	}
	return v
}

func looksTyped(v string) bool {
	if v == "true" || v == "false" || v == "null" {
		return true
	}
	_, err := strconv.ParseFloat(v, 64)
	return err == nil
}

// FormatMetrics is one format's measured render cost for a fixed Surface — the
// ablation row. EstimatedTokens uses the same coarse bytes/4 heuristic memq.Stats
// already uses (memq/exec.go tokenEstimate), so a cross-package comparison stays on
// one consistent yardstick rather than mixing estimators.
type FormatMetrics struct {
	Format          Format `json:"format"`
	Bytes           int    `json:"bytes"`
	EstimatedTokens int    `json:"estimated_tokens"`
}

// SweepFormats renders the SAME Surface under every named format (or every registered
// format, if formats is empty) and reports each one's byte/token cost — the ablation
// half of the format axis: hold the content constant, vary only the encoding, read the
// size delta straight off deterministic encoders (no model, $0, like internal/ablate's
// feature sweep does for kernel knobs). An unknown format name fails the whole sweep
// closed rather than silently skipping it, so a typo in `--ablate-formats` cannot
// produce a report that quietly measured fewer arms than requested.
func SweepFormats(s Surface, formats []Format) ([]FormatMetrics, error) {
	if len(formats) == 0 {
		for _, name := range KnownFormats() {
			formats = append(formats, Format(name))
		}
	}
	out := make([]FormatMetrics, 0, len(formats))
	for _, f := range formats {
		body, err := Encode(f, s)
		if err != nil {
			return nil, err
		}
		out = append(out, FormatMetrics{Format: f, Bytes: len(body), EstimatedTokens: tokenEstimate(len(body))})
	}
	return out, nil
}

func tokenEstimate(n int) int {
	if n <= 0 {
		return 0
	}
	return (n + 3) / 4
}
