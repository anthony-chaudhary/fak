package agent

import (
	"strings"
	"testing"
)

// TestContainsEmptyJSONArray unit-tests the RepairEmptyToolResultContent fast-path guard. It MUST
// detect every shape isEmptyJSONArray/skipSpace call empty (`[` + any JSON whitespace + `]`) so the
// guard never skips a body the repair loop would edit, and it MUST NOT match a non-empty array (that
// would only forfeit the fast path, the harmless direction, but we pin it to catch a scan bug).
func TestContainsEmptyJSONArray(t *testing.T) {
	present := []string{
		`[]`,
		`[ ]`,
		"[\t]",
		"[\n\r ]",
		`{"content":[]}`,
		`{"a":1,"content":[  ]}`,
		"prefix [\n\t\r ] suffix",
	}
	for _, s := range present {
		if !containsEmptyJSONArray([]byte(s)) {
			t.Errorf("containsEmptyJSONArray(%q) = false, want true (empty array present)", s)
		}
	}
	absent := []string{
		``,
		`[1]`,
		`[{"type":"text"}]`,
		`["a"]`,
		`{"content":[{"type":"text","text":"ok"}]}`,
		`[x]`,
		`no brackets here`,
	}
	for _, s := range absent {
		if containsEmptyJSONArray([]byte(s)) {
			t.Errorf("containsEmptyJSONArray(%q) = true, want false (no empty array)", s)
		}
	}
}

// TestFastPathIdentityLargeBody proves the fast-path guards leave a large, realistic no-op body
// byte-identical with the exact same non-fired outcome the full parse would report — a pure speedup
// with no observable behavior change. The body carries many tool_results with NON-empty content and
// NO tool_reference block, the overwhelmingly common wire shape both functions run on.
func TestFastPathIdentityLargeBody(t *testing.T) {
	body := noopToolResultBody(400)

	out, outcome := SanitizeAnthropicToolReferences([]byte(body))
	if outcome.Reason != ToolRefReasonNoToolRef || outcome.Converted != 0 {
		t.Fatalf("sanitize: reason=%q converted=%d, want NoToolRef/0", outcome.Reason, outcome.Converted)
	}
	if string(out) != body {
		t.Fatalf("sanitize mutated a no-op body")
	}

	rep, repOut := RepairEmptyToolResultContent([]byte(body))
	if repOut.Reason != EmptyContentReasonNoEmpty || repOut.Repaired != 0 {
		t.Fatalf("repair: reason=%q repaired=%d, want NoEmpty/0", repOut.Reason, repOut.Repaired)
	}
	if string(rep) != body {
		t.Fatalf("repair mutated a no-op body")
	}
}

// TestFastPathStillFires guards against an over-eager skip: a body that genuinely needs each
// transform must still be edited even when it is large and mostly no-op, so the guard's substring /
// empty-array scan is a true SUPERSET, not a filter that drops real work.
func TestFastPathStillFires(t *testing.T) {
	// A real tool_reference buried after many benign tool_results still converts.
	withRef := `{"messages":[` + strings.Repeat(
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":"ok"}]}]},`, 200) +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"r","content":[{"type":"tool_reference","tool_name":"Read"}]}]}]}`
	_, outcome := SanitizeAnthropicToolReferences([]byte(withRef))
	if outcome.Reason != ToolRefReasonNone || outcome.Converted != 1 {
		t.Fatalf("buried tool_reference not converted: reason=%q converted=%d", outcome.Reason, outcome.Converted)
	}

	// A real empty content array buried after many benign tool_results still repairs.
	withEmpty := `{"messages":[` + strings.Repeat(
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":"ok"}]}]},`, 200) +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"e","content":[]}]}]}`
	_, repOut := RepairEmptyToolResultContent([]byte(withEmpty))
	if repOut.Reason != EmptyContentReasonNone || repOut.Repaired != 1 {
		t.Fatalf("buried empty content not repaired: reason=%q repaired=%d", repOut.Reason, repOut.Repaired)
	}
}

// noopToolResultBody builds a Messages body of n tool_result turns with non-empty text content and
// no tool_reference / empty array — the common no-op wire shape the fast path is built to skip.
func noopToolResultBody(n int) string {
	var b strings.Builder
	b.WriteString(`{"model":"claude-opus-4-8","messages":[`)
	for i := 0; i < n; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(`{"role":"user","content":[{"type":"tool_result","tool_use_id":"call","content":[{"type":"text","text":"a moderately long tool result payload standing in for real output on the wire"}]}]}`)
	}
	b.WriteString(`]}`)
	return b.String()
}

// BenchmarkSanitizeToolReferencesNoop measures the per-wire cost of the tool_reference sanitizer on
// the common no-op body (no tool_reference present). The fast-path substring guard should keep this
// far below the full-parse cost the pre-guard code paid on every wire.
func BenchmarkSanitizeToolReferencesNoop(b *testing.B) {
	raw := []byte(noopToolResultBody(400))
	b.ReportAllocs()
	b.SetBytes(int64(len(raw)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = SanitizeAnthropicToolReferences(raw)
	}
}

// BenchmarkRepairEmptyContentNoop measures the per-wire cost of the empty-content gate on the common
// no-op body (no empty array present). The fast-path scan should keep this far below the full-parse
// cost the pre-guard code paid on every wire.
func BenchmarkRepairEmptyContentNoop(b *testing.B) {
	raw := []byte(noopToolResultBody(400))
	b.ReportAllocs()
	b.SetBytes(int64(len(raw)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = RepairEmptyToolResultContent(raw)
	}
}

// BenchmarkSanitizeToolReferencesFullParse is the apples-to-apples counterfactual: a body that DEFEATS
// the substring guard (the literal "tool_reference" appears inside a benign text value, so the guard
// misses) but carries no real block — forcing the full decodeArrayElements + per-message scan the
// fast path exists to skip, with no splice. The gap vs. BenchmarkSanitizeToolReferencesNoop is the
// per-wire decode+scan cost the guard removes for the common no-op wire; it grows with turn count.
func BenchmarkSanitizeToolReferencesFullParse(b *testing.B) {
	raw := []byte(fullParseNoEditBody(400))
	b.ReportAllocs()
	b.SetBytes(int64(len(raw)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = SanitizeAnthropicToolReferences(raw)
	}
}

// BenchmarkRepairEmptyContentFullParse is the counterfactual for the empty-content gate: a body whose
// empty array sits in a NON-tool_result field (an empty tool-schema list), defeating the guard's
// scan but yielding no repair — the full decode+scan the fast path skips for a body with no empty
// array at all.
func BenchmarkRepairEmptyContentFullParse(b *testing.B) {
	raw := []byte(fullParseNoEmptyEditBody(400))
	b.ReportAllocs()
	b.SetBytes(int64(len(raw)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = RepairEmptyToolResultContent(raw)
	}
}

// fullParseNoEditBody is noopToolResultBody with the literal "tool_reference" planted inside a text
// value, so SanitizeAnthropicToolReferences' substring guard misses and the full parse runs, but the
// edit loop finds no real tool_reference block (NoToolRef).
func fullParseNoEditBody(n int) string {
	var b strings.Builder
	b.WriteString(`{"model":"claude-opus-4-8","messages":[`)
	for i := 0; i < n; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		text := "a moderately long tool result payload standing in for real output on the wire"
		if i == 0 {
			text = "the word tool_reference appears here in prose, not as a block type"
		}
		b.WriteString(`{"role":"user","content":[{"type":"tool_result","tool_use_id":"call","content":[{"type":"text","text":"` + text + `"}]}]}`)
	}
	b.WriteString(`]}`)
	return b.String()
}

// fullParseNoEmptyEditBody is noopToolResultBody with an empty array planted in a non-tool_result
// field, so RepairEmptyToolResultContent's scan matches and the full parse runs, but no tool_result
// content is empty (NoEmpty).
func fullParseNoEmptyEditBody(n int) string {
	var b strings.Builder
	b.WriteString(`{"model":"claude-opus-4-8","stop_sequences":[],"messages":[`)
	for i := 0; i < n; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(`{"role":"user","content":[{"type":"tool_result","tool_use_id":"call","content":[{"type":"text","text":"a moderately long tool result payload standing in for real output on the wire"}]}]}`)
	}
	b.WriteString(`]}`)
	return b.String()
}
