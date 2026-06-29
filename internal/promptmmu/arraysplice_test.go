package promptmmu

import (
	"encoding/json"
	"testing"
)

// TestArraySplicePoints covers the shared cached-prefix boundary primitive the
// system-prompt MMU (syspromptmmu Rung 2) uses to splice the system[] overlay past the
// SAME breakpoint discipline CompactInboundTools applies to tools[].
func TestArraySplicePoints(t *testing.T) {
	// A system[] of three blocks with the cache_control breakpoint on the MIDDLE block:
	// the cached prefix ends after block 1, the last element ends after block 2.
	b0 := `{"type":"text","text":"spine"}`
	b1 := `{"type":"text","text":"policy","cache_control":{"type":"ephemeral"}}`
	b2 := `{"type":"text","text":"overlay"}`
	raw := []byte(`{"model":"x","system":[` + b0 + `,` + b1 + `,` + b2 + `],"messages":[]}`)

	breakIdx, prefixEnd, lastElemEnd, ok := ArraySplicePoints(raw, "system")
	if !ok {
		t.Fatal("expected an anchor, got ok=false")
	}
	if breakIdx != 1 {
		t.Errorf("breakIdx = %d, want 1", breakIdx)
	}
	// prefixEnd must land just past block 1's closing brace.
	if string(raw[prefixEnd-1]) != "}" {
		t.Errorf("prefixEnd %d does not end on a brace: %q", prefixEnd, raw[prefixEnd-1])
	}
	wantPrefix := `{"model":"x","system":[` + b0 + `,` + b1
	if string(raw[:prefixEnd]) != wantPrefix {
		t.Errorf("prefix = %q\nwant     %q", raw[:prefixEnd], wantPrefix)
	}
	// lastElemEnd must land just past block 2 (before the array close).
	if string(raw[lastElemEnd:lastElemEnd+2]) != "]," {
		t.Errorf("lastElemEnd %d not at array close: %q", lastElemEnd, raw[lastElemEnd:lastElemEnd+2])
	}

	// The same primitive works on tools[] (its original home).
	tools := []byte(`{"tools":[{"name":"a","cache_control":{"type":"ephemeral"}},{"name":"b"}]}`)
	if _, _, _, ok := ArraySplicePoints(tools, "tools"); !ok {
		t.Error("tools[] with a breakpoint should anchor")
	}
}

// TestArraySplicePointsFailSafe asserts ok=false (fail-safe, no offsets) for every
// ambiguous shape, exactly as CompactInboundTools returns identity.
func TestArraySplicePointsFailSafe(t *testing.T) {
	cases := map[string][]byte{
		"empty":            nil,
		"not-object":       []byte(`["a"]`),
		"absent-array":     []byte(`{"model":"x"}`),
		"bare-string":      []byte(`{"system":"a plain string"}`),
		"empty-array":      []byte(`{"system":[]}`),
		"no-cache-control": []byte(`{"system":[{"type":"text","text":"foo"}]}`),
	}
	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			if _, _, _, ok := ArraySplicePoints(raw, "system"); ok {
				t.Errorf("%s: expected ok=false", name)
			}
		})
	}

	// Sanity: a JSON object decodes (so the not-object case is genuinely the object check).
	var probe map[string]json.RawMessage
	if json.Unmarshal([]byte(`{"system":[]}`), &probe) != nil {
		t.Fatal("fixture sanity: object should decode")
	}
}
