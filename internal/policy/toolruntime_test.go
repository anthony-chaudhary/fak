package policy

import (
	"strings"
	"testing"
)

// TestToolRuntimeEnvelopeResolvesFromTheManifest is the seam-5 acceptance
// line: the capability manifest grants a per-tool runtime envelope, and
// EnvelopeFor resolves it — exact row first, "*" catch-all second, nothing
// invented when neither exists.
func TestToolRuntimeEnvelopeResolvesFromTheManifest(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{
		"version": "fak-policy/v1",
		"allow": ["Bash", "Read"],
		"tool_runtime": [
			{"tool": "Bash", "deadline_ms": 600000, "heartbeat_every_ms": 30000},
			{"tool": "*", "deadline_ms": 120000}
		]
	}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	if rt.ToolRuntime == nil {
		t.Fatal("Runtime.ToolRuntime is nil for a manifest that declares envelopes")
	}
	// Exact row wins over the catch-all.
	r, ok := rt.ToolRuntime.EnvelopeFor("Bash")
	if !ok || r.DeadlineMS != 600000 || r.HeartbeatEveryMS != 30000 {
		t.Fatalf("EnvelopeFor(Bash) = %+v, %v; want the exact row", r, ok)
	}
	// A tool with no exact row falls to "*".
	r, ok = rt.ToolRuntime.EnvelopeFor("Read")
	if !ok || r.DeadlineMS != 120000 || r.HeartbeatEveryMS != 0 {
		t.Fatalf("EnvelopeFor(Read) = %+v, %v; want the catch-all row", r, ok)
	}
	// Identity tolerates surrounding whitespace but stays case-sensitive
	// (tool names are exact identifiers on the floor, same as Allow).
	if _, ok := rt.ToolRuntime.EnvelopeFor("  Bash "); !ok {
		t.Fatal("EnvelopeFor(\"  Bash \") missed the exact row")
	}
	if r, _ := rt.ToolRuntime.EnvelopeFor("bash"); r.DeadlineMS != 120000 {
		t.Fatalf("EnvelopeFor(bash) = %+v; lowercase must NOT match the exact Bash row (falls to catch-all)", r)
	}
}

// TestToolRuntimeAbsentGrantsNothing: no tool_runtime block means a nil table,
// and a nil table resolves nothing — absent config never invents an envelope.
func TestToolRuntimeAbsentGrantsNothing(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{"version": "fak-policy/v1", "allow": ["Read"]}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	if rt.ToolRuntime != nil {
		t.Fatalf("Runtime.ToolRuntime = %+v, want nil for a manifest with no tool_runtime block", rt.ToolRuntime)
	}
	// Nil-safety is part of the contract: embedders call through without a guard.
	if r, ok := rt.ToolRuntime.EnvelopeFor("Bash"); ok {
		t.Fatalf("nil table EnvelopeFor(Bash) = %+v, true; want zero, false", r)
	}
	if rows := rt.ToolRuntime.Rules(); rows != nil {
		t.Fatalf("nil table Rules() = %+v, want nil", rows)
	}
	// A tool with no row and no catch-all resolves nothing.
	rt2, err := ParseRuntime([]byte(`{
		"tool_runtime": [{"tool": "Bash", "deadline_ms": 1000}]
	}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	if r, ok := rt2.ToolRuntime.EnvelopeFor("Read"); ok {
		t.Fatalf("EnvelopeFor(Read) = %+v, true; want zero, false (no row, no catch-all)", r)
	}
}

// TestToolRuntimeRefusesBadRowsAtLoad: an empty tool name, a negative value,
// an all-zero row, and a duplicate tool each fail loud at LOAD — never at
// spawn time.
func TestToolRuntimeRefusesBadRowsAtLoad(t *testing.T) {
	cases := []struct {
		name, manifest, wantErr string
	}{
		{"empty tool", `{"tool_runtime": [{"tool": " ", "deadline_ms": 1}]}`, "tool is required"},
		{"negative deadline", `{"tool_runtime": [{"tool": "Bash", "deadline_ms": -1}]}`, "non-negative"},
		{"negative heartbeat", `{"tool_runtime": [{"tool": "Bash", "heartbeat_every_ms": -5}]}`, "non-negative"},
		{"all-zero row", `{"tool_runtime": [{"tool": "Bash"}]}`, "grants nothing"},
		{"duplicate tool", `{"tool_runtime": [{"tool": "Bash", "deadline_ms": 1}, {"tool": "Bash", "deadline_ms": 2}]}`, "duplicate"},
		{"duplicate after trim", `{"tool_runtime": [{"tool": "Bash", "deadline_ms": 1}, {"tool": " Bash ", "deadline_ms": 2}]}`, "duplicate"},
	}
	for _, tc := range cases {
		if _, err := ParseRuntime([]byte(tc.manifest)); err == nil {
			t.Errorf("%s: ParseRuntime accepted %s; want error containing %q", tc.name, tc.manifest, tc.wantErr)
		} else if !strings.Contains(err.Error(), tc.wantErr) {
			t.Errorf("%s: error = %q, want it to contain %q", tc.name, err, tc.wantErr)
		}
	}
}

// TestToolRuntimeSummaryRendersTheGrant: the operator summary names every
// envelope row (sorted, deterministic) and states the honest default when the
// block is absent.
func TestToolRuntimeSummaryRendersTheGrant(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{
		"tool_runtime": [
			{"tool": "WebFetch", "deadline_ms": 60000},
			{"tool": "Bash", "deadline_ms": 600000, "heartbeat_every_ms": 30000}
		]
	}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	sum := SummaryRuntime(rt)
	if !strings.Contains(sum, "tool runtime       : 2 envelope(s)") {
		t.Fatalf("summary missing the envelope count:\n%s", sum)
	}
	if !strings.Contains(sum, "Bash -> deadline_ms=600000 heartbeat_every_ms=30000") ||
		!strings.Contains(sum, "WebFetch -> deadline_ms=60000 heartbeat_every_ms=0") {
		t.Fatalf("summary missing an envelope row:\n%s", sum)
	}
	bash := strings.Index(sum, "Bash ->")
	web := strings.Index(sum, "WebFetch ->")
	if bash < 0 || web < 0 || bash > web {
		t.Fatalf("summary rows not sorted by tool name:\n%s", sum)
	}
	empty, err := ParseRuntime([]byte(`{"allow": ["Read"]}`))
	if err != nil {
		t.Fatalf("ParseRuntime(empty): %v", err)
	}
	if sum := SummaryRuntime(empty); !strings.Contains(sum, "tool runtime       : (none — fold defaults apply)") {
		t.Fatalf("summary missing the absent-block line:\n%s", sum)
	}
}
