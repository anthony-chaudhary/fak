package gateway

// served_age_fresh_test.go — the two model-facing freshness affordances on top of the
// served-inline vDSO path:
//   1. a tier-2 served result tells the model HOW STALE it is ("~Nm old"), and
//   2. the model can FORCE A FRESH read by re-calling with "_fak_fresh": true, which
//      bypasses the cache probe so the call passes through to the client to actually run.

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// TestCacheAgeLabel is a pure-function test of the age renderer — no clock, no server, so
// it is fully deterministic. It proves the coarse seconds/minutes/hours buckets and that a
// missing/absent age_ms (tier-1 pure, tier-3 static, or a non-vdso caller) renders no age.
func TestCacheAgeLabel(t *testing.T) {
	cases := []struct {
		name    string
		meta    map[string]string
		wantLbl string
		wantOK  bool
	}{
		{"seconds", map[string]string{"age_ms": "45000"}, "45s", true},
		{"minutes", map[string]string{"age_ms": "180000"}, "3m", true},
		{"hours", map[string]string{"age_ms": "7200000"}, "2h", true},
		{"zero", map[string]string{"age_ms": "0"}, "0s", true},
		{"no-age-key (tier-1/3)", map[string]string{"tier": "1"}, "", false},
		{"nil meta", nil, "", false},
		{"garbage", map[string]string{"age_ms": "abc"}, "", false},
		{"negative", map[string]string{"age_ms": "-5"}, "", false},
	}
	for _, c := range cases {
		got, ok := cacheAgeLabel(c.meta)
		if ok != c.wantOK || got != c.wantLbl {
			t.Errorf("%s: cacheAgeLabel(%v) = (%q,%v), want (%q,%v)", c.name, c.meta, got, ok, c.wantLbl, c.wantOK)
		}
	}
}

// TestServedToolLineRendersAgeAndFreshHint proves the tier-2 served line names the age AND
// advertises the force-fresh marker, while a no-age (tier-1/3) line stays the plain form.
func TestServedToolLineRendersAgeAndFreshHint(t *testing.T) {
	withAge := servedToolLine("get_doc", []byte(`{"x":1}`), map[string]string{"age_ms": "180000"})
	if !strings.Contains(withAge, "~3m old") {
		t.Errorf("tier-2 line missing age: %q", withAge)
	}
	if !strings.Contains(withAge, fakFreshMarker) {
		t.Errorf("tier-2 line must advertise the force-fresh marker %q: %q", fakFreshMarker, withAge)
	}

	noAge := servedToolLine("calculate", []byte(`{"sum":2}`), map[string]string{"tier": "1"})
	if strings.Contains(noAge, "old") || strings.Contains(noAge, fakFreshMarker) {
		t.Errorf("tier-1 line must carry no age/fresh clause: %q", noAge)
	}
	if !strings.Contains(noAge, "served from cache") {
		t.Errorf("tier-1 line should still say served from cache: %q", noAge)
	}
}

// TestCallRequestsFresh is a pure-function test of the force-fresh marker parse.
func TestCallRequestsFresh(t *testing.T) {
	cases := []struct {
		args string
		want bool
	}{
		{`{"_fak_fresh":true}`, true},
		{`{"id":"x","_fak_fresh":true}`, true},
		{`{"_fak_fresh":false}`, false},
		{`{"id":"x"}`, false},
		{``, false},
		{`not json`, false},
		{`{"_fak_fresh":"true"}`, false}, // a string, not the bool — must not trip
	}
	for _, c := range cases {
		if got := callRequestsFresh(c.args); got != c.want {
			t.Errorf("callRequestsFresh(%q) = %v, want %v", c.args, got, c.want)
		}
	}
}

// TestForceFreshBypassesServedInline is the end-to-end loop: warm the vDSO, then a
// re-proposed read WITH the _fak_fresh marker must NOT be served inline — it passes
// through as a surviving tool_use the client actually runs. A re-proposal WITHOUT the
// marker is served inline (the contrast that proves the marker is what bypasses).
func TestForceFreshBypassesServedInline(t *testing.T) {
	srv, _ := newSharingServer(t, vdso.Global)
	const tool = "get_doc"
	warmServedRead(t, srv, tool, `{"id":"x"}`) // fills tier-2

	// Without the marker: served inline (the call is dropped, answered from cache).
	plain := proposeMessagesTurn(t, srv, []agent.ToolCall{
		{ID: "c1", Type: "function", Function: agent.Func{Name: tool, Arguments: `{"id":"x"}`}},
	})
	if tu, _ := countToolUse(plain); tu != 0 {
		t.Fatalf("plain re-read should be served inline (0 tool_use), got %d", tu)
	}

	// With the marker: bypass the cache, the call survives as a tool_use the client runs.
	fresh := proposeMessagesTurn(t, srv, []agent.ToolCall{
		{ID: "c2", Type: "function", Function: agent.Func{Name: tool, Arguments: `{"id":"x","_fak_fresh":true}`}},
	})
	tu, _ := countToolUse(fresh)
	if tu != 1 {
		t.Fatalf("force-fresh re-read must survive as a tool_use the client runs, got %d tool_use", tu)
	}
	if fresh.StopReason != "tool_use" {
		t.Fatalf("force-fresh turn stop_reason=%q, want tool_use (a call survived)", fresh.StopReason)
	}
}
