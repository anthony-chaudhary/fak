package gateway

// served_max_age_test.go — #1349: the OPERATOR's deterministic freshness ceiling on the
// served-inline vDSO path, the counterpart to the model-driven _fak_fresh in
// served_age_fresh_test.go. A per-tool max-age (TTL) says "a hit for this tool older than
// D must NOT be served from cache"; adjudicateProposedServed then declines the inline
// serve and passes the call through to the client to run fresh (the same already-tested
// fall-through as a cache miss / _fak_fresh). This is the first ENFORCED consumer of
// abi.ConsistencyBoundedStale's staleness bound (previously recorded but never enforced).

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// ageStubFastPath is a fixed-age tier-2 fast path: it answers a single tool with a chosen
// body and a chosen age_ms, so the TTL gate is exercised DETERMINISTICALLY without any
// dependence on wall-clock elapsed time (a real fill+re-read can be 0ms apart). It is the
// in-package analogue of injecting the vDSO clock.
type ageStubFastPath struct {
	tool  string
	body  string
	ageMs string
}

func (f ageStubFastPath) Caps() []abi.Capability { return nil }

func (f ageStubFastPath) Lookup(ctx context.Context, c *abi.ToolCall) (*abi.Result, bool) {
	if c == nil || c.Tool != f.tool {
		return nil, false
	}
	return &abi.Result{
		Call:    c,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(f.body)},
		Status:  abi.StatusOK,
		Meta:    map[string]string{"served_by": "vdso", "tier": "2", "age_ms": f.ageMs},
	}, true
}

// newAgeStubServer wires an isolated chain (inline backend + echo engine + allow-all
// adjudicator) with the given fast path registered, and returns a served gateway bound to
// it — the served_sharing_test.go recipe with a controllable-age fast path swapped for the
// real vDSO so the TTL assertions are hermetic.
func newAgeStubServer(t *testing.T, fp abi.FastPath) *Server {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, allowAllAdj{})
	abi.RegisterFastPath(1, fp)

	srv, err := New(Config{EngineID: "test", Model: "test", VDSO: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv
}

// TestServedHitOverMaxAge is a pure-function test of the TTL decision — no server, no
// clock, so it is fully deterministic. It pins the fail-open shape: no config, no ceiling
// for the tool, a non-positive ceiling, a missing/garbage/negative age, and the exact
// boundary all leave the serve UNGATED; only a positive ceiling strictly exceeded gates.
func TestServedHitOverMaxAge(t *testing.T) {
	s := &Server{maxAgeByTool: map[string]time.Duration{
		"get_quote": 30 * time.Second,
		"get_slow":  0, // non-positive ceiling => no gate even though the key is present
	}}
	cases := []struct {
		name string
		tool string
		meta map[string]string
		want bool
	}{
		{"over ceiling", "get_quote", map[string]string{"age_ms": "45000"}, true},
		{"under ceiling", "get_quote", map[string]string{"age_ms": "15000"}, false},
		{"equal boundary is served", "get_quote", map[string]string{"age_ms": "30000"}, false},
		{"tool has no ceiling", "get_other", map[string]string{"age_ms": "999999"}, false},
		{"zero ceiling disables", "get_slow", map[string]string{"age_ms": "999999"}, false},
		{"no age_ms (tier-1/3)", "get_quote", map[string]string{"tier": "1"}, false},
		{"garbage age", "get_quote", map[string]string{"age_ms": "abc"}, false},
		{"negative age", "get_quote", map[string]string{"age_ms": "-5"}, false},
		{"nil meta", "get_quote", nil, false},
	}
	for _, c := range cases {
		if got := s.servedHitOverMaxAge(c.tool, c.meta); got != c.want {
			t.Errorf("%s: servedHitOverMaxAge(%q,%v)=%v, want %v", c.name, c.tool, c.meta, got, c.want)
		}
	}

	// An empty/nil config never gates — the byte-identical-to-today path.
	var empty Server
	if empty.servedHitOverMaxAge("get_quote", map[string]string{"age_ms": "999999"}) {
		t.Error("empty config must never gate")
	}

	// SetToolMaxAge lifecycle: install a ceiling, then a non-positive duration removes it.
	empty.SetToolMaxAge("get_quote", 10*time.Second)
	if !empty.servedHitOverMaxAge("get_quote", map[string]string{"age_ms": "20000"}) {
		t.Error("after SetToolMaxAge(10s), a 20s hit must be over-age")
	}
	empty.SetToolMaxAge("get_quote", 0)
	if empty.servedHitOverMaxAge("get_quote", map[string]string{"age_ms": "20000"}) {
		t.Error("SetToolMaxAge(0) must remove the ceiling")
	}
}

// TestServedInline_MaxAgeCeiling is the end-to-end witness: a fixed 60s-old tier-2 hit is
// served inline with NO ceiling and UNDER a generous ceiling, but a 30s ceiling (below the
// 60s age) declines the inline serve so the call passes through as a tool_use the client
// runs fresh — the operator's hard freshness bound, deterministic via the fixed-age hit.
func TestServedInline_MaxAgeCeiling(t *testing.T) {
	const tool = "get_quote"
	srv := newAgeStubServer(t, ageStubFastPath{tool: tool, body: `{"px":42}`, ageMs: "60000"}) // 60s old

	// No ceiling configured: the 60s hit is served inline (dropped, 0 tool_use).
	base := proposeMessagesTurn(t, srv, []agent.ToolCall{
		{ID: "c0", Type: "function", Function: agent.Func{Name: tool, Arguments: `{"id":"x"}`}},
	})
	if tu, _ := countToolUse(base); tu != 0 {
		t.Fatalf("no ceiling: a 60s hit should be served inline (0 tool_use), got %d", tu)
	}

	// Ceiling 30s < 60s age: the hit is OVER age -> declined -> survives as a tool_use the
	// client runs fresh (the _fak_fresh fall-through, now operator-driven).
	srv.SetToolMaxAge(tool, 30*time.Second)
	over := proposeMessagesTurn(t, srv, []agent.ToolCall{
		{ID: "c1", Type: "function", Function: agent.Func{Name: tool, Arguments: `{"id":"x"}`}},
	})
	if tu, _ := countToolUse(over); tu != 1 {
		t.Fatalf("over-age: a 60s hit past a 30s ceiling must pass through as a tool_use, got %d", tu)
	}
	if over.StopReason != "tool_use" {
		t.Fatalf("over-age: stop_reason=%q, want tool_use (the call survived to run fresh)", over.StopReason)
	}

	// Ceiling 120s > 60s age: under the ceiling -> served inline again (the contrast that
	// proves the ceiling — not some unrelated change — is what declined the serve above).
	srv.SetToolMaxAge(tool, 120*time.Second)
	under := proposeMessagesTurn(t, srv, []agent.ToolCall{
		{ID: "c2", Type: "function", Function: agent.Func{Name: tool, Arguments: `{"id":"x"}`}},
	})
	if tu, txt := countToolUse(under); tu != 0 || !strings.Contains(txt, "served from cache") {
		t.Fatalf("under-age: a 60s hit within a 120s ceiling must be served inline, got tool_use=%d text=%q", tu, txt)
	}
}
