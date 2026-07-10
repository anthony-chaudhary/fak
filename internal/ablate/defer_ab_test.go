package ablate

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// The #3628 witness: the defer ON/OFF arm over ONE canonical tool-heavy workload reports a
// provider resident-token delta AND a held-accuracy figure. Both arms bind the same
// WorkloadHash (the frozen body's sha256), arming defer removes cold defs from the resident
// slice (positive delta), and every pre-defer tool stays reachable (held-accuracy 1.0).
func TestDeferABSweepReportsResidentDeltaAndHeldAccuracy(t *testing.T) {
	rep, err := DeferABOverCanonical()
	if err != nil {
		t.Fatalf("DeferABOverCanonical: %v", err)
	}
	if rep.ColdDeferred <= 0 {
		t.Fatalf("cold-tool deferral must fire over the canonical registry; ColdDeferred=%d", rep.ColdDeferred)
	}

	// Two arms over the SAME workload: OFF keeps every def resident, ON keeps only the hot
	// core + tool_search_tool, so the resident slice shrinks and its token count drops.
	if rep.OnResidentTools >= rep.OffResidentTools {
		t.Fatalf("defer ON resident tools (%d) must be fewer than OFF (%d)", rep.OnResidentTools, rep.OffResidentTools)
	}
	if rep.ResidentTokenDelta != rep.OffResidentTokens-rep.OnResidentTokens {
		t.Fatalf("ResidentTokenDelta=%d, want off-on=%d", rep.ResidentTokenDelta, rep.OffResidentTokens-rep.OnResidentTokens)
	}
	if rep.ResidentTokenDelta <= 0 {
		t.Fatalf("provider resident-token delta must be positive (defer faults cold defs out); got %d (off=%d on=%d)",
			rep.ResidentTokenDelta, rep.OffResidentTokens, rep.OnResidentTokens)
	}

	// Held-accuracy: no tool advertised OFF went missing ON — deferral cost the agent nothing.
	if !rep.AccuracyHeld() || rep.HeldAccuracy != 1.0 {
		t.Fatalf("held-accuracy must be 1.0 (no dropped tool); got %.4f held=%d/%d dropped=%v",
			rep.HeldAccuracy, rep.HeldToolCount, rep.TotalToolCount, rep.DroppedTools)
	}
	if rep.TotalToolCount != rep.OffResidentTools {
		t.Fatalf("held-accuracy denominator (%d) must be the OFF resident tool count (%d)", rep.TotalToolCount, rep.OffResidentTools)
	}
	if len(rep.DroppedTools) != 0 {
		t.Fatalf("no tool should be dropped by deferral; got %v", rep.DroppedTools)
	}
	if !rep.CachePrefixByteIdentical {
		t.Fatalf("non-tools cache prefix must be byte-identical across arms")
	}

	// One WorkloadHash binds both arms — and it is exactly the frozen body's sha256.
	want := sha256.Sum256(gateway.CanonicalDeferABBody())
	if rep.WorkloadHash != hex.EncodeToString(want[:]) {
		t.Fatalf("WorkloadHash=%q, want sha256(canonical body)=%q", rep.WorkloadHash, hex.EncodeToString(want[:]))
	}

	row := rep.SweepRow()
	for _, want := range []string{"held-accuracy", "resident tokens", "cold defs deferred"} {
		if !strings.Contains(row, want) {
			t.Fatalf("SweepRow %q missing %q", row, want)
		}
	}
	if len(rep.JSON()) == 0 {
		t.Fatalf("JSON artifact must render")
	}
}

// The arm is deterministic turn-over-turn (the cache-stability property the whole defer
// lever rests on): the same frozen body yields the same WorkloadHash, delta, and accuracy.
func TestDeferABSweepDeterministic(t *testing.T) {
	a, err := DeferABOverCanonical()
	if err != nil {
		t.Fatalf("first sweep: %v", err)
	}
	b, err := DeferABOverCanonical()
	if err != nil {
		t.Fatalf("second sweep: %v", err)
	}
	if a.WorkloadHash != b.WorkloadHash {
		t.Fatalf("WorkloadHash not stable: %q vs %q", a.WorkloadHash, b.WorkloadHash)
	}
	if a.ResidentTokenDelta != b.ResidentTokenDelta || a.HeldAccuracy != b.HeldAccuracy {
		t.Fatalf("arm not deterministic: delta %d/%d accuracy %.4f/%.4f",
			a.ResidentTokenDelta, b.ResidentTokenDelta, a.HeldAccuracy, b.HeldAccuracy)
	}
}

// Fail closed, never fabricate a zero-delta row: an empty body and a stand-down (the lever
// did not fire — here an already-armed body re-fed) both error instead of reporting a no-op.
func TestDeferABSweepFailsClosed(t *testing.T) {
	if _, err := DeferABSweep(nil); err == nil {
		t.Fatalf("empty body must fail closed")
	}
	armed := gateway.DeferColdToolsAB(gateway.CanonicalDeferABBody()).Armed
	if _, err := DeferABSweep(armed); err == nil {
		t.Fatalf("stand-down (already-armed body) must fail closed, not report a fabricated zero delta")
	}
}

// The held-accuracy figure is FALSIFIABLE, not trivially 1.0: a tool advertised OFF that is
// absent from the armed body's tools[] (a def DROPPED rather than deferred) is counted as a
// reachability regression and named. This is the negative control that keeps the accuracy
// signal honest.
func TestHeldAccuracyCatchesDroppedTool(t *testing.T) {
	off := []agent.ToolDef{
		{Function: agent.ToolDefFunction{Name: "Read"}},
		{Function: agent.ToolDefFunction{Name: "Bash"}},
		{Function: agent.ToolDefFunction{Name: "mcp__x__cold"}},
	}
	// Armed body advertises Read + Bash + tool_search_tool but DROPS mcp__x__cold entirely.
	dropped := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[` +
		`{"name":"Read","input_schema":{"type":"object"}},` +
		`{"name":"Bash","input_schema":{"type":"object"}},` +
		`{"name":"tool_search_tool","input_schema":{"type":"object"}}]}`)
	held, total, missing := heldAccuracy(off, dropped)
	if held != 2 || total != 3 {
		t.Fatalf("dropped-tool case: held=%d total=%d, want 2/3", held, total)
	}
	if len(missing) != 1 || missing[0] != "mcp__x__cold" {
		t.Fatalf("dropped tool must be named; got %v", missing)
	}

	// Positive control: advertise the cold def too (deferred, still in tools[]) — all held.
	kept := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[` +
		`{"name":"Read","input_schema":{"type":"object"}},` +
		`{"name":"Bash","input_schema":{"type":"object"}},` +
		`{"name":"mcp__x__cold","input_schema":{"type":"object"},"defer_loading":true},` +
		`{"name":"tool_search_tool","input_schema":{"type":"object"}}]}`)
	held, total, missing = heldAccuracy(off, kept)
	if held != 3 || total != 3 || len(missing) != 0 {
		t.Fatalf("all-reachable case: held=%d total=%d missing=%v, want 3/3/none", held, total, missing)
	}
}
