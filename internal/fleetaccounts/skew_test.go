package fleetaccounts

import (
	"strings"
	"testing"
)

// seat is a tiny fixture helper: one seat-pool row keyed by its own tag, carrying the
// named live workers.
func seatRow(tag string, workers ...string) Seat {
	return Seat{Seat: "dir:" + tag, Tag: tag, Account: ".claude-" + tag, Workers: workers}
}

// The witness for issue #1805: a fixture with imbalanced assignments emits a skew warning
// BEFORE the heavy pool becomes a hard throttle or auth failure.
func TestDetectSeatSkewFlagsImbalancedPool(t *testing.T) {
	pool := SeatPool{Seats: []Seat{
		seatRow("hot", "w1", "w2", "w3", "w4"), // 4 workers piled on one pool
		seatRow("cool-a", "w5"),                // peers nearly idle
		seatRow("cool-b"),
	}}

	warnings := DetectSeatSkew(pool, DefaultSkewPolicy())
	if len(warnings) != 1 {
		t.Fatalf("DetectSeatSkew = %d warnings, want exactly 1: %+v", len(warnings), warnings)
	}
	w := warnings[0]
	if w.Tag != "hot" {
		t.Errorf("skew flagged tag %q, want the overloaded pool %q", w.Tag, "hot")
	}
	if w.Load != 4 {
		t.Errorf("skew load = %d, want 4", w.Load)
	}
	// peer mean = (1+0)/2 = 0.5, ratio computed against a floor of 1.0 => 4.0x.
	if w.PeerMean != 0.5 {
		t.Errorf("peer mean = %.2f, want 0.50", w.PeerMean)
	}
	if w.Ratio != 4.0 {
		t.Errorf("ratio = %.2f, want 4.00", w.Ratio)
	}

	rendered := RenderSeatSkew(warnings)
	if !strings.Contains(rendered, "ACCOUNT-POOL SKEW") {
		t.Errorf("render missing skew header:\n%s", rendered)
	}
	if !strings.Contains(rendered, "hot") || !strings.Contains(rendered, "4 live workers") {
		t.Errorf("render missing the flagged pool detail:\n%s", rendered)
	}
}

// A balanced fan-out must NOT warn — the detector may not cry skew on healthy spread.
func TestDetectSeatSkewBalancedPoolIsQuiet(t *testing.T) {
	pool := SeatPool{Seats: []Seat{
		seatRow("a", "w1"),
		seatRow("b", "w2"),
		seatRow("c", "w3"),
	}}
	if got := DetectSeatSkew(pool, DefaultSkewPolicy()); len(got) != 0 {
		t.Fatalf("balanced pool flagged %d skew(s), want 0: %+v", len(got), got)
	}
	if r := RenderSeatSkew(nil); r != "" {
		t.Errorf("RenderSeatSkew(nil) = %q, want empty", r)
	}
}

// A mild lead (3 vs 2/2) is within the default factor and must stay quiet: this guards
// against a trigger-happy detector that would warn on every uneven-but-fine spread.
func TestDetectSeatSkewMildLeadIsQuiet(t *testing.T) {
	pool := SeatPool{Seats: []Seat{
		seatRow("lead", "w1", "w2", "w3"),
		seatRow("b", "w4", "w5"),
		seatRow("c", "w6", "w7"),
	}}
	if got := DetectSeatSkew(pool, DefaultSkewPolicy()); len(got) != 0 {
		t.Fatalf("mild lead flagged %d skew(s), want 0: %+v", len(got), got)
	}
}

// Two dirs sharing one Anthropic account (same PoolKey) aggregate into one pool's load,
// so the skew is measured per rate-limit pool, not per dir.
func TestDetectSeatSkewAggregatesSharedPool(t *testing.T) {
	pool := SeatPool{Seats: []Seat{
		{Seat: "uuid:shared", Tag: "dir1", Account: ".claude-dir1", Workers: []string{"w1", "w2"}},
		{Seat: "uuid:shared", Tag: "dir2", Account: ".claude-dir2", Workers: []string{"w3"}},
		seatRow("idle-a"),
		seatRow("idle-b"),
	}}
	warnings := DetectSeatSkew(pool, DefaultSkewPolicy())
	if len(warnings) != 1 {
		t.Fatalf("shared-pool skew = %d warnings, want 1: %+v", len(warnings), warnings)
	}
	if warnings[0].Pool != "uuid:shared" || warnings[0].Load != 3 {
		t.Errorf("aggregated pool = %+v, want pool uuid:shared load 3", warnings[0])
	}
}

// Fewer than two pools => no peer baseline => no skew possible.
func TestDetectSeatSkewSinglePoolNoPeers(t *testing.T) {
	pool := SeatPool{Seats: []Seat{seatRow("solo", "w1", "w2", "w3", "w4")}}
	if got := DetectSeatSkew(pool, DefaultSkewPolicy()); got != nil {
		t.Fatalf("single pool flagged skew %+v, want nil (no peers)", got)
	}
}
