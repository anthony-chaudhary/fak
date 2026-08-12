package model

// ep_decode_coord_prefix_test.go — #5553: the prefix-reuse blocker the serve wiring had to
// settle before the coordinator could be installed at all.
//
// epCheckMirror fails closed when rank 0 announces a forward at a position the follower's
// mirror never computed, and that is EXACTLY what a restored KV prefix produces: rank 0
// runs zero forwards over the matched prefix, so its next PREFILL lands at pos>0 against a
// follower mirror still at 0. The cure taken is to disable prefix reuse while a coordinator
// is installed, and EPDecodeCoordinated is the flag the decode driver above this package
// (agent.inKernelPlannerPrefixReuseSupported) reads to do it.
//
// The two tests below pin the two halves: the flag tracks the coordinator, and the failure
// it exists to prevent is real (a follower REFUSES a reused-prefix announcement rather than
// reducing a partial computed from a different context).

import (
	"strings"
	"testing"
)

// TestEPDecodeCoordinatedTracksTheInstalledCoordinator pins the accessor the decode driver
// keys prefix reuse off. A nil model and an ordinary (non-sharded) serve must both report
// false — that is what keeps every existing single-process path on the reuse fast path.
func TestEPDecodeCoordinatedTracksTheInstalledCoordinator(t *testing.T) {
	var nilModel *Model
	if nilModel.EPDecodeCoordinated() {
		t.Fatalf("nil model reports a coordinated decode — the accessor must be nil-safe, or every planner built before a model loads would drop prefix reuse")
	}

	m := NewSynthetic(llamaArchConfig())
	if m.EPDecodeCoordinated() {
		t.Fatalf("a freshly built model reports a coordinated decode; every ordinary serve installs no coordinator and must keep prefix reuse")
	}

	m.SetEPDecodeCoordinator(&EPDecodeCoordinator{})
	if !m.EPDecodeCoordinated() {
		t.Fatalf("model with a coordinator installed reports EPDecodeCoordinated()=false — the decode driver would then resume a coordinated session from a restored prefix and desync every follower (#5553)")
	}

	m.SetEPDecodeCoordinator(nil)
	if m.EPDecodeCoordinated() {
		t.Fatalf("clearing the coordinator left EPDecodeCoordinated()=true; the flag must track the field, not latch")
	}
}

// TestEPFollowerRefusesAReusedPrefixAnnouncement is the defect the decision above avoids,
// made concrete: a rank 0 that resumed from a cached prefix announces PREFILL at pos>0, and
// the follower's fresh mirror is at 0. Without the reuse gate this is what a coordinated
// serve would hit on its first cache hit.
//
// It is asserted at epCheckMirror rather than over a live group because the refusal IS the
// protocol's contract; the multi-rank consequence (the reduce summing a partial computed
// from a different context) stays inferred from the AllReduce contract, not measured — no
// multi-rank hardware was involved.
func TestEPFollowerRefusesAReusedPrefixAnnouncement(t *testing.T) {
	m := NewSynthetic(llamaArchConfig())
	mirror := m.NewSession() // a fresh follower mirror: nothing prefilled, so epSeqLen()==0

	// Rank 0 matched 7 cached tokens and prefills only the divergent suffix, so it announces
	// its PREFILL at position 7.
	err := epCheckMirror(&DistComm{rank: 1}, mirror, epOpPrefill, 7)
	if err == nil {
		t.Fatalf("follower accepted a PREFILL at pos 7 against a mirror at pos 0 — a reused prefix would then reduce hidden states computed from different contexts, which is the one failure this protocol must never take silently (#5553)")
	}
	if !strings.Contains(err.Error(), "mirror desync") {
		t.Fatalf("refusal = %q, want the mirror-desync refusal so an operator can tell this apart from a transport error", err)
	}

	// And the same announcement against a mirror that DID compute those positions is fine —
	// otherwise the test above would pass for a coordinator that refused everything.
	mirror.Prefill([]int{1, 2, 3, 4, 5, 6, 7})
	if err := epCheckMirror(&DistComm{rank: 1}, mirror, epOpStep, 7); err != nil {
		t.Fatalf("follower refused an IN-SYNC announcement at pos 7: %v", err)
	}
}
