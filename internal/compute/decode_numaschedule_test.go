package compute

import (
	"reflect"
	"testing"
)

// topo is a tiny builder: node ids each with a contiguous CPU block of `coresPerNode`,
// laid out da33-style (node k owns cpus [k*coresPerNode, (k+1)*coresPerNode)).
func topo(nodes, coresPerNode int) []NUMANodeTopology {
	out := make([]NUMANodeTopology, nodes)
	for n := 0; n < nodes; n++ {
		cpus := make([]int, coresPerNode)
		for c := 0; c < coresPerNode; c++ {
			cpus[c] = n*coresPerNode + c
		}
		out[n] = NUMANodeTopology{NodeID: n, CPUs: cpus}
	}
	return out
}

func TestScheduleDecodeNUMA_EvenDistribution(t *testing.T) {
	// da33 shape: 8 nodes, 32 cores/node, 64 decode workers ⇒ 8 workers per node. The
	// shape drives both the call and the assertions, so the invariant asserted is
	// "one placement per requested worker", not the number 64.
	const (
		nodes        = 8
		coresPerNode = 32
		workers      = 64
		perNode      = workers / nodes
	)
	s := ScheduleDecodeNUMA(topo(nodes, coresPerNode), workers)
	if !s.Eligible || s.Reason != DecodeNUMAScheduleEligible {
		t.Fatalf("want eligible, got reason=%q", s.Reason)
	}
	if len(s.Placements) != workers {
		t.Fatalf("want %d placements, got %d", workers, len(s.Placements))
	}
	if s.Oversubscribed {
		t.Fatalf("64 workers over 8×32 cores must not oversubscribe")
	}
	want := []int{8, 8, 8, 8, 8, 8, 8, 8}
	if !reflect.DeepEqual(s.PerNodeWorkers, want) {
		t.Fatalf("per-node workers = %v, want %v", s.PerNodeWorkers, want)
	}
	// Every worker index appears exactly once, block-wise ascending, pinned to its node's CPUs.
	for i, p := range s.Placements {
		if p.Worker != i {
			t.Fatalf("placement %d has worker %d, want ascending", i, p.Worker)
		}
		wantNode := i / perNode
		if p.NodeID != wantNode {
			t.Fatalf("worker %d homed on node %d, want %d (block-wise)", i, p.NodeID, wantNode)
		}
		if p.CPUs[0] != wantNode*coresPerNode || len(p.CPUs) != coresPerNode {
			t.Fatalf("worker %d cpus = %v, want node %d's %d-core block", i, p.CPUs, wantNode, coresPerNode)
		}
	}
}

func TestScheduleDecodeNUMA_Remainder(t *testing.T) {
	// 10 workers over 4 nodes ⇒ 3,3,2,2 (first `10 mod 4 = 2` nodes take the extra).
	s := ScheduleDecodeNUMA(topo(4, 8), 10)
	if !s.Eligible {
		t.Fatalf("want eligible, got %q", s.Reason)
	}
	if want := []int{3, 3, 2, 2}; !reflect.DeepEqual(s.PerNodeWorkers, want) {
		t.Fatalf("per-node = %v, want %v", s.PerNodeWorkers, want)
	}
	total := 0
	for _, c := range s.PerNodeWorkers {
		total += c
	}
	if total != 10 {
		t.Fatalf("per-node counts sum to %d, want 10", total)
	}
}

func TestScheduleDecodeNUMA_Oversubscribed(t *testing.T) {
	// 20 workers over 4 nodes of 2 cores ⇒ 5 workers/node > 2 cores ⇒ oversubscribed,
	// but the placement still covers EVERY requested worker (the kernel time-slices
	// within the node) — that coverage relation is the invariant, not the count 20.
	const workers = 20
	s := ScheduleDecodeNUMA(topo(4, 2), workers)
	if !s.Eligible {
		t.Fatalf("want eligible, got %q", s.Reason)
	}
	if !s.Oversubscribed {
		t.Fatalf("5 workers over 2 cores/node must flag oversubscribed")
	}
	if len(s.Placements) != workers {
		t.Fatalf("want %d placements even when oversubscribed, got %d", workers, len(s.Placements))
	}
}

func TestScheduleDecodeNUMA_DeterministicUnderNodeReorder(t *testing.T) {
	// The schedule must be a function of the node SET, not the discovery order.
	forward := topo(4, 8)
	reversed := []NUMANodeTopology{forward[3], forward[2], forward[1], forward[0]}
	a := ScheduleDecodeNUMA(forward, 9)
	b := ScheduleDecodeNUMA(reversed, 9)
	if !reflect.DeepEqual(a.Placements, b.Placements) {
		t.Fatalf("schedule differs under node reorder:\n forward=%+v\n reversed=%+v", a.Placements, b.Placements)
	}
}

func TestScheduleDecodeNUMA_CPUSetIsolation(t *testing.T) {
	// Mutating a placement's CPUs must not corrupt the caller's topology.
	tp := topo(2, 4)
	s := ScheduleDecodeNUMA(tp, 2)
	s.Placements[0].CPUs[0] = -999
	if tp[0].CPUs[0] == -999 {
		t.Fatalf("placement CPUs must be an isolated copy, not alias the topology")
	}
}

func TestScheduleDecodeNUMA_Refusals(t *testing.T) {
	cases := []struct {
		name    string
		topo    []NUMANodeTopology
		workers int
		reason  DecodeNUMAScheduleReason
	}{
		{"zero workers", topo(4, 8), 0, DecodeNUMAScheduleInvalidWorkers},
		{"negative workers", topo(4, 8), -1, DecodeNUMAScheduleInvalidWorkers},
		{"single node", topo(1, 8), 8, DecodeNUMAScheduleInsufficientNodes},
		{"empty topology", nil, 8, DecodeNUMAScheduleInsufficientNodes},
		{"node without cpus", []NUMANodeTopology{{NodeID: 0, CPUs: []int{0}}, {NodeID: 1, CPUs: nil}}, 4, DecodeNUMAScheduleInvalidTopology},
		{"negative node id", []NUMANodeTopology{{NodeID: -1, CPUs: []int{0}}, {NodeID: 1, CPUs: []int{1}}}, 4, DecodeNUMAScheduleInvalidTopology},
		{"duplicate node id", []NUMANodeTopology{{NodeID: 2, CPUs: []int{0}}, {NodeID: 2, CPUs: []int{1}}}, 4, DecodeNUMAScheduleInvalidTopology},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := ScheduleDecodeNUMA(tc.topo, tc.workers)
			if s.Eligible {
				t.Fatalf("want refusal, got eligible")
			}
			if s.Reason != tc.reason {
				t.Fatalf("reason = %q, want %q", s.Reason, tc.reason)
			}
			if s.Placements != nil || s.PerNodeWorkers != nil {
				t.Fatalf("refusal must carry an empty plan, got placements=%v pernode=%v", s.Placements, s.PerNodeWorkers)
			}
		})
	}
}
