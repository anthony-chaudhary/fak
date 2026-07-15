package compute

import "testing"

func TestPlanNUMAReplicasExactDeterministicPlan(t *testing.T) {
	snapshot := numaReplicaSnapshot{
		policy:        numaReplicaPolicyUnconstrained,
		topologyKnown: true,
		nodes: []numaReplicaNode{
			{id: 3, freeBytes: 96, memoryKnown: true},
			{id: 1, freeBytes: 80, memoryKnown: true},
			{id: 2, freeBytes: 72, memoryKnown: true},
		},
	}
	got := planNUMAReplicas(snapshot, 32, 8)
	if !got.Eligible || got.Reason != NUMAReplicaPlanEligible {
		t.Fatalf("plan refused: %+v", got)
	}
	if got.RequiredPerNodeBytes != 40 || got.TotalReplicaBytes != 96 || got.TotalRequiredBytes != 120 {
		t.Fatalf("plan totals are not exact: %+v", got)
	}
	if len(got.Targets) != 3 {
		t.Fatalf("target count = %d, want 3", len(got.Targets))
	}
	wantIDs := []int{1, 2, 3}
	wantFree := []int64{80, 72, 96}
	for i, target := range got.Targets {
		if target.NodeID != wantIDs[i] || target.FreeBytes != wantFree[i] ||
			target.ReplicaBytes != 32 || target.ReserveBytes != 8 || target.RequiredBytes != 40 {
			t.Fatalf("target[%d] = %+v, want node=%d free=%d replica=32 reserve=8 required=40", i, target, wantIDs[i], wantFree[i])
		}
	}
}

// TestPlanNUMAReplicasRejectsStarvedNodeDespiteAggregateCapacity is the observed
// failure shape without embedding any host identity: total free memory is ample,
// but one target cannot meet its own replica-plus-reserve safety floor.
func TestPlanNUMAReplicasRejectsStarvedNodeDespiteAggregateCapacity(t *testing.T) {
	snapshot := eligibleNUMAReplicaSnapshot(
		numaReplicaNode{id: 0, freeBytes: 72, memoryKnown: true},
		numaReplicaNode{id: 1, freeBytes: 168, memoryKnown: true},
	)
	got := planNUMAReplicas(snapshot, 64, 16) // aggregate 240 >= 2*(64+16), node 0 < 80
	assertNUMAReplicaRefusal(t, got, NUMAReplicaPlanNodeBelowFloor)
}

func TestPlanNUMAReplicasFailClosedInputs(t *testing.T) {
	tests := []struct {
		name     string
		snapshot numaReplicaSnapshot
		replica  int64
		reserve  int64
		reason   NUMAReplicaPlanReason
	}{
		{
			name:     "unsupported policy",
			snapshot: numaReplicaSnapshot{policy: numaReplicaPolicyUnsupported, topologyKnown: true},
			replica:  1,
			reason:   NUMAReplicaPlanUnsupported,
		},
		{
			name:     "unknown topology",
			snapshot: numaReplicaSnapshot{policy: numaReplicaPolicyUnconstrained},
			replica:  1,
			reason:   NUMAReplicaPlanUnsupported,
		},
		{
			name: "constrained policy",
			snapshot: numaReplicaSnapshot{
				policy:        numaReplicaPolicyConstrained,
				topologyKnown: true,
				nodes:         []numaReplicaNode{{id: 0, freeBytes: 8, memoryKnown: true}, {id: 1, freeBytes: 8, memoryKnown: true}},
			},
			replica: 1,
			reason:  NUMAReplicaPlanConstrainedPolicy,
		},
		{
			name:     "one CPU-bearing node",
			snapshot: eligibleNUMAReplicaSnapshot(numaReplicaNode{id: 0, freeBytes: 8, memoryKnown: true}),
			replica:  1,
			reason:   NUMAReplicaPlanInsufficientNodes,
		},
		{
			name: "unknown node memory",
			snapshot: eligibleNUMAReplicaSnapshot(
				numaReplicaNode{id: 0, freeBytes: 8, memoryKnown: true},
				numaReplicaNode{id: 1, memoryKnown: false},
			),
			replica: 1,
			reason:  NUMAReplicaPlanUnknownNodeMemory,
		},
		{
			name: "duplicate node identity",
			snapshot: eligibleNUMAReplicaSnapshot(
				numaReplicaNode{id: 0, freeBytes: 8, memoryKnown: true},
				numaReplicaNode{id: 0, freeBytes: 8, memoryKnown: true},
			),
			replica: 1,
			reason:  NUMAReplicaPlanUnsupported,
		},
		{
			name:     "zero replica",
			snapshot: eligibleNUMAReplicaSnapshot(),
			replica:  0,
			reason:   NUMAReplicaPlanInvalidSize,
		},
		{
			name:     "negative reserve",
			snapshot: eligibleNUMAReplicaSnapshot(),
			replica:  1,
			reserve:  -1,
			reason:   NUMAReplicaPlanInvalidSize,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assertNUMAReplicaRefusal(t, planNUMAReplicas(tc.snapshot, tc.replica, tc.reserve), tc.reason)
		})
	}
}

func TestPlanNUMAReplicasRejectsArithmeticOverflow(t *testing.T) {
	const maxInt64 = int64(maxInt64Uint64)
	roomy := eligibleNUMAReplicaSnapshot(
		numaReplicaNode{id: 0, freeBytes: maxInt64, memoryKnown: true},
		numaReplicaNode{id: 1, freeBytes: maxInt64, memoryKnown: true},
		numaReplicaNode{id: 2, freeBytes: maxInt64, memoryKnown: true},
	)
	t.Run("per-node addition", func(t *testing.T) {
		assertNUMAReplicaRefusal(t, planNUMAReplicas(roomy, maxInt64, 1), NUMAReplicaPlanArithmeticOverflow)
	})
	t.Run("aggregate multiplication", func(t *testing.T) {
		assertNUMAReplicaRefusal(t, planNUMAReplicas(roomy, maxInt64/2, 0), NUMAReplicaPlanArithmeticOverflow)
	})
}

func eligibleNUMAReplicaSnapshot(nodes ...numaReplicaNode) numaReplicaSnapshot {
	return numaReplicaSnapshot{policy: numaReplicaPolicyUnconstrained, topologyKnown: true, nodes: nodes}
}

func assertNUMAReplicaRefusal(t *testing.T, got NUMAReplicaPlan, reason NUMAReplicaPlanReason) {
	t.Helper()
	if got.Eligible || got.Reason != reason {
		t.Fatalf("got eligible=%v reason=%q, want refusal reason=%q: %+v", got.Eligible, got.Reason, reason, got)
	}
	if got.Targets != nil {
		t.Fatalf("refusal exposed a partial target plan: %+v", got.Targets)
	}
}
