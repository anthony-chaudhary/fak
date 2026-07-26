package compute

import (
	"testing"
)

// eligiblePlan builds an eligible plan for `nodes` targets sized to replicaBytes, bypassing the
// host snapshot so the store contract is testable on any platform (the planner's own refusal
// logic is covered by numa_replica_plan_test.go).
func eligiblePlan(nodes int, replicaBytes int64) NUMAReplicaPlan {
	targets := make([]NUMAReplicaTarget, nodes)
	for i := range targets {
		targets[i] = NUMAReplicaTarget{NodeID: i, ReplicaBytes: replicaBytes, RequiredBytes: replicaBytes}
	}
	return NUMAReplicaPlan{
		Eligible:     true,
		Reason:       NUMAReplicaPlanEligible,
		ReplicaBytes: replicaBytes,
		Targets:      targets,
	}
}

func srcBytes(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i*31 + 7) // non-trivial pattern: a zeroed or misaligned copy cannot pass
	}
	return b
}

// TestBuildNUMAReplicas_BitIdentity is the load-bearing correctness contract: every replica must
// be byte-identical to the source, because the decode GEMV reads weights from whichever replica
// its node holds. A drifting replica would silently change model output.
func TestBuildNUMAReplicas_BitIdentity(t *testing.T) {
	src := srcBytes(4096 + 17) // deliberately not a page multiple
	set, err := BuildNUMAReplicas(src, eligiblePlan(4, int64(len(src))))
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer set.Free()

	if set.Len() != 4 {
		t.Fatalf("Len = %d, want 4", set.Len())
	}
	if err := VerifyNUMAReplicas(src, set); err != nil {
		t.Fatalf("bit-identity: %v", err)
	}
	for _, node := range set.Nodes() {
		r := set.For(node)
		if len(r) != len(src) {
			t.Fatalf("node %d replica len = %d, want %d", node, len(r), len(src))
		}
	}
}

// TestBuildNUMAReplicas_DistinctBackingStores pins that replicas are real copies, not aliases of
// the source or of each other — the whole point is N physically distinct copies on N nodes.
func TestBuildNUMAReplicas_DistinctBackingStores(t *testing.T) {
	src := srcBytes(1024)
	set, err := BuildNUMAReplicas(src, eligiblePlan(3, int64(len(src))))
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer set.Free()

	a, b := set.For(0), set.For(1)
	a[0] = 0xAA
	b[0] = 0xBB
	if src[0] == 0xAA || src[0] == 0xBB {
		t.Fatalf("replica aliases the source region")
	}
	if a[0] == b[0] {
		t.Fatalf("replicas on node 0 and 1 share a backing store")
	}
}

// TestBuildNUMAReplicas_NodeLookup pins For()'s node→replica mapping, including a miss.
func TestBuildNUMAReplicas_NodeLookup(t *testing.T) {
	src := srcBytes(512)
	plan := eligiblePlan(2, int64(len(src)))
	plan.Targets[0].NodeID = 2 // ascending but not zero-based
	plan.Targets[1].NodeID = 5
	set, err := BuildNUMAReplicas(src, plan)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer set.Free()

	if got := set.Nodes(); len(got) != 2 || got[0] != 2 || got[1] != 5 {
		t.Fatalf("Nodes = %v, want [2 5]", got)
	}
	if set.For(2) == nil || set.For(5) == nil {
		t.Fatalf("For must return a replica for every planned node")
	}
	if set.For(3) != nil {
		t.Fatalf("For(3) must be nil — no replica planned on that node")
	}
}

// TestBuildNUMAReplicas_Refusals pins that a refused/mismatched plan never yields a partial set.
func TestBuildNUMAReplicas_Refusals(t *testing.T) {
	src := srcBytes(256)
	cases := []struct {
		name string
		src  []byte
		plan NUMAReplicaPlan
	}{
		{"empty source", nil, eligiblePlan(2, 256)},
		{"ineligible plan", src, NUMAReplicaPlan{Eligible: false, Reason: NUMAReplicaPlanConstrainedPolicy}},
		{"eligible but no targets", src, NUMAReplicaPlan{Eligible: true, Reason: NUMAReplicaPlanEligible, ReplicaBytes: 256}},
		{"size disagreement", src, eligiblePlan(2, 999)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			set, err := BuildNUMAReplicas(tc.src, tc.plan)
			if err == nil {
				set.Free()
				t.Fatalf("want refusal, got a replica set")
			}
			if set != nil {
				t.Fatalf("a refusal must return a nil set, got %+v", set)
			}
		})
	}
}

// TestNUMAReplicaSet_NilSafe pins the zero/nil contract the decode path relies on when the
// planner refuses: every accessor degrades quietly rather than panicking in the hot path.
func TestNUMAReplicaSet_NilSafe(t *testing.T) {
	var s *NUMAReplicaSet
	if s.Len() != 0 || s.Bound() != 0 || s.Nodes() != nil || s.For(0) != nil {
		t.Fatalf("nil set accessors must be zero-valued")
	}
	if s.Label() != "replicas=none" {
		t.Fatalf("nil Label = %q, want replicas=none", s.Label())
	}
	if err := s.Free(); err != nil {
		t.Fatalf("nil Free = %v, want nil", err)
	}
}
