package model

import (
	"math"
	"testing"
)

// Admission-only preflight must match the ordinary gather plan's refusal and
// boundary decisions while eliminating its discarded heap-backed shard table per token.
// fak-test:runtime fast est=10ms lane=default
func TestV41EPPrefillGatherAdmissionAllocations(t *testing.T) {
	owners, err := ExpertParallelPlan(10, 3)
	if err != nil {
		t.Fatal(err)
	}
	balanced := []routePick{{expert: 0, weight: 1}, {expert: 5, weight: 0}, {expert: 9, weight: -1}}
	skewed := []routePick{{expert: 9, weight: 1}, {expert: 9, weight: 2}, {expert: 9, weight: 3}}
	limit := (v41EPMaxWords - len(owners.Shards)*v41EPHeaderWords) / len(balanced)
	badOwners := TPPlan{Dim: owners.Dim, Shards: append([]TPShard(nil), owners.Shards...)}
	badOwners.Shards[1].Lo++
	overlapOwners := TPPlan{Dim: owners.Dim, Shards: append([]TPShard(nil), owners.Shards...)}
	overlapOwners.Shards[1].Lo--
	for _, tc := range []struct {
		name   string
		owners TPPlan
		picks  []routePick
		width  int
	}{
		{"empty", owners, nil, 5120},
		{"balanced", owners, balanced, 5120},
		{"duplicates", owners, skewed, 5120},
		{"last-fitting-width", owners, balanced, limit},
		{"overflow", owners, skewed, limit + 1},
		{"gap-owners", badOwners, balanced, 5120},
		{"overlap-owners", overlapOwners, balanced, 5120},
		{"empty-owners", TPPlan{Dim: owners.Dim}, balanced, 5120},
		{"zero-width", owners, balanced, 0},
		{"large-width", owners, balanced, int(^uint(0) >> 1)},
		{"negative-expert", owners, []routePick{{expert: -1, weight: 1}}, 5120},
		{"outside-expert", owners, []routePick{{expert: owners.Dim, weight: 1}}, 5120},
		{"nonfinite", owners, []routePick{{expert: 0, weight: float32(math.NaN())}}, 5120},
		{"invalid-before-overflow", owners, []routePick{{expert: 0, weight: 1}, {expert: 1, weight: 1}, {expert: 2, weight: float32(math.Inf(1))}}, limit + 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, want := v41EPGatherPlan(tc.picks, tc.owners, tc.width)
			got := v41EPPrefillGatherAdmitted(tc.picks, tc.owners, tc.width)
			if (got == nil) != (want == nil) || got != nil && got.Error() != want.Error() {
				t.Fatalf("admission=%v, ordinary plan=%v", got, want)
			}
		})
	}
	var plan TPPlan
	before := testing.AllocsPerRun(100, func() { plan, err = v41EPGatherPlan(skewed, owners, 5120) })
	if err != nil || len(plan.Shards) != len(owners.Shards) {
		t.Fatalf("ordinary plan=%+v err=%v", plan, err)
	}
	after := testing.AllocsPerRun(100, func() { err = v41EPPrefillGatherAdmitted(skewed, owners, 5120) })
	if err != nil || after != 0 || before != 1 {
		t.Fatalf("ordinary=%g admission=%g allocations, err=%v; want one removed for this three-owner fixture", before, after, err)
	}
	t.Logf("ordinary=%g admission=%g allocations per token", before, after)
}

// Window selection needs the complete geometry admission but never the
// returned per-owner request slice used by transport.
// fak-test:runtime fast est=10ms lane=default
func TestV41EPPanelAdmissionAllocations(t *testing.T) {
	owners, err := ExpertParallelPlan(10, 3)
	if err != nil {
		t.Fatal(err)
	}
	picks := make([][]routePick, v41EPOwnerPanelTokens)
	for i := range picks {
		picks[i] = []routePick{{expert: 9, weight: 1}, {expert: 9, weight: 2}}
	}
	const width = 5120
	ordinary, err := v41EPPanelPlan(owners, picks, width)
	if err != nil {
		t.Fatal(err)
	}
	admitted, err := v41EPPanelAdmitted(owners, picks, width)
	if err != nil || admitted.requests != nil || admitted.payloadWords != ordinary.payloadWords || admitted.replyWords != ordinary.replyWords || admitted.liveBytes != ordinary.liveBytes {
		t.Fatalf("admission=%+v ordinary=%+v err=%v", admitted, ordinary, err)
	}
	base := v41EPPanelHeaderWords + len(picks)*v41EPHeaderWords
	if len(ordinary.requests) != 3 || ordinary.requests[0] != base || ordinary.requests[1] != base || ordinary.requests[2] != base+len(picks)*2*width {
		t.Fatalf("skewed request lengths changed: %v", ordinary.requests)
	}
	before := testing.AllocsPerRun(100, func() { ordinary, err = v41EPPanelPlan(owners, picks, width) })
	after := testing.AllocsPerRun(100, func() { admitted, err = v41EPPanelAdmitted(owners, picks, width) })
	if err != nil || before != 1 || after != 0 {
		t.Fatalf("ordinary=%g admission=%g allocations, err=%v", before, after, err)
	}
	last := -1
	window := testing.AllocsPerRun(100, func() { last = v41EPPrefillWindowEnd(owners, picks, width, 0) })
	if last != len(picks) || window != 0 {
		t.Fatalf("window end=%d allocations=%g", last, window)
	}
	picks[0][0].expert = owners.Dim
	if _, err := v41EPPanelAdmitted(owners, picks, width); err == nil {
		t.Fatal("invalid pick was admitted")
	}
	t.Logf("ordinary=%g admission=%g window=%g allocations", before, after, window)
}
