package blastlease_test

import (
	"context"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/blastlease"
	"github.com/anthony-chaudhary/fak/internal/blastradius"
	"github.com/anthony-chaudhary/fak/internal/knownbad"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

func TestComposability(t *testing.T) {
	tempDir := initTestRepo(t)
	store := leaseref.NewInDir(tempDir)
	ctx := context.Background()
	now := time.Unix(1_700_000_000, 0)

	// Step 1 & 2: Acquire multiple active leases on distinct trees using real leaseref store.
	leasesToAcquire := []leaseref.Record{
		{
			ID:          "gateway",
			TreeGlobs:   []string{"internal/gateway/**"},
			Holder:      "worker-1",
			AcquiredAt:  now.Unix(),
			TTLSeconds:  600,
			Description: "gateway worker",
		},
		{
			ID:          "engine",
			TreeGlobs:   []string{"internal/engine/**"},
			Holder:      "worker-2",
			AcquiredAt:  now.Unix(),
			TTLSeconds:  600,
			Description: "engine worker",
		},
		{
			ID:          "unrelated",
			TreeGlobs:   []string{"cmd/unrelated/**"},
			Holder:      "worker-3",
			AcquiredAt:  now.Unix(),
			TTLSeconds:  600,
			Description: "unrelated worker",
		},
	}

	for _, rec := range leasesToAcquire {
		if _, err := store.Acquire(ctx, rec); err != nil {
			t.Fatalf("store.Acquire(%q) failed: %v", rec.ID, err)
		}
	}

	// Step 3: Call blastlease.Live(tempDir, now)
	liveLeases, err := blastlease.Live(tempDir, now)
	if err != nil {
		t.Fatalf("blastlease.Live(%q, %v) failed: %v", tempDir, now, err)
	}

	if len(liveLeases) != 3 {
		t.Fatalf("blastlease.Live returned %d leases, want 3", len(liveLeases))
	}

	byLane := make(map[string]blastradius.Lease, len(liveLeases))
	for _, l := range liveLeases {
		byLane[l.Lane] = l
	}

	lGateway, ok := byLane["gateway"]
	if !ok {
		t.Fatalf("missing projected lease for lane 'gateway'")
	}
	lEngine, ok := byLane["engine"]
	if !ok {
		t.Fatalf("missing projected lease for lane 'engine'")
	}
	lUnrelated, ok := byLane["unrelated"]
	if !ok {
		t.Fatalf("missing projected lease for lane 'unrelated'")
	}

	// Step 4: Verify that knownbad.TreesIntersect([]string{"internal/gateway/**"}, lease.TreeGlobs)
	// matches Lease 1, does NOT match Lease 2 or 3.
	gatewayQuery := []string{"internal/gateway/**"}
	if !knownbad.TreesIntersect(gatewayQuery, lGateway.TreeGlobs) {
		t.Errorf("knownbad.TreesIntersect(%v, gateway.TreeGlobs %v) = false, want true", gatewayQuery, lGateway.TreeGlobs)
	}
	if knownbad.TreesIntersect(gatewayQuery, lEngine.TreeGlobs) {
		t.Errorf("knownbad.TreesIntersect(%v, engine.TreeGlobs %v) = true, want false", gatewayQuery, lEngine.TreeGlobs)
	}
	if knownbad.TreesIntersect(gatewayQuery, lUnrelated.TreeGlobs) {
		t.Errorf("knownbad.TreesIntersect(%v, unrelated.TreeGlobs %v) = true, want false", gatewayQuery, lUnrelated.TreeGlobs)
	}

	// Step 5: Verify that knownbad.TreesIntersect([]string{"internal/**"}, lease.TreeGlobs)
	// matches Lease 1 and 2, but not Lease 3.
	internalQuery := []string{"internal/**"}
	if !knownbad.TreesIntersect(internalQuery, lGateway.TreeGlobs) {
		t.Errorf("knownbad.TreesIntersect(%v, gateway.TreeGlobs %v) = false, want true", internalQuery, lGateway.TreeGlobs)
	}
	if !knownbad.TreesIntersect(internalQuery, lEngine.TreeGlobs) {
		t.Errorf("knownbad.TreesIntersect(%v, engine.TreeGlobs %v) = false, want true", internalQuery, lEngine.TreeGlobs)
	}
	if knownbad.TreesIntersect(internalQuery, lUnrelated.TreeGlobs) {
		t.Errorf("knownbad.TreesIntersect(%v, unrelated.TreeGlobs %v) = true, want false", internalQuery, lUnrelated.TreeGlobs)
	}

	// Step 6: Prove that the real pipeline composition (leaseref -> blastlease -> knownbad)
	// functions end-to-end.
	var matchedInternalLanes []string
	for _, l := range liveLeases {
		if knownbad.TreesIntersect(internalQuery, l.TreeGlobs) {
			matchedInternalLanes = append(matchedInternalLanes, l.Lane)
		}
	}
	sort.Strings(matchedInternalLanes)
	wantInternalLanes := []string{"engine", "gateway"}
	if !reflect.DeepEqual(matchedInternalLanes, wantInternalLanes) {
		t.Fatalf("pipeline matched internal lanes = %v, want %v", matchedInternalLanes, wantInternalLanes)
	}

	var matchedGatewayLanes []string
	for _, l := range liveLeases {
		if knownbad.TreesIntersect(gatewayQuery, l.TreeGlobs) {
			matchedGatewayLanes = append(matchedGatewayLanes, l.Lane)
		}
	}
	wantGatewayLanes := []string{"gateway"}
	if !reflect.DeepEqual(matchedGatewayLanes, wantGatewayLanes) {
		t.Fatalf("pipeline matched gateway lanes = %v, want %v", matchedGatewayLanes, wantGatewayLanes)
	}
}
