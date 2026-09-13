package deepseekv4moe

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func mustSyntheticStream(t *testing.T, name string, seed int64, layers, experts, topK, accesses int) CachePolicyStream {
	t.Helper()
	stream, err := SyntheticCachePolicyStream(name, seed, layers, experts, topK, accesses)
	if err != nil {
		t.Fatalf("SyntheticCachePolicyStream(%q) error = %v", name, err)
	}
	return stream
}

func mustReplay(t *testing.T, stream CachePolicyStream, capacity int) CachePolicyBenchReceipt {
	t.Helper()
	receipt, err := CachePolicyReplay(stream, capacity)
	if err != nil {
		t.Fatalf("CachePolicyReplay(%q, %d) error = %v", stream.Name, capacity, err)
	}
	return receipt
}

func TestCachePolicyBenchAllArms(t *testing.T) {
	stream := mustSyntheticStream(t, "zipf", 7, 4, 32, 2, 2000)
	receipt := mustReplay(t, stream, 64)

	if len(receipt.Rows) != 5 {
		t.Fatalf("len(receipt.Rows) = %d, want 5", len(receipt.Rows))
	}
	wantArms := CachePolicyArms()
	if len(wantArms) != len(receipt.Rows) {
		t.Fatalf("len(CachePolicyArms()) = %d, want %d", len(wantArms), len(receipt.Rows))
	}
	for i, want := range wantArms {
		if receipt.Rows[i].Policy != want {
			t.Fatalf("row %d policy = %v, want %v", i, receipt.Rows[i].Policy, want)
		}
	}
	if receipt.Schema != CachePolicyBenchSchema {
		t.Fatalf("receipt.Schema = %q, want %q", receipt.Schema, CachePolicyBenchSchema)
	}
	if receipt.Measurement != CachePolicyMeasurementSimulated {
		t.Fatalf("receipt.Measurement = %q, want %q", receipt.Measurement, CachePolicyMeasurementSimulated)
	}

	for _, row := range receipt.Rows {
		if row.Accesses != len(stream.Groups) {
			t.Fatalf("policy %v Accesses = %d, want %d", row.Policy, row.Accesses, len(stream.Groups))
		}
		if row.Hits+row.Misses != row.Accesses {
			t.Fatalf("policy %v Hits(%d)+Misses(%d) != Accesses(%d)", row.Policy, row.Hits, row.Misses, row.Accesses)
		}
		if !row.HitRateKnown {
			t.Fatalf("policy %v HitRateKnown = false, want true", row.Policy)
		}
	}

	opt := receipt.Rows[len(receipt.Rows)-1]
	if opt.Policy != CachePolicyOPT {
		t.Fatalf("last row policy = %v, want OPT", opt.Policy)
	}
	if diff := opt.VersusOPT - 1.0; diff < -1e-9 || diff > 1e-9 {
		t.Fatalf("OPT VersusOPT = %v, want 1.0", opt.VersusOPT)
	}
}

func TestCachePolicyBenchDeterministic(t *testing.T) {
	stream := mustSyntheticStream(t, "zipf-det", 7, 4, 32, 2, 2000)
	r1 := mustReplay(t, stream, 64)
	r2 := mustReplay(t, stream, 64)
	if !reflect.DeepEqual(r1, r2) {
		t.Fatalf("replay not deterministic:\nfirst  %+v\nsecond %+v", r1, r2)
	}
}

func TestCachePolicyBenchOPTBound(t *testing.T) {
	stream := mustSyntheticStream(t, "zipf-opt", 7, 4, 32, 2, 2000)
	receipt := mustReplay(t, stream, 32)
	for _, row := range receipt.Rows {
		if row.Policy == CachePolicyOPT {
			continue
		}
		if row.Hits > receipt.OPTHits {
			t.Fatalf("policy %v Hits = %d exceeds OPT hits %d", row.Policy, row.Hits, receipt.OPTHits)
		}
		if row.VersusOPT > 1.0+1e-9 {
			t.Fatalf("policy %v VersusOPT = %v > 1.0", row.Policy, row.VersusOPT)
		}
	}
	// The 32-expert stream has more than 63 distinct groups, so the oracle degrades to the
	// farthest-next-use heuristic. The upper bound still holds, but the receipt must say so
	// rather than letting callers read "vs OPT" as an exact Belady margin.
	if distinctGroups(stream) <= 63 {
		t.Fatalf("test setup: stream has %d distinct groups, expected >63 to exercise the fallback", distinctGroups(stream))
	}
	if receipt.OPTExact {
		t.Fatalf("receipt.OPTExact = true on a >63-distinct-group stream, want false (greedy fallback)")
	}
}

func TestCachePolicyBenchExactOPTUpperBound(t *testing.T) {
	// A <=63-distinct-group stream takes the cursor's exact Belady DP path, so OPTExact is
	// true and VersusOPT is a genuine offline-optimum margin, not a greedy approximation.
	stream := mustSyntheticStream(t, "exact-opt", 5, 2, 8, 1, 40)
	receipt := mustReplay(t, stream, 3)
	if !receipt.OPTExact {
		t.Fatalf("receipt.OPTExact = false on a %d-group stream, want true", distinctGroups(stream))
	}
	for _, row := range receipt.Rows {
		if row.Policy == CachePolicyOPT {
			continue
		}
		if row.Hits > receipt.OPTHits {
			t.Fatalf("policy %v Hits = %d exceeds exact OPT hits %d", row.Policy, row.Hits, receipt.OPTHits)
		}
	}
	if got := receipt.Rows[len(receipt.Rows)-1].VersusOPT; got < 1.0-1e-9 || got > 1.0+1e-9 {
		t.Fatalf("OPT VersusOPT = %v, want 1.0", got)
	}
}

// distinctGroups counts the unique (layer, expert) groups in a stream.
func distinctGroups(stream CachePolicyStream) int {
	seen := make(map[ExpertGroup]struct{}, len(stream.Groups))
	for _, g := range stream.Groups {
		seen[g] = struct{}{}
	}
	return len(seen)
}

func TestCachePolicyBenchLRUMatchesShipped(t *testing.T) {
	// The LRU-vs-LRU equivalence needs no OPT oracle; use the oracle-free path so the
	// test does not pay compute.BeladyKVReplayOracle's exponential exact DP (which cost
	// ~30s for these 32 distinct groups and proved nothing about LRU).
	stream := mustSyntheticStream(t, "lru-check", 3, 2, 16, 1, 500)
	rows, err := CachePolicyReplayOnline(stream, 4)
	if err != nil {
		t.Fatalf("CachePolicyReplayOnline error = %v", err)
	}

	routes := make([]ExpertRoute, 0, len(stream.Groups))
	for _, g := range stream.Groups {
		routes = append(routes, ExpertRoute{Layer: g.Layer, Experts: []int{g.Expert}})
	}
	trace, err := SimulateExpertCache(routes, 4, 2, 16, 1)
	if err != nil {
		t.Fatalf("SimulateExpertCache error = %v", err)
	}

	var lru *CachePolicyRow
	for i := range rows {
		if rows[i].Policy == CachePolicyLRU {
			lru = &rows[i]
			break
		}
	}
	if lru == nil {
		t.Fatalf("rows have no LRU row: %+v", rows)
	}
	if lru.Hits != int(trace.Hits) {
		t.Fatalf("LRU Hits = %d, SimulateExpertCache Hits = %d", lru.Hits, int(trace.Hits))
	}
	if lru.Misses != int(trace.PageIns) {
		t.Fatalf("LRU Misses = %d, SimulateExpertCache PageIns = %d", lru.Misses, int(trace.PageIns))
	}
}

func TestCachePolicyMarginCapacitySensitivity(t *testing.T) {
	// Re-tests the upstream X8/Y7 claim that capacity, not eviction policy,
	// sets the routed-expert hit rate: at a generous capacity the whole working
	// set fits and the per-policy spread collapses, while a tight cache leaves
	// the policies diverging.
	stream := mustSyntheticStream(t, "sensitivity", 11, 6, 64, 2, 4000)
	tightCapacity := 8
	if compute.DefaultSmallBatchExpertBudget < tightCapacity {
		tightCapacity = compute.DefaultSmallBatchExpertBudget
	}
	tight := mustReplay(t, stream, tightCapacity)
	generous := mustReplay(t, stream, 512)

	tightMargin := CachePolicyMargin(tight)
	generousMargin := CachePolicyMargin(generous)
	if generousMargin > tightMargin+1e-9 {
		t.Fatalf("generous margin %v > tight margin %v; policy spread should collapse as capacity grows", generousMargin, tightMargin)
	}
	if generousMargin >= 0.05 {
		t.Fatalf("generous capacity margin = %v, want < 0.05 (capacity should set the hit rate)", generousMargin)
	}
}

func TestCasePolicyBenchInvalidInputs(t *testing.T) {
	tests := []struct {
		name string
		run  func() error
		want error
	}{
		{
			name: "zero capacity",
			run: func() error {
				_, err := CachePolicyReplay(CachePolicyStream{Name: "bad"}, 0)
				return err
			},
			want: ErrInvalidTraceCapacity,
		},
		{
			name: "top-k exceeds experts",
			run: func() error {
				_, err := SyntheticCachePolicyStream("bad-topk", 1, 2, 4, 8, 100)
				return err
			},
			want: ErrInvalidTraceShape,
		},
		{
			name: "zero accesses",
			run: func() error {
				_, err := SyntheticCachePolicyStream("bad-access", 1, 2, 4, 1, 0)
				return err
			},
			want: ErrInvalidTraceShape,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.run(); !errors.Is(err, tc.want) {
				t.Fatalf("error = %v, want %v", err, tc.want)
			}
		})
	}
}

func TestCasePolicyBenchEmptyStream(t *testing.T) {
	receipt, err := CachePolicyReplay(CachePolicyStream{Name: "empty"}, 4)
	if err != nil {
		t.Fatalf("CachePolicyReplay(empty) error = %v", err)
	}
	if len(receipt.Rows) != len(CachePolicyArms()) {
		t.Fatalf("len(receipt.Rows) = %d, want %d", len(receipt.Rows), len(CachePolicyArms()))
	}
	for _, row := range receipt.Rows {
		if row.HitRateKnown {
			t.Fatalf("policy %v HitRateKnown = true, want false on empty stream", row.Policy)
		}
		if row.HitRate != 0 {
			t.Fatalf("policy %v HitRate = %v, want 0 (no phantom rate)", row.Policy, row.HitRate)
		}
		if row.Accesses != 0 {
			t.Fatalf("policy %v Accesses = %d, want 0", row.Policy, row.Accesses)
		}
	}
}
