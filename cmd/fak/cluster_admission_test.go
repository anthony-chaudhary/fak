// cluster_admission_test.go — pins the converged-round admission planner
// ADAPTed from TensorFold zig/src/cluster/admission.zig @9e8c368b0b4c2c7892a0c9f0f302d7c164016209
// (Apache-2.0; see THIRD_PARTY_NOTICES.md): the per-rank fitsAt budget check, the
// monotone maxStreams/maxContext searches, the over-budget refusal sentinel, and
// the 16-step chunkRows share bound. Closed contracts only: exact values and
// errors.Is on the sentinel, never error prose.

package main

import (
	"errors"
	"testing"
)

// TestClusterAdmissionFitsAtAndChunkRows is the contract witness for the
// admission planner. Two ranks; rank 1 binds. Per rank: need = weights + static
// + margin + kv*streams*context + state*streams. Rank 0 fixed need
// 500_010_000 against a 1_000_000_000 limit; rank 1 fixed need 450_010_000
// against 900_000_000, so the binding product is streams*context <= 449_990.
func TestClusterAdmissionFitsAtAndChunkRows(t *testing.T) {
	ranks := []admissionRank{
		{GPULimit: 1_000_000_000, Weights: 400_000_000, Margin: 100_000_000, KVPerToken: 1_000, StatePerStream: 10_000},
		{GPULimit: 900_000_000, Weights: 350_000_000, Margin: 100_000_000, KVPerToken: 1_000, StatePerStream: 10_000},
	}

	// fitsAt: the exact boundary and the tight rank.
	if !fitsAt(ranks, 4, 100_000) {
		t.Fatal("fitsAt(4, 100000) = false, want true (both ranks inside their limits)")
	}
	if fitsAt(ranks, 5, 100_000) {
		t.Fatal("fitsAt(5, 100000) = true, want false (rank 1 over budget)")
	}
	if !fitsAt(ranks, 1, 449_990) {
		t.Fatal("fitsAt(1, 449990) = false, want true (rank 1 exactly at its limit)")
	}
	if fitsAt(ranks, 1, 449_991) {
		t.Fatal("fitsAt(1, 449991) = true, want false (rank 1 one token over)")
	}

	// maxStreams: exact value, and monotonicity down to zero at the binding context.
	if got := maxStreams(ranks, 100_000); got != 4 {
		t.Fatalf("maxStreams(100000) = %d, want 4", got)
	}
	for s := int64(0); s <= 4; s++ {
		if !fitsAt(ranks, s, 100_000) {
			t.Fatalf("monotone break: fitsAt(%d, 100000) = false below the admitted count", s)
		}
	}

	// maxContext: the memory ceiling when uncapped, then the model cap.
	if got := maxContext(ranks, admissionShape{MaxContext: 500_000}, 1); got != 449_990 {
		t.Fatalf("maxContext(1) = %d, want 449990 (memory ceiling)", got)
	}
	if got := maxContext(ranks, admissionShape{MaxContext: 100_000}, 4); got != 100_000 {
		t.Fatalf("maxContext(4) = %d, want 100000 (model cap)", got)
	}

	// chunkRows: linear round cost 10ms/decode-row + 1ms/prompt-row, decode = 4.
	linear := func(decode, prompt int64) float64 { return float64(decode)*10 + float64(prompt) }
	if got := chunkRows(linear, 4, 0.25); got != 16 {
		t.Fatalf("chunkRows(share=0.25) = %d, want 16 (floor: the next step already breaks the bound)", got)
	}
	if got := chunkRows(linear, 4, 1.0); got != 32 {
		t.Fatalf("chunkRows(share=1.0) = %d, want 32 (the 48-row step exceeds base*2)", got)
	}
	if got := chunkRows(linear, 4, 1e9); got != 4096 {
		t.Fatalf("chunkRows(huge share) = %d, want 4096 (cap)", got)
	}
	// Share bound honored: below the cap, the returned step's NEXT step violates
	// the bound.
	if c := chunkRows(linear, 4, 1.0); c < 4096 && linear(4, c+16) <= 40*(1+1.0) {
		t.Fatalf("chunkRows = %d does not honor the share bound (next step still fits)", c)
	}

	// Over-budget refusal: even one stream at context 1 must fit or the planner
	// refuses with the closed sentinel.
	tight := []admissionRank{
		{GPULimit: 450_000_000, Weights: 350_000_000, Margin: 100_000_000, KVPerToken: 1_000, StatePerStream: 10_000},
	}
	if got := maxStreams(tight, 1); got != 0 {
		t.Fatalf("maxStreams(tight) = %d, want 0 (even one stream does not fit)", got)
	}
	cfg := admissionConfig{
		Widths:         []int{1},
		Limits:         []int64{450_000_000},
		TotalWeights:   350_000_000,
		Margin:         100_000_000,
		KVPerToken:     1_000,
		StatePerStream: 10_000,
		MaxContext:     100_000,
		Share:          0.25,
		DecodeMS:       10,
		PromptMS:       1,
	}
	if _, err := planAdmission(cfg); !errors.Is(err, errAdmissionOverBudget) {
		t.Fatalf("planAdmission(tight) err = %v, want errAdmissionOverBudget via errors.Is", err)
	}
}

// TestClusterAdmissionPlanEndToEnd pins the planner policy: admit the longest
// context one stream can hold, then the most streams that survive at that
// context, with the tiling plan built through planFromWidths.
func TestClusterAdmissionPlanEndToEnd(t *testing.T) {
	cfg := admissionConfig{
		Widths:         []int{1, 1},
		Limits:         []int64{1_000_000_000, 900_000_000},
		TotalWeights:   750_000_000, // 375_000_000 per rank
		Margin:         100_000_000,
		KVPerToken:     1_000,
		StatePerStream: 10_000,
		MaxContext:     100_000,
		Share:          0.25,
		DecodeMS:       10,
		PromptMS:       1,
	}
	ap, err := planAdmission(cfg)
	if err != nil {
		t.Fatalf("planAdmission: %v", err)
	}
	// Fixed need 475_010_000 per rank; binding rank 1 gives streams*context <=
	// 424_990. Context caps at 100_000 -> 4 streams; chunk floor 16.
	if ap.Streams != 4 || ap.Context != 100_000 || ap.ChunkRows != 16 {
		t.Fatalf("plan = streams %d context %d chunk %d, want 4 / 100000 / 16", ap.Streams, ap.Context, ap.ChunkRows)
	}
	if ap.Plan.Dim != 2 || len(ap.Plan.Shards) != 2 {
		t.Fatalf("plan tiling = %+v, want dim=2 with 2 shards", ap.Plan)
	}
}

// TestClusterAdmissionShardWeights pins the conservative weight split: each rank
// gets ceil(total*width/dim), so no rank is under-accounted.
func TestClusterAdmissionShardWeights(t *testing.T) {
	got := admissionShardWeights(100, []int{1, 2})
	if got[0] != 34 || got[1] != 67 {
		t.Fatalf("shardWeights(100, [1,2]) = %v, want [34 67]", got)
	}
	if got := admissionShardWeights(0, []int{1, 1}); got[0] != 0 || got[1] != 0 {
		t.Fatalf("shardWeights(0) = %v, want zeros", got)
	}
}
