package model

import (
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// collective_bridge_test.go — the gates for BackendCollective (collective_bridge.go), the
// model→HAL collective adapter that de-risks the native-753B Pillar-3 multi-GPU seam. They
// pin two things, both with NO multi-GPU hardware (the cpu-ref CollectiveBackend is the only
// backend behind the seam today):
//
//   - BackendCollective == LocalCollective byte-for-byte (max|Δ|=0) for AllReduceSum and
//     AllGather, so routing the in-process TP primitives through the HAL collective changes
//     the communicator, not the math;
//   - ForwardTP driven by BackendCollective == ForwardTP driven by LocalCollective at
//     max|Δ|=0 across rank combos — the end-to-end "the bridge is a faithful drop-in" rung;
//   - the fail-closed contract (nil/non-collective backend at construction; empty, ragged,
//     and mis-width parts at reduce) holds, so swapping the communicator never loosens it.

func bridgeRandVec(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}
	return v
}

func mustBackendColl(t *testing.T) *BackendCollective {
	t.Helper()
	bc, err := NewBackendCollective(compute.Default())
	if err != nil {
		t.Fatalf("NewBackendCollective(cpu-ref): %v", err)
	}
	return bc
}

// TestBackendCollectiveMatchesLocal pins the bridge byte-for-byte against LocalCollective for
// both methods, over several rank counts. AllReduceSum uses equal-length partials; AllGather
// uses each rank's plan-shard width (NewTPPlan gives near-even, deliberately uneven, bands).
// max|Δ|=0 — a reduction that reordered, dropped, or double-counted a rank would be caught.
func TestBackendCollectiveMatchesLocal(t *testing.T) {
	bc := mustBackendColl(t)
	var local LocalCollective
	rng := rand.New(rand.NewSource(485))

	for _, ranks := range []int{1, 2, 3, 5} {
		// AllReduceSum: equal-length partials (a non-round length exercises fdot's tail).
		const n = 17
		parts := make([][]float32, ranks)
		for r := range parts {
			parts[r] = bridgeRandVec(rng, n)
		}
		wantR, err := local.AllReduceSum(parts)
		if err != nil {
			t.Fatalf("local AllReduceSum ranks=%d: %v", ranks, err)
		}
		gotR, err := bc.AllReduceSum(parts)
		if err != nil {
			t.Fatalf("bridge AllReduceSum ranks=%d: %v", ranks, err)
		}
		if len(gotR) != len(wantR) {
			t.Fatalf("AllReduceSum ranks=%d len = %d, want %d", ranks, len(gotR), len(wantR))
		}
		for i := range wantR {
			if gotR[i] != wantR[i] {
				t.Fatalf("AllReduceSum ranks=%d [%d] = %v, want %v (not bit-exact vs LocalCollective)", ranks, i, gotR[i], wantR[i])
			}
		}

		// AllGather: rank-shard widths from a real plan over a dim divisible-ish by ranks.
		dim := 6 * ranks // > ranks so NewTPPlan is valid; uneven only when ranks ∤ dim, but covers both
		plan, err := NewTPPlan(dim, ranks)
		if err != nil {
			t.Fatalf("NewTPPlan(%d,%d): %v", dim, ranks, err)
		}
		gparts := make([][]float32, ranks)
		for r, s := range plan.Shards {
			gparts[r] = bridgeRandVec(rng, s.Width())
		}
		wantG, err := local.AllGather(gparts, plan)
		if err != nil {
			t.Fatalf("local AllGather ranks=%d: %v", ranks, err)
		}
		gotG, err := bc.AllGather(gparts, plan)
		if err != nil {
			t.Fatalf("bridge AllGather ranks=%d: %v", ranks, err)
		}
		if len(gotG) != len(wantG) {
			t.Fatalf("AllGather ranks=%d len = %d, want %d", ranks, len(gotG), len(wantG))
		}
		for i := range wantG {
			if gotG[i] != wantG[i] {
				t.Fatalf("AllGather ranks=%d [%d] = %v, want %v (not bit-exact vs LocalCollective)", ranks, i, gotG[i], wantG[i])
			}
		}
	}
}

// TestForwardTPViaBackendCollective is the end-to-end gate: a full ForwardTP whose collective
// is the HAL bridge reproduces the same ForwardTP driven by LocalCollective bit-for-bit
// (max|Δ|=0) across rank combos. The bridge swaps the communicator, not the reduction order,
// so the recombined logits are byte-identical — the proof a real CollectiveBackend can be
// dropped in behind ForwardTP without moving a number.
func TestForwardTPViaBackendCollective(t *testing.T) {
	m := NewSynthetic(tpFwdBaseCfg())
	ids := []int{1, 2, 3, 4, 5, 6}
	bc := mustBackendColl(t)
	combos := []struct{ attn, ffn int }{{1, 1}, {2, 2}, {4, 4}, {2, 8}, {4, 3}}
	for _, c := range combos {
		ref, err := m.ForwardTP(ids, TPConfig{AttnRanks: c.attn, FFNRanks: c.ffn, Coll: LocalCollective{}})
		if err != nil {
			t.Fatalf("ForwardTP via Local (a=%d f=%d): %v", c.attn, c.ffn, err)
		}
		got, err := m.ForwardTP(ids, TPConfig{AttnRanks: c.attn, FFNRanks: c.ffn, Coll: bc})
		if err != nil {
			t.Fatalf("ForwardTP via bridge (a=%d f=%d): %v", c.attn, c.ffn, err)
		}
		exact, mx := bitExactRows(got.Logits, ref.Logits)
		if !exact {
			t.Fatalf("ForwardTP via BackendCollective != via LocalCollective (a=%d f=%d): max|Δ|=%.3e, want bit-exact 0", c.attn, c.ffn, mx)
		}
	}
}

// TestNewBackendCollectiveFailsClosed proves a nil backend is rejected at construction, and
// that cpu-ref (which advertises Caps().Collective) is accepted — the two ends of the
// discovery contract.
func TestNewBackendCollectiveFailsClosed(t *testing.T) {
	if _, err := NewBackendCollective(nil); err == nil {
		t.Fatalf("NewBackendCollective(nil) should fail closed")
	}
	if _, err := NewBackendCollective(compute.Default()); err != nil {
		t.Fatalf("NewBackendCollective(cpu-ref) should succeed: %v", err)
	}
}

// TestBackendCollectiveFailsClosedLikeLocal pins that the bridge rejects the SAME malformed
// inputs LocalCollective does — empty AllReduceSum, ragged partials, and an AllGather rank
// whose width disagrees with the plan — so swapping the communicator does not loosen the
// fail-closed contract (error text may differ; the refusal must not).
func TestBackendCollectiveFailsClosedLikeLocal(t *testing.T) {
	bc := mustBackendColl(t)
	var local LocalCollective

	// empty
	if _, e := local.AllReduceSum(nil); e == nil {
		t.Fatal("local AllReduceSum(nil) should error")
	}
	if _, e := bc.AllReduceSum(nil); e == nil {
		t.Fatal("bridge AllReduceSum(nil) should error")
	}

	// ragged partials
	rag := [][]float32{{1, 2, 3}, {4, 5}}
	if _, e := local.AllReduceSum(rag); e == nil {
		t.Fatal("local AllReduceSum(ragged) should error")
	}
	if _, e := bc.AllReduceSum(rag); e == nil {
		t.Fatal("bridge AllReduceSum(ragged) should error")
	}

	// AllGather rank width disagrees with the plan
	plan, err := NewTPPlan(4, 2) // shards width 2,2
	if err != nil {
		t.Fatalf("NewTPPlan: %v", err)
	}
	bad := [][]float32{{1, 2}, {3}} // rank 1 width 1 != 2
	if _, e := local.AllGather(bad, plan); e == nil {
		t.Fatal("local AllGather(bad width) should error")
	}
	if _, e := bc.AllGather(bad, plan); e == nil {
		t.Fatal("bridge AllGather(bad width) should error")
	}
}
