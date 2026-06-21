package model

import (
	"math"
	"testing"
)

// tensor_parallel_test.go — the correctness gates for the tensor-parallel (intra-layer)
// sharding seam. The headline (TestColumnParallelMatMulBitExact) is the TP analog of
// pipeline.go's "separately-loaded stages == monolith": a matmul sharded across ranks
// over its OUTPUT features and AllGather-recombined is BIT-IDENTICAL (max|Δ|=0) to the
// single-device matRows. The row-parallel gates make the one place TP must reassociate a
// reduction honest: exact vs a shard-grouped reference, and within documented round-off of
// the monolith. The FFN gate is the real composed "TP within a layer" proof.

// tpRand is the package's deterministic pseudo-random generator (same LCG as
// parallel_test.go) so these gates need no rng dependency and reproduce exactly.
func tpRand(n int, seed uint64) []float32 {
	v := make([]float32, n)
	s := seed
	for i := range v {
		s = s*6364136223846793005 + 1442695040888963407
		v[i] = float32(int64(s>>40))/float32(1<<23) - 0.5
	}
	return v
}

// TestColumnParallelMatMulBitExact is the headline: a column-parallel (output-sharded)
// matmul recombined by AllGather equals the single-device matRows BIT-FOR-BIT, across
// rank counts and shapes. This holds because each output row is computed by exactly one
// rank in the identical fdot order — the same argument that makes parMatRows bit-exact —
// so tensor-sharding the output features adds zero numeric drift.
func TestColumnParallelMatMulBitExact(t *testing.T) {
	cases := []struct{ out, in int }{
		{576, 576}, {192, 576}, {1536, 576}, {49152, 576}, {7, 3}, {1, 1}, {8, 64},
	}
	for _, c := range cases {
		w := tpRand(c.out*c.in, uint64(c.out*131+c.in))
		x := tpRand(c.in, 99)
		ref := matRows(w, x, c.out, c.in)
		for _, ranks := range []int{1, 2, 3, 4, 8} {
			if ranks > c.out {
				continue
			}
			plan, err := NewTPPlan(c.out, ranks)
			if err != nil {
				t.Fatalf("NewTPPlan(out=%d, ranks=%d): %v", c.out, ranks, err)
			}
			got, err := ColumnParallelMatMul(w, x, c.out, c.in, plan, LocalCollective{})
			if err != nil {
				t.Fatalf("ColumnParallelMatMul[%dx%d] ranks=%d: %v", c.out, c.in, ranks, err)
			}
			if len(got) != c.out {
				t.Fatalf("ColumnParallelMatMul len = %d, want %d", len(got), c.out)
			}
			for o := 0; o < c.out; o++ {
				if math.Float32bits(got[o]) != math.Float32bits(ref[o]) {
					t.Fatalf("ColumnParallelMatMul[%dx%d] ranks=%d o=%d: %v != monolith %v (NOT bit-identical)",
						c.out, c.in, ranks, o, got[o], ref[o])
				}
			}
		}
	}
}

// TestRowParallelMatMulMatchesShardReference pins RowParallelMatMul to its shard-grouped
// reference BIT-FOR-BIT — the row-parallel path adds nothing beyond the reassociation the
// reference already models — and confirms it stays within the documented ~1e-6 round-off
// of the monolithic matRows (NOT bit-exact there, by design: the AllReduce reassociates
// the contraction). A 1-rank plan must still match the monolith exactly (one shard == one
// fdot over the whole row).
func TestRowParallelMatMulMatchesShardReference(t *testing.T) {
	cases := []struct{ out, in int }{
		{576, 576}, {192, 1536}, {64, 4096}, {1, 4096}, {8, 7},
	}
	for _, c := range cases {
		w := tpRand(c.out*c.in, uint64(c.out*977+c.in))
		x := tpRand(c.in, 7)
		mono := matRows(w, x, c.out, c.in)
		for _, ranks := range []int{1, 2, 3, 4, 8} {
			if ranks > c.in {
				continue
			}
			plan, err := NewTPPlan(c.in, ranks)
			if err != nil {
				t.Fatalf("NewTPPlan(in=%d, ranks=%d): %v", c.in, ranks, err)
			}
			got, err := RowParallelMatMul(w, x, c.out, c.in, plan, LocalCollective{})
			if err != nil {
				t.Fatalf("RowParallelMatMul[%dx%d] ranks=%d: %v", c.out, c.in, ranks, err)
			}
			ref, err := RowParallelReference(w, x, c.out, c.in, plan)
			if err != nil {
				t.Fatalf("RowParallelReference[%dx%d] ranks=%d: %v", c.out, c.in, ranks, err)
			}
			for o := 0; o < c.out; o++ {
				if math.Float32bits(got[o]) != math.Float32bits(ref[o]) {
					t.Fatalf("RowParallelMatMul[%dx%d] ranks=%d o=%d: %v != shard reference %v (NOT bit-identical)",
						c.out, c.in, ranks, o, got[o], ref[o])
				}
			}
			// 1 rank is one fdot over the whole row -> identical to the monolith.
			if ranks == 1 {
				for o := 0; o < c.out; o++ {
					if math.Float32bits(got[o]) != math.Float32bits(mono[o]) {
						t.Fatalf("RowParallelMatMul[%dx%d] ranks=1 o=%d not bit-identical to monolith", c.out, c.in, o)
					}
				}
			}
			// Multi-rank: within the documented reassociation round-off, never garbage.
			// Tight relative bound for non-trivial outputs; absolute floor only on the
			// near-zero outputs the relative test skips (so a gross error there can't hide,
			// and a legitimately large output magnitude can't trip it falsely).
			var maxAbsSmall, maxRel float64
			for o := 0; o < c.out; o++ {
				d := math.Abs(float64(got[o] - mono[o]))
				den := math.Abs(float64(mono[o]))
				if den > 1e-6 {
					if r := d / den; r > maxRel {
						maxRel = r
					}
				} else if d > maxAbsSmall {
					maxAbsSmall = d
				}
			}
			if maxRel > 1e-4 || maxAbsSmall > 1e-3 {
				t.Fatalf("RowParallelMatMul[%dx%d] ranks=%d drift rel %.2e absSmall %.2e vs monolith exceeds round-off bound",
					c.out, c.in, ranks, maxRel, maxAbsSmall)
			}
		}
	}
}

// TestTensorParallelFFNMatchesMonolith is the composed "TP within a layer" proof: the
// Megatron FFN (gate/up column-parallel, down row-parallel, one AllReduce) reproduces the
// single-device SwiGLU FFN within the down-projection's AllReduce round-off, while each
// rank's pre-down activation band is BIT-EXACT vs the monolith's matching slice.
func TestTensorParallelFFNMatchesMonolith(t *testing.T) {
	cases := []struct{ h, inter int }{
		{64, 256}, {128, 512}, {16, 48}, {576, 1536},
	}
	for _, c := range cases {
		gateW := tpRand(c.inter*c.h, uint64(c.inter*31+c.h+1))
		upW := tpRand(c.inter*c.h, uint64(c.inter*37+c.h+2))
		downW := tpRand(c.h*c.inter, uint64(c.h*41+c.inter+3))
		x := tpRand(c.h, 5)

		// Monolithic SwiGLU FFN reference: g=gate·x; u=up·x; a=silu(g)*u; out=down·a.
		g := matRows(gateW, x, c.inter, c.h)
		u := matRows(upW, x, c.inter, c.h)
		act := make([]float32, c.inter)
		for i := 0; i < c.inter; i++ {
			act[i] = silu(g[i]) * u[i]
		}
		monoOut := matRows(downW, act, c.h, c.inter)

		for _, ranks := range []int{1, 2, 4, 8} {
			if ranks > c.inter {
				continue
			}
			plan, err := NewTPPlan(c.inter, ranks)
			if err != nil {
				t.Fatalf("NewTPPlan(inter=%d, ranks=%d): %v", c.inter, ranks, err)
			}
			// Per-rank activation band must be bit-exact vs the monolith's slice (the
			// column-parallel half of the block touches no reduction order).
			for _, s := range plan.Shards {
				gSh := matRows(shardWeightRows(gateW, c.h, s.Lo, s.Hi), x, s.Width(), c.h)
				uSh := matRows(shardWeightRows(upW, c.h, s.Lo, s.Hi), x, s.Width(), c.h)
				for i := 0; i < s.Width(); i++ {
					a := silu(gSh[i]) * uSh[i]
					if math.Float32bits(a) != math.Float32bits(act[s.Lo+i]) {
						t.Fatalf("TP FFN[h=%d,I=%d] ranks=%d rank %d act i=%d not bit-identical to monolith",
							c.h, c.inter, ranks, s.Rank, i)
					}
				}
			}
			out, err := TensorParallelFFN(gateW, upW, downW, x, c.h, c.inter, plan, LocalCollective{})
			if err != nil {
				t.Fatalf("TensorParallelFFN[h=%d,I=%d] ranks=%d: %v", c.h, c.inter, ranks, err)
			}
			if len(out) != c.h {
				t.Fatalf("TensorParallelFFN out len = %d, want %d", len(out), c.h)
			}
			// Bit-exact vs the shard-grouped rank-order reference (max|Δ|=0): pins the down
			// AllReduce's reduction order, which the loose vs-monolith bound cannot see (a
			// reordered/dropped/double-counted reduction stays within round-off). This is the
			// only multi-rank correctness signal that is EXACT.
			ref, err := TensorParallelFFNReference(gateW, upW, downW, x, c.h, c.inter, plan)
			if err != nil {
				t.Fatalf("TensorParallelFFNReference[h=%d,I=%d] ranks=%d: %v", c.h, c.inter, ranks, err)
			}
			for o := 0; o < c.h; o++ {
				if math.Float32bits(out[o]) != math.Float32bits(ref[o]) {
					t.Fatalf("TP FFN[h=%d,I=%d] ranks=%d o=%d: %v != rank-order reference %v (AllReduce order not pinned)",
						c.h, c.inter, ranks, o, out[o], ref[o])
				}
			}
			if ranks == 1 {
				for o := 0; o < c.h; o++ {
					if math.Float32bits(out[o]) != math.Float32bits(monoOut[o]) {
						t.Fatalf("TP FFN[h=%d,I=%d] ranks=1 o=%d not bit-identical to monolith", c.h, c.inter, o)
					}
				}
			}
			var maxRel, maxAbsSmall float64
			for o := 0; o < c.h; o++ {
				d := math.Abs(float64(out[o] - monoOut[o]))
				den := math.Abs(float64(monoOut[o]))
				if den > 1e-6 {
					if r := d / den; r > maxRel {
						maxRel = r
					}
				} else if d > maxAbsSmall {
					maxAbsSmall = d
				}
			}
			// Reassociation round-off only: ~1e-7 relative in practice. Tight relative bound
			// for large coordinates; absolute floor only on the near-zero coordinates the
			// relative test skips (so neither a gross error there nor a legitimately large
			// unnormalized output trips it falsely).
			if maxRel > 1e-4 || maxAbsSmall > 1e-3 {
				t.Fatalf("TP FFN[h=%d,I=%d] ranks=%d drift rel %.2e absSmall %.2e exceeds round-off bound", c.h, c.inter, ranks, maxRel, maxAbsSmall)
			}
		}
	}
}

// TestTPPlanValidate pins the fail-closed planning rules: near-even tiling, full coverage,
// and rejection of degenerate rank counts — so a bad partition is refused before any rank
// loads a shard (the TP analog of PartitionPlan.Validate at plan time).
func TestTPPlanValidate(t *testing.T) {
	// Near-even split: 10 over 3 ranks -> widths 4,3,3, contiguous, complete.
	p, err := NewTPPlan(10, 3)
	if err != nil {
		t.Fatalf("NewTPPlan(10,3): %v", err)
	}
	wantWidths := []int{4, 3, 3}
	if len(p.Shards) != 3 {
		t.Fatalf("got %d shards, want 3", len(p.Shards))
	}
	lo := 0
	for r, s := range p.Shards {
		if s.Rank != r {
			t.Fatalf("shard %d Rank = %d", r, s.Rank)
		}
		if s.Width() != wantWidths[r] {
			t.Fatalf("shard %d width = %d, want %d", r, s.Width(), wantWidths[r])
		}
		if s.Lo != lo {
			t.Fatalf("shard %d Lo = %d, want %d (gap/overlap)", r, s.Lo, lo)
		}
		lo = s.Hi
	}
	if lo != 10 {
		t.Fatalf("coverage ends at %d, want 10", lo)
	}

	// Degenerate rank counts fail closed.
	if _, err := NewTPPlan(4, 0); err == nil {
		t.Fatalf("NewTPPlan(4,0) should fail (zero ranks)")
	}
	if _, err := NewTPPlan(4, 5); err == nil {
		t.Fatalf("NewTPPlan(4,5) should fail (more ranks than dim)")
	}
	if _, err := NewTPPlan(0, 1); err == nil {
		t.Fatalf("NewTPPlan(0,1) should fail (empty dim)")
	}

	// A hand-built plan with a gap is rejected by Validate.
	bad := TPPlan{Dim: 10, Shards: []TPShard{{Rank: 0, Lo: 0, Hi: 4}, {Rank: 1, Lo: 5, Hi: 10}}}
	if err := bad.Validate(); err == nil {
		t.Fatalf("Validate should reject a gap between shards")
	}
	// Incomplete coverage is rejected.
	short := TPPlan{Dim: 10, Shards: []TPShard{{Rank: 0, Lo: 0, Hi: 4}, {Rank: 1, Lo: 4, Hi: 8}}}
	if err := short.Validate(); err == nil {
		t.Fatalf("Validate should reject incomplete coverage")
	}
}

// TestLocalCollectiveFailsClosed pins the collective seam's fail-closed contract: AllGather
// rejects a mis-sized rank output, and AllReduceSum rejects ragged partials — a real NCCL
// collective swapped in behind this interface inherits the same boundary checks.
func TestLocalCollectiveFailsClosed(t *testing.T) {
	plan, err := NewTPPlan(6, 3) // widths 2,2,2
	if err != nil {
		t.Fatalf("NewTPPlan: %v", err)
	}
	c := LocalCollective{}
	// Correct AllGather concatenates in rank order.
	got, err := c.AllGather([][]float32{{1, 2}, {3, 4}, {5, 6}}, plan)
	if err != nil {
		t.Fatalf("AllGather: %v", err)
	}
	want := []float32{1, 2, 3, 4, 5, 6}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("AllGather[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	// Mis-sized rank output rejected.
	if _, err := c.AllGather([][]float32{{1, 2}, {3}, {5, 6}}, plan); err == nil {
		t.Fatalf("AllGather should reject a rank output whose width != shard width")
	}
	// Wrong number of parts rejected.
	if _, err := c.AllGather([][]float32{{1, 2}, {3, 4}}, plan); err == nil {
		t.Fatalf("AllGather should reject a part count != shard count")
	}
	// AllReduceSum adds in rank order; ragged partials rejected.
	sum, err := c.AllReduceSum([][]float32{{1, 2, 3}, {10, 20, 30}})
	if err != nil {
		t.Fatalf("AllReduceSum: %v", err)
	}
	for i, w := range []float32{11, 22, 33} {
		if sum[i] != w {
			t.Fatalf("AllReduceSum[%d] = %v, want %v", i, sum[i], w)
		}
	}
	if _, err := c.AllReduceSum([][]float32{{1, 2, 3}, {10, 20}}); err == nil {
		t.Fatalf("AllReduceSum should reject ragged partials")
	}
}
