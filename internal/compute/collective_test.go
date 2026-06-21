package compute

import (
	"math"
	"testing"
)

// collective_test.go — the correctness gates for the CollectiveBackend seam (collective.go),
// the tensor-parallel cross-rank reduction at the HAL. They prove the CPU reference is the
// single-box EXACT collective a real NCCL/RDMA backend must reproduce:
//
//   - AllReduceSum / AllGather reduce/concatenate in RANK ORDER, bit-for-bit;
//   - ReduceScatter (the dual of AllGather, the sequence-parallel collective) scatters the
//     same rank-order sum, so AllGather∘ReduceScatter == AllReduceSum byte-for-byte;
//   - AllToAll (the transpose/redistribution collective) is an INVOLUTION, and ReduceScatter is
//     recoverable as AllToAll + a local per-rank reduce — both byte-for-byte, NCCL's construction;
//   - the single-rank case is the IDENTITY (the HAL twin of ForwardTP(ranks=1)==Forward —
//     the "bit-exact vs single-device" rung, witnessable with no multi-GPU hardware);
//   - a row-parallel matmul reassembled through AllReduceSum is bit-exact vs its rank-order
//     reference and within documented round-off of the monolithic device matmul;
//   - the fail-closed contract (no parts, ragged, non-F32, unready, foreign backend) holds —
//     the cross-backend reduction a real communicator rejects.

// asCollective fetches the CPU reference as a CollectiveBackend, asserting it advertises and
// implements the optional seam (the discovery idiom the doc describes: Caps flag + type-assert).
func asCollective(t *testing.T) (*cpuBackend, CollectiveBackend) {
	t.Helper()
	c := cpu()
	if !c.Caps().Collective {
		t.Fatalf("cpu-ref Caps().Collective = false, want true")
	}
	cb, ok := Backend(c).(CollectiveBackend)
	if !ok {
		t.Fatalf("cpu-ref does not implement CollectiveBackend despite advertising the cap")
	}
	return c, cb
}

// TestCollectiveAllReduceSumRankOrder pins AllReduceSum: a new tensor equal to the rank-order
// sum of the partials, bit-for-bit, and the single-part identity. The rank-order reference is
// computed independently here so a reduction that reordered, dropped, or double-counted a rank
// is caught at max|Δ|=0 (a defect the loose vs-monolith bound cannot see).
func TestCollectiveAllReduceSumRankOrder(t *testing.T) {
	c, cb := asCollective(t)
	var s lcg = 7
	const n = 17 // exercises fdot's lane+tail split downstream and a non-round length here
	for _, ranks := range []int{1, 2, 3, 5} {
		parts := make([]Tensor, ranks)
		raw := make([][]float32, ranks)
		for r := 0; r < ranks; r++ {
			raw[r] = randVec(&s, n)
			parts[r] = NewF32(c, []int{n}, append([]float32(nil), raw[r]...))
		}
		// independent rank-order reference
		want := make([]float32, n)
		copy(want, raw[0])
		for r := 1; r < ranks; r++ {
			for i := 0; i < n; i++ {
				want[i] += raw[r][i]
			}
		}
		out, err := cb.AllReduceSum(parts)
		if err != nil {
			t.Fatalf("AllReduceSum ranks=%d: %v", ranks, err)
		}
		got := c.Read(out)
		if len(got) != n {
			t.Fatalf("AllReduceSum ranks=%d len = %d, want %d", ranks, len(got), n)
		}
		for i := 0; i < n; i++ {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("AllReduceSum ranks=%d i=%d: %v != rank-order reference %v (order not pinned)", ranks, i, got[i], want[i])
			}
		}
		// single rank is the identity: byte-for-byte the lone part.
		if ranks == 1 {
			for i := 0; i < n; i++ {
				if math.Float32bits(got[i]) != math.Float32bits(raw[0][i]) {
					t.Fatalf("AllReduceSum ranks=1 i=%d not identity to the single part", i)
				}
			}
		}
	}
}

// TestCollectiveAllGatherRankOrder pins AllGather: rank-ordered concatenation of (possibly
// uneven) shards, and the single-part identity.
func TestCollectiveAllGatherRankOrder(t *testing.T) {
	c, cb := asCollective(t)
	var s lcg = 11
	widths := []int{4, 1, 7, 2} // deliberately uneven — gather must not assume equal shards
	parts := make([]Tensor, len(widths))
	var want []float32
	for r, w := range widths {
		v := randVec(&s, w)
		parts[r] = NewF32(c, []int{w}, append([]float32(nil), v...))
		want = append(want, v...)
	}
	out, err := cb.AllGather(parts)
	if err != nil {
		t.Fatalf("AllGather: %v", err)
	}
	got := c.Read(out)
	if len(got) != len(want) {
		t.Fatalf("AllGather len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("AllGather i=%d: %v != rank-order concat %v", i, got[i], want[i])
		}
	}
	// single part identity
	one := randVec(&s, 5)
	out1, err := cb.AllGather([]Tensor{NewF32(c, []int{5}, append([]float32(nil), one...))})
	if err != nil {
		t.Fatalf("AllGather single: %v", err)
	}
	g1 := c.Read(out1)
	for i := range one {
		if math.Float32bits(g1[i]) != math.Float32bits(one[i]) {
			t.Fatalf("AllGather single-part i=%d not identity", i)
		}
	}
}

// TestCollectiveReduceScatter pins ReduceScatter: it is the rank-order AllReduceSum scattered
// into P equal shards, so (1) each shard equals the matching 1/P band of the full reduction,
// (2) AllGather(ReduceScatter(parts)) reconstructs AllReduceSum(parts) byte-for-byte — the
// dual-of-AllGather identity that pins the seam against the already-proven methods — and (3)
// the single-rank case is the identity. N is chosen divisible by every rank count exercised.
func TestCollectiveReduceScatter(t *testing.T) {
	c, cb := asCollective(t)
	var s lcg = 23
	const n = 24 // divisible by 1,2,3,4,6 — the rank counts exercised
	for _, ranks := range []int{1, 2, 3, 4, 6} {
		parts := make([]Tensor, ranks)
		raw := make([][]float32, ranks)
		for r := 0; r < ranks; r++ {
			raw[r] = randVec(&s, n)
			parts[r] = NewF32(c, []int{n}, append([]float32(nil), raw[r]...))
		}
		// The already-pinned AllReduceSum is the oracle for the full reduction.
		full, err := cb.AllReduceSum(parts)
		if err != nil {
			t.Fatalf("AllReduceSum ranks=%d: %v", ranks, err)
		}
		fullV := c.Read(full)

		shards, err := cb.ReduceScatter(parts)
		if err != nil {
			t.Fatalf("ReduceScatter ranks=%d: %v", ranks, err)
		}
		if len(shards) != ranks {
			t.Fatalf("ReduceScatter ranks=%d returned %d shards, want %d", ranks, len(shards), ranks)
		}
		shardLen := n / ranks
		// (1) each shard is the matching contiguous band of the full reduction, bit-for-bit.
		for r := 0; r < ranks; r++ {
			sv := c.Read(shards[r])
			if len(sv) != shardLen {
				t.Fatalf("ReduceScatter ranks=%d shard %d len = %d, want %d", ranks, r, len(sv), shardLen)
			}
			for i := 0; i < shardLen; i++ {
				if math.Float32bits(sv[i]) != math.Float32bits(fullV[r*shardLen+i]) {
					t.Fatalf("ReduceScatter ranks=%d shard %d i=%d: %v != AllReduceSum band %v",
						ranks, r, i, sv[i], fullV[r*shardLen+i])
				}
			}
		}
		// (2) the defining identity: AllGather of the scattered shards == AllReduceSum, byte-for-byte.
		regathered, err := cb.AllGather(shards)
		if err != nil {
			t.Fatalf("AllGather(ReduceScatter) ranks=%d: %v", ranks, err)
		}
		rg := c.Read(regathered)
		if len(rg) != n {
			t.Fatalf("AllGather(ReduceScatter) ranks=%d len = %d, want %d", ranks, len(rg), n)
		}
		for i := 0; i < n; i++ {
			if math.Float32bits(rg[i]) != math.Float32bits(fullV[i]) {
				t.Fatalf("AllGather(ReduceScatter) ranks=%d i=%d: %v != AllReduceSum %v (AllReduce != AllGather∘ReduceScatter)",
					ranks, i, rg[i], fullV[i])
			}
		}
		// (3) single rank is the identity: one shard equal to the lone part.
		if ranks == 1 {
			sv := c.Read(shards[0])
			for i := 0; i < n; i++ {
				if math.Float32bits(sv[i]) != math.Float32bits(raw[0][i]) {
					t.Fatalf("ReduceScatter ranks=1 i=%d not identity to the single part", i)
				}
			}
		}
	}
}

// TestCollectiveAllToAll pins AllToAll: the block transpose that hands shard r of every input
// rank to output rank r. It checks (1) the result against an independent transpose reference,
// (2) the INVOLUTION AllToAll(AllToAll(parts)) == parts byte-for-byte, (3) the single-rank
// identity, and (4) the cross-collective gate ReduceScatter == AllToAll + a local per-rank reduce
// byte-for-byte — NCCL's own construction, which ties this method to the already-proven seam.
func TestCollectiveAllToAll(t *testing.T) {
	c, cb := asCollective(t)
	var s lcg = 29
	const n = 24 // divisible by 1,2,3,4,6 — the rank counts exercised
	for _, ranks := range []int{1, 2, 3, 4, 6} {
		parts := make([]Tensor, ranks)
		raw := make([][]float32, ranks)
		for r := 0; r < ranks; r++ {
			raw[r] = randVec(&s, n)
			parts[r] = NewF32(c, []int{n}, append([]float32(nil), raw[r]...))
		}
		shard := n / ranks

		out, err := cb.AllToAll(parts)
		if err != nil {
			t.Fatalf("AllToAll ranks=%d: %v", ranks, err)
		}
		if len(out) != ranks {
			t.Fatalf("AllToAll ranks=%d returned %d parts, want %d", ranks, len(out), ranks)
		}
		// (1) independent transpose reference: out[r]'s k-th block is rank k's shard r.
		for r := 0; r < ranks; r++ {
			ov := c.Read(out[r])
			if len(ov) != n {
				t.Fatalf("AllToAll ranks=%d out[%d] len = %d, want %d", ranks, r, len(ov), n)
			}
			for k := 0; k < ranks; k++ {
				for j := 0; j < shard; j++ {
					if math.Float32bits(ov[k*shard+j]) != math.Float32bits(raw[k][r*shard+j]) {
						t.Fatalf("AllToAll ranks=%d out[%d] elem %d = %v != rank %d shard %d elem %v (transpose not pinned)",
							ranks, r, k*shard+j, ov[k*shard+j], k, r, raw[k][r*shard+j])
					}
				}
			}
		}
		// (2) involution: applying the transpose twice restores the input byte-for-byte.
		back, err := cb.AllToAll(out)
		if err != nil {
			t.Fatalf("AllToAll(AllToAll) ranks=%d: %v", ranks, err)
		}
		for r := 0; r < ranks; r++ {
			bv := c.Read(back[r])
			for i := 0; i < n; i++ {
				if math.Float32bits(bv[i]) != math.Float32bits(raw[r][i]) {
					t.Fatalf("AllToAll involution ranks=%d rank %d i=%d: %v != original %v", ranks, r, i, bv[i], raw[r][i])
				}
			}
		}
		// (3) single rank is the identity: the lone vector unchanged.
		if ranks == 1 {
			ov := c.Read(out[0])
			for i := 0; i < n; i++ {
				if math.Float32bits(ov[i]) != math.Float32bits(raw[0][i]) {
					t.Fatalf("AllToAll ranks=1 i=%d not identity to the single part", i)
				}
			}
		}
		// (4) ReduceScatter == AllToAll + local per-rank elementwise reduce, byte-for-byte. Both
		// sum the same shards in the SAME rank order, so the result is bit-identical, not merely
		// within round-off — the identity that pins AllToAll to the proven ReduceScatter.
		rs, err := cb.ReduceScatter(parts)
		if err != nil {
			t.Fatalf("ReduceScatter ranks=%d: %v", ranks, err)
		}
		for r := 0; r < ranks; r++ {
			ov := c.Read(out[r])
			local := make([]float32, shard)
			copy(local, ov[0:shard])
			for k := 1; k < ranks; k++ {
				for j := 0; j < shard; j++ {
					local[j] += ov[k*shard+j]
				}
			}
			rsv := c.Read(rs[r])
			for j := 0; j < shard; j++ {
				if math.Float32bits(local[j]) != math.Float32bits(rsv[j]) {
					t.Fatalf("ranks=%d rank %d j=%d: AllToAll+localReduce %v != ReduceScatter %v (RS != A2A∘reduce)",
						ranks, r, j, local[j], rsv[j])
				}
			}
		}
	}
}

// TestCollectiveRowParallelMatMul is the composed proof at the HAL: a matmul sharded across
// its CONTRACTION dimension (each rank holds a column band of W and the matching x segment,
// computes a partial via the device MatMul, and the partials are AllReduceSum'd) is bit-exact
// vs the rank-order reference and within documented round-off of the monolithic device MatMul.
// ranks=1 (one band == the full row) is bit-exact vs the monolith — the single-device rung.
func TestCollectiveRowParallelMatMul(t *testing.T) {
	c, cb := asCollective(t)
	cases := []struct{ out, in int }{{8, 64}, {32, 96}, {1, 50}, {16, 7}}
	for _, cs := range cases {
		var s lcg = lcg(uint64(cs.out*131 + cs.in))
		W := randVec(&s, cs.out*cs.in) // [out,in] row-major
		x := randVec(&s, cs.in)
		mono := c.Read(c.MatMul(NewF32(c, []int{cs.out, cs.in}, W), NewF32(c, []int{cs.in}, x)))

		for _, ranks := range []int{1, 2, 3, 4} {
			if ranks > cs.in {
				continue
			}
			bands := tileDim(cs.in, ranks)
			parts := make([]Tensor, len(bands))
			for r, b := range bands {
				lo, hi := b[0], b[1]
				wsh := hi - lo
				packed := make([]float32, cs.out*wsh) // column band [out, hi-lo]
				for o := 0; o < cs.out; o++ {
					copy(packed[o*wsh:(o+1)*wsh], W[o*cs.in+lo:o*cs.in+hi])
				}
				xseg := append([]float32(nil), x[lo:hi]...)
				parts[r] = c.MatMul(NewF32(c, []int{cs.out, wsh}, packed), NewF32(c, []int{wsh}, xseg))
			}
			// rank-order reference: sum the partials in rank order, independently of the collective.
			ref := make([]float32, cs.out)
			copy(ref, c.Read(parts[0]))
			for r := 1; r < len(parts); r++ {
				pr := c.Read(parts[r])
				for o := 0; o < cs.out; o++ {
					ref[o] += pr[o]
				}
			}
			out, err := cb.AllReduceSum(parts)
			if err != nil {
				t.Fatalf("row-parallel AllReduceSum[%dx%d] ranks=%d: %v", cs.out, cs.in, ranks, err)
			}
			got := c.Read(out)
			for o := 0; o < cs.out; o++ {
				if math.Float32bits(got[o]) != math.Float32bits(ref[o]) {
					t.Fatalf("row-parallel[%dx%d] ranks=%d o=%d: %v != rank-order reference %v (AllReduce order not pinned)",
						cs.out, cs.in, ranks, o, got[o], ref[o])
				}
			}
			if ranks == 1 {
				for o := 0; o < cs.out; o++ {
					if math.Float32bits(got[o]) != math.Float32bits(mono[o]) {
						t.Fatalf("row-parallel[%dx%d] ranks=1 o=%d not bit-identical to monolith", cs.out, cs.in, o)
					}
				}
			}
			// multi-rank: within the reassociation round-off, never garbage.
			var maxRel, maxAbsSmall float64
			for o := 0; o < cs.out; o++ {
				d := math.Abs(float64(got[o] - mono[o]))
				den := math.Abs(float64(mono[o]))
				if den > 1e-6 {
					if rr := d / den; rr > maxRel {
						maxRel = rr
					}
				} else if d > maxAbsSmall {
					maxAbsSmall = d
				}
			}
			if maxRel > 1e-4 || maxAbsSmall > 1e-3 {
				t.Fatalf("row-parallel[%dx%d] ranks=%d drift rel %.2e absSmall %.2e exceeds round-off bound",
					cs.out, cs.in, ranks, maxRel, maxAbsSmall)
			}
		}
	}
}

// TestCollectiveFailsClosed pins the fail-closed boundary: no parts, ragged AllReduce partials,
// a non-F32 part, an unready part, and a part owned by a DIFFERENT backend are each rejected —
// the cross-backend reduction a real communicator must refuse.
func TestCollectiveFailsClosed(t *testing.T) {
	c, cb := asCollective(t)
	ok := func() Tensor { return NewF32(c, []int{3}, []float32{1, 2, 3}) }

	if _, err := cb.AllReduceSum(nil); err == nil {
		t.Fatalf("AllReduceSum should reject zero parts")
	}
	if _, err := cb.AllGather(nil); err == nil {
		t.Fatalf("AllGather should reject zero parts")
	}
	// ragged partials
	if _, err := cb.AllReduceSum([]Tensor{ok(), NewF32(c, []int{2}, []float32{9, 9})}); err == nil {
		t.Fatalf("AllReduceSum should reject ragged partials")
	}
	// non-F32 part (a Q8 weight owned by c)
	q := QuantizeQ8(c, []int{1, 32}, randVecN(32), 32)
	if _, err := cb.AllReduceSum([]Tensor{ok(), q}); err == nil {
		t.Fatalf("AllReduceSum should reject a non-f32 part")
	}
	// unready part (nil buffer)
	unready := makeTensor(c, F32, RowMajor, []int{3}, nil, nil)
	if _, err := cb.AllReduceSum([]Tensor{ok(), unready}); err == nil {
		t.Fatalf("AllReduceSum should reject an unready part")
	}
	// foreign backend: a tensor owned by a DIFFERENT backend (a distinct type — the realistic
	// CUDA-tensor-vs-host-tensor case; cpuBackend is zero-size so two *cpuBackend pointers alias).
	foreign := NewF32(foreignBackend{c}, []int{3}, []float32{4, 5, 6})
	if _, err := cb.AllReduceSum([]Tensor{ok(), foreign}); err == nil {
		t.Fatalf("AllReduceSum should reject a part owned by a different backend")
	}
	if _, err := cb.AllGather([]Tensor{ok(), foreign}); err == nil {
		t.Fatalf("AllGather should reject a part owned by a different backend")
	}
	// ReduceScatter shares the collectF32 boundary, PLUS its own indivisible-length rule:
	// the reduced length must be a multiple of the rank count (real reduce-scatter requires
	// sendcount % nranks == 0), so two length-3 partials over 2 ranks is rejected.
	if _, err := cb.ReduceScatter(nil); err == nil {
		t.Fatalf("ReduceScatter should reject zero parts")
	}
	if _, err := cb.ReduceScatter([]Tensor{ok(), NewF32(c, []int{2}, []float32{9, 9})}); err == nil {
		t.Fatalf("ReduceScatter should reject ragged partials")
	}
	if _, err := cb.ReduceScatter([]Tensor{ok(), ok()}); err == nil {
		t.Fatalf("ReduceScatter should reject a reduced length not divisible by the rank count")
	}
	if _, err := cb.ReduceScatter([]Tensor{ok(), foreign}); err == nil {
		t.Fatalf("ReduceScatter should reject a part owned by a different backend")
	}
	// AllToAll shares the collectF32 boundary plus ReduceScatter's indivisible-length rule (real
	// all-to-all requires sendcount % nranks == 0): zero parts, ragged partials, a per-rank length
	// not divisible by the rank count, and a foreign-backend part are each rejected.
	if _, err := cb.AllToAll(nil); err == nil {
		t.Fatalf("AllToAll should reject zero parts")
	}
	if _, err := cb.AllToAll([]Tensor{ok(), NewF32(c, []int{2}, []float32{9, 9})}); err == nil {
		t.Fatalf("AllToAll should reject ragged partials")
	}
	if _, err := cb.AllToAll([]Tensor{ok(), ok()}); err == nil {
		t.Fatalf("AllToAll should reject a per-rank length not divisible by the rank count")
	}
	if _, err := cb.AllToAll([]Tensor{ok(), foreign}); err == nil {
		t.Fatalf("AllToAll should reject a part owned by a different backend")
	}
}

// tileDim splits [0,dim) into `ranks` near-even contiguous bands (first dim%ranks bands get one
// extra index) — a local mirror of model.NewTPPlan's tiling for the row-parallel proof.
func tileDim(dim, ranks int) [][2]int {
	base, extra := dim/ranks, dim%ranks
	out := make([][2]int, ranks)
	lo := 0
	for r := 0; r < ranks; r++ {
		w := base
		if r < extra {
			w++
		}
		out[r] = [2]int{lo, lo + w}
		lo += w
	}
	return out
}

// randVecN is a tiny fixed-seed vector for the fail-closed Q8 construction (no shared lcg state).
func randVecN(n int) []float32 {
	var s lcg = 1234567
	return randVec(&s, n)
}

// foreignBackend is a DISTINCT backend type wrapping the reference's kernels — a stand-in for a
// real device backend (e.g. a CUDA target). Its tensors carry a different dynamic type than
// *cpuBackend, so the collective's interface-identity affinity check rejects them, exercising
// the cross-backend contract honestly (a same-type *cpuBackend would alias, being zero-size).
type foreignBackend struct{ *cpuBackend }

func (foreignBackend) Name() string { return "foreign" }
