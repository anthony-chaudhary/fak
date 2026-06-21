package model

import "fmt"

// tensor_parallel.go — tensor-parallel (intra-layer) sharding for native multi-GPU
// serving. This is the "tensor parallelism within a layer" lever that
// GLM-5.2-NATIVE-ENGINE-GAP names as the remaining multi-A100 step AFTER the
// pipeline-parallel arc (partition.go / pipeline.go) landed. Pipeline parallelism
// splits the layer STACK across workers and crosses a hidden state at each boundary;
// tensor parallelism splits a SINGLE layer's matmuls across workers and crosses a
// collective (AllGather / AllReduce) inside the layer. The two compose: a real
// 8×A100 plan is a grid of pipeline stages × tensor-parallel ranks.
//
// The decomposition here is the canonical Megatron-LM one, and the repo's existing
// numeric discipline (parallel.go) is exactly what makes it honest:
//
//   - COLUMN-PARALLEL (shard the OUTPUT features): y = x·Wᵀ, W split into row-bands
//     [W_0; W_1; …]; rank r computes y_r = x·W_rᵀ over its output band, and the parts
//     are AllGather-concatenated in rank order into the full y. Each output element
//     y[o] is still fdot(W[o], x) computed by exactly ONE rank in the SAME inner order
//     as the monolithic matRows — only the assignment of which rank computes which
//     output row differs. That is precisely parMatRows' argument, so column-parallel is
//     BIT-EXACT vs the single-device matmul (max|Δ|=0). This is the TP analog of
//     pipeline.go's "separately-loaded stages == monolith" proof.
//
//   - ROW-PARALLEL (shard the CONTRACTION dim): y = x·Wᵀ, W split into column-bands
//     and x into matching segments; rank r computes a PARTIAL y over its slice of the
//     contraction, and the parts are AllReduce-summed. This REASSOCIATES the reduction
//     (sum-of-per-shard-fdots, not one fdot over the whole row), so it is NOT bit-exact
//     vs the monolithic matRows — it drifts ~1e-6, the same non-associativity parallel.go
//     already documents for fdot. It IS bit-exact vs a shard-grouped reference (the sum,
//     in rank order, of each shard's fdot), which is the invariant the gate pins.
//
// Megatron uses exactly these two: attention is QKV column-parallel (shard heads) then
// output-proj row-parallel; the FFN is gate/up column-parallel then down row-parallel —
// one AllReduce per block, the intermediate never gathered. TensorParallelFFN below is
// that composed block, and its test is the real "TP within a layer" proof.
//
// The Collective interface is the AllGather/AllReduce seam, the TP counterpart of
// pipeline.go's StageTransport: LocalCollective is the bit-exact single-box default, and
// the "real NCCL collective" is a swap (implement two methods), not a rewrite of the
// matmul decomposition. Nothing here is wired into the live Forward path yet — like
// WithLayerWindow/PipelineStage before it, this lands the proven primitive the
// forward-path attention/FFN sharding builds on.

// TPShard is the half-open [Lo,Hi) band of a sharded dimension one rank owns. For a
// column-parallel matmul the sharded dimension is the OUTPUT features; for a
// row-parallel matmul it is the CONTRACTION (input) dimension.
type TPShard struct {
	Rank   int
	Lo, Hi int
}

// Width is the number of indices this shard owns.
func (s TPShard) Width() int { return s.Hi - s.Lo }

// TPPlan is a validated tiling of a single dimension [0,Dim) into Ranks contiguous,
// non-overlapping, complete shards — the tensor-parallel analog of PartitionPlan
// (which tiles the LAYER stack). It is dimension-agnostic: the same plan over Dim==out
// drives a column-parallel matmul and over Dim==in drives a row-parallel one.
type TPPlan struct {
	Dim    int
	Shards []TPShard
}

// NewTPPlan tiles [0,dim) into `ranks` near-even contiguous shards (the first
// dim%ranks shards get one extra index, so widths differ by at most 1). It fails closed
// rather than emit a degenerate plan: ranks must be in [1,dim] so no shard is empty —
// an empty shard would mean a rank with no work, which the collective then cannot place.
func NewTPPlan(dim, ranks int) (TPPlan, error) {
	if dim <= 0 {
		return TPPlan{}, fmt.Errorf("model: TPPlan dim = %d, want > 0", dim)
	}
	if ranks <= 0 {
		return TPPlan{}, fmt.Errorf("model: TPPlan ranks = %d, want > 0", ranks)
	}
	if ranks > dim {
		return TPPlan{}, fmt.Errorf("model: TPPlan ranks = %d > dim = %d (would leave a rank with no work)", ranks, dim)
	}
	base := dim / ranks
	extra := dim % ranks
	shards := make([]TPShard, ranks)
	lo := 0
	for r := 0; r < ranks; r++ {
		w := base
		if r < extra {
			w++
		}
		shards[r] = TPShard{Rank: r, Lo: lo, Hi: lo + w}
		lo += w
	}
	p := TPPlan{Dim: dim, Shards: shards}
	if err := p.Validate(); err != nil {
		return TPPlan{}, err
	}
	return p, nil
}

// Validate fails closed on any malformed tiling, mirroring PartitionPlan.Validate:
// the shards must tile [0,Dim) contiguously (first Lo==0, last Hi==Dim, each Lo ==
// previous Hi — no gap, no overlap), every shard non-empty and in range, and rank r
// labelled r so AllGather's rank-ordered concatenation is unambiguous.
func (p TPPlan) Validate() error {
	if p.Dim <= 0 {
		return fmt.Errorf("model: TPPlan Dim = %d, want > 0", p.Dim)
	}
	if len(p.Shards) == 0 {
		return fmt.Errorf("model: TPPlan has no shards")
	}
	for i, s := range p.Shards {
		if s.Rank != i {
			return fmt.Errorf("model: TPPlan shard %d has Rank = %d, want %d (rank order drives AllGather concat)", i, s.Rank, i)
		}
		if s.Lo < 0 || s.Hi <= s.Lo || s.Hi > p.Dim {
			return fmt.Errorf("model: TPPlan shard %d band [%d,%d) invalid for dim %d", i, s.Lo, s.Hi, p.Dim)
		}
		if i == 0 {
			if s.Lo != 0 {
				return fmt.Errorf("model: TPPlan first shard starts at %d, want 0", s.Lo)
			}
		} else if s.Lo != p.Shards[i-1].Hi {
			return fmt.Errorf("model: TPPlan shard %d starts at %d but shard %d ends at %d (gap or overlap)", i, s.Lo, i-1, p.Shards[i-1].Hi)
		}
	}
	if last := p.Shards[len(p.Shards)-1].Hi; last != p.Dim {
		return fmt.Errorf("model: TPPlan last shard ends at %d, want %d (incomplete coverage)", last, p.Dim)
	}
	return nil
}

// Collective is the rank-to-rank reduction seam — the tensor-parallel counterpart of
// pipeline.go's StageTransport. A column-parallel matmul ends in AllGather (concatenate
// each rank's output band); a row-parallel matmul ends in AllReduceSum (sum each rank's
// partial). This is the seam the "real NCCL collective" plugs into: LocalCollective is
// the single-box, bit-exact default, and a fleet swaps in an NCCL/RDMA implementation
// without touching the matmul decomposition or the gates that pin it.
//
// Both methods fail closed on ragged input (parts of mismatched length, or a part whose
// width disagrees with the plan) rather than producing a silently-truncated result.
type Collective interface {
	// AllGather concatenates the per-rank parts in rank order into one slice. parts[r]
	// must be exactly p.Shards[r].Width() long. Used to recombine a column-parallel
	// matmul's output bands.
	AllGather(parts [][]float32, p TPPlan) ([]float32, error)
	// AllReduceSum returns the element-wise sum of equal-length per-rank partials, added
	// in rank order (a FIXED order, so the result is deterministic and the row-parallel
	// gate is exact vs its shard-grouped reference). Used to recombine a row-parallel
	// matmul's partial sums.
	AllReduceSum(parts [][]float32) ([]float32, error)
}

// LocalCollective is the default single-box Collective: AllGather concatenates, and
// AllReduceSum adds in rank order. It is exact by construction and is what the in-process
// TP primitives use. A real fleet swaps this for an NCCL/RDMA implementation; the bytes
// it exchanges are exactly these per-rank slices, so the two are interchangeable.
type LocalCollective struct{}

// AllGather concatenates parts[0]‖parts[1]‖… after checking each rank's width against the
// plan, so a mis-sized rank output is rejected at the boundary instead of shifting every
// downstream feature.
func (LocalCollective) AllGather(parts [][]float32, p TPPlan) ([]float32, error) {
	if len(parts) != len(p.Shards) {
		return nil, fmt.Errorf("model: AllGather got %d parts, plan has %d shards", len(parts), len(p.Shards))
	}
	out := make([]float32, 0, p.Dim)
	for r, s := range p.Shards {
		if len(parts[r]) != s.Width() {
			return nil, fmt.Errorf("model: AllGather rank %d part len = %d, want shard width %d", r, len(parts[r]), s.Width())
		}
		out = append(out, parts[r]...)
	}
	if len(out) != p.Dim {
		return nil, fmt.Errorf("model: AllGather produced %d elements, want dim %d", len(out), p.Dim)
	}
	return out, nil
}

// AllReduceSum sums equal-length partials in rank order: acc = parts[0]; acc += parts[r]
// for r=1.. . The fixed order makes the result deterministic and bit-identical to the
// shard-grouped reference the row-parallel gate compares against.
func (LocalCollective) AllReduceSum(parts [][]float32) ([]float32, error) {
	if len(parts) == 0 {
		return nil, fmt.Errorf("model: AllReduceSum has no parts")
	}
	n := len(parts[0])
	acc := make([]float32, n)
	copy(acc, parts[0])
	for r := 1; r < len(parts); r++ {
		if len(parts[r]) != n {
			return nil, fmt.Errorf("model: AllReduceSum part %d len = %d, want %d (ragged partials)", r, len(parts[r]), n)
		}
		for i := 0; i < n; i++ {
			acc[i] += parts[r][i]
		}
	}
	return acc, nil
}

// shardWeightRows returns the contiguous output-row band W[lo:hi) of a row-major
// [out,in] weight — the column-parallel shard, a zero-copy view since output rows are
// already contiguous. (Naming follows HF: y = x·Wᵀ, so an OUTPUT feature is a ROW of W.)
func shardWeightRows(w []float32, in, lo, hi int) []float32 {
	return w[lo*in : hi*in]
}

// shardWeightColumns gathers the contraction-dim band [lo,hi) from every row of a
// row-major [out,in] weight into a packed [out, hi-lo] matrix — the row-parallel shard.
// Unlike the column-parallel case this is a real copy, because a contraction-dim band is
// strided across the original rows. The matching input segment is x[lo:hi].
func shardWeightColumns(w []float32, out, in, lo, hi int) []float32 {
	wsh := hi - lo
	dst := make([]float32, out*wsh)
	for o := 0; o < out; o++ {
		copy(dst[o*wsh:(o+1)*wsh], w[o*in+lo:o*in+hi])
	}
	return dst
}

// ColumnParallelMatMul computes y = x·Wᵀ (W row-major [out,in]) sharded across the OUTPUT
// features by plan (plan.Dim must == out): each rank runs matRows over its output band,
// then coll.AllGather concatenates the bands in rank order. The result is BIT-EXACT vs the
// single-device matRows(w,x,out,in): every y[o] is fdot(W[o],x) computed by one rank in
// the identical inner order; only which rank owns row o differs. A nil collective defaults
// to LocalCollective.
func ColumnParallelMatMul(w, x []float32, out, in int, plan TPPlan, coll Collective) ([]float32, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	if plan.Dim != out {
		return nil, fmt.Errorf("model: ColumnParallelMatMul plan.Dim = %d, want out = %d", plan.Dim, out)
	}
	if len(x) != in {
		return nil, fmt.Errorf("model: ColumnParallelMatMul x len = %d, want in = %d", len(x), in)
	}
	if len(w) != out*in {
		return nil, fmt.Errorf("model: ColumnParallelMatMul w len = %d, want out*in = %d", len(w), out*in)
	}
	if coll == nil {
		coll = LocalCollective{}
	}
	parts := make([][]float32, len(plan.Shards))
	for r, s := range plan.Shards {
		wsh := shardWeightRows(w, in, s.Lo, s.Hi)
		parts[r] = matRows(wsh, x, s.Width(), in)
	}
	return coll.AllGather(parts, plan)
}

// RowParallelMatMul computes y = x·Wᵀ (W row-major [out,in]) sharded across the
// CONTRACTION dimension by plan (plan.Dim must == in): rank r holds the [Lo,Hi) column
// band of W and the matching x segment, computes a PARTIAL out-vector, then
// coll.AllReduceSum adds the partials in rank order. This reassociates the reduction, so
// it is NOT bit-exact vs the monolithic matRows (it drifts ~1e-6, the fdot non-associativity
// parallel.go documents); it IS bit-exact vs RowParallelReference, which the gate pins. A
// nil collective defaults to LocalCollective.
func RowParallelMatMul(w, x []float32, out, in int, plan TPPlan, coll Collective) ([]float32, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	if plan.Dim != in {
		return nil, fmt.Errorf("model: RowParallelMatMul plan.Dim = %d, want in = %d", plan.Dim, in)
	}
	if len(x) != in {
		return nil, fmt.Errorf("model: RowParallelMatMul x len = %d, want in = %d", len(x), in)
	}
	if len(w) != out*in {
		return nil, fmt.Errorf("model: RowParallelMatMul w len = %d, want out*in = %d", len(w), out*in)
	}
	if coll == nil {
		coll = LocalCollective{}
	}
	parts := make([][]float32, len(plan.Shards))
	for r, s := range plan.Shards {
		wsh := shardWeightColumns(w, out, in, s.Lo, s.Hi)
		parts[r] = matRows(wsh, x[s.Lo:s.Hi], out, s.Width())
	}
	return coll.AllReduceSum(parts)
}

// RowParallelReference is the shard-grouped oracle RowParallelMatMul is pinned to
// bit-for-bit: for each output o, sum (in rank order) the fdot of W's [Lo,Hi) column band
// against x's matching segment. It deliberately reproduces the reassociated reduction —
// it is what the monolithic matRows WOULD compute if it summed per-shard partials instead
// of one fdot over the whole row — so the gate proves the sharded path adds nothing beyond
// the unavoidable, documented round-off, not a logic error.
func RowParallelReference(w, x []float32, out, in int, plan TPPlan) ([]float32, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	if plan.Dim != in {
		return nil, fmt.Errorf("model: RowParallelReference plan.Dim = %d, want in = %d", plan.Dim, in)
	}
	y := make([]float32, out)
	for o := 0; o < out; o++ {
		row := w[o*in : o*in+in]
		var acc float32
		for _, s := range plan.Shards {
			acc += fdot(row[s.Lo:s.Hi], x[s.Lo:s.Hi])
		}
		y[o] = acc
	}
	return y, nil
}

// TensorParallelFFN is the composed Megatron FFN block sharded across `plan` over the
// INTERMEDIATE dimension I (plan.Dim must == I): gate and up are COLUMN-parallel (each
// rank computes its I-band of silu(gate·x)*up·x locally — the intermediate is never
// gathered), and down is ROW-parallel over that same I-band (each rank's down columns
// against its local activation), with a SINGLE AllReduceSum producing the [H] output. It
// is the real "tensor parallelism within a layer" demonstration: one collective per block,
// matching the single-device FFN within the down-projection's documented AllReduce
// round-off (~1e-5), while each rank's activation band is bit-exact vs the monolith's
// corresponding slice.
//
// gateW/upW are row-major [I,H]; downW is row-major [H,I]; x is [H]; the result is [H]. A
// nil collective defaults to LocalCollective.
func TensorParallelFFN(gateW, upW, downW, x []float32, h, inter int, plan TPPlan, coll Collective) ([]float32, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	if plan.Dim != inter {
		return nil, fmt.Errorf("model: TensorParallelFFN plan.Dim = %d, want intermediate = %d", plan.Dim, inter)
	}
	if len(x) != h {
		return nil, fmt.Errorf("model: TensorParallelFFN x len = %d, want hidden = %d", len(x), h)
	}
	if len(gateW) != inter*h || len(upW) != inter*h {
		return nil, fmt.Errorf("model: TensorParallelFFN gate/up len = %d/%d, want inter*hidden = %d", len(gateW), len(upW), inter*h)
	}
	if len(downW) != h*inter {
		return nil, fmt.Errorf("model: TensorParallelFFN down len = %d, want hidden*inter = %d", len(downW), h*inter)
	}
	if coll == nil {
		coll = LocalCollective{}
	}
	return coll.AllReduceSum(tensorParallelFFNPartials(gateW, upW, downW, x, h, inter, plan))
}

// tensorParallelFFNPartials computes each rank's [hidden] down-projection partial — the
// per-rank result the FFN's single AllReduceSum combines. Factored out so TensorParallelFFN
// (which reduces via the Collective) and TensorParallelFFNReference (which sums in rank
// order directly) share the IDENTICAL partial computation and differ ONLY in the reduction,
// which is what the bit-exact reference is meant to pin. Assumes shapes are already
// validated by the caller.
func tensorParallelFFNPartials(gateW, upW, downW, x []float32, h, inter int, plan TPPlan) [][]float32 {
	parts := make([][]float32, len(plan.Shards))
	for r, s := range plan.Shards {
		w := s.Width()
		// Column-parallel gate/up over the I-band: bit-exact vs the monolith's [Lo,Hi)
		// slice of g and u.
		gShard := matRows(shardWeightRows(gateW, h, s.Lo, s.Hi), x, w, h)
		uShard := matRows(shardWeightRows(upW, h, s.Lo, s.Hi), x, w, h)
		for i := 0; i < w; i++ {
			gShard[i] = silu(gShard[i]) * uShard[i]
		}
		// Row-parallel down over the same I-band: this rank's down columns [Lo,Hi) against
		// its local activation, a [hidden] partial. The AllReduceSum is the one collective.
		downShard := shardWeightColumns(downW, h, inter, s.Lo, s.Hi)
		parts[r] = matRows(downShard, gShard, h, w)
	}
	return parts
}

// sumPartialsRankOrder sums equal-length per-rank partials in RANK ORDER (acc = parts[0];
// acc += parts[r]), independently of any Collective. It is the bit-exact spec the
// row-parallel reductions must satisfy: a Collective (Local today, NCCL later) is correct
// only if its AllReduceSum equals this. References use it so a reduction that reorders,
// drops, or double-counts a rank — which the loose vs-monolith bound cannot see, since it
// stays within round-off — is caught at max|Δ|=0.
func sumPartialsRankOrder(parts [][]float32) []float32 {
	if len(parts) == 0 {
		return nil
	}
	acc := append([]float32(nil), parts[0]...)
	for r := 1; r < len(parts); r++ {
		for i := range acc {
			acc[i] += parts[r][i]
		}
	}
	return acc
}

// TensorParallelFFNReference is the shard-grouped bit-exact oracle for TensorParallelFFN:
// the identical per-rank partials, summed in rank order directly rather than through the
// Collective. Pinning TensorParallelFFN == this at max|Δ|=0 proves the collective reduces
// the down-projection partials in rank order — the contract the row-parallel bit-exactness
// depends on — independently of the loose vs-monolith round-off bound. Validation mirrors
// TensorParallelFFN so the reference rejects the same malformed inputs.
func TensorParallelFFNReference(gateW, upW, downW, x []float32, h, inter int, plan TPPlan) ([]float32, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	if plan.Dim != inter {
		return nil, fmt.Errorf("model: TensorParallelFFNReference plan.Dim = %d, want intermediate = %d", plan.Dim, inter)
	}
	if len(x) != h {
		return nil, fmt.Errorf("model: TensorParallelFFNReference x len = %d, want hidden = %d", len(x), h)
	}
	if len(gateW) != inter*h || len(upW) != inter*h || len(downW) != h*inter {
		return nil, fmt.Errorf("model: TensorParallelFFNReference weight shape mismatch")
	}
	return sumPartialsRankOrder(tensorParallelFFNPartials(gateW, upW, downW, x, h, inter, plan)), nil
}
