package compute

// prefill.go — the host-tractable scaffold for issue #9 (B-001), "close the prefill
// throughput gap vs llama.cpp." The acceptance bullet that pins prefill "within 1.2×
// llama.cpp Q8_0" sustained at P=256/512/1024 is a WALL-CLOCK measurement on a CUDA
// device: it cannot be witnessed on a CUDA-less host, and is deferred to a CUDA bench
// node (see the report). This file ships the three pieces the perf work stands on, each
// the honest, hardware-independent form of one scope bullet:
//
//   1. PrefillRoofline / PrefillCostModel — "profile the prefill bottleneck." An ANALYTIC
//      roofline: per stage, the EXACT FLOPs and the EXACT bytes moved (weights +
//      activations), and their ratio (arithmetic intensity). It needs no timer and
//      fabricates no throughput; it locates the bottleneck STRUCTURALLY — which stage
//      dominates the FLOPs, and whether a stage is compute- or memory-bound relative to a
//      caller-supplied ridge point. Counting work is exact; only timing needs the device.
//      The model surfaces the key long-sequence fact for free: attention's FLOPs grow as
//      O(P²) while every projection/FFN GEMM grows as O(P), so attention overtakes the
//      GEMMs past some crossover length — and naive attention's intensity is a constant
//      ~0.5 FLOP/byte (deeply memory-bound), which is exactly why "optimize attention for
//      long sequences" is its own scope bullet (the flash/paged-attention motivation).
//
//   2. PrefillGEMM — "implement a batched-GEMM prefill kernel." A tiled panel×tile GEMM
//      skeleton (the blocking shape a device kernel uses), CPU-ref correct and BIT-EXACT
//      to Backend.BatchedMatMul: every output cell Y[t,o] is the same fdot(row_o, x_t),
//      so a blocked visit order touches the same independent cells and writes the same
//      bytes. Tiling changes only locality, never the result — so the "bit-exact results
//      unchanged" acceptance bullet holds by construction (and is witnessed in the tests).
//
//   3. PrefillGraphCapturer + CapturePrefillGraph — "CUDA graph for prefill." The capturer
//      interface is pure Go and ALWAYS compiled. The CUDA backend (cuda.go, built only
//      under -tags cuda) already implements GraphBegin/GraphEndLaunch/GraphReset, so it
//      satisfies this interface there; on a non-CUDA build no backend implements it,
//      CapturePrefillGraph takes the plain-execution fallback, and the whole file still
//      compiles and runs. That is the "CUDA-graph wiring stubs guarded so non-CUDA builds
//      compile" requirement: the seam exists at the HAL today, the device fills it later.

// q8Block is the Q8_0 group size the cost model charges per-block scale bytes against
// (matches QuantSpec.Block for Q8_0 / llama.cpp's block_q8_0). It is used only to size
// the scale side-channel in weightBytes; it is not a kernel parameter here.
const q8Block = 32

// StageCost is the analytic cost of one prefill stage (summed over the layers it appears
// in). FLOPs and the byte counts are EXACT — a count of the arithmetic and the operands
// the stage must touch — not a measurement; Intensity = FLOPs/(WeightBytes+ActBytes) is
// the roofline arithmetic intensity (FLOP per byte moved). No wall-clock time appears
// here by design: time needs the target device, the counts do not.
type StageCost struct {
	Name        string  // stable stage id, e.g. "q_proj", "attn", "ffn_down", "lm_head"
	FLOPs       int64   // multiply+add counted as 2 flops
	WeightBytes int64   // resident weight bytes the stage streams (0 for attention)
	ActBytes    int64   // activation bytes read+written (f32 traffic)
	Intensity   float64 // FLOPs / (WeightBytes+ActBytes) — roofline arithmetic intensity
}

// totalBytes is the denominator of Intensity, exposed so the caller can roofline against a
// machine's bandwidth without re-deriving it.
func (s StageCost) totalBytes() int64 { return s.WeightBytes + s.ActBytes }

// Bound classifies the stage on a roofline given a ridge point (the machine's peak
// FLOP/s ÷ peak bytes/s, in FLOP/byte). It bakes in NO hardware constants — the ridge is
// the caller's, measured on the target device — so this stays honest on a host that
// cannot measure either peak. A stage is compute-bound when its intensity sits above the
// ridge, memory-bound when below.
func (s StageCost) Bound(ridge float64) string {
	if s.Intensity >= ridge {
		return "compute"
	}
	return "memory"
}

// PrefillGeometry is the model shape + prefill length the cost model reasons over. It is
// the subset of model.Config the prefill arithmetic depends on; passed explicitly so the
// compute package holds no model state (the same discipline as KVConfig).
type PrefillGeometry struct {
	DModel      int   // residual width (a.k.a. hidden size)
	NHeads      int   // query heads
	NKVHeads    int   // key/value heads (GQA: ≤ NHeads)
	HeadDim     int   // per-head width
	DFF         int   // FFN inner width (gate/up out, down in)
	NLayers     int   // transformer blocks
	Vocab       int   // LM head output width
	P           int   // prefill length (the panel height) — P=256/512/1024 are the targets
	WeightDtype Dtype // weight storage format (F32 / Q8_0 / …) — selects weightBytes
}

// weightBytes is the resident size of an [out,in] weight in the geometry's dtype. For
// Q8_0 it charges the int8 codes PLUS the per-block f32 scales (the side-channel a real
// Q8 weight carries), so the memory-bound roofline of a quantized prefill is not
// understated; for float dtypes it is the dense element size.
func (g PrefillGeometry) weightBytes(out, in int) int64 {
	o, i := int64(out), int64(in)
	switch g.WeightDtype {
	case Q8_0:
		nblk := int64((in + q8Block - 1) / q8Block)
		return o*i + o*nblk*4 // codes (1B each) + per-block scales (4B each)
	default:
		return o * i * int64(g.WeightDtype.Bytes())
	}
}

// gemmCost is the analytic cost of a single Y[P,out] = X[P,in] @ Wᵀ panel GEMM: 2·P·out·in
// flops, the weight stream, and the f32 activation read (X) + write (Y). It is the shared
// shape behind every projection and FFN matmul in the prefill path.
func (g PrefillGeometry) gemmCost(name string, out, in, rows int) StageCost {
	flops := int64(2) * int64(rows) * int64(out) * int64(in)
	wb := g.weightBytes(out, in)
	ab := int64(4) * (int64(rows)*int64(in) + int64(rows)*int64(out))
	return StageCost{Name: name, FLOPs: flops, WeightBytes: wb, ActBytes: ab,
		Intensity: ratio(flops, wb+ab)}
}

// attnCost is the analytic cost of naive (un-tiled) causal self-attention over the P-token
// prefill, summed over heads and the two matmuls (scores Q·Kᵀ and the ΣwV output). Causal
// masking makes the work Σ_i(i+1) ≈ P²/2 keys per head, so FLOPs ≈ 2·NHeads·P²·HeadDim.
// It carries NO weights; its ActBytes is the KV re-read traffic a naive attention pays
// (each query row reads HeadDim×2 floats per causal key) — the traffic flash/paged
// attention exists to cut. The resulting intensity is a P-independent ~0.5 FLOP/byte:
// naive attention is memory-bound at every length, which is the long-sequence bottleneck.
func (g PrefillGeometry) attnCost() StageCost {
	P := int64(g.P)
	hd := int64(g.HeadDim)
	nh := int64(g.NHeads)
	causalPairs := P * P // 2·(P²/2): the i-th query attends i+1 keys, summed ≈ P²/2, ×2 matmuls
	flops := int64(2) * nh * causalPairs * hd
	// naive KV traffic: HeadDim floats per (query,key) for each of K and V, over the causal
	// triangle, every head — the re-read a fused attention avoids.
	actBytes := int64(4) * (nh * causalPairs * hd)
	return StageCost{Name: "attn", FLOPs: flops, WeightBytes: 0, ActBytes: actBytes,
		Intensity: ratio(flops, actBytes)}
}

// PrefillCostModel returns the per-stage analytic cost of the whole prefill at the given
// geometry: every per-layer stage summed across NLayers, then the single last-position LM
// head (prefill needs only the final token's logits). The slice is in forward order. This
// is the "profile the prefill bottleneck" deliverable — exact work counts that locate the
// bottleneck without a timer (Dominant picks the heaviest; the attn O(P²) term is what
// overtakes the GEMMs at long P).
func PrefillCostModel(g PrefillGeometry) []StageCost {
	qOut := g.NHeads * g.HeadDim
	kvOut := g.NKVHeads * g.HeadDim
	nL := g.NLayers

	perLayer := []StageCost{
		g.gemmCost("q_proj", qOut, g.DModel, g.P),
		g.gemmCost("k_proj", kvOut, g.DModel, g.P),
		g.gemmCost("v_proj", kvOut, g.DModel, g.P),
		g.attnCost(),
		g.gemmCost("o_proj", g.DModel, qOut, g.P),
		g.gemmCost("ffn_gate", g.DFF, g.DModel, g.P),
		g.gemmCost("ffn_up", g.DFF, g.DModel, g.P),
		g.gemmCost("ffn_down", g.DModel, g.DFF, g.P),
	}
	out := make([]StageCost, 0, len(perLayer)+1)
	for _, s := range perLayer {
		out = append(out, scaleStage(s, nL)) // sum the stage across all layers
	}
	// LM head runs once over the LAST position only (prefill emits one next-token dist).
	out = append(out, g.gemmCost("lm_head", g.Vocab, g.DModel, 1))
	return out
}

// PrefillRoofline aggregates a cost-model slice into the totals a profiler reports: total
// FLOPs, total bytes, the overall intensity, and the dominant stage (the bottleneck to
// attack first). It is a pure fold over PrefillCostModel's output.
type PrefillRoofline struct {
	Stages     []StageCost
	TotalFLOPs int64
	TotalBytes int64
	Intensity  float64
	Dominant   StageCost // the single heaviest stage by FLOPs
}

// Profile builds the roofline summary for a geometry — the one-call entry point a harness
// uses to answer "where is the prefill time going, structurally, at this P?".
func Profile(g PrefillGeometry) PrefillRoofline {
	stages := PrefillCostModel(g)
	r := PrefillRoofline{Stages: stages}
	for _, s := range stages {
		r.TotalFLOPs += s.FLOPs
		r.TotalBytes += s.totalBytes()
		if s.FLOPs > r.Dominant.FLOPs {
			r.Dominant = s
		}
	}
	r.Intensity = ratio(r.TotalFLOPs, r.TotalBytes)
	return r
}

// scaleStage multiplies a per-layer stage's counts by the layer count (intensity is a
// ratio, so it is unchanged by the scaling).
func scaleStage(s StageCost, n int) StageCost {
	s.FLOPs *= int64(n)
	s.WeightBytes *= int64(n)
	s.ActBytes *= int64(n)
	return s
}

func ratio(num, den int64) float64 {
	if den == 0 {
		return 0
	}
	return float64(num) / float64(den)
}

// ---- batched-GEMM prefill kernel skeleton ---------------------------------------

// PrefillGEMM computes Y[P,out] = X[P,in] @ Wᵀ over an out-tile × token-tile blocking —
// the panel×tile loop shape a device GEMM kernel uses to keep a weight row hot across the
// whole token panel — and is BIT-EXACT to Backend.BatchedMatMul. Each output cell
// Y[t,o] = fdot(W[o,:], X[t,:]) is independent, so any visit order computes the identical
// fdot and writes the identical bytes; tileO/tileT change only cache locality, never the
// result. A non-positive tile size falls back to a sane default, and a tile larger than
// its dimension simply spans the whole dimension — so the skeleton is safe at any P,
// including the P=256/512/1024 targets, and the bit-identity gate (R2) survives adoption.
//
// w is the row-major [out,in] weight, X the row-major [P,in] activation panel; the
// returned Y is row-major [P,out]. It is deliberately a bare-slice kernel (no Tensor
// plumbing) so it reads as the kernel a device lowers, and so the bit-exactness test can
// compare it cell-for-cell against fdot and BatchedMatMul.
func PrefillGEMM(w, X []float32, out, in, P, tileO, tileT int) []float32 {
	if tileO <= 0 {
		tileO = 64
	}
	if tileT <= 0 {
		tileT = 64
	}
	Y := make([]float32, P*out)
	for o0 := 0; o0 < out; o0 += tileO {
		oEnd := o0 + tileO
		if oEnd > out {
			oEnd = out
		}
		for t0 := 0; t0 < P; t0 += tileT {
			tEnd := t0 + tileT
			if tEnd > P {
				tEnd = P
			}
			for o := o0; o < oEnd; o++ {
				row := w[o*in : o*in+in]
				for t := t0; t < tEnd; t++ {
					Y[t*out+o] = fdot(row, X[t*in:t*in+in])
				}
			}
		}
	}
	return Y
}

// ---- CUDA-graph wiring seam (pure Go, always compiled) --------------------------

// PrefillGraphCapturer is the OPTIONAL capability a backend implements to capture a
// prefill op-stream into a replayable device graph — the CUDA-graph seam at the HAL. The
// CUDA backend (cuda.go) already exposes GraphBegin/GraphEndLaunch/GraphReset and so
// satisfies this interface under -tags cuda; the forward loop discovers it the same way
// it discovers every optional capability — a type-assertion on the Backend — and falls
// back to plain execution when it is absent (every non-CUDA build today). Because the
// interface is declared here, in the always-compiled file, the seam exists and type-checks
// with or without CUDA: the device just fills it in later.
type PrefillGraphCapturer interface {
	Backend
	// GraphBegin opens capture for one prefill replay unit; false means "not captured,
	// run eagerly" (the device may decline, e.g. capture disabled or unwarmed).
	GraphBegin() bool
	// GraphEndLaunch instantiates + launches + fences the captured graph.
	GraphEndLaunch()
	// GraphReset drops a kept exec graph (bound to one session's buffer addresses).
	GraphReset()
}

// CapturePrefillGraph runs body once, wrapped in a device graph capture when be supports
// it AND consents (GraphBegin returns true), and reports whether the capture path was
// taken. When be is not a PrefillGraphCapturer — every non-CUDA build, since cpu-ref does
// not implement it — body runs directly and the result is false. This is the guarded
// wiring the forward loop calls so the same prefill code path serves both a CUDA device
// (captured) and the CPU reference (eager), with no build tag at the call site.
func CapturePrefillGraph(be Backend, body func()) (captured bool) {
	gc, ok := be.(PrefillGraphCapturer)
	if !ok {
		body()
		return false
	}
	if !gc.GraphBegin() {
		body() // backend declined capture this unit — run eagerly
		return false
	}
	body()
	gc.GraphEndLaunch()
	return true
}

// ResetPrefillGraph drops any kept prefill exec graph on be, if be is a capturer (a no-op
// otherwise). The forward loop calls it at a session boundary, where the captured graph's
// bound buffer addresses are about to change.
func ResetPrefillGraph(be Backend) {
	if gc, ok := be.(PrefillGraphCapturer); ok {
		gc.GraphReset()
	}
}
