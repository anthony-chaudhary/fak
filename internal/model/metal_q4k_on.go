//go:build darwin && arm64 && cgo

package model

// metal_q4k_on.go — the Metal q4_k prefill GEMM dispatch (built on Apple Silicon with cgo).
// When s.MetalQ4K is set and a device is present, the resident-Q4_K hybrid prefill's
// q4_k-majority projection/MLP GEMMs run on the GPU via internal/metalgemm's q4_k dequant-GEMM
// (q4k.m) instead of the CPU q4kGemm. Each weight is uploaded once and cached per *Model.
//
// On Apple unified memory the q4_k upload path wraps the model's resident raw bytes with a
// no-copy MTLBuffer when Metal accepts the pointer. If it falls back to a copied Metal buffer,
// MetalQ4K can still be paired with FAK_Q4K_FREE_CPU=1 once all q4_k matmuls are GPU-routed.

import (
	"os"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

const (
	q4kMLPOutputSlabMaxTokens = 512
	q4kMLPOutputSlabMaxBytes  = 71_303_168
	q4kMLPOutputSlabMaxFloats = q4kMLPOutputSlabMaxBytes / 4
)

var (
	metalQ4KMu sync.Mutex
	// metalQ4KW caches one GPU q4_k weight handle per (model, weight-name). A nil entry is
	// cached too (upload failed / table full) so we don't retry the upload every token.
	metalQ4KW = map[*Model]map[string]*metalgemm.Q4KWeight{}
	// metalQ6KW caches one GPU Q6_K weight handle per (model, weight-name) for the fused MLP's
	// Q6_K down_proj. Same nil-caching policy as metalQ4KW. Guarded by metalQ4KMu.
	metalQ6KW = map[*Model]map[string]*metalgemm.Q6KWeight{}
	// metalQ8KW caches one GPU Q8_0 weight handle per (model, weight-name) for Q8-minority
	// prefill projections in the resident-Q4_K lane (full-attn q/k and Qwen3.6 linear_attn.*).
	// Same nil-caching policy as metalQ4KW. Guarded by metalQ4KMu.
	metalQ8KW = map[*Model]map[string]*metalgemm.Q8Weight{}
	// metalQ8Budget caches, per *Model, whether the Q8-minority GPU upload fits the device's
	// working-set budget (computed once by metalQ8UploadAllowed). Guarded by metalQ4KMu.
	metalQ8Budget = map[*Model]bool{}
	// metalQ8Exact records the immutable, all-or-nothing Qwen3.8 no-copy publication. Its order is
	// the canonical 272-name runtime band and therefore also the reverse teardown order.
	metalQ8Exact = map[*Model]*metalQ8ExactState{}
	// freeCPUCopyAfterUpload, when set, drops qt.raw after a successful GPU upload for single
	// residency. Default OFF: the CPU prefill/decode fallbacks (q4kGemm/q4kMatRows) still read
	// qt.raw and panic on nil when the GPU path isn't taken for some tensor (#1067). Opt in with
	// FAK_Q4K_FREE_CPU=1 only once every q4_k matmul is provably GPU-routed.
	freeCPUCopyAfterUpload = os.Getenv("FAK_Q4K_FREE_CPU") == "1"
	// q4kMMOnce guards the one-time FAK_Q4K_MM read that flips the batched GEMM to the simdgroup-
	// matrix kernel (metalgemm.SetGEMMUseMM). Read lazily on first prefill weight-upload, not at
	// package init, because it is a cgo call that needs the Metal device present.
	q4kMMOnce sync.Once
)

type metalQ8ExactState struct {
	names   []string
	handles []*metalgemm.Q8Weight
	err     error
}

func init() {
	releaseModelQ4KHandles = releaseMetalQ4KResidency
}

func (s *Session) metalExecution(operation metalgemm.ExecutionOperation, call func(*metalgemm.ExecutionObservation)) {
	observation := metalgemm.NewExecutionObservation(operation)
	call(observation)
	snapshot, err := observation.Snapshot()
	if s.PhaseProfiler != nil {
		s.PhaseProfiler.recordMetal(snapshot, err)
	}
	if err == nil {
		s.observeQwen35MetalExecutionSnapshot(snapshot)
	}
}

func (s *Session) recordMetalFallback(route MetalFallbackRoute) {
	if s != nil && s.PhaseProfiler != nil {
		s.PhaseProfiler.recordMetalFallback(route)
	}
}

func (s *Session) q4kGemmDispatch(name string, qt *q4kTensor, Xf []float32, P int) []float32 {
	if !s.MetalQ4K || !metalgemm.Available() {
		return q4kGemm(qt, Xf, P)
	}
	Y := make([]float32, P*qt.out)
	if !s.M.withMetalQ4K(name, qt, func(w *metalgemm.Q4KWeight) {
		if P == 1 {
			s.metalExecution(metalgemm.ExecutionQ4KGEMV, func(observation *metalgemm.ExecutionObservation) {
				w.GEMVWithEvents(Xf, Y, observation)
			})
			return
		}
		s.metalExecution(metalgemm.ExecutionQ4KGEMM, func(observation *metalgemm.ExecutionObservation) {
			w.GEMMWithEvents(Xf, P, Y, observation)
		})
	}) {
		route := MetalFallbackQ4KGEMMCPU
		if P == 1 {
			route = MetalFallbackQ4KGEMVPanelCPU
		}
		s.recordMetalFallback(route)
		if qt.lazy != nil {
			panic("model: lazy Q4_K Metal GEMM upload failed: " + name)
		}
		return q4kGemm(qt, Xf, P)
	}
	return Y
}

// q4kGemmGroupDispatch groups Q4_K projections that share one f32 activation panel Xf[P, in].
// Single-row panels use decode GEMV; larger panels use batched GEMM. Typical groups are a
// layer's q/k/v, gate/up, or GDN in_proj quad. Each group uses one Metal command buffer,
// collapsing the per-weight submit/sync round-trips that dominate this path.
// It returns one result slice per name with the q4_k-resident members filled and every other member
// (Q8/Q6_K minority, or a declined upload) left nil, so the caller fills those via its existing
// per-weight `proj`. Returns nil entirely — caller loops per-weight — unless MetalQ4K is on, a
// device is present, AND at least two members are q4_k-resident (so a command buffer is worth
// amortizing). Each filled slice is [P*out] token-major and uses the same P-specific kernel as
// q4kGemmDispatch.
func (s *Session) q4kGemmGroupDispatch(names []string, Xf []float32, P int) [][]float32 {
	if !s.MetalQ4K || !metalgemm.Available() || P <= 0 {
		return nil
	}
	n := len(names)
	ws := make([]*metalgemm.Q4KWeight, 0, n)
	pos := make([]int, 0, n) // index in names of each grouped (q4_k-resident, uploaded) member
	for i, name := range names {
		qt := s.M.q4kw[name]
		if qt == nil {
			continue
		}
		w := s.M.metalQ4KWeight(name, qt)
		if w == nil {
			continue
		}
		ws = append(ws, w)
		pos = append(pos, i)
	}
	if len(ws) < 2 {
		return nil // not enough resident members to amortize a command buffer
	}
	var grouped [][]float32
	if P == 1 {
		s.metalExecution(metalgemm.ExecutionQ4KGEMVGroup, func(observation *metalgemm.ExecutionObservation) {
			grouped = metalgemm.GEMVGroupWithEvents(ws, Xf, observation)
		})
	} else {
		s.metalExecution(metalgemm.ExecutionQ4KGEMMGroup, func(observation *metalgemm.ExecutionObservation) {
			if q4kMLPOutputSlabSelected(s.Q4KGateUpOutputSlab, names, pos, P) {
				need := 0
				for _, w := range ws {
					if w.Out > (q4kMLPOutputSlabMaxFloats-need)/P {
						need = q4kMLPOutputSlabMaxFloats + 1
						break
					}
					need += P * w.Out
				}
				if need <= q4kMLPOutputSlabMaxFloats {
					slab := s.q4kMLPOutputSlab
					allocated := cap(slab) < need
					if allocated {
						slab = make([]float32, need)
					} else {
						slab = slab[:need]
					}
					grouped = metalgemm.GEMMGroupIntoWithEvents(ws, Xf, P, slab, observation)
					if grouped != nil {
						// Do not publish new backing into Session until the synchronous Metal call
						// has completed and returned its aliases.
						s.q4kMLPOutputSlab = slab
						stats := &s.q4kMLPOutputSlabStats
						stats.Calls++
						if allocated {
							stats.Allocations++
						} else {
							stats.Reuses++
						}
						if highWater := uint64(len(slab)) * 4; highWater > stats.HighWaterBytes {
							stats.HighWaterBytes = highWater
						}
					}
					return
				}
			}
			grouped = metalgemm.GEMMGroupWithEvents(ws, Xf, P, observation)
		})
	}
	if grouped == nil {
		route := MetalFallbackQ4KGEMMGroupDispatch
		if P == 1 {
			route = MetalFallbackQ4KGEMVGroupDispatch
		}
		s.recordMetalFallback(route)
		return nil
	}
	out := make([][]float32, n)
	for j, i := range pos {
		out[i] = grouped[j]
	}
	// Ungrouped members (Q8/Q6_K minority, or a declined upload) stay nil; the caller fills them
	// via its per-weight proj so the panel-quantized Q8 path is reused unchanged.
	return out
}

// q4kMLPOutputSlabSelected narrows experimental reuse to an exact same-layer gate/up pair. The
// feature remains default-off pending the #9102 Mac KEEP gate; every other group and panel keeps
// GEMMGroupWithEvents' call-owned allocation.
func q4kMLPOutputSlabSelected(enabled bool, names []string, pos []int, P int) bool {
	if !enabled || P <= 1 || P > q4kMLPOutputSlabMaxTokens || len(names) != 2 || len(pos) != 2 || pos[0] != 0 || pos[1] != 1 {
		return false
	}
	const gateSuffix = "mlp.gate_proj.weight"
	const upSuffix = "mlp.up_proj.weight"
	if !strings.HasSuffix(names[0], gateSuffix) || !strings.HasSuffix(names[1], upSuffix) {
		return false
	}
	return strings.TrimSuffix(names[0], gateSuffix) == strings.TrimSuffix(names[1], upSuffix)
}

// q8GemmDispatch is the prefill-GEMM twin for the Q8-minority projections in the resident-Q4_K
// path. Pure CPU builds and non-Metal sessions use qGemm8. With MetalQ4K enabled, the Q8 panel
// runs through metalgemm's batched Q8 GEMM, so Qwen3.6's full-attn q/k and linear_attn.* no longer
// cap prefill on the CPU side (#1087). The caller supplies the already-quantized activation panel.
func (s *Session) q8GemmDispatch(name string, qt *q8Tensor, Xq *q8Panel) []float32 {
	if !s.MetalQ4K || !metalgemm.Available() {
		return qGemm8(qt, Xq)
	}
	w := s.M.metalQ8Weight(name, qt)
	if w == nil {
		s.recordMetalFallback(MetalFallbackQ8GEMMCPU)
		return qGemm8(qt, Xq)
	}
	Y := make([]float32, Xq.P*qt.out)
	s.metalExecution(metalgemm.ExecutionQ8GEMM, func(observation *metalgemm.ExecutionObservation) {
		w.GEMMWithEvents(Xq.q, Xq.d, Xq.P, Y, observation)
	})
	return Y
}

// q8GemmGroupDispatch is the grouped prefill-GEMM twin for Q8-minority projections that share one
// activation panel. Qwen3.6 linear_attn in-proj groups are all Q8 because the reorder keeps them out
// of raw-Q4_K residency; grouping them pays one Metal command-buffer roundtrip instead of one per
// projection. Non-Q8 names are left nil for the caller's existing fallback.
func (s *Session) q8GemmGroupDispatch(names []string, Xq *q8Panel, P int) [][]float32 {
	if os.Getenv("FAK_Q8_GEMM_GROUP") != "1" || !s.MetalQ4K || !metalgemm.Available() || Xq == nil || P <= 0 {
		return nil
	}
	n := len(names)
	ws := make([]*metalgemm.Q8Weight, 0, n)
	pos := make([]int, 0, n)
	for i, name := range names {
		if s.M.q4kw[name] != nil || s.M.kqw[name] != nil {
			continue
		}
		qt := s.M.q8w[name]
		if qt == nil {
			continue
		}
		w := s.M.metalQ8Weight(name, qt)
		if w == nil {
			continue
		}
		ws = append(ws, w)
		pos = append(pos, i)
	}
	if len(ws) < 2 {
		return nil
	}
	var grouped [][]float32
	s.metalExecution(metalgemm.ExecutionQ8GEMMGroup, func(observation *metalgemm.ExecutionObservation) {
		grouped = metalgemm.GEMMGroupQ8WithEvents(ws, Xq.q, Xq.d, P, observation)
	})
	if grouped == nil {
		s.recordMetalFallback(MetalFallbackQ8GEMMGroupDispatch)
		return nil
	}
	out := make([][]float32, n)
	for j, i := range pos {
		out[i] = grouped[j]
	}
	return out
}

// kQuantGemmDispatch is the prefill-GEMM twin for resident K-quant tensors in the q4_k_m mix. The
// dense down_proj tensors load as Q6_K in kqw; under MetalQ4K they can use the resident Q6_K GEMM
// instead of the CPU kQuantMatRowsIntoBatch loop. Other K-quant kinds stay on the proven CPU path.
func (s *Session) kQuantGemmDispatch(name string, qt *kQuantTensor, Xf []float32, P int) []float32 {
	Y := make([]float32, P*qt.out)
	if !s.MetalQ4K || !metalgemm.Available() || qt.kind != kindQ6K {
		kQuantMatRowsIntoBatch(qt, Xf, P, Y)
		return Y
	}
	w := s.M.metalQ6KWeight(name, qt)
	if w == nil {
		s.recordMetalFallback(MetalFallbackQ6KGEMMCPU)
		kQuantMatRowsIntoBatch(qt, Xf, P, Y)
		return Y
	}
	s.metalExecution(metalgemm.ExecutionQ6KGEMM, func(observation *metalgemm.ExecutionObservation) {
		w.GEMMWithEvents(Xf, P, Y, observation)
	})
	return Y
}

// kQuantMatRowsIntoDispatch is the decode/head GEMV twin for resident K-quant tensors. Qwen3.6
// q4_k_m commonly stores the LM head as Q6_K in kqw; under MetalQ4K this keeps that head projection
// on the same resident Metal path as Q6_K MLP down_proj instead of paying the CPU kQuantMatRows
// escape every generated token.
func (s *Session) kQuantMatRowsIntoDispatch(name string, qt *kQuantTensor, xf, y []float32) {
	if !s.MetalQ4K || !metalgemm.Available() || qt.kind != kindQ6K {
		kQuantMatRowsInto(qt, xf, y)
		return
	}
	w := s.M.metalQ6KWeight(name, qt)
	if w == nil {
		s.recordMetalFallback(MetalFallbackQ6KGEMVCPU)
		kQuantMatRowsInto(qt, xf, y)
		return
	}
	s.metalExecution(metalgemm.ExecutionQ6KGEMV, func(observation *metalgemm.ExecutionObservation) {
		w.GEMVWithEvents(xf, y, observation)
	})
}

// q4kMatRowsDispatch is the decode-GEMV twin of q4kGemmDispatch: under MetalQ4K it runs the q4_k
// GEMV on the GPU (q4k_gemv) instead of the CPU q4kMatRows. Routing BOTH decode and prefill q4_k
// matmuls to the GPU is what lets metalQ4KWeight free the CPU copy (single residency) — the fix
// for the double-residency memory pressure that made the 27B Metal path a regression.
func (s *Session) q4kMatRowsDispatch(name string, qt *q4kTensor, xf []float32) []float32 {
	if !s.MetalQ4K || !metalgemm.Available() {
		return q4kMatRows(qt, xf)
	}
	y := make([]float32, qt.out)
	if !s.M.withMetalQ4K(name, qt, func(w *metalgemm.Q4KWeight) {
		s.metalExecution(metalgemm.ExecutionQ4KGEMV, func(observation *metalgemm.ExecutionObservation) {
			w.GEMVWithEvents(xf, y, observation)
		})
	}) {
		s.recordMetalFallback(MetalFallbackQ4KGEMVCPU)
		if qt.lazy != nil {
			panic("model: lazy Q4_K Metal GEMV upload failed: " + name)
		}
		return q4kMatRows(qt, xf)
	}
	return y
}

// q8MatRowsDispatch is the decode-GEMV twin for Q8-minority projections in the resident-Q4_K
// lane. Qwen3.6's linear_attn.* and some full-attention projections are not raw-Q4_K eligible, so
// leaving them on CPU creates a host/device ping-pong inside the otherwise-resident Metal decode.
// With MetalQ4K enabled and the Q8 upload budget satisfied, this runs the Q8_0 GEMV on Metal; if
// upload is declined (tight unified-memory budget, no device, or table full) it preserves the
// existing CPU qMatRows fallback.
func (s *Session) q8MatRowsDispatch(name string, qt *q8Tensor, xf []float32) []float32 {
	if !s.MetalQ4K || !metalgemm.Available() {
		return qMatRows(qt, s.quantizeVecQ8(xf))
	}
	w := s.M.metalQ8Weight(name, qt)
	if w == nil {
		s.recordMetalFallback(MetalFallbackQ8GEMVCPU)
		return qMatRows(qt, s.quantizeVecQ8(xf))
	}
	qv := s.quantizeVecQ8(xf)
	y := make([]float32, qt.out)
	s.metalExecution(metalgemm.ExecutionQ8GEMV, func(observation *metalgemm.ExecutionObservation) {
		w.GEMVWithEvents(qv.q, qv.d, y, observation)
	})
	return y
}

// q4kGroupDispatch runs a group of matmuls that share one f32 activation xf (a layer's gate/up,
// q/k/v, or the GDN in_proj quad) through resident Metal paths: q4_k-resident members go through
// metalgemm.GEMVGroup on the f32 activation, and Q8-minority members go through GEMVGroupQ8 on one
// shared Q8_0 activation. Any Q6_K member, declined upload, or singleton not already covered falls
// back to the per-call dispatch. Returns nil — so the caller (mulGroup) loops the historical path —
// only when no member could be Metal-routed. Results are bit-identical to calling the per-name
// dispatches, up to the existing Metal float-order tolerance.
func (s *Session) q4kGroupDispatch(names []string, xf []float32, outs []int) [][]float32 {
	if !s.MetalQ4K || !metalgemm.Available() {
		return nil
	}
	n := len(names)
	out := make([][]float32, n)
	routed := false

	q4ws := make([]*metalgemm.Q4KWeight, 0, n)
	q4pos := make([]int, 0, n) // index in names of each grouped (q4_k-resident, uploaded) member
	q8ws := make([]*metalgemm.Q8Weight, 0, n)
	q8pos := make([]int, 0, n) // index in names of each grouped (Q8-minority, uploaded) member
	for i, name := range names {
		if qt := s.M.q4kw[name]; qt != nil {
			w := s.M.metalQ4KWeight(name, qt) // uploads once + frees the CPU copy on success
			if w == nil {
				continue
			}
			q4ws = append(q4ws, w)
			q4pos = append(q4pos, i)
			continue
		}
		if s.M.kqw[name] != nil {
			continue // Q5_K/Q6_K stays on the proven k-quant CPU path unless a fused MLP handles it.
		}
		qt := s.M.q8w[name]
		if qt == nil {
			continue
		}
		w := s.M.metalQ8Weight(name, qt)
		if w == nil {
			continue
		}
		q8ws = append(q8ws, w)
		q8pos = append(q8pos, i)
	}

	if len(q4ws) >= 2 {
		var grouped [][]float32
		s.metalExecution(metalgemm.ExecutionQ4KGEMVGroup, func(observation *metalgemm.ExecutionObservation) {
			grouped = metalgemm.GEMVGroupWithEvents(q4ws, xf, observation)
		})
		if grouped != nil {
			for j, i := range q4pos {
				out[i] = grouped[j]
			}
			routed = true
		} else {
			s.recordMetalFallback(MetalFallbackQ4KGEMVGroupDispatch)
		}
	}

	var qv q8Vec
	qvOK := false
	getQ8 := func() q8Vec {
		if !qvOK {
			qv = s.quantizeVecQ8(xf)
			qvOK = true
		}
		return qv
	}
	if len(q8ws) > 0 {
		q := getQ8()
		var grouped [][]float32
		s.metalExecution(metalgemm.ExecutionQ8GEMVGroup, func(observation *metalgemm.ExecutionObservation) {
			grouped = metalgemm.GEMVGroupQ8WithEvents(q8ws, q.q, q.d, observation)
		})
		if grouped != nil {
			for j, i := range q8pos {
				out[i] = grouped[j]
			}
			routed = true
		} else {
			s.recordMetalFallback(MetalFallbackQ8GEMVGroupDispatch)
		}
	}

	if !routed {
		return nil
	}

	// Fill every member not covered by a grouped Metal dispatch, exactly as sessionQ4KKernel.mul.
	for i, name := range names {
		if out[i] != nil {
			continue
		}
		if qt := s.M.q4kw[name]; qt != nil {
			out[i] = s.q4kMatRowsDispatch(name, qt, xf) // GPU upload declined → its own dispatch
		} else if qt := s.M.kqw[name]; qt != nil {
			y := make([]float32, qt.out)
			s.kQuantMatRowsIntoDispatch(name, qt, xf, y)
			out[i] = y
		} else {
			qt := s.M.q8(name)
			q := getQ8()
			s.recordMetalFallback(MetalFallbackQ4KGroupQ8CPU)
			out[i] = qMatRows(qt, q)
		}
	}
	return out
}

// q4kFusedMLP runs the dense SwiGLU MLP (gate/up/silu/down) for one decode token entirely on the
// GPU in ONE command buffer (the intermediate-wide buffer stays resident) when MetalQ4K is on and
// all three weights are q4_k-resident + uploaded. Returns nil otherwise so the caller uses the
// per-matmul path. The Metal kernel is silu-only and adds no bias, so the caller must gate on a
// non-GELU activation and bias-free MLP. Bit-identical to the per-matmul path up to GPU float-order.
func (s *Session) q4kFusedMLP(gateName, upName, downName string, x []float32) []float32 {
	if !s.MetalQ4K || !metalgemm.Available() {
		return nil
	}
	gt, ut := s.M.q4kw[gateName], s.M.q4kw[upName]
	if gt == nil || ut == nil {
		return nil
	}
	gw := s.M.metalQ4KWeight(gateName, gt)
	uw := s.M.metalQ4KWeight(upName, ut)
	if gw == nil || uw == nil {
		return nil
	}
	// Down is Q4_K-resident → the all-Q4_K fused path (unchanged).
	if dt := s.M.q4kw[downName]; dt != nil {
		dw := s.M.metalQ4KWeight(downName, dt)
		if dw == nil {
			return nil
		}
		y := make([]float32, dt.out)
		ok := false
		s.metalExecution(metalgemm.ExecutionQ4KFusedMLP, func(observation *metalgemm.ExecutionObservation) {
			ok = metalgemm.FusedMLPWithEvents(gw, uw, dw, x, y, observation)
		})
		if !ok {
			s.recordMetalFallback(MetalFallbackFusedMLPDispatch)
			return nil
		}
		return y
	}
	// Down is Q6_K-resident (the q4_k_m down_proj case) → the mixed-quant fused path. This is the
	// Stage B cap-lift: previously a Q6_K down made q4kFusedMLP decline and every such expert fell
	// to the per-matmul path; now gate/up stay Q4_K and only stage 3 runs the Q6_K GEMV.
	if dq := s.M.kqw[downName]; dq != nil && dq.kind == kindQ6K {
		dw := s.M.metalQ6KWeight(downName, dq)
		if dw == nil {
			return nil
		}
		y := make([]float32, dq.out)
		ok := false
		s.metalExecution(metalgemm.ExecutionQ4KFusedMLPQ6Down, func(observation *metalgemm.ExecutionObservation) {
			ok = metalgemm.FusedMLPQ6DownWithEvents(gw, uw, dw, x, y, observation)
		})
		if !ok {
			s.recordMetalFallback(MetalFallbackFusedMLPQ6DownDispatch)
			return nil
		}
		return y
	}
	return nil
}

// q4kFusedMLPBatch runs the top-k routed experts' fused SwiGLU MLP (Q4_K gate/up, Q6_K down) in ONE
// Metal command buffer and returns each expert's [H] output (row e = experts[e]'s down result). It is
// the batched decode lever (#1382): the per-expert q4kFusedMLP fires one command buffer per expert, so
// a top-k MoE layer pays the ~360us submit/sync k×; this pays it once. Declines (returns nil, caller
// runs the per-expert loop) unless MetalQ4K is on, every expert's gate/up is resident Q4_K and its down
// is resident Q6_K, and the whole batch shares one geometry — the q4_k_m residency the fused path needs.
// The gate-weighted sum stays on the host so the routed-delta reduction order matches the loop exactly.
func (s *Session) q4kFusedMLPBatch(gate, up, down []string, x []float32) [][]float32 {
	if !s.MetalQ4K || !metalgemm.Available() {
		return nil
	}
	n := len(gate)
	if n == 0 || len(up) != n || len(down) != n {
		return nil
	}
	gws := make([]*metalgemm.Q4KWeight, n)
	uws := make([]*metalgemm.Q4KWeight, n)
	dws := make([]*metalgemm.Q6KWeight, n)
	var dout int
	for e := 0; e < n; e++ {
		gt, ut := s.M.q4kw[gate[e]], s.M.q4kw[up[e]]
		if gt == nil || ut == nil {
			return nil
		}
		dq := s.M.kqw[down[e]]
		if dq == nil || dq.kind != kindQ6K {
			return nil // a Q4_K-down expert (or missing) — not this fused path's residency
		}
		gws[e] = s.M.metalQ4KWeight(gate[e], gt)
		uws[e] = s.M.metalQ4KWeight(up[e], ut)
		dws[e] = s.M.metalQ6KWeight(down[e], dq)
		if gws[e] == nil || uws[e] == nil || dws[e] == nil {
			return nil
		}
		dout = dq.out
	}
	ycat := make([]float32, n*dout)
	ok := false
	s.metalExecution(metalgemm.ExecutionQ4KFusedMLPQ6DownBatch, func(observation *metalgemm.ExecutionObservation) {
		ok = metalgemm.FusedMLPQ6DownBatchWithEvents(gws, uws, dws, x, ycat, observation)
	})
	if !ok {
		s.recordMetalFallback(MetalFallbackFusedMLPBatchDispatch)
		return nil
	}
	out := make([][]float32, n)
	for e := 0; e < n; e++ {
		out[e] = ycat[e*dout : (e+1)*dout]
	}
	return out
}

// metalQ6KWeight returns this model's GPU Q6_K handle for `name`, uploading the raw 210-B blocks
// once (cached per *Model, nil cached too). The Q6_K resident store backs the fused MLP's down_proj
// when a q4_k_m GGUF quantizes down to Q6_K; gate/up stay Q4_K via metalQ4KWeight.
func (m *Model) metalQ6KWeight(name string, qt *kQuantTensor) *metalgemm.Q6KWeight {
	metalQ4KMu.Lock()
	defer metalQ4KMu.Unlock()
	tbl := metalQ6KW[m]
	if tbl == nil {
		tbl = map[string]*metalgemm.Q6KWeight{}
		metalQ6KW[m] = tbl
	}
	if w, ok := tbl[name]; ok {
		return w
	}
	w := metalgemm.UploadQ6K(qt.raw, qt.out, qt.in)
	tbl[name] = w
	return w
}

// metalQ8UploadAllowed reports (and caches per *Model) whether this model's Q8-minority
// projections may be uploaded to the GPU without breaching the device working-set budget. When it
// returns false, both the bulk pre-upload (metalQ8Weights) and the lazy per-call upload
// (metalQ8Weight) decline, so the Q8 minority stays on the proven CPU qGemm8 path — exactly the
// pre-#1087 behavior, which serves the 27B on a 36 GiB Mac without the OOM. Callers already hold
// no lock; this takes metalQ4KMu to guard the cache.
func (m *Model) metalQ8UploadAllowed() bool {
	metalQ4KMu.Lock()
	defer metalQ4KMu.Unlock()
	if v, ok := metalQ8Budget[m]; ok {
		return v
	}
	r := m.ResidentReport()
	deviceTotal := int64(0)
	if total, ok := metalgemm.DeviceMemoryTotal(); ok {
		deviceTotal = int64(total)
	}
	allowed := q8UploadFits(r.TotalResidentBytes, r.Q8Bytes, deviceTotal, os.Getenv("FAK_METAL_Q8_UPLOAD"))
	metalQ8Budget[m] = allowed
	return allowed
}

// metalQ8Weight returns this model's GPU Q8 handle for `name`, uploading the Q8_0 codes/scales
// once (cached per *Model, nil cached too). It backs batched Metal prefill for the Q8-minority
// projections in the resident-Q4_K lane. Declines (returns nil, caller stays on CPU qGemm8) when
// the device working-set budget can't absorb the additive Q8 GPU copy (metalQ8UploadAllowed).
func (m *Model) metalQ8Weight(name string, qt *q8Tensor) *metalgemm.Q8Weight {
	if _, err := qwen38MetalQ8RuntimeNames(m.Cfg); err == nil {
		if m.promoteMetalQ8Residency() != nil {
			return nil
		}
		metalQ4KMu.Lock()
		defer metalQ4KMu.Unlock()
		return metalQ8KW[m][name]
	}
	// Budget gate BEFORE the lock (metalQ8UploadAllowed takes metalQ4KMu itself; sync.Mutex is
	// not reentrant). On a tight device the additive Q8 GPU copy would OOM the serve, so decline
	// here and let q8GemmDispatch fall back to the CPU qGemm8 — the pre-#1087, non-OOM path.
	if !m.metalQ8UploadAllowed() {
		return nil
	}
	metalQ4KMu.Lock()
	defer metalQ4KMu.Unlock()
	tbl := metalQ8KW[m]
	if tbl == nil {
		tbl = map[string]*metalgemm.Q8Weight{}
		metalQ8KW[m] = tbl
	}
	if w, ok := tbl[name]; ok {
		return w
	}
	w := metalgemm.UploadQ8(qt.q, qt.d, qt.out, qt.in)
	tbl[name] = w
	return w
}

func (m *Model) promoteMetalQ8Residency() error {
	names, err := qwen38MetalQ8RuntimeNames(m.Cfg)
	if err != nil {
		return err
	}
	metalQ4KMu.Lock()
	defer metalQ4KMu.Unlock()
	if state, ok := metalQ8Exact[m]; ok {
		return state.err
	}
	r := m.ResidentReport()
	deviceTotal := int64(0)
	if total, ok := metalgemm.DeviceMemoryTotal(); ok {
		deviceTotal = int64(total)
	}
	if err := q8AliasFits(r.TotalResidentBytes, deviceTotal, os.Getenv("FAK_METAL_Q8_UPLOAD")); err != nil {
		metalQ8Exact[m] = &metalQ8ExactState{err: err}
		return err
	}
	handles, err := buildAllOrNothing(names, func(name string) (*metalgemm.Q8Weight, error) {
		if m.q4kw[name] != nil || m.kqw[name] != nil {
			return nil, &MetalQ8ResidencyUnavailableError{Reason: "promised Q8 projection resolved to another quant type: " + name}
		}
		qt := m.q8w[name]
		if qt == nil {
			return nil, &MetalQ8ResidencyUnavailableError{Reason: "missing promised Q8 projection: " + name}
		}
		w := metalgemm.AliasQ8(qt.q, qt.d, qt.out, qt.in)
		if w == nil || !w.NoCopy() {
			if w != nil {
				w.Release()
			}
			return nil, &MetalQ8ResidencyUnavailableError{Reason: "no-copy Metal alias declined: " + name}
		}
		return w, nil
	}, func(w *metalgemm.Q8Weight) { w.Release() })
	if err != nil {
		metalQ8Exact[m] = &metalQ8ExactState{err: err}
		return err
	}
	tbl := make(map[string]*metalgemm.Q8Weight, len(names))
	for i, name := range names {
		tbl[name] = handles[i]
	}
	metalQ8KW[m] = tbl // immutable publication: readers only retrieve handles after this assignment.
	metalQ8Exact[m] = &metalQ8ExactState{names: append([]string(nil), names...), handles: handles}
	return nil
}

func (m *Model) releaseMetalQ8Residency() {
	metalQ4KMu.Lock()
	state := metalQ8Exact[m]
	delete(metalQ8Exact, m)
	tbl := metalQ8KW[m]
	delete(metalQ8KW, m)
	delete(metalQ8Budget, m)
	metalQ4KMu.Unlock()
	if state != nil && len(state.handles) > 0 {
		for i := len(state.handles) - 1; i >= 0; i-- {
			state.handles[i].Release()
		}
		return
	}
	// Preserve UploadQ8 compatibility while giving copied per-model handles deterministic teardown.
	for _, w := range tbl {
		if w != nil {
			w.Release()
		}
	}
}

// metalQ4KWeights uploads all Q4_K projection weights for this model to the GPU once,
// caching them per *Model. This is the prefill-weight-upload twin of metalWeights(): it
// uploads every q4_k-resident projection (q/k/v/o, gate/up/down) upfront so the prefill
// loop never incurs a per-call GPU round-trip. The lazy upload path in metalQ4KWeight
// caps warm prefill at ~7x under llama.cpp (#1113); calling this before the layer loop
// restores the full prefill speed by amortizing all H2D copies up front. Returns the map
// (read-only) so the caller can verify upload success; nil on non-Metal builds.
func (m *Model) metalQ4KWeights() map[string]bool {
	if !metalgemm.Available() {
		return nil
	}
	// Opt into the simdgroup-matrix (hardware MMA) batched GEMM when FAK_Q4K_MM=1. Measured ~1.3x
	// the scalar register-tile GEMM at the Qwen3.6-27B gate/up prefill shape (1465 vs 1113 GFLOP/s
	// on the M3 Pro), cosine 1.0 vs the CPU f32 reference. Default OFF (the scalar kernel stays the
	// proven path) until the MMA variant earns auto-enable. Set once per process — cheap and
	// idempotent on the metalgemm side.
	q4kMMOnce.Do(func() { metalgemm.SetGEMMUseMM(os.Getenv("FAK_Q4K_MM") == "1") })
	uploaded := map[string]bool{}
	cfg := m.Cfg
	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(str string) string { return layerName(l, str) }
		for _, name := range denseProjectionNames(lp) {
			qt := m.q4kw[name]
			if qt == nil {
				continue // Q8 minority — not a q4_k-resident projection
			}
			if qt.lazy != nil {
				uploaded[name] = true // lazy weights stage at the operation seam
				continue
			}
			w := m.metalQ4KWeight(name, qt)
			uploaded[name] = w != nil
		}
	}
	return uploaded
}

// metalQ8Weights uploads the Q8-minority projection weights for this model to the GPU once. It
// deliberately skips names already present in q4kw or kqw: those route through Q4_K/Q6_K resident
// kernels, and uploading their Q8 copies would waste unified memory.
func (m *Model) metalQ8Weights() map[string]bool {
	if !metalgemm.Available() {
		return nil
	}
	if names, err := qwen38MetalQ8RuntimeNames(m.Cfg); err == nil {
		if m.promoteMetalQ8Residency() != nil {
			return nil
		}
		uploaded := make(map[string]bool, len(names))
		for _, name := range names {
			uploaded[name] = true
		}
		return uploaded
	}
	// Skip the whole bulk pre-upload when the device budget can't absorb the additive Q8 GPU
	// copy — otherwise the 7 GiB projection store doubles and the serve is SIGKILLed at first
	// prefill (#1087 OOM). metalQ8Weight would decline each tensor anyway; returning early keeps
	// the intent legible and avoids the pointless per-tensor budget re-checks.
	if !m.metalQ8UploadAllowed() {
		return nil
	}
	uploaded := map[string]bool{}
	add := func(name string) {
		if m.q4kw[name] != nil || m.kqw[name] != nil {
			return
		}
		qt := m.q8w[name]
		if qt == nil {
			return
		}
		w := m.metalQ8Weight(name, qt)
		uploaded[name] = w != nil
	}
	cfg := m.Cfg
	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(str string) string { return layerName(l, str) }
		for _, name := range denseProjectionNames(lp) {
			add(name)
		}
		if cfg.isLinearAttnLayer(l) {
			for _, name := range []string{
				lp("linear_attn.in_proj_qkv.weight"), lp("linear_attn.in_proj_z.weight"),
				lp("linear_attn.in_proj_b.weight"), lp("linear_attn.in_proj_a.weight"),
				lp("linear_attn.out_proj.weight"),
			} {
				add(name)
			}
		}
	}
	return uploaded
}

func denseProjectionNames(lp func(string) string) []string {
	return []string{
		lp("self_attn.q_proj.weight"), lp("self_attn.k_proj.weight"),
		lp("self_attn.v_proj.weight"), lp("self_attn.o_proj.weight"),
		lp("mlp.gate_proj.weight"), lp("mlp.up_proj.weight"), lp("mlp.down_proj.weight"),
	}
}

func (m *Model) withMetalQ4K(name string, qt *q4kTensor, use func(*metalgemm.Q4KWeight)) bool {
	if qt == nil {
		return false
	}
	w := m.metalQ4KWeight(name, qt)
	if w == nil {
		return false
	}
	use(w)
	return true
}

// metalQ4KWeight returns this model's GPU q4_k handle for name, uploading the raw blocks once.
// A streamed tensor inside a retained mmap shard is bound directly through the offset-aware
// no-copy upload. Other descriptors read into page-aligned storage only on first use. In both
// cases the cached handle owns the runtime residency and later tokens do no checkpoint I/O.
// Resident tensors keep the historical CPU-fallback lifetime below.
func (m *Model) metalQ4KWeight(name string, qt *q4kTensor) *metalgemm.Q4KWeight {
	metalQ4KMu.Lock()
	defer metalQ4KMu.Unlock()
	tbl := metalQ4KW[m]
	if tbl == nil {
		tbl = map[string]*metalgemm.Q4KWeight{}
		metalQ4KW[m] = tbl
	}
	if w, ok := tbl[name]; ok {
		return w
	}
	var w *metalgemm.Q4KWeight
	raw := qt.raw
	mappedAttempt := false
	copiedUpload := false
	if qt.lazy != nil {
		if span, offset, ok := qt.mappedRaw(); ok {
			mappedAttempt = true
			w = metalgemm.UploadQ4KMappedSpan(span, offset, qt.out, qt.in)
		}
		if w == nil {
			copiedUpload = true
			var err error
			raw, err = qt.materializeRaw()
			if err != nil {
				panic("model: lazy Q4_K read " + name + ": " + err.Error())
			}
			raw = pageAlignResidentBytes(raw)
		}
	}
	if w == nil {
		w = metalgemm.UploadQ4K(raw, qt.out, qt.in)
	}
	if qt.lazy != nil {
		switch {
		case w == nil:
			recordQ4KResidencyOutcome(m, name, qt.lazy.Bytes, q4kResidencyUploadFailure)
		case mappedAttempt && !copiedUpload:
			recordQ4KResidencyOutcome(m, name, qt.lazy.Bytes, q4kResidencyMappedSuccess)
		case copiedUpload && len(qt.lazy.MappedSpan) > 0:
			// The mapping was present but either descriptor validation or Metal aliasing declined,
			// and the copied model-owned bytes subsequently uploaded successfully.
			recordQ4KResidencyOutcome(m, name, qt.lazy.Bytes, q4kResidencyMappedDeclineCopiedUpload)
		}
	}
	tbl[name] = w // cache nil too, so a failed upload doesn't retry every token
	if qt.lazy == nil && w != nil && freeCPUCopyAfterUpload && !w.NoCopy() {
		// Drop the CPU copy → single residency (~16 GB for 27B vs ~30 GB doubled). UNSAFE
		// unless EVERY q4_k matmul for this weight — decode GEMV *and* batched prefill GEMM —
		// is guaranteed to run on the GPU: the CPU fallbacks q4kGemm/q4kMatRows read qt.raw and
		// panic on a nil slice (#1067, the multi-K-prompt prefill crash). Gated OFF by default;
		// FAK_Q4K_FREE_CPU=1 opts back into single residency once the prefill path is fully
		// GPU-routed and the CPU fallback is provably unreachable. A no-copy Metal buffer already
		// aliases qt.raw, so keeping the slice costs no duplicate storage and preserves fallback.
		qt.raw = nil
	}
	return w
}

// releaseMetalQ4KResidency drops every handle owned by m before its retained
// checkpoint mapping is closed. Handles are removed from the cache first so no
// later lookup can rediscover a buffer whose backing is about to be unmapped.
func releaseMetalQ4KResidency(m *Model) {
	metalQ4KMu.Lock()
	tbl := metalQ4KW[m]
	delete(metalQ4KW, m)
	metalQ4KMu.Unlock()
	for _, w := range tbl {
		if w != nil {
			w.Release()
		}
	}
}
