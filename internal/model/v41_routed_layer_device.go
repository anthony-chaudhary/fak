package model

import (
	"errors"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// v41LayerExpertBatchEnv disables the per-layer routed-expert batch when set to
// "0", so the per-pick fused seam stays one switch away for A/B and bisection.
const v41LayerExpertBatchEnv = "FAK_V41_LAYER_EXPERT_BATCH"

// v41RoutedIDEnv set to "0" keeps the batched layer on its per-pick dispatches
// instead of the one-dispatch-per-stage id kernels, for A/B and bisection.
const v41RoutedIDEnv = "FAK_V41_ROUTED_ID"

// v41RoutedLayerFunc runs every routed pick of one full-profile decode layer on
// the device and returns their routed sum (#13820). Each pick issues the
// kernels v41RoutedExpertFunc issues, in the same order, and the sum adds the
// BF16-published rows in v41RoutedSumOrder, so the result is bit-identical to
// the per-pick seam followed by v41RoutedSumFunc. Only the per-pick status and
// output readbacks become one end-of-layer readback. x is the BF16-published
// expert input. Declined leaves the layer to the per-pick loop; error is a
// selected failure typed exactly as the per-pick loop would raise it.
type v41RoutedLayerFunc func(layer int, picks []routePick, x []float32) ([]float32, v41RoutedExpertOutcome, error)

// v41RoutedLayerProbe names no resident: CheckAdmission on it asks whether
// the ring can still pin the given bytes.
const v41RoutedLayerProbe polymodel.ModelID = "v41-routed-layer-probe"

// v41RoutedLayerFunc binds the batched routed layer, or nil when the fused
// per-pick seam is unbound or the backend cannot defer its status records.
func (s *Session) v41RoutedLayerFunc() v41RoutedLayerFunc {
	if os.Getenv(v41LayerExpertBatchEnv) == "0" || s.v41RoutedExpertFunc() == nil {
		return nil
	}
	cfg := s.M.Cfg
	H, I := cfg.HiddenSize, cfg.MoEIntermediateSize
	limit := float32(cfg.SwigluLimit)
	deferred, ok := s.Backend.(compute.V41DeferredStatusBackend)
	if !ok || H <= 0 || I <= 0 || !deferred.SupportsV41DeferredStatus(H) {
		return nil
	}
	idBackend, _ := s.Backend.(compute.V41RoutedIDBackend)
	if idBackend != nil && (os.Getenv(v41RoutedIDEnv) == "0" || !idBackend.V41RoutedIDAvailable()) {
		idBackend = nil
	}
	return func(layer int, picks []routePick, x []float32) (result []float32, outcome v41RoutedExpertOutcome, cause error) {
		if len(picks) == 0 || len(x) != H {
			return nil, v41RoutedExpertDeclined, nil
		}
		for _, pick := range picks {
			if !finite32(pick.weight) {
				return nil, v41RoutedExpertDeclined, nil
			}
		}
		names := make([][3]string, len(picks))
		for slot, pick := range picks {
			stem := "ffn.experts." + itoa(pick.expert)
			names[slot] = [3]string{layerName(layer, stem+".w1.weight"), layerName(layer, stem+".w3.weight"), layerName(layer, stem+".w2.weight")}
			if s.M.has(names[slot][2][:len(names[slot][2])-len("weight")] + "bias") {
				return nil, v41RoutedExpertDeclined, nil
			}
		}
		s.ensureOpenBackendSession()
		path, stage := "v41-routed-expert", "gate/up"
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: path, Layer: layer, Stage: stage, Cause: err}
			s.halFailure = closed
			s.Close()
			return closed
		}
		var opened int64
		dispatched, noted := 0, 0
		defer func() {
			if outcome == v41RoutedExpertDeclined {
				return
			}
			for ; noted < dispatched; noted++ {
				s.M.v41NoteIncrementalDeviceDispatch(true, v41FirstOpened(noted, opened))
				s.M.v41NoteIncrementalDeviceDispatch(false, 0)
			}
		}()
		defer func() {
			if r := recover(); r != nil {
				err, isErr := r.(error)
				if isErr {
					var closed *BackendForwardOperationError
					if errors.As(err, &closed) {
						panic(r)
					}
					var backend *compute.BackendError
					if errors.As(err, &backend) {
						result, outcome, cause = nil, v41RoutedExpertError, closeFailure(err)
						return
					}
				}
				if converted, ok := compute.ConvertCUDAPanic(r, "", ""); ok {
					if isErr {
						converted = err
					}
					result, outcome, cause = nil, v41RoutedExpertError, closeFailure(converted)
					return
				}
				if !isErr {
					err = fmt.Errorf("unclassified backend panic: %v", r)
				}
				closeFailure(err)
				panic(r)
			}
		}()

		// Page every pick in before recording any work: a mapped page-in after
		// recording has started would flush the step batch.
		var releases []func()
		defer func() {
			for i := len(releases) - 1; i >= 0; i-- {
				releases[i]()
			}
		}()
		weights := make([]v41RoutedLayerWeights, len(picks))
		admitted := func() bool {
			if r := s.routedExpertRing(names[0][0]); r != nil {
				done := s.ringEnter(r)
				defer done()
				if !s.v41RoutedLayerRingFits(r, names) {
					return false
				}
			}
			for slot := range picks {
				gateW, releaseGate, upW, releaseUp, _, ok := expertInputDeviceAdmitted(s, names[slot][0], names[slot][1], x, H)
				if !ok {
					return false
				}
				releases = append(releases, releaseGate, releaseUp)
				downW, releaseDown, ok := s.q4kExpertDownDeviceWeight(names[slot][2])
				if !ok {
					return false
				}
				releases = append(releases, releaseDown)
				weights[slot] = v41RoutedLayerWeights{gateW, upW, downW}
			}
			return true
		}
		if !admitted() {
			return nil, v41RoutedExpertDeclined, nil
		}

		var owned []compute.Tensor
		defer func() {
			for i := len(owned) - 1; i >= 0; i-- {
				s.Backend.Free(owned[i])
			}
		}()
		var statuses []compute.V41DeferredStatus
		consumed := false
		defer func() {
			if consumed {
				return
			}
			for _, st := range statuses {
				s.Backend.Free(st.Record)
			}
		}()
		opened = s.M.v41NowNanos()
		xd := s.uploadHostF32([]int{H}, x, compute.MemoryActivation, "moe expert gate/up activation")
		owned = append(owned, xd)
		useID := idBackend != nil && v41RoutedIDAdmits(idBackend, weights, len(picks))
		// Verdicts decode per pick: the per-pick arm interleaves each pick's
		// SwiGLU and publication records, the id arm returns all SwiGLU records
		// and then all publication records.
		swigluVerdict := func(slot int) int { return 2 * slot }
		publishVerdict := func(slot int) int { return 2*slot + 1 }
		var sumRow compute.Tensor
		if useID {
			swigluVerdict = func(slot int) int { return slot }
			publishVerdict = func(slot int) int { return len(picks) + slot }
			gates, ups, downs := make([]compute.Tensor, len(picks)), make([]compute.Tensor, len(picks)), make([]compute.Tensor, len(picks))
			slots, routeWeights := make([]int, len(picks)), make([]float32, len(picks))
			for slot, pick := range picks {
				gates[slot], ups[slot], downs[slot] = weights[slot].gate, weights[slot].up, weights[slot].down
				slots[slot], routeWeights[slot] = slot, pick.weight
			}
			stage = "gate/up"
			dispatched = len(picks)
			ids, err := idBackend.V41RoutedIDs(slots, routeWeights, v41RoutedSumOrder(picks))
			if err != nil {
				return nil, v41RoutedExpertError, closeFailure(err)
			}
			owned = append(owned, ids)
			g, err := idBackend.V41MatVecID(gates, ids, len(picks), xd)
			if err != nil {
				return nil, v41RoutedExpertError, closeFailure(err)
			}
			owned = append(owned, g)
			u, err := idBackend.V41MatVecID(ups, ids, len(picks), xd)
			if err != nil {
				return nil, v41RoutedExpertError, closeFailure(err)
			}
			owned = append(owned, u)
			stage = "activation"
			activation, activationStatus, err := idBackend.V41RoutedSwiGLUIDDeferred(g, u, ids, len(picks), limit)
			if err != nil {
				return nil, v41RoutedExpertError, closeFailure(err)
			}
			owned = append(owned, activation)
			statuses = append(statuses, activationStatus)
			stage = "down"
			out, err := idBackend.V41MatVecID(downs, ids, len(picks), activation)
			if err != nil {
				return nil, v41RoutedExpertError, closeFailure(err)
			}
			owned = append(owned, out)
			path, stage = "v41-routed-sum", "routed-sum"
			sum, publishStatus, err := idBackend.V41RoutedPublishSumDeferred(out, ids, len(picks), compute.V41BF16PublishRoutedDown)
			if err != nil {
				return nil, v41RoutedExpertError, closeFailure(err)
			}
			owned = append(owned, sum)
			statuses = append(statuses, publishStatus)
			if sum.Buf() == nil || sum.Dtype != compute.F32 || len(sum.Shape) != 1 || sum.Shape[0] != H {
				return nil, v41RoutedExpertError, closeFailure(errV41BF16BoundaryResult)
			}
			sumRow = sum
			s.M.v41NoteRoutedLayerDispatches(true, v41RoutedIDLayerDispatches)
		}
		published := make([]compute.Tensor, len(picks))
		for slot, pick := range picks {
			if useID {
				break
			}
			stage = "gate/up"
			dispatched = slot + 1
			g := s.Backend.MatMul(weights[slot].gate, xd)
			owned = append(owned, g)
			u := s.Backend.MatMul(weights[slot].up, xd)
			owned = append(owned, u)
			stage = "activation"
			activation, activationStatus, err := deferred.V41RoutedSwiGLUDeferred(g, u, limit, pick.weight)
			if err != nil {
				return nil, v41RoutedExpertError, closeFailure(err)
			}
			owned = append(owned, activation)
			statuses = append(statuses, activationStatus)
			stage = "down"
			out := s.Backend.MatMul(weights[slot].down, activation)
			owned = append(owned, out)
			row, publishStatus, err := deferred.V41BF16PublishDeferred(out, compute.V41BF16PublishRoutedDown)
			if err != nil {
				return nil, v41RoutedExpertError, closeFailure(err)
			}
			owned = append(owned, row)
			statuses = append(statuses, publishStatus)
			if row.Buf() == nil || row.Buf() == out.Buf() || row.Dtype != compute.F32 || len(row.Shape) != 1 || row.Shape[0] != H {
				return nil, v41RoutedExpertError, closeFailure(errV41BF16BoundaryResult)
			}
			published[slot] = row
		}
		if !useID {
			path, stage = "v41-routed-sum", "accumulator upload"
			acc := s.uploadHostF32([]int{H}, make([]float32, H), compute.MemoryActivation, "V4.1 routed-sum accumulator")
			owned = append(owned, acc)
			stage = "routed-sum"
			v41DeviceAccumulate(s.Backend, acc, published, v41RoutedSumOrder(picks))
			sumRow = acc
			s.M.v41NoteRoutedLayerDispatches(false, v41RoutedPerPickDispatches*len(picks)+len(picks))
		}
		path, stage = "v41-routed-expert", "readback"
		consumed = true
		verdicts, rows, err := deferred.V41ReadDeferred(statuses, []compute.Tensor{sumRow})
		if err != nil {
			return nil, v41RoutedExpertError, closeFailure(err)
		}
		if len(verdicts) != 2*len(picks) || len(rows) != 1 {
			return nil, v41RoutedExpertError, closeFailure(errV41ExpertResult)
		}
		// Verdicts decode in slot order, so the lowest faulting pick reports
		// exactly the stage the per-pick loop would have stopped at.
		for slot := range picks {
			s.M.v41NoteIncrementalDeviceDispatch(true, v41FirstOpened(slot, opened))
			s.M.v41NoteIncrementalDeviceDispatch(false, 0)
			noted = slot + 1
			stage = "activation"
			if err := verdicts[swigluVerdict(slot)]; err != nil {
				dispatched = noted
				var fault *compute.V41RoutedSwiGLUFault
				if errors.As(err, &fault) {
					faultStage := "gate/up"
					if fault.Stage == compute.V41RoutedSwiGLUStageDownOperand {
						faultStage = "down"
					}
					return nil, v41RoutedExpertError, v41ExpertOperationErr(layer, faultStage, errV41ExpertResult)
				}
				return nil, v41RoutedExpertError, closeFailure(err)
			}
			s.M.v41NoteExpertActivation(1, 0, 16, v41FirstOpened(slot, opened))
			stage = "down"
			if err := verdicts[publishVerdict(slot)]; err != nil {
				dispatched = noted
				var numeric *compute.V41BF16CheckedArithmeticError
				if errors.As(err, &numeric) {
					return nil, v41RoutedExpertError, v41ExpertOperationErr(layer, stage, errV41ExpertResult)
				}
				return nil, v41RoutedExpertError, closeFailure(err)
			}
			s.M.v41NoteExpertContraction()
		}
		path, stage = "v41-routed-sum", "readback"
		sum := rows[0]
		if len(sum) != H {
			return nil, v41RoutedExpertError, closeFailure(v41ExpertOperationErr(layer, "sum", errV41ExpertResult))
		}
		for _, v := range sum {
			if !finite32(v) {
				return nil, v41RoutedExpertError, closeFailure(v41ExpertOperationErr(layer, "sum", errV41ExpertResult))
			}
		}
		return sum, v41RoutedExpertHandled, nil
	}
}

// v41RoutedLayerWeights is one pick's staged and held device projections.
type v41RoutedLayerWeights struct{ gate, up, down compute.Tensor }

// Routed dispatches one batched layer records: the per-pick arm issues gate,
// up, SwiGLU, down and publication per pick plus one sum add per pick; the id
// arm issues gate, up, SwiGLU, down and publication+sum once for every pick.
const (
	v41RoutedPerPickDispatches = 5
	v41RoutedIDLayerDispatches = 5
)

// v41RoutedIDAdmits reports whether one id dispatch per projection can serve
// every pick: within the slot bound, and each projection's slot table one
// dtype and shape the backend's id GEMV serves. Otherwise the layer keeps its
// per-pick dispatches.
func v41RoutedIDAdmits(be compute.V41RoutedIDBackend, weights []v41RoutedLayerWeights, picks int) bool {
	if picks <= 0 || picks > compute.V41RoutedIDMaxPicks || len(weights) != picks {
		return false
	}
	same := func(a, b compute.Tensor) bool {
		return a.Dtype == b.Dtype && len(a.Shape) == 2 && len(b.Shape) == 2 && a.Shape[0] == b.Shape[0] && a.Shape[1] == b.Shape[1]
	}
	first := weights[0]
	for _, w := range []compute.Tensor{first.gate, first.up, first.down} {
		if !be.SupportsV41RoutedIDWeight(w.Dtype) || len(w.Shape) != 2 || w.Shape[1]%256 != 0 {
			return false
		}
	}
	for _, w := range weights[1:] {
		if !same(w.gate, first.gate) || !same(w.up, first.up) || !same(w.down, first.down) {
			return false
		}
	}
	return true
}

func v41FirstOpened(slot int, opened int64) int64 {
	if slot == 0 {
		return opened
	}
	return 0
}

// v41RoutedLayerRingFits reports whether the ring can pin every pick's three
// projections at once beside its other pinned residents. The per-pick seam
// pins one expert at a time, so a ring that only fits that declines here
// instead of refusing in strict mode. Caller holds the ring span.
func (s *Session) v41RoutedLayerRingFits(r *pagedRing, names [][3]string) bool {
	seen := map[polymodel.ModelID]bool{}
	var unpinned int64
	for _, triple := range names {
		for _, name := range triple {
			w, ok := s.resolveExpertWeight(name)
			if !ok {
				return false
			}
			var bytes int64
			switch {
			case w.ck != nil:
				bytes = w.ck.bytes
			case w.q4 != nil:
				bytes = q4kResidentBytes(w.q4)
			case w.kq != nil:
				bytes = kQuantResidentBytes(w.kq)
			}
			id := polymodel.ModelID(w.halKey())
			if bytes <= 0 || seen[id] {
				continue
			}
			seen[id] = true
			if m, resident := r.pool.Get(id); resident && m.Pinned {
				continue
			}
			unpinned += bytes
		}
	}
	return r.pool.CheckAdmission(polymodel.Model{ID: v41RoutedLayerProbe, WeightBytes: unpinned}) == nil
}

// v41RoutedLayerStep offers one full-profile decode layer's picks to the
// batched routed layer. handled=false leaves the per-pick loop to run, which
// also keeps every malformed-pick refusal on its original path.
func (m *Model) v41RoutedLayerStep(l int, picks []routePick, ffnX []float32, scratch *v41ProjScratch) ([]float32, bool, error) {
	if scratch == nil || scratch.routedLayer == nil || len(picks) == 0 {
		return nil, false, nil
	}
	cfg := m.Cfg
	H := cfg.HiddenSize
	if H <= 0 || cfg.MoEIntermediateSize <= 0 {
		return nil, false, nil
	}
	for _, pick := range picks {
		if !finite32(pick.weight) {
			return nil, false, nil
		}
	}
	// The per-pick loop publishes this same row before its first pick.
	x, err := v41RoutedBF16Publish(l, "gate/up", ffnX, H, scratch.gatherOps.bf16PublishFunc())
	if err != nil {
		return nil, true, err
	}
	sum, outcome, cause := scratch.routedLayer(l, picks, x)
	switch outcome {
	case v41RoutedExpertHandled:
		if cause != nil {
			return nil, true, cause
		}
		if len(sum) != H {
			return nil, true, v41ExpertOperationErr(l, "sum", errV41DeviceSumResult)
		}
		return sum, true, nil
	case v41RoutedExpertError:
		if cause == nil {
			cause = v41ExpertOperationErr(l, "gate/up", errV41ExpertResult)
		}
		return nil, true, cause
	case v41RoutedExpertDeclined:
		return nil, false, nil
	default:
		return nil, true, v41ExpertOperationErr(l, "gate/up", errV41ExpertResult)
	}
}
