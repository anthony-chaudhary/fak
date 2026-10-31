package model

import (
	"errors"
	"fmt"
	"math"
	"os"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41CompressorRingEnv set to "0" keeps decode on the host-staged compressor
// group (per-row projection readbacks, then an uploaded gated pool), for A/B
// and bisection.
const v41CompressorRingEnv = "FAK_V41_COMPRESSOR_RING"

var (
	errV41CompressorRingResult = errors.New("model: invalid selected V4.1 compressor ring result")
	errV41CompressorRingDesync = errors.New("model: V4.1 compressor ring is out of step with the retained group")
)

// v41CompressorRing keeps each ratio >= 2 compressor-source layer's current
// group on the device. Adapted from DeepInfer compress_and_update_state_decode
// (Apache-2.0, deepseek-ai/DeepInfer deepinfer/model_executor/layers/
// tilelang_ops/compress.py at 774ce244b87cc0eb8d4b94ab9f5de2ab22049f01); no
// upstream source is copied. Every decode token projects its carrier through
// wkv/wgate into a ring row without a readback, and the group-closing token
// pools the ring in the same dispatch and reads back one row. The host
// attention state stays the source of truth, exactly like v41ResidentKV: each
// ring row remembers the position and input bits it was projected from, and a
// row that does not match the retained group (rollback, restore, fork, reset,
// prefill seed) is re-projected from the host-retained input before the group
// closes.
type v41CompressorRing struct {
	s      *Session
	ops    compute.V41CompressorRingBackend
	layers map[int]*v41CompressorRingLayer
	stats  v41CompressorRingStats
}

// v41CompressorRingStats counts successful device work for witnesses.
type v41CompressorRingStats struct {
	Stores, Emits, Reseeds int
}

type v41CompressorRingLayer struct {
	ratio, width int
	kv, scores   compute.Tensor
	slots        []v41CompressorRingSlot
}

type v41CompressorRingSlot struct {
	valid bool
	pos   int
	input []float32
}

func (r v41CompressorRingSlot) holds(pos int, input []float32) bool {
	if !r.valid || r.pos != pos || len(r.input) != len(input) {
		return false
	}
	for i, v := range input {
		if math.Float32bits(v) != math.Float32bits(r.input[i]) {
			return false
		}
	}
	return true
}

// v41CompressorRingBinding binds the ring when the backend offers it.
func (s *Session) v41CompressorRingBinding() *v41CompressorRing {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory || os.Getenv(v41CompressorRingEnv) == "0" {
		return nil
	}
	ops, ok := s.Backend.(compute.V41CompressorRingBackend)
	if !ok || !ops.SupportsV41CompressorRing() {
		return nil
	}
	return &v41CompressorRing{s: s, ops: ops, layers: map[int]*v41CompressorRingLayer{}}
}

func v41CompressorRingKey(layer int, part string) string {
	return fmt.Sprintf("v41-compressor-ring/%d/%s", layer, part)
}

// v41CompressorRingStep is one admitted layer step. It exists only when the
// session's canonical native projection would run both compressor leaves on
// this backend, so the ring rows carry exactly the bits that path reads back.
type v41CompressorRingStep struct {
	ring          *v41CompressorRing
	model         *Model
	layer, ratio  int
	width, hidden int
	dtypes        [2]compute.Dtype
}

func (o *v41DeviceGatherOps) compressorRingStep(m *Model, layer, ratio int, scratch *v41ProjScratch) *v41CompressorRingStep {
	if o == nil {
		return nil
	}
	return o.compressorRing.step(m, layer, ratio, scratch)
}

func (r *v41CompressorRing) step(m *Model, layer, ratio int, scratch *v41ProjScratch) *v41CompressorRingStep {
	if r == nil || m == nil || ratio < 2 || m.lora != nil || m.prism != nil || scratch == nil || scratch.projectionRoute == nil {
		return nil
	}
	s := r.s
	binding := scratch.projectionRoute
	native := binding.nativeProjection
	if native.core == nil || binding.projection == nil || native.owner != s || native.model != m || s.M != m ||
		!backendHasComparableIdentity(native.backend) || native.backend != s.Backend ||
		s.halClosed || s.halFailure != nil || !s.modelWeightsHeld || s.halW == nil {
		return nil
	}
	width, hidden := v41CompressorWidth(m.Cfg), m.Cfg.HiddenSize
	if compute.ValidateV41CompressorRingAlloc(ratio, width) != nil || hidden <= 0 {
		return nil
	}
	st := &v41CompressorRingStep{ring: r, model: m, layer: layer, ratio: ratio, width: width, hidden: hidden}
	for i, leaf := range v41ColdCompressorLeaves {
		name := layerName(layer, leaf)
		dtype, ok := s.v41HeadProjectionDtype(name, width, hidden)
		if !ok {
			return nil
		}
		if qt := m.kqw[name]; qt != nil && !m.has(name) && m.q8w[name] == nil && m.q4kw[name] == nil &&
			qt.kind == kindQ6K && !s.useHALKQuantWeight(qt) {
			return nil
		}
		st.dtypes[i] = dtype
	}
	return st
}

func (r *v41CompressorRing) adopt(key string, t compute.Tensor) {
	s := r.s
	if old, ok := s.halW[key]; ok && old.Buf() != t.Buf() {
		if _, borrowed := s.borrowedHALW[key]; !borrowed {
			s.Backend.Free(old)
		}
	}
	s.halW[key] = t
}

func (r *v41CompressorRing) current(key string, t compute.Tensor) bool {
	held, ok := r.s.halW[key]
	return ok && t.Buf() != nil && held.Buf() == t.Buf()
}

func (r *v41CompressorRing) layer(layer, ratio, width int) (*v41CompressorRingLayer, error) {
	kvKey, scoreKey := v41CompressorRingKey(layer, "kv"), v41CompressorRingKey(layer, "scores")
	if lr := r.layers[layer]; lr != nil && lr.ratio == ratio && lr.width == width && r.current(kvKey, lr.kv) && r.current(scoreKey, lr.scores) {
		return lr, nil
	}
	kv, scores, err := r.ops.V41CompressorRingAlloc(ratio, width)
	if err != nil {
		return nil, err
	}
	r.adopt(kvKey, kv)
	r.adopt(scoreKey, scores)
	lr := &v41CompressorRingLayer{ratio: ratio, width: width, kv: kv, scores: scores, slots: make([]v41CompressorRingSlot, ratio)}
	r.layers[layer] = lr
	return lr, nil
}

// fits reports whether this step can carry the group; a malformed group keeps
// the host-staged path and its own refusal.
func (st *v41CompressorRingStep) fits(ratio int, retained [][]float32, input []float32) bool {
	if st == nil || ratio != st.ratio || len(retained) >= ratio || len(input) != st.hidden {
		return false
	}
	for _, row := range retained {
		if len(row) != st.hidden {
			return false
		}
	}
	return true
}

// store projects input into group slot len(retained). Rows the ring does not
// hold for retained[0:k) are re-projected first. The group-closing store
// returns the raw pooled row (before the BF16/RMSNorm tail); any other store
// returns nil. Every failure after selection closes the session.
func (st *v41CompressorRingStep) store(retained [][]float32, retainedPos []int, input []float32, pos int) ([]float32, error) {
	r, s := st.ring, st.ring.s
	return v41SelectedDeviceCall(s, st.layer, "v41-compressor-ring", func(stage *string) ([]float32, error) {
		k := len(retained)
		if k >= st.ratio || len(retainedPos) != k || len(input) != st.hidden {
			return nil, errV41CompressorRingDesync
		}
		*stage = "ring"
		lr, err := r.layer(st.layer, st.ratio, st.width)
		if err != nil {
			return nil, err
		}
		for j := 0; j < k; j++ {
			if len(retained[j]) != st.hidden || retainedPos[j] != pos-k+j {
				return nil, errV41CompressorRingDesync
			}
			if lr.slots[j].holds(retainedPos[j], retained[j]) {
				continue
			}
			*stage = "reseed"
			if _, err := st.project(lr, j, retained[j], retainedPos[j], false); err != nil {
				return nil, err
			}
			r.stats.Reseeds++
		}
		emit := k+1 == st.ratio
		*stage = "store"
		out, err := st.project(lr, k, input, pos, emit)
		if err != nil || !emit {
			return nil, err
		}
		defer s.Backend.Free(out)
		*stage = "readback"
		pooled, err := compute.DecodeV41CompressorRing(append([]float32(nil), s.Backend.Read(out)...), st.width)
		if err != nil {
			return nil, err
		}
		r.stats.Emits++
		return pooled, nil
	})
}

// project runs both compressor leaves on the canonical device weights with one
// activation upload and records the ring store. Weight staging failures panic
// with their payload; they become the selected operation's error here.
func (st *v41CompressorRingStep) project(lr *v41CompressorRingLayer, slot int, input []float32, pos int, emit bool) (out compute.Tensor, err error) {
	s, m := st.ring.s, st.model
	opened := m.v41NowNanos()
	rows := 0
	defer func() { m.v41NoteCompressorProjection(2, 0, rows, 0, int64(len(input))*4, 0, opened) }()
	lr.slots[slot].valid = false
	var weights [2]compute.Tensor
	if cause := func() (cause error) {
		defer func() {
			if r := recover(); r != nil {
				payload, ok := r.(v41HeadPayloadError)
				if !ok {
					panic(r)
				}
				cause = payload.cause
			}
		}()
		s.ensureOpenBackendSession()
		for i, leaf := range v41ColdCompressorLeaves {
			name := layerName(st.layer, leaf)
			weights[i] = s.v41HeadWeightHAL(name, st.dtypes[i], st.width, st.hidden)
			if st.dtypes[i] == compute.Q4_K {
				if q := m.q4kw[name]; q4kHostCopyReleasable(s, name, q, weights[i].Ready()) {
					q.raw = nil
				}
			}
		}
		return nil
	}(); cause != nil {
		return compute.Tensor{}, cause
	}
	for _, v := range input {
		if !finite32(v) {
			return compute.Tensor{}, errV41CompressorRingResult
		}
	}
	x := s.uploadHostF32([]int{st.hidden}, input, compute.MemoryActivation, "V4.1 compressor ring activation")
	defer s.Backend.Free(x)
	if producer, ok := st.ring.ops.(compute.V41CompressorRingProducerBackend); ok {
		out, err = producer.V41CompressorRingProjectStore(lr.kv, lr.scores, weights[0], weights[1], x, slot, st.ratio, st.width, emit)
		if err != nil {
			return compute.Tensor{}, err
		}
		if emit && !v41MHCFreshOutput(s, out, []int{2, st.width}, lr.kv, lr.scores, x, weights[0], weights[1]) {
			if out.Buf() != nil {
				s.Backend.Free(out)
			}
			return compute.Tensor{}, errV41CompressorRingResult
		}
	} else {
		var projected [2]compute.Tensor
		for i := range weights {
			projected[i] = s.Backend.MatMul(weights[i], x)
			defer s.Backend.Free(projected[i])
			if y := projected[i]; y.Buf() == nil || y.Buf() == x.Buf() || y.Dtype != compute.F32 || y.Layout != compute.RowMajor || y.Quant != nil || y.Numel() != st.width {
				return compute.Tensor{}, errV41CompressorRingResult
			}
		}
		out, err = st.ring.ops.V41CompressorRingStore(lr.kv, lr.scores, projected[0], projected[1], slot, st.ratio, st.width, emit)
		if err != nil {
			return compute.Tensor{}, err
		}
		if emit && !v41MHCFreshOutput(s, out, []int{2, st.width}, lr.kv, lr.scores, projected[0], projected[1], x) {
			if out.Buf() != nil {
				s.Backend.Free(out)
			}
			return compute.Tensor{}, errV41CompressorRingResult
		}

	}
	rows = 2
	st.ring.stats.Stores++
	lr.slots[slot] = v41CompressorRingSlot{valid: true, pos: pos, input: append([]float32(nil), input...)}
	return out, nil
}
