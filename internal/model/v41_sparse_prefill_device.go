package model

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"math"
)

type v41SparsePrefillCall struct {
	backend compute.V41SparsePrefillBackend
	live    func() bool
}

// Select before QKV effects. Replaced callbacks cannot acquire this capability
// by wrapping the ordinary callback; only a live canonical session binding may.
func (m *Model) v41PrepareSparsePrefill(st *v41ForwardState, scratch *v41ProjScratch, full bool, l, seq, heads, dim int, trace *v41ActivationTraceState) *v41SparsePrefillCall {
	if m == nil || st == nil || scratch == nil || !full || seq < 2 || seq > math.MaxInt32/32768 || heads != 64 || dim != 512 || m.Cfg.QKRopeHeadDim != 64 || m.Cfg.OGroups != 8 || trace != nil || m.lora != nil || m.prism != nil || tensorLookupProbe != nil || m.v41CompressorHostCustomClock() {
		return nil
	}
	excluded := func() bool {
		return m.Cfg.NumHeads != 64 || m.Cfg.HeadDim != 512 || m.Cfg.QKRopeHeadDim != 64 || m.Cfg.OGroups != 8 || scratch.tailRoPE != nil || scratch.gatherOps.bf16PublishFunc() != nil || scratch.gatherOps.inverseOutputFunc() != nil || scratch.activationTrace != nil || tensorLookupProbe != nil || m.v41CompressorHostCustomClock()
	}
	if excluded() {
		return nil
	}
	binding := scratch.sharedAttentionBinding
	if binding == nil || binding.ordinary == nil {
		return nil
	}
	n := binding.native
	s := n.owner
	if n.core == nil || s == nil || n.model != m || s.M != m || !backendHasComparableIdentity(n.backend) || !backendHasComparableIdentity(s.Backend) || n.backend != s.Backend || st.callbackOwner != s || st.sharedAttentionBinding != binding || s.v41Forward == nil || s.v41Forward.sharedAttentionBinding != binding {
		return nil
	}
	scope, epoch, forward, backend := s.backendExecutionScope, s.cacheGeometryEpoch, s.v41Forward, s.Backend
	if scope == nil || !scope.live || scope.owner == nil {
		return nil
	}
	owner := scope.owner
	var config []byte
	frozenPlan, planErr := v41AttentionPlanFor(m.Cfg, l, m.v41AttentionRolesCached())
	if planErr != nil {
		return nil
	}
	live := func() bool {
		if excluded() {
			return false
		}
		plan, e := v41AttentionPlanFor(m.Cfg, l, m.v41AttentionRolesCached())
		if e != nil || plan != frozenPlan {
			return false
		}
		if config != nil {
			current, e := json.Marshal(m.Cfg)
			if e != nil || !bytes.Equal(config, current) {
				return false
			}
		}
		if s.M != m || !backendHasComparableIdentity(s.Backend) || s.Backend != backend || s.halClosed || s.halFailure != nil || !s.modelWeightsHeld || s.v41Forward != forward || forward.callbackOwner != s || forward.sharedAttentionBinding != binding || st.callbackOwner != s || st.sharedAttentionBinding != binding || scratch.sharedAttentionBinding != binding || s.cacheGeometryEpoch != epoch || s.backendExecutionScope != scope || !scope.live || scope.owner != owner || m.lora != nil || m.prism != nil {
			return false
		}
		for _, member := range scope.sessions {
			if member == s {
				return true
			}
		}
		return false
	}
	b, ok := backend.(compute.V41SparsePrefillBackend)
	if !ok {
		return nil
	}
	config, planErr = json.Marshal(m.Cfg)
	if planErr != nil {
		return nil
	}
	if !live() || !b.SupportsV41SparsePrefill() || !live() {
		return nil
	}
	return &v41SparsePrefillCall{backend: b, live: live}
}

// Infallible bit capture preserves the legacy wkv/row-wise normalization error
// order. Validation of the captured pre-RoPE panel occurs only at contraction.
func v41SparsePrefillBits(x []float32) []uint16 {
	out := make([]uint16, len(x))
	for i, v := range x {
		b := math.Float32bits(v)
		if b&0x7f800000 == 0x7f800000 {
			out[i] = uint16(b>>16) | 1
		} else {
			out[i] = uint16((b + 0x7fff + ((b >> 16) & 1)) >> 16)
		}
	}
	return out
}

func (m *Model) v41SparsePrefillContract(call *v41SparsePrefillCall, l, seq int, src v41LayerSources, qkv v41LayerQKV, indexList []int32, sink []float32) ([][]float32, error) {
	fail := func(s string) ([][]float32, error) {
		return nil, v41StageErr(v41StageAttention, l, fmt.Errorf("%w: sparse panel %s", ErrV41ForwardStage, s))
	}
	if call == nil || !call.live() {
		return fail("session ownership changed")
	}
	if len(qkv.preRoPE) != seq*32768 || len(qkv.kvRows) != seq || len(sink) != 64 {
		return fail("shape")
	}
	rows := seq + len(src.sharedKV)
	if rows > math.MaxInt32/512 || seq > math.MaxInt32/32768 {
		return fail("overflow")
	}
	x := compute.V41SparsePrefillInput{Queries: seq, Keys: rows, TableRows: seq, Q: qkv.preRoPE, Scale: m.Cfg.attnScale(), Sink: append([]float32(nil), sink...), Lengths: make([]int32, seq), Positions: make([]uint32, seq)}
	// Validate every original row and ID before masking. The selected whole-panel
	// contract may report cross-row malformed inputs earlier than scalar attention.
	for _, bank := range [][][]float32{qkv.kvRows, src.sharedKV} {
		for _, row := range bank {
			if len(row) != 512 {
				return fail("KV shape")
			}
			for _, v := range row {
				if !finite32(v) {
					return fail("nonfinite KV")
				}
			}
			x.KV = append(x.KV, v41SparsePrefillBits(row)...)
		}
	}
	compressed := src.plan.Ratio > 0 || src.plan.Role == V41AttentionRoleReader
	width := src.plan.topKWidth()
	if indexList != nil && (width <= 0 || len(indexList) != seq*width) {
		return fail("index coverage")
	}
	for _, id := range indexList {
		if id < -1 || int(id) >= len(src.sharedKV) {
			return fail("original compressed ID out of range")
		}
	}
	lists := make([][]int32, seq)
	for t := 0; t < seq; t++ {
		keys := v41PlainWindowKeys(t, m.Cfg.windowForLayer(l))
		ids := make([]int32, 0, len(keys)+len(src.sharedKV))
		for _, key := range keys {
			ids = append(ids, int32(key))
		}
		if compressed {
			ratio := src.plan.kvGroupSize(m.Cfg)
			if ratio <= 0 {
				return fail("ratio")
			}
			var chosen []int32
			if indexList != nil {
				chosen = indexList[t*width : (t+1)*width]
			} else {
				chosen = make([]int32, len(src.sharedKV))
				for i := range chosen {
					chosen[i] = int32(i)
				}
			}
			for _, id := range chosen {
				if id < 0 || t < ratio-1 || int(id) > (t-(ratio-1))/ratio {
					ids = append(ids, -1)
				} else {
					ids = append(ids, int32(seq)+id)
				}
			}
		}
		lists[t] = ids
		if len(ids) > x.TopK {
			x.TopK = len(ids)
		}
		x.Positions[t] = uint32(t)
		cos, sin := v41RopeTableForLayer(m.Cfg, l, t)
		if len(cos) != 32 || len(sin) != 32 {
			return fail("RoPE table")
		}
		x.CosSin = append(x.CosSin, cos...)
		x.CosSin = append(x.CosSin, sin...)
	}
	if x.TopK > math.MaxInt32-7 {
		return fail("topk overflow")
	}
	x.TopK = (x.TopK + 7) &^ 7
	if x.TopK == 0 {
		x.TopK = 8
	}
	if seq > math.MaxInt32/x.TopK {
		return fail("index allocation overflow")
	}
	x.Indices = make([]int32, seq*x.TopK)
	for i := range x.Indices {
		x.Indices[i] = -1
	}
	for t, ids := range lists {
		copy(x.Indices[t*x.TopK:], ids)
		x.Lengths[t] = int32(len(ids))
	}
	if !call.live() {
		return fail("session ownership changed before dispatch")
	}
	opened := m.v41NowNanos()
	out, err := call.backend.V41SparsePrefill(x)
	m.v41NoteAttentionContraction(opened)
	if err != nil {
		return nil, v41StageErr(v41StageAttention, l, err)
	}
	if !call.live() {
		return fail("session ownership changed after dispatch")
	}
	if len(out.Values) != seq*32768 || len(out.MaxLogits) != seq*64 || len(out.LSE) != seq*64 {
		return fail("output shape")
	}
	result := make([][]float32, seq)
	for t := range result {
		row := out.Values[t*32768 : (t+1)*32768]
		for _, v := range row {
			if !finite32(v) {
				return fail("nonfinite output")
			}
		}
		result[t] = append([]float32(nil), row...)
	}
	// Fused donor already applied inverse RoPE; never rotate this result twice.
	return result, nil
}
