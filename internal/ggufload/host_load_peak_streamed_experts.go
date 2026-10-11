package ggufload

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// streamedDenseMaterializeWindowBytes mirrors model.q4kMaterializeWindowBytes: a lazy dense
// tensor faulted without a mapping allocates its page-aligned payload once and fills it through
// one reusable read window of this size.
const streamedDenseMaterializeWindowBytes int64 = 64 << 20

// EstimateStreamedExpertHostLoadPeak predicts the host anonymous peak of the streamed-expert load
// (LoadModelQ4KStreamedExperts) whose dense side is held as a bounded streamed working set
// (WithStreamedDenseQ4KWorkingSet). It is the MoE route EstimateQ4KHostLoadPeak does not qualify.
//
// Routed-expert slabs the tier owns and V4.1 Engram tables are never materialized, so they are not
// charged; the tier's own host retention budget is. Dense tensors the loader holds as range
// descriptors are charged at min(eligible, declared working set), and the largest of them is
// charged again as the one-tensor staging transient its device upload materializes. Every other
// tensor is an owned copy: native Q8 for a 2-D quant weight, F32 otherwise, and F32 for a name
// with no canonical mapping, so an unrecognized tensor overcharges rather than escaping the gate.
func (s *WeightSource) EstimateStreamedExpertHostLoadPeak(opts ...Q4KLoadOption) (HostLoadPeak, error) {
	if s == nil || s.File == nil {
		return HostLoadPeak{}, fmt.Errorf("gguf: streamed-expert host peak requires a weight source")
	}
	cfg, err := s.File.Config()
	if err != nil {
		return HostLoadPeak{}, err
	}
	o, err := resolveQ4KLoadOptions(cfg, opts)
	if err != nil {
		return HostLoadPeak{}, err
	}
	if !o.streamedExperts || !o.streamedDenseBounded || o.expertShardSet {
		return HostLoadPeak{}, fmt.Errorf("%w: streamed-expert host peak requires streamed experts with a bounded streamed-dense working set", ErrQ4KLoadEstimateUnsupported)
	}
	shards, err := s.FusedExpertTensors()
	if err != nil {
		return HostLoadPeak{}, err
	}
	streamed := map[string]bool{}
	for _, sh := range shards {
		for _, f := range sh.Fused {
			streamed[f.Name] = true
		}
	}

	var out HostLoadPeak
	byType := map[TensorType]*HostLoadTypeRow{}
	var windows []int64
	var eligible, largestLazy int64
	kvb := kvbBufferSim{buffered: map[int]int64{}}
	add := func(dst *int64, n int64) error {
		if n < 0 || *dst > math.MaxInt64-n {
			return fmt.Errorf("gguf: streamed-expert host peak estimate overflows int64")
		}
		*dst += n
		return nil
	}
	for _, info := range s.File.Tensors {
		if streamed[info.Name] ||
			(archIsDeepSeek41(cfg.ModelType) && deepseek41EngramTableTensor(info.Name)) ||
			(archShipsMTPOrVisionSidecar(cfg.ModelType) && glmMoeDsaMTPOrVisionTensor(info.Name)) {
			continue
		}
		ch, err := streamedTensorHostCharge(cfg, o, info)
		if err != nil {
			return HostLoadPeak{}, err
		}
		if ch.skip {
			continue
		}
		row := byType[info.Type]
		if row == nil {
			row = &HostLoadTypeRow{Type: info.Type.String()}
			byType[info.Type] = row
		}
		row.Tensors++
		for _, d := range []struct {
			dst *int64
			n   int64
		}{
			{&row.PayloadBytes, ch.payload}, {&row.StreamBytes, ch.stream}, {&eligible, ch.stream},
			{&row.Q8Bytes, ch.q8}, {&row.F32Bytes, ch.f32}, {&out.ArenaCopyBytes, ch.f32},
		} {
			if err := add(d.dst, d.n); err != nil {
				return HostLoadPeak{}, err
			}
		}
		largestLazy = max(largestLazy, ch.stream)
		window, err := kvb.admit(ch)
		if err != nil {
			return HostLoadPeak{}, err
		}
		windows = append(windows, window)
	}

	for _, r := range byType {
		out.Rows = append(out.Rows, *r)
		if err := add(&out.SteadyAnonBytes, r.AnonBytes()); err != nil {
			return HostLoadPeak{}, err
		}
	}
	sort.Slice(out.Rows, func(i, j int) bool {
		if a, b := out.Rows[i].AnonBytes(), out.Rows[j].AnonBytes(); a != b {
			return a > b
		}
		return out.Rows[i].Type < out.Rows[j].Type
	})
	if err := add(&out.SteadyAnonBytes, min(eligible, max(o.streamedDenseBytes, 0))); err != nil {
		return HostLoadPeak{}, err
	}
	if err := add(&out.SteadyAnonBytes, max(o.streamedExpertBytes, 0)); err != nil {
		return HostLoadPeak{}, err
	}
	out.Workers = loadWorkers()
	if out.WorkerWindowBytes, err = maxContiguousWindowSum(windows, out.Workers); err != nil {
		return HostLoadPeak{}, err
	}
	staging := int64(0)
	if largestLazy > 0 {
		staging, err = sumWindow(largestLazy, min(largestLazy, streamedDenseMaterializeWindowBytes))
		if err != nil {
			return HostLoadPeak{}, err
		}
	}
	// The tensor windows (load), Build's arena coalescing copy, and the device upload's staging
	// are disjoint phases of one load.
	out.TransientBytes = max(out.WorkerWindowBytes, out.ArenaCopyBytes, staging)
	out.PeakAnonBytes = out.SteadyAnonBytes
	if err := add(&out.PeakAnonBytes, out.TransientBytes); err != nil {
		return HostLoadPeak{}, err
	}
	return out, nil
}

// streamedTensorCharge is one non-streamed tensor's share of the streamed-expert host load.
type streamedTensorCharge struct {
	skip    bool  // dropped by the loader (QuantSourceTensorName keep=false)
	payload int64 // packed bytes on disk
	stream  int64 // bounded streamed-dense candidate (a range descriptor; no anon bytes at load)
	q8      int64 // steady native Q8 (a 2-D quant weight, plus a tied head's Q8 copy)
	f32     int64 // steady F32 builder chunk; Build coalesces it into the arena with a second copy
	window  int64 // the tensor's own worker-window transient
	// kvbLayer >= 0 marks an MLA KV-b half (attn_k_b/attn_v_b) that the collector buffers as an
	// owned f32 copy of kvbF32 bytes until its partner arrives.
	kvbLayer int
	kvbF32   int64
}

// streamedTensorHostCharge mirrors computeQ4KTensorWork + applyQ4KTensorWork for one tensor of
// the streamed-expert load. Each window holds the raw read and the f32 dequant; a Q8 weight also
// holds the normalized f32 it quantizes from, an F32 tensor holds the normalized copy when its
// canonical transform reallocates, and an MLA KV-b half holds the append copy it buffers.
func streamedTensorHostCharge(cfg model.Config, o q4kLoadOptions, info TensorInfo) (streamedTensorCharge, error) {
	ch := streamedTensorCharge{kvbLayer: -1}
	payloadU, err := tensorPayloadBytes(info)
	if err != nil {
		return ch, err
	}
	elemsU, err := tensorElems(info)
	if err != nil {
		return ch, err
	}
	if ch.payload, err = checkedEstimateUint64(payloadU, "payload", info.Name); err != nil {
		return ch, err
	}
	f32, err := checkedEstimateMulUint64(elemsU, 4, "f32 bytes", info.Name)
	if err != nil {
		return ch, err
	}
	q8U, err := estimateNativeQ8LogicalBytes(elemsU)
	if err != nil {
		return ch, err
	}
	q8, err := checkedEstimateUint64(q8U, "q8 bytes", info.Name)
	if err != nil {
		return ch, err
	}
	if archUsesMLAMoELayout(cfg.ModelType) {
		if layer, _, ok := glmMoeDsaSplitKVB(info.Name); ok {
			// The merged kv_b_proj lands as F32 or Q8; charge the larger per half.
			ch.f32 = max(ch.payload, f32, q8)
			ch.kvbLayer, ch.kvbF32 = layer, f32
			ch.window, err = sumWindow(ch.payload, f32, f32)
			return ch, err
		}
	}
	canon, mapped := CanonicalTensorNameArch(info.Name, cfg.ModelType)
	if mapped {
		var keep bool
		if canon, keep = model.QuantSourceTensorName(cfg, canon); !keep {
			ch.skip = true
			return ch, nil
		}
	}
	if mapped && streamedDenseLazy(cfg, o, info.Type, canon) {
		ch.stream = ch.payload
		return ch, nil
	}
	shape, shapeErr := modelShapeFromGGUFDims(info.Name, info.Dims)
	if mapped && shapeErr == nil && len(shape) == 2 && shape[1]%32 == 0 && model.IsQuantWeight(canon) {
		ch.q8 = q8
		ch.window, err = sumWindow(ch.payload, f32, f32)
		return ch, err
	}
	ch.f32 = max(ch.payload, f32)
	copied := int64(0)
	if !mapped || normalizeCanonicalMayCopy(canon, cfg) {
		copied = f32
	}
	if ch.window, err = sumWindow(ch.payload, f32, copied); err != nil {
		return ch, err
	}
	// A tied embedding stays F32 for the row gather and is also quantized as the LM head.
	if cfg.TieWordEmbeddings && canon == "model.embed_tokens.weight" && shapeErr == nil && len(shape) == 2 {
		ch.q8 = q8
		ch.window, err = sumWindow(ch.window, q8)
	}
	return ch, err
}

// kvbBufferSim replays the collector's MLA KV-b pairing (bufferGLMKVBHalf) in header order: each
// half's owned f32 copy stays buffered until its partner arrives, and the merge then holds the
// buffered pair, the merged tensor and its normalized copy at once.
type kvbBufferSim struct {
	buffered map[int]int64 // layer -> f32 bytes of the half waiting for its partner
	held     int64         // sum of buffered
}

// admit returns ch's window including every KV-b half still buffered at that point and, for the
// half that completes a pair, the merge transient.
func (k *kvbBufferSim) admit(ch streamedTensorCharge) (int64, error) {
	if ch.kvbLayer < 0 {
		return sumWindow(ch.window, k.held)
	}
	partner, ok := k.buffered[ch.kvbLayer]
	if !ok {
		k.buffered[ch.kvbLayer] = ch.kvbF32
		k.held += ch.kvbF32
		return sumWindow(ch.window, k.held)
	}
	delete(k.buffered, ch.kvbLayer)
	k.held -= partner
	pair, err := sumWindow(partner, ch.kvbF32)
	if err != nil {
		return 0, err
	}
	return sumWindow(ch.window, k.held, pair, pair, pair)
}

// normalizeCanonicalMayCopy reports whether normalizeCanonicalTensorData can return a fresh
// slice for canon rather than its input: every qwen35-hybrid transform and the rotary unpermute.
func normalizeCanonicalMayCopy(canon string, cfg model.Config) bool {
	if cfg.IsQwen35Hybrid() {
		return true
	}
	return strings.HasSuffix(canon, ".self_attn.q_proj.weight") || strings.HasSuffix(canon, ".self_attn.k_proj.weight")
}

// streamedDenseLazy is the loader's streamed-dense dispatch (computeQ4KTensorWork): an eligible
// dense Q4_K, or a retained eligible Q2_K/Q3_K/Q5_K/Q6_K, becomes a range descriptor.
func streamedDenseLazy(cfg model.Config, o q4kLoadOptions, t TensorType, canon string) bool {
	if !o.streamedDenseQ4K {
		return false
	}
	if t == TensorQ4_K {
		return model.ResidentQ4KEligible(cfg, canon)
	}
	return denseKQuantRetained(o, t) && lazyDenseKQuantBoundedEligible(cfg, t, canon)
}

// StreamedDenseWorkingSetFromHeadroom sizes a bounded streamed-dense working set from the host
// memory left after a load's fixed peak (margin included), holding back reserve of that room. A
// host that cannot hold the fixed peak gets a one-byte bound rather than 0, because a zero working
// set declares stream-through with unbounded memoization; AdmitHostLoadPeak then refuses the load.
func StreamedDenseWorkingSetFromHeadroom(fixed HostLoadPeak, availBytes, marginBytes int64, reserve float64) int64 {
	room := availBytes - max(marginBytes, 0) - fixed.PeakAnonBytes
	return max(int64(float64(room)*(1-reserve)), 1)
}
