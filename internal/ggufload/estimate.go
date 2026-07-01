package ggufload

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// estimate.go — the load-time device-fit pre-check for the GGUF loader (issue #709; the
// capacity-bridge Plank 5, docs/explainers/hardware-limits-and-capacity.md). The Go load
// paths (WeightSource.QuantModelProfile / QuantModelQ4K) allocate optimistically (the
// make([]byte, ...) in TensorBytes, the dequant buffers); a model too big for the box
// OOM-panics mid-load, losing the sizing context of what was needed. EstimateLoadBytes lifts
// the bytes the loader will demand OFF THE HEADER ALONE (no tensor read, no full load), and
// FitOnDevice turns compute.FitsOnDevice's verdict into a typed refusal BEFORE the allocation
// — fail-open on a backend that cannot probe (cpu-ref), so the portable floor loads exactly
// as before.
//
// EstimateLoadBytes sums each tensor's on-disk block payload (tensorPayloadBytes: the bytes
// the loader must read). For the memory-lean resident paths this is the resident footprint to
// within a small constant factor: the direct-Q4_K path holds those bytes RAW (byte-for-byte
// the ggml layout the forward reads), and the Q8_0 re-quant path's resident bytes are the same
// order (out*in int8 codes + per-block f32 scales). It is therefore a faithful order-of-
// magnitude fit proxy for the lean paths the GGUF loader exists to serve; the f32 dequant path
// (WeightSource.Model / LoadModel) resident is larger (elems*4), so treat it as a lower bound
// there. Either way an oversize model exceeds a small KNOWN ceiling and is refused; an unknown
// ceiling (cpu-ref) never is.

// EstimateLoadBytes reports the GGUF weight payload the loader will read, summed from the
// parsed header's tensor directory WITHOUT reading a single tensor — so a caller can ask
// "will this fit?" before the make/append in QuantModelProfile/QuantModelQ4K. It walks
// s.File.Tensors (the header) and sums each tensor's block payload (tensorPayloadBytes), the
// same sizing TensorBytes uses to allocate its read buffer, so the estimate tracks the real
// allocation rather than guessing. Safe to call right after OpenWeights (which parses only the
// header); it never touches a tensor byte.
func (s *WeightSource) EstimateLoadBytes() (int64, error) {
	var total uint64
	for _, info := range s.File.Tensors {
		n, err := tensorPayloadBytes(info)
		if err != nil {
			return 0, fmt.Errorf("gguf: estimate tensor %s: %w", info.Name, err)
		}
		total += n
	}
	if total > math.MaxInt64 {
		return 0, fmt.Errorf("gguf: estimated load bytes %d overflow int64", total)
	}
	return int64(total), nil
}

// EstimateF32LoadBytes reports the resident f32 footprint of loading this GGUF through
// LoadModel/WeightSource.Model. Unlike EstimateLoadBytes (the raw/lean payload proxy),
// this counts every tensor by element-count * 4 bytes because the default dequant path
// expands quantized GGUF payloads into f32 resident weights. It is intentionally an
// upper-bound header estimate over the tensor directory; unmapped or later-skipped
// tensors still count, which is the safe direction for a pre-load capacity refusal.
func (s *WeightSource) EstimateF32LoadBytes() (int64, error) {
	var total uint64
	for _, info := range s.File.Tensors {
		elems, err := tensorElems(info)
		if err != nil {
			return 0, fmt.Errorf("gguf: estimate f32 tensor %s: %w", info.Name, err)
		}
		if elems > math.MaxUint64/4 || total > math.MaxUint64-elems*4 {
			return 0, fmt.Errorf("gguf: estimated f32 load bytes overflow uint64")
		}
		total += elems * 4
	}
	if total > math.MaxInt64 {
		return 0, fmt.Errorf("gguf: estimated f32 load bytes %d overflow int64", total)
	}
	return int64(total), nil
}

// EstimateLoadMemoryPlan is the classed form of EstimateLoadBytes. The whole estimate is a
// weights demand because the GGUF loader is admitting resident model weights; KV-cache,
// scratchpad, and offload staging are reserved separately by the caller's headroom or by a
// richer multi-demand plan.
func (s *WeightSource) EstimateLoadMemoryPlan() (compute.MemoryPlan, error) {
	byDType := map[string]uint64{}
	for _, info := range s.File.Tensors {
		n, err := tensorPayloadBytes(info)
		if err != nil {
			return nil, fmt.Errorf("gguf: estimate tensor %s: %w", info.Name, err)
		}
		dtype := ggufTensorDTypeLabel(info.Type)
		if byDType[dtype] > math.MaxUint64-n {
			return nil, fmt.Errorf("gguf: estimated load bytes overflow uint64")
		}
		byDType[dtype] += n
	}
	return ggufMemoryPlanByDType(compute.MemoryWeights, compute.MemoryScopeDevice, "gguf-load", byDType)
}

// EstimateExpertParallelLoadMemoryPlan estimates the resident per-rank GGUF weight plan for an
// expert-parallel MoE load. Non-expert tensors are replicated on every rank; batched routed-expert
// blobs are charged only for the busiest rank's contiguous expert band. The estimate is header-only
// like EstimateLoadMemoryPlan: it reads no tensor payloads, and it preserves dtype rows so the
// capacity refusal still names the storage mix.
func (s *WeightSource) EstimateExpertParallelLoadMemoryPlan(ranks int) (compute.MemoryPlan, error) {
	if ranks <= 1 {
		return s.EstimateLoadMemoryPlan()
	}
	cfg, err := s.File.Config()
	if err != nil {
		return nil, err
	}
	if !archUsesGGUFBatchedMoEExperts(cfg.ModelType) || cfg.NumExperts <= 0 {
		return s.EstimateLoadMemoryPlan()
	}
	if _, err := model.ExpertParallelPlan(cfg.NumExperts, ranks); err != nil {
		return nil, err
	}
	band := compute.ExpertParallelLargestBandExperts(cfg.NumExperts, ranks)
	replicatedByDType := map[string]uint64{}
	expertByDType := map[string]uint64{}
	for _, info := range s.File.Tensors {
		if glmMoeDsaSkipGGUFTensorForType(cfg.ModelType, info.Name) {
			continue
		}
		n, err := tensorPayloadBytes(info)
		if err != nil {
			return nil, fmt.Errorf("gguf: estimate expert-parallel tensor %s: %w", info.Name, err)
		}
		shardedExpert := false
		if _, _, ok := glmMoeDsaBatchedExpert(info.Name); ok {
			n, err = scaleExpertBandBytes(n, band, cfg.NumExperts)
			if err != nil {
				return nil, fmt.Errorf("gguf: estimate expert-parallel tensor %s: %w", info.Name, err)
			}
			shardedExpert = true
		}
		dtype := ggufTensorDTypeLabel(info.Type)
		byDType := replicatedByDType
		if shardedExpert {
			byDType = expertByDType
		}
		if byDType[dtype] > math.MaxUint64-n {
			return nil, fmt.Errorf("gguf: estimated expert-parallel load bytes overflow uint64")
		}
		byDType[dtype] += n
	}
	replicated, err := ggufMemoryPlanByDType(compute.MemoryWeights, compute.MemoryScopeDevice, "gguf-ep-replicated-load", replicatedByDType)
	if err != nil {
		return nil, err
	}
	sharded, err := ggufMemoryPlanByDType(compute.MemoryWeights, compute.MemoryScopeDevice, "gguf-ep-routed-expert-shard", expertByDType)
	if err != nil {
		return nil, err
	}
	return append(replicated, sharded...), nil
}

func scaleExpertBandBytes(total uint64, band, experts int) (uint64, error) {
	if band <= 0 || experts <= 0 {
		return 0, nil
	}
	num, den := uint64(band), uint64(experts)
	q, r := total/den, total%den
	if q > math.MaxUint64/num {
		return 0, fmt.Errorf("expert band byte estimate overflows uint64")
	}
	out := q * num
	if r == 0 {
		return out, nil
	}
	if r > math.MaxUint64/num {
		return 0, fmt.Errorf("expert band byte estimate overflows uint64")
	}
	rem := r * num
	add := rem / den
	if rem%den != 0 {
		add++
	}
	if out > math.MaxUint64-add {
		return 0, fmt.Errorf("expert band byte estimate overflows uint64")
	}
	return out + add, nil
}

// EstimateF32LoadMemoryPlan is the classed form of EstimateF32LoadBytes for the f32-resident
// device load path.
func (s *WeightSource) EstimateF32LoadMemoryPlan() (compute.MemoryPlan, error) {
	want, err := s.EstimateF32LoadBytes()
	if err != nil {
		return nil, err
	}
	return compute.MemoryPlan{{Class: compute.MemoryWeights, Bytes: want, Detail: "gguf-f32-load", DType: compute.F32.String()}}, nil
}

// EstimateCPUOffloadExpertsMemoryPlan estimates the --cpu-offload-experts placement without
// reading tensor payloads. Dense/router/attention tensors remain device-scoped weights; routed
// and shared expert tensors are host-scoped offload bytes. The partition uses the same canonical
// tensor names as the runtime split kernel (model.CPUOffloadExpertWeight), with GLM-DSA's batched
// routed-expert GGUF blobs classified before their loader-time 1->E split.
func (s *WeightSource) EstimateCPUOffloadExpertsMemoryPlan() (compute.MemoryPlan, error) {
	arch, _ := s.File.String("general.architecture")
	modelType := canonicalGGUFArch(arch)
	type key struct {
		class compute.MemoryClass
		scope compute.MemoryScope
		dtype string
	}
	by := map[key]uint64{}
	for _, info := range s.File.Tensors {
		if glmMoeDsaSkipGGUFTensorForType(modelType, info.Name) {
			continue
		}
		n, err := tensorPayloadBytes(info)
		if err != nil {
			return nil, fmt.Errorf("gguf: estimate offload tensor %s: %w", info.Name, err)
		}
		hostExpert, err := tensorCPUOffloadExpert(info.Name, modelType)
		if err != nil {
			return nil, err
		}
		k := key{class: compute.MemoryWeights, scope: compute.MemoryScopeDevice, dtype: ggufTensorDTypeLabel(info.Type)}
		if hostExpert {
			k = key{class: compute.MemoryOffload, scope: compute.MemoryScopeHost, dtype: ggufTensorDTypeLabel(info.Type)}
		}
		if by[k] > math.MaxUint64-n {
			return nil, fmt.Errorf("gguf: estimated device bytes overflow uint64")
		}
		by[k] += n
	}
	keys := make([]key, 0, len(by))
	for k := range by {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].scope != keys[j].scope {
			return keys[i].scope < keys[j].scope
		}
		if keys[i].class != keys[j].class {
			return keys[i].class < keys[j].class
		}
		return keys[i].dtype < keys[j].dtype
	})
	plan := make(compute.MemoryPlan, 0, len(keys))
	for _, k := range keys {
		total := by[k]
		if total == 0 {
			continue
		}
		if total > math.MaxInt64 {
			return nil, fmt.Errorf("gguf: estimated offload memory plan overflows int64")
		}
		detail := "gguf-device-dense-load"
		if k.scope == compute.MemoryScopeHost {
			detail = "gguf-host-expert-offload"
		}
		plan = append(plan, compute.MemoryDemand{
			Class:  k.class,
			Bytes:  int64(total),
			Detail: detail,
			Scope:  k.scope,
			DType:  k.dtype,
		})
	}
	return plan, nil
}

func ggufTensorDTypeLabel(t TensorType) string {
	label := strings.ToLower(strings.TrimSpace(t.String()))
	if label == "" {
		return "unknown"
	}
	return label
}

func ggufMemoryPlanByDType(class compute.MemoryClass, scope compute.MemoryScope, detail string, byDType map[string]uint64) (compute.MemoryPlan, error) {
	dtypes := make([]string, 0, len(byDType))
	for dtype := range byDType {
		dtypes = append(dtypes, dtype)
	}
	sort.Strings(dtypes)
	plan := make(compute.MemoryPlan, 0, len(dtypes))
	for _, dtype := range dtypes {
		total := byDType[dtype]
		if total == 0 {
			continue
		}
		if total > math.MaxInt64 {
			return nil, fmt.Errorf("gguf: estimated load bytes %d overflow int64", total)
		}
		plan = append(plan, compute.MemoryDemand{
			Class:  class,
			Bytes:  int64(total),
			Detail: detail,
			Scope:  scope,
			DType:  dtype,
		})
	}
	return plan, nil
}

func tensorCPUOffloadExpert(name, modelType string) (bool, error) {
	if modelType == "glm_moe_dsa" {
		// The MTP ("nextn") head + vision tower are skipped at load (the text forward never
		// reads them). Byte-accounting callers drop them before calling this helper; direct
		// callers still get a non-expert classification instead of a mapping error. This must
		// match the loader skip (glmMoeDsaSkipGGUFTensor) or the estimate would reject a real
		// GLM-5.2 checkpoint the loader happily loads.
		if glmMoeDsaSkipGGUFTensor(name) {
			return false, nil
		}
		if _, _, ok := glmMoeDsaSplitKVB(name); ok {
			return false, nil
		}
	}
	if archUsesGGUFBatchedMoEExperts(modelType) {
		if _, _, ok := glmMoeDsaBatchedExpert(name); ok {
			return true, nil
		}
	}
	canon, ok := CanonicalTensorNameArch(name, modelType)
	if !ok {
		return false, fmt.Errorf("gguf: no canonical mapping for tensor %s", name)
	}
	return model.CPUOffloadExpertWeight(canon), nil
}

// FitOnDevice is the load-time device-fit refusal for a GGUF WeightSource: it estimates the
// load bytes off the header and returns a *compute.FitError ("needs ~W GiB, device has ~A
// GiB") ONLY when be is a capacity-reporting backend that KNOWS the model exceeds its ceiling.
// A backend that cannot probe (the cpu-ref floor, a device without a memory query) reports
// unknown capacity, so this returns nil — the load proceeds unchanged and the portable floor
// is never blocked (the fail-open contract). Call it BEFORE QuantModel / QuantModelQ4K to turn
// an oversize model into a typed refusal instead of an OOM panic; headroom in [0,1) reserves
// that fraction of the budget for the KV cache / activations / per-op scratch that do not
// pass through this single check (see compute.FitsOnDevice).
func (s *WeightSource) FitOnDevice(be compute.Backend, headroom float64) error {
	plan, err := s.EstimateLoadMemoryPlan()
	if err != nil {
		return err
	}
	return compute.RefuseMemoryPlanIfTooBig(be, plan, headroom)
}

// FitF32OnDevice is FitOnDevice for the f32-resident GGUF load path. Use this before
// LoadModel/WeightSource.Model when a device backend will hold f32 weights.
func (s *WeightSource) FitF32OnDevice(be compute.Backend, headroom float64) error {
	plan, err := s.EstimateF32LoadMemoryPlan()
	if err != nil {
		return err
	}
	return compute.RefuseMemoryPlanIfTooBig(be, plan, headroom)
}

// FitCPUOffloadExpertsOnDevice is FitOnDevice for the --cpu-offload-experts placement. Host
// expert bytes remain visible as MemoryOffload demands but do not count against device capacity.
func (s *WeightSource) FitCPUOffloadExpertsOnDevice(be compute.Backend, headroom float64) error {
	plan, err := s.EstimateCPUOffloadExpertsMemoryPlan()
	if err != nil {
		return err
	}
	return compute.RefuseMemoryPlanIfTooBig(be, plan, headroom)
}
