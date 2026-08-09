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

// RoutedExpertActiveSet is the header-only MoE active-set Lane F (#3074) pins from the witnessed
// GGUF header. It closes BOTH roofline inputs the ceiling doc had only ESTIMATED:
//
//   - active-bytes/token — the single-stream decode divisor. RoutedResident is the resident bytes
//     of ALL routed experts (the full unsharded band — for GLM-5.2 UD-Q4_K_M ~414 GiB across 256
//     experts, i.e. PerExpert ~1.619 GiB; the 51.80 GiB EP-8 figure is one rank's 32-expert shard,
//     not the whole band). ActivePerToken = K × PerExpert is the routed stream a token pulls;
//     ActiveBytesPerToken adds the NonExpertResident (attention / dense / shared-expert / router /
//     embedding) stream, so it is the full per-token byte divisor (an UPPER BOUND — the true decode
//     stream is a little lower, since the token-embedding read is a gather, not a full sweep).
//     ActiveBytesPerTokenSwept applies exactly that correction: when embeddings are UNTIED it drops
//     the input token_embd table (a get_rows gather, not swept), giving the header-tight sweep.
//   - active-params/token — the FLOP divisor. Params are element counts (tensorElems), quant-
//     independent: PerExpertParams = RoutedParams / NumExperts, and ActiveParamsPerToken =
//     K × PerExpertParams + NonExpertParams.
//
// Everything here is DERIVED header arithmetic, not the box-side per-op byte trace; the per-op trace
// remains the separate GPU witness. The two active-*/token fields are 0 until K (expert_used_count)
// is present in the header — the one scalar this derivation waits on.
type RoutedExpertActiveSet struct {
	NumExperts     int   // expert_count
	ExpertsUsed    int   // expert_used_count (K); 0 when the header omits it (active-*/token pending)
	MoELayers      int   // distinct block ordinals carrying a batched routed-expert tensor — the DENOMINATOR that turns the whole-model routed band into a per-layer one (a hybrid checkpoint's dense prefix carries none, so this is not block_count)
	RoutedResident int64 // sum of every batched routed-expert tensor payload (all experts, unsharded)
	PerExpert      int64 // RoutedResident / NumExperts — one expert's resident bytes across every MoE layer
	ActivePerToken int64 // K × PerExpert — routed-expert bytes a decoded token streams (0 when K unread)

	RoutedParams         int64 // routed-expert element count (all experts, unsharded)
	PerExpertParams      int64 // RoutedParams / NumExperts
	NonExpertResident    int64 // resident bytes of every tensor that is NOT a batched routed expert
	NonExpertParams      int64 // element count of every non-routed tensor
	ActiveBytesPerToken  int64 // ActivePerToken + NonExpertResident — full per-token divisor, UPPER BOUND (0 when K unread)
	ActiveParamsPerToken int64 // K × PerExpertParams + NonExpertParams (0 when K unread)

	// Embedding-gather correction (the one named looseness in the ActiveBytesPerToken UPPER BOUND).
	// The input token-embedding is read by a get_rows GATHER at decode — one row (~hidden bytes) per
	// token, not the whole [vocab×hidden] table — yet the full table sits in NonExpertResident. When
	// embeddings are UNTIED (a distinct output.weight/lm_head carries the output projection, which IS
	// swept per token), that table is counted but not swept, so the upper bound overcounts it.
	InputEmbedResident       int64 // token_embd.weight resident bytes (0 if absent from the header)
	InputEmbedGather         int64 // bytes subtracted for the input-embedding gather: InputEmbedResident when UNTIED, else 0 (tied ⇒ the table IS the swept output projection, no saving)
	ActiveBytesPerTokenSwept int64 // ActiveBytesPerToken − InputEmbedGather — the header-tight per-token sweep (≤ the upper bound). 0 when K unread.
}

// RoutedExpertActiveSet derives the Lane F (#3074) active-set from the header alone — it reads NO
// tensor payloads. In one pass over the parsed tensor directory it sums, per tensor, the on-disk
// block payload (tensorPayloadBytes) and the element count (tensorElems), splitting each into the
// batched routed-expert band (the tensors EstimateExpertParallelLoadMemoryPlan shards) and the
// non-expert remainder (attention / dense / shared-expert / router / embedding). Dividing the routed
// band by expert_count gives per-expert resident bytes and params; multiplying by expert_used_count
// (K) gives the routed stream a single decoded token pulls — K experts fire per MoE layer and the
// summed band already spans every MoE layer, so K×(band/experts) is the whole active routed stream.
// Adding the non-expert remainder yields active-bytes/token and active-params/token, the two divisors
// the GPU-server roofline had only ESTIMATED; the live per-op trace remains the separate box-side
// witness. ok=false for a model with no batched expert axis (non-MoE, or an arch that does not carry
// routed experts as batched GGUF blobs).
func (s *WeightSource) RoutedExpertActiveSet() (RoutedExpertActiveSet, bool, error) {
	cfg, err := s.File.Config()
	if err != nil {
		return RoutedExpertActiveSet{}, false, err
	}
	if !archUsesGGUFBatchedMoEExperts(cfg.ModelType) || cfg.NumExperts <= 0 {
		return RoutedExpertActiveSet{}, false, nil
	}
	var routedBytes, routedElems, totalBytes, totalElems, inputEmbedBytes uint64
	moeLayers := map[int]struct{}{}
	for _, info := range s.File.Tensors {
		n, err := tensorPayloadBytes(info)
		if err != nil {
			return RoutedExpertActiveSet{}, false, fmt.Errorf("gguf: active-set tensor %s: %w", info.Name, err)
		}
		e, err := tensorElems(info)
		if err != nil {
			return RoutedExpertActiveSet{}, false, fmt.Errorf("gguf: active-set tensor %s: %w", info.Name, err)
		}
		if totalBytes > math.MaxUint64-n || totalElems > math.MaxUint64-e {
			return RoutedExpertActiveSet{}, false, fmt.Errorf("gguf: active-set totals overflow uint64")
		}
		totalBytes += n
		totalElems += e
		if layer, _, ok := glmMoeDsaBatchedExpert(info.Name); ok {
			routedBytes += n
			routedElems += e
			moeLayers[layer] = struct{}{}
		}
		if strings.EqualFold(info.Name, "token_embd.weight") {
			inputEmbedBytes = n
		}
	}
	if routedBytes == 0 {
		return RoutedExpertActiveSet{}, false, nil
	}
	if totalBytes > math.MaxInt64 || totalElems > math.MaxInt64 {
		return RoutedExpertActiveSet{}, false, fmt.Errorf("gguf: active-set totals %d/%d overflow int64", totalBytes, totalElems)
	}
	perExpert := int64(routedBytes) / int64(cfg.NumExperts)
	perExpertParams := int64(routedElems) / int64(cfg.NumExperts)
	nonExpertBytes := int64(totalBytes - routedBytes)
	nonExpertParams := int64(totalElems - routedElems)
	as := RoutedExpertActiveSet{
		NumExperts:        cfg.NumExperts,
		ExpertsUsed:       cfg.NumExpertsPerTok,
		MoELayers:         len(moeLayers),
		RoutedResident:    int64(routedBytes),
		PerExpert:         perExpert,
		RoutedParams:      int64(routedElems),
		PerExpertParams:   perExpertParams,
		NonExpertResident: nonExpertBytes,
		NonExpertParams:   nonExpertParams,
	}
	if cfg.NumExpertsPerTok > 0 {
		k := int64(cfg.NumExpertsPerTok)
		if perExpert != 0 && k > math.MaxInt64/perExpert {
			return RoutedExpertActiveSet{}, false, fmt.Errorf("gguf: active-bytes/token overflows int64")
		}
		if perExpertParams != 0 && k > math.MaxInt64/perExpertParams {
			return RoutedExpertActiveSet{}, false, fmt.Errorf("gguf: active-params/token overflows int64")
		}
		as.ActivePerToken = perExpert * k
		routedActiveParams := perExpertParams * k
		if as.ActivePerToken > math.MaxInt64-nonExpertBytes || routedActiveParams > math.MaxInt64-nonExpertParams {
			return RoutedExpertActiveSet{}, false, fmt.Errorf("gguf: active-set/token + non-expert overflows int64")
		}
		as.ActiveBytesPerToken = as.ActivePerToken + nonExpertBytes
		as.ActiveParamsPerToken = routedActiveParams + nonExpertParams

		// Embedding-gather correction. The input token-embedding is a get_rows GATHER at decode
		// (one row per token, ~hidden bytes — negligible), not a full sweep of the [vocab×hidden]
		// table. When embeddings are UNTIED, a distinct output.weight/lm_head carries the swept
		// output projection, so the token_embd table sits in NonExpertResident yet is not swept:
		// subtract it to get the header-tight per-token sweep. When TIED, the same table IS the
		// swept output projection (there is no separate output.weight), so there is no saving.
		as.InputEmbedResident = int64(inputEmbedBytes)
		if !cfg.TieWordEmbeddings && inputEmbedBytes > 0 && int64(inputEmbedBytes) < as.ActiveBytesPerToken {
			as.InputEmbedGather = int64(inputEmbedBytes)
		}
		as.ActiveBytesPerTokenSwept = as.ActiveBytesPerToken - as.InputEmbedGather
	}
	return as, true, nil
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
	return s.estimateCPUOffloadExpertsMemoryPlan(1)
}

// EstimateCPUOffloadExpertsExpertParallelMemoryPlan is EstimateCPUOffloadExpertsMemoryPlan for a
// SHARDED expert-parallel rank: the same device/host partition, except batched routed-expert blobs
// are charged only for the BUSIEST rank's contiguous band. A --cpu-offload-experts rank in a
// sharded EP serve admits only its band [Lo,Hi) into the host expert pool (WithExpertShard), so
// charging the whole routed set to every rank overstates host demand ~ranks-fold and fires the
// host-scope refusal on a serve that actually fits (#4952). It shards on the SAME
// compute.ExpertParallelLargestBandExperts math as EstimateExpertParallelLoadMemoryPlan and the
// authoritative rank-local gate (#2997), so no two estimates on the load path can disagree.
//
// Shared experts stay replicated — only the batched routed blobs shard. ranks <= 1, a non-batched-
// MoE arch, or an unconfigurable header all fall back to the full-model plan, byte-identical.
func (s *WeightSource) EstimateCPUOffloadExpertsExpertParallelMemoryPlan(ranks int) (compute.MemoryPlan, error) {
	return s.estimateCPUOffloadExpertsMemoryPlan(ranks)
}

func (s *WeightSource) estimateCPUOffloadExpertsMemoryPlan(ranks int) (compute.MemoryPlan, error) {
	arch, _ := s.File.String("general.architecture")
	modelType := canonicalGGUFArch(arch)
	// band>0 only for a real sharded EP rank on a batched-MoE arch; everything else keeps the
	// full-model accounting untouched (and never reads Config, so no new error path appears on
	// the ranks<=1 call this method has always served).
	band, experts := 0, 0
	if ranks > 1 {
		cfg, err := s.File.Config()
		if err != nil {
			return nil, err
		}
		if archUsesGGUFBatchedMoEExperts(cfg.ModelType) && cfg.NumExperts > 0 {
			if _, err := model.ExpertParallelPlan(cfg.NumExperts, ranks); err != nil {
				return nil, err
			}
			band, experts = compute.ExpertParallelLargestBandExperts(cfg.NumExperts, ranks), cfg.NumExperts
		}
	}
	type key struct {
		class compute.MemoryClass
		scope compute.MemoryScope
		dtype string
		shard bool
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
		shard := false
		if experts > 0 {
			if _, _, ok := glmMoeDsaBatchedExpert(info.Name); ok {
				n, err = scaleExpertBandBytes(n, band, experts)
				if err != nil {
					return nil, fmt.Errorf("gguf: estimate offload tensor %s: %w", info.Name, err)
				}
				shard = true
			}
		}
		k := key{class: compute.MemoryWeights, scope: compute.MemoryScopeDevice, dtype: ggufTensorDTypeLabel(info.Type), shard: shard}
		if hostExpert {
			k = key{class: compute.MemoryOffload, scope: compute.MemoryScopeHost, dtype: ggufTensorDTypeLabel(info.Type), shard: shard}
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
		if keys[i].dtype != keys[j].dtype {
			return keys[i].dtype < keys[j].dtype
		}
		return !keys[i].shard && keys[j].shard
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
			// Keep the replicated host bytes (shared experts, charged to every rank) and the
			// per-rank routed band in SEPARATE rows, so a refusal message says which of the two
			// the host pool could not take. Only a sharded rank ever emits the -shard row.
			detail = "gguf-host-expert-offload"
			if k.shard {
				detail = "gguf-host-expert-offload-shard"
			}
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
	// The MTP ("nextn") head + vision tower carry no canonical HF mapping. Classify them as
	// non-expert (never a mapping error) via the UNGATED union: when model.RetainMTP retains
	// the MTP head, the offload estimator no longer skips it before this helper, so it must
	// still classify (as a device weight) rather than fall through to CanonicalTensorNameArch
	// and reject a real GLM-5.2, DeepSeek-V3, or Qwen3.6 checkpoint.
	if archShipsMTPOrVisionSidecar(modelType) && glmMoeDsaMTPOrVisionTensor(name) {
		return false, nil
	}
	if archUsesMLAMoELayout(modelType) {
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
