package ggufload

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func estimateVulkanStreamedExperts(in PreflightInput) (preflightEstimate, error) {
	if !in.StreamedExperts || !in.VulkanMixedQ4K || in.Backend == nil || in.Backend.Name() != "vulkan" ||
		in.Lean || in.OffloadExperts || in.SingleResidencyQ4K ||
		in.StreamedExpertDeviceRingBytes <= 0 || in.StreamedDenseWorkingSetBytes <= 0 {
		return preflightEstimate{}, fmt.Errorf("%w: streamed preflight requires Vulkan mixed weights and positive ring and dense working-set bounds", ErrQ4KLoadEstimateUnsupported)
	}
	s := in.Source
	if s == nil || s.File == nil {
		return preflightEstimate{}, fmt.Errorf("%w: streamed preflight requires a weight source", ErrQ4KLoadEstimateUnsupported)
	}
	cfg, err := s.File.Config()
	if err != nil {
		return preflightEstimate{}, err
	}
	if in.ResidentQ2KEmbedding {
		if err := s.validateResidentQ2KEmbedding(cfg); err != nil {
			return preflightEstimate{}, err
		}
	}
	opts := []Q4KLoadOption{WithDenseKQuantResident(false), WithDenseQ2KResident(true),
		WithStreamedExperts(0), WithStreamedExpertDeviceRing(in.StreamedExpertDeviceRingBytes),
		WithStreamedDenseQ4KWorkingSet(in.StreamedDenseWorkingSetBytes), WithQ2KEmbeddingResident(in.ResidentQ2KEmbedding)}
	o, err := resolveQ4KLoadOptions(cfg, opts)
	if err != nil {
		return preflightEstimate{}, err
	}
	shards, err := s.FusedExpertTensors()
	if err != nil {
		return preflightEstimate{}, err
	}
	tier, err := buildExpertCheckpointTier(shards, 0)
	if err != nil {
		return preflightEstimate{}, err
	}
	if tier == nil {
		return preflightEstimate{}, fmt.Errorf("%w: checkpoint has no stageable routed expert slabs", ErrQ4KLoadEstimateUnsupported)
	}
	unstageable, err := s.UnstageableRoutedExpertSlabs()
	if err != nil {
		return preflightEstimate{}, err
	}
	if len(unstageable) > 0 {
		return preflightEstimate{}, fmt.Errorf("%w: unstageable routed expert slabs: %s", ErrQ4KLoadEstimateUnsupported, strings.Join(unstageable, ", "))
	}
	if err := tier.SetDeviceRingBudget(in.StreamedExpertDeviceRingBytes); err != nil {
		return preflightEstimate{}, err
	}
	peak, err := s.EstimateStreamedExpertHostLoadPeak(opts...)
	if err != nil {
		return preflightEstimate{}, err
	}
	streamed := map[string]bool{}
	for _, shard := range shards {
		for _, fused := range shard.Fused {
			streamed[fused.Name] = true
		}
	}
	deviceBytes := in.StreamedExpertDeviceRingBytes
	var readBytes, gatheredBytes, hostQ8Alignment int64
	for _, info := range s.File.Tensors {
		if streamed[info.Name] || (archShipsMTPOrVisionSidecar(cfg.ModelType) && glmMoeDsaMTPOrVisionTensor(info.Name)) {
			continue
		}
		if archIsDeepSeek41(cfg.ModelType) && deepseek41EngramTableTensor(info.Name) {
			payload, err := tensorPayloadBytes(info)
			if err != nil {
				return preflightEstimate{}, err
			}
			n, err := checkedEstimateUint64(payload, "Engram payload", info.Name)
			if err == nil {
				gatheredBytes, err = checkedEstimateAdd(gatheredBytes, n, "Engram gathered bytes")
			}
			if err != nil {
				return preflightEstimate{}, err
			}
			continue
		}
		ch, err := streamedTensorHostCharge(cfg, o, info)
		if err != nil {
			return preflightEstimate{}, err
		}
		if ch.skip {
			continue
		}
		if canon, mapped := CanonicalTensorNameArch(info.Name, cfg.ModelType); mapped {
			if name, keep := model.QuantSourceTensorName(cfg, canon); keep {
				for layer := 0; layer < cfg.NumLayers; layer++ {
					prefix := fmt.Sprintf("model.layers.%d.", layer)
					if name == prefix+"self_attn.qkv_proj.weight" || name == prefix+"mlp.gate_up_proj.weight" {
						return preflightEstimate{}, fmt.Errorf("%w: fused projection child allocations are not sized for %s", ErrQ4KLoadEstimateUnsupported, info.Name)
					}
				}
			}
		}
		readBytes, err = checkedEstimateAdd(readBytes, ch.payload, "streamed dense read bytes")
		if err != nil {
			return preflightEstimate{}, err
		}
		packedDeviceBytes := ch.packed
		if in.ResidentQ2KEmbedding && info.Type == TensorQ2_K && info.Name == "token_embd.weight" {
			packedDeviceBytes = 0
		}
		for _, n := range []int64{ch.stream, packedDeviceBytes, ch.f32} {
			aligned, err := conservativePageAllocation(n)
			if err == nil {
				deviceBytes, err = checkedEstimateAdd(deviceBytes, aligned, "streamed dense device bytes")
			}
			if err != nil {
				return preflightEstimate{}, err
			}
		}
		if ch.q8 > 0 {
			shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
			if err != nil {
				return preflightEstimate{}, err
			}
			_, allocation, err := estimateQ8Allocation(info.Name, shape)
			if err == nil {
				deviceBytes, err = checkedEstimateAdd(deviceBytes, allocation, "streamed dense Q8 device bytes")
			}
			if err == nil {
				hostQ8Alignment, err = checkedEstimateAdd(hostQ8Alignment, allocation-ch.q8, "streamed host Q8 alignment")
			}
			if err != nil {
				return preflightEstimate{}, err
			}
		}
	}
	gathered := (ActivatedExpertFit{HostGatheredBytes: gatheredBytes}).HostGatheredPageReserve()
	hostExtra, err := checkedEstimateAdd(gathered, hostQ8Alignment, "streamed host extra residency")
	if err != nil {
		return preflightEstimate{}, err
	}
	hostBytes, err := checkedEstimateAdd(peak.PeakAnonBytes, hostExtra, "streamed host peak")
	if err != nil {
		return preflightEstimate{}, err
	}
	hostResident, err := checkedEstimateAdd(peak.SteadyAnonBytes, hostExtra, "streamed host resident bytes")
	if err != nil {
		return preflightEstimate{}, err
	}
	plan := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeHost, Bytes: hostBytes, Detail: "gguf-vulkan-streamed-host-peak", DType: "mixed"},
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: deviceBytes, Detail: "gguf-vulkan-streamed-dense-and-ring", DType: "mixed"},
	}
	return preflightEstimate{plan: plan, readBytes: readBytes, hostResidentBytes: hostResident,
		deviceResidentBytes: deviceBytes, stagingBytes: peak.TransientBytes}, nil
}
