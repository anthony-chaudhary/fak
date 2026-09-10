package model

import (
	"fmt"
	"math"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func qwen35MTPResidentPrefixLen(target *Session) int {
	if target == nil {
		return 0
	}
	if target.Backend != nil {
		if target.halKV == nil {
			return 0
		}
		return target.halKV.Len()
	}
	if target.Cache == nil {
		return 0
	}
	return target.Cache.Len()
}

func qwen35MTPDepthNTargetMatchesCommitted(target *Session, committed []int) bool {
	resident := qwen35MTPResidentPrefixLen(target)
	if target == nil || resident > len(committed) {
		return false
	}
	return qwen35MTPDepthNTargetHasEvaluatedPrefix(target, committed[:resident])
}

func qwen35MTPDepthNTargetHasEvaluatedPrefix(target *Session, prefix []int) bool {
	if target == nil || len(prefix) > qwen35MTPResidentPrefixLen(target) {
		return false
	}
	var lineage []uint32
	if target.Backend != nil {
		if target.halLineage.fault != "" || len(target.halLineage.ids) < len(prefix) {
			return false
		}
		lineage = target.halLineage.ids
	} else {
		if target.Cache == nil || target.Cache.lineage.fault != "" || len(target.Cache.lineage.ids) < len(prefix) {
			return false
		}
		lineage = target.Cache.lineage.ids
	}
	target.targetHiddenMu.RLock()
	defer target.targetHiddenMu.RUnlock()
	if len(target.targetHidden) < len(prefix) || len(target.targetHiddenTokens) < len(prefix) {
		return false
	}
	for i, token := range prefix {
		if token < 0 || uint64(token) > math.MaxUint32 || lineage[i] != uint32(token) || target.targetHiddenTokens[i] != token || len(target.targetHidden[i]) != target.M.Cfg.HiddenSize {
			return false
		}
	}
	return true
}

// qwen35MTPPrepareTargetCapture publishes capture only after resident draft
// construction succeeds. A warm/restored target is accepted only when every
// resident position already has exact raw-hidden and token lineage.
func qwen35MTPPrepareTargetCapture(target *Session) error {
	resident := qwen35MTPResidentPrefixLen(target)
	target.targetHiddenMu.Lock()
	defer target.targetHiddenMu.Unlock()
	if resident == 0 {
		if len(target.targetHidden) != 0 || len(target.targetHiddenTokens) != 0 {
			return &Qwen35MTPSpecDecodeUnsupportedError{Reason: "fresh target has stale raw-hidden history"}
		}
		target.captureTargetHidden = true
		return nil
	}
	if !target.captureTargetHidden || len(target.targetHidden) != resident || len(target.targetHiddenTokens) != resident {
		return &Qwen35MTPSpecDecodeUnsupportedError{Reason: "restored target lacks complete raw-hidden history; cold reset and prefill after constructing the MTP draft session"}
	}
	var lineage []uint32
	if target.Backend != nil {
		if target.halLineage.fault != "" || len(target.halLineage.ids) != resident {
			return &Qwen35MTPSpecDecodeUnsupportedError{Reason: "restored target lacks exact device token lineage"}
		}
		lineage = target.halLineage.ids
	} else {
		if target.Cache == nil || target.Cache.lineage.fault != "" || len(target.Cache.lineage.ids) != resident {
			return &Qwen35MTPSpecDecodeUnsupportedError{Reason: "restored target lacks exact host token lineage"}
		}
		lineage = target.Cache.lineage.ids
	}
	for pos := 0; pos < resident; pos++ {
		if len(target.targetHidden[pos]) != target.M.Cfg.HiddenSize || target.targetHiddenTokens[pos] < 0 || uint64(target.targetHiddenTokens[pos]) > math.MaxUint32 || uint32(target.targetHiddenTokens[pos]) != lineage[pos] {
			return &Qwen35MTPSpecDecodeUnsupportedError{Reason: fmt.Sprintf("restored target raw-hidden history diverges at position %d", pos)}
		}
		for _, value := range target.targetHidden[pos] {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return &Qwen35MTPSpecDecodeUnsupportedError{Reason: fmt.Sprintf("restored target raw-hidden history is non-finite at position %d", pos)}
			}
		}
	}
	return nil
}

func validateQwen35MTPVulkanTarget(target *Session) error {
	if target.Backend == nil || !strings.EqualFold(target.Backend.Name(), "vulkan") {
		return &Qwen35MTPSpecDecodeUnsupportedError{Reason: "resident MTP target capture currently requires the Vulkan backend"}
	}
	if (target.Quant && !target.Q4K) || target.Q4 || target.F16 || target.GPTQ || target.Metal || target.MetalQ4K || target.PrecisionPolicy != nil {
		return &Qwen35MTPSpecDecodeUnsupportedError{Reason: "Vulkan MTP target supports the retained f32 or Q4_K model path only"}
	}
	if _, split := target.validateDenseGPULayers(); split {
		return &Qwen35MTPSpecDecodeUnsupportedError{Reason: "Vulkan MTP target excludes split host/device layer placement"}
	}
	if _, advertised, err := qwen35SequencePrefillBackend(target.Backend); err != nil || !advertised {
		if err != nil {
			return err
		}
		return &Qwen35MTPSpecDecodeUnsupportedError{Reason: "Vulkan target lacks the resident Qwen3.8 sequence operation"}
	}
	if cap, ok := target.Backend.(compute.Qwen35SequenceAllLogitsBackend); !ok || cap.Qwen35SequenceAllLogitsPath() != compute.Qwen35SequenceAllLogitsPath {
		return &Qwen35MTPSpecDecodeUnsupportedError{Reason: "Vulkan target lacks all-row final projection"}
	}
	if cap, ok := target.Backend.(compute.Qwen35SequenceRawHiddenBackend); !ok || cap.Qwen35SequenceRawHiddenPath() != compute.Qwen35SequenceRawHiddenPath {
		return &Qwen35MTPSpecDecodeUnsupportedError{Reason: "Vulkan target lacks exact raw-hidden capture"}
	}
	if cap, ok := target.Backend.(compute.Qwen35SequencePrefixReplayBackend); !ok || cap.Qwen35SequencePrefixReplayPath() != compute.Qwen35SequencePrefixReplayPath {
		return &Qwen35MTPSpecDecodeUnsupportedError{Reason: "Vulkan target lacks retained prefix replay"}
	}
	return nil
}
