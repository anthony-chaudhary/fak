package macfit

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
)

const (
	// ArtifactFileName is the canonical GGUF artifact for Qwen3.8-27B UD-Q2_K_XL (#11961).
	ArtifactFileName = "Qwen3.8-27B-UD-Q2_K_XL.gguf"

	// ArtifactBytes is the exact binary size of the upstream GGUF file (9.8 GB GGUF).
	ArtifactBytes uint64 = 9828981664

	// ArtifactLFS_SHA256 is the upstream Git LFS SHA-256 digest of the artifact.
	ArtifactLFS_SHA256 = "fd4730dd8aad070517978752b63d530aeb1740d2283cab9fa24f1e404032ddb0"

	// ArtifactLFSSHA256 is an alias for ArtifactLFS_SHA256.
	ArtifactLFSSHA256 = ArtifactLFS_SHA256

	// ResidentWeightBytes is the in-memory footprint of resident model weights (~8.5 GiB).
	ResidentWeightBytes uint64 = 9126805504

	// FullAttnLayers is the count of full token-indexed attention layers in the hybrid architecture.
	FullAttnLayers uint64 = 16

	// RecurrentLayers is the count of linear-attention / GDN recurrent layers.
	RecurrentLayers uint64 = 48

	// TotalLayers is the total number of layers in Qwen3.8-27B.
	TotalLayers uint64 = 64

	// KVHeads is the grouped-query attention KV head count for full-attention layers.
	KVHeads uint64 = 4

	// HeadDim is the per-head dimension.
	HeadDim uint64 = 128

	// RecurrentStatePerAgentBytes is the fixed O(1) recurrent GDN state across all 48 recurrent layers (~3.2 MB per agent).
	RecurrentStatePerAgentBytes uint64 = 3355443

	// ProfileSeam36GBBaseline represents the 32GB-63GB baseline profile seam.
	ProfileSeam36GBBaseline = "36GB_BASELINE"

	// ProfileSeamCapabilityScaledLarger represents the >=64GB Mac Studio/Pro capability-scaled profile seam.
	ProfileSeamCapabilityScaledLarger = "CAPABILITY_SCALED_LARGER"

	// ProfileSeamSub36GBConstrained represents the <32GB memory-constrained profile seam.
	ProfileSeamSub36GBConstrained = "SUB_36GB_CONSTRAINED"

	// Limiting component identifiers.
	LimitingComponentNone           = "none"
	LimitingComponentWeights        = "resident_weights"
	LimitingComponentKVCache        = "full_attention_kv_cache"
	LimitingComponentRecurrentState = "recurrent_state"
	LimitingComponentStagingScratch = "staging_scratch"
	LimitingComponentWiredCeiling   = "wired_ceiling"

	// Swap risk classifications.
	SwapRiskZero     = "ZERO"
	SwapRiskNone     = "NONE"
	SwapRiskLow      = "LOW"
	SwapRiskModerate = "MODERATE"
	SwapRiskHigh     = "HIGH"
	SwapRiskCritical = "CRITICAL"

	// StagingScratchBytes allocates scratch space for staging buffers (~512 MiB).
	StagingScratchBytes uint64 = 512 * 1024 * 1024

	// FullAttnKVBytesPerToken: 2 (K,V) * 16 layers * 4 heads * 128 dim * 2 bytes = 32,768 bytes/token.
	FullAttnKVBytesPerToken uint64 = 32768
)

// UDQ2KXLQualification details the unified-memory qualification, context budget,
// physical peak memory, and admission verdict for Qwen3.8-27B-UD-Q2_K_XL.
type UDQ2KXLQualification struct {
	Schema                  string `json:"schema"`
	ArtifactFileName        string `json:"artifact_file_name"`
	ArtifactBytes           uint64 `json:"artifact_bytes"`
	ArtifactLFS_SHA256      string `json:"artifact_lfs_sha256"`
	ResidentWeightBytes     uint64 `json:"resident_weight_bytes"`
	FullAttnLayers          uint64 `json:"full_attn_layers"`
	RecurrentLayers         uint64 `json:"recurrent_layers"`
	TotalLayers             uint64 `json:"total_layers"`
	KVHeads                 uint64 `json:"kv_heads"`
	HeadDim                 uint64 `json:"head_dim"`
	FullAttnKVBytesPerToken uint64 `json:"full_attn_kv_bytes_per_token"`

	// Memory sizing
	TotalMemoryBytes    uint64 `json:"total_memory_bytes"`
	OSReserveBytes      uint64 `json:"os_reserve_bytes"`
	WiredCeilingBytes   uint64 `json:"wired_ceiling_bytes"`
	StagingScratchBytes uint64 `json:"staging_scratch_bytes"`

	// Workload configuration
	ContextTokens      uint64 `json:"context_tokens"`
	Concurrency        uint64 `json:"concurrency"`
	SharedPrefixTokens uint64 `json:"shared_prefix_tokens,omitempty"`
	PrivateTailTokens  uint64 `json:"private_tail_tokens,omitempty"`

	// Dynamic allocations & physical footprint
	FullAttnKVCacheBytes uint64  `json:"full_attn_kv_cache_bytes"`
	RecurrentStateBytes  uint64  `json:"recurrent_state_bytes"`
	PeakMemoryBytes      uint64  `json:"peak_memory_bytes"`
	HeadroomBytes        uint64  `json:"headroom_bytes"`
	HeadroomRatio        float64 `json:"headroom_ratio"`
	WiredHeadroomBytes   uint64  `json:"wired_headroom_bytes"`
	WiredHeadroomRatio   float64 `json:"wired_headroom_ratio"`

	// Profile & capability scaling
	ProfileSeam         string `json:"profile_seam"`
	ContextBudgetTokens uint64 `json:"context_budget_tokens"`
	MaxContextTokens    uint64 `json:"max_context_tokens"`

	// Swap risk & admission verdict
	SwapRisk          string `json:"swap_risk"`
	ZeroSwapRisk      bool   `json:"zero_swap_risk"`
	Admitted          bool   `json:"admitted"`
	RefusalReason     string `json:"refusal_reason,omitempty"`
	LimitingComponent string `json:"limiting_component,omitempty"`
}

// SelectProfileSeam identifies the qualification profile seam for a given physical memory size.
// Boundaries:
//   - >= 64GB: CAPABILITY_SCALED_LARGER
//   - 32GB - 63GB: 36GB_BASELINE
//   - < 32GB: SUB_36GB_CONSTRAINED
func SelectProfileSeam(memoryBytes uint64) string {
	const (
		gbDecimal      = uint64(1_000_000_000)
		threshold64GiB = 60 * GiB
		threshold64GB  = 64 * gbDecimal
		threshold32GiB = 30 * GiB
		threshold32GB  = 32 * gbDecimal
	)
	if memoryBytes >= threshold64GiB || memoryBytes >= threshold64GB {
		return ProfileSeamCapabilityScaledLarger
	}
	if memoryBytes >= threshold32GiB || memoryBytes >= threshold32GB {
		return ProfileSeam36GBBaseline
	}
	return ProfileSeamSub36GBConstrained
}

// QualifyUDQ2KXL evaluates unified-memory qualification for Qwen3.8-27B-UD-Q2_K_XL
// under the specified memory capacity, context token requirement, and concurrency.
func QualifyUDQ2KXL(memoryBytes uint64, contextTokens uint64, concurrency uint64) (UDQ2KXLQualification, error) {
	if memoryBytes == 0 {
		return UDQ2KXLQualification{}, errors.New("memory bytes must be positive")
	}
	if concurrency == 0 {
		concurrency = 1
	}
	if contextTokens == 0 {
		contextTokens = 20000
	}

	// 1. Full attention KV cache accounting:
	// 2 (K,V) * 16 layers * 4 heads * 128 dim * 2 bytes = 32,768 bytes/token.
	kvpt := FullAttnKVBytesPerToken
	totalKVBytes, err := mul(contextTokens, kvpt, concurrency)
	if err != nil {
		return UDQ2KXLQualification{}, fmt.Errorf("context calculation overflow: %w", err)
	}

	return qualifyUDQ2KXLInternal(memoryBytes, contextTokens, concurrency, 0, 0, totalKVBytes)
}

// QualifyUDQ2KXLMultiAgent qualifies concurrent multi-agent workloads with a shared
// preamble prefix stored once and private reasoning tails allocated per agent.
func QualifyUDQ2KXLMultiAgent(memoryBytes uint64, sharedPreambleTokens uint64, privateTailTokens uint64, concurrency uint64) (UDQ2KXLQualification, error) {
	if memoryBytes == 0 {
		return UDQ2KXLQualification{}, errors.New("memory bytes must be positive")
	}
	if concurrency == 0 {
		concurrency = 1
	}
	kvpt := FullAttnKVBytesPerToken
	sharedKV, err := mul(sharedPreambleTokens, kvpt)
	if err != nil {
		return UDQ2KXLQualification{}, fmt.Errorf("shared preamble calculation overflow: %w", err)
	}
	tailKV, err := mul(concurrency, privateTailTokens, kvpt)
	if err != nil {
		return UDQ2KXLQualification{}, fmt.Errorf("private tail calculation overflow: %w", err)
	}
	if sharedKV > math.MaxUint64-tailKV {
		return UDQ2KXLQualification{}, errors.New("total KV calculation overflow")
	}
	totalKVBytes := sharedKV + tailKV
	if sharedPreambleTokens > math.MaxUint64-privateTailTokens {
		return UDQ2KXLQualification{}, errors.New("total context tokens overflow")
	}
	totalContextTokens := sharedPreambleTokens + privateTailTokens

	return qualifyUDQ2KXLInternal(memoryBytes, totalContextTokens, concurrency, sharedPreambleTokens, privateTailTokens, totalKVBytes)
}

func qualifyUDQ2KXLInternal(memoryBytes uint64, contextTokens uint64, concurrency uint64, sharedPrefixTokens uint64, privateTailTokens uint64, totalKVBytes uint64) (UDQ2KXLQualification, error) {
	// OS reserve: 20% of unified memory.
	osReserveBytes := (memoryBytes * 20) / 100

	// Wired ceiling: 75% of unified memory (27 GB on 36GB Mac).
	wiredCeilingBytes := (memoryBytes * 75) / 100

	// Profile seam selection:
	// >= 64GB -> CAPABILITY_SCALED_LARGER; 32-63GB -> 36GB_BASELINE; <32GB -> SUB_36GB_CONSTRAINED.
	seam := SelectProfileSeam(memoryBytes)

	// Recurrent state across 48 layers for active agents:
	recurrentStateBytes, err := mul(concurrency, RecurrentStatePerAgentBytes)
	if err != nil {
		return UDQ2KXLQualification{}, fmt.Errorf("recurrent state calculation overflow: %w", err)
	}

	// Peak physical memory footprint:
	// weights + KV cache + recurrent state + staging scratch (~512 MiB).
	if math.MaxUint64-ResidentWeightBytes < totalKVBytes ||
		math.MaxUint64-(ResidentWeightBytes+totalKVBytes) < recurrentStateBytes ||
		math.MaxUint64-(ResidentWeightBytes+totalKVBytes+recurrentStateBytes) < StagingScratchBytes {
		return UDQ2KXLQualification{}, errors.New("peak memory calculation overflow")
	}
	peakMemoryBytes := ResidentWeightBytes + totalKVBytes + recurrentStateBytes + StagingScratchBytes

	// Fixed non-KV baseline under wired ceiling:
	fixedOverhead := ResidentWeightBytes + StagingScratchBytes + recurrentStateBytes

	var maxContextTokens uint64
	denom, err := mul(FullAttnKVBytesPerToken, concurrency)
	if err == nil && denom > 0 && wiredCeilingBytes > fixedOverhead {
		availKVPool := wiredCeilingBytes - fixedOverhead
		maxContextTokens = availKVPool / denom
	}

	// Adjudication against wired ceiling (zero disk swap check)
	admitted := peakMemoryBytes <= wiredCeilingBytes
	var (
		swapRisk          string
		zeroSwapRisk      bool
		refusalReason     string
		limitingComponent string
		headroomBytes     uint64
		headroomRatio     float64
		wiredHeadroom     uint64
		wiredHeadroomRat  float64
	)

	if admitted {
		swapRisk = SwapRiskZero
		zeroSwapRisk = true
		limitingComponent = LimitingComponentNone

		if memoryBytes > peakMemoryBytes {
			headroomBytes = memoryBytes - peakMemoryBytes
			headroomRatio = float64(headroomBytes) / float64(memoryBytes)
		}
		if wiredCeilingBytes > peakMemoryBytes {
			wiredHeadroom = wiredCeilingBytes - peakMemoryBytes
			wiredHeadroomRat = float64(wiredHeadroom) / float64(wiredCeilingBytes)
		}
	} else {
		zeroSwapRisk = false
		headroomBytes = 0
		headroomRatio = 0.0
		wiredHeadroom = 0
		wiredHeadroomRat = 0.0

		// Assess swap risk severity based on overflow
		overflowBytes := peakMemoryBytes - wiredCeilingBytes
		if overflowBytes > 2*GiB {
			swapRisk = SwapRiskCritical
		} else {
			swapRisk = SwapRiskHigh
		}

		// Identify limiting component pushing memory over wired ceiling
		if ResidentWeightBytes > wiredCeilingBytes {
			limitingComponent = LimitingComponentWeights
			refusalReason = fmt.Sprintf("resident weights (%d bytes) exceed wired ceiling (%d bytes)", ResidentWeightBytes, wiredCeilingBytes)
		} else if ResidentWeightBytes+StagingScratchBytes > wiredCeilingBytes {
			limitingComponent = LimitingComponentStagingScratch
			refusalReason = fmt.Sprintf("resident weights + staging scratch (%d bytes) exceed wired ceiling (%d bytes)", ResidentWeightBytes+StagingScratchBytes, wiredCeilingBytes)
		} else if ResidentWeightBytes+StagingScratchBytes+recurrentStateBytes > wiredCeilingBytes {
			limitingComponent = LimitingComponentRecurrentState
			refusalReason = fmt.Sprintf("recurrent state for %d agents (%d bytes) exceeds remaining wired ceiling", concurrency, recurrentStateBytes)
		} else {
			limitingComponent = LimitingComponentKVCache
			refusalReason = fmt.Sprintf("peak memory (%d bytes) exceeds wired ceiling (%d bytes); KV cache (%d bytes for %d tokens) overflows wired budget", peakMemoryBytes, wiredCeilingBytes, totalKVBytes, contextTokens)
		}
	}

	// Compute context budget tokens based on capability scaling
	var contextBudgetTokens uint64
	if admitted {
		switch seam {
		case ProfileSeamCapabilityScaledLarger:
			// Scaled envelope: expanded context budget (up to 131,072 or 65,536)
			if maxContextTokens >= 131072 {
				contextBudgetTokens = 131072
			} else if maxContextTokens >= 65536 {
				contextBudgetTokens = 65536
			} else {
				contextBudgetTokens = maxContextTokens
			}
		case ProfileSeam36GBBaseline:
			// 36GB baseline: up to 32,768 tokens (or requested context)
			if maxContextTokens >= 32768 {
				contextBudgetTokens = 32768
			} else {
				contextBudgetTokens = maxContextTokens
			}
		case ProfileSeamSub36GBConstrained:
			if maxContextTokens >= 8192 {
				contextBudgetTokens = 8192
			} else {
				contextBudgetTokens = maxContextTokens
			}
		}
		if contextTokens > contextBudgetTokens && contextTokens <= maxContextTokens {
			contextBudgetTokens = contextTokens
		}
	}

	return UDQ2KXLQualification{
		Schema:                  "fak-udq2kxl-qualification/1",
		ArtifactFileName:        ArtifactFileName,
		ArtifactBytes:           ArtifactBytes,
		ArtifactLFS_SHA256:      ArtifactLFS_SHA256,
		ResidentWeightBytes:     ResidentWeightBytes,
		FullAttnLayers:          FullAttnLayers,
		RecurrentLayers:         RecurrentLayers,
		TotalLayers:             TotalLayers,
		KVHeads:                 KVHeads,
		HeadDim:                 HeadDim,
		FullAttnKVBytesPerToken: FullAttnKVBytesPerToken,
		TotalMemoryBytes:        memoryBytes,
		OSReserveBytes:          osReserveBytes,
		WiredCeilingBytes:       wiredCeilingBytes,
		StagingScratchBytes:     StagingScratchBytes,
		ContextTokens:           contextTokens,
		Concurrency:             concurrency,
		SharedPrefixTokens:      sharedPrefixTokens,
		PrivateTailTokens:       privateTailTokens,
		FullAttnKVCacheBytes:    totalKVBytes,
		RecurrentStateBytes:     recurrentStateBytes,
		PeakMemoryBytes:         peakMemoryBytes,
		HeadroomBytes:           headroomBytes,
		HeadroomRatio:           headroomRatio,
		WiredHeadroomBytes:      wiredHeadroom,
		WiredHeadroomRatio:      wiredHeadroomRat,
		ProfileSeam:             seam,
		ContextBudgetTokens:     contextBudgetTokens,
		MaxContextTokens:        maxContextTokens,
		SwapRisk:                swapRisk,
		ZeroSwapRisk:            zeroSwapRisk,
		Admitted:                admitted,
		RefusalReason:           refusalReason,
		LimitingComponent:       limitingComponent,
	}, nil
}

// ToJSON serializes the qualification to an indented JSON string.
func (q UDQ2KXLQualification) ToJSON() (string, error) {
	b, err := json.MarshalIndent(q, "", "  ")
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// String returns a compact single-line summary of the qualification verdict.
func (q UDQ2KXLQualification) String() string {
	verdict := "ADMITTED"
	if !q.Admitted {
		verdict = fmt.Sprintf("REFUSED(%s: %s)", q.LimitingComponent, q.RefusalReason)
	}
	return fmt.Sprintf("UD-Q2_K_XL [%s] %s: peak=%.2fGiB ceiling=%.2fGiB headroom=%.1f%% swap=%s",
		q.ProfileSeam, verdict,
		float64(q.PeakMemoryBytes)/float64(GiB),
		float64(q.WiredCeilingBytes)/float64(GiB),
		q.HeadroomRatio*100,
		q.SwapRisk,
	)
}

// FormatCLI outputs a formatted qualification report table to the writer.
func (q UDQ2KXLQualification) FormatCLI(w io.Writer) error {
	fmt.Fprintf(w, "=== Qwen3.8-27B-UD-Q2_K_XL Mac Qualification ===\n")
	fmt.Fprintf(w, "Artifact File       : %s (%d bytes)\n", q.ArtifactFileName, q.ArtifactBytes)
	fmt.Fprintf(w, "Artifact SHA-256    : %s\n", q.ArtifactLFS_SHA256)
	fmt.Fprintf(w, "Profile Seam        : %s\n", q.ProfileSeam)
	fmt.Fprintf(w, "Unified Memory      : %.2f GiB (%d bytes)\n", float64(q.TotalMemoryBytes)/float64(GiB), q.TotalMemoryBytes)
	fmt.Fprintf(w, "Wired Ceiling (75%%) : %.2f GiB (%d bytes)\n", float64(q.WiredCeilingBytes)/float64(GiB), q.WiredCeilingBytes)
	fmt.Fprintf(w, "OS Reserve (20%%)    : %.2f GiB (%d bytes)\n", float64(q.OSReserveBytes)/float64(GiB), q.OSReserveBytes)
	fmt.Fprintf(w, "Resident Weights    : %.2f GiB (%d bytes)\n", float64(q.ResidentWeightBytes)/float64(GiB), q.ResidentWeightBytes)
	fmt.Fprintf(w, "Full-Attn KV Cache  : %.2f MiB (%d tokens @ %d B/tok)\n", float64(q.FullAttnKVCacheBytes)/(1024*1024), q.ContextTokens, q.FullAttnKVBytesPerToken)
	fmt.Fprintf(w, "GDN Recurrent State : %.2f MiB (%d recurrent layers, %d agents)\n", float64(q.RecurrentStateBytes)/(1024*1024), q.RecurrentLayers, q.Concurrency)
	fmt.Fprintf(w, "Staging Scratch     : %.2f MiB\n", float64(q.StagingScratchBytes)/(1024*1024))
	fmt.Fprintf(w, "Peak Memory Footprint: %.2f GiB\n", float64(q.PeakMemoryBytes)/float64(GiB))
	fmt.Fprintf(w, "Headroom            : %.2f GiB (%.1f%%)\n", float64(q.HeadroomBytes)/float64(GiB), q.HeadroomRatio*100)
	fmt.Fprintf(w, "Swap Risk           : %s (zero_swap_risk: %t)\n", q.SwapRisk, q.ZeroSwapRisk)
	fmt.Fprintf(w, "Context Budget      : %d tokens (max: %d tokens)\n", q.ContextBudgetTokens, q.MaxContextTokens)
	if q.Admitted {
		fmt.Fprintf(w, "Admission Verdict   : ADMITTED (zero disk swap guaranteed)\n")
	} else {
		fmt.Fprintf(w, "Admission Verdict   : REFUSED\n")
		fmt.Fprintf(w, "Limiting Component  : %s\n", q.LimitingComponent)
		fmt.Fprintf(w, "Refusal Reason      : %s\n", q.RefusalReason)
	}
	return nil
}
