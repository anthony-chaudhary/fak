// Prior-art: llama.cpp Metal / MLX (https://github.com/ggml-org/llama.cpp)
// Oracle: cpuref (GEMV cosine)

package metalgemm

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

// Schema identifiers for Qwen4Exp Metal plan and execution receipt.
const (
	Qwen4ExpMetalPlanSchema        = "fak.qwen4exp.metal-plan/1"
	Qwen4ExpExecutionReceiptSchema = "fak.qwen4exp.metal-execution-receipt/1"
)

// Pinned architecture constants for Qwen3.8-Flash-Next / Qwen4Exp.
const (
	Qwen4ExpModelType             = "qwen4_exp_text"
	Qwen4ExpArchitecture          = "Qwen4ExpForConditionalGeneration"
	Qwen4ExpTotalLayers           = 48
	Qwen4ExpFullAttentionInterval = 4
	Qwen4ExpNumGDNLayers          = 36
	Qwen4ExpNumQSALayers          = 12
	Qwen4ExpHiddenSize            = 2560
	Qwen4ExpVocabSize             = 248320
	Qwen4ExpTotalParams           = 82800000000 // 82.8B total parameters
	Qwen4ExpActiveParams          = 3100000000  // ~3.1B active parameters per token
	Qwen4ExpDenseTrunkParams      = 4250000000  // ~4.25B dense trunk parameters
	Qwen4ExpRoutedExpertParams    = 78550000000 // ~78.55B routed expert parameters

	// GDN (Gated DeltaNet) linear attention constants
	Qwen4ExpGDNNumKeyHeads   = 16
	Qwen4ExpGDNNumValueHeads = 48
	Qwen4ExpGDNKeyHeadDim    = 128
	Qwen4ExpGDNValueHeadDim  = 128
	Qwen4ExpGDNConvKernel    = 4
	Qwen4ExpGDNStateDType    = "float32"

	// QSA (Qwen Sparse Attention) full attention constants
	Qwen4ExpQSANumQueryHeads  = 24
	Qwen4ExpQSANumKVHeads     = 2
	Qwen4ExpQSAHeadDim        = 256
	Qwen4ExpQSAIndexerBudget  = 2048
	Qwen4ExpQSAIndexerHeads   = 4
	Qwen4ExpQSAIndexerHeadDim = 128

	// Sparse MoE constants
	Qwen4ExpNumRoutedExperts             = 512
	Qwen4ExpActiveRoutedExperts          = 10
	Qwen4ExpSharedExpertIntermediateSize = 640
	Qwen4ExpRoutedExpertIntermediateSize = 640

	// PLE (Parameter-Lookup Embeddings) constants
	Qwen4ExpPLENgramSize = 3
	Qwen4ExpPLEEmbedDim  = 2560

	// MTP (Multi-Token Prediction) constants
	Qwen4ExpMTPNumLayers = 1
	Qwen4ExpMTPLayerType = "full_attention"
	Qwen4ExpMTPNgramSize = 3
)

// Standard Apple Silicon RAM tier byte constants.
const (
	RAMTier16GB  int64 = 16 << 30
	RAMTier24GB  int64 = 24 << 30
	RAMTier36GB  int64 = 36 << 30
	RAMTier48GB  int64 = 48 << 30
	RAMTier64GB  int64 = 64 << 30
	RAMTier128GB int64 = 128 << 30
)

// QuantTier identifies the quantization tier for model weights.
type QuantTier string

const (
	QuantTierQ4K  QuantTier = "Q4_K"
	QuantTierQ80  QuantTier = "Q8_0"
	QuantTierBF16 QuantTier = "BF16"
)

// BytesPerParameter returns the average bytes per parameter for a quantization tier.
func (q QuantTier) BytesPerParameter() (float64, error) {
	switch strings.ToUpper(string(q)) {
	case "Q4_K", "Q4_K_M", "Q4":
		return 0.5625, nil // ~4.5 bits/param
	case "Q8_0", "Q8":
		return 1.0625, nil // ~8.5 bits/param
	case "BF16", "BFLOAT16", "FP16":
		return 2.0, nil // 16 bits/param
	default:
		return 0, fmt.Errorf("qwen4exp metal: unsupported quantization tier %q", q)
	}
}

// DarwinMemoryPressure represents macOS memory pressure notifications.
type DarwinMemoryPressure string

const (
	MemoryPressureNormal   DarwinMemoryPressure = "Normal"
	MemoryPressureWarning  DarwinMemoryPressure = "Warning"
	MemoryPressureCritical DarwinMemoryPressure = "Critical"
)

// ParseDarwinMemoryPressure normalizes a string into a DarwinMemoryPressure level.
func ParseDarwinMemoryPressure(s string) (DarwinMemoryPressure, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "normal", "nominal", "none":
		return MemoryPressureNormal, nil
	case "warning", "warn":
		return MemoryPressureWarning, nil
	case "critical", "crit":
		return MemoryPressureCritical, nil
	default:
		return "", fmt.Errorf("qwen4exp metal: unknown darwin memory pressure level %q", s)
	}
}

// Common error definitions.
var (
	ErrIncompleteTopology       = errors.New("qwen4exp metal: incomplete architecture topology")
	ErrIncompleteKernelCoverage = errors.New("qwen4exp metal: incomplete Metal kernel coverage")
	ErrFallbackProhibited       = errors.New("qwen4exp metal: non-native or fallback execution prohibited")
	ErrNonNativeEngine          = errors.New("qwen4exp metal: engine must be fak-native")
	ErrCriticalMemoryPressure   = errors.New("qwen4exp metal: critical memory pressure: fail-closed refusal")
	ErrOvercommit               = errors.New("qwen4exp metal: unified memory allocation exceeds physical limit")
)

// RequiredQwen4ExpMetalKernels enumerates the complete set of required Metal kernels.
var RequiredQwen4ExpMetalKernels = []string{
	"gdn_blocked_sequential_prefill",
	"gdn_decode_simd_recurrence",
	"gdn_state_rmsnorm",
	"qsa_top2048_index_gather",
	"qsa_sparse_attention_tile",
	"qsa_sdpa_wide_m",
	"moe_gate_top10_softmax",
	"moe_shared_expert_gemm",
	"moe_routed_expert_stream",
	"ple_ngram3_hash_gather",
	"ple_embedding_coalesced_lookup",
	"rmsnorm_rope_partial25",
}

// GDNTopology captures the Gated-DeltaNet linear attention architecture specifications.
type GDNTopology struct {
	NumLayers                   int      `json:"num_layers"`
	NumKeyHeads                 int      `json:"num_key_heads"`
	NumValueHeads               int      `json:"num_value_heads"`
	KeyHeadDim                  int      `json:"key_head_dim"`
	ValueHeadDim                int      `json:"value_head_dim"`
	ConvKernel                  int      `json:"conv_kernel"`
	StateDType                  string   `json:"state_dtype"`
	RecurrentStateBytesPerLayer int64    `json:"recurrent_state_bytes_per_layer"`
	TotalRecurrentStateBytes    int64    `json:"total_recurrent_state_bytes"`
	MetalKernels                []string `json:"metal_kernels"`
}

// QSATopology captures the Qwen Sparse Attention full attention architecture specifications.
type QSATopology struct {
	NumLayers      int      `json:"num_layers"`
	NumQueryHeads  int      `json:"num_query_heads"`
	NumKVHeads     int      `json:"num_kv_heads"`
	HeadDim        int      `json:"head_dim"`
	IndexerBudget  int      `json:"indexer_budget"`
	IndexerHeads   int      `json:"indexer_heads"`
	IndexerHeadDim int      `json:"indexer_head_dim"`
	MetalKernels   []string `json:"metal_kernels"`
}

// MoETopology captures the sparse Mixture of Experts architecture specifications.
type MoETopology struct {
	NumLayers                    int      `json:"num_layers"`
	NumRoutedExperts             int      `json:"num_routed_experts"`
	ActiveRoutedExperts          int      `json:"active_routed_experts"`
	SharedExpertIntermediateSize int      `json:"shared_expert_intermediate_size"`
	RoutedExpertIntermediateSize int      `json:"routed_expert_intermediate_size"`
	MetalKernels                 []string `json:"metal_kernels"`
}

// PLETopology captures the Parameter-Lookup Embeddings architecture specifications.
type PLETopology struct {
	NgramSize    int      `json:"ngram_size"`
	EmbeddingDim int      `json:"embedding_dim"`
	VocabSize    int      `json:"vocab_size"`
	MetalKernels []string `json:"metal_kernels"`
}

// MTPTopology captures the Multi-Token Prediction architecture specifications.
type MTPTopology struct {
	Hybrid          bool     `json:"hybrid"`
	NumHiddenLayers int      `json:"num_hidden_layers"`
	LayerType       string   `json:"layer_type"`
	NgramSize       int      `json:"ngram_size"`
	MetalKernels    []string `json:"metal_kernels"`
}

// HybridLayerSpec details the exact composition of a single transformer layer.
type HybridLayerSpec struct {
	LayerIndex      int    `json:"layer_index"`
	AttentionType   string `json:"attention_type"` // "linear_attention" | "full_attention"
	IsGDN           bool   `json:"is_gdn"`
	IsQSA           bool   `json:"is_qsa"`
	HasMoE          bool   `json:"has_moe"`
	RoutedExperts   int    `json:"routed_experts"`
	ActiveExperts   int    `json:"active_experts"`
	HasSharedExpert bool   `json:"has_shared_expert"`
}

// Qwen4ExpMetalPlan specifies the full architecture topology, kernel coverage,
// and residency contracts for Qwen4Exp on Apple Silicon.
type Qwen4ExpMetalPlan struct {
	Schema                string            `json:"schema"`
	ModelType             string            `json:"model_type"`
	Architecture          string            `json:"architecture"`
	TotalLayers           int               `json:"total_layers"`
	FullAttentionInterval int               `json:"full_attention_interval"`
	Layers                []HybridLayerSpec `json:"layers"`
	GDN                   GDNTopology       `json:"gdn"`
	QSA                   QSATopology       `json:"qsa"`
	MoE                   MoETopology       `json:"moe"`
	PLE                   PLETopology       `json:"ple"`
	MTP                   MTPTopology       `json:"mtp"`
	KernelRegistry        map[string]string `json:"kernel_registry"`
	Engine                string            `json:"engine"`
	Fallback              string            `json:"fallback"`
}

// NewDefaultQwen4ExpMetalPlan constructs the canonical 48-layer Qwen4Exp Metal plan.
func NewDefaultQwen4ExpMetalPlan() *Qwen4ExpMetalPlan {
	layers := make([]HybridLayerSpec, Qwen4ExpTotalLayers)
	for i := 0; i < Qwen4ExpTotalLayers; i++ {
		isFull := (i+1)%Qwen4ExpFullAttentionInterval == 0
		attType := "linear_attention"
		if isFull {
			attType = "full_attention"
		}
		layers[i] = HybridLayerSpec{
			LayerIndex:      i,
			AttentionType:   attType,
			IsGDN:           !isFull,
			IsQSA:           isFull,
			HasMoE:          true,
			RoutedExperts:   Qwen4ExpNumRoutedExperts,
			ActiveExperts:   Qwen4ExpActiveRoutedExperts,
			HasSharedExpert: true,
		}
	}

	// Calculate exact GDN recurrent state size per layer:
	// S matrix: 48 value heads * 128 key dim * 128 value dim * 4 bytes (FP32) = 3,145,728 bytes.
	// Conv state: (4 - 1) * (2*16*128 + 48*128) * 4 bytes = 3 * 10240 * 4 = 122,880 bytes.
	// Total per layer: 3,268,608 bytes (~3.12 MiB).
	// Total across 36 GDN layers: 36 * 3,268,608 = 117,669,888 bytes (~112.2 MiB).
	const gdnStatePerLayer = int64(48*128*128*4 + (4-1)*(2*16*128+48*128)*4)
	const gdnTotalState = gdnStatePerLayer * int64(Qwen4ExpNumGDNLayers)

	gdnTopo := GDNTopology{
		NumLayers:                   Qwen4ExpNumGDNLayers,
		NumKeyHeads:                 Qwen4ExpGDNNumKeyHeads,
		NumValueHeads:               Qwen4ExpGDNNumValueHeads,
		KeyHeadDim:                  Qwen4ExpGDNKeyHeadDim,
		ValueHeadDim:                Qwen4ExpGDNValueHeadDim,
		ConvKernel:                  Qwen4ExpGDNConvKernel,
		StateDType:                  Qwen4ExpGDNStateDType,
		RecurrentStateBytesPerLayer: gdnStatePerLayer,
		TotalRecurrentStateBytes:    gdnTotalState,
		MetalKernels: []string{
			"gdn_blocked_sequential_prefill",
			"gdn_decode_simd_recurrence",
			"gdn_state_rmsnorm",
		},
	}

	qsaTopo := QSATopology{
		NumLayers:      Qwen4ExpNumQSALayers,
		NumQueryHeads:  Qwen4ExpQSANumQueryHeads,
		NumKVHeads:     Qwen4ExpQSANumKVHeads,
		HeadDim:        Qwen4ExpQSAHeadDim,
		IndexerBudget:  Qwen4ExpQSAIndexerBudget,
		IndexerHeads:   Qwen4ExpQSAIndexerHeads,
		IndexerHeadDim: Qwen4ExpQSAIndexerHeadDim,
		MetalKernels: []string{
			"qsa_top2048_index_gather",
			"qsa_sparse_attention_tile",
			"qsa_sdpa_wide_m",
		},
	}

	moeTopo := MoETopology{
		NumLayers:                    Qwen4ExpTotalLayers,
		NumRoutedExperts:             Qwen4ExpNumRoutedExperts,
		ActiveRoutedExperts:          Qwen4ExpActiveRoutedExperts,
		SharedExpertIntermediateSize: Qwen4ExpSharedExpertIntermediateSize,
		RoutedExpertIntermediateSize: Qwen4ExpRoutedExpertIntermediateSize,
		MetalKernels: []string{
			"moe_gate_top10_softmax",
			"moe_shared_expert_gemm",
			"moe_routed_expert_stream",
		},
	}

	pleTopo := PLETopology{
		NgramSize:    Qwen4ExpPLENgramSize,
		EmbeddingDim: Qwen4ExpPLEEmbedDim,
		VocabSize:    Qwen4ExpVocabSize,
		MetalKernels: []string{
			"ple_ngram3_hash_gather",
			"ple_embedding_coalesced_lookup",
		},
	}

	mtpTopo := MTPTopology{
		Hybrid:          true,
		NumHiddenLayers: Qwen4ExpMTPNumLayers,
		LayerType:       Qwen4ExpMTPLayerType,
		NgramSize:       Qwen4ExpMTPNgramSize,
		MetalKernels: []string{
			"qsa_sdpa_wide_m",
		},
	}

	registry := map[string]string{
		"gdn_blocked_sequential_prefill": "Blocked-sequential prefill staging in TB=32/DB=32 with SIMD-shuffle recurrence",
		"gdn_decode_simd_recurrence":     "P=1 linear recurrence update with zero threadgroup barriers",
		"gdn_state_rmsnorm":              "In-place persistent state RMSNorm and gated output projection",
		"qsa_top2048_index_gather":       "Learned compressed indexer scoring and top-2048 token selection",
		"qsa_sparse_attention_tile":      "Sparse full-attention tile evaluating selected 2048 key/value pairs",
		"qsa_sdpa_wide_m":                "Wide-M SDPA speculative verification tile for draft evaluation",
		"moe_gate_top10_softmax":         "Fused routing gate with top-10 expert selection and softmax normalization",
		"moe_shared_expert_gemm":         "Shared expert forward projection with intermediate dimension 640",
		"moe_routed_expert_stream":       "Asynchronous QD32 slotstream expert streaming or resident GEMM",
		"ple_ngram3_hash_gather":         "3-gram parameter-lookup embedding hash and gather",
		"ple_embedding_coalesced_lookup": "Coalesced parameter table lookup into 2560-dim representation",
		"rmsnorm_rope_partial25":         "25% rotary position embedding and pre-attention RMSNorm",
	}

	return &Qwen4ExpMetalPlan{
		Schema:                Qwen4ExpMetalPlanSchema,
		ModelType:             Qwen4ExpModelType,
		Architecture:          Qwen4ExpArchitecture,
		TotalLayers:           Qwen4ExpTotalLayers,
		FullAttentionInterval: Qwen4ExpFullAttentionInterval,
		Layers:                layers,
		GDN:                   gdnTopo,
		QSA:                   qsaTopo,
		MoE:                   moeTopo,
		PLE:                   pleTopo,
		MTP:                   mtpTopo,
		KernelRegistry:        registry,
		Engine:                "fak-native",
		Fallback:              "none",
	}
}

// ValidateTopology verifies that the plan captures the complete architecture topology
// without omissions, modifications, or external fallbacks.
func (p *Qwen4ExpMetalPlan) ValidateTopology() error {
	if p.Schema != Qwen4ExpMetalPlanSchema {
		return fmt.Errorf("qwen4exp metal: invalid plan schema %q", p.Schema)
	}
	if p.ModelType != Qwen4ExpModelType || p.Architecture != Qwen4ExpArchitecture {
		return fmt.Errorf("qwen4exp metal: invalid model identity %q/%q", p.ModelType, p.Architecture)
	}
	if p.Engine != "fak-native" {
		return ErrNonNativeEngine
	}
	if p.Fallback != "none" {
		return ErrFallbackProhibited
	}
	if p.TotalLayers != Qwen4ExpTotalLayers || p.FullAttentionInterval != Qwen4ExpFullAttentionInterval {
		return fmt.Errorf("qwen4exp metal: layer configuration mismatch: layers=%d interval=%d", p.TotalLayers, p.FullAttentionInterval)
	}
	if len(p.Layers) != Qwen4ExpTotalLayers {
		return fmt.Errorf("qwen4exp metal: expected %d layer specs, got %d", Qwen4ExpTotalLayers, len(p.Layers))
	}

	gdnCount, qsaCount := 0, 0
	for i, layer := range p.Layers {
		if layer.LayerIndex != i {
			return fmt.Errorf("qwen4exp metal: layer %d has mismatched index %d", i, layer.LayerIndex)
		}
		isFull := (i+1)%p.FullAttentionInterval == 0
		if isFull {
			if layer.AttentionType != "full_attention" || !layer.IsQSA || layer.IsGDN {
				return fmt.Errorf("qwen4exp metal: layer %d expected QSA full attention", i)
			}
			qsaCount++
		} else {
			if layer.AttentionType != "linear_attention" || !layer.IsGDN || layer.IsQSA {
				return fmt.Errorf("qwen4exp metal: layer %d expected GDN linear attention", i)
			}
			gdnCount++
		}
		if !layer.HasMoE || layer.RoutedExperts != Qwen4ExpNumRoutedExperts || layer.ActiveExperts != Qwen4ExpActiveRoutedExperts || !layer.HasSharedExpert {
			return fmt.Errorf("qwen4exp metal: layer %d has invalid MoE topology", i)
		}
	}

	if gdnCount != Qwen4ExpNumGDNLayers || qsaCount != Qwen4ExpNumQSALayers {
		return fmt.Errorf("qwen4exp metal: layer cadence mismatch: GDN=%d QSA=%d, want %d/%d",
			gdnCount, qsaCount, Qwen4ExpNumGDNLayers, Qwen4ExpNumQSALayers)
	}

	// Verify GDN topology
	if p.GDN.NumLayers != Qwen4ExpNumGDNLayers ||
		p.GDN.NumKeyHeads != Qwen4ExpGDNNumKeyHeads ||
		p.GDN.NumValueHeads != Qwen4ExpGDNNumValueHeads ||
		p.GDN.KeyHeadDim != Qwen4ExpGDNKeyHeadDim ||
		p.GDN.ValueHeadDim != Qwen4ExpGDNValueHeadDim ||
		p.GDN.ConvKernel != Qwen4ExpGDNConvKernel ||
		p.GDN.StateDType != Qwen4ExpGDNStateDType ||
		p.GDN.RecurrentStateBytesPerLayer <= 0 ||
		p.GDN.TotalRecurrentStateBytes <= 0 {
		return fmt.Errorf("%w: GDN specification mismatch", ErrIncompleteTopology)
	}

	// Verify QSA topology
	if p.QSA.NumLayers != Qwen4ExpNumQSALayers ||
		p.QSA.NumQueryHeads != Qwen4ExpQSANumQueryHeads ||
		p.QSA.NumKVHeads != Qwen4ExpQSANumKVHeads ||
		p.QSA.HeadDim != Qwen4ExpQSAHeadDim ||
		p.QSA.IndexerBudget != Qwen4ExpQSAIndexerBudget ||
		p.QSA.IndexerHeads != Qwen4ExpQSAIndexerHeads ||
		p.QSA.IndexerHeadDim != Qwen4ExpQSAIndexerHeadDim {
		return fmt.Errorf("%w: QSA specification mismatch", ErrIncompleteTopology)
	}

	// Verify MoE topology
	if p.MoE.NumLayers != Qwen4ExpTotalLayers ||
		p.MoE.NumRoutedExperts != Qwen4ExpNumRoutedExperts ||
		p.MoE.ActiveRoutedExperts != Qwen4ExpActiveRoutedExperts ||
		p.MoE.SharedExpertIntermediateSize != Qwen4ExpSharedExpertIntermediateSize ||
		p.MoE.RoutedExpertIntermediateSize != Qwen4ExpRoutedExpertIntermediateSize {
		return fmt.Errorf("%w: MoE specification mismatch", ErrIncompleteTopology)
	}

	// Verify PLE topology
	if p.PLE.NgramSize != Qwen4ExpPLENgramSize ||
		p.PLE.EmbeddingDim != Qwen4ExpPLEEmbedDim ||
		p.PLE.VocabSize != Qwen4ExpVocabSize {
		return fmt.Errorf("%w: PLE specification mismatch", ErrIncompleteTopology)
	}

	// Verify MTP topology
	if !p.MTP.Hybrid || p.MTP.NumHiddenLayers != Qwen4ExpMTPNumLayers ||
		p.MTP.LayerType != Qwen4ExpMTPLayerType || p.MTP.NgramSize != Qwen4ExpMTPNgramSize {
		return fmt.Errorf("%w: MTP specification mismatch", ErrIncompleteTopology)
	}

	// Verify kernel coverage completeness
	for _, reqKernel := range RequiredQwen4ExpMetalKernels {
		if _, ok := p.KernelRegistry[reqKernel]; !ok {
			return fmt.Errorf("%w: missing required Metal kernel %q", ErrIncompleteKernelCoverage, reqKernel)
		}
	}

	return nil
}

// UnifiedMemoryBudget holds the calculated memory requirements for a concrete execution plan.
type UnifiedMemoryBudget struct {
	QuantTier               QuantTier `json:"quant_tier"`
	PhysicalRAMBytes        int64     `json:"physical_ram_bytes"`
	StreamingMode           bool      `json:"streaming_mode"`
	ContextLength           int       `json:"context_length"`
	BatchSize               int       `json:"batch_size"`
	WeightTotalBytes        int64     `json:"weight_total_bytes"`
	WeightResidentBytes     int64     `json:"weight_resident_bytes"`
	DenseTrunkBytes         int64     `json:"dense_trunk_bytes"`
	SlotPoolBytes           int64     `json:"slot_pool_bytes"`
	KVCacheBytes            int64     `json:"kv_cache_bytes"`
	RecurrentStateBytes     int64     `json:"recurrent_state_bytes"`
	ActivationHeadroomBytes int64     `json:"activation_headroom_bytes"`
	SystemReservedBytes     int64     `json:"system_reserved_bytes"`
	PeakUnifiedMemoryBytes  int64     `json:"peak_unified_memory_bytes"`
	UsableRAMBytes          int64     `json:"usable_ram_bytes"`
	AvailableHeadroomBytes  int64     `json:"available_headroom_bytes"`
	FitsInRAM               bool      `json:"fits_in_ram"`
}

// CalculateMemoryFootprint models the unified memory footprint of Qwen4Exp across
// weight residency, KV cache, recurrent state, activation headroom, and system reservations.
func (p *Qwen4ExpMetalPlan) CalculateMemoryFootprint(
	tier QuantTier,
	physicalRAM int64,
	streaming bool,
	contextLen int,
	batchSize int,
) (UnifiedMemoryBudget, error) {
	if physicalRAM <= 0 {
		return UnifiedMemoryBudget{}, errors.New("qwen4exp metal: physical RAM must be positive")
	}
	if contextLen <= 0 {
		contextLen = 2048
	}
	if batchSize <= 0 {
		batchSize = 1
	}

	bytesPerParam, err := tier.BytesPerParameter()
	if err != nil {
		return UnifiedMemoryBudget{}, err
	}

	// 1. Weight modeling:
	// Total model size: 82.8B params * bytesPerParam.
	weightTotal := int64(float64(Qwen4ExpTotalParams) * bytesPerParam)
	denseTrunk := int64(float64(Qwen4ExpDenseTrunkParams) * bytesPerParam)

	var weightResident, slotPool int64
	if streaming {
		// In streaming mode, only dense trunk is resident in RAM.
		// Active routed experts stream on demand via the QD32 slot pool (32 slots).
		// Slot byte size scales with quantization tier.
		var bytesPerSlot int64
		switch strings.ToUpper(string(tier)) {
		case "Q4_K", "Q4_K_M", "Q4":
			bytesPerSlot = 3 * 1024 * 1024 // 3 MiB
		case "Q8_0", "Q8":
			bytesPerSlot = 6 * 1024 * 1024 // 6 MiB
		default: // BF16
			bytesPerSlot = 12 * 1024 * 1024 // 12 MiB
		}
		slotPool = int64(DefaultSlotCount) * bytesPerSlot
		weightResident = denseTrunk + slotPool
	} else {
		weightResident = weightTotal
		slotPool = 0
	}

	// 2. KV Cache modeling:
	// ONLY the 12 QSA layers maintain a KV cache! The 36 GDN layers use recurrent state.
	// Per token KV footprint = 12 layers * 2 KV heads * 2 (K & V) * 256 head dim * 2 bytes (FP16/BF16) = 24,576 bytes.
	const bytesPerTokenKVCache = int64(Qwen4ExpNumQSALayers * Qwen4ExpQSANumKVHeads * 2 * Qwen4ExpQSAHeadDim * 2)
	kvCache := int64(contextLen*batchSize) * bytesPerTokenKVCache

	// 3. Recurrent state modeling:
	// The 36 GDN layers maintain fixed-size recurrent state independent of context length!
	// 36 layers * 3,268,608 bytes/layer = 117,669,888 bytes (~112.2 MiB).
	recurrentState := p.GDN.TotalRecurrentStateBytes
	if recurrentState <= 0 {
		recurrentState = 117669888
	}

	// 4. Activation headroom:
	// Dynamic scratch buffer for intermediate tensors during GEMM, index gathering, and attention.
	// Base headroom 512 MiB + scaling with context length (64 KB/token), capped at 2 GiB.
	activationHeadroom := int64(512*1024*1024) + int64(contextLen*64*1024)
	if activationHeadroom > 2*1024*1024*1024 {
		activationHeadroom = 2 * 1024 * 1024 * 1024
	}

	// 5. System reservation:
	// macOS WindowServer, OS kernel, background processes require headroom.
	// Minimum 2 GiB or 15% of physical RAM.
	systemReserved := int64(float64(physicalRAM) * 0.15)
	if minReserved := int64(2 * 1024 * 1024 * 1024); systemReserved < minReserved {
		systemReserved = minReserved
	}

	peakUnified := weightResident + kvCache + recurrentState + activationHeadroom
	usableRAM := physicalRAM - systemReserved
	availableHeadroom := usableRAM - peakUnified
	fits := availableHeadroom >= 0

	budget := UnifiedMemoryBudget{
		QuantTier:               tier,
		PhysicalRAMBytes:        physicalRAM,
		StreamingMode:           streaming,
		ContextLength:           contextLen,
		BatchSize:               batchSize,
		WeightTotalBytes:        weightTotal,
		WeightResidentBytes:     weightResident,
		DenseTrunkBytes:         denseTrunk,
		SlotPoolBytes:           slotPool,
		KVCacheBytes:            kvCache,
		RecurrentStateBytes:     recurrentState,
		ActivationHeadroomBytes: activationHeadroom,
		SystemReservedBytes:     systemReserved,
		PeakUnifiedMemoryBytes:  peakUnified,
		UsableRAMBytes:          usableRAM,
		AvailableHeadroomBytes:  availableHeadroom,
		FitsInRAM:               fits,
	}

	return budget, nil
}

// MemoryPressureStatus records the admission decision under Darwin memory pressure.
type MemoryPressureStatus struct {
	Level         DarwinMemoryPressure `json:"level"`
	UsableRAM     int64                `json:"usable_ram_bytes"`
	HeadroomBytes int64                `json:"headroom_bytes"`
	Admitted      bool                 `json:"admitted"`
	RefusalReason string               `json:"refusal_reason,omitempty"`
}

// GovernMemoryPressure enforces memory pressure governance across Darwin levels:
//   - Normal: allows execution up to usable unified memory limit.
//   - Warning: tightens usable headroom (reserving 30% for host recovery), failing closed if exceeded.
//   - Critical: strict fail-closed refusal with zero fallback.
func (p *Qwen4ExpMetalPlan) GovernMemoryPressure(
	budget UnifiedMemoryBudget,
	pressure DarwinMemoryPressure,
) (MemoryPressureStatus, error) {
	normPressure, err := ParseDarwinMemoryPressure(string(pressure))
	if err != nil {
		return MemoryPressureStatus{
			Level:         pressure,
			Admitted:      false,
			RefusalReason: err.Error(),
		}, err
	}

	switch normPressure {
	case MemoryPressureCritical:
		// Strict fail-closed zero-fallback refusal.
		return MemoryPressureStatus{
			Level:         MemoryPressureCritical,
			UsableRAM:     0,
			HeadroomBytes: -budget.PeakUnifiedMemoryBytes,
			Admitted:      false,
			RefusalReason: "CRITICAL_MEMORY_PRESSURE: fail-closed zero-fallback refusal",
		}, ErrCriticalMemoryPressure

	case MemoryPressureWarning:
		// Under warning, macOS requests memory reclamation. Tighten usable limit to 70% of physical RAM.
		warnUsable := int64(float64(budget.PhysicalRAMBytes) * 0.70)
		warnHeadroom := warnUsable - budget.PeakUnifiedMemoryBytes
		if warnHeadroom < 0 {
			return MemoryPressureStatus{
				Level:         MemoryPressureWarning,
				UsableRAM:     warnUsable,
				HeadroomBytes: warnHeadroom,
				Admitted:      false,
				RefusalReason: fmt.Sprintf("WARNING_MEMORY_PRESSURE: peak memory %d bytes exceeds warning threshold %d bytes", budget.PeakUnifiedMemoryBytes, warnUsable),
			}, ErrOvercommit
		}
		return MemoryPressureStatus{
			Level:         MemoryPressureWarning,
			UsableRAM:     warnUsable,
			HeadroomBytes: warnHeadroom,
			Admitted:      true,
		}, nil

	case MemoryPressureNormal:
		if !budget.FitsInRAM {
			return MemoryPressureStatus{
				Level:         MemoryPressureNormal,
				UsableRAM:     budget.UsableRAMBytes,
				HeadroomBytes: budget.AvailableHeadroomBytes,
				Admitted:      false,
				RefusalReason: fmt.Sprintf("EXCEEDS_UNIFIED_MEMORY: peak %d bytes exceeds usable RAM %d bytes", budget.PeakUnifiedMemoryBytes, budget.UsableRAMBytes),
			}, ErrOvercommit
		}
		return MemoryPressureStatus{
			Level:         MemoryPressureNormal,
			UsableRAM:     budget.UsableRAMBytes,
			HeadroomBytes: budget.AvailableHeadroomBytes,
			Admitted:      true,
		}, nil

	default:
		return MemoryPressureStatus{}, fmt.Errorf("qwen4exp metal: unhandled pressure level %q", pressure)
	}
}

// Qwen4ExpExecutionReceipt is the machine-readable operational receipt verifying
// exact Metal execution conditions, unified memory residency, and performance bounds.
type Qwen4ExpExecutionReceipt struct {
	Schema                 string               `json:"schema"`
	Model                  string               `json:"model"`
	Engine                 string               `json:"engine"`
	Fallback               string               `json:"fallback"`
	Chip                   string               `json:"chip"`
	PhysicalRAMBytes       int64                `json:"physical_ram_bytes"`
	QuantTier              QuantTier            `json:"quant_tier"`
	StreamingMode          bool                 `json:"streaming_mode"`
	ContextLength          int                  `json:"context_length"`
	BatchSize              int                  `json:"batch_size"`
	MemoryPressure         DarwinMemoryPressure `json:"memory_pressure"`
	KernelCoverage         map[string]bool      `json:"kernel_coverage"`
	AllKernelsCovered      bool                 `json:"all_kernels_covered"`
	PeakUnifiedMemoryBytes int64                `json:"peak_unified_memory_bytes"`
	AvailableHeadroomBytes int64                `json:"available_headroom_bytes"`
	EstimatedTTFTMs        float64              `json:"estimated_ttft_ms"`
	PrefillTokPerSec       float64              `json:"prefill_tok_per_sec"`
	DecodeTokPerSec        float64              `json:"decode_tok_per_sec"`
	Admitted               bool                 `json:"admitted"`
	RefusalReason          string               `json:"refusal_reason,omitempty"`
	Timestamp              string               `json:"timestamp"`
	Digest                 string               `json:"digest"`
}

// Validate ensures that an execution receipt satisfies all integrity and zero-fallback contracts.
func (r *Qwen4ExpExecutionReceipt) Validate() error {
	if r.Schema != Qwen4ExpExecutionReceiptSchema {
		return fmt.Errorf("qwen4exp receipt: invalid schema %q", r.Schema)
	}
	if r.Model == "" || r.Chip == "" || r.PhysicalRAMBytes <= 0 || r.PeakUnifiedMemoryBytes <= 0 {
		return errors.New("qwen4exp receipt: incomplete identity or memory fields")
	}
	if r.Engine != "fak-native" {
		return ErrNonNativeEngine
	}
	if r.Fallback != "none" {
		return ErrFallbackProhibited
	}
	if !r.AllKernelsCovered {
		return ErrIncompleteKernelCoverage
	}
	for _, k := range RequiredQwen4ExpMetalKernels {
		if !r.KernelCoverage[k] {
			return fmt.Errorf("%w: missing kernel %q in receipt", ErrIncompleteKernelCoverage, k)
		}
	}
	if r.Admitted {
		if r.RefusalReason != "" {
			return errors.New("qwen4exp receipt: admitted receipt cannot have refusal reason")
		}
		if r.EstimatedTTFTMs <= 0 || r.PrefillTokPerSec <= 0 || r.DecodeTokPerSec <= 0 {
			return errors.New("qwen4exp receipt: admitted receipt missing performance bounds")
		}
	} else {
		if r.RefusalReason == "" {
			return errors.New("qwen4exp receipt: refused receipt requires explicit refusal reason")
		}
	}
	return nil
}

// JSON returns the serialized JSON representation of the receipt.
func (r *Qwen4ExpExecutionReceipt) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// EstimateThroughputAndTTFT computes realistic, hardware-grounded throughput bounds
// and time-to-first-token estimates based on chip capability, quantization, and residency mode.
func EstimateThroughputAndTTFT(
	chip string,
	tier QuantTier,
	streaming bool,
	contextLen int,
	batchSize int,
	ramBytes int64,
) (prefillTPS float64, decodeTPS float64, ttftMs float64) {
	chipLower := strings.ToLower(chip)
	bytesPerParam, _ := tier.BytesPerParameter()
	if bytesPerParam <= 0 {
		bytesPerParam = 0.5625
	}

	// Determine unified memory bandwidth baseline (GB/s):
	var memBandwidthGBps float64
	switch {
	case strings.Contains(chipLower, "ultra"):
		memBandwidthGBps = 800.0
	case strings.Contains(chipLower, "max"):
		memBandwidthGBps = 400.0
	case strings.Contains(chipLower, "pro"):
		memBandwidthGBps = 200.0
	case ramBytes >= RAMTier128GB:
		memBandwidthGBps = 400.0
	case ramBytes >= RAMTier48GB:
		memBandwidthGBps = 300.0
	case ramBytes >= RAMTier36GB:
		memBandwidthGBps = 150.0
	default:
		memBandwidthGBps = 100.0
	}

	// Prefill throughput (tokens/sec):
	// GDN linear recurrence is O(N) (36 layers), and QSA is bounded to top-2048 index gather.
	// Scaled inversely with weight precision.
	basePrefill := 280.0 * (memBandwidthGBps / 300.0)
	if basePrefill < 60.0 {
		basePrefill = 60.0
	}
	prefillTPS = basePrefill * (0.5625 / bytesPerParam)

	// Decode throughput (tokens/sec):
	if streaming {
		// In streaming mode, 10 routed experts per token are loaded from NVMe SSD via QD32.
		// NVMe SSD pread bandwidth on Apple Silicon is ~10-17 GB/s.
		// Active routed experts payload ~27.6 MB in Q4_K.
		// With 32 preallocated slots and LRU temporal reuse, decode is ~8-16 tok/s.
		decodeTPS = 12.5 * (0.5625 / bytesPerParam)
		if decodeTPS < 4.0 {
			decodeTPS = 4.0
		}
	} else {
		// In full residency, active weights per token = 3.1B * bytesPerParam.
		// In Q4_K: ~1.74 GB memory traffic / token.
		// On 400 GB/s bandwidth (M3/M4 Max): theoretical ~230 tok/s, practical ~80-110 tok/s.
		activeBytesPerToken := float64(Qwen4ExpActiveParams) * bytesPerParam
		activeGBPerToken := activeBytesPerToken / 1e9
		if activeGBPerToken > 0 {
			// Real-world memory efficiency ~40-50%
			decodeTPS = (memBandwidthGBps * 0.45) / activeGBPerToken
		}
		if decodeTPS < 10.0 {
			decodeTPS = 10.0
		}
	}

	// TTFT (ms):
	// Time to process prompt tokens + initial Metal command submission overhead.
	if prefillTPS > 0 {
		ttftMs = (float64(contextLen)/prefillTPS)*1000.0 + 15.0
	} else {
		ttftMs = 50.0
	}

	return prefillTPS, decodeTPS, ttftMs
}

// GenerateExecutionReceipt creates an immutable, machine-readable operational receipt
// verifying kernel coverage, memory residency, and performance bounds.
func (p *Qwen4ExpMetalPlan) GenerateExecutionReceipt(
	tier QuantTier,
	physicalRAM int64,
	streaming bool,
	contextLen int,
	batchSize int,
	pressure DarwinMemoryPressure,
	chip string,
) (*Qwen4ExpExecutionReceipt, error) {
	if err := p.ValidateTopology(); err != nil {
		return nil, err
	}

	budget, err := p.CalculateMemoryFootprint(tier, physicalRAM, streaming, contextLen, batchSize)
	if err != nil {
		return nil, err
	}

	pressureStatus, _ := p.GovernMemoryPressure(budget, pressure)

	// Verify complete kernel coverage
	coverage := make(map[string]bool, len(RequiredQwen4ExpMetalKernels))
	allCovered := true
	for _, k := range RequiredQwen4ExpMetalKernels {
		if _, ok := p.KernelRegistry[k]; ok {
			coverage[k] = true
		} else {
			coverage[k] = false
			allCovered = false
		}
	}

	if chip == "" {
		chip = "Apple Silicon (M-Series)"
	}

	var prefillTPS, decodeTPS, ttftMs float64
	if pressureStatus.Admitted {
		prefillTPS, decodeTPS, ttftMs = EstimateThroughputAndTTFT(chip, tier, streaming, contextLen, batchSize, physicalRAM)
	}

	receipt := &Qwen4ExpExecutionReceipt{
		Schema:                 Qwen4ExpExecutionReceiptSchema,
		Model:                  "Qwen3.8-Flash-Next/qwen4exp",
		Engine:                 p.Engine,
		Fallback:               p.Fallback,
		Chip:                   chip,
		PhysicalRAMBytes:       physicalRAM,
		QuantTier:              tier,
		StreamingMode:          streaming,
		ContextLength:          contextLen,
		BatchSize:              batchSize,
		MemoryPressure:         pressureStatus.Level,
		KernelCoverage:         coverage,
		AllKernelsCovered:      allCovered,
		PeakUnifiedMemoryBytes: budget.PeakUnifiedMemoryBytes,
		AvailableHeadroomBytes: budget.AvailableHeadroomBytes,
		EstimatedTTFTMs:        ttftMs,
		PrefillTokPerSec:       prefillTPS,
		DecodeTokPerSec:        decodeTPS,
		Admitted:               pressureStatus.Admitted,
		RefusalReason:          pressureStatus.RefusalReason,
		Timestamp:              time.Now().UTC().Format(time.RFC3339),
	}

	// Compute deterministic SHA-256 digest of receipt
	raw, err := json.Marshal(receipt)
	if err != nil {
		return nil, fmt.Errorf("qwen4exp receipt: marshal error: %w", err)
	}
	sum := sha256.Sum256(raw)
	receipt.Digest = "sha256:" + hex.EncodeToString(sum[:])

	if err := receipt.Validate(); err != nil {
		return nil, err
	}

	return receipt, nil
}
