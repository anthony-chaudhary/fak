package servereceipt

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Schema constants: the facts document is the input contract, the receipt is
// the plan output, the validate doc is the re-verification output, and the
// profiles doc is the built-in catalog listing.
const (
	FactsSchema    = "fak.serve-receipt.facts/1"
	ReceiptSchema  = "fak.serve-receipt.receipt/1"
	ValidateSchema = "fak.serve-receipt.validate/1"
	ProfilesSchema = "fak.serve-receipt.profiles/1"
)

// UnitsNote is the single units convention, rendered into every receipt so no
// number is ever read under the wrong unit. GiB = 2^30 bytes; GB = 10^9 bytes.
const UnitsNote = "VRAM per GPU read as GiB (2^30 B) — vendor \"GB\" listings treated as GiB (pkg/hwspecs convention); weights/state/KV bytes exact; activation reserve in decimal GB (10^9 B)"

// GibBytes is the GiB (2^30 bytes); GbBytes the decimal GB (10^9 bytes).
const (
	GibBytes int64 = 1 << 30
	GbBytes  int64 = 1_000_000_000
)

// Warning strings are fixed literals so a recomputed warning list is
// byte-comparable during validate.
const (
	WarnFP4NotNative = "NVFP4 checkpoints require Blackwell SM≥10.0"
	WarnPCIeWideTP   = "wide TP all-reduce over PCIe; PP-heavy layouts preferred (measured: TP4 6–7 tok/s vs TP2+PP2 46–49 on 397B MoE, flashinfer#2577)"
	WarnDCPSlow      = "DCP8 decode measured 7–25% slower than TP8 (recipes.vllm.ai K3.1-Pro)"
)

// AnchorCalibration is the S8 production-runtime anchor the offload model is
// calibrated against (issue #14438). Rendered verbatim into every --curve
// block so the ESTIMATE rows always cite their measured anchor.
const AnchorCalibration = "calibration: 16×RTX PRO 6000, K3 MXFP4, KV_OFFLOADING_SIZE=32, HOST_KV_BACKEND=native → ~1.02M physical FP8 KV tokens, 55.8 tok/s target-only / 122.7 DSpark / 155.1 DFlash (github.com/local-inference-lab/rtx6kpro production-runtime)"

// CurvePenaltyNote is the fixed decode-step context line under the miss-rate
// rows: at the anchor's 55.8 tok/s the device step is ~18 ms/token, so even
// the worst swept miss rate is far below 2% of a step.
const CurvePenaltyNote = "device decode step ≈ 18 ms/token at the 55.8 tok/s anchor → offload penalty <2% at m=0.25"

// Verdict is the plan outcome. INFEASIBLE exits 3; TIGHT and
// OFFLOAD_REQUIRED exit 0 but are clearly marked; OK exits 0.
type Verdict string

const (
	VerdictOK              Verdict = "OK"
	VerdictTight           Verdict = "TIGHT"
	VerdictInfeasible      Verdict = "INFEASIBLE"
	VerdictOffloadRequired Verdict = "OFFLOAD_REQUIRED"
)

// KV variant names (issue #14438). --kv-variant selects one; facts may
// override any of them (or add new names) via facts.kv_variants.
const (
	KVVariantFP8            = "fp8"
	KVVariantNVFP4          = "nvfp4"
	KVVariantNVFP4MXFP4Keys = "nvfp4_mxfp4keys"
)

// KnownKVVariantNames lists the built-in variant names in selection order.
var KnownKVVariantNames = []string{KVVariantFP8, KVVariantNVFP4, KVVariantNVFP4MXFP4Keys}

// WitnessChecklist is the fixed post-plan verification list rendered into
// every receipt: what a human or agent must run before trusting the plan.
var WitnessChecklist = []string{
	"load witness: checkpoint loads to completion on this TP×PP layout without OOM",
	"3-prompt accuracy spot-check on the served model",
	"vllm bench serve sweep at the planned max-num-seqs",
	"KV-headroom log cross-check against max_cached_tokens",
}

// KVShape is the per-token KV/latent cache shape (MLA-style latent + indexer).
type KVShape struct {
	FullAttnLayers       int  `json:"full_attn_layers"`
	KVLoraRank           int  `json:"kv_lora_rank"`
	QKRopeHeadDim        int  `json:"qk_rope_head_dim"`
	IndexerKeyDim        int  `json:"indexer_key_dim"`
	LatentDTypeBytes     int  `json:"latent_dtype_bytes"`
	IndexerKeyDTypeBytes int  `json:"indexer_key_dtype_bytes"`
	IndexerScaleBytes    int  `json:"indexer_scale_bytes"`
	TPReplicated         bool `json:"tp_replicated"`
}

// KVVariantBytes is one named KV variant's per-token component bytes for ONE
// full-attention layer — latent + indexer key + indexer scale — before the
// layer count and the TP rule are applied. Facts may override the built-in
// variants (or add new names) via facts.kv_variants; absent an override, the
// built-ins are derived from the base kv fields (see resolveKVVariant).
type KVVariantBytes struct {
	LatentBytes       int64 `json:"latent_bytes"`
	IndexerKeyBytes   int64 `json:"indexer_key_bytes"`
	IndexerScaleBytes int64 `json:"indexer_scale_bytes"`
}

// Facts is one model's measured/cited serving facts (fak.serve-receipt.facts/1).
// The routed-expert fields (issue #14438) are optional: when present they make
// the weight/expert offload-rejection arithmetic exact instead of estimated.
type Facts struct {
	Schema            string   `json:"schema"`
	Model             string   `json:"model"`
	WeightsBytes      int64    `json:"weights_bytes"`
	ObservedAt        string   `json:"observed_at"`
	SourceURLs        []string `json:"source_urls"`
	Notes             []string `json:"notes,omitempty"`
	NumAttentionHeads int      `json:"num_attention_heads,omitempty"` // 0 = unknown: skip divisibility check
	KV                KVShape  `json:"kv"`
	StatePerSeqBytes  int64    `json:"state_per_seq_bytes"`
	StateTPSharded    bool     `json:"state_tp_sharded"`
	// Offload-aware optional fields (issue #14438), all absent in v1 facts:
	KVVariants        map[string]KVVariantBytes `json:"kv_variants,omitempty"` // per-variant per-token component bytes; overrides the derived built-ins
	RoutedExperts     int                       `json:"routed_experts,omitempty"`
	ExpertsPerToken   int                       `json:"experts_per_token,omitempty"`
	ExpertBytesPacked int64                     `json:"expert_bytes_packed,omitempty"` // routed-expert weight bytes (scales included when the checkpoint packs them together)
	ExpertScaleBytes  int64                     `json:"expert_scale_bytes,omitempty"`  // separate scale-tensor bytes when stored apart from the packed payload
}

// GPUProfile is one accelerator's serving-relevant shape. VRAMGB is GiB
// (2^30 bytes) per the UnitsNote convention.
type GPUProfile struct {
	Key       string  `json:"key"`
	VRAMGB    float64 `json:"vram_gb"`
	SM        string  `json:"sm"`
	FP4Native bool    `json:"fp4_native"`
	NVLink    bool    `json:"nvlink"`
	PCIe      string  `json:"pcie,omitempty"`
}

// Config is the serving layout being priced. Zero-valued offload fields
// (issue #14438) mean "unset" and take the documented defaults (DefaultDCP,
// DefaultKVVariant, DefaultHostPoolUsability, DefaultHostBandwidthGBs), which
// is what keeps v1 receipts recomputing identically under validate.
type Config struct {
	TP                  int     `json:"tp"`
	PP                  int     `json:"pp"`
	MaxModelLen         int     `json:"max_model_len,omitempty"` // 0 = not requested
	MaxNumSeqs          int     `json:"max_num_seqs"`
	GPUMemUtil          float64 `json:"gpu_mem_util"`
	ActivationReserveGB float64 `json:"activation_reserve_gb"` // decimal GB (10^9 B)
	// Offload-aware fields (issue #14438), absent in v1 configs:
	HostKVGiB         float64 `json:"host_kv_gib,omitempty"`         // total host KV pool, GiB (2^30 B); 0 = device-only
	HostPoolUsability float64 `json:"host_pool_usability,omitempty"` // usable fraction of the host pool; 0 = default 0.5
	DCP               int     `json:"dcp,omitempty"`                 // decode-context parallel: KV per-GPU ÷ N; 0 = 1
	KVVariant         string  `json:"kv_variant,omitempty"`          // fp8 (default) | nvfp4 | nvfp4_mxfp4keys
	HostBandwidthGBs  float64 `json:"host_bandwidth_gbs,omitempty"`  // host↔device bandwidth for the penalty math; 0 = default 50
	Curve             bool    `json:"curve,omitempty"`               // attach the --curve ESTIMATE block to the receipt
}

// Plan defaults for the offload-aware config fields. HostPoolUsability 0.5 is
// calibrated to the S8 anchor: 32 GiB configured (KV_OFFLOADING_SIZE=32,
// native backend) → ~1.02M physical FP8 KV tokens observed ⇒ ~16 GiB effective,
// i.e. half the configured pool. The mechanism (per-block metadata, alignment,
// allocator overhead in the native host backend) is not directly measured —
// 0.5 is the anchor-derived effective fraction. HostBandwidthGBs 50 is the
// realistic host↔device figure for PCIe 5.0 x16 (≈64 GB/s theoretical).
const (
	DefaultHostPoolUsability = 0.5
	DefaultHostBandwidthGBs  = 50.0
	DefaultKVVariant         = KVVariantFP8
	DefaultDCP               = 1
)

// Arithmetic carries every intermediate of the plan, in bytes and tokens, so
// the rendered receipt shows its work and validate can recompute each field.
type Arithmetic struct {
	Units                  string `json:"units"`
	GPUsTotal              int    `json:"gpus_total"`
	WeightsPerGPUBytes     int64  `json:"weights_per_gpu_bytes"`
	KVBytesPerTokenPerGPU  int64  `json:"kv_bytes_per_token_per_gpu"`
	KVFormula              string `json:"kv_formula"`
	StatePerGPUPerSeqBytes int64  `json:"state_per_gpu_per_seq_bytes"`
	VRAMBytesPerGPU        int64  `json:"vram_bytes_per_gpu"`
	UtilBudgetBytes        int64  `json:"util_budget_bytes"`
	UsablePerGPUBytes      int64  `json:"usable_per_gpu_bytes"`
	ReservedStateBytes     int64  `json:"reserved_state_bytes"`
	ActivationReserveBytes int64  `json:"activation_reserve_bytes"`
	MaxCachedTokens        int64  `json:"max_cached_tokens"`
	RequestedMaxModelLen   int    `json:"requested_max_model_len"`
	HeadroomTokens         int64  `json:"headroom_tokens"` // max_cached − requested; with no request, all cached tokens are headroom
	// Offload-aware fields (issue #14438). PhysicalTokens is the capacity the
	// plan actually promises: device KV pool plus the usable host pool, at the
	// selected variant and DCP. DevicePoolBytes is the per-GPU device KV
	// leftover (clamped ≥ 0); UsableHostBytes is the per-DCP-rank share of the
	// host pool (the offloaded cache mirrors the device layout, so with DCP
	// each rank stores 1/N of each token on host too — the per-rank view keeps
	// both sides of the ratio on the same rank). With host 0 and dcp 1 this
	// reduces exactly to MaxCachedTokens.
	KVVariant               string  `json:"kv_variant,omitempty"`
	KVVariantFormula        string  `json:"kv_variant_formula,omitempty"` // per-layer component derivation for non-fp8 variants
	DCP                     int     `json:"dcp,omitempty"`
	KVEffectivePerTokenGPU  int64   `json:"kv_effective_per_token_per_gpu,omitempty"` // kv_bytes_per_token_per_gpu ÷ dcp
	DevicePoolBytes         int64   `json:"device_pool_bytes,omitempty"`
	HostPoolBytes           int64   `json:"host_pool_bytes,omitempty"` // --host-kv-gib × 2^30
	HostPoolUsability       float64 `json:"host_pool_usability,omitempty"`
	UsableHostBytes         int64   `json:"usable_host_bytes,omitempty"` // per-rank share: pool × usability ÷ dcp
	HostBandwidthGBs        float64 `json:"host_bandwidth_gbs,omitempty"`
	PhysicalTokens          int64   `json:"physical_tokens"`
	PhysicalFormula         string  `json:"physical_formula"`
	ExpertBytesPerExpert    int64   `json:"expert_bytes_per_expert,omitempty"`    // routed-expert bytes/expert (packed + scales) ÷ routed_experts
	ExpertUnionExperts      int64   `json:"expert_union_experts,omitempty"`       // min(routed_experts, max_num_seqs × experts_per_token)
	ExpertUnionBytes        int64   `json:"expert_union_bytes,omitempty"`         // union × bytes/expert
	ExpertSecondsPerForward float64 `json:"expert_seconds_per_forward,omitempty"` // union_bytes ÷ (host_bw × 10^9)
	ExpertEstimated         bool    `json:"expert_estimated,omitempty"`           // bytes/expert estimated from the weights share (no measured expert bytes in facts)
}

// Curve is the --curve block (issue #14438): every row is an ESTIMATE anchored
// to AnchorCalibration; the capacity rows reuse the exact physical_tokens
// arithmetic with the host pool swept, and the miss rows are bytes÷bandwidth
// only (no fixed PCIe transaction latency).
type Curve struct {
	Anchor      string             `json:"anchor"`
	Capacity    []CurveCapacityRow `json:"capacity"`
	MissLatency []CurveMissRow     `json:"miss_latency"`
	PenaltyNote string             `json:"penalty_note"`
}

// CurveCapacityRow is one point of the host-pool sweep.
type CurveCapacityRow struct {
	HostGiB        float64 `json:"host_gib"`
	PhysicalTokens int64   `json:"physical_tokens"`
}

// CurveMissRow is one miss-rate point of the host-miss latency estimate.
type CurveMissRow struct {
	MissRate       float64 `json:"miss_rate"`
	ExtraLatencyUS float64 `json:"extra_latency_us"` // per token, microseconds
}

// Receipt is the plan output (fak.serve-receipt.receipt/1): verdict first,
// then the exact inputs and every intermediate.
type Receipt struct {
	Schema           string     `json:"schema"`
	Model            string     `json:"model"`
	Verdict          Verdict    `json:"verdict"`
	Reason           string     `json:"reason,omitempty"`
	GPU              GPUProfile `json:"gpu"`
	Config           Config     `json:"config"`
	Facts            Facts      `json:"facts"`
	Arithmetic       Arithmetic `json:"arithmetic"`
	Warnings         []string   `json:"warnings"`
	SourceURLs       []string   `json:"source_urls"`
	ObservedAt       string     `json:"observed_at"`
	WitnessChecklist []string   `json:"witness_checklist"`
	Curve            *Curve     `json:"curve,omitempty"` // present only when --curve is passed
}

// builtinProfiles is the closed built-in catalog. Custom hardware enters via
// `--gpu @file.json`, never by editing this table from a caller.
var builtinProfiles = []GPUProfile{
	{Key: "rtx-pro-6000-blackwell-96gb", VRAMGB: 96, SM: "12.0", FP4Native: true, NVLink: false, PCIe: "5.0 x16"},
	{Key: "b200-180gb", VRAMGB: 180, SM: "10.0", FP4Native: true, NVLink: true},
	{Key: "gb300-288gb", VRAMGB: 288, SM: "10.3", FP4Native: true, NVLink: true},
	{Key: "rtx-6000-ada-48gb", VRAMGB: 48, SM: "8.9", FP4Native: false, NVLink: false},
}

// AllProfiles returns the built-in catalog sorted by key.
func AllProfiles() []GPUProfile {
	out := append([]GPUProfile(nil), builtinProfiles...)
	sort.Slice(out, func(i, j int) bool { return out[i].Key < out[j].Key })
	return out
}

// ProfileKeys returns the built-in catalog keys sorted.
func ProfileKeys() []string {
	keys := make([]string, 0, len(builtinProfiles))
	for _, p := range builtinProfiles {
		keys = append(keys, p.Key)
	}
	sort.Strings(keys)
	return keys
}

// LookupProfile resolves a built-in profile by exact key (case-insensitive).
func LookupProfile(key string) (GPUProfile, bool) {
	for _, p := range builtinProfiles {
		if strings.EqualFold(p.Key, strings.TrimSpace(key)) {
			return p, true
		}
	}
	return GPUProfile{}, false
}

// LoadFacts reads and schema-checks a facts document from disk.
func LoadFacts(path string) (Facts, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Facts{}, fmt.Errorf("read facts: %w", err)
	}
	var f Facts
	if err := json.Unmarshal(raw, &f); err != nil {
		return Facts{}, fmt.Errorf("parse facts %s: %w", path, err)
	}
	if f.Schema != FactsSchema {
		return Facts{}, fmt.Errorf("facts %s: schema %q, want %q", path, f.Schema, FactsSchema)
	}
	return f, nil
}

// LoadGPUProfile resolves the --gpu spec: a built-in profile key, or
// "@path.json" for a custom profile with the same JSON shape.
func LoadGPUProfile(spec string) (GPUProfile, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return GPUProfile{}, errors.New("--gpu is required (built-in key or @file.json)")
	}
	if strings.HasPrefix(spec, "@") {
		path := strings.TrimPrefix(spec, "@")
		raw, err := os.ReadFile(path)
		if err != nil {
			return GPUProfile{}, fmt.Errorf("read gpu profile: %w", err)
		}
		var p GPUProfile
		if err := json.Unmarshal(raw, &p); err != nil {
			return GPUProfile{}, fmt.Errorf("parse gpu profile %s: %w", path, err)
		}
		if strings.TrimSpace(p.Key) == "" {
			p.Key = strings.TrimSuffix(filepath.Base(path), ".json")
		}
		if err := checkProfileShape(p); err != nil {
			return GPUProfile{}, fmt.Errorf("gpu profile %s: %w", path, err)
		}
		return p, nil
	}
	p, ok := LookupProfile(spec)
	if !ok {
		return GPUProfile{}, fmt.Errorf("no built-in gpu profile %q (catalog: %s; or pass @file.json)", spec, strings.Join(ProfileKeys(), ", "))
	}
	return p, nil
}

func checkProfileShape(p GPUProfile) error {
	if p.VRAMGB <= 0 {
		return fmt.Errorf("vram_gb %g: must be > 0", p.VRAMGB)
	}
	if strings.TrimSpace(p.SM) == "" {
		return errors.New("sm is empty (e.g. \"12.0\")")
	}
	return nil
}

// ceilDiv returns ⌈a/b⌉ for positive a, b.
func ceilDiv(a, b int64) int64 {
	return (a + b - 1) / b
}

// KVBytesPerTokenPerGPU is the per-token KV/latent cache bytes held by ONE
// GPU: full_attn_layers × ((kv_lora_rank+qk_rope)×latent_dtype_bytes +
// indexer_key_dim×indexer_key_dtype_bytes + indexer_scale_bytes), divided by
// TP only when the latent state is NOT tp_replicated.
func KVBytesPerTokenPerGPU(kv KVShape, tp int) int64 {
	per := int64(kv.FullAttnLayers) *
		(int64(kv.KVLoraRank+kv.QKRopeHeadDim)*int64(kv.LatentDTypeBytes) +
			int64(kv.IndexerKeyDim)*int64(kv.IndexerKeyDTypeBytes) +
			int64(kv.IndexerScaleBytes))
	if !kv.TPReplicated {
		per /= int64(tp)
	}
	return per
}

// KVFormula renders the per-token KV formula with the facts' own numbers, for
// the arithmetic block and for validate's field-by-field comparison.
func KVFormula(kv KVShape, tp int) string {
	s := fmt.Sprintf("%d × ((%d+%d)×%d + %d×%d + %d)",
		kv.FullAttnLayers, kv.KVLoraRank, kv.QKRopeHeadDim, kv.LatentDTypeBytes,
		kv.IndexerKeyDim, kv.IndexerKeyDTypeBytes, kv.IndexerScaleBytes)
	if !kv.TPReplicated {
		s += fmt.Sprintf(" / tp %d", tp)
	}
	return s
}

// kvFormulaFor renders the arithmetic block's kv_bytes_per_token_per_gpu
// formula. fp8 (the default variant) keeps the v1 KVFormula rendering
// byte-identical; a variant formula (already rendered with the layer count by
// resolveKVVariant) gets the TP suffix appended, mirroring the v1 shape.
func kvFormulaFor(kv KVShape, variantFormula string, tp int) string {
	if variantFormula == "" {
		return KVFormula(kv, tp)
	}
	if !kv.TPReplicated {
		return variantFormula + fmt.Sprintf(" / tp %d", tp)
	}
	return variantFormula
}

// resolveKVVariant resolves --kv-variant to per-layer component bytes.
//
// A facts.kv_variants entry with the requested name always wins (explicit
// beats derived). Otherwise the built-ins are derived from the base kv fields:
//
//	fp8:                latent (rank+rope)×latent_dtype_bytes; keys dim×key_dtype_bytes; scale bytes — the v1 default
//	nvfp4:              latent ceil(rank/2) + ceil(rank/16) + rope (FP4 payload at 2 elements/byte + one FP8
//	                    group scale per 16 elements on the compressed latent; the RoPE key stays FP8);
//	                    keys dim×key_dtype_bytes; scale bytes
//	nvfp4_mxfp4keys:    same NVFP4 latent; keys ceil(dim/2) + ceil(dim/32) (MXFP4: FP4 payload + one FP8
//	                    scale per 32 elements); no separate per-token scale bytes (folded into the
//	                    MXFP4 block scales)
//
// For the Kimi K3.1-Pro shape (24 layers, rank 512, rope 64, key dim 128,
// scale 4) the derivations give exactly fp8 24×(576+128+4)=16,992 B,
// nvfp4 24×(352+132)=11,616 B, nvfp4_mxfp4keys 24×(352+68)=10,080 B.
func resolveKVVariant(kv KVShape, variants map[string]KVVariantBytes, name string) (KVVariantBytes, string, error) {
	if strings.TrimSpace(name) == "" {
		name = DefaultKVVariant
	}
	if v, ok := variants[name]; ok {
		if v.LatentBytes <= 0 || v.IndexerKeyBytes < 0 || v.IndexerScaleBytes < 0 {
			return KVVariantBytes{}, "", fmt.Errorf("facts.kv_variants[%s]: latent_bytes %d must be > 0 and key/scale bytes ≥ 0", name, v.LatentBytes)
		}
		return v, fmt.Sprintf("%d × (%d + %d + %d) (facts.kv_variants[%s])", kv.FullAttnLayers, v.LatentBytes, v.IndexerKeyBytes, v.IndexerScaleBytes, name), nil
	}
	switch name {
	case KVVariantFP8:
		v := KVVariantBytes{
			LatentBytes:       int64(kv.KVLoraRank+kv.QKRopeHeadDim) * int64(kv.LatentDTypeBytes),
			IndexerKeyBytes:   int64(kv.IndexerKeyDim) * int64(kv.IndexerKeyDTypeBytes),
			IndexerScaleBytes: int64(kv.IndexerScaleBytes),
		}
		return v, "", nil // fp8 keeps the v1 KVFormula rendering byte-identical
	case KVVariantNVFP4:
		latent := ceilDiv(int64(kv.KVLoraRank), 2) + ceilDiv(int64(kv.KVLoraRank), 16) + int64(kv.QKRopeHeadDim)
		key := int64(kv.IndexerKeyDim) * int64(kv.IndexerKeyDTypeBytes)
		v := KVVariantBytes{LatentBytes: latent, IndexerKeyBytes: key, IndexerScaleBytes: int64(kv.IndexerScaleBytes)}
		return v, fmt.Sprintf("%d × (%d + %d + %d) (nvfp4 latent = ceil(%d/2)+ceil(%d/16)+%d = %d; fp8 keys = %d×%d+%d = %d)",
			kv.FullAttnLayers, v.LatentBytes, v.IndexerKeyBytes, v.IndexerScaleBytes,
			kv.KVLoraRank, kv.KVLoraRank, kv.QKRopeHeadDim, latent,
			kv.IndexerKeyDim, kv.IndexerKeyDTypeBytes, kv.IndexerScaleBytes, v.IndexerKeyBytes+v.IndexerScaleBytes), nil
	case KVVariantNVFP4MXFP4Keys:
		latent := ceilDiv(int64(kv.KVLoraRank), 2) + ceilDiv(int64(kv.KVLoraRank), 16) + int64(kv.QKRopeHeadDim)
		key := ceilDiv(int64(kv.IndexerKeyDim), 2) + ceilDiv(int64(kv.IndexerKeyDim), 32)
		v := KVVariantBytes{LatentBytes: latent, IndexerKeyBytes: key}
		return v, fmt.Sprintf("%d × (%d + %d + 0) (nvfp4 latent = ceil(%d/2)+ceil(%d/16)+%d = %d; mxfp4 keys = ceil(%d/2)+ceil(%d/32) = %d)",
			kv.FullAttnLayers, v.LatentBytes, v.IndexerKeyBytes,
			kv.KVLoraRank, kv.KVLoraRank, kv.QKRopeHeadDim, latent,
			kv.IndexerKeyDim, kv.IndexerKeyDim, key), nil
	default:
		return KVVariantBytes{}, "", fmt.Errorf("kv variant %q: unknown (known: %s; or override/add it in facts.kv_variants)", name, strings.Join(KnownKVVariantNames, ", "))
	}
}

// KVBytesPerTokenPerGPUVariant applies the layer count and the base TP rule to
// one resolved variant's per-layer component bytes.
func KVBytesPerTokenPerGPUVariant(kv KVShape, v KVVariantBytes, tp int) int64 {
	per := int64(kv.FullAttnLayers) * (v.LatentBytes + v.IndexerKeyBytes + v.IndexerScaleBytes)
	if !kv.TPReplicated {
		per /= int64(tp)
	}
	return per
}

// validateInputs rejects structurally impossible inputs with a usage-class
// error (CLI exit 2), distinct from a computed INFEASIBLE verdict (exit 3).
// Zero-valued offload fields are "unset" and legal here (they take the
// documented defaults in Plan); only negative/out-of-range/unknown values
// refuse.
func validateInputs(f Facts, gpu GPUProfile, cfg Config) error {
	if f.Schema != FactsSchema {
		return fmt.Errorf("facts schema %q: want %q", f.Schema, FactsSchema)
	}
	if strings.TrimSpace(f.Model) == "" {
		return errors.New("facts.model is empty")
	}
	if f.WeightsBytes <= 0 {
		return fmt.Errorf("facts.weights_bytes %d: must be > 0", f.WeightsBytes)
	}
	if f.KV.FullAttnLayers <= 0 {
		return fmt.Errorf("facts.kv.full_attn_layers %d: must be > 0", f.KV.FullAttnLayers)
	}
	if cfg.TP < 1 || cfg.PP < 1 {
		return fmt.Errorf("tp/pp must be ≥ 1 (got tp=%d pp=%d)", cfg.TP, cfg.PP)
	}
	if cfg.MaxNumSeqs < 1 {
		return fmt.Errorf("max_num_seqs %d: must be ≥ 1", cfg.MaxNumSeqs)
	}
	if cfg.MaxModelLen < 0 {
		return fmt.Errorf("max_model_len %d: must be ≥ 0 (0 = not requested)", cfg.MaxModelLen)
	}
	if cfg.GPUMemUtil <= 0 || cfg.GPUMemUtil > 1 {
		return fmt.Errorf("gpu_mem_util %g: must be in (0, 1]", cfg.GPUMemUtil)
	}
	if cfg.ActivationReserveGB < 0 {
		return fmt.Errorf("activation_reserve_gb %g: must be ≥ 0", cfg.ActivationReserveGB)
	}
	if cfg.DCP < 0 {
		return fmt.Errorf("dcp %d: must be ≥ 0 (0 = unset → %d)", cfg.DCP, DefaultDCP)
	}
	if cfg.HostKVGiB < 0 {
		return fmt.Errorf("host_kv_gib %g: must be ≥ 0", cfg.HostKVGiB)
	}
	if cfg.HostPoolUsability < 0 || cfg.HostPoolUsability > 1 {
		return fmt.Errorf("host_pool_usability %g: must be in [0, 1] (0 = unset → %g)", cfg.HostPoolUsability, DefaultHostPoolUsability)
	}
	if cfg.HostBandwidthGBs < 0 {
		return fmt.Errorf("host_bandwidth_gbs %g: must be ≥ 0 (0 = unset → %g)", cfg.HostBandwidthGBs, DefaultHostBandwidthGBs)
	}
	if _, _, err := resolveKVVariant(f.KV, f.KVVariants, cfg.KVVariant); err != nil {
		return err
	}
	if err := checkProfileShape(gpu); err != nil {
		return fmt.Errorf("gpu profile: %w", err)
	}
	if KVBytesPerTokenPerGPU(f.KV, cfg.TP) <= 0 {
		return errors.New("facts.kv yields 0 bytes/token (check full_attn_layers, ranks, dtype bytes); cannot size the cache")
	}
	return nil
}

// physicalTokens is the offload-aware capacity promise (issue #14438):
//
//	physical_tokens = floor((device_pool_B + usable_host_B) / (kv_per_token_per_gpu / dcp))
//
// where device_pool_B is ONE GPU's device KV leftover and usable_host_B is the
// per-DCP-rank share of the host pool (total × usability ÷ dcp). DCP divides
// the per-GPU KV because a DCP rank stores 1/N of each token — on device AND
// on host, since the offloaded cache mirrors the device layout — so both sides
// of the ratio are per-rank. Equivalently (multiplying through by dcp):
// floor((device_pool×dcp + host_total×usability) / kv_per_token_per_gpu); with
// device_pool 0 the dcp factors cancel and the anchor reduces to
// host_total × usability / kv_per_token (the S8 calibration form).
func physicalTokens(devicePool, hostTotalBytes int64, usability float64, kvPerTok int64, dcp int) int64 {
	if dcp < 1 {
		dcp = 1
	}
	denom := kvPerTok / int64(dcp) // integer floor, documented
	if denom <= 0 {
		return 0
	}
	usableHost := int64(float64(hostTotalBytes)*usability) / int64(dcp)
	if usableHost < 0 {
		usableHost = 0
	}
	return (devicePool + usableHost) / denom
}

// computeCurve sweeps the host pool for the capacity rows and prices the
// miss-rate rows; the device pool and all other flags are held at the
// receipt's values. Every row is an ESTIMATE anchored to AnchorCalibration.
func computeCurve(devicePool, kvPerTok int64, cfg Config) *Curve {
	dcp := cfg.DCP
	if dcp < 1 {
		dcp = DefaultDCP
	}
	curve := &Curve{Anchor: AnchorCalibration, PenaltyNote: CurvePenaltyNote}
	for _, gib := range []float64{0, 16, 32, 64, 128} {
		host := int64(gib * float64(GibBytes))
		curve.Capacity = append(curve.Capacity, CurveCapacityRow{
			HostGiB:        gib,
			PhysicalTokens: physicalTokens(devicePool, host, cfg.HostPoolUsability, kvPerTok, dcp),
		})
	}
	bw := cfg.HostBandwidthGBs
	if bw <= 0 {
		bw = DefaultHostBandwidthGBs
	}
	perTokBytes := kvPerTok / int64(dcp)
	for _, m := range []float64{0.05, 0.10, 0.25} {
		curve.MissLatency = append(curve.MissLatency, CurveMissRow{
			MissRate:       m,
			ExtraLatencyUS: m * float64(perTokBytes) / (bw * 1e9) * 1e6, // bytes ÷ (B/s) → s → µs
		})
	}
	return curve
}

// Plan computes the serving-config receipt. The only floors are documented
// ones: the util budget (vram_bytes × util, floored once to whole bytes), the
// final token division, the per-rank host share (pool × usability ÷ dcp), and
// kv_per_token ÷ dcp. Everything between is exact integer byte math.
//
// Verdicts (issue #14438): weights over budget still gate HARD (INFEASIBLE,
// with the routed-expert offload-rejection warning when the facts allow it);
// a device KV shortfall no longer kills — when the host pool covers the
// request (or is the only positive pool) the verdict is OFFLOAD_REQUIRED
// (exit 0, marked); TIGHT/OK otherwise, as in v1.
func Plan(f Facts, gpu GPUProfile, cfg Config) (Receipt, error) {
	if err := validateInputs(f, gpu, cfg); err != nil {
		return Receipt{}, err
	}

	// Normalize the unset offload fields to their documented defaults (the
	// receipt stores the effective config, so validate recomputes identically).
	if cfg.DCP == 0 {
		cfg.DCP = DefaultDCP
	}
	if cfg.KVVariant == "" {
		cfg.KVVariant = DefaultKVVariant
	}
	if cfg.HostPoolUsability == 0 {
		cfg.HostPoolUsability = DefaultHostPoolUsability
	}
	if cfg.HostBandwidthGBs == 0 {
		cfg.HostBandwidthGBs = DefaultHostBandwidthGBs
	}

	gpus := cfg.TP * cfg.PP
	weightsPerGPU := ceilDiv(f.WeightsBytes, int64(gpus))
	variant, variantFormula, err := resolveKVVariant(f.KV, f.KVVariants, cfg.KVVariant)
	if err != nil {
		return Receipt{}, err // unreachable after validateInputs; kept for safety
	}
	kvPerTok := KVBytesPerTokenPerGPUVariant(f.KV, variant, cfg.TP)

	statePerGPUPerSeq := f.StatePerSeqBytes
	if f.StateTPSharded {
		statePerGPUPerSeq /= int64(cfg.TP) // integer floor, documented in the arithmetic block
	}

	vramBytes := int64(math.Floor(gpu.VRAMGB * float64(GibBytes)))
	utilBudget := int64(math.Floor(float64(vramBytes) * cfg.GPUMemUtil))
	usable := utilBudget - weightsPerGPU
	reservedState := statePerGPUPerSeq * int64(cfg.MaxNumSeqs)
	activationBytes := int64(math.Floor(cfg.ActivationReserveGB * float64(GbBytes)))

	maxCached := int64(0)
	if leftover := usable - reservedState - activationBytes; leftover > 0 {
		maxCached = leftover / kvPerTok // integer floor; clamped at 0 otherwise
	}
	devicePool := usable - reservedState - activationBytes
	if devicePool < 0 {
		devicePool = 0 // clamped: the device carries no KV when weights+state+reserve fill the card
	}

	hostPool := int64(cfg.HostKVGiB * float64(GibBytes))
	physical := physicalTokens(devicePool, hostPool, cfg.HostPoolUsability, kvPerTok, cfg.DCP)
	usableHost := int64(float64(hostPool)*cfg.HostPoolUsability) / int64(cfg.DCP)
	physicalFormula := fmt.Sprintf("floor((device_pool %d B + usable_host %d B) / (kv %d B/token ÷ dcp %d))",
		devicePool, usableHost, kvPerTok, cfg.DCP)

	verdict := VerdictOK
	var reasons []string
	weightsInfeasible := false
	if usable <= 0 {
		verdict = VerdictInfeasible
		weightsInfeasible = true
		reasons = append(reasons, fmt.Sprintf("usable_per_gpu ≤ 0 after weights: %d B util budget vs %d B weights_per_gpu", utilBudget, weightsPerGPU))
	}
	if f.NumAttentionHeads > 0 && f.NumAttentionHeads%cfg.TP != 0 {
		verdict = VerdictInfeasible
		weightsInfeasible = true
		reasons = append(reasons, fmt.Sprintf("num_attention_heads %d not divisible by tp %d", f.NumAttentionHeads, cfg.TP))
	}

	var warnings []string
	if !gpu.FP4Native {
		warnings = append(warnings, WarnFP4NotNative)
	}
	if cfg.TP > 2 && !gpu.NVLink {
		warnings = append(warnings, WarnPCIeWideTP)
	}
	if cfg.DCP >= 8 {
		warnings = append(warnings, WarnDCPSlow)
	}

	// Routed-expert host offload is rejected with arithmetic whenever the
	// weights gate is HARD-infeasible and the facts name the expert shape:
	// serving cannot fall back to streaming routed experts from host per
	// forward, because every forward activates a large UNION of experts across
	// the batch — the per-forward host read dwarfs a decode step. KV offload
	// stays available (per-token reads are small and prefix-stable).
	var bytesPerExpert, unionExperts, unionBytes int64
	var secondsPerForward float64
	var expertEstimated bool
	if weightsInfeasible && f.RoutedExperts > 0 && f.ExpertsPerToken > 0 {
		if f.ExpertBytesPacked > 0 || f.ExpertScaleBytes > 0 {
			bytesPerExpert = (f.ExpertBytesPacked + f.ExpertScaleBytes) / int64(f.RoutedExperts)
		} else {
			bytesPerExpert = f.WeightsBytes / int64(f.RoutedExperts) // estimate: experts carry ~the whole checkpoint share
			expertEstimated = true
		}
		unionExperts = min(int64(f.RoutedExperts), int64(cfg.MaxNumSeqs)*int64(f.ExpertsPerToken))
		unionBytes = unionExperts * bytesPerExpert
		secondsPerForward = float64(unionBytes) / (cfg.HostBandwidthGBs * 1e9)
		warnings = append(warnings, fmt.Sprintf(
			"weight/expert host offload rejected: ~%.1f s/forward (union of ~%d experts × %.2f GB) — KV offload only",
			secondsPerForward, unionExperts, float64(bytesPerExpert)/1e9))
	}

	// Device KV shortfall no longer kills: when the host pool covers the
	// request (or is the only positive pool) the verdict is OFFLOAD_REQUIRED.
	if verdict != VerdictInfeasible {
		shortOfLen := cfg.MaxModelLen > 0 && maxCached < int64(cfg.MaxModelLen)
		noDeviceHeadroom := cfg.MaxModelLen == 0 && maxCached <= 0
		if shortOfLen || noDeviceHeadroom {
			reach := physical > 0 // no request: a positive pool is the bar
			if cfg.MaxModelLen > 0 {
				reach = physical >= int64(cfg.MaxModelLen)
			}
			switch {
			case usableHost > 0 && reach:
				verdict = VerdictOffloadRequired
				if cfg.MaxModelLen > 0 {
					reasons = append(reasons, fmt.Sprintf(
						"device KV budget covers %s of the requested max-model-len %s; the host pool covers the rest — physical_tokens %s at kv %s B/token ÷ dcp %d (exit 0, marked: decode reads miss to host)",
						comma(maxCached), comma(int64(cfg.MaxModelLen)), comma(physical), comma(kvPerTok), cfg.DCP))
				} else {
					reasons = append(reasons, fmt.Sprintf(
						"device KV budget is exhausted (max_cached_tokens %s); the host pool carries the cache — physical_tokens %s at kv %s B/token ÷ dcp %d (exit 0, marked: decode reads miss to host)",
						comma(maxCached), comma(physical), comma(kvPerTok), cfg.DCP))
				}
			case shortOfLen:
				verdict = VerdictTight
				reasons = append(reasons, fmt.Sprintf("max_cached_tokens %d < requested max-model-len %d", maxCached, cfg.MaxModelLen))
			}
		}
	}

	ar := Arithmetic{
		Units:                   UnitsNote,
		GPUsTotal:               gpus,
		WeightsPerGPUBytes:      weightsPerGPU,
		KVBytesPerTokenPerGPU:   kvPerTok,
		KVFormula:               kvFormulaFor(f.KV, variantFormula, cfg.TP),
		StatePerGPUPerSeqBytes:  statePerGPUPerSeq,
		VRAMBytesPerGPU:         vramBytes,
		UtilBudgetBytes:         utilBudget,
		UsablePerGPUBytes:       usable,
		ReservedStateBytes:      reservedState,
		ActivationReserveBytes:  activationBytes,
		MaxCachedTokens:         maxCached,
		RequestedMaxModelLen:    cfg.MaxModelLen,
		HeadroomTokens:          maxCached - int64(cfg.MaxModelLen),
		KVVariant:               cfg.KVVariant,
		KVVariantFormula:        variantFormula,
		DCP:                     cfg.DCP,
		KVEffectivePerTokenGPU:  kvPerTok / int64(cfg.DCP),
		DevicePoolBytes:         devicePool,
		HostPoolBytes:           hostPool,
		HostPoolUsability:       cfg.HostPoolUsability,
		UsableHostBytes:         usableHost,
		HostBandwidthGBs:        cfg.HostBandwidthGBs,
		PhysicalTokens:          physical,
		PhysicalFormula:         physicalFormula,
		ExpertBytesPerExpert:    bytesPerExpert,
		ExpertUnionExperts:      unionExperts,
		ExpertUnionBytes:        unionBytes,
		ExpertSecondsPerForward: secondsPerForward,
		ExpertEstimated:         expertEstimated,
	}

	receipt := Receipt{
		Schema:           ReceiptSchema,
		Model:            f.Model,
		Verdict:          verdict,
		Reason:           strings.Join(reasons, "; "),
		GPU:              gpu,
		Config:           cfg,
		Facts:            f,
		Arithmetic:       ar,
		Warnings:         warnings,
		SourceURLs:       f.SourceURLs,
		ObservedAt:       f.ObservedAt,
		WitnessChecklist: append([]string(nil), WitnessChecklist...),
	}
	if cfg.Curve {
		receipt.Curve = computeCurve(devicePool, kvPerTok, cfg)
	}
	return receipt, nil
}
