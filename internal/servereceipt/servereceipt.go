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

// Verdict is the plan outcome. INFEASIBLE exits 3; TIGHT exits 0 but is
// clearly marked; OK exits 0.
type Verdict string

const (
	VerdictOK         Verdict = "OK"
	VerdictTight      Verdict = "TIGHT"
	VerdictInfeasible Verdict = "INFEASIBLE"
)

// Warning strings are fixed literals so a recomputed warning list is
// byte-comparable during validate.
const (
	WarnFP4NotNative = "NVFP4 checkpoints require Blackwell SM≥10.0"
	WarnPCIeWideTP   = "wide TP all-reduce over PCIe; PP-heavy layouts preferred (measured: TP4 6–7 tok/s vs TP2+PP2 46–49 on 397B MoE, flashinfer#2577)"
)

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

// Facts is one model's measured/cited serving facts (fak.serve-receipt.facts/1).
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

// Config is the serving layout being priced.
type Config struct {
	TP                  int     `json:"tp"`
	PP                  int     `json:"pp"`
	MaxModelLen         int     `json:"max_model_len,omitempty"` // 0 = not requested
	MaxNumSeqs          int     `json:"max_num_seqs"`
	GPUMemUtil          float64 `json:"gpu_mem_util"`
	ActivationReserveGB float64 `json:"activation_reserve_gb"` // decimal GB (10^9 B)
}

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

// validateInputs rejects structurally impossible inputs with a usage-class
// error (CLI exit 2), distinct from a computed INFEASIBLE verdict (exit 3).
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
	if err := checkProfileShape(gpu); err != nil {
		return fmt.Errorf("gpu profile: %w", err)
	}
	if KVBytesPerTokenPerGPU(f.KV, cfg.TP) <= 0 {
		return errors.New("facts.kv yields 0 bytes/token (check full_attn_layers, ranks, dtype bytes); cannot size the cache")
	}
	return nil
}

// Plan computes the serving-config receipt. The only floors are documented
// ones: the util budget (vram_bytes × util, floored once to whole bytes) and
// the final token division. Everything between is exact integer byte math.
func Plan(f Facts, gpu GPUProfile, cfg Config) (Receipt, error) {
	if err := validateInputs(f, gpu, cfg); err != nil {
		return Receipt{}, err
	}

	gpus := cfg.TP * cfg.PP
	weightsPerGPU := ceilDiv(f.WeightsBytes, int64(gpus))
	kvPerTok := KVBytesPerTokenPerGPU(f.KV, cfg.TP)

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

	verdict := VerdictOK
	var reasons []string
	if usable <= 0 {
		verdict = VerdictInfeasible
		reasons = append(reasons, fmt.Sprintf("usable_per_gpu ≤ 0 after weights: %d B util budget vs %d B weights_per_gpu", utilBudget, weightsPerGPU))
	}
	if f.NumAttentionHeads > 0 && f.NumAttentionHeads%cfg.TP != 0 {
		verdict = VerdictInfeasible
		reasons = append(reasons, fmt.Sprintf("num_attention_heads %d not divisible by tp %d", f.NumAttentionHeads, cfg.TP))
	}
	if verdict != VerdictInfeasible && cfg.MaxModelLen > 0 && maxCached < int64(cfg.MaxModelLen) {
		verdict = VerdictTight
		reasons = append(reasons, fmt.Sprintf("max_cached_tokens %d < requested max-model-len %d", maxCached, cfg.MaxModelLen))
	}

	var warnings []string
	if !gpu.FP4Native {
		warnings = append(warnings, WarnFP4NotNative)
	}
	if cfg.TP > 2 && !gpu.NVLink {
		warnings = append(warnings, WarnPCIeWideTP)
	}

	ar := Arithmetic{
		Units:                  UnitsNote,
		GPUsTotal:              gpus,
		WeightsPerGPUBytes:     weightsPerGPU,
		KVBytesPerTokenPerGPU:  kvPerTok,
		KVFormula:              KVFormula(f.KV, cfg.TP),
		StatePerGPUPerSeqBytes: statePerGPUPerSeq,
		VRAMBytesPerGPU:        vramBytes,
		UtilBudgetBytes:        utilBudget,
		UsablePerGPUBytes:      usable,
		ReservedStateBytes:     reservedState,
		ActivationReserveBytes: activationBytes,
		MaxCachedTokens:        maxCached,
		RequestedMaxModelLen:   cfg.MaxModelLen,
		HeadroomTokens:         maxCached - int64(cfg.MaxModelLen),
	}

	return Receipt{
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
	}, nil
}
