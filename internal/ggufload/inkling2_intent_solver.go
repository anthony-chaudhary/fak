package ggufload

import (
	"fmt"
	"math"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// ServeProfile names the high-level serving intent a user wants the reconciler
// to satisfy without hand-tuning KV dtypes, context bounds, or prefill chunks.
type ServeProfile string

const (
	ServeProfileCoding  ServeProfile = "coding"
	ServeProfileChat    ServeProfile = "chat"
	ServeProfileBatch   ServeProfile = "batch"
	ServeProfileLongDoc ServeProfile = "long-doc"
	ServeProfileDemo    ServeProfile = "demo"
)

// ExpertOverrides carries explicit operator pins that always take precedence
// over profile defaults and automatic ladders. Each non-zero/non-nil override
// is recorded in ResolvedPlan.OverridesApplied.
type ExpertOverrides struct {
	Ctx          int
	KVDType      *Inkling2KVDType
	PrefillChunk int
	Spec         *bool
}

// ServeIntent is the declarative user request reconciled against a
// HardwareEnvelope to produce a concrete ResolvedPlan.
type ServeIntent struct {
	Profile             ServeProfile
	MinContext          int  // 0 = profile default
	TargetContext       int  // 0 = profile default
	Concurrency         int  // 0 = profile default
	AllowExperimentalKV bool // gates opt-in fp8/nvfp4 until KL default-on clears
	ExpertOverrides     ExpertOverrides
}

// HardwareEnvelope describes the post-weight device memory and kernel
// capabilities available to host the resident KV cache, arenas, and draft.
type HardwareEnvelope struct {
	FreeVRAMBytes  int64
	SMArch         int // e.g. 120 for Blackwell, 89 for Ada, 0 for Metal/CPU
	HasNVFP4       bool
	HasFP8         bool
	HasFP16        bool
	HasDSparkDraft bool
}

// ResolvedPlan is the reconciled execution plan produced by
// ResolveInkling2ServeIntent.
type ResolvedPlan struct {
	Profile          ServeProfile
	Ctx              int
	Concurrency      int
	KVDType          Inkling2KVDType
	PrefillChunk     int
	EnableSpec       bool
	SpillLayers      int
	EstimatedBytes   int64
	HeadroomBytes    int64
	OverridesApplied []string
	Summary          string
}

type inkling2ProfileDefaults struct {
	concurrency int
	minCtx      int
	targetCtx   int
	preferSpec  bool
}

// inkling2DSparkDraftReserveBytes is the conservative device footprint reserved
// for the DSpark speculative draft weights, tap buffers, and verify logits when
// speculative decoding is enabled.
const inkling2DSparkDraftReserveBytes int64 = 384 << 20

// inkling2PrefillChunkLadder is the ordered prefill chunk descent tried when
// fitting a target context into free VRAM without spilling.
var inkling2PrefillChunkLadder = []int{512, 256, 128, 64}

func inkling2ProfileSpec(p ServeProfile) (ServeProfile, inkling2ProfileDefaults, error) {
	if p == "" {
		p = ServeProfileCoding
	}
	switch p {
	case ServeProfileCoding:
		return p, inkling2ProfileDefaults{concurrency: 1, minCtx: 65536, targetCtx: 131072, preferSpec: true}, nil
	case ServeProfileChat, ServeProfileDemo:
		return p, inkling2ProfileDefaults{concurrency: 1, minCtx: 16384, targetCtx: 32768, preferSpec: true}, nil
	case ServeProfileBatch:
		return p, inkling2ProfileDefaults{concurrency: 4, minCtx: 16384, targetCtx: 32768, preferSpec: false}, nil
	case ServeProfileLongDoc:
		return p, inkling2ProfileDefaults{concurrency: 1, minCtx: 131072, targetCtx: 262144, preferSpec: true}, nil
	default:
		return "", inkling2ProfileDefaults{}, fmt.Errorf("inkling2 intent: unknown serve profile %q", p)
	}
}

func inkling2HardwareSupportsKVDType(hw HardwareEnvelope, dtype Inkling2KVDType) bool {
	switch dtype {
	case Inkling2KVF32:
		return true
	case Inkling2KVF16:
		return hw.HasFP16
	case Inkling2KVFP8:
		return hw.HasFP8
	case Inkling2KVNVFP4:
		return hw.HasNVFP4
	default:
		return false
	}
}

// inkling2IntentFit prices a candidate (ctx, concurrency, dtype, chunk, specBytes)
// using the exact resident KV and non-KV byte accounting (inkling2ResidentKVBytesFor,
// inkling2ResidentOtherBytes, and inkling2KVFitFor).
func inkling2IntentFit(hp model.Inkling2LoopHParams, ctx, concurrency int, dtype Inkling2KVDType, chunk int, specBytes, free int64) (estimated int64, fits bool) {
	batched := chunk > 0
	other := inkling2ResidentOtherBytes(hp, inkling2ArenaFloats, chunk, 0) + specBytes
	var extraSlots int64
	if concurrency > 1 {
		extraSlots = int64(concurrency-1) * inkling2ResidentKVBytesFor(hp, ctx, batched, chunk, dtype)
	}
	kv, err := inkling2KVFitFor(hp, ctx, batched, chunk, dtype, other+extraSlots, free)
	return kv + other + extraSlots, err == nil
}

// inkling2IntentMaxCtx finds the largest 256-aligned context in [0, ceil] that
// fits concurrency slots of dtype KV + non-KV buffers inside free minus device
// headroom, using inkling2ResidentKVBytesFor and inkling2ResidentOtherBytes.
func inkling2IntentMaxCtx(hp model.Inkling2LoopHParams, concurrency int, dtype Inkling2KVDType, chunk, ceil int, specBytes, free int64) int {
	budget := free - inkling2DeviceHeadroom(free)
	other := inkling2ResidentOtherBytes(hp, inkling2ArenaFloats, chunk, 0) + specBytes
	left := budget - other
	if left <= 0 || concurrency <= 0 {
		return 0
	}
	perSlotBudget := left / int64(concurrency)
	n := inkling2MaxResidentCtxFor(hp, chunk > 0, chunk, dtype, ceil, perSlotBudget)
	n = n / 256 * 256
	if n < Inkling2ResidentCtxMin {
		return 0
	}
	return n
}

func inkling2EstimateSpillLayers(hp model.Inkling2LoopHParams, overBytes, free int64) int {
	if overBytes <= 0 || hp.Layers <= 0 {
		return 0
	}
	perLayer := max(int64(256<<20), free/int64(hp.Layers))
	layers := int((overBytes + perLayer - 1) / perLayer)
	return min(hp.Layers, max(1, layers))
}

// ResolveInkling2ServeIntent reconciles a declarative ServeIntent against hp
// and hw into a concrete ResolvedPlan using the exact resident VRAM accounting
// of inkling2loop_resident.go.
func ResolveInkling2ServeIntent(hp model.Inkling2LoopHParams, intent ServeIntent, hw HardwareEnvelope) (ResolvedPlan, error) {
	if hp.Layers <= 0 || hp.KVHeads <= 0 || hp.HeadDim <= 0 || len(hp.LocalLayer) != hp.Layers {
		return ResolvedPlan{}, fmt.Errorf("inkling2 intent: invalid model hparams (layers=%d kv_heads=%d head_dim=%d local_len=%d)",
			hp.Layers, hp.KVHeads, hp.HeadDim, len(hp.LocalLayer))
	}
	if hw.FreeVRAMBytes <= 0 {
		return ResolvedPlan{}, fmt.Errorf("inkling2 intent: free VRAM must be positive (got %d)", hw.FreeVRAMBytes)
	}

	profile, defaults, err := inkling2ProfileSpec(intent.Profile)
	if err != nil {
		return ResolvedPlan{}, err
	}

	concurrency := defaults.concurrency
	if intent.Concurrency > 0 {
		concurrency = intent.Concurrency
	}
	minCtx := defaults.minCtx
	if intent.MinContext > 0 {
		minCtx = intent.MinContext
	}
	targetCtx := defaults.targetCtx
	if intent.TargetContext > 0 {
		targetCtx = intent.TargetContext
	}
	if targetCtx < minCtx {
		targetCtx = minCtx
	}

	var overrides []string
	ov := intent.ExpertOverrides

	if ov.Ctx > 0 {
		minCtx = ov.Ctx
		targetCtx = ov.Ctx
		overrides = append(overrides, fmt.Sprintf("ctx=%d", ov.Ctx))
	}

	var dtypes []Inkling2KVDType
	if ov.KVDType != nil {
		dt := *ov.KVDType
		if !inkling2HardwareSupportsKVDType(hw, dt) {
			return ResolvedPlan{}, fmt.Errorf("inkling2 intent: expert override kv_dtype=%s is not supported by hardware (sm_%d)", dt, hw.SMArch)
		}
		dtypes = []Inkling2KVDType{dt}
		overrides = append(overrides, fmt.Sprintf("kv_dtype=%s", dt))
	} else {
		if hw.HasFP16 {
			dtypes = append(dtypes, Inkling2KVF16)
		} else {
			dtypes = append(dtypes, Inkling2KVF32)
		}
		if intent.AllowExperimentalKV {
			if hw.HasFP8 {
				dtypes = append(dtypes, Inkling2KVFP8)
			}
			if hw.HasNVFP4 {
				dtypes = append(dtypes, Inkling2KVNVFP4)
			}
		}
	}

	var chunks []int
	if ov.PrefillChunk > 0 {
		chunks = []int{ov.PrefillChunk}
		overrides = append(overrides, fmt.Sprintf("prefill_chunk=%d", ov.PrefillChunk))
	} else {
		chunks = inkling2PrefillChunkLadder
	}

	if ov.Spec != nil {
		overrides = append(overrides, fmt.Sprintf("spec=%t", *ov.Spec))
	}

	var (
		chosenCtx   int
		chosenDType Inkling2KVDType
		chosenChunk int
		chosenEst   int64
		spillLayers int
		found       bool
	)

	// Phase 1: reach targetCtx with 0 spill, preferring highest KV precision then widest prefill chunk.
	for _, dt := range dtypes {
		for _, ch := range chunks {
			if est, ok := inkling2IntentFit(hp, targetCtx, concurrency, dt, ch, 0, hw.FreeVRAMBytes); ok {
				chosenCtx, chosenDType, chosenChunk, chosenEst, found = targetCtx, dt, ch, est, true
				break
			}
		}
		if found {
			break
		}
	}

	// Phase 2: targetCtx did not fit all-resident; maximize ctx in [minCtx, targetCtx) with 0 spill.
	if !found && ov.Ctx == 0 {
		bestCtx := 0
		for _, dt := range dtypes {
			for _, ch := range chunks {
				fitCtx := inkling2IntentMaxCtx(hp, concurrency, dt, ch, targetCtx, 0, hw.FreeVRAMBytes)
				if fitCtx >= minCtx && fitCtx > bestCtx {
					est, ok := inkling2IntentFit(hp, fitCtx, concurrency, dt, ch, 0, hw.FreeVRAMBytes)
					if ok {
						bestCtx = fitCtx
						chosenCtx, chosenDType, chosenChunk, chosenEst, found = fitCtx, dt, ch, est, true
					}
				}
			}
		}
	}

	// Phase 3: even minCtx (or expert-pinned ctx) exceeds free VRAM budget; report honest spill.
	if !found {
		chosenDType = dtypes[len(dtypes)-1]
		chosenChunk = chunks[len(chunks)-1]
		chosenCtx = minCtx
		chosenEst, _ = inkling2IntentFit(hp, chosenCtx, concurrency, chosenDType, chosenChunk, 0, hw.FreeVRAMBytes)
		budget := hw.FreeVRAMBytes - inkling2DeviceHeadroom(hw.FreeVRAMBytes)
		spillLayers = inkling2EstimateSpillLayers(hp, chosenEst-budget, hw.FreeVRAMBytes)
	}

	// Phase 4: speculative decode admission.
	enableSpec := false
	if ov.Spec != nil {
		enableSpec = *ov.Spec
		if enableSpec {
			est, ok := inkling2IntentFit(hp, chosenCtx, concurrency, chosenDType, chosenChunk, inkling2DSparkDraftReserveBytes, hw.FreeVRAMBytes)
			chosenEst = est
			if !ok {
				budget := hw.FreeVRAMBytes - inkling2DeviceHeadroom(hw.FreeVRAMBytes)
				spillLayers = inkling2EstimateSpillLayers(hp, chosenEst-budget, hw.FreeVRAMBytes)
			}
		}
	} else if defaults.preferSpec && hw.HasDSparkDraft && spillLayers == 0 {
		if est, ok := inkling2IntentFit(hp, chosenCtx, concurrency, chosenDType, chosenChunk, inkling2DSparkDraftReserveBytes, hw.FreeVRAMBytes); ok {
			enableSpec = true
			chosenEst = est
		}
	}

	plan := ResolvedPlan{
		Profile:          profile,
		Ctx:              chosenCtx,
		Concurrency:      concurrency,
		KVDType:          chosenDType,
		PrefillChunk:     chosenChunk,
		EnableSpec:       enableSpec,
		SpillLayers:      spillLayers,
		EstimatedBytes:   chosenEst,
		HeadroomBytes:    hw.FreeVRAMBytes - chosenEst,
		OverridesApplied: overrides,
	}
	plan.Summary = formatInkling2ResolvedPlanSummary(plan, hw)
	return plan, nil
}

func formatInkling2ResolvedPlanSummary(p ResolvedPlan, hw HardwareEnvelope) string {
	vramStr := formatInkling2GiBCompact(hw.FreeVRAMBytes)
	archStr := "cpu/metal"
	if hw.SMArch > 0 {
		archStr = fmt.Sprintf("sm_%d", hw.SMArch)
	}
	ctxStr := fmt.Sprintf("%d", p.Ctx)
	if p.Ctx >= 1024 && p.Ctx%1024 == 0 {
		ctxStr = fmt.Sprintf("%dk", p.Ctx/1024)
	}
	if p.Concurrency > 1 {
		ctxStr = fmt.Sprintf("%s x%d", ctxStr, p.Concurrency)
	}
	residency := "all-resident"
	if p.SpillLayers > 0 {
		residency = "spilled"
	}
	specStr := ""
	if p.EnableSpec {
		specStr = ", spec=on"
	}
	ovStr := ""
	if len(p.OverridesApplied) > 0 {
		ovStr = fmt.Sprintf(" [overrides: %s]", strings.Join(p.OverridesApplied, ", "))
	}
	return fmt.Sprintf("%s profile (%s %s): %s %s via %s KV, chunk=%d%s, %d spill%s",
		p.Profile, vramStr, archStr, ctxStr, residency, p.KVDType, p.PrefillChunk, specStr, p.SpillLayers, ovStr)
}

func formatInkling2GiBCompact(b int64) string {
	gib := float64(b) / float64(int64(1)<<30)
	if math.Abs(gib-math.Round(gib)) < 0.05 {
		return fmt.Sprintf("%.0f GiB", math.Round(gib))
	}
	return fmt.Sprintf("%.1f GiB", gib)
}
