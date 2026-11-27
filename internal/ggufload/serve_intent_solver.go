package ggufload

import (
	"fmt"
	"math"
	"strings"
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
	KVDType      *ServeKVDType
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

// ResolvedPlan is the reconciled execution plan produced by ResolveServeIntent.
type ResolvedPlan struct {
	Profile          ServeProfile
	Ctx              int
	Concurrency      int
	KVDType          ServeKVDType
	PrefillChunk     int
	EnableSpec       bool
	SpillLayers      int
	EstimatedBytes   int64
	HeadroomBytes    int64
	OverridesApplied []string
	Summary          string
}

type serveProfileDefaults struct {
	concurrency int
	minCtx      int
	targetCtx   int
	preferSpec  bool
}

// servePrefillChunkLadder is the ordered prefill chunk descent tried when
// fitting a target context into free VRAM without spilling.
var servePrefillChunkLadder = []int{512, 256, 128, 64}

func resolveServeProfileSpec(p ServeProfile) (ServeProfile, serveProfileDefaults, error) {
	if p == "" {
		p = ServeProfileCoding
	}
	switch p {
	case ServeProfileCoding:
		return p, serveProfileDefaults{concurrency: 1, minCtx: 65536, targetCtx: 131072, preferSpec: true}, nil
	case ServeProfileChat, ServeProfileDemo:
		return p, serveProfileDefaults{concurrency: 1, minCtx: 16384, targetCtx: 32768, preferSpec: true}, nil
	case ServeProfileBatch:
		return p, serveProfileDefaults{concurrency: 4, minCtx: 16384, targetCtx: 32768, preferSpec: false}, nil
	case ServeProfileLongDoc:
		return p, serveProfileDefaults{concurrency: 1, minCtx: 131072, targetCtx: 262144, preferSpec: true}, nil
	default:
		return "", serveProfileDefaults{}, fmt.Errorf("serve intent: unknown serve profile %q", p)
	}
}

func hardwareSupportsServeKVDType(hw HardwareEnvelope, dtype ServeKVDType) bool {
	switch dtype {
	case ServeKVF32:
		return true
	case ServeKVF16:
		return hw.HasFP16
	case ServeKVFP8:
		return hw.HasFP8
	case ServeKVNVFP4:
		return hw.HasNVFP4
	default:
		return false
	}
}

func serveIntentFit(budget ModelMemoryBudget, ctx, concurrency int, dtype ServeKVDType, chunk int, specBytes, free int64) (estimated int64, fits bool) {
	batched := chunk > 0
	other := budget.ResidentOtherBytes(chunk) + specBytes
	var extraSlots int64
	if concurrency > 1 {
		extraSlots = int64(concurrency-1) * budget.ResidentKVBytes(ctx, batched, chunk, dtype)
	}
	kv, err := budget.KVFit(ctx, batched, chunk, dtype, other+extraSlots, free)
	return kv + other + extraSlots, err == nil
}

func serveIntentMaxCtx(budget ModelMemoryBudget, concurrency int, dtype ServeKVDType, chunk, ceil int, specBytes, free int64) int {
	c := budget.EnvelopeConstraints("")
	avail := free - budget.DeviceHeadroom(free)
	other := budget.ResidentOtherBytes(chunk) + specBytes
	left := avail - other
	if left <= 0 || concurrency <= 0 {
		return 0
	}
	perSlotBudget := left / int64(concurrency)
	n := budget.MaxResidentCtx(chunk > 0, chunk, dtype, ceil, perSlotBudget)
	n = n / 256 * 256
	if n < c.MinCtx {
		return 0
	}
	return n
}

func estimateServeSpillLayers(layersCount int, overBytes, free int64) int {
	if overBytes <= 0 || layersCount <= 0 {
		return 0
	}
	perLayer := max(int64(256<<20), free/int64(layersCount))
	layers := int((overBytes + perLayer - 1) / perLayer)
	return min(layersCount, max(1, layers))
}

// ResolveServeIntent reconciles a declarative ServeIntent against budget and hw
// into a concrete ResolvedPlan.
func ResolveServeIntent(budget ModelMemoryBudget, intent ServeIntent, hw HardwareEnvelope) (ResolvedPlan, error) {
	if budget == nil {
		return ResolvedPlan{}, fmt.Errorf("serve intent: model memory budget is required")
	}
	if err := budget.Validate(); err != nil {
		return ResolvedPlan{}, err
	}
	if hw.FreeVRAMBytes <= 0 {
		return ResolvedPlan{}, fmt.Errorf("serve intent: free VRAM must be positive (got %d)", hw.FreeVRAMBytes)
	}

	profile, defaults, err := resolveServeProfileSpec(intent.Profile)
	if err != nil {
		return ResolvedPlan{}, err
	}

	c := budget.EnvelopeConstraints("")

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

	var dtypes []ServeKVDType
	if ov.KVDType != nil {
		dt := *ov.KVDType
		if !hardwareSupportsServeKVDType(hw, dt) {
			return ResolvedPlan{}, fmt.Errorf("serve intent: expert override kv_dtype=%s is not supported by hardware (sm_%d)", dt, hw.SMArch)
		}
		dtypes = []ServeKVDType{dt}
		overrides = append(overrides, fmt.Sprintf("kv_dtype=%s", dt))
	} else {
		if hw.HasFP16 {
			dtypes = append(dtypes, ServeKVF16)
		} else {
			dtypes = append(dtypes, ServeKVF32)
		}
		if intent.AllowExperimentalKV {
			if hw.HasFP8 {
				dtypes = append(dtypes, ServeKVFP8)
			}
			if hw.HasNVFP4 {
				dtypes = append(dtypes, ServeKVNVFP4)
			}
		}
	}

	var chunks []int
	if ov.PrefillChunk > 0 {
		chunks = []int{ov.PrefillChunk}
		overrides = append(overrides, fmt.Sprintf("prefill_chunk=%d", ov.PrefillChunk))
	} else {
		chunks = servePrefillChunkLadder
	}

	if ov.Spec != nil {
		overrides = append(overrides, fmt.Sprintf("spec=%t", *ov.Spec))
	}

	var (
		chosenCtx   int
		chosenDType ServeKVDType
		chosenChunk int
		chosenEst   int64
		spillLayers int
		found       bool
	)

	// Phase 1: reach targetCtx with 0 spill, preferring highest KV precision then widest prefill chunk.
	for _, dt := range dtypes {
		for _, ch := range chunks {
			if est, ok := serveIntentFit(budget, targetCtx, concurrency, dt, ch, 0, hw.FreeVRAMBytes); ok {
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
				fitCtx := serveIntentMaxCtx(budget, concurrency, dt, ch, targetCtx, 0, hw.FreeVRAMBytes)
				if fitCtx >= minCtx && fitCtx > bestCtx {
					est, ok := serveIntentFit(budget, fitCtx, concurrency, dt, ch, 0, hw.FreeVRAMBytes)
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
		chosenEst, _ = serveIntentFit(budget, chosenCtx, concurrency, chosenDType, chosenChunk, 0, hw.FreeVRAMBytes)
		avail := hw.FreeVRAMBytes - budget.DeviceHeadroom(hw.FreeVRAMBytes)
		spillLayers = estimateServeSpillLayers(budget.Layers(), chosenEst-avail, hw.FreeVRAMBytes)
	}

	// Phase 4: speculative decode admission.
	enableSpec := false
	draftReserve := c.DraftReserveBytes
	if ov.Spec != nil {
		enableSpec = *ov.Spec
		if enableSpec {
			est, ok := serveIntentFit(budget, chosenCtx, concurrency, chosenDType, chosenChunk, draftReserve, hw.FreeVRAMBytes)
			chosenEst = est
			if !ok {
				avail := hw.FreeVRAMBytes - budget.DeviceHeadroom(hw.FreeVRAMBytes)
				spillLayers = estimateServeSpillLayers(budget.Layers(), chosenEst-avail, hw.FreeVRAMBytes)
			}
		}
	} else if defaults.preferSpec && hw.HasDSparkDraft && spillLayers == 0 {
		if est, ok := serveIntentFit(budget, chosenCtx, concurrency, chosenDType, chosenChunk, draftReserve, hw.FreeVRAMBytes); ok {
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
	plan.Summary = formatResolvedPlanSummary(plan, hw)
	return plan, nil
}

func formatResolvedPlanSummary(p ResolvedPlan, hw HardwareEnvelope) string {
	vramStr := formatServeGiBCompact(hw.FreeVRAMBytes)
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

func formatServeGiBCompact(b int64) string {
	gib := float64(b) / float64(int64(1)<<30)
	if math.Abs(gib-math.Round(gib)) < 0.05 {
		return fmt.Sprintf("%.0f GiB", math.Round(gib))
	}
	return fmt.Sprintf("%.1f GiB", gib)
}
