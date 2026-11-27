package ggufload

import (
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const (
	// inkling2EnvelopeWideChunkFreeBytes is the post-weight free device memory
	// threshold above which auto-tuning selects DefaultChunk=1024 instead of 512.
	inkling2EnvelopeWideChunkFreeBytes = int64(6) << 30

	// inkling2EnvelopeMinChunk is the lowest prefill chunk rung used for
	// step-down under VRAM pressure and for LongCtxC1Threshold sizing.
	inkling2EnvelopeMinChunk = 128
)

// Inkling2ServeEnvelope holds the solved serving parameters for an Inkling-2
// model on a given device memory budget and capability set.
type Inkling2ServeEnvelope struct {
	MaxCtx             int
	DefaultChunk       int
	MinChunk           int
	TargetConcurrency  int
	KVDtype            Inkling2KVDType
	LongCtxC1Threshold int
}

// Inkling2ServeCaps describes the backend kernel and runtime capabilities
// available for Inkling-2 serving.
type Inkling2ServeCaps struct {
	BackendName      string
	BatchEligible    bool
	F16Ready         bool
	FP8Ready         bool
	FP8KLQualified   bool
	NVFP4Ready       bool
	NVFP4KLQualified bool
	SpecActive       bool
}

// Inkling2ServeOverrides carries explicit CLI flag and FAK_INKLING2_* env
// overrides that act as hard pins over solved envelope defaults.
type Inkling2ServeOverrides struct {
	RequestedCtx   int              // --native-context-tokens
	EnvCtx         int              // FAK_INKLING2_CTX
	KVDtype        *Inkling2KVDType // FAK_INKLING2_BATCH_KV or FAK_INKLING2_KV
	PrefillChunk   int              // FAK_INKLING2_PREFILL_CHUNK
	ServeBatchRows int              // FAK_INKLING2_SERVE_BATCH_ROWS
	BatchDisabled  bool             // FAK_INKLING2_SERVE_BATCH=0/false/off or FAK_INKLING2_BATCH_SEQOPS=0
}

// ResolveInkling2ServeEnvelope computes the unified VRAM-envelope parameters
// from exact inkling2BatchBytes and inkling2ResidentKVBytesFor math, honoring
// explicit overrides as hard pins.
func ResolveInkling2ServeEnvelope(hp model.Inkling2LoopHParams, freeBytes int64, caps Inkling2ServeCaps, overrides Inkling2ServeOverrides) Inkling2ServeEnvelope {
	kvDtype := resolveEnvelopeKVDtype(caps, overrides)

	knownFree := freeBytes > 0 && freeBytes != compute.FreeUnknown
	defaultChunk, minChunk := resolveEnvelopeChunk(hp, freeBytes, knownFree, caps, overrides, kvDtype)
	maxCtx := resolveEnvelopeMaxCtx(hp, freeBytes, knownFree, caps, overrides, kvDtype, defaultChunk)
	targetC := resolveEnvelopeConcurrency(hp, freeBytes, knownFree, caps, overrides, kvDtype, defaultChunk, maxCtx)
	c1Thresh := resolveEnvelopeLongCtxC1Threshold(hp, freeBytes, knownFree, caps, overrides, kvDtype, minChunk, maxCtx)

	return Inkling2ServeEnvelope{
		MaxCtx:             maxCtx,
		DefaultChunk:       defaultChunk,
		MinChunk:           minChunk,
		TargetConcurrency:  targetC,
		KVDtype:            kvDtype,
		LongCtxC1Threshold: c1Thresh,
	}
}

func resolveEnvelopeKVDtype(caps Inkling2ServeCaps, overrides Inkling2ServeOverrides) Inkling2KVDType {
	if overrides.KVDtype != nil {
		return *overrides.KVDtype
	}
	if caps.SpecActive {
		return Inkling2KVF32
	}
	if caps.NVFP4Ready && caps.NVFP4KLQualified {
		return Inkling2KVNVFP4
	}
	if caps.FP8Ready && caps.FP8KLQualified {
		return Inkling2KVFP8
	}
	if caps.F16Ready {
		return Inkling2KVF16
	}
	return Inkling2KVF32
}

func resolveEnvelopeChunk(hp model.Inkling2LoopHParams, freeBytes int64, knownFree bool, caps Inkling2ServeCaps, overrides Inkling2ServeOverrides, kvDtype Inkling2KVDType) (defChunk, minChunk int) {
	if overrides.PrefillChunk > 0 {
		return overrides.PrefillChunk, min(overrides.PrefillChunk, inkling2EnvelopeMinChunk)
	}
	if caps.BackendName == "metal" {
		if !knownFree {
			return inkling2PrefillChunkMetal, inkling2PrefillChunkBase
		}
		for _, c := range inkling2MetalPrefillChunkSteps {
			if inkling2EnvelopeBatchFits(hp, 1, Inkling2LoopResidentContext, c, kvDtype, freeBytes) {
				return c, inkling2PrefillChunkBase
			}
		}
		return inkling2PrefillChunkBase, inkling2PrefillChunkBase
	}
	if !knownFree {
		return inkling2PrefillChunkBase, inkling2EnvelopeMinChunk
	}
	want := inkling2PrefillChunkBase
	if freeBytes >= inkling2EnvelopeWideChunkFreeBytes {
		want = inkling2PrefillChunkWide
	}
	for _, c := range inkling2BatchChunkLadder(want) {
		if inkling2EnvelopeBatchFits(hp, 1, Inkling2LoopResidentContext, c, kvDtype, freeBytes) {
			return c, inkling2EnvelopeMinChunk
		}
	}
	return inkling2EnvelopeMinChunk, inkling2EnvelopeMinChunk
}

func resolveEnvelopeMaxCtx(hp model.Inkling2LoopHParams, freeBytes int64, knownFree bool, caps Inkling2ServeCaps, overrides Inkling2ServeOverrides, kvDtype Inkling2KVDType, defaultChunk int) int {
	if overrides.RequestedCtx > 0 {
		if hard := inkling2ResidentHardMax(caps.BackendName); hard > 0 && overrides.RequestedCtx > hard {
			return hard
		}
		return overrides.RequestedCtx
	}
	if overrides.EnvCtx > 0 {
		return overrides.EnvCtx
	}
	if !knownFree {
		return Inkling2LoopResidentContext
	}
	ceiling := Inkling2AutoCtxCeiling
	if kvDtype != Inkling2KVF32 && freeBytes >= 8<<30 {
		ceiling = Inkling2ResidentCtxFitMax
	}
	if hard := inkling2ResidentHardMax(caps.BackendName); hard > 0 && ceiling > hard {
		ceiling = hard
	}

	batched := caps.BatchEligible && !overrides.BatchDisabled
	prefillChunk := 0
	if batched && hp.Hidden > 0 {
		prefillChunk = max(defaultChunk, inkling2PrefillChunkBase)
	}
	other := inkling2ResidentOtherBytes(hp, inkling2ArenaFloats, prefillChunk, 0)
	budget := freeBytes - inkling2DeviceHeadroom(freeBytes)
	left := budget - other
	var fit int
	if left > 0 {
		fit = inkling2MaxResidentCtxFor(hp, batched, prefillChunk, kvDtype, ceiling, left)
		fit = fit / 256 * 256
	}
	if fit < Inkling2ResidentCtxMin {
		return Inkling2LoopResidentContext
	}
	return max(Inkling2LoopResidentContext, min(fit, ceiling))
}

func resolveEnvelopeConcurrency(hp model.Inkling2LoopHParams, freeBytes int64, knownFree bool, caps Inkling2ServeCaps, overrides Inkling2ServeOverrides, kvDtype Inkling2KVDType, defaultChunk, maxCtx int) int {
	if overrides.ServeBatchRows > 0 {
		return min(overrides.ServeBatchRows, Inkling2LoopMaxBatch)
	}
	if !caps.BatchEligible || overrides.BatchDisabled {
		return 1
	}
	if !knownFree {
		return 1
	}
	refCtx := min(maxCtx, Inkling2AutoCtxCeiling)
	best := 1
	for c := Inkling2LoopMaxBatch; c >= 2; c-- {
		if inkling2EnvelopeBatchFits(hp, c, refCtx, defaultChunk, kvDtype, freeBytes) {
			best = c
			break
		}
	}
	return best
}

func resolveEnvelopeLongCtxC1Threshold(hp model.Inkling2LoopHParams, freeBytes int64, knownFree bool, caps Inkling2ServeCaps, overrides Inkling2ServeOverrides, kvDtype Inkling2KVDType, minChunk, maxCtx int) int {
	if !caps.BatchEligible || overrides.BatchDisabled {
		return 0
	}
	if !knownFree {
		return maxCtx
	}
	if inkling2EnvelopeBatchFits(hp, 2, maxCtx, minChunk, kvDtype, freeBytes) {
		return maxCtx
	}
	lo, hi := 0, maxCtx
	for lo < hi {
		mid := lo + (hi-lo+1)/2
		if inkling2EnvelopeBatchFits(hp, 2, mid, minChunk, kvDtype, freeBytes) {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return lo / inkling2BatchCtxAlign * inkling2BatchCtxAlign
}

// inkling2EnvelopeBatchNeedBytes prices C concurrent sequences at ctx tokens and
// prefill chunk using inkling2BatchBytes and inkling2ResidentKVBytesFor.
func inkling2EnvelopeBatchNeedBytes(hp model.Inkling2LoopHParams, C, ctx, chunk int, kvDtype Inkling2KVDType) int64 {
	if ctx <= 0 {
		ctx = 1
	}
	keep, scr := inkling2PrefillArenaFloats(hp, C, 0)
	baseCaps := inkling2ResidentCaps(hp, 0, true, chunk)
	caps := inkling2BatchCaps(hp, baseCaps, ctx, chunk)
	half := kvDtype != Inkling2KVF32
	need := inkling2BatchBytes(hp, caps, C, keep, scr, half)
	if kvDtype == Inkling2KVFP8 || kvDtype == Inkling2KVNVFP4 {
		delta := inkling2ResidentKVBytesFor(hp, ctx, true, chunk, kvDtype) - inkling2ResidentKVBytesFor(hp, ctx, true, chunk, Inkling2KVF16)
		need += int64(C) * delta
	}
	if chunk > C {
		pKeep, pScr := inkling2PrefillArenaFloats(hp, chunk, 0)
		need += 4 * (int64(pKeep) + int64(pScr))
	}
	return need
}

func inkling2EnvelopeBatchFits(hp model.Inkling2LoopHParams, C, ctx, chunk int, kvDtype Inkling2KVDType, freeBytes int64) bool {
	if freeBytes <= 0 || freeBytes == compute.FreeUnknown {
		return false
	}
	need := inkling2EnvelopeBatchNeedBytes(hp, C, ctx, chunk, kvDtype)
	return need+inkling2BatchHeadroom <= freeBytes
}

// Inkling2ServeCapsFromBackend inspects m, be, and lookup to build Inkling2ServeCaps.
func Inkling2ServeCapsFromBackend(m *Inkling2LoopModel, be compute.Backend, lookup func(string) (string, bool)) Inkling2ServeCaps {
	if lookup == nil {
		lookup = os.LookupEnv
	}
	var caps Inkling2ServeCaps
	if be != nil {
		caps.BackendName = be.Name()
	}
	if m != nil && inkling2SpecDepthFor(m) > 0 {
		caps.SpecActive = true
	}
	if be == nil {
		return caps
	}
	if _, ok := be.(compute.Inkling2PrefillDevice); ok {
		caps.BatchEligible = Inkling2LoopBatchEligible(m, be)
	} else if _, ok := be.(compute.Inkling2BatchDevice); ok {
		caps.BatchEligible = true
	}

	batchDisabled := inkling2ServeEnvOptOut(lookup, "FAK_INKLING2_SERVE_BATCH") ||
		inkling2ServeEnvOptOut(lookup, "FAK_INKLING2_BATCH_SEQOPS")
	if !batchDisabled {
		seq := inkling2BatchSeqOps(be, lookup)
		if half, err := inkling2BatchKVHalfFor(be, seq, lookup); err == nil && half {
			caps.F16Ready = true
		}
	} else if dtype, err := Inkling2KVDTypeFrom(lookup); err == nil && dtype == Inkling2KVF16 {
		if inkling2BackendHalfKVErr(be) == nil {
			caps.F16Ready = true
		}
	}

	if inkling2BackendFP8KVErr(be) == nil {
		caps.FP8Ready = true
	}
	if inkling2BackendNVFP4KVErr(be) == nil {
		caps.NVFP4Ready = true
	}
	return caps
}

// Inkling2ServeOverridesFromLookup parses CLI requestedCtx and FAK_INKLING2_*
// environment overrides into Inkling2ServeOverrides.
func Inkling2ServeOverridesFromLookup(requestedCtx int, lookup func(string) (string, bool)) Inkling2ServeOverrides {
	if lookup == nil {
		lookup = os.LookupEnv
	}
	var ov Inkling2ServeOverrides
	ov.RequestedCtx = requestedCtx
	if !inkling2ResidentCtxAutoSet(lookup) {
		ov.EnvCtx = inkling2ResidentCtxFrom(lookup)
	}
	if v, ok := lookup("FAK_INKLING2_BATCH_KV"); ok && strings.TrimSpace(v) != "" {
		if dt, ok := inkling2ParseKVDTypeToken(v); ok {
			ov.KVDtype = &dt
		}
	} else if v, ok := lookup("FAK_INKLING2_KV"); ok && strings.TrimSpace(v) != "" {
		if dt, ok := inkling2ParseKVDTypeToken(v); ok {
			ov.KVDtype = &dt
		}
	}
	if v, ok := lookup(inkling2PrefillChunkEnv); ok && strings.TrimSpace(v) != "" {
		if n, err := inkling2PrefillChunkParse(v, true); err == nil {
			ov.PrefillChunk = n
		}
	}
	if v, ok := lookup("FAK_INKLING2_SERVE_BATCH_ROWS"); ok && strings.TrimSpace(v) != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n >= 1 {
			ov.ServeBatchRows = min(n, Inkling2LoopMaxBatch)
		}
	}
	ov.BatchDisabled = inkling2ServeEnvOptOut(lookup, "FAK_INKLING2_SERVE_BATCH") ||
		inkling2ServeEnvOptOut(lookup, "FAK_INKLING2_BATCH_SEQOPS")
	return ov
}

// ResolveInkling2ServeEnvelopeFor resolves the serving envelope for m on be
// using device free memory, backend capabilities, and explicit overrides.
func ResolveInkling2ServeEnvelopeFor(m *Inkling2LoopModel, be compute.Backend, requestedCtx int, lookup func(string) (string, bool)) Inkling2ServeEnvelope {
	if lookup == nil {
		lookup = os.LookupEnv
	}
	var hp model.Inkling2LoopHParams
	if m != nil {
		hp = m.HP
	}
	free := int64(compute.FreeUnknown)
	if be != nil {
		if _, f, known := compute.DeviceMemoryInfo(be); known && f > 0 {
			free = f
		}
	}
	caps := Inkling2ServeCapsFromBackend(m, be, lookup)
	ov := Inkling2ServeOverridesFromLookup(requestedCtx, lookup)
	return ResolveInkling2ServeEnvelope(hp, free, caps, ov)
}

func inkling2ParseKVDTypeToken(v string) (Inkling2KVDType, bool) {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "nvfp4", "fp4", "e2m1":
		return Inkling2KVNVFP4, true
	case "fp8", "e4m3", "e4m3fn":
		return Inkling2KVFP8, true
	case "f16", "fp16":
		return Inkling2KVF16, true
	case "f32", "fp32":
		return Inkling2KVF32, true
	default:
		return Inkling2KVF32, false
	}
}

func inkling2ServeEnvOptOut(lookup func(string) (string, bool), key string) bool {
	v, ok := lookup(key)
	if !ok {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "0", "false", "off":
		return true
	}
	return false
}
