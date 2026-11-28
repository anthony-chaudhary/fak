package ggufload

import (
	"github.com/anthony-chaudhary/fak/internal/compute"
)

const (
	// serveEnvelopeWideChunkFreeBytes is the post-weight free device memory
	// threshold above which auto-tuning selects DefaultChunk=1024 instead of 512.
	serveEnvelopeWideChunkFreeBytes = int64(6) << 30

	// serveEnvelopeMinChunk is the lowest prefill chunk rung used for
	// step-down under VRAM pressure and for LongCtxC1Threshold sizing.
	serveEnvelopeMinChunk = 128
)

// ModelMemoryBudget abstracts model-specific VRAM and KV-cache byte accounting
// for the model-agnostic serving envelope and intent solvers.
type ModelMemoryBudget interface {
	// Validate checks that the underlying model hyperparameters are valid for planning.
	Validate() error
	// Layers returns the number of transformer layers for spill estimation.
	Layers() int
	// DeviceHeadroom returns the bytes reserved for allocator/runtime headroom on freeBytes.
	DeviceHeadroom(freeBytes int64) int64
	// ResidentKVBytes returns the KV-cache bytes for one sequence of ctx tokens.
	ResidentKVBytes(ctx int, batched bool, chunk int, dtype ServeKVDType) int64
	// ResidentOtherBytes returns non-KV resident device bytes (arenas, local rings, etc.) for chunk.
	ResidentOtherBytes(chunk int) int64
	// KVFit checks whether a single sequence at ctx with other non-KV bytes fits in freeBytes.
	KVFit(ctx int, batched bool, chunk int, dtype ServeKVDType, other, freeBytes int64) (int64, error)
	// MaxResidentCtx returns the largest context in [0, ceil] fitting in budget bytes.
	MaxResidentCtx(batched bool, chunk int, dtype ServeKVDType, ceil int, budget int64) int
	// BatchNeedBytes returns total device bytes needed for concurrency sequences at ctx and chunk.
	BatchNeedBytes(concurrency, ctx, chunk int, dtype ServeKVDType) int64
	// EnvelopeConstraints returns model-family bounds and defaults for envelope/intent solving.
	EnvelopeConstraints(backendName string) ServeEnvelopeConstraints
}

// ServeEnvelopeConstraints carries model-family bounds and defaults consumed by
// ResolveServeEnvelope and ResolveServeIntent.
type ServeEnvelopeConstraints struct {
	DefaultCtx             int
	MinCtx                 int
	AutoCtxCeiling         int
	ResidentCtxFitMax      int
	HardMaxCtx             int
	MaxBatch               int
	PrefillChunkBase       int
	PrefillChunkWide       int
	PrefillChunkMetal      int
	MetalPrefillChunkSteps []int
	BatchChunkLadder       func(want int) []int
	BatchHeadroom          int64
	BatchCtxAlign          int
	DraftReserveBytes      int64
	SupportsBatchedPrefill bool
}

// ServeEnvelope holds the solved serving parameters for a model on a given
// device memory budget and capability set.
type ServeEnvelope struct {
	MaxCtx             int
	DefaultChunk       int
	MinChunk           int
	TargetConcurrency  int
	KVDtype            ServeKVDType
	LongCtxC1Threshold int
}

// ServeCaps describes the backend kernel and runtime capabilities available
// for serving.
type ServeCaps struct {
	BackendName      string
	BatchEligible    bool
	F16Ready         bool
	FP8Ready         bool
	FP8KLQualified   bool
	NVFP4Ready       bool
	NVFP4KLQualified bool
	SpecActive       bool
}

// ServeOverrides carries explicit CLI flag and env overrides that act as hard
// pins over solved envelope defaults.
type ServeOverrides struct {
	RequestedCtx   int           // --native-context-tokens
	EnvCtx         int           // model-specific env ctx
	KVDtype        *ServeKVDType // explicit KV dtype override
	PrefillChunk   int           // explicit prefill chunk override
	ServeBatchRows int           // explicit batch rows override
	BatchDisabled  bool          // explicit batch opt-out
}

// ResolveServeEnvelope computes the unified VRAM-envelope parameters from
// budget, free device memory, backend capabilities, and explicit overrides.
func ResolveServeEnvelope(budget ModelMemoryBudget, freeBytes int64, caps ServeCaps, overrides ServeOverrides) ServeEnvelope {
	c := budget.EnvelopeConstraints(caps.BackendName)
	kvDtype := resolveServeEnvelopeKVDtype(caps, overrides)

	knownFree := freeBytes > 0 && freeBytes != compute.FreeUnknown
	defaultChunk, minChunk := resolveServeEnvelopeChunk(budget, c, freeBytes, knownFree, caps, overrides, kvDtype)
	maxCtx := resolveServeEnvelopeMaxCtx(budget, c, freeBytes, knownFree, caps, overrides, kvDtype, defaultChunk)
	targetC := resolveServeEnvelopeConcurrency(budget, c, freeBytes, knownFree, caps, overrides, kvDtype, defaultChunk, maxCtx)
	c1Thresh := resolveServeEnvelopeLongCtxC1Threshold(budget, c, freeBytes, knownFree, caps, overrides, kvDtype, minChunk, maxCtx)

	return ServeEnvelope{
		MaxCtx:             maxCtx,
		DefaultChunk:       defaultChunk,
		MinChunk:           minChunk,
		TargetConcurrency:  targetC,
		KVDtype:            kvDtype,
		LongCtxC1Threshold: c1Thresh,
	}
}

func resolveServeEnvelopeKVDtype(caps ServeCaps, overrides ServeOverrides) ServeKVDType {
	if overrides.KVDtype != nil {
		return *overrides.KVDtype
	}
	if caps.SpecActive {
		return ServeKVF32
	}
	if caps.NVFP4Ready && caps.NVFP4KLQualified {
		return ServeKVNVFP4
	}
	if caps.FP8Ready && caps.FP8KLQualified {
		return ServeKVFP8
	}
	if caps.F16Ready {
		return ServeKVF16
	}
	return ServeKVF32
}

func resolveServeEnvelopeChunk(budget ModelMemoryBudget, c ServeEnvelopeConstraints, freeBytes int64, knownFree bool, caps ServeCaps, overrides ServeOverrides, kvDtype ServeKVDType) (defChunk, minChunk int) {
	if overrides.PrefillChunk > 0 {
		return overrides.PrefillChunk, min(overrides.PrefillChunk, serveEnvelopeMinChunk)
	}
	if caps.BackendName == "metal" {
		if !knownFree {
			return c.PrefillChunkMetal, c.PrefillChunkBase
		}
		for _, chunk := range c.MetalPrefillChunkSteps {
			if serveEnvelopeBatchFits(budget, c, 1, c.DefaultCtx, chunk, kvDtype, freeBytes) {
				return chunk, c.PrefillChunkBase
			}
		}
		return c.PrefillChunkBase, c.PrefillChunkBase
	}
	if !knownFree {
		return c.PrefillChunkBase, serveEnvelopeMinChunk
	}
	want := c.PrefillChunkBase
	if freeBytes >= serveEnvelopeWideChunkFreeBytes {
		want = c.PrefillChunkWide
	}
	ladder := []int{want}
	if c.BatchChunkLadder != nil {
		ladder = c.BatchChunkLadder(want)
	}
	for _, chunk := range ladder {
		if serveEnvelopeBatchFits(budget, c, 1, c.DefaultCtx, chunk, kvDtype, freeBytes) {
			return chunk, serveEnvelopeMinChunk
		}
	}
	return serveEnvelopeMinChunk, serveEnvelopeMinChunk
}

func resolveServeEnvelopeMaxCtx(budget ModelMemoryBudget, c ServeEnvelopeConstraints, freeBytes int64, knownFree bool, caps ServeCaps, overrides ServeOverrides, kvDtype ServeKVDType, defaultChunk int) int {
	if overrides.RequestedCtx > 0 {
		if c.HardMaxCtx > 0 && overrides.RequestedCtx > c.HardMaxCtx {
			return c.HardMaxCtx
		}
		return overrides.RequestedCtx
	}
	if overrides.EnvCtx > 0 {
		return overrides.EnvCtx
	}
	if !knownFree {
		return c.DefaultCtx
	}
	ceiling := c.AutoCtxCeiling
	if kvDtype != ServeKVF32 && freeBytes >= 8<<30 {
		ceiling = c.ResidentCtxFitMax
	}
	if c.HardMaxCtx > 0 && ceiling > c.HardMaxCtx {
		ceiling = c.HardMaxCtx
	}

	batched := caps.BatchEligible && !overrides.BatchDisabled
	prefillChunk := 0
	if batched && c.SupportsBatchedPrefill {
		prefillChunk = max(defaultChunk, c.PrefillChunkBase)
	}
	other := budget.ResidentOtherBytes(prefillChunk)
	avail := freeBytes - budget.DeviceHeadroom(freeBytes)
	left := avail - other
	var fit int
	if left > 0 {
		fit = budget.MaxResidentCtx(batched, prefillChunk, kvDtype, ceiling, left)
		fit = fit / 256 * 256
	}
	if fit < c.MinCtx {
		return c.DefaultCtx
	}
	return max(c.DefaultCtx, min(fit, ceiling))
}

func resolveServeEnvelopeConcurrency(budget ModelMemoryBudget, c ServeEnvelopeConstraints, freeBytes int64, knownFree bool, caps ServeCaps, overrides ServeOverrides, kvDtype ServeKVDType, defaultChunk, maxCtx int) int {
	if overrides.ServeBatchRows > 0 {
		return min(overrides.ServeBatchRows, c.MaxBatch)
	}
	if !caps.BatchEligible || overrides.BatchDisabled {
		return 1
	}
	if !knownFree {
		return 1
	}
	refCtx := min(maxCtx, c.AutoCtxCeiling)
	best := 1
	for conc := c.MaxBatch; conc >= 2; conc-- {
		if serveEnvelopeBatchFits(budget, c, conc, refCtx, defaultChunk, kvDtype, freeBytes) {
			best = conc
			break
		}
	}
	return best
}

func resolveServeEnvelopeLongCtxC1Threshold(budget ModelMemoryBudget, c ServeEnvelopeConstraints, freeBytes int64, knownFree bool, caps ServeCaps, overrides ServeOverrides, kvDtype ServeKVDType, minChunk, maxCtx int) int {
	if !caps.BatchEligible || overrides.BatchDisabled {
		return 0
	}
	if !knownFree {
		return maxCtx
	}
	if serveEnvelopeBatchFits(budget, c, 2, maxCtx, minChunk, kvDtype, freeBytes) {
		return maxCtx
	}
	lo, hi := 0, maxCtx
	for lo < hi {
		mid := lo + (hi-lo+1)/2
		if serveEnvelopeBatchFits(budget, c, 2, mid, minChunk, kvDtype, freeBytes) {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	if c.BatchCtxAlign > 0 {
		return lo / c.BatchCtxAlign * c.BatchCtxAlign
	}
	return lo
}

func serveEnvelopeBatchFits(budget ModelMemoryBudget, c ServeEnvelopeConstraints, concurrency, ctx, chunk int, kvDtype ServeKVDType, freeBytes int64) bool {
	if freeBytes <= 0 || freeBytes == compute.FreeUnknown {
		return false
	}
	need := budget.BatchNeedBytes(concurrency, ctx, chunk, kvDtype)
	return need+c.BatchHeadroom <= freeBytes
}
