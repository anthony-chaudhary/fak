package ggufload

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// Inkling2ServeEnvelope holds the solved serving parameters for an Inkling-2
// model on a given device memory budget and capability set.
type Inkling2ServeEnvelope = ServeEnvelope

// Inkling2ServeCaps describes the backend kernel and runtime capabilities
// available for Inkling-2 serving.
type Inkling2ServeCaps = ServeCaps

// Inkling2ServeOverrides carries explicit CLI flag and FAK_INKLING2_* env
// overrides that act as hard pins over solved envelope defaults.
type Inkling2ServeOverrides = ServeOverrides

// inkling2DSparkDraftReserveBytes is the conservative device footprint reserved
// for the DSpark speculative draft weights, tap buffers, and verify logits when
// speculative decoding is enabled.
const inkling2DSparkDraftReserveBytes int64 = 384 << 20

// inkling2MemoryBudget adapts model.Inkling2LoopHParams and the exact
// inkling2loop_resident.go memory formulas to ModelMemoryBudget.
type inkling2MemoryBudget struct {
	hp model.Inkling2LoopHParams
}

func (b inkling2MemoryBudget) Validate() error {
	if b.hp.Layers <= 0 || b.hp.KVHeads <= 0 || b.hp.HeadDim <= 0 || len(b.hp.LocalLayer) != b.hp.Layers {
		return fmt.Errorf("inkling2 intent: invalid model hparams (layers=%d kv_heads=%d head_dim=%d local_len=%d)",
			b.hp.Layers, b.hp.KVHeads, b.hp.HeadDim, len(b.hp.LocalLayer))
	}
	return nil
}

func (b inkling2MemoryBudget) Layers() int {
	return b.hp.Layers
}

func (b inkling2MemoryBudget) DeviceHeadroom(freeBytes int64) int64 {
	return inkling2DeviceHeadroom(freeBytes)
}

func (b inkling2MemoryBudget) ResidentKVBytes(ctx int, batched bool, chunk int, dtype ServeKVDType) int64 {
	return inkling2ResidentKVBytesFor(b.hp, ctx, batched, chunk, dtype)
}

func (b inkling2MemoryBudget) ResidentOtherBytes(chunk int) int64 {
	return inkling2ResidentOtherBytes(b.hp, inkling2ArenaFloats, chunk, 0)
}

func (b inkling2MemoryBudget) KVFit(ctx int, batched bool, chunk int, dtype ServeKVDType, other, freeBytes int64) (int64, error) {
	return inkling2KVFitFor(b.hp, ctx, batched, chunk, dtype, other, freeBytes)
}

func (b inkling2MemoryBudget) MaxResidentCtx(batched bool, chunk int, dtype ServeKVDType, ceil int, budget int64) int {
	return inkling2MaxResidentCtxFor(b.hp, batched, chunk, dtype, ceil, budget)
}

func (b inkling2MemoryBudget) BatchNeedBytes(C, ctx, chunk int, kvDtype ServeKVDType) int64 {
	return inkling2EnvelopeBatchNeedBytes(b.hp, C, ctx, chunk, kvDtype)
}

func (b inkling2MemoryBudget) EnvelopeConstraints(backendName string) ServeEnvelopeConstraints {
	return ServeEnvelopeConstraints{
		DefaultCtx:             Inkling2LoopResidentContext,
		MinCtx:                 Inkling2ResidentCtxMin,
		AutoCtxCeiling:         Inkling2AutoCtxCeiling,
		ResidentCtxFitMax:      Inkling2ResidentCtxFitMax,
		HardMaxCtx:             inkling2ResidentHardMax(backendName),
		MaxBatch:               Inkling2LoopMaxBatch,
		PrefillChunkBase:       inkling2PrefillChunkBase,
		PrefillChunkWide:       inkling2PrefillChunkWide,
		PrefillChunkMetal:      inkling2PrefillChunkMetal,
		MetalPrefillChunkSteps: inkling2MetalPrefillChunkSteps,
		BatchChunkLadder:       inkling2BatchChunkLadder,
		BatchHeadroom:          inkling2BatchHeadroom,
		BatchCtxAlign:          inkling2BatchCtxAlign,
		DraftReserveBytes:      inkling2DSparkDraftReserveBytes,
		SupportsBatchedPrefill: b.hp.Hidden > 0,
	}
}

// ResolveInkling2ServeEnvelope computes the unified VRAM-envelope parameters
// from exact inkling2BatchBytes and inkling2ResidentKVBytesFor math, honoring
// explicit overrides as hard pins.
func ResolveInkling2ServeEnvelope(hp model.Inkling2LoopHParams, freeBytes int64, caps Inkling2ServeCaps, overrides Inkling2ServeOverrides) Inkling2ServeEnvelope {
	return ResolveServeEnvelope(inkling2MemoryBudget{hp: hp}, freeBytes, caps, overrides)
}

// inkling2EnvelopeBatchFits reports whether C concurrent sequences at ctx and
// chunk fit within freeBytes including batch headroom.
func inkling2EnvelopeBatchFits(hp model.Inkling2LoopHParams, C, ctx, chunk int, kvDtype Inkling2KVDType, freeBytes int64) bool {
	b := inkling2MemoryBudget{hp: hp}
	return serveEnvelopeBatchFits(b, b.EnvelopeConstraints(""), C, ctx, chunk, kvDtype, freeBytes)
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
	dt, ok := ParseServeKVDTypeToken(v)
	if !ok || dt == Inkling2KVAsym {
		return Inkling2KVF32, false
	}
	return dt, true
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
