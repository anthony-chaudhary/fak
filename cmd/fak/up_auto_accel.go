package main

import (
	"fmt"
	"io"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/engine/specutil"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/macfit"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

const (
	// maxTurnkeySingleNodeGPUs caps single-node auto-acceleration at 1–8 GPUs;
	// multi-node / cluster / RDMA fabrics are explicitly out of scope for `fak up`.
	maxTurnkeySingleNodeGPUs = 8

	upAccelStatusActive   = "active"
	upAccelStatusDisabled = "disabled"
)

// Test seams for deterministic single-node GPU topology and MTP qualification tests.
var (
	upDetectGPUStats          = compute.SystemGPUStats
	upDetectUnifiedMemory     = macfit.DetectUnifiedMemory
	upSpecQualificationMatrix = specutil.DefaultMatrix
	upLoadModelPlacedFn       = loadServeInKernelModelPlacedErr
)

// upGPUTopology summarizes single-node 1–8 GPU VRAM or Apple Silicon UMA capacity.
type upGPUTopology struct {
	Mode           string            `json:"mode"`
	DeviceCount    int               `json:"device_count"`
	DetectedCount  int               `json:"detected_count"`
	ClampedFrom    int               `json:"clamped_from,omitempty"`
	TotalVRAMBytes uint64            `json:"total_vram_bytes,omitempty"`
	UsedVRAMBytes  uint64            `json:"used_vram_bytes,omitempty"`
	FreeVRAMBytes  uint64            `json:"free_vram_bytes,omitempty"`
	Devices        []compute.GPUStat `json:"devices,omitempty"`
	UnifiedMemory  bool              `json:"unified_memory,omitempty"`
	EffectiveBytes uint64            `json:"effective_bytes"`
	OverrideSource string            `json:"override_source,omitempty"`
}

// upAccelerationFeature captures the live activation status and concrete rationale
// for one auto-acceleration subsystem.
type upAccelerationFeature struct {
	Status string `json:"status"`
	Reason string `json:"reason"`
	Detail string `json:"detail,omitempty"`
}

// upAccelerationEnvelope summarizes the resolved ggufload ServeIntent + ServeEnvelope.
type upAccelerationEnvelope struct {
	Profile           string `json:"profile"`
	KVDType           string `json:"kv_dtype"`
	MaxCtx            int    `json:"max_ctx"`
	TargetConcurrency int    `json:"target_concurrency"`
	DefaultChunk      int    `json:"default_chunk"`
	LongCtxC1Cutoff   int    `json:"long_ctx_c1_cutoff,omitempty"`
	SpillLayers       int    `json:"spill_layers,omitempty"`
	IntentSummary     string `json:"intent_summary,omitempty"`
}

// upAccelerationReport backs both the startup acceleration banner and /healthz "acceleration".
type upAccelerationReport struct {
	Mode             string                           `json:"mode"`
	Backend          string                           `json:"backend"`
	GPUCount         int                              `json:"gpu_count"`
	GPUDetectedCount int                              `json:"gpu_detected_count,omitempty"`
	TotalVRAMBytes   uint64                           `json:"total_vram_bytes,omitempty"`
	FreeVRAMBytes    uint64                           `json:"free_vram_bytes,omitempty"`
	Envelope         *upAccelerationEnvelope          `json:"envelope,omitempty"`
	Features         map[string]upAccelerationFeature `json:"features"`
}

func upEnvFalse(name string) bool {
	raw := strings.TrimSpace(strings.ToLower(os.Getenv(name)))
	switch raw {
	case "0", "false", "off", "no", "disabled":
		return true
	}
	return false
}

func upScavengeLoaderHeapEnabled() (bool, string) {
	raw := strings.TrimSpace(os.Getenv("FAK_UP_SCAVENGE_LOADER_HEAP"))
	if raw == "1" || strings.EqualFold(raw, "true") {
		return true, fmt.Sprintf("debug.FreeOSMemory enabled via FAK_UP_SCAVENGE_LOADER_HEAP=%s", raw)
	}
	if raw == "" {
		return false, "disabled: opt-in only until startup performance is qualified (set FAK_UP_SCAVENGE_LOADER_HEAP=1 or true)"
	}
	return false, fmt.Sprintf("disabled: opt-in requires FAK_UP_SCAVENGE_LOADER_HEAP=1 or true (got %q)", raw)
}

func upFitShrinkEnabled() (bool, string) {
	if raw := strings.TrimSpace(os.Getenv("FAK_UP_FIT_SHRINK")); raw != "" && upEnvFalse("FAK_UP_FIT_SHRINK") {
		return false, fmt.Sprintf("disabled via FAK_UP_FIT_SHRINK=%s", raw)
	}
	return true, "runServeFitCommit adaptive context/allocation retry active by default"
}

// resolveTurnkeyTopologyAndMemory discovers 1–8 single-node discrete GPUs via
// compute.SystemGPUStats / compute.AggregateGPUStats, falling back to Apple Silicon
// unified memory on darwin/arm64 or host RAM on CPU-only nodes. Explicit
// --memory-gib or FAK_UP_MEMORY_BYTES overrides are always honored.
func resolveTurnkeyTopologyAndMemory(memoryGiB float64) upGPUTopology {
	topo := upGPUTopology{
		UnifiedMemory: runtime.GOOS == "darwin" && runtime.GOARCH == "arm64",
	}

	if stats, ok := upDetectGPUStats(); ok && len(stats) > 0 {
		valid := make([]compute.GPUStat, 0, len(stats))
		for _, st := range stats {
			if st.VRAMTotalBytes > 0 {
				valid = append(valid, st)
			}
		}
		if len(valid) > 0 {
			topo.DetectedCount = len(valid)
			if len(valid) > maxTurnkeySingleNodeGPUs {
				topo.ClampedFrom = len(valid)
				valid = valid[:maxTurnkeySingleNodeGPUs]
			}
			if used, total, _, aggOK := compute.AggregateGPUStats(valid); aggOK && total > 0 {
				topo.DeviceCount = len(valid)
				topo.Devices = valid
				topo.TotalVRAMBytes = total
				topo.UsedVRAMBytes = used
				if total > used {
					topo.FreeVRAMBytes = total - used
				}
			}
		}
	}

	if memoryGiB > 0 {
		topo.EffectiveBytes = uint64(memoryGiB * float64(macfit.GiB))
		topo.OverrideSource = "flag:--memory-gib"
		switch {
		case topo.UnifiedMemory:
			topo.Mode = "apple-silicon-uma"
		case topo.DeviceCount > 0:
			topo.Mode = "single-node-gpu"
		default:
			topo.Mode = "host-ram-fallback"
		}
		return topo
	}

	if raw := strings.TrimSpace(os.Getenv("FAK_UP_MEMORY_BYTES")); raw != "" {
		mem, err := upDetectUnifiedMemory()
		if err != nil || mem == 0 {
			mem = 16 * macfit.GiB
		}
		topo.EffectiveBytes = mem
		topo.OverrideSource = "env:FAK_UP_MEMORY_BYTES"
		switch {
		case topo.UnifiedMemory:
			topo.Mode = "apple-silicon-uma"
		case topo.DeviceCount > 0:
			topo.Mode = "single-node-gpu"
		default:
			topo.Mode = "host-ram-fallback"
		}
		return topo
	}

	if topo.UnifiedMemory {
		mem, err := upDetectUnifiedMemory()
		if err != nil || mem == 0 {
			mem = 16 * macfit.GiB
		}
		topo.Mode = "apple-silicon-uma"
		topo.EffectiveBytes = mem
		return topo
	}

	if topo.DeviceCount > 0 && topo.TotalVRAMBytes > 0 {
		topo.Mode = "single-node-gpu"
		topo.EffectiveBytes = topo.TotalVRAMBytes
		return topo
	}

	mem, err := upDetectUnifiedMemory()
	if err != nil || mem == 0 {
		mem = 16 * macfit.GiB
	}
	topo.Mode = "host-ram-fallback"
	topo.EffectiveBytes = mem
	return topo
}

func turnkeyTopologyFitBudget(topo upGPUTopology, planMemoryBytes uint64) *serveFitBudget {
	if topo.OverrideSource == "" && topo.Mode == "single-node-gpu" && topo.FreeVRAMBytes > 0 {
		budget := planMemoryBytes
		if topo.FreeVRAMBytes < budget {
			budget = topo.FreeVRAMBytes
		}
		return turnkeyStableFitBudget(budget)
	}
	return turnkeyStableFitBudget(planMemoryBytes)
}

func upFormatGiB(b uint64) string {
	return fmt.Sprintf("%.1f GiB", float64(b)/float64(macfit.GiB))
}

func upTopologyFeature(topo upGPUTopology) upAccelerationFeature {
	switch {
	case topo.Mode == "single-node-gpu" && topo.DeviceCount > 0:
		msg := fmt.Sprintf("single-node %d GPU topology, %s total VRAM (%s free)",
			topo.DeviceCount, upFormatGiB(topo.TotalVRAMBytes), upFormatGiB(topo.FreeVRAMBytes))
		if topo.ClampedFrom > 0 {
			msg = fmt.Sprintf("%s; clamped from %d detected GPUs to 1-%d single-node limit", msg, topo.ClampedFrom, maxTurnkeySingleNodeGPUs)
		}
		if topo.OverrideSource != "" {
			msg = fmt.Sprintf("%s (memory budget overridden via %s)", msg, topo.OverrideSource)
		}
		return upAccelerationFeature{
			Status: upAccelStatusActive,
			Reason: msg,
			Detail: fmt.Sprintf("%dgpu/%s", topo.DeviceCount, upFormatGiB(topo.TotalVRAMBytes)),
		}
	case topo.Mode == "apple-silicon-uma" || topo.UnifiedMemory:
		msg := fmt.Sprintf("Apple Silicon unified memory (%s)", upFormatGiB(topo.EffectiveBytes))
		if topo.OverrideSource != "" {
			msg = fmt.Sprintf("%s via %s", msg, topo.OverrideSource)
		}
		return upAccelerationFeature{
			Status: upAccelStatusActive,
			Reason: msg,
			Detail: fmt.Sprintf("uma/%s", upFormatGiB(topo.EffectiveBytes)),
		}
	default:
		reason := fmt.Sprintf("no discrete GPUs (0) or Apple Silicon UMA detected; sized from host RAM (%s)", upFormatGiB(topo.EffectiveBytes))
		if topo.OverrideSource != "" {
			reason = fmt.Sprintf("no discrete GPUs detected; sized via %s (%s)", topo.OverrideSource, upFormatGiB(topo.EffectiveBytes))
		}
		return upAccelerationFeature{
			Status: upAccelStatusDisabled,
			Reason: reason,
			Detail: fmt.Sprintf("host/%s", upFormatGiB(topo.EffectiveBytes)),
		}
	}
}

// turnkeyModelMemoryBudget adapts a turnkey tier or loaded model config into
// ggufload.ModelMemoryBudget for ResolveServeIntent / ResolveServeEnvelope.
type turnkeyModelMemoryBudget struct {
	layers        int
	weightsBytes  int64
	kvBytesPerTok int64 // baseline F32 bytes per token across full-attention KV layers
	headroomRatio float64
}

func newTurnkeyModelMemoryBudget(plan macfit.TurnkeyProfile, m *fakmodel.Model) turnkeyModelMemoryBudget {
	headroom := plan.HeadroomRatio
	if headroom <= 0 || headroom >= 0.5 {
		headroom = 0.15
	}
	weights := int64(plan.Tier.WeightBytes)
	if weights <= 0 {
		weights = 4 << 30
	}
	if m != nil && m.Cfg.NumLayers > 0 {
		kvHeads := m.Cfg.NumKVHeads
		if kvHeads <= 0 {
			kvHeads = m.Cfg.NumHeads
		}
		if kvHeads <= 0 {
			kvHeads = 8
		}
		headDim := m.Cfg.HeadDim
		if headDim <= 0 {
			headDim = 128
		}
		kvLayers := m.Cfg.NumLayers
		if interval := m.Cfg.FullAttentionInterval; interval > 1 && kvLayers >= interval {
			kvLayers = kvLayers / interval
			if kvLayers < 1 {
				kvLayers = 1
			}
		}
		// K + V tensors stored at baseline F32 (4 bytes/elem) before KV dtype scaling.
		perTok := int64(kvLayers) * int64(kvHeads) * int64(headDim) * 2 * 4
		if perTok <= 0 {
			perTok = 16384
		}
		return turnkeyModelMemoryBudget{
			layers:        m.Cfg.NumLayers,
			weightsBytes:  weights,
			kvBytesPerTok: perTok,
			headroomRatio: headroom,
		}
	}
	layers := 32
	perTok := int64(64 * 1024)
	switch strings.ToUpper(plan.Tier.Name) {
	case "70B":
		layers = 80
		perTok = 320 * 1024
	case "27B":
		layers = 64
		perTok = 128 * 1024
	case "7B":
		layers = 36
		perTok = 48 * 1024
	case "3B":
		layers = 36
		perTok = 32 * 1024
	}
	return turnkeyModelMemoryBudget{
		layers:        layers,
		weightsBytes:  weights,
		kvBytesPerTok: perTok,
		headroomRatio: headroom,
	}
}

func (b turnkeyModelMemoryBudget) Validate() error {
	if b.layers <= 0 || b.kvBytesPerTok <= 0 {
		return fmt.Errorf("invalid turnkey model memory budget: layers=%d kvPerTok=%d", b.layers, b.kvBytesPerTok)
	}
	return nil
}

func (b turnkeyModelMemoryBudget) Layers() int { return b.layers }

func (b turnkeyModelMemoryBudget) DeviceHeadroom(freeBytes int64) int64 {
	if freeBytes <= 0 {
		return 0
	}
	hr := int64(float64(freeBytes) * b.headroomRatio)
	const minReserve = 256 << 20
	if freeBytes > 4*minReserve && hr < minReserve {
		return minReserve
	}
	return hr
}

func turnkeyKVDTypeScale(dtype ggufload.ServeKVDType) (num, den int64) {
	switch dtype {
	case ggufload.Inkling2KVF16:
		return 1, 2
	case ggufload.Inkling2KVFP8:
		return 1, 4
	case ggufload.Inkling2KVNVFP4:
		return 1, 8
	default:
		// F32 baseline (or unrecognized conservative fallback).
		return 1, 1
	}
}

func (b turnkeyModelMemoryBudget) ResidentKVBytes(ctx int, batched bool, chunk int, dtype ggufload.ServeKVDType) int64 {
	if ctx <= 0 {
		return 0
	}
	num, den := turnkeyKVDTypeScale(dtype)
	effCtx := int64(ctx)
	if batched && chunk > 0 {
		effCtx += int64(chunk)
	}
	return (effCtx * b.kvBytesPerTok * num) / den
}

func (b turnkeyModelMemoryBudget) ResidentOtherBytes(chunk int) int64 {
	scratch := int64(128 << 20)
	if chunk > 512 {
		scratch += int64(chunk-512) * 128 * 1024
	}
	return b.weightsBytes + scratch
}

func (b turnkeyModelMemoryBudget) KVFit(ctx int, batched bool, chunk int, dtype ggufload.ServeKVDType, other, freeBytes int64) (int64, error) {
	need := other + b.ResidentKVBytes(ctx, batched, chunk, dtype) + b.DeviceHeadroom(freeBytes)
	if freeBytes > 0 && need > freeBytes {
		return need, fmt.Errorf("need %d bytes exceeds free %d bytes", need, freeBytes)
	}
	return need, nil
}

func (b turnkeyModelMemoryBudget) MaxResidentCtx(batched bool, chunk int, dtype ggufload.ServeKVDType, ceil int, budget int64) int {
	if budget <= 0 || ceil <= 0 {
		return 0
	}
	avail := budget - b.ResidentOtherBytes(chunk) - b.DeviceHeadroom(budget)
	if avail <= 0 {
		return 0
	}
	num, den := turnkeyKVDTypeScale(dtype)
	perTok := (b.kvBytesPerTok * num) / den
	if perTok <= 0 {
		return ceil
	}
	tokens := int(avail / perTok)
	if batched && chunk > 0 {
		tokens -= chunk
	}
	if tokens < 0 {
		return 0
	}
	if tokens > ceil {
		return ceil
	}
	return tokens
}

func (b turnkeyModelMemoryBudget) BatchNeedBytes(C, ctx, chunk int, kvDtype ggufload.ServeKVDType) int64 {
	if C < 1 {
		C = 1
	}
	return b.ResidentOtherBytes(chunk) + int64(C)*b.ResidentKVBytes(ctx, true, chunk, kvDtype)
}

func (b turnkeyModelMemoryBudget) EnvelopeConstraints(backendName string) ggufload.ServeEnvelopeConstraints {
	return ggufload.ServeEnvelopeConstraints{
		DefaultCtx:             4096,
		MinCtx:                 512,
		AutoCtxCeiling:         32768,
		ResidentCtxFitMax:      131072,
		HardMaxCtx:             131072,
		MaxBatch:               32,
		PrefillChunkBase:       512,
		PrefillChunkWide:       2048,
		PrefillChunkMetal:      512,
		MetalPrefillChunkSteps: []int{2048, 1024, 512},
		BatchCtxAlign:          256,
		SupportsBatchedPrefill: true,
	}
}

func upFreeMemoryForEnvelope(topo upGPUTopology, backend compute.Backend, fallbackBytes uint64) int64 {
	if backend != nil {
		if _, free, ok := compute.DeviceMemoryInfo(backend); ok && free > 0 {
			return free
		}
	}
	if topo.Mode == "single-node-gpu" && topo.FreeVRAMBytes > 0 {
		return int64(topo.FreeVRAMBytes)
	}
	if topo.EffectiveBytes > 0 {
		return int64(topo.EffectiveBytes)
	}
	if fallbackBytes > 0 {
		return int64(fallbackBytes)
	}
	return 16 << 30
}

func upCUDAArchitectureSM(backend compute.Backend) (compute.CUDADeviceReceipt, int, bool) {
	if backend == nil {
		return compute.CUDADeviceReceipt{}, 0, false
	}
	rcpt, ok := compute.CUDADeviceReceiptOf(backend)
	if !ok {
		return compute.CUDADeviceReceipt{}, 0, false
	}
	sm := 0
	if trimmed := strings.TrimPrefix(strings.ToLower(strings.TrimSpace(rcpt.Tier)), "sm_"); trimmed != "" {
		if v, err := strconv.Atoi(trimmed); err == nil {
			sm = v
		}
	}
	return rcpt, sm, true
}

func solveTurnkeyServeEnvelope(plan macfit.TurnkeyProfile, kvPrec fakmodel.KVPrecision, topo upGPUTopology, backend compute.Backend, m *fakmodel.Model, mtpAdmitted bool) (*upAccelerationEnvelope, error) {
	backendName := "cpu"
	if backend != nil && strings.TrimSpace(backend.Name()) != "" {
		backendName = backend.Name()
	}
	budget := newTurnkeyModelMemoryBudget(plan, m)
	freeBytes := upFreeMemoryForEnvelope(topo, backend, plan.MemoryBytes)

	_, sm, hasCUDA := upCUDAArchitectureSM(backend)
	hw := ggufload.HardwareEnvelope{
		FreeVRAMBytes:  freeBytes,
		SMArch:         sm,
		HasFP16:        backendName != "cpu" && backendName != "cpu-ref",
		HasFP8:         hasCUDA && sm >= 89,
		HasNVFP4:       hasCUDA && sm >= 100,
		HasDSparkDraft: mtpAdmitted,
	}
	intent := ggufload.ServeIntent{
		Profile: ggufload.ServeProfileCoding,
	}
	if kvPrec == fakmodel.KVPrecisionQ8_0 {
		dt := ggufload.Inkling2KVFP8
		intent.ExpertOverrides.KVDType = &dt
		intent.AllowExperimentalKV = true
	}

	resolvedPlan, err := ggufload.ResolveServeIntent(budget, intent, hw)
	if err != nil {
		return nil, err
	}

	caps := ggufload.ServeCaps{
		BackendName:    backendName,
		BatchEligible:  true,
		F16Ready:       hw.HasFP16,
		FP8Ready:       hw.HasFP8,
		FP8KLQualified: hw.HasFP8,
		NVFP4Ready:     hw.HasNVFP4,
	}
	var ov ggufload.ServeOverrides
	if kvPrec == fakmodel.KVPrecisionQ8_0 {
		dt := ggufload.Inkling2KVFP8
		ov.KVDtype = &dt
	}
	env := ggufload.ResolveServeEnvelope(budget, freeBytes, caps, ov)

	maxCtx := resolvedPlan.Ctx
	if env.MaxCtx > 0 && (maxCtx == 0 || env.MaxCtx < maxCtx) {
		maxCtx = env.MaxCtx
	}
	conc := env.TargetConcurrency
	if resolvedPlan.Concurrency > conc {
		conc = resolvedPlan.Concurrency
	}
	return &upAccelerationEnvelope{
		Profile:           string(resolvedPlan.Profile),
		KVDType:           resolvedPlan.KVDType.String(),
		MaxCtx:            maxCtx,
		TargetConcurrency: conc,
		DefaultChunk:      env.DefaultChunk,
		LongCtxC1Cutoff:   env.LongCtxC1Threshold,
		SpillLayers:       resolvedPlan.SpillLayers,
		IntentSummary:     resolvedPlan.Summary,
	}, nil
}

type upFittedLoadResult struct {
	Model           *fakmodel.Model
	Q4K             bool
	Profile         *gateway.ModelLoadProfile
	Attempts        int
	Shrunk          bool
	EffectiveCtx    int
	ShrinkReason    string
	MTPDroppedOnOOM bool
}

func loadTurnkeyModelWithFitCommit(modelPath string, backend compute.Backend, contextTokens int, fit *serveFitBudget, retainMTP bool, minContextTokens int) (upFittedLoadResult, error) {
	res := upFittedLoadResult{
		Attempts:     1,
		EffectiveCtx: contextTokens,
	}
	tryLoadOnce := func(ctxTok int, fitBudget *serveFitBudget, attempts *int) (*fakmodel.Model, bool, *gateway.ModelLoadProfile, gateway.StartupPhase, compute.MemoryPlan, error) {
		*attempts++
		m, q4k, profile, phase, plan, loadErr := upLoadModelPlacedFn(modelPath, backend, false, false, ctxTok, nil, 1, fitBudget)
		if loadErr != nil && retainMTP && !res.MTPDroppedOnOOM && serveNoteMTPRetentionFallback(loadErr, os.Getenv) {
			*attempts++
			res.MTPDroppedOnOOM = true
			m, q4k, profile, phase, plan, loadErr = upLoadModelPlacedFn(modelPath, backend, false, false, ctxTok, nil, 1, fitBudget)
		}
		return m, q4k, profile, phase, plan, loadErr
	}

	enabled, _ := upFitShrinkEnabled()
	if !enabled {
		attempts := 0
		m, q4k, profile, _, _, err := tryLoadOnce(contextTokens, fit, &attempts)
		res.Attempts = attempts
		res.Model, res.Q4K, res.Profile = m, q4k, profile
		return res, err
	}

	if minContextTokens <= 0 || minContextTokens > contextTokens {
		minContextTokens = contextTokens
		if minContextTokens > 2048 {
			minContextTokens = 2048
		}
	}
	var sizing serveFitBudget
	if fit != nil {
		sizing = *fit
	}
	cfg := serveFitCommitConfig{
		OriginalTokens: contextTokens,
		SizingFit:      sizing,
		DeviceFit:      fit,
		Arm:            upBackendLabel(backend),
		Log:            logServeFitCommitAttempt,
	}

	var attempts int
	lastCtx := contextTokens
	var commitRes serveFitCommitResult
	var err error
	if sizing.Base > 0 {
		commitRes, err = runServeFitCommit(cfg, func(l serveFitCommitLayout) (serveFitCommitResult, error) {
			lastCtx = l.ContextTokens
			if lastCtx < minContextTokens {
				lastCtx = minContextTokens
			}
			fitBudget := l.DeviceFit
			if fitBudget == nil && l.SizingFit.Base > 0 {
				copied := l.SizingFit
				fitBudget = &copied
			}
			m, q4k, profile, phase, plan, loadErr := tryLoadOnce(lastCtx, fitBudget, &attempts)
			return serveFitCommitResult{Model: m, Q4K: q4k, Profile: profile, Phase: phase, Plan: plan}, loadErr
		})
	} else {
		m, q4k, profile, phase, plan, loadErr := tryLoadOnce(contextTokens, fit, &attempts)
		commitRes = serveFitCommitResult{Model: m, Q4K: q4k, Profile: profile, Phase: phase, Plan: plan}
		err = loadErr
	}
	if err != nil && serveLoadFailureIsAllocation(err) {
		for tryCtx := lastCtx / 2; tryCtx >= minContextTokens && tryCtx < lastCtx; tryCtx /= 2 {
			lastCtx = tryCtx
			m, q4k, profile, phase, plan, loadErr := tryLoadOnce(tryCtx, fit, &attempts)
			commitRes = serveFitCommitResult{Model: m, Q4K: q4k, Profile: profile, Phase: phase, Plan: plan}
			err = loadErr
			if err == nil || !serveLoadFailureIsAllocation(err) {
				break
			}
		}
	}
	if attempts > 0 {
		res.Attempts = attempts
	}
	res.EffectiveCtx = lastCtx
	if err != nil {
		return res, err
	}
	res.Model, res.Q4K, res.Profile = commitRes.Model, commitRes.Q4K, commitRes.Profile
	if res.EffectiveCtx != contextTokens || attempts > 1 {
		res.Shrunk = res.EffectiveCtx != contextTokens
		res.ShrinkReason = fmt.Sprintf("fit-shrink retry converged in %d attempt(s) (%d -> %d context tokens)", res.Attempts, contextTokens, res.EffectiveCtx)
	}
	return res, nil
}

func upBackendLabel(be compute.Backend) string {
	if be == nil || strings.TrimSpace(be.Name()) == "" {
		return "cpu"
	}
	return be.Name()
}

func upConfigureCUDAGraphAndMMQ(backend compute.Backend, m *fakmodel.Model, q4k bool, contextTokens int, freeBytes int64) (graphActive bool, graphReason string, mmqArm string, mmqActive bool, mmqReason string) {
	bName := upBackendLabel(backend)
	if bName != "cuda" {
		graphReason = fmt.Sprintf("backend is %s (CUDA graph decode capture requires CUDA)", bName)
		mmqReason = fmt.Sprintf("backend is %s (CUDA MMQ prefill kernels require CUDA)", bName)
		return false, graphReason, "", false, mmqReason
	}

	if upEnvFalse("FAK_CUDA_GRAPH") {
		graphReason = fmt.Sprintf("disabled via FAK_CUDA_GRAPH=%s", strings.TrimSpace(os.Getenv("FAK_CUDA_GRAPH")))
	} else {
		compute.EnableCUDAGraph()
		if m != nil {
			sizing := fakmodel.CUDAGraphKVSizingFor(m.Cfg, backend, true, contextTokens, freeBytes)
			if sizing.Eligible && sizing.Prealloc.Preallocate {
				compute.SetCUDAGraphKVCapacity(contextTokens)
				graphActive = true
				graphReason = fmt.Sprintf("CUDA graph decode capture active (kv_positions=%d, rows=%d)", contextTokens, sizing.KVRows)
			} else {
				graphReason = fmt.Sprintf("CUDA graph runtime enabled, model KV prealloc skipped (%s)", sizing.Prealloc.Reason)
			}
		} else {
			graphActive = true
			graphReason = "CUDA graph runtime capture enabled"
		}
	}

	rcpt, hasRcpt := compute.CUDADeviceReceiptOf(backend)
	_, sm, _ := upCUDAArchitectureSM(backend)
	arm := string(compute.DefaultCUDAQ4KPrefillArm(sm))
	mmqArm = arm
	if upEnvFalse("FAK_CUDA_MMQ") {
		mmqReason = fmt.Sprintf("disabled via FAK_CUDA_MMQ=%s", strings.TrimSpace(os.Getenv("FAK_CUDA_MMQ")))
		return graphActive, graphReason, arm, false, mmqReason
	}
	if !q4k {
		mmqReason = "loaded checkpoint is not resident Q4_K"
		return graphActive, graphReason, arm, false, mmqReason
	}
	if arm == "" {
		if hasRcpt && rcpt.Tier != "" {
			mmqReason = fmt.Sprintf("GPU %s (%s) has no qualified Q4K MMQ prefill arm", rcpt.DeviceName, rcpt.Tier)
		} else {
			mmqReason = "CUDA device compute capability unavailable for Q4K MMQ selection"
		}
		return graphActive, graphReason, "", false, mmqReason
	}
	mmqActive = true
	devLabel := "cuda"
	if hasRcpt && rcpt.DeviceName != "" {
		devLabel = fmt.Sprintf("%s (%s)", rcpt.DeviceName, rcpt.Tier)
	}
	mmqReason = fmt.Sprintf("resident Q4_K MMQ prefill arm %q active on %s", arm, devLabel)
	return graphActive, graphReason, arm, mmqActive, mmqReason
}

func upIsSpecQualified(backendName, modelID string, m *fakmodel.Model) bool {
	matrix := upSpecQualificationMatrix()
	if matrix == nil {
		return false
	}
	candidates := []string{modelID, resolveTurnkeyModelRef(modelID)}
	if m != nil && strings.TrimSpace(m.Cfg.Name) != "" {
		candidates = append(candidates, m.Cfg.Name)
	}
	for _, cand := range candidates {
		cand = strings.TrimSpace(cand)
		if cand != "" && matrix.IsQualified(backendName, cand) {
			return true
		}
	}
	return false
}

func upShouldRetainResidentMTP(metal bool, backend compute.Backend, modelID, modelPath string, hasMetalRetainHook bool) (retain bool, specMode string, envForced bool, qualified bool, reason string) {
	var rawEnv string
	for _, name := range []string{"FAK_SPECULATIVE", "SPECULATIVE"} {
		if v := strings.ToLower(strings.TrimSpace(os.Getenv(name))); v != "" {
			rawEnv = v
			break
		}
	}
	if serveSpeculativeOptOutTokens[rawEnv] {
		return false, "off", false, false, "disabled via speculative=off"
	}
	resolved, err := resolveSpeculativeMode("", os.Getenv)
	if err != nil {
		return false, "off", false, false, err.Error()
	}
	envForced = rawEnv != ""
	specMode = resolved
	if specMode == "" {
		specMode = "auto"
	}
	if metal {
		return hasMetalRetainHook, specMode, envForced, hasMetalRetainHook, ""
	}
	bName := upBackendLabel(backend)
	if bName != "cuda" && bName != "vulkan" {
		return false, specMode, envForced, false, fmt.Sprintf("backend is %s (resident MTP requires Metal, CUDA, or Vulkan)", bName)
	}
	qualified = upIsSpecQualified(bName, modelID, nil)
	if !(specMode == serveSpeculativeMTP || (specMode == "auto" && (envForced || qualified))) {
		return false, specMode, envForced, false, fmt.Sprintf("pair (%s, %s) not HW-WITNESSED in specutil qualification matrix (set FAK_SPECULATIVE=mtp to force)", bName, modelID)
	}
	mType := serveMTPPeekModelType(modelPath)
	if mType != "" && !serveMTPArchConsumes(mType) {
		return false, specMode, envForced, qualified, fmt.Sprintf("model architecture %q does not carry an MTP head", mType)
	}
	if ok, replayReason := serveMTPBackendReplayStatus(serveSpeculativeMTP, false, backend); !ok {
		return false, specMode, envForced, qualified, replayReason
	}
	return true, specMode, envForced, qualified, ""
}

func wireTurnkeyMTPSpecAndEvaluate(planner *agent.InKernelPlanner, metal bool, backend compute.Backend, m *fakmodel.Model, modelID, specMode string, envForced, mtpDroppedOnOOM bool, metalAdmitted bool, metalReason string) (active bool, engine, reason string) {
	if metal {
		if metalAdmitted {
			return true, "metal-mtp", "Metal MTP speculative decoding admitted by qualification catalog"
		}
		if strings.TrimSpace(metalReason) == "" {
			metalReason = "Metal MTP not admitted for model/context"
		}
		return false, "", metalReason
	}

	bName := upBackendLabel(backend)
	if bName != "cuda" && bName != "vulkan" {
		return false, "", fmt.Sprintf("backend is %s (resident MTP requires Metal, CUDA, or Vulkan)", bName)
	}
	if specMode == "off" {
		return false, "", "disabled via FAK_SPECULATIVE/SPECULATIVE=off"
	}
	if mtpDroppedOnOOM {
		return false, "", "MTP head dropped during fit-shrink allocation fallback"
	}
	if m == nil || !m.Cfg.HasMTPHead() {
		return false, "", "loaded checkpoint has no embedded MTP weights"
	}
	if ok, replayReason := serveMTPBackendReplayStatus(serveSpeculativeMTP, false, backend); !ok {
		return false, "", replayReason
	}
	qualified := upIsSpecQualified(bName, modelID, m)
	if !envForced && !qualified && specMode != serveSpeculativeMTP {
		return false, "", fmt.Sprintf("pair (%s, %s) not HW-WITNESSED in specutil qualification matrix (set FAK_SPECULATIVE=mtp to force)", bName, modelID)
	}
	if planner == nil {
		return false, "", "in-kernel planner unavailable"
	}
	if err := planner.EnableVulkanMTP(fakmodel.Qwen35MTPMaxDraftDepth); err != nil {
		return false, "", fmt.Sprintf("EnableVulkanMTP failed: %v", err)
	}
	return true, fmt.Sprintf("%s-mtp", bName), fmt.Sprintf("resident %s MTP active (depth=%d, target-verified)", bName, fakmodel.Qwen35MTPMaxDraftDepth)
}

func upEvaluateRadixKVStatus(m *fakmodel.Model, backend compute.Backend) (active bool, reason string) {
	if os.Getenv("FAK_INKERNEL_RADIX") == "off" {
		return false, "disabled via FAK_INKERNEL_RADIX=off"
	}
	if m == nil {
		return false, "in-kernel model not loaded"
	}
	var supported bool
	if backend != nil {
		supported = m.Cfg.InKernelBackendPrefixReuseSupportedFor(backend)
	} else {
		supported = m.Cfg.InKernelBackendPrefixReuseSupported()
	}
	if !supported {
		return false, fmt.Sprintf("prefix KV reuse unsupported for architecture on %s", upBackendLabel(backend))
	}
	return true, "radix KV prefix reuse active by default across turns"
}

func buildTurnkeyAccelerationReport(topo upGPUTopology, plan macfit.TurnkeyProfile, kvPrec fakmodel.KVPrecision, mockMode bool, native *turnkeyNativeResources) upAccelerationReport {
	features := make(map[string]upAccelerationFeature, 10)
	features["topology"] = upTopologyFeature(topo)

	backendName := "cpu"
	if mockMode {
		backendName = "mock"
	} else if native != nil && strings.TrimSpace(native.Startup.BackendName) != "" {
		backendName = native.Startup.BackendName
	}

	var envelope *upAccelerationEnvelope
	if native != nil && native.Startup.Envelope != nil {
		envelope = native.Startup.Envelope
	} else {
		var be compute.Backend
		var m *fakmodel.Model
		var mtp bool
		if native != nil {
			be = native.Backend
			m = native.Model
			mtp = native.Startup.MTPActive
		}
		if env, err := solveTurnkeyServeEnvelope(plan, kvPrec, topo, be, m, mtp); err == nil {
			envelope = env
		}
	}

	if envelope != nil {
		features["serve_envelope"] = upAccelerationFeature{
			Status: upAccelStatusActive,
			Reason: fmt.Sprintf("profile=%s kv=%s max_ctx=%d concurrency=%d chunk=%d",
				envelope.Profile, envelope.KVDType, envelope.MaxCtx, envelope.TargetConcurrency, envelope.DefaultChunk),
			Detail: fmt.Sprintf("ctx=%d/c=%d", envelope.MaxCtx, envelope.TargetConcurrency),
		}
	} else {
		features["serve_envelope"] = upAccelerationFeature{
			Status: upAccelStatusDisabled,
			Reason: "serve envelope solver unavailable for zero memory budget",
		}
	}

	if kvPrec == fakmodel.KVPrecisionQ8_0 {
		features["kv_precision"] = upAccelerationFeature{
			Status: upAccelStatusActive,
			Reason: "q8_0 dense-mixed KV storage active (~2x context capacity)",
			Detail: "q8_0",
		}
	} else {
		features["kv_precision"] = upAccelerationFeature{
			Status: upAccelStatusDisabled,
			Reason: "using exact f32 KV default; pass --kv-precision q8_0 or set FAK_UP_KV_PRECISION=q8_0 for ~2x context",
			Detail: "f32",
		}
	}

	if mockMode || native == nil {
		reason := "mock engine active (in-kernel acceleration inactive)"
		if !mockMode {
			reason = "in-kernel runtime not started (planning / dry-run view)"
			if topo.Mode == "single-node-gpu" || topo.UnifiedMemory {
				scavOn, scavReason := upScavengeLoaderHeapEnabled()
				scavStatus := upAccelStatusActive
				if !scavOn {
					scavStatus = upAccelStatusDisabled
				}
				features["loader_heap_scavenge"] = upAccelerationFeature{Status: scavStatus, Reason: scavReason}
				fitOn, fitReason := upFitShrinkEnabled()
				fitStatus := upAccelStatusActive
				if !fitOn {
					fitStatus = upAccelStatusDisabled
				}
				features["fit_shrink_retry"] = upAccelerationFeature{Status: fitStatus, Reason: fitReason}
				if os.Getenv("FAK_INKERNEL_RADIX") == "off" {
					features["radix_kv_cache"] = upAccelerationFeature{Status: upAccelStatusDisabled, Reason: "disabled via FAK_INKERNEL_RADIX=off"}
				} else {
					features["radix_kv_cache"] = upAccelerationFeature{Status: upAccelStatusActive, Reason: "radix KV prefix reuse active by default"}
				}
				features["batch_coalesce"] = upAccelerationFeature{Status: upAccelStatusActive, Reason: "continuous batched decode active by default"}
				features["speculative_mtp"] = upAccelerationFeature{Status: upAccelStatusDisabled, Reason: "pending runtime backend & qualification check"}
				features["cuda_graph"] = upAccelerationFeature{Status: upAccelStatusDisabled, Reason: "pending CUDA runtime probe"}
				features["cuda_mmq"] = upAccelerationFeature{Status: upAccelStatusDisabled, Reason: "pending CUDA runtime probe"}
				mode := topo.Mode
				if mode == "" {
					mode = "turnkey"
				}
				return upAccelerationReport{
					Mode:             mode,
					Backend:          backendName,
					GPUCount:         topo.DeviceCount,
					GPUDetectedCount: topo.DetectedCount,
					TotalVRAMBytes:   topo.TotalVRAMBytes,
					FreeVRAMBytes:    topo.FreeVRAMBytes,
					Envelope:         envelope,
					Features:         features,
				}
			}
		}
		for _, k := range []string{"fit_shrink_retry", "loader_heap_scavenge", "speculative_mtp", "cuda_graph", "cuda_mmq", "radix_kv_cache", "batch_coalesce"} {
			features[k] = upAccelerationFeature{Status: upAccelStatusDisabled, Reason: reason}
		}
		mode := topo.Mode
		if mockMode {
			mode = "mock"
		}
		return upAccelerationReport{
			Mode:             mode,
			Backend:          backendName,
			GPUCount:         topo.DeviceCount,
			GPUDetectedCount: topo.DetectedCount,
			TotalVRAMBytes:   topo.TotalVRAMBytes,
			FreeVRAMBytes:    topo.FreeVRAMBytes,
			Envelope:         envelope,
			Features:         features,
		}
	}

	st := native.Startup
	if st.FitShrinkEnabled {
		reason := fmt.Sprintf("fit-shrink retry active (%d load attempt(s))", max(1, st.FitShrinkAttempts))
		if st.FitShrinkReason != "" {
			reason = st.FitShrinkReason
		}
		features["fit_shrink_retry"] = upAccelerationFeature{
			Status: upAccelStatusActive,
			Reason: reason,
			Detail: fmt.Sprintf("attempts=%d/ctx=%d", max(1, st.FitShrinkAttempts), max(0, st.EffectiveContextTokens)),
		}
	} else {
		_, r := upFitShrinkEnabled()
		features["fit_shrink_retry"] = upAccelerationFeature{Status: upAccelStatusDisabled, Reason: r}
	}

	if st.HeapScavenged {
		features["loader_heap_scavenge"] = upAccelerationFeature{
			Status: upAccelStatusActive,
			Reason: st.HeapScavengeReason,
		}
	} else {
		reason := st.HeapScavengeReason
		if reason == "" {
			_, reason = upScavengeLoaderHeapEnabled()
		}
		features["loader_heap_scavenge"] = upAccelerationFeature{
			Status: upAccelStatusDisabled,
			Reason: reason,
		}
	}

	if st.MTPActive {
		features["speculative_mtp"] = upAccelerationFeature{
			Status: upAccelStatusActive,
			Reason: st.MTPReason,
			Detail: st.MTPEngine,
		}
	} else {
		reason := st.MTPReason
		if reason == "" {
			reason = "speculative MTP inactive"
		}
		features["speculative_mtp"] = upAccelerationFeature{
			Status: upAccelStatusDisabled,
			Reason: reason,
		}
	}

	if st.CUDAGraphActive {
		features["cuda_graph"] = upAccelerationFeature{
			Status: upAccelStatusActive,
			Reason: st.CUDAGraphReason,
		}
	} else {
		reason := st.CUDAGraphReason
		if reason == "" {
			reason = fmt.Sprintf("backend is %s (CUDA graph requires CUDA)", backendName)
		}
		features["cuda_graph"] = upAccelerationFeature{Status: upAccelStatusDisabled, Reason: reason}
	}

	if st.CUDAMMQActive {
		features["cuda_mmq"] = upAccelerationFeature{
			Status: upAccelStatusActive,
			Reason: st.CUDAMMQReason,
			Detail: st.CUDAMMQArm,
		}
	} else {
		reason := st.CUDAMMQReason
		if reason == "" {
			reason = fmt.Sprintf("backend is %s (CUDA MMQ requires CUDA)", backendName)
		}
		features["cuda_mmq"] = upAccelerationFeature{Status: upAccelStatusDisabled, Reason: reason}
	}

	if st.RadixActive {
		features["radix_kv_cache"] = upAccelerationFeature{
			Status: upAccelStatusActive,
			Reason: st.RadixReason,
		}
	} else {
		reason := st.RadixReason
		if reason == "" {
			reason = "radix KV prefix reuse inactive"
		}
		features["radix_kv_cache"] = upAccelerationFeature{Status: upAccelStatusDisabled, Reason: reason}
	}

	features["batch_coalesce"] = upAccelerationFeature{
		Status: upAccelStatusActive,
		Reason: "continuous batched decode active on in-kernel planner",
	}

	mode := topo.Mode
	if mode == "" {
		mode = "turnkey"
	}
	return upAccelerationReport{
		Mode:             mode,
		Backend:          backendName,
		GPUCount:         topo.DeviceCount,
		GPUDetectedCount: topo.DetectedCount,
		TotalVRAMBytes:   topo.TotalVRAMBytes,
		FreeVRAMBytes:    topo.FreeVRAMBytes,
		Envelope:         envelope,
		Features:         features,
	}
}

var upAccelerationFeatureOrder = []string{
	"topology",
	"serve_envelope",
	"fit_shrink_retry",
	"loader_heap_scavenge",
	"speculative_mtp",
	"kv_precision",
	"cuda_graph",
	"cuda_mmq",
	"radix_kv_cache",
	"batch_coalesce",
}

func printTurnkeyAccelerationSummary(w io.Writer, report upAccelerationReport) {
	if w == nil || len(report.Features) == 0 {
		return
	}
	fmt.Fprintf(w, "Auto-acceleration (mode=%s, backend=%s, gpus=%d):\n", report.Mode, report.Backend, report.GPUCount)
	seen := make(map[string]bool, len(report.Features))
	for _, key := range upAccelerationFeatureOrder {
		f, ok := report.Features[key]
		if !ok {
			continue
		}
		seen[key] = true
		fmt.Fprintf(w, "  [%-8s] %-22s %s\n", f.Status, key+":", f.Reason)
	}
	if len(seen) < len(report.Features) {
		extra := make([]string, 0, len(report.Features)-len(seen))
		for k := range report.Features {
			if !seen[k] {
				extra = append(extra, k)
			}
		}
		sort.Strings(extra)
		for _, k := range extra {
			f := report.Features[k]
			fmt.Fprintf(w, "  [%-8s] %-22s %s\n", f.Status, k+":", f.Reason)
		}
	}
}
