package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/engine/specutil"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/macfit"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

type stubAutoAccelBackend struct {
	compute.Backend
	name       string
	replayPath string
}

func (b *stubAutoAccelBackend) Name() string { return b.name }

func (b *stubAutoAccelBackend) Caps() compute.Caps {
	return compute.Caps{DeviceMemory: b.name == "cuda" || b.name == "vulkan"}
}

func (b *stubAutoAccelBackend) Qwen35SequencePrefixReplayPath() string { return b.replayPath }

// fak-test:runtime fast est=1s lane=default
func TestUpAutoAccelSingleNode1To8GPUTopology(t *testing.T) {
	origGPU := upDetectGPUStats
	origUMA := upDetectUnifiedMemory
	t.Cleanup(func() {
		upDetectGPUStats = origGPU
		upDetectUnifiedMemory = origUMA
	})
	t.Setenv("FAK_UP_MEMORY_BYTES", "")

	// 4 discrete GPUs on a single node aggregate cleanly.
	upDetectGPUStats = func() ([]compute.GPUStat, bool) {
		return []compute.GPUStat{
			{Index: 0, VRAMTotalBytes: 24 << 30, VRAMUsedBytes: 2 << 30},
			{Index: 1, VRAMTotalBytes: 24 << 30, VRAMUsedBytes: 2 << 30},
			{Index: 2, VRAMTotalBytes: 24 << 30, VRAMUsedBytes: 2 << 30},
			{Index: 3, VRAMTotalBytes: 24 << 30, VRAMUsedBytes: 2 << 30},
		}, true
	}
	upDetectUnifiedMemory = func() (uint64, error) { return 32 * macfit.GiB, nil }

	topo4 := resolveTurnkeyTopologyAndMemory(0)
	if topo4.DeviceCount != 4 || topo4.DetectedCount != 4 || topo4.ClampedFrom != 0 {
		t.Fatalf("unexpected 4-GPU topology: %+v", topo4)
	}
	if !topo4.UnifiedMemory && topo4.Mode != "single-node-gpu" {
		t.Fatalf("expected single-node-gpu mode on non-UMA host, got %q", topo4.Mode)
	}
	if topo4.TotalVRAMBytes != 96<<30 || topo4.FreeVRAMBytes != 88<<30 {
		t.Fatalf("unexpected 4-GPU aggregate VRAM: total=%d free=%d", topo4.TotalVRAMBytes, topo4.FreeVRAMBytes)
	}

	// Nodes reporting >8 GPUs are clamped to the 1-8 single-node envelope (no multi-node scale-out).
	upDetectGPUStats = func() ([]compute.GPUStat, bool) {
		out := make([]compute.GPUStat, 10)
		for i := range out {
			out[i] = compute.GPUStat{Index: i, VRAMTotalBytes: 24 << 30, VRAMUsedBytes: 4 << 30}
		}
		return out, true
	}
	topoClamped := resolveTurnkeyTopologyAndMemory(0)
	if topoClamped.DeviceCount != maxTurnkeySingleNodeGPUs || topoClamped.ClampedFrom != 10 {
		t.Fatalf("expected clamp 10 -> 8 GPUs, got count=%d clamped_from=%d", topoClamped.DeviceCount, topoClamped.ClampedFrom)
	}

	// Zero discrete GPUs falls back cleanly without failing.
	upDetectGPUStats = func() ([]compute.GPUStat, bool) { return nil, false }
	topoFallback := resolveTurnkeyTopologyAndMemory(0)
	if topoFallback.DeviceCount != 0 || topoFallback.EffectiveBytes == 0 {
		t.Fatalf("unexpected 0-GPU fallback topology: %+v", topoFallback)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestUpAutoAccelDefaultsAndOptOuts(t *testing.T) {
	t.Setenv("FAK_UP_SCAVENGE_LOADER_HEAP", "")
	t.Setenv("FAK_UP_FIT_SHRINK", "")
	if ok, reason := upScavengeLoaderHeapEnabled(); ok || !strings.Contains(reason, "disabled") {
		t.Fatalf("upScavengeLoaderHeapEnabled() = (%v, %q), want opt-in disabled by default", ok, reason)
	}
	if ok, _ := upFitShrinkEnabled(); !ok {
		t.Fatal("upFitShrinkEnabled() = false by default, want true")
	}

	t.Setenv("FAK_UP_SCAVENGE_LOADER_HEAP", "true")
	if ok, reason := upScavengeLoaderHeapEnabled(); !ok || !strings.Contains(reason, "enabled") {
		t.Fatalf("scavenge opt-in = (%v, %q), want enabled", ok, reason)
	}
	t.Setenv("FAK_UP_SCAVENGE_LOADER_HEAP", "0")
	if ok, reason := upScavengeLoaderHeapEnabled(); ok || !strings.Contains(reason, "disabled") {
		t.Fatalf("scavenge opt-out = (%v, %q), want disabled", ok, reason)
	}
	t.Setenv("FAK_UP_FIT_SHRINK", "off")
	if ok, reason := upFitShrinkEnabled(); ok || !strings.Contains(reason, "disabled") {
		t.Fatalf("fit-shrink opt-out = (%v, %q), want disabled", ok, reason)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestUpAutoAccelFitShrinkRetryAndMTPOOMDrop(t *testing.T) {
	origLoad := upLoadModelPlacedFn
	t.Cleanup(func() {
		upLoadModelPlacedFn = origLoad
		fakmodel.SetRetainMTP(false)
	})
	t.Setenv("FAK_UP_FIT_SHRINK", "")
	t.Setenv("FAK_SPECULATIVE", "")
	t.Setenv("SPECULATIVE", "")

	// Case 1: FitTooBig shrinks context tokens from 4096 -> 2048 while keeping RetainMTP intact.
	var seenCtx []int
	fakmodel.SetRetainMTP(true)
	upLoadModelPlacedFn = func(_ string, _ compute.Backend, _ bool, _ bool, ctx int, _ *ggufload.ExpertShard, _ int, _ *serveFitBudget) (*fakmodel.Model, bool, *gateway.ModelLoadProfile, gateway.StartupPhase, compute.MemoryPlan, error) {
		seenCtx = append(seenCtx, ctx)
		if ctx > 2048 {
			return nil, false, nil, gateway.StartupPhase{}, nil, &compute.FitError{Verdict: compute.FitTooBig, Want: 32 << 30, Avail: 16 << 30}
		}
		return &fakmodel.Model{}, true, &gateway.ModelLoadProfile{Mode: "resident"}, gateway.StartupPhase{}, nil, nil
	}

	res, err := loadTurnkeyModelWithFitCommit("stub.gguf", &stubAutoAccelBackend{name: "cuda"}, 4096, nil, true, 512)
	if err != nil {
		t.Fatalf("loadTurnkeyModelWithFitCommit(fit shrink) error: %v", err)
	}
	if res.MTPDroppedOnOOM || !fakmodel.RetainMTP {
		t.Fatalf("FitTooBig should shrink context without dropping MTP, got dropped=%v retainMTP=%v", res.MTPDroppedOnOOM, fakmodel.RetainMTP)
	}
	if !res.Shrunk || res.EffectiveCtx != 2048 || res.Attempts != 2 {
		t.Fatalf("unexpected fit-shrink result: %+v (seenCtx=%v)", res, seenCtx)
	}

	// Case 2: An MTP-specific retention error drops RetainMTP once and retries cleanly.
	upLoadModelPlacedFn = func(_ string, _ compute.Backend, _ bool, _ bool, _ int, _ *ggufload.ExpertShard, _ int, _ *serveFitBudget) (*fakmodel.Model, bool, *gateway.ModelLoadProfile, gateway.StartupPhase, compute.MemoryPlan, error) {
		if fakmodel.RetainMTP {
			return nil, false, nil, gateway.StartupPhase{}, nil, errors.New("incomplete mtp tensor layout")
		}
		return &fakmodel.Model{}, true, &gateway.ModelLoadProfile{Mode: "resident"}, gateway.StartupPhase{}, nil, nil
	}
	resMTP, err := loadTurnkeyModelWithFitCommit("stub.gguf", &stubAutoAccelBackend{name: "cuda"}, 4096, nil, true, 512)
	if err != nil {
		t.Fatalf("loadTurnkeyModelWithFitCommit(MTP fallback) error: %v", err)
	}
	if !resMTP.MTPDroppedOnOOM || fakmodel.RetainMTP || resMTP.Attempts != 2 {
		t.Fatalf("expected one-shot MTP fallback in 2 attempts, got dropped=%v retain=%v attempts=%d", resMTP.MTPDroppedOnOOM, fakmodel.RetainMTP, resMTP.Attempts)
	}

	// Verify disabled FAK_UP_FIT_SHRINK=0 still propagates non-MTP errors directly.
	t.Setenv("FAK_UP_FIT_SHRINK", "0")
	upLoadModelPlacedFn = func(_ string, _ compute.Backend, _ bool, _ bool, _ int, _ *ggufload.ExpertShard, _ int, _ *serveFitBudget) (*fakmodel.Model, bool, *gateway.ModelLoadProfile, gateway.StartupPhase, compute.MemoryPlan, error) {
		return nil, false, nil, gateway.StartupPhase{}, nil, errors.New("corrupt gguf")
	}
	if _, err := loadTurnkeyModelWithFitCommit("stub.gguf", nil, 4096, nil, false, 512); err == nil {
		t.Fatal("expected error with FAK_UP_FIT_SHRINK=0")
	}
}

// fak-test:runtime fast est=1s lane=default
func TestUpAutoAccelSpecMatrixQualification(t *testing.T) {
	origMatrix := upSpecQualificationMatrix
	t.Cleanup(func() { upSpecQualificationMatrix = origMatrix })

	mat := specutil.NewMatrix()
	mat.Record(specutil.Qualification{
		Key:           specutil.QualificationKey{Backend: "cuda", Model: "qwen38:27b-q4_k_m"},
		Qualified:     true,
		EvidenceClass: "HW-WITNESSED",
	})
	upSpecQualificationMatrix = func() *specutil.Matrix { return mat }

	t.Setenv("FAK_SPECULATIVE", "")
	t.Setenv("SPECULATIVE", "")
	be := &stubAutoAccelBackend{name: "cuda", replayPath: compute.Qwen35SequencePrefixReplayPath}
	retain, mode, _, qualified, reason := upShouldRetainResidentMTP(false, be, "qwen38:27b-q4_k_m", os.DevNull, false)
	if !retain || mode != "auto" || !qualified || reason != "" {
		t.Fatalf("qualified auto MTP = (retain=%v, mode=%q, qualified=%v, reason=%q), want retain=true", retain, mode, qualified, reason)
	}

	t.Setenv("FAK_SPECULATIVE", "off")
	retainOff, modeOff, _, _, _ := upShouldRetainResidentMTP(false, be, "qwen38:27b-q4_k_m", os.DevNull, false)
	if retainOff || modeOff != "off" {
		t.Fatalf("opt-out MTP = (retain=%v, mode=%q), want (false, off)", retainOff, modeOff)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestUpAutoAccelHealthzAndSummary(t *testing.T) {
	topo := upGPUTopology{
		Mode:           "single-node-gpu",
		DeviceCount:    2,
		DetectedCount:  2,
		TotalVRAMBytes: 48 << 30,
		FreeVRAMBytes:  44 << 30,
		EffectiveBytes: 48 << 30,
	}
	plan, err := macfit.ConfigureTurnkeyWithOptions(topo.EffectiveBytes, macfit.TurnkeyOptions{KVPrecision: fakmodel.KVPrecisionFP32})
	if err != nil {
		t.Fatalf("ConfigureTurnkeyWithOptions: %v", err)
	}
	be := &stubAutoAccelBackend{name: "cuda"}
	env, err := solveTurnkeyServeEnvelope(plan, fakmodel.KVPrecisionFP32, topo, be, nil, false)
	if err != nil {
		t.Fatalf("solveTurnkeyServeEnvelope: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ts, err := startTurnkeyServerWithTopology(ctx, plan, topo, "127.0.0.1:0", true)
	if err != nil {
		t.Fatalf("startTurnkeyServerWithTopology(mock): %v", err)
	}
	defer ts.Close()

	report := buildTurnkeyAccelerationReport(topo, plan, fakmodel.KVPrecisionFP32, false, &turnkeyNativeResources{
		Startup: turnkeyNativeStartup{
			BackendName:        "cuda",
			FitShrinkEnabled:   true,
			FitShrinkAttempts:  1,
			HeapScavenged:      true,
			HeapScavengeReason: "debug.FreeOSMemory active by default during model load",
			RadixActive:        true,
			RadixReason:        "radix KV prefix reuse active by default across turns",
			Envelope:           env,
		},
		Backend: be,
	})
	var banner bytes.Buffer
	printTurnkeyAccelerationSummary(&banner, report)
	for _, want := range []string{"topology:", "serve_envelope:", "fit_shrink_retry:", "loader_heap_scavenge:", "radix_kv_cache:"} {
		if !strings.Contains(banner.String(), want) {
			t.Fatalf("banner missing %q:\n%s", want, banner.String())
		}
	}

	ts.accelReport = report
	rec := httptest.NewRecorder()
	ts.handleHealthz(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("/healthz HTTP %d, want 200", rec.Code)
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode /healthz JSON: %v", err)
	}
	acc, ok := payload["acceleration"].(map[string]any)
	if !ok {
		t.Fatalf("/healthz missing acceleration object: %+v", payload)
	}
	feats, ok := acc["features"].(map[string]any)
	if !ok || len(feats) < 8 {
		t.Fatalf("/healthz missing features map: %+v", acc)
	}
	for _, k := range []string{"topology", "serve_envelope", "fit_shrink_retry", "loader_heap_scavenge", "speculative_mtp", "cuda_graph", "cuda_mmq", "radix_kv_cache"} {
		entry, ok := feats[k].(map[string]any)
		if !ok || entry["status"] == "" || entry["reason"] == "" {
			t.Fatalf("/healthz feature %q missing status/reason: %+v", k, feats[k])
		}
	}
}
