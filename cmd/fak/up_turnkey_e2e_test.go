package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/macfit"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=2s lane=default
func TestUpTurnkeyE2ESingleNodeAutoAccelAndHealthz(t *testing.T) {
	origGPU := upDetectGPUStats
	origUMA := upDetectUnifiedMemory
	t.Cleanup(func() {
		upDetectGPUStats = origGPU
		upDetectUnifiedMemory = origUMA
	})
	t.Setenv("FAK_UP_MEMORY_BYTES", "")
	t.Setenv("FAK_INKERNEL_RADIX", "")

	// 4x24 GiB single-node discrete GPU topology.
	upDetectGPUStats = func() ([]compute.GPUStat, bool) {
		return []compute.GPUStat{
			{Index: 0, VRAMTotalBytes: 24 << 30, VRAMUsedBytes: 2 << 30},
			{Index: 1, VRAMTotalBytes: 24 << 30, VRAMUsedBytes: 2 << 30},
			{Index: 2, VRAMTotalBytes: 24 << 30, VRAMUsedBytes: 2 << 30},
			{Index: 3, VRAMTotalBytes: 24 << 30, VRAMUsedBytes: 2 << 30},
		}, true
	}
	upDetectUnifiedMemory = func() (uint64, error) { return 64 * macfit.GiB, nil }

	topo := resolveTurnkeyTopologyAndMemory(0)
	if topo.DeviceCount != 4 || topo.TotalVRAMBytes != 96<<30 || topo.FreeVRAMBytes != 88<<30 {
		t.Fatalf("unexpected 4-GPU topology: %+v", topo)
	}

	plan, err := macfit.ConfigureTurnkeyWithOptions(topo.EffectiveBytes, macfit.TurnkeyOptions{
		KVPrecision: fakmodel.KVPrecisionFP32,
	})
	if err != nil {
		t.Fatalf("ConfigureTurnkeyWithOptions: %v", err)
	}

	env, err := solveTurnkeyServeEnvelope(plan, fakmodel.KVPrecisionFP32, topo, &stubAutoAccelBackend{name: "cuda"}, nil, false)
	if err != nil {
		t.Fatalf("solveTurnkeyServeEnvelope: %v", err)
	}
	if env.MaxCtx <= 0 || env.TargetConcurrency < 1 || env.DefaultChunk <= 0 {
		t.Fatalf("invalid solved serve envelope: %+v", env)
	}

	planner := newTurnkeyWarmE2EPlanner(t, "synthetic-turnkey-e2e")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	ts, err := startTurnkeyServerWithTopology(ctx, plan, topo, "127.0.0.1:0", false, planner)
	if err != nil {
		t.Fatalf("startTurnkeyServerWithTopology: %v", err)
	}
	defer ts.Close()

	baseURL := fmt.Sprintf("http://%s", ts.listener.Addr().String())
	resp, err := http.Get(baseURL + "/healthz")
	if err != nil {
		t.Fatalf("GET /healthz: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /healthz status = %d, want 200", resp.StatusCode)
	}
	var healthz map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&healthz); err != nil {
		t.Fatalf("decode /healthz: %v", err)
	}
	acc, ok := healthz["acceleration"].(map[string]any)
	if !ok {
		t.Fatalf("/healthz missing acceleration block: %+v", healthz)
	}
	if gotGPUs, _ := acc["gpu_count"].(float64); int(gotGPUs) != 4 {
		t.Fatalf("/healthz acceleration.gpu_count = %v, want 4", acc["gpu_count"])
	}
	feats, ok := acc["features"].(map[string]any)
	if !ok {
		t.Fatalf("/healthz acceleration missing features: %+v", acc)
	}
	for _, key := range []string{"topology", "serve_envelope", "radix_kv_cache", "batch_coalesce"} {
		entry, ok := feats[key].(map[string]any)
		if !ok || entry["status"] != upAccelStatusActive {
			t.Fatalf("/healthz feature %q = %+v, want status=%q", key, feats[key], upAccelStatusActive)
		}
	}

	// Verify two sequential HTTP /v1/chat/completions turns share and reuse the prefix.
	sharedSystem := "You are a turnkey mid-market coding assistant with deterministic prefix reuse across turns."
	postChat := func(userMsg string) {
		t.Helper()
		payload, _ := json.Marshal(map[string]any{
			"model": "synthetic-turnkey-e2e",
			"messages": []agent.Message{
				{Role: agent.RoleSystem, Content: sharedSystem},
				{Role: agent.RoleUser, Content: userMsg},
			},
			"max_tokens": 3,
		})
		r, err := http.Post(baseURL+"/v1/chat/completions", "application/json", bytes.NewReader(payload))
		if err != nil {
			t.Fatalf("POST /v1/chat/completions: %v", err)
		}
		defer r.Body.Close()
		if r.StatusCode != http.StatusOK {
			t.Fatalf("POST /v1/chat/completions status = %d, want 200", r.StatusCode)
		}
		var out gateway.ChatResponse
		if err := json.NewDecoder(r.Body).Decode(&out); err != nil {
			t.Fatalf("decode chat response: %v", err)
		}
		if len(out.Choices) == 0 {
			t.Fatalf("chat response has no choices: %+v", out)
		}
	}

	postChat("first turn question alpha")
	postChat("second turn question beta")
	if matched := lastMatchedPrefix(t, planner); matched <= 0 {
		t.Fatalf("expected second turn to reuse shared prefix tokens, got MatchedPrefix=%d", matched)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestUpTurnkeyE2EFitShrinkAndMTPFallback(t *testing.T) {
	origLoad := upLoadModelPlacedFn
	t.Cleanup(func() {
		upLoadModelPlacedFn = origLoad
		fakmodel.SetRetainMTP(false)
	})
	t.Setenv("FAK_UP_FIT_SHRINK", "")
	t.Setenv("FAK_SPECULATIVE", "")
	t.Setenv("SPECULATIVE", "")

	// FitTooBig shrinks context 8192 -> 4096 -> 2048 while preserving MTP retention.
	fakmodel.SetRetainMTP(true)
	upLoadModelPlacedFn = func(_ string, _ compute.Backend, _ bool, _ bool, ctx int, _ *ggufload.ExpertShard, _ int, _ *serveFitBudget) (*fakmodel.Model, bool, *gateway.ModelLoadProfile, gateway.StartupPhase, compute.MemoryPlan, error) {
		if ctx > 2048 {
			return nil, false, nil, gateway.StartupPhase{}, nil, &compute.FitError{Verdict: compute.FitTooBig, Want: 48 << 30, Avail: 24 << 30}
		}
		return &fakmodel.Model{}, true, &gateway.ModelLoadProfile{Mode: "resident"}, gateway.StartupPhase{}, nil, nil
	}

	res, err := loadTurnkeyModelWithFitCommit("stub.gguf", &stubAutoAccelBackend{name: "cuda"}, 8192, nil, true, 512)
	if err != nil {
		t.Fatalf("loadTurnkeyModelWithFitCommit(fit shrink): %v", err)
	}
	if !res.Shrunk || res.EffectiveCtx != 2048 || res.MTPDroppedOnOOM {
		t.Fatalf("unexpected fit-shrink result: %+v", res)
	}

	// An MTP-specific layout error drops MTP once and succeeds on retry.
	upLoadModelPlacedFn = func(_ string, _ compute.Backend, _ bool, _ bool, _ int, _ *ggufload.ExpertShard, _ int, _ *serveFitBudget) (*fakmodel.Model, bool, *gateway.ModelLoadProfile, gateway.StartupPhase, compute.MemoryPlan, error) {
		if fakmodel.RetainMTP {
			return nil, false, nil, gateway.StartupPhase{}, nil, errors.New("incomplete mtp tensor layout")
		}
		return &fakmodel.Model{}, true, &gateway.ModelLoadProfile{Mode: "resident"}, gateway.StartupPhase{}, nil, nil
	}
	resMTP, err := loadTurnkeyModelWithFitCommit("stub.gguf", &stubAutoAccelBackend{name: "cuda"}, 4096, nil, true, 512)
	if err != nil {
		t.Fatalf("loadTurnkeyModelWithFitCommit(MTP fallback): %v", err)
	}
	if !resMTP.MTPDroppedOnOOM || resMTP.Attempts != 2 || fakmodel.RetainMTP {
		t.Fatalf("unexpected MTP fallback result: %+v (RetainMTP=%v)", resMTP, fakmodel.RetainMTP)
	}
}
