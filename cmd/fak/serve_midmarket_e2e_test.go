package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/macfit"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=2s lane=default
func TestServeMidMarket1To8GPUVRAMEnvelopeFit(t *testing.T) {
	origGPU := upDetectGPUStats
	origUMA := upDetectUnifiedMemory
	t.Cleanup(func() {
		upDetectGPUStats = origGPU
		upDetectUnifiedMemory = origUMA
	})
	t.Setenv("FAK_UP_MEMORY_BYTES", "")
	upDetectUnifiedMemory = func() (uint64, error) { return 64 * macfit.GiB, nil }

	cases := []struct {
		name         string
		detectedGPUs int
		wantGPUs     int
		perGPUVRAM   uint64
		perGPUUsed   uint64
	}{
		{name: "1x24GiB", detectedGPUs: 1, wantGPUs: 1, perGPUVRAM: 24 << 30, perGPUUsed: 2 << 30},
		{name: "2x24GiB", detectedGPUs: 2, wantGPUs: 2, perGPUVRAM: 24 << 30, perGPUUsed: 2 << 30},
		{name: "4x24GiB", detectedGPUs: 4, wantGPUs: 4, perGPUVRAM: 24 << 30, perGPUUsed: 2 << 30},
		{name: "8x24GiB", detectedGPUs: 8, wantGPUs: 8, perGPUVRAM: 24 << 30, perGPUUsed: 2 << 30},
		{name: "10x24GiB_clamped_to_8", detectedGPUs: 10, wantGPUs: 8, perGPUVRAM: 24 << 30, perGPUUsed: 2 << 30},
	}

	be := &stubAutoAccelBackend{name: "cuda"}
	// Fixed 27B tier to verify monotonic context/concurrency scaling for a given model as GPUs scale 1->8.
	fixedPlan, err := macfit.ConfigureTurnkeyWithOptions(48*macfit.GiB, macfit.TurnkeyOptions{
		KVPrecision: fakmodel.KVPrecisionFP32,
	})
	if err != nil {
		t.Fatalf("ConfigureTurnkeyWithOptions(fixed): %v", err)
	}
	prevFixedCtx := 0
	prevAutoWeight := uint64(0)
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			upDetectGPUStats = func() ([]compute.GPUStat, bool) {
				stats := make([]compute.GPUStat, tc.detectedGPUs)
				for i := range stats {
					stats[i] = compute.GPUStat{
						Index:          i,
						VRAMTotalBytes: tc.perGPUVRAM,
						VRAMUsedBytes:  tc.perGPUUsed,
					}
				}
				return stats, true
			}

			topo := resolveTurnkeyTopologyAndMemory(0)
			if topo.DeviceCount != tc.wantGPUs {
				t.Fatalf("DeviceCount = %d, want %d", topo.DeviceCount, tc.wantGPUs)
			}
			wantTotal := uint64(tc.wantGPUs) * tc.perGPUVRAM
			wantFree := uint64(tc.wantGPUs) * (tc.perGPUVRAM - tc.perGPUUsed)
			if topo.TotalVRAMBytes != wantTotal || topo.FreeVRAMBytes != wantFree {
				t.Fatalf("VRAM aggregate = (total=%d, free=%d), want (total=%d, free=%d)",
					topo.TotalVRAMBytes, topo.FreeVRAMBytes, wantTotal, wantFree)
			}

			plan, err := macfit.ConfigureTurnkeyWithOptions(topo.EffectiveBytes, macfit.TurnkeyOptions{
				KVPrecision: fakmodel.KVPrecisionFP32,
			})
			if err != nil {
				t.Fatalf("ConfigureTurnkeyWithOptions: %v", err)
			}
			if plan.Tier.WeightBytes < prevAutoWeight {
				t.Fatalf("auto-selected tier WeightBytes regressed: %d < %d", plan.Tier.WeightBytes, prevAutoWeight)
			}
			prevAutoWeight = plan.Tier.WeightBytes

			fitBudget := turnkeyTopologyFitBudget(topo, plan.MemoryBytes)
			if fitBudget == nil || fitBudget.Base == 0 {
				t.Fatalf("turnkeyTopologyFitBudget returned empty budget: %+v", fitBudget)
			}
			if !topo.UnifiedMemory && uint64(fitBudget.Base) > topo.FreeVRAMBytes {
				t.Fatalf("fitBudget.Base (%d) exceeds FreeVRAMBytes (%d)", fitBudget.Base, topo.FreeVRAMBytes)
			}

			env, err := solveTurnkeyServeEnvelope(plan, fakmodel.KVPrecisionFP32, topo, be, nil, false)
			if err != nil {
				t.Fatalf("solveTurnkeyServeEnvelope(auto): %v", err)
			}
			if env.MaxCtx <= 0 || env.TargetConcurrency < 1 || env.DefaultChunk <= 0 {
				t.Fatalf("invalid auto envelope: %+v", env)
			}

			fixedEnv, err := solveTurnkeyServeEnvelope(fixedPlan, fakmodel.KVPrecisionFP32, topo, be, nil, false)
			if err != nil {
				t.Fatalf("solveTurnkeyServeEnvelope(fixed): %v", err)
			}
			if fixedEnv.MaxCtx < prevFixedCtx {
				t.Fatalf("fixed-tier MaxCtx regressed as GPU count grew: %d < %d", fixedEnv.MaxCtx, prevFixedCtx)
			}
			prevFixedCtx = fixedEnv.MaxCtx
		})
	}
}

// fak-test:runtime fast est=3s lane=default
func TestServeMidMarketOverlapSchedulingAndPrefixReuse(t *testing.T) {
	origGPU := upDetectGPUStats
	t.Cleanup(func() { upDetectGPUStats = origGPU })
	t.Setenv("FAK_INKERNEL_RADIX", "")

	upDetectGPUStats = func() ([]compute.GPUStat, bool) {
		return []compute.GPUStat{
			{Index: 0, VRAMTotalBytes: 24 << 30, VRAMUsedBytes: 2 << 30},
			{Index: 1, VRAMTotalBytes: 24 << 30, VRAMUsedBytes: 2 << 30},
		}, true
	}

	topo := resolveTurnkeyTopologyAndMemory(0)
	plan, err := macfit.ConfigureTurnkeyWithOptions(topo.EffectiveBytes, macfit.TurnkeyOptions{
		KVPrecision: fakmodel.KVPrecisionFP32,
	})
	if err != nil {
		t.Fatalf("ConfigureTurnkeyWithOptions: %v", err)
	}

	planner := newTurnkeyWarmE2EPlanner(t, "synthetic-midmarket-e2e")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	ts, err := startTurnkeyServerWithTopology(ctx, plan, topo, "127.0.0.1:0", false, planner)
	if err != nil {
		t.Fatalf("startTurnkeyServerWithTopology: %v", err)
	}
	defer ts.Close()

	baseURL := fmt.Sprintf("http://%s", ts.listener.Addr().String())
	sharedSystem := "Mid-market 2-GPU serving system prompt: always preserve deterministic prefix KV reuse across overlapped requests."

	sendChat := func(reqCtx context.Context, suffix string) (*gateway.ChatResponse, error) {
		body, err := json.Marshal(map[string]any{
			"model": "synthetic-midmarket-e2e",
			"messages": []agent.Message{
				{Role: agent.RoleSystem, Content: sharedSystem},
				{Role: agent.RoleUser, Content: suffix},
			},
			"max_tokens": 3,
		})
		if err != nil {
			return nil, err
		}
		req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, baseURL+"/v1/chat/completions", bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
		}
		var out gateway.ChatResponse
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			return nil, err
		}
		return &out, nil
	}

	// Seed the radix prefix cache with an initial turn.
	if _, err := sendChat(ctx, "seed-turn"); err != nil {
		t.Fatalf("seed request failed: %v", err)
	}

	// Schedule 4 overlapped follow-up requests sharing the same system prefix via OverlapRunner.
	runner := engine.NewOverlapRunner[string](4)
	defer runner.Close()

	var maxInFlight atomic.Int64
	const numTasks = 4
	for i := 0; i < numTasks; i++ {
		taskID := fmt.Sprintf("midmarket-req-%d", i)
		suffix := fmt.Sprintf("overlapped-turn-%d", i)
		if _, err := runner.Submit(ctx, engine.InFlightTask[string]{
			ID: taskID,
			Execute: func(taskCtx context.Context) (string, error) {
				cur := runner.InFlightCount()
				for {
					prev := maxInFlight.Load()
					if cur <= prev || maxInFlight.CompareAndSwap(prev, cur) {
						break
					}
				}
				out, err := sendChat(taskCtx, suffix)
				if err != nil {
					return "", err
				}
				if len(out.Choices) == 0 {
					return "", fmt.Errorf("empty choices for %s", taskID)
				}
				return out.Choices[0].Message.Content, nil
			},
		}); err != nil {
			t.Fatalf("OverlapRunner.Submit(%s): %v", taskID, err)
		}
	}

	results, err := runner.Drain(ctx)
	if err != nil {
		t.Fatalf("OverlapRunner.Drain: %v", err)
	}
	if len(results) != numTasks {
		t.Fatalf("Drain returned %d results, want %d", len(results), numTasks)
	}
	for _, r := range results {
		if r.Err != nil || !r.Committed {
			t.Fatalf("task %s failed: committed=%v err=%v", r.ID, r.Committed, r.Err)
		}
	}

	decisions := planner.TurnTaxDecisions()
	if len(decisions) < 1+numTasks {
		t.Fatalf("expected at least %d turn-tax decisions, got %d", 1+numTasks, len(decisions))
	}
	reused := 0
	for _, d := range decisions[1:] {
		if d.Decision.MatchedPrefix > 0 {
			reused++
		}
	}
	if reused == 0 {
		t.Fatalf("expected radix prefix reuse on overlapped follow-up turns, got 0/%d hits", len(decisions)-1)
	}
}
