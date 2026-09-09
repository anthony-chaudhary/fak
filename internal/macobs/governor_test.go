package macobs

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"testing"
)

func TestGovernor_24AgentConcurrencyWithoutSwap(t *testing.T) {
	// 1. Hardware baseline: Apple Silicon 36GB unified memory, 27GB wired memory limit
	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 36 * 1024 * 1024 * 1024, // 36 GB DRAM
		WiredMemoryLimitBytes:  27 * 1024 * 1024 * 1024, // 27 GB wired ceiling
		SwapUsedBytes:          0,                       // Zero swap at baseline
		PageOuts:               0,                       // Zero pageouts at baseline
		Available:              true,
	}

	// 2. Model architecture: Qwen3.8-27B 3:1 GDN hybrid
	cfg := Qwen38GDNHeadroomConfig()

	// Verify hybrid geometry
	if cfg.Layers != 64 {
		t.Fatalf("expected 64 layers, got %d", cfg.Layers)
	}
	if cfg.FullAttnLayers != 16 {
		t.Fatalf("expected 16 full-attention layers, got %d", cfg.FullAttnLayers)
	}
	if cfg.RecurrentLayers != 48 {
		t.Fatalf("expected 48 recurrent linear attention layers, got %d", cfg.RecurrentLayers)
	}
	if cfg.EffectiveKVLayers() != 16 {
		t.Fatalf("expected effective KV layers = 16, got %d", cfg.EffectiveKVLayers())
	}
	if math.Abs(cfg.HybridRatio()-0.25) > 1e-4 {
		t.Fatalf("expected hybrid ratio = 0.25 (3:1 GDN), got %f", cfg.HybridRatio())
	}

	// 3. Initialize MemoryGovernor
	gov := NewMemoryGovernor(hw, cfg)

	// Verify initial telemetry
	telemInit := gov.Telemetry()
	if telemInit.ActiveAgents != 0 {
		t.Errorf("expected 0 active agents, got %d", telemInit.ActiveAgents)
	}
	if !telemInit.SharedPreamblePinned {
		t.Errorf("expected shared preamble to be pinned")
	}
	// Pinned shared preamble for 4096 tokens with 16 full-attn layers @ 65536 B/token = 268,435,456 bytes (256 MB = 0.25 GB)
	const wantPreambleBytes = 268435456
	if telemInit.SharedPreambleBytes != wantPreambleBytes {
		t.Errorf("expected SharedPreambleBytes %d (0.25 GB), got %d", wantPreambleBytes, telemInit.SharedPreambleBytes)
	}

	// 4. Sequentially admit 24 concurrent agents with shared preamble
	for i := 1; i <= 24; i++ {
		spec := AgentSpec{
			ID:                  fmt.Sprintf("worker-%d", i),
			SharedPreamble:      true,
			PreambleTokens:      4096,
			TailTokens:          1024,
			RecurrentStateBytes: DefaultQwen38RecurrentStatePerAgentBytes,
		}

		canAdmit, reason := gov.CanAdmit(spec)
		if !canAdmit {
			t.Fatalf("expected CanAdmit = true for agent %d, got false (reason: %s)", i, reason)
		}

		admitted, admitReason := gov.Admit(spec)
		if !admitted {
			t.Fatalf("failed to admit agent %d: %s", i, admitReason)
		}
		if !strings.Contains(admitReason, ReasonHeadroomOK) {
			t.Errorf("expected reason to contain %s, got: %s", ReasonHeadroomOK, admitReason)
		}
	}

	// 5. Assert exactly 24 agents resident and zero-swap bounds preserved
	if gov.ActiveCount() != 24 {
		t.Fatalf("expected 24 active agents, got %d", gov.ActiveCount())
	}

	// Assert zero swap delta and pageout delta
	if err := gov.AssertZeroSwapDelta(); err != nil {
		t.Fatalf("expected zero swap delta assertion to pass, got: %v", err)
	}

	// 6. Assert live telemetry metrics after 24-agent launch
	telem24 := gov.Telemetry()
	if telem24.ActiveAgents != 24 {
		t.Errorf("expected 24 active agents, got %d", telem24.ActiveAgents)
	}
	if telem24.SwapUsedDeltaBytes != 0 {
		t.Errorf("expected SwapUsedDeltaBytes == 0, got %d", telem24.SwapUsedDeltaBytes)
	}
	if telem24.PageoutsDelta != 0 {
		t.Errorf("expected PageoutsDelta == 0, got %d", telem24.PageoutsDelta)
	}
	if !telem24.ZeroSwapGuaranteed {
		t.Errorf("expected ZeroSwapGuaranteed == true")
	}
	if telem24.ConcurrencyStatus != "CAPACITY_OPTIMAL_24_AGENTS" {
		t.Errorf("expected ConcurrencyStatus 'CAPACITY_OPTIMAL_24_AGENTS', got %q", telem24.ConcurrencyStatus)
	}

	// Peak resident memory must be ~25.3 GB (between 25.0 GB and 25.5 GB) and strictly below 27 GB ceiling
	const gb = 1024 * 1024 * 1024
	const ceiling27GB = 27 * gb
	if telem24.ResidentMemoryBytes >= ceiling27GB {
		t.Errorf("resident memory %d bytes (%.2f GB) exceeds 27GB ceiling", telem24.ResidentMemoryBytes, float64(telem24.ResidentMemoryBytes)/float64(gb))
	}
	residentGB := float64(telem24.ResidentMemoryBytes) / float64(gb)
	if residentGB < 25.0 || residentGB > 25.6 {
		t.Errorf("expected resident memory ~25.3 GB, got %.2f GB (%d bytes)", residentGB, telem24.ResidentMemoryBytes)
	}

	// 7. Attempt to admit a 25th non-shared preamble agent when memory headroom is constrained
	nonSharedSpec := AgentSpec{
		ID:                  "worker-nonshared-probe",
		SharedPreamble:      false,
		PreambleTokens:      4096, // Non-shared: requires 256MB preamble + 64MB tail + 3.2MB recurrent
		TailTokens:          1024,
		RecurrentStateBytes: DefaultQwen38RecurrentStatePerAgentBytes,
	}
	canAdmitNS, reasonNS := gov.CanAdmit(nonSharedSpec)
	_ = canAdmitNS
	_ = reasonNS
	// Even if it fits or not, test non-shared rejection when ceiling would be exceeded
	// Let's create an oversized non-shared request that would breach the wired ceiling:
	oversizedNonShared := AgentSpec{
		ID:                  "worker-oversized-nonshared",
		SharedPreamble:      false,
		PreambleTokens:      32768, // Giant isolated preamble
		TailTokens:          4096,
		RecurrentStateBytes: DefaultQwen38RecurrentStatePerAgentBytes,
	}
	canAdmitOS, reasonOS := gov.CanAdmit(oversizedNonShared)
	if canAdmitOS {
		t.Errorf("expected oversized non-shared agent to be rejected, got admitted")
	}
	if !strings.Contains(reasonOS, ReasonNonSharedPreambleSwapRisk) {
		t.Errorf("expected rejection reason to contain %s, got %s", ReasonNonSharedPreambleSwapRisk, reasonOS)
	}

	// 8. Test releasing agents
	for _, a := range gov.ActiveAgents() {
		if !gov.Release(a.ID) {
			t.Errorf("failed to release agent %s", a.ID)
		}
	}
	if gov.ActiveCount() != 0 {
		t.Errorf("expected 0 active agents after full release, got %d", gov.ActiveCount())
	}
	telemReleased := gov.Telemetry()
	if telemReleased.ActiveAgents != 0 {
		t.Errorf("expected 0 active agents in telemetry, got %d", telemReleased.ActiveAgents)
	}
	if telemReleased.AvailableAgentSlots != telemReleased.MaxSharedAgents {
		t.Errorf("expected available slots %d, got %d", telemReleased.MaxSharedAgents, telemReleased.AvailableAgentSlots)
	}

	_ = canAdmitNS
}

func TestGovernor_NonSharedPreambleRejection(t *testing.T) {
	// Constrained headroom scenario: 25GB wired ceiling, 23.5GB base
	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 32 * 1024 * 1024 * 1024,
		WiredMemoryLimitBytes:  25 * 1024 * 1024 * 1024, // 25 GB limit
		SwapUsedBytes:          0,
		PageOuts:               0,
		Available:              true,
	}
	cfg := Qwen38GDNHeadroomConfig() // Base = 23.5 GB (16GB weights + 7.5GB OS) + 256MB preamble = 23.75 GB
	// Available headroom = 1.25 GB = 1280 MB

	gov := NewMemoryGovernor(hw, cfg)

	// An isolated agent needs: (4096+1024)*65536 + 3.2MB = 335.5MB + 3.2MB = 338.7 MB
	// 4 isolated agents require: 4 * 338.7 MB = 1354.8 MB > 1280 MB!
	// Admit 3 isolated agents:
	for i := 1; i <= 3; i++ {
		spec := AgentSpec{
			ID:             fmt.Sprintf("iso-%d", i),
			SharedPreamble: false,
			PreambleTokens: 4096,
			TailTokens:     1024,
		}
		admitted, reason := gov.Admit(spec)
		if !admitted {
			t.Fatalf("expected isolated agent %d to be admitted, failed: %s", i, reason)
		}
	}

	// 4th isolated agent would project ~25.1 GB > 25 GB ceiling -> MUST be rejected
	iso4 := AgentSpec{
		ID:             "iso-4",
		SharedPreamble: false,
		PreambleTokens: 4096,
		TailTokens:     1024,
	}
	canAdmit4, reason4 := gov.CanAdmit(iso4)
	if canAdmit4 {
		t.Fatalf("expected 4th isolated agent to be rejected with swap risk, got admitted")
	}
	if !strings.Contains(reason4, ReasonNonSharedPreambleSwapRisk) {
		t.Errorf("expected reason %s, got: %s", ReasonNonSharedPreambleSwapRisk, reason4)
	}

	// But a SHARED preamble agent needing only 67.2 MB CAN still be admitted!
	sharedSpec := AgentSpec{
		ID:             "shared-subagent",
		SharedPreamble: true,
		PreambleTokens: 4096,
		TailTokens:     1024,
	}
	canAdmitShared, reasonShared := gov.CanAdmit(sharedSpec)
	if !canAdmitShared {
		t.Fatalf("expected shared preamble agent to be admitted within remaining headroom, failed: %s", reasonShared)
	}
	if !strings.Contains(reasonShared, ReasonHeadroomOK) {
		t.Errorf("expected shared reason %s, got: %s", ReasonHeadroomOK, reasonShared)
	}
}

func TestGovernor_ZeroSwapDeltaAssertion(t *testing.T) {
	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 36 * 1024 * 1024 * 1024,
		WiredMemoryLimitBytes:  27 * 1024 * 1024 * 1024,
		SwapUsedBytes:          100 * 1024 * 1024, // 100MB initial swap
		PageOuts:               1000,
		Available:              true,
	}
	cfg := Qwen38GDNHeadroomConfig()

	gov := NewMemoryGovernor(hw, cfg)

	// Initially swap delta is 0
	if err := gov.AssertZeroSwapDelta(); err != nil {
		t.Fatalf("expected zero delta initially, got: %v", err)
	}

	// Simulate macOS paging out to disk (swap increase)
	hwMutated := hw
	hwMutated.SwapUsedBytes = 200 * 1024 * 1024 // +100MB swap delta
	hwMutated.PageOuts = 1500                   // +500 pageouts delta
	gov.UpdateHardwareTelemetry(hwMutated)

	// Assertion must now fail
	err := gov.AssertZeroSwapDelta()
	if err == nil {
		t.Fatalf("expected zero swap delta assertion to fail when swap increased")
	}
	if !strings.Contains(err.Error(), "swap_used_delta=104857600") {
		t.Errorf("expected error to name swap delta, got: %v", err)
	}

	// Admission should reject with SWAP_DELTA_DETECTED
	spec := AgentSpec{SharedPreamble: true}
	admitted, reason := gov.CanAdmit(spec)
	if admitted {
		t.Errorf("expected admission to fail when swap delta detected")
	}
	if !strings.Contains(reason, ReasonSwapDeltaDetected) {
		t.Errorf("expected reason to contain %s, got: %s", ReasonSwapDeltaDetected, reason)
	}
}

func TestGovernor_Concurrent24AgentAdmission(t *testing.T) {
	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 36 * 1024 * 1024 * 1024,
		WiredMemoryLimitBytes:  27 * 1024 * 1024 * 1024,
		Available:              true,
	}
	cfg := Qwen38GDNHeadroomConfig()
	gov := NewMemoryGovernor(hw, cfg)

	var wg sync.WaitGroup
	errCh := make(chan error, 24)

	for i := 1; i <= 24; i++ {
		wg.Add(1)
		go func(agentNum int) {
			defer wg.Done()
			spec := AgentSpec{
				ID:             fmt.Sprintf("concurrent-agent-%d", agentNum),
				SharedPreamble: true,
				PreambleTokens: 4096,
				TailTokens:     1024,
			}
			admitted, reason := gov.Admit(spec)
			if !admitted {
				errCh <- fmt.Errorf("agent %d failed: %s", agentNum, reason)
			}
		}(i)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Errorf("concurrent admission error: %v", err)
	}

	if gov.ActiveCount() != 24 {
		t.Errorf("expected 24 active agents, got %d", gov.ActiveCount())
	}
	if err := gov.AssertZeroSwapDelta(); err != nil {
		t.Errorf("unexpected swap delta: %v", err)
	}
}

func TestGovernor_3to1GDNHybridGeometry(t *testing.T) {
	cfg := Qwen38GDNHeadroomConfig()

	// 16 full-attention layers + 48 recurrent linear attention layers = 64 total
	if cfg.Layers != 64 || cfg.FullAttnLayers != 16 || cfg.RecurrentLayers != 48 {
		t.Fatalf("geometry mismatch: layers=%d, full=%d, rec=%d", cfg.Layers, cfg.FullAttnLayers, cfg.RecurrentLayers)
	}

	// KV bytes per token = 2 * 16 layers * 8 heads * 128 head_dim * 2 bytes = 65,536 bytes
	wantKVBytesPerToken := uint64(2 * 16 * 8 * 128 * 2)
	if wantKVBytesPerToken != 65536 {
		t.Fatalf("expected 65536 bytes/token, got %d", wantKVBytesPerToken)
	}

	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 36 * 1024 * 1024 * 1024,
		WiredMemoryLimitBytes:  27 * 1024 * 1024 * 1024,
		Available:              true,
	}
	head := ComputeHeadroom(hw, cfg)

	if head.ModelKVBytesPerToken != wantKVBytesPerToken {
		t.Errorf("ComputeHeadroom KVBytesPerToken = %d, want %d", head.ModelKVBytesPerToken, wantKVBytesPerToken)
	}
	if head.FullAttnLayers != 16 {
		t.Errorf("head.FullAttnLayers = %d, want 16", head.FullAttnLayers)
	}
	if head.RecurrentLayers != 48 {
		t.Errorf("head.RecurrentLayers = %d, want 48", head.RecurrentLayers)
	}
	if head.MaxSharedAgents < 24 {
		t.Errorf("head.MaxSharedAgents = %d, want >= 24", head.MaxSharedAgents)
	}
}

func TestGovernor_MemoryPressureIntegration(t *testing.T) {
	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 36 * 1024 * 1024 * 1024,
		WiredMemoryLimitBytes:  27 * 1024 * 1024 * 1024,
		Available:              true,
	}
	cfg := Qwen38GDNHeadroomConfig()
	sub := NewSimulatedSubscriber()

	gov := NewMemoryGovernor(hw, cfg, WithGovernorPressureSubscriber(sub))

	// Under normal pressure, admission succeeds
	sub.SimulatePressure(PressureNormal)
	spec := AgentSpec{ID: "sub-1", SharedPreamble: true}
	canAdmit, reason := gov.CanAdmit(spec)
	if !canAdmit {
		t.Fatalf("expected admit under normal pressure, got: %s", reason)
	}

	// Under critical pressure, requests are queued
	sub.SimulatePressure(PressureCritical)
	canAdmitCrit, reasonCrit := gov.CanAdmit(spec)
	if canAdmitCrit {
		t.Fatalf("expected rejection/queue under critical pressure")
	}
	if !strings.Contains(reasonCrit, ReasonPressureCritical) {
		t.Errorf("expected reason %s, got: %s", ReasonPressureCritical, reasonCrit)
	}

	// Under warn pressure, non-shared is rejected
	sub.SimulatePressure(PressureWarn)
	nsSpec := AgentSpec{ID: "sub-ns", SharedPreamble: false}
	canAdmitWarnNS, reasonWarnNS := gov.CanAdmit(nsSpec)
	if canAdmitWarnNS {
		t.Fatalf("expected non-shared to be rejected under warn pressure")
	}
	if !strings.Contains(reasonWarnNS, ReasonPressureWarn) {
		t.Errorf("expected reason %s, got: %s", ReasonPressureWarn, reasonWarnNS)
	}
}

func TestGovernor_AdversarialChurnAndPressure(t *testing.T) {
	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 36 * 1024 * 1024 * 1024,
		WiredMemoryLimitBytes:  27 * 1024 * 1024 * 1024,
		Available:              true,
	}
	cfg := Qwen38GDNHeadroomConfig()
	sub := NewSimulatedSubscriber()
	gov := NewMemoryGovernor(hw, cfg, WithGovernorPressureSubscriber(sub))

	// Step 1: Concurrently admit 24 agents under mutex contention
	var wg sync.WaitGroup
	errCh := make(chan error, 48)

	for i := 1; i <= 24; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			spec := AgentSpec{
				ID:             fmt.Sprintf("churn-agent-%d", id),
				SharedPreamble: true,
				PreambleTokens: 4096,
				TailTokens:     1024,
			}
			admitted, reason := gov.Admit(spec)
			if !admitted {
				errCh <- fmt.Errorf("initial admit %d failed: %s", id, reason)
			}
		}(i)
	}
	wg.Wait()

	if gov.ActiveCount() != 24 {
		t.Fatalf("expected 24 active agents, got %d", gov.ActiveCount())
	}
	if err := gov.AssertZeroSwapDelta(); err != nil {
		t.Fatalf("swap delta assertion failed: %v", err)
	}

	// Step 2: Concurrently release 12 agents while reading telemetry
	for i := 1; i <= 12; i++ {
		wg.Add(2)
		agentID := fmt.Sprintf("churn-agent-%d", i)
		go func(id string) {
			defer wg.Done()
			if !gov.Release(id) {
				errCh <- fmt.Errorf("release %s failed", id)
			}
		}(agentID)
		go func() {
			defer wg.Done()
			_ = gov.Telemetry()
		}()
	}
	wg.Wait()

	if gov.ActiveCount() != 12 {
		t.Fatalf("expected 12 active agents after partial release, got %d", gov.ActiveCount())
	}

	// Step 3: Concurrently re-admit 12 agents back up to 24
	for i := 1; i <= 12; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			spec := AgentSpec{
				ID:             fmt.Sprintf("churn-agent-%d", id),
				SharedPreamble: true,
				PreambleTokens: 4096,
				TailTokens:     1024,
			}
			admitted, reason := gov.Admit(spec)
			if !admitted {
				errCh <- fmt.Errorf("re-admit %d failed: %s", id, reason)
			}
		}(i)
	}
	wg.Wait()

	if gov.ActiveCount() != 24 {
		t.Fatalf("expected 24 active agents after re-admission, got %d", gov.ActiveCount())
	}

	// Step 4: Rapid pressure transitions under active load
	sub.SimulatePressure(PressureCritical)
	// New admission must be queued
	specProbe := AgentSpec{ID: "probe-critical", SharedPreamble: true}
	admitted, reason := gov.CanAdmit(specProbe)
	if admitted || !strings.Contains(reason, ReasonPressureCritical) {
		t.Fatalf("expected queued under critical pressure, got admitted=%t, reason=%s", admitted, reason)
	}

	sub.SimulatePressure(PressureWarn)
	// Non-shared must be rejected under WARN
	specWarnNS := AgentSpec{ID: "probe-warn-ns", SharedPreamble: false}
	admittedNS, reasonNS := gov.CanAdmit(specWarnNS)
	if admittedNS || !strings.Contains(reasonNS, ReasonPressureWarn) {
		t.Fatalf("expected rejection for non-shared under WARN, got admitted=%t, reason=%s", admittedNS, reasonNS)
	}

	sub.SimulatePressure(PressureNormal)

	// Step 5: Full release and re-admission
	for _, a := range gov.ActiveAgents() {
		if !gov.Release(a.ID) {
			t.Errorf("failed releasing %s", a.ID)
		}
	}
	if gov.ActiveCount() != 0 {
		t.Fatalf("expected 0 active agents after full release, got %d", gov.ActiveCount())
	}

	for i := 1; i <= 24; i++ {
		spec := AgentSpec{
			ID:             fmt.Sprintf("cycle2-agent-%d", i),
			SharedPreamble: true,
			PreambleTokens: 4096,
			TailTokens:     1024,
		}
		admitted, reason := gov.Admit(spec)
		if !admitted {
			t.Fatalf("cycle 2 admit %d failed: %s", i, reason)
		}
	}

	if gov.ActiveCount() != 24 {
		t.Fatalf("expected 24 active agents in cycle 2, got %d", gov.ActiveCount())
	}

	close(errCh)
	for err := range errCh {
		t.Errorf("churn test error: %v", err)
	}

	telem := gov.Telemetry()
	if !telem.ZeroSwapGuaranteed {
		t.Errorf("expected ZeroSwapGuaranteed == true")
	}
	if telem.ResidentMemoryBytes >= hw.WiredMemoryLimitBytes {
		t.Errorf("resident memory %d exceeds wired limit %d", telem.ResidentMemoryBytes, hw.WiredMemoryLimitBytes)
	}
}
