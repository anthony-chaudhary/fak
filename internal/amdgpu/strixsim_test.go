package amdgpu

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestStrixHaloFullSim verifies the end-to-end discrete-event agent simulation
// and verification report for AMD Strix Halo 128GB with 500 agents and 100 turns.
func TestStrixHaloFullSim(t *testing.T) {
	cfg := StrixSimConfig{
		Platform:           StrixHalo128GB,
		TargetAgents:       500,
		TargetTurns:        100,
		MaxContextTokens:   262144,
		CommonPrefixTokens: 16384,
		TokensPerTurn:      16,
		ModelWeightsBytes:  16 * 1024 * 1024 * 1024, // 16 GiB (27B ROCmFP4)
		MTPAcceptanceRate:  0.82,
		VocabSize:          152064,
		NumRanks:           2,
	}

	report, err := RunStrixHaloSim(cfg)
	if err != nil {
		t.Fatalf("RunStrixHaloSim failed: %v", err)
	}
	if report == nil {
		t.Fatal("RunStrixHaloSim returned nil report")
	}

	if !report.VerifiedParity {
		t.Errorf("expected VerifiedParity to be true; got false")
	}

	// Verify Summary string output
	summary := report.Summary()
	if !strings.Contains(summary, "RYZEN AI MAX+ 395") {
		t.Errorf("expected summary to contain 'RYZEN AI MAX+ 395', got:\n%s", summary)
	}
	if !strings.Contains(summary, "VERIFIED_PARITY") {
		t.Errorf("expected summary to contain 'VERIFIED_PARITY', got:\n%s", summary)
	}

	// Verify JSON export
	jsonData, err := report.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON failed: %v", err)
	}
	var unmarshaled map[string]any
	if err := json.Unmarshal(jsonData, &unmarshaled); err != nil {
		t.Fatalf("unmarshaling ToJSON failed: %v", err)
	}
	if unmarshaled["arch"] != "gfx1151" {
		t.Errorf("expected JSON arch to be gfx1151, got %v", unmarshaled["arch"])
	}
}

// TestStrixHaloBreadthEfficiency verifies the 10x+ breadth metrics, including
// 500 concurrent agents, >99% prefix sharing memory savings, and concurrency admission.
func TestStrixHaloBreadthEfficiency(t *testing.T) {
	cfg := StrixSimConfig{
		Platform:           StrixHalo128GB,
		TargetAgents:       500,
		TargetTurns:        100,
		MaxContextTokens:   262144,
		CommonPrefixTokens: 16384,
		TokensPerTurn:      16,
	}

	report, err := RunStrixHaloSim(cfg)
	if err != nil {
		t.Fatalf("RunStrixHaloSim failed: %v", err)
	}

	if report.AgentCount != 500 {
		t.Errorf("expected 500 agents, got %d", report.AgentCount)
	}

	// Common prefix sharing savings ratio must exceed 99.0%
	if report.CommonPrefixSavingsRatio <= 0.99 {
		t.Errorf("expected CommonPrefixSavingsRatio > 0.99, got %f (%.2f%%)",
			report.CommonPrefixSavingsRatio, report.CommonPrefixSavingsRatio*100.0)
	}

	// Breadth memory efficiency gain must exceed 10x
	if report.BreadthMemoryEfficiencyGain < 10.0 {
		t.Errorf("expected BreadthMemoryEfficiencyGain >= 10.0x, got %.2fx",
			report.BreadthMemoryEfficiencyGain)
	}

	// Concurrency admission must be admitted
	if !report.ConcurrencyAdmitted || report.ConcurrencyAdmissionVerdict != "ADMITTED" {
		t.Errorf("expected concurrency ADMITTED, got verdict=%s, admitted=%t",
			report.ConcurrencyAdmissionVerdict, report.ConcurrencyAdmitted)
	}

	// Total KV memory with sharing must be within available KV budget
	totalGiB := float64(report.TotalKVMemoryWithSharingBytes) / (1 << 30)
	if totalGiB > report.AvailableKVBudgetGiB {
		t.Errorf("total KV memory (%.2f GiB) exceeds available KV budget (%.2f GiB)",
			totalGiB, report.AvailableKVBudgetGiB)
	}
}

// TestStrixHaloDepthLongevity verifies 100+ turns depth, context scaling up to 262,144 tokens,
// TurboQuant KV compression (~75% VRAM savings), MTP speculative acceptance (75-88%),
// bounded forget gate stability, and candidate top-2 wire savings (>99.9%).
func TestStrixHaloDepthLongevity(t *testing.T) {
	cfg := StrixSimConfig{
		Platform:           StrixHalo128GB,
		TargetAgents:       1,
		TargetTurns:        150, // 100+ turns
		MaxContextTokens:   262144,
		CommonPrefixTokens: 16384,
		TokensPerTurn:      64,
		MTPAcceptanceRate:  0.80,
	}

	report, err := RunStrixHaloSim(cfg)
	if err != nil {
		t.Fatalf("RunStrixHaloSim failed: %v", err)
	}

	if report.MaxTurns < 100 {
		t.Errorf("expected MaxTurns >= 100, got %d", report.MaxTurns)
	}
	if report.ContextTokensPerAgent > report.MaxContextTokens {
		t.Errorf("expected context tokens (%d) <= max context tokens (%d)",
			report.ContextTokensPerAgent, report.MaxContextTokens)
	}

	// TurboQuant KV compression
	if report.KVPrecision != "K=Q8, V=turbo4" {
		t.Errorf("expected KV precision 'K=Q8, V=turbo4', got %s", report.KVPrecision)
	}
	if report.TurboQuantVRAMSavings < 74.9 || report.TurboQuantVRAMSavings > 75.1 {
		t.Errorf("expected TurboQuantVRAMSavings ~75%%, got %.2f%%", report.TurboQuantVRAMSavings)
	}
	if report.OverallKVSavingsPercentage < 60.0 {
		t.Errorf("expected OverallKVSavingsPercentage > 60%%, got %.2f%%", report.OverallKVSavingsPercentage)
	}

	// MTP speculative draft acceptance
	if report.MTPAcceptanceRate < 0.75 || report.MTPAcceptanceRate > 0.88 {
		t.Errorf("expected MTPAcceptanceRate in [0.75, 0.88], got %f", report.MTPAcceptanceRate)
	}
	if report.MTPEffectiveSpeedup <= 3.0 {
		t.Errorf("expected MTPEffectiveSpeedup > 3.0x, got %.2fx", report.MTPEffectiveSpeedup)
	}

	// Bounded forget gate stability across turns
	if !report.ForgetGateBounded {
		t.Errorf("expected ForgetGateBounded to be true")
	}
	if !report.ForgetGateStableAcrossTurns {
		t.Errorf("expected ForgetGateStableAcrossTurns to be true")
	}
	if report.ForgetGateMinDecay < compute.MinBoundedDecay {
		t.Errorf("expected ForgetGateMinDecay >= %f, got %f", compute.MinBoundedDecay, report.ForgetGateMinDecay)
	}
	if report.ForgetGateMaxDecay > compute.MaxBoundedDecay {
		t.Errorf("expected ForgetGateMaxDecay <= %f, got %f", compute.MaxBoundedDecay, report.ForgetGateMaxDecay)
	}

	// Top-2 candidate wire savings (>99.9%)
	if report.WireSavingsPercentage <= 99.9 {
		t.Errorf("expected WireSavingsPercentage > 99.9%%, got %.4f%%", report.WireSavingsPercentage)
	}
}

// TestStrixHaloHardwareDispatch verifies AQL 64-byte packet, PM4 Type-3 sequence,
// standalone HSACO generation for gfx1151, and KPACK target resolution.
func TestStrixHaloHardwareDispatch(t *testing.T) {
	cfg := StrixSimConfig{
		Platform: StrixHalo128GB,
	}

	report, err := RunStrixHaloSim(cfg)
	if err != nil {
		t.Fatalf("RunStrixHaloSim failed: %v", err)
	}

	// AQL validation
	if !report.AQLPacketValid {
		t.Errorf("expected AQLPacketValid to be true")
	}
	if report.AQLPacketSize != 64 {
		t.Errorf("expected AQLPacketSize == 64, got %d", report.AQLPacketSize)
	}

	// PM4 validation
	if !report.PM4StreamValid {
		t.Errorf("expected PM4StreamValid to be true")
	}
	if report.PM4DwordCount == 0 {
		t.Errorf("expected PM4DwordCount > 0")
	}

	// HSACO validation
	if report.HSACOBinarySize == 0 {
		t.Errorf("expected HSACOBinarySize > 0")
	}
	if report.HSACOTarget != "amdgcn-amd-amdhsa--gfx1151" {
		t.Errorf("expected HSACOTarget 'amdgcn-amd-amdhsa--gfx1151', got %s", report.HSACOTarget)
	}

	// KPACK resolution
	if !report.KPACKResolved {
		t.Errorf("expected KPACKResolved to be true")
	}
	if report.KPACKTarget != "gfx1151" {
		t.Errorf("expected KPACKTarget 'gfx1151', got %s", report.KPACKTarget)
	}
}

// TestStrixHaloAdmissionRefusal verifies that when workload exceeds UMA capacity,
// the engine refuses admission cleanly without panic or corruption.
func TestStrixHaloAdmissionRefusal(t *testing.T) {
	// Massive workload exceeding 120 GiB UMA aperture
	cfg := StrixSimConfig{
		Platform:           StrixHalo128GB,
		TargetAgents:       50000, // 50,000 agents will far exceed 96 GiB budget
		TargetTurns:        100,
		CommonPrefixTokens: 16384,
		TokensPerTurn:      128,
	}

	report, err := RunStrixHaloSim(cfg)
	if err != nil {
		t.Fatalf("RunStrixHaloSim failed: %v", err)
	}

	if report.ConcurrencyAdmitted {
		t.Errorf("expected ConcurrencyAdmitted to be false for 50,000 agents")
	}
	if report.ConcurrencyAdmissionVerdict != "REFUSED" {
		t.Errorf("expected ConcurrencyAdmissionVerdict == 'REFUSED', got %s", report.ConcurrencyAdmissionVerdict)
	}
	if report.VerifiedParity {
		t.Errorf("expected VerifiedParity to be false when concurrency is refused")
	}
}

// TestStrixHaloInputValidation verifies boundary checks on configuration.
func TestStrixHaloInputValidation(t *testing.T) {
	// Invalid target agents
	_, err := RunStrixHaloSim(StrixSimConfig{TargetAgents: -1})
	if err == nil {
		t.Errorf("expected error for negative target agents")
	}

	// Invalid target turns
	_, err = RunStrixHaloSim(StrixSimConfig{TargetTurns: -5})
	if err == nil {
		t.Errorf("expected error for negative target turns")
	}

	// Invalid MTP acceptance rate (> 1.0)
	_, err = RunStrixHaloSim(StrixSimConfig{MTPAcceptanceRate: 1.5})
	if err == nil {
		t.Errorf("expected error for MTP acceptance rate > 1.0")
	}

	// Unsupported platform
	_, err = RunStrixHaloSim(StrixSimConfig{Platform: "unsupported_gpu"})
	if err == nil {
		t.Errorf("expected error for unsupported platform")
	}
}
