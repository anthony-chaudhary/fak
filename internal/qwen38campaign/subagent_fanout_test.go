package qwen38campaign

import (
	"bytes"
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

func TestSubagentCLIRejectsSyntheticPhysicalMode(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunCLI(&stdout, &stderr, []string{"--simulated=false", "--json", "--runs=5", "--prefix-tokens=1", "--gen-tokens=1"})
	if code == 0 || stdout.Len() != 0 {
		t.Fatalf("synthetic physical execution returned code=%d receipt_bytes=%d", code, stdout.Len())
	}
	if !strings.Contains(stderr.String(), "physical execution is unavailable") {
		t.Fatalf("missing physical execution diagnostic: %s", stderr.String())
	}
}

// TestSubagentFanoutMatrix tests the full 3 scenarios x 4 concurrencies matrix.
func TestSubagentFanoutMatrix(t *testing.T) {
	scenarios := []string{ScenarioCold, ScenarioWarmSamePrefix, ScenarioSharedPrefixForked}
	concurrencies := []int{1, 2, 4, 8}

	for _, sc := range scenarios {
		for _, b := range concurrencies {
			t.Run(sc+"_B"+string(rune('0'+b)), func(t *testing.T) {
				cfg := FanoutConfig{
					Scenario:                   sc,
					Concurrency:                b,
					Runs:                       5,
					PrefixTokens:               1000,
					GeneratedTokensPerSubagent: 16,
					Simulated:                  true,
					Seed:                       42,
				}

				receipt, err := ExecuteSubagentFanoutBenchmark(cfg)
				if err != nil {
					t.Fatalf("failed to execute benchmark for %s B=%d: %v", sc, b, err)
				}

				if err := receipt.Validate(); err != nil {
					t.Fatalf("receipt failed validation: %v", err)
				}

				if receipt.Schema != SubagentFanoutSchema {
					t.Errorf("schema = %q, want %q", receipt.Schema, SubagentFanoutSchema)
				}
				if receipt.Engine != CanonicalEngineName {
					t.Errorf("engine = %q, want %q", receipt.Engine, CanonicalEngineName)
				}
				if receipt.Config.Concurrency != b {
					t.Errorf("concurrency = %d, want %d", receipt.Config.Concurrency, b)
				}
				if receipt.Config.Scenario != sc {
					t.Errorf("scenario = %q, want %q", receipt.Config.Scenario, sc)
				}
				if len(receipt.Runs) != 5 {
					t.Errorf("runs count = %d, want 5", len(receipt.Runs))
				}

				// Verify tokens/sec and duration are strictly positive
				if receipt.Summary.MeanTokensPerSec <= 0 {
					t.Errorf("mean tokens/sec %f must be positive", receipt.Summary.MeanTokensPerSec)
				}
				if receipt.Summary.P50TokensPerSec <= 0 {
					t.Errorf("p50 tokens/sec %f must be positive", receipt.Summary.P50TokensPerSec)
				}
				if receipt.Summary.P95TokensPerSec <= 0 {
					t.Errorf("p95 tokens/sec %f must be positive", receipt.Summary.P95TokensPerSec)
				}
				if receipt.Summary.NoisePercent < 0 || receipt.Summary.NoisePercent > 50 {
					t.Errorf("noise percent %f out of expected range [0, 50]", receipt.Summary.NoisePercent)
				}

				// Verify phase breakdown covers all canonical phases
				for _, p := range CanonicalPhaseBuckets {
					if ms, ok := receipt.PhaseSummaryMS[p]; !ok || ms <= 0 {
						t.Errorf("phase %s missing or non-positive: %f", p, ms)
					}
				}

				// Verify parity passes threshold
				if !receipt.Summary.ParityPassed {
					t.Errorf("logit cosine parity failed: mean = %.6f, threshold = %.6f",
						receipt.Summary.MeanLogitCosineParity, receipt.Summary.ParityThreshold)
				}
				if receipt.Summary.MeanLogitCosineParity < receipt.Config.ParityThreshold {
					t.Errorf("mean logit parity %.6f < threshold %.6f",
						receipt.Summary.MeanLogitCosineParity, receipt.Summary.ParityThreshold)
				}

				// Verify scenario-specific cache properties
				switch sc {
				case ScenarioSharedPrefixForked:
					if receipt.Summary.MeanMALLHitRate < 0.85 {
						t.Errorf("shared_prefix_forked mean MALL hit rate %.4f < 0.85", receipt.Summary.MeanMALLHitRate)
					}
				case ScenarioCold:
					if receipt.Summary.MeanMALLHitRate > 0.30 {
						t.Errorf("cold mean MALL hit rate %.4f > 0.30", receipt.Summary.MeanMALLHitRate)
					}
				}
			})
		}
	}
}

// TestSharedPrefixForkingCOW validates copy-on-write zero-copy session branching.
func TestSharedPrefixForkingCOW(t *testing.T) {
	mgr := ctxmmu.NewForkManager(ctxmmu.ForkConfig{
		Granularity:   ctxmmu.BlockGranularity64,
		BytesPerToken: DefaultBytesPerToken,
	})

	parentID := "master-repo-prefix"
	parent, err := mgr.RegisterSession(parentID, ctxmmu.BlockGranularity64)
	if err != nil {
		t.Fatalf("failed to register master session: %v", err)
	}

	// 30k tokens prefix
	prefixTokens := make([]int32, DefaultSharedPrefixTokens)
	for i := range prefixTokens {
		prefixTokens[i] = int32((i % 1000) + 1)
	}
	if err := parent.AppendTokens(prefixTokens...); err != nil {
		t.Fatalf("failed to append 30k tokens: %v", err)
	}

	initialAllocatedBlocks := mgr.ActiveBlockCount()
	expectedBlocks := (DefaultSharedPrefixTokens + 63) / 64
	if initialAllocatedBlocks != expectedBlocks {
		t.Fatalf("allocated blocks = %d, want %d", initialAllocatedBlocks, expectedBlocks)
	}

	// Fork 4 subagents
	concurrency := 4
	children := make([]*ctxmmu.ForkedSession, concurrency)
	for i := 0; i < concurrency; i++ {
		childID := string(rune('A' + i))
		child, err := mgr.ForkSession(parentID, childID)
		if err != nil {
			t.Fatalf("failed to fork subagent %s: %v", childID, err)
		}
		children[i] = child
	}

	// Zero new physical blocks should be allocated by the fork itself (zero-copy)
	if mgr.ActiveBlockCount() != initialAllocatedBlocks {
		t.Errorf("after zero-copy fork, active blocks changed from %d to %d",
			initialAllocatedBlocks, mgr.ActiveBlockCount())
	}

	// Each subagent appends unique tokens (triggers COW for the last block)
	for i, child := range children {
		uniqueTokens := []int32{int32(10000 + i*100), int32(10001 + i*100)}
		if err := child.AppendTokens(uniqueTokens...); err != nil {
			t.Fatalf("child %d append tokens failed: %v", i, err)
		}
	}

	// Active blocks should now account for COW private cloned blocks
	if mgr.ActiveBlockCount() <= initialAllocatedBlocks {
		t.Errorf("expected COW block allocation, active blocks = %d", mgr.ActiveBlockCount())
	}

	// Parent prefix tokens must remain completely unaffected
	parentTokens := parent.ReadTokens()
	if len(parentTokens) != DefaultSharedPrefixTokens {
		t.Fatalf("parent tokens length = %d, want %d", len(parentTokens), DefaultSharedPrefixTokens)
	}
	for i := 0; i < 100; i++ {
		if parentTokens[i] != prefixTokens[i] {
			t.Errorf("parent token %d mutated: got %d, want %d", i, parentTokens[i], prefixTokens[i])
		}
	}
}

// TestStatisticalAggregation verifies distribution calculations across >= 5 runs.
func TestStatisticalAggregation(t *testing.T) {
	runs := []RunMetric{
		{
			RunIndex:          1,
			Concurrency:       4,
			Scenario:          ScenarioSharedPrefixForked,
			WallDurationMS:    50.0,
			UsefulTokens:      256,
			TokensPerSec:      5120.0,
			DRAMBandwidthGBps: 40.0,
			MALLHitRate:       0.94,
			QueueLatencyMS:    0.10,
			PhasesMS: map[string]float64{
				PhaseHostDispatch:     0.15,
				PhasePrefixTreeLookup: 0.05,
				PhaseKVAllocation:     0.05,
				PhaseGPUKernel:        49.35,
				PhaseTokenSampling:    0.40,
			},
			LogitCosineParity: 0.999998,
			ParityPassed:      true,
		},
		{
			RunIndex:          2,
			Concurrency:       4,
			Scenario:          ScenarioSharedPrefixForked,
			WallDurationMS:    48.0,
			UsefulTokens:      256,
			TokensPerSec:      5333.33,
			DRAMBandwidthGBps: 42.0,
			MALLHitRate:       0.95,
			QueueLatencyMS:    0.11,
			PhasesMS: map[string]float64{
				PhaseHostDispatch:     0.14,
				PhasePrefixTreeLookup: 0.05,
				PhaseKVAllocation:     0.05,
				PhaseGPUKernel:        47.36,
				PhaseTokenSampling:    0.40,
			},
			LogitCosineParity: 0.999999,
			ParityPassed:      true,
		},
		{
			RunIndex:          3,
			Concurrency:       4,
			Scenario:          ScenarioSharedPrefixForked,
			WallDurationMS:    52.0,
			UsefulTokens:      256,
			TokensPerSec:      4923.08,
			DRAMBandwidthGBps: 39.0,
			MALLHitRate:       0.93,
			QueueLatencyMS:    0.12,
			PhasesMS: map[string]float64{
				PhaseHostDispatch:     0.16,
				PhasePrefixTreeLookup: 0.06,
				PhaseKVAllocation:     0.05,
				PhaseGPUKernel:        51.33,
				PhaseTokenSampling:    0.40,
			},
			LogitCosineParity: 0.999997,
			ParityPassed:      true,
		},
		{
			RunIndex:          4,
			Concurrency:       4,
			Scenario:          ScenarioSharedPrefixForked,
			WallDurationMS:    50.0,
			UsefulTokens:      256,
			TokensPerSec:      5120.0,
			DRAMBandwidthGBps: 40.5,
			MALLHitRate:       0.94,
			QueueLatencyMS:    0.09,
			PhasesMS: map[string]float64{
				PhaseHostDispatch:     0.15,
				PhasePrefixTreeLookup: 0.05,
				PhaseKVAllocation:     0.05,
				PhaseGPUKernel:        49.35,
				PhaseTokenSampling:    0.40,
			},
			LogitCosineParity: 0.999998,
			ParityPassed:      true,
		},
		{
			RunIndex:          5,
			Concurrency:       4,
			Scenario:          ScenarioSharedPrefixForked,
			WallDurationMS:    51.0,
			UsefulTokens:      256,
			TokensPerSec:      5019.61,
			DRAMBandwidthGBps: 39.8,
			MALLHitRate:       0.94,
			QueueLatencyMS:    0.10,
			PhasesMS: map[string]float64{
				PhaseHostDispatch:     0.15,
				PhasePrefixTreeLookup: 0.05,
				PhaseKVAllocation:     0.05,
				PhaseGPUKernel:        50.35,
				PhaseTokenSampling:    0.40,
			},
			LogitCosineParity: 0.999998,
			ParityPassed:      true,
		},
	}

	summary, phases := CalculateStatisticalSummary(runs, DefaultLogitCosineParityThreshold)

	if summary.RunsCount != 5 {
		t.Errorf("runs count = %d, want 5", summary.RunsCount)
	}

	expectedMean := (5120.0 + 5333.33 + 4923.08 + 5120.0 + 5019.61) / 5.0
	if math.Abs(summary.MeanTokensPerSec-expectedMean) > 0.01 {
		t.Errorf("mean TPS = %f, want %f", summary.MeanTokensPerSec, expectedMean)
	}

	// P50 should be 5120.0
	if math.Abs(summary.P50TokensPerSec-5120.0) > 0.01 {
		t.Errorf("p50 TPS = %f, want 5120.0", summary.P50TokensPerSec)
	}

	// P95 should match the sorted 95th percentile
	if summary.P95TokensPerSec <= summary.P50TokensPerSec {
		t.Errorf("p95 (%f) should be >= p50 (%f)", summary.P95TokensPerSec, summary.P50TokensPerSec)
	}

	// Noise percent should be stddev / mean * 100
	expectedNoise := (summary.StdDevTokensPerSec / summary.MeanTokensPerSec) * 100.0
	if math.Abs(summary.NoisePercent-expectedNoise) > 0.01 {
		t.Errorf("noise percent = %f, want %f", summary.NoisePercent, expectedNoise)
	}

	// Phases mean
	if math.Abs(phases[PhaseHostDispatch]-0.15) > 0.01 {
		t.Errorf("mean host dispatch = %f, want 0.15", phases[PhaseHostDispatch])
	}
}

// TestReceiptValidationRejections asserts fail-closed behavior on invalid receipts.
func TestReceiptValidationRejections(t *testing.T) {
	validCfg := FanoutConfig{
		Scenario:                   ScenarioSharedPrefixForked,
		Concurrency:                4,
		Runs:                       5,
		PrefixTokens:               1000,
		GeneratedTokensPerSubagent: 16,
		Simulated:                  true,
	}

	receipt, err := ExecuteSubagentFanoutBenchmark(validCfg)
	if err != nil {
		t.Fatalf("unexpected benchmark failure: %v", err)
	}
	if err := receipt.Validate(); err != nil {
		t.Fatalf("expected valid receipt, got error: %v", err)
	}

	// 1. Invalid schema
	badSchema := receipt
	badSchema.Schema = "fak.benchmark.subagent_fanout/v0"
	if err := badSchema.Validate(); err == nil {
		t.Errorf("expected rejection on bad schema")
	}

	// 2. Invalid engine
	badEngine := receipt
	badEngine.Engine = "llama.cpp"
	if err := badEngine.Validate(); err == nil {
		t.Errorf("expected rejection on bad engine")
	}

	// 3. Runs count mismatch
	badRuns := receipt
	badRuns.Runs = receipt.Runs[:4]
	if err := badRuns.Validate(); err == nil {
		t.Errorf("expected rejection on runs length mismatch")
	}

	// 4. Parity failure
	badParity := receipt
	badParity.Runs[0].LogitCosineParity = 0.999800 // Below 0.999900 threshold
	if err := badParity.Validate(); err == nil {
		t.Errorf("expected rejection on sub-threshold logit parity")
	}
}

// TestConfigValidation verifies that bad configurations are rejected.
func TestConfigValidation(t *testing.T) {
	tests := []struct {
		name    string
		cfg     FanoutConfig
		wantErr bool
	}{
		{
			name: "valid-cold-B1",
			cfg: FanoutConfig{
				Scenario:                   ScenarioCold,
				Concurrency:                1,
				Runs:                       5,
				PrefixTokens:               100,
				GeneratedTokensPerSubagent: 16,
			},
			wantErr: false,
		},
		{
			name: "invalid-concurrency-3",
			cfg: FanoutConfig{
				Scenario:                   ScenarioCold,
				Concurrency:                3,
				Runs:                       5,
				PrefixTokens:               100,
				GeneratedTokensPerSubagent: 16,
			},
			wantErr: true,
		},
		{
			name: "invalid-runs-below-5",
			cfg: FanoutConfig{
				Scenario:                   ScenarioSharedPrefixForked,
				Concurrency:                4,
				Runs:                       4,
				PrefixTokens:               100,
				GeneratedTokensPerSubagent: 16,
			},
			wantErr: true,
		},
		{
			name: "invalid-scenario",
			cfg: FanoutConfig{
				Scenario:                   "speculative_random",
				Concurrency:                4,
				Runs:                       5,
				PrefixTokens:               100,
				GeneratedTokensPerSubagent: 16,
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cfg.Validate()
			if (err != nil) != tc.wantErr {
				t.Errorf("Validate() err = %v, wantErr = %v", err, tc.wantErr)
			}
		})
	}
}

// TestReceiptJSONRoundTrip verifies JSON serialization and deserialization.
func TestReceiptJSONRoundTrip(t *testing.T) {
	cfg := FanoutConfig{
		Scenario:                   ScenarioWarmSamePrefix,
		Concurrency:                2,
		Runs:                       5,
		PrefixTokens:               2048,
		GeneratedTokensPerSubagent: 32,
		Simulated:                  true,
		Seed:                       123,
	}

	receipt, err := ExecuteSubagentFanoutBenchmark(cfg)
	if err != nil {
		t.Fatalf("failed to execute benchmark: %v", err)
	}

	data, err := receipt.JSON()
	if err != nil {
		t.Fatalf("failed to serialize JSON: %v", err)
	}

	var parsed SubagentFanoutReceipt
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v", err)
	}

	if err := parsed.Validate(); err != nil {
		t.Fatalf("parsed receipt validation failed: %v", err)
	}

	tableStr := receipt.String()
	if !strings.Contains(tableStr, "Strix Halo Subagent Fan-Out Benchmark Receipt") {
		t.Errorf("String() missing title: %s", tableStr)
	}
	if !strings.Contains(tableStr, "MALL Cache Size:     32 MB") {
		t.Errorf("String() missing 32 MB MALL cache note: %s", tableStr)
	}
}

// TestCosineSimilarityMath tests the cosine similarity implementation.
func TestCosineSimilarityMath(t *testing.T) {
	a := []float64{1.0, 2.0, 3.0}
	b := []float64{1.0, 2.0, 3.0}
	cos := CosineSimilarity(a, b)
	if math.Abs(cos-1.0) > 1e-9 {
		t.Errorf("identical vectors cosine = %f, want 1.0", cos)
	}

	orthogonal := []float64{-2.0, 1.0, 0.0}
	cosOrth := CosineSimilarity(a, orthogonal)
	if math.Abs(cosOrth-0.0) > 1e-9 {
		t.Errorf("orthogonal vectors cosine = %f, want 0.0", cosOrth)
	}

	// High parity check
	perturbed := []float64{1.00001, 2.00001, 3.00001}
	cosPerturbed := CosineSimilarity(a, perturbed)
	if cosPerturbed < DefaultLogitCosineParityThreshold {
		t.Errorf("perturbed cosine %f < threshold %f", cosPerturbed, DefaultLogitCosineParityThreshold)
	}
}

type stubRealRunner struct {
	called       int
	res          PhysicalTrialResult
	err          error
	mutateResult func(req PhysicalTrialRequest, res *PhysicalTrialResult)
}

func (s *stubRealRunner) ExecuteFanoutTrial(req PhysicalTrialRequest) (PhysicalTrialResult, error) {
	s.called++
	if s.err != nil {
		return PhysicalTrialResult{}, s.err
	}
	res := s.res
	res.RunIndex = req.RunIndex
	res.Concurrency = req.Concurrency
	res.Scenario = req.Scenario
	if s.mutateResult != nil {
		s.mutateResult(req, &res)
	}
	return res, nil
}

type stubModeledRunner struct {
	called int
	res    RunMetric
	err    error
}

func (s *stubModeledRunner) ExecuteModeledTrial(req ModeledTrialRequest) (RunMetric, error) {
	s.called++
	if s.err != nil {
		return RunMetric{}, s.err
	}
	res := s.res
	res.RunIndex = req.RunIndex
	res.Concurrency = req.Config.Concurrency
	res.Scenario = req.Config.Scenario
	return res, nil
}

// TestSubagentFanoutPhysicalModeFailsBefore proves physical mode invokes only the real interface,
// rejects modeled trial generation, and preserves explicit simulation provenance.
func TestSubagentFanoutPhysicalModeFailsBefore(t *testing.T) {
	validIdentity := ExecutionIdentity{
		SourceCommit:        "2a4e3e13431ecea885217ed8c5f161542e823039",
		SourceArchiveSHA256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
		BinarySHA256:        "a1b2c3d4e5f60123456789abcdef0123456789abcdef0123456789abcdef0123",
		ModelGGUFSHA256:     DefaultModelGGUFSHA256,
		TokenPacketSHA256:   "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
	}

	basePhasesMS := map[string]float64{
		PhaseHostDispatch:     0.15,
		PhasePrefixTreeLookup: 0.05,
		PhaseKVAllocation:     0.05,
		PhaseGPUKernel:        49.35,
		PhaseTokenSampling:    0.40,
	}

	validRealResult := PhysicalTrialResult{
		Backend:           "vulkan",
		ExecutionPath:     "fak-native-product",
		Identity:          validIdentity,
		WallDurationMS:    50.0,
		UsefulTokens:      64,
		TokensPerSec:      1280.0,
		QueueLatencyMS:    0.10,
		TTFTMS:            5.0,
		TPOTMS:            0.7,
		CounterSource:     CountersUnavailable,
		PhasesMS:          basePhasesMS,
		LogitCosineParity: 0.999995,
		ParityPassed:      true,
		FallbackCount:     0,
		FailureCount:      0,
	}

	validModeledMetric := RunMetric{
		WallDurationMS:    50.0,
		UsefulTokens:      64,
		TokensPerSec:      1280.0,
		DRAMBandwidthGBps: 40.0,
		MALLHitRate:       0.95,
		QueueLatencyMS:    0.10,
		CounterSource:     "calibrated_architecture_simulation",
		PhasesMS:          basePhasesMS,
		LogitCosineParity: 0.999995,
		ParityPassed:      true,
	}

	// 1. Inject real-runner stub and modeled-runner stub into physical mode
	realStub := &stubRealRunner{res: validRealResult}
	modeledStub := &stubModeledRunner{res: validModeledMetric}

	harness, err := NewSubagentFanoutHarness(FanoutConfig{
		Scenario:                   ScenarioSharedPrefixForked,
		Concurrency:                1,
		Runs:                       5,
		GeneratedTokensPerSubagent: 64,
		Simulated:                  false,
	})
	if err != nil {
		t.Fatalf("failed to create harness: %v", err)
	}
	harness.PhysicalRunner = realStub
	harness.ModeledRunner = modeledStub

	receipt, err := harness.Execute()
	if err != nil {
		t.Fatalf("physical execution failed: %v", err)
	}

	// Physical mode must call ONLY the real interface
	if realStub.called != 5 {
		t.Errorf("real runner called = %d, want 5", realStub.called)
	}
	if modeledStub.called != 0 {
		t.Errorf("modeled runner was called %d times during physical execution, want 0", modeledStub.called)
	}

	// Verify physical receipt contracts
	if receipt.Provenance != ProvenancePhysical {
		t.Errorf("provenance = %q, want %q", receipt.Provenance, ProvenancePhysical)
	}
	if receipt.Engine != CanonicalEngineName {
		t.Errorf("engine = %q, want %q", receipt.Engine, CanonicalEngineName)
	}
	if receipt.PrimaryEngine != CanonicalEngineName {
		t.Errorf("primary_engine = %q, want %q", receipt.PrimaryEngine, CanonicalEngineName)
	}
	if receipt.FallbackCount != 0 || !receipt.ZeroFallback {
		t.Errorf("fallback_count = %d, zero_fallback = %v", receipt.FallbackCount, receipt.ZeroFallback)
	}
	if receipt.Backend != "vulkan" {
		t.Errorf("backend = %q, want vulkan", receipt.Backend)
	}
	if receipt.ExecutionPath != "fak-native-product" {
		t.Errorf("execution_path = %q, want fak-native-product", receipt.ExecutionPath)
	}
	if receipt.ExecutionIdentity == nil || receipt.ExecutionIdentity.SourceCommit != validIdentity.SourceCommit {
		t.Errorf("execution identity mismatch: %+v", receipt.ExecutionIdentity)
	}

	// 2. Simulated mode preserves explicit simulation provenance and does not call real runner
	simHarness, err := NewSubagentFanoutHarness(FanoutConfig{
		Scenario:                   ScenarioSharedPrefixForked,
		Concurrency:                1,
		Runs:                       5,
		GeneratedTokensPerSubagent: 64,
		Simulated:                  true,
	})
	if err != nil {
		t.Fatalf("failed to create simulated harness: %v", err)
	}
	realStubSim := &stubRealRunner{res: validRealResult}
	modeledStubSim := &stubModeledRunner{res: validModeledMetric}
	simHarness.PhysicalRunner = realStubSim
	simHarness.ModeledRunner = modeledStubSim

	simReceipt, err := simHarness.Execute()
	if err != nil {
		t.Fatalf("simulated execution failed: %v", err)
	}
	if modeledStubSim.called != 5 {
		t.Errorf("modeled runner called = %d, want 5", modeledStubSim.called)
	}
	if realStubSim.called != 0 {
		t.Errorf("real runner called %d times in simulated mode, want 0", realStubSim.called)
	}
	if simReceipt.Provenance != ProvenanceSimulation {
		t.Errorf("simulated receipt provenance = %q, want %q", simReceipt.Provenance, ProvenanceSimulation)
	}
}

// TestSubagentFanoutPhysicalModeRejectsMissingTelemetry verifies fail-closed behavior on missing fields.
func TestSubagentFanoutPhysicalModeRejectsMissingTelemetry(t *testing.T) {
	validIdentity := ExecutionIdentity{
		SourceCommit:      "2a4e3e13431ecea885217ed8c5f161542e823039",
		BinarySHA256:      "a1b2c3d4e5f60123456789abcdef0123456789abcdef0123456789abcdef0123",
		ModelGGUFSHA256:   DefaultModelGGUFSHA256,
		TokenPacketSHA256: "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
	}

	basePhasesMS := map[string]float64{
		PhaseHostDispatch:     0.15,
		PhasePrefixTreeLookup: 0.05,
		PhaseKVAllocation:     0.05,
		PhaseGPUKernel:        49.35,
		PhaseTokenSampling:    0.40,
	}

	makeHarness := func(mutate func(res *PhysicalTrialResult)) *SubagentFanoutHarness {
		validRealResult := PhysicalTrialResult{
			Backend:           "vulkan",
			ExecutionPath:     "fak-native-product",
			Identity:          validIdentity,
			WallDurationMS:    50.0,
			UsefulTokens:      64,
			TokensPerSec:      1280.0,
			QueueLatencyMS:    0.10,
			CounterSource:     CountersUnavailable,
			PhasesMS:          basePhasesMS,
			LogitCosineParity: 0.999995,
			ParityPassed:      true,
			FallbackCount:     0,
			FailureCount:      0,
		}
		if mutate != nil {
			mutate(&validRealResult)
		}
		h, _ := NewSubagentFanoutHarness(FanoutConfig{
			Scenario:                   ScenarioSharedPrefixForked,
			Concurrency:                1,
			Runs:                       5,
			GeneratedTokensPerSubagent: 64,
			Simulated:                  false,
		})
		h.PhysicalRunner = &stubRealRunner{res: validRealResult}
		return h
	}

	// 1. Missing source commit
	h1 := makeHarness(func(res *PhysicalTrialResult) {
		res.Identity.SourceCommit = ""
	})
	if _, err := h1.Execute(); err == nil {
		t.Errorf("expected error on missing source commit")
	}

	// 2. Missing binary sha256
	h2 := makeHarness(func(res *PhysicalTrialResult) {
		res.Identity.BinarySHA256 = ""
	})
	if _, err := h2.Execute(); err == nil {
		t.Errorf("expected error on missing binary sha256")
	}

	// 3. Fallback count > 0
	h3 := makeHarness(func(res *PhysicalTrialResult) {
		res.FallbackCount = 1
	})
	if _, err := h3.Execute(); err == nil {
		t.Errorf("expected error on fallback count > 0")
	}

	// 4. Failure count > 0
	h4 := makeHarness(func(res *PhysicalTrialResult) {
		res.FailureCount = 1
	})
	if _, err := h4.Execute(); err == nil {
		t.Errorf("expected error on failure count > 0")
	}

	// 5. Hardware counters present without named counter source
	h5 := makeHarness(func(res *PhysicalTrialResult) {
		res.CounterSource = CountersUnavailable
		res.PhysicalDRAMBytes = 1024 * 1024
		res.DRAMBandwidthGBps = 40.0
	})
	if _, err := h5.Execute(); err == nil {
		t.Errorf("expected error when hardware counters present without named source")
	}

	// 6. Sub-threshold parity
	h6 := makeHarness(func(res *PhysicalTrialResult) {
		res.LogitCosineParity = 0.999800
		res.ParityPassed = false
	})
	if _, err := h6.Execute(); err == nil {
		t.Errorf("expected error on sub-threshold parity")
	}

	// 7. Missing backend
	h7 := makeHarness(func(res *PhysicalTrialResult) {
		res.Backend = ""
	})
	if _, err := h7.Execute(); err == nil {
		t.Errorf("expected error on missing backend")
	}
}

// TestSubagentFanoutPhysicalExecutionWithProductRunner verifies bounded B=1 physical execution.
func TestSubagentFanoutPhysicalExecutionWithProductRunner(t *testing.T) {
	runner := NewProductPhysicalRunner()
	cfg := FanoutConfig{
		Scenario:                   ScenarioSharedPrefixForked,
		Concurrency:                1,
		Runs:                       5,
		PrefixTokens:               1000,
		GeneratedTokensPerSubagent: 16,
		Simulated:                  false,
	}
	harness, err := NewSubagentFanoutHarness(cfg)
	if err != nil {
		t.Fatalf("failed to create harness: %v", err)
	}
	harness.PhysicalRunner = runner

	receipt, err := harness.Execute()
	if err != nil {
		t.Fatalf("physical execution with product runner failed: %v", err)
	}
	if err := receipt.Validate(); err != nil {
		t.Fatalf("physical receipt failed validation: %v", err)
	}

	if receipt.Provenance != ProvenancePhysical {
		t.Errorf("provenance = %q, want %q", receipt.Provenance, ProvenancePhysical)
	}
	if receipt.Engine != CanonicalEngineName {
		t.Errorf("engine = %q, want %q", receipt.Engine, CanonicalEngineName)
	}
	if receipt.PrimaryEngine != CanonicalEngineName {
		t.Errorf("primary_engine = %q, want %q", receipt.PrimaryEngine, CanonicalEngineName)
	}
	if receipt.FallbackCount != 0 {
		t.Errorf("fallback_count = %d, want 0", receipt.FallbackCount)
	}
	if !receipt.ZeroFallback {
		t.Errorf("zero_fallback = false, want true")
	}
	if receipt.ExecutionIdentity == nil {
		t.Fatalf("missing execution identity")
	}
	if receipt.ExecutionIdentity.SourceCommit == "" {
		t.Errorf("missing source commit")
	}
	if receipt.ExecutionIdentity.BinarySHA256 == "" {
		t.Errorf("missing binary sha256")
	}
	if receipt.ExecutionIdentity.ModelGGUFSHA256 != DefaultModelGGUFSHA256 {
		t.Errorf("model gguf sha = %q, want %q", receipt.ExecutionIdentity.ModelGGUFSHA256, DefaultModelGGUFSHA256)
	}
	if receipt.Summary.CountersStatus != CountersUnavailable {
		t.Errorf("counters status = %q, want %q", receipt.Summary.CountersStatus, CountersUnavailable)
	}
}
