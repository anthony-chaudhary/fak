package main

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func TestBenchSubagentFanoutHelp(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runBenchSubagentFanout(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Fatalf("expected exit 0 for --help, got %d, stderr=%s", code, stderr.String())
	}
	for _, want := range []string{
		"-model",
		"-quant",
		"-mem-fraction",
		"-fanout",
		"-arms",
		"-verify-contract",
	} {
		if !strings.Contains(stderr.String(), want) {
			t.Errorf("help output missing flag %q:\n%s", want, stderr.String())
		}
	}
}

func TestBenchSubagentFanoutContractValidation(t *testing.T) {
	t.Run("CompliantConfig", func(t *testing.T) {
		cfg := &FanoutBenchConfig{
			Model:          "Qwen/Qwen2.5-Coder-7B-Instruct",
			Quantization:   "Q4_K_M",
			MemoryFraction: 0.85,
			FanoutSweep:    []int{1, 4, 8, 16, 32},
			Arms:           []string{"no_reuse", "sglang", "vllm", "fak"},
		}
		val := validateContractInvariants(cfg)
		if !val.Compliant {
			t.Fatalf("expected config to be compliant, violations: %v", val.Violations)
		}
		if !val.AllFourArmsPresent {
			t.Errorf("expected all 4 arms present")
		}
		if !val.FixedMemoryFractionVerified {
			t.Errorf("expected fixed memory fraction 0.85 verified")
		}
		if !val.FanoutSweepVerified {
			t.Errorf("expected fanout sweep [1,4,8,16,32] verified")
		}
	})

	t.Run("MissingArmViolation", func(t *testing.T) {
		cfg := &FanoutBenchConfig{
			Model:          "Qwen/Qwen2.5-Coder-7B-Instruct",
			Quantization:   "Q4_K_M",
			MemoryFraction: 0.85,
			FanoutSweep:    []int{1, 4, 8, 16, 32},
			Arms:           []string{"no_reuse", "sglang", "fak"}, // missing vllm
		}
		val := validateContractInvariants(cfg)
		if val.Compliant {
			t.Fatal("expected compliance failure when arm is missing")
		}
		if val.AllFourArmsPresent {
			t.Error("AllFourArmsPresent should be false")
		}
		foundViolation := false
		for _, v := range val.Violations {
			if strings.Contains(v, "vllm") {
				foundViolation = true
				break
			}
		}
		if !foundViolation {
			t.Errorf("expected violation mentioning vllm, got: %v", val.Violations)
		}
	})

	t.Run("InvalidMemoryFractionViolation", func(t *testing.T) {
		cfg := &FanoutBenchConfig{
			Model:          "Qwen/Qwen2.5-Coder-7B-Instruct",
			Quantization:   "Q4_K_M",
			MemoryFraction: 0.90, // Contract requires 0.85
			FanoutSweep:    []int{1, 4, 8, 16, 32},
			Arms:           []string{"no_reuse", "sglang", "vllm", "fak"},
		}
		val := validateContractInvariants(cfg)
		if val.Compliant {
			t.Fatal("expected compliance failure when memory fraction != 0.85")
		}
		if val.FixedMemoryFractionVerified {
			t.Error("FixedMemoryFractionVerified should be false")
		}
	})

	t.Run("IncompleteFanoutSweepViolation", func(t *testing.T) {
		cfg := &FanoutBenchConfig{
			Model:          "Qwen/Qwen2.5-Coder-7B-Instruct",
			Quantization:   "Q4_K_M",
			MemoryFraction: 0.85,
			FanoutSweep:    []int{1, 4, 8}, // missing 16, 32
			Arms:           []string{"no_reuse", "sglang", "vllm", "fak"},
		}
		val := validateContractInvariants(cfg)
		if val.Compliant {
			t.Fatal("expected compliance failure when sweep is incomplete")
		}
		if val.FanoutSweepVerified {
			t.Error("FanoutSweepVerified should be false")
		}
	})
}

func TestBenchSubagentFanoutSimulationRunJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{
		"-fanout", "1,4,8,16,32",
		"-arms", "no_reuse,sglang,vllm,fak",
		"-model", "Qwen/Qwen2.5-Coder-7B-Instruct",
		"-quant", "Q4_K_M",
		"-mem-fraction", "0.85",
		"-trials", "2",
		"-json",
	}

	code := runBenchSubagentFanout(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("runBenchSubagentFanout failed with code %d, stderr: %s", code, stderr.String())
	}

	var receipt SubagentFanoutReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to decode receipt JSON: %v\nOutput: %s", err, stdout.String())
	}

	if receipt.Schema != SubagentFanoutComparisonSchema {
		t.Errorf("schema = %q, want %q", receipt.Schema, SubagentFanoutComparisonSchema)
	}
	if receipt.ContractIssue != ContractIssue6036 {
		t.Errorf("contract issue = %q, want %q", receipt.ContractIssue, ContractIssue6036)
	}
	if !receipt.Contract.Compliant {
		t.Errorf("contract audit should pass, violations: %v", receipt.Contract.Violations)
	}
	if math.Abs(receipt.MemoryFraction-0.85) > 1e-4 {
		t.Errorf("memory fraction = %f, want 0.85", receipt.MemoryFraction)
	}

	// Must contain 5 fanout values * 4 arms = 20 cells
	wantCells := len(CanonicalFanoutSweep) * len(CanonicalArms)
	if len(receipt.Results) != wantCells {
		t.Fatalf("got %d results, want %d", len(receipt.Results), wantCells)
	}

	for _, res := range receipt.Results {
		if res.Error != "" {
			t.Errorf("cell N=%d Arm=%s returned error: %s", res.FanoutN, res.Arm, res.Error)
		}
		// TTFT percentiles order check
		if res.TTFT.P50 > res.TTFT.P95 || res.TTFT.P95 > res.TTFT.P99 {
			t.Errorf("invalid TTFT percentile ordering for N=%d Arm=%s: p50=%.2f, p95=%.2f, p99=%.2f",
				res.FanoutN, res.Arm, res.TTFT.P50, res.TTFT.P95, res.TTFT.P99)
		}
		// ITL jitter check
		if res.ITL.JitterMs < 0 {
			t.Errorf("negative ITL jitter for N=%d Arm=%s: %f", res.FanoutN, res.Arm, res.ITL.JitterMs)
		}
		// Hit rate check for no-reuse arm
		if res.Arm == ArmNoReuse && res.PrefixHitRate != 0.0 {
			t.Errorf("ArmNoReuse should have 0.0 hit rate, got %f", res.PrefixHitRate)
		}
		// At N > 1, reuse arms must have positive hit rate
		if res.Arm != ArmNoReuse && res.FanoutN > 1 && res.PrefixHitRate <= 0.0 {
			t.Errorf("Arm %s at N=%d should have positive hit rate, got %f", res.Arm, res.FanoutN, res.PrefixHitRate)
		}
	}
}

func TestPercentileCalculation(t *testing.T) {
	sorted := []float64{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}

	p50 := calcPercentile(sorted, 50.0)
	if math.Abs(p50-55.0) > 1e-4 {
		t.Errorf("p50 = %f, want 55.0", p50)
	}

	p0 := calcPercentile(sorted, 0.0)
	if math.Abs(p0-10.0) > 1e-4 {
		t.Errorf("p0 = %f, want 10.0", p0)
	}

	p100 := calcPercentile(sorted, 100.0)
	if math.Abs(p100-100.0) > 1e-4 {
		t.Errorf("p100 = %f, want 100.0", p100)
	}
}

func TestDistributionComputations(t *testing.T) {
	samples := []float64{10.0, 12.0, 14.0, 16.0, 18.0, 20.0}
	dist := computeDistribution(samples)
	if dist.Count != 6 {
		t.Errorf("count = %d, want 6", dist.Count)
	}
	if math.Abs(dist.Mean-15.0) > 1e-4 {
		t.Errorf("mean = %f, want 15.0", dist.Mean)
	}
	if dist.Min != 10.0 || dist.Max != 20.0 {
		t.Errorf("min/max = %f/%f, want 10/20", dist.Min, dist.Max)
	}

	itl := computeITLStats(samples)
	if itl.Count != 6 {
		t.Errorf("count = %d, want 6", itl.Count)
	}
	if itl.JitterMs <= 0 {
		t.Errorf("jitter should be positive, got %f", itl.JitterMs)
	}
}

func TestPrettyRenderOutput(t *testing.T) {
	cfg := &FanoutBenchConfig{
		Model:          "Qwen/Qwen2.5-Coder-7B-Instruct",
		Quantization:   "Q4_K_M",
		MemoryFraction: 0.85,
		FanoutSweep:    []int{1, 4},
		Arms:           []string{"no_reuse", "fak"},
		Trials:         1,
		PrefixTokens:   1024,
		SuffixTokens:   128,
		DecodeTokens:   16,
		Seed:           42,
	}
	harness := NewFanoutBenchmarkHarness(cfg)
	receipt, err := harness.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	receipt.Contract = validateContractInvariants(cfg)

	var buf bytes.Buffer
	renderPrettyReceipt(&buf, receipt)
	output := buf.String()

	if !strings.Contains(output, "APPLES-TO-APPLES SUBAGENT FANOUT BENCHMARK HARNESS") {
		t.Errorf("missing header in pretty render:\n%s", output)
	}
	if !strings.Contains(output, "No-reuse baseline") || !strings.Contains(output, "FAK native engine") {
		t.Errorf("missing arm labels in pretty render:\n%s", output)
	}
}
