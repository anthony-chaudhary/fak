package qwen38campaign

import (
	"testing"
)

var (
	benchSinkHw      HardwareInfo
	benchSinkReceipt SubagentFanoutReceipt
	benchSinkErr     error
)

func BenchmarkFanoutConfigValidate(b *testing.B) {
	configs := []FanoutConfig{
		{
			Scenario:                   ScenarioCold,
			Concurrency:                1,
			Runs:                       5,
			PrefixTokens:               100,
			GeneratedTokensPerSubagent: 16,
			Simulated:                  true,
		},
		{
			Scenario:                   ScenarioWarmSamePrefix,
			Concurrency:                4,
			Runs:                       5,
			PrefixTokens:               500,
			GeneratedTokensPerSubagent: 32,
			Simulated:                  true,
		},
		{
			Scenario:                   ScenarioSharedPrefixForked,
			Concurrency:                8,
			Runs:                       10,
			PrefixTokens:               1000,
			GeneratedTokensPerSubagent: 64,
			Simulated:                  true,
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cfg := configs[i%len(configs)]
		benchSinkErr = cfg.Validate()
	}
}

func BenchmarkDefaultHardwareInfo(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkHw = DefaultHardwareInfo()
	}
}

func BenchmarkRunCampaignSimulation(b *testing.B) {
	cfg := FanoutConfig{
		Scenario:                   ScenarioCold,
		Concurrency:                1,
		Runs:                       5,
		PrefixTokens:               100,
		GeneratedTokensPerSubagent: 8,
		Simulated:                  true,
		Seed:                       42,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		receipt, err := ExecuteSubagentFanoutBenchmark(cfg)
		if err != nil {
			b.Fatalf("ExecuteSubagentFanoutBenchmark failed: %v", err)
		}
		benchSinkReceipt = receipt
	}
}
