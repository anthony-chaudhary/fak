package macbench

import (
	"strings"
	"testing"
)

func validAgenticComparisonPacket() AgenticComparisonPacket {
	hardware := ComparisonHardware{
		Model:       "Mac15,7",
		Chip:        "Apple M3 Pro",
		MemoryBytes: 38654705664, // 36 GiB
	}
	osInfo := ComparisonOS{
		Name:    "macOS",
		Version: "26.6.2",
		Build:   "25G83",
	}
	qualityPolicy := ComparisonQualityPolicy{
		ID:           "strict-token-parity",
		Version:      "1",
		SHA256:       strings.Repeat("8", 64),
		MinimumScore: 1.0,
	}

	workload := AgenticWorkloadShape{
		Concurrency:        4,
		Horizon:            20,
		SharedPrefixTokens: 4096,
		TurnDeltaTokens:    128,
		TurnOutputTokens:   64,
	}

	fakArm := AgenticComparisonArm{
		Name:              "fak-native",
		Engine:            "fak-native",
		Runtime:           "inkernel",
		RuntimeRevision:   "r652+g839b1d44",
		EvidenceKind:      "modeled",
		PrefixStrategy:    "radix-shared",
		PrefixEvalCount:   1,
		PromptTokens:      483840,
		ReusedTokens:      469504,
		ReuseRatio:        0.97037,
		TotalWallMS:       410900.0,
		PrefillMS:         182400.0,
		DecodeMS:          228500.0,
		QueueContentionMS: 1600.0,
		P50TTFTMS:         12.6,
		P95TTFTMS:         12.9,
		PeakMemoryMB:      22208.0,
		AgentsPerGB:       0.18,
		EffectiveTokS:     12.46,
		Quality: ComparisonQualityResult{
			PolicyRef:     "strict-token-parity",
			PolicyVersion: "1",
			PolicySHA256:  strings.Repeat("8", 64),
			Passed:        true,
			Score:         1.0,
			ResultPath:    "fak-native-quality.json",
			ResultSHA256:  strings.Repeat("a", 64),
		},
		RawResult: ComparisonRawResult{
			Path:   "fak-native-raw.json",
			SHA256: strings.Repeat("b", 64),
		},
		Repro: []string{"go run ./cmd/fak macbench many-agent -c 4 --model Qwen3.8-27B --horizon 20 --json"},
	}

	llamaArm := AgenticComparisonArm{
		Name:              "llama.cpp",
		Engine:            "llama.cpp",
		Runtime:           "reference",
		RuntimeRevision:   "b9828",
		EvidenceKind:      "modeled",
		PrefixStrategy:    "slot-isolated",
		PrefixEvalCount:   4,
		PromptTokens:      483840,
		ReusedTokens:      0,
		ReuseRatio:        0.0,
		TotalWallMS:       786100.0,
		PrefillMS:         504800.0,
		DecodeMS:          281300.0,
		QueueContentionMS: 946400.0,
		P50TTFTMS:         84480.0,
		P95TTFTMS:         253440.0,
		PeakMemoryMB:      25792.0,
		AgentsPerGB:       0.16,
		EffectiveTokS:     6.51,
		Quality: ComparisonQualityResult{
			PolicyRef:     "strict-token-parity",
			PolicyVersion: "1",
			PolicySHA256:  strings.Repeat("8", 64),
			Passed:        true,
			Score:         1.0,
			ResultPath:    "llama.cpp-quality.json",
			ResultSHA256:  strings.Repeat("c", 64),
		},
		RawResult: ComparisonRawResult{
			Path:   "llama.cpp-raw.json",
			SHA256: strings.Repeat("d", 64),
		},
		Repro: []string{"python3 internal/model/bench_llamacpp_turn_agents.py --turns 20 --agents 4 --prefix 4096"},
	}

	return AgenticComparisonPacket{
		Schema:            AgenticComparisonSchema,
		GeneratedAt:       "2026-09-05T12:00:00Z",
		CampaignID:        "issue-3809-mac-agentic-4x-qwen38-20260905",
		HostID:            strings.Repeat("6", 64),
		Provenance:        "MODELED",
		IsPhysicalSilicon: false,
		UnmodeledEffects: []string{
			"thermal_dvfs_throttling",
			"memory_bus_contention",
			"metal_command_buffer_sync_latency",
		},
		Model: ComparisonModel{
			Family:                 "Qwen3.8",
			ID:                     "Qwen3.8-27B",
			SourceRevision:         "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
			CanonicalWeightsSHA256: "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169",
			Quant:                  "Q4_K_M",
		},
		Hardware:      hardware,
		OS:            osInfo,
		Workload:      workload,
		QualityPolicy: qualityPolicy,
		Arms:          []AgenticComparisonArm{fakArm, llamaArm},
		Summary: AgenticSummary{
			SpeedupRatio:   1.91,
			MemorySavedMB:  3584.0,
			TTFTSpeedupP50: 6704.76,
			Verified:       true,
		},
		MinSpeedupRatio: 1.50,
	}
}

func TestValidateAgenticComparisonPacket_HappyPath(t *testing.T) {
	packet := validAgenticComparisonPacket()
	if err := ValidateAgenticComparisonPacket(packet); err != nil {
		t.Fatalf("ValidateAgenticComparisonPacket failed on valid packet: %v", err)
	}
}

func TestValidateAgenticComparisonPacket_LegacyAccounting(t *testing.T) {
	packet := validAgenticComparisonPacket()
	for i := range packet.Arms {
		packet.Arms[i].TotalWallMS = packet.Arms[i].PrefillMS + packet.Arms[i].DecodeMS + packet.Arms[i].QueueContentionMS
	}
	packet.Summary.SpeedupRatio = packet.Arms[1].TotalWallMS / packet.Arms[0].TotalWallMS
	if err := ValidateAgenticComparisonPacket(packet); err != nil {
		t.Fatalf("expected legacy packet with queue contention in total_wall_ms to pass: %v", err)
	}
}

func TestValidateAgenticComparisonPacket_FailsClosed(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(*AgenticComparisonPacket)
		wantErr string
	}{
		{
			name: "wrong schema",
			mutate: func(p *AgenticComparisonPacket) {
				p.Schema = "fak.macbench.comparison.v1"
			},
			wantErr: "schema",
		},
		{
			name: "invalid provenance",
			mutate: func(p *AgenticComparisonPacket) {
				p.Provenance = "MEASURED_UNVERIFIED"
			},
			wantErr: "provenance",
		},
		{
			name: "modeled without unmodeled effects",
			mutate: func(p *AgenticComparisonPacket) {
				p.UnmodeledEffects = nil
			},
			wantErr: "unmodeled_effects",
		},
		{
			name: "non-Qwen3.8 model family",
			mutate: func(p *AgenticComparisonPacket) {
				p.Model.Family = "Qwen3.6"
			},
			wantErr: "model.family",
		},
		{
			name: "non-Qwen3.8 model ID",
			mutate: func(p *AgenticComparisonPacket) {
				p.Model.ID = "Llama-3.3-70B"
			},
			wantErr: "model.id",
		},
		{
			name: "quantization mismatch",
			mutate: func(p *AgenticComparisonPacket) {
				p.Model.Quant = "Q8_0"
			},
			wantErr: "model.quant",
		},
		{
			name: "invalid canonical weights sha256",
			mutate: func(p *AgenticComparisonPacket) {
				p.Model.CanonicalWeightsSHA256 = "not-a-valid-sha"
			},
			wantErr: "model.canonical_weights_sha256",
		},
		{
			name: "concurrency below minimum",
			mutate: func(p *AgenticComparisonPacket) {
				p.Workload.Concurrency = 1
			},
			wantErr: "workload.concurrency",
		},
		{
			name: "horizon below minimum",
			mutate: func(p *AgenticComparisonPacket) {
				p.Workload.Horizon = 2
			},
			wantErr: "workload.horizon",
		},
		{
			name: "shared prefix below minimum",
			mutate: func(p *AgenticComparisonPacket) {
				p.Workload.SharedPrefixTokens = 512
			},
			wantErr: "workload.shared_prefix_tokens",
		},
		{
			name: "missing fak-native arm",
			mutate: func(p *AgenticComparisonPacket) {
				p.Arms = p.Arms[1:] // only llama.cpp
			},
			wantErr: "arms",
		},
		{
			name: "fak-native wrong runtime",
			mutate: func(p *AgenticComparisonPacket) {
				p.Arms[0].Runtime = "gateway"
			},
			wantErr: "fak-native.runtime",
		},
		{
			name: "fak-native prefix evaluated multiple times",
			mutate: func(p *AgenticComparisonPacket) {
				p.Arms[0].PrefixEvalCount = 4
			},
			wantErr: "fak-native.prefix_eval_count",
		},
		{
			name: "fak-native TTFT p50 regressed (> 25ms)",
			mutate: func(p *AgenticComparisonPacket) {
				p.Arms[0].P50TTFTMS = 150.0
			},
			wantErr: "fak-native.p50_ttft_ms",
		},
		{
			name: "fak-native zero token reuse",
			mutate: func(p *AgenticComparisonPacket) {
				p.Arms[0].ReusedTokens = 0
				p.Arms[0].ReuseRatio = 0.0
			},
			wantErr: "fak-native.reused_tokens",
		},
		{
			name: "llama.cpp wrong runtime",
			mutate: func(p *AgenticComparisonPacket) {
				p.Arms[1].Runtime = "inkernel"
			},
			wantErr: "llama.cpp.runtime",
		},
		{
			name: "speedup ratio below 1.50x threshold",
			mutate: func(p *AgenticComparisonPacket) {
				p.Summary.SpeedupRatio = 1.20
				p.Arms[1].TotalWallMS = p.Arms[0].TotalWallMS * 1.20
				p.Arms[1].PrefillMS = p.Arms[1].TotalWallMS - p.Arms[1].DecodeMS
			},
			wantErr: "summary.speedup_ratio",
		},
		{
			name: "boundary accounting mismatch in fak-native",
			mutate: func(p *AgenticComparisonPacket) {
				p.Arms[0].PrefillMS += 50000.0 // makes sum != total_wall_ms
			},
			wantErr: "fak-native.boundary",
		},
		{
			name: "quality failed on arm",
			mutate: func(p *AgenticComparisonPacket) {
				p.Arms[0].Quality.Passed = false
			},
			wantErr: "quality.passed",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			packet := validAgenticComparisonPacket()
			tc.mutate(&packet)
			err := ValidateAgenticComparisonPacket(packet)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected error containing %q, got: %v", tc.wantErr, err)
			}
		})
	}
}
