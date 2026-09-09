package macbench

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestValidateAgenticMTPPacket_NodeMacOSA(t *testing.T) {
	packet, raw, quality := NodeMacOSA24AgentMTPPacket()
	if err := ValidateAgenticMTPPacket(packet); err != nil {
		t.Fatalf("NodeMacOSA24AgentMTPPacket validation failed: %v", err)
	}

	if packet.Workload.Concurrency != 24 {
		t.Fatalf("expected concurrency 24, got %d", packet.Workload.Concurrency)
	}
	if packet.Summary.AggregateDecodeTokS < 300.0 {
		t.Fatalf("expected aggregate decode >= 300.0 tok/s, got %.2f", packet.Summary.AggregateDecodeTokS)
	}
	if packet.Summary.AcceptanceRate < 0.75 {
		t.Fatalf("expected acceptance rate >= 0.75, got %.3f", packet.Summary.AcceptanceRate)
	}
	if packet.Summary.DraftDepth < 3 || packet.Summary.DraftDepth > 4 {
		t.Fatalf("expected draft depth 3 or 4, got %d", packet.Summary.DraftDepth)
	}
	if !packet.Summary.ZeroFallback {
		t.Fatal("expected zero fallback")
	}
	if !packet.Summary.Verified {
		t.Fatal("expected summary verified to be true")
	}
	if packet.Provenance != "PHYSICAL_SILICON" || !packet.IsPhysicalSilicon {
		t.Fatalf("expected PHYSICAL_SILICON provenance, got %q (is_physical=%t)", packet.Provenance, packet.IsPhysicalSilicon)
	}

	_ = raw
	_ = quality
}

func TestValidateAgenticMTPPacket_FailsClosed(t *testing.T) {
	tests := []struct {
		name string
		edit func(*AgenticMTPPacket)
		want string
	}{
		{
			name: "wrong schema",
			edit: func(p *AgenticMTPPacket) { p.Schema = "fak.macbench.agentic-mtp.v2" },
			want: "schema",
		},
		{
			name: "invalid generated_at",
			edit: func(p *AgenticMTPPacket) { p.GeneratedAt = "not-a-timestamp" },
			want: "generated_at",
		},
		{
			name: "empty campaign_id",
			edit: func(p *AgenticMTPPacket) { p.CampaignID = "" },
			want: "campaign_id",
		},
		{
			name: "placeholder host_id",
			edit: func(p *AgenticMTPPacket) { p.HostID = strings.Repeat("0", 64) },
			want: "host_id",
		},
		{
			name: "non-physical provenance",
			edit: func(p *AgenticMTPPacket) { p.Provenance = "MODELED" },
			want: "provenance",
		},
		{
			name: "is_physical_silicon false",
			edit: func(p *AgenticMTPPacket) { p.IsPhysicalSilicon = false },
			want: "is_physical_silicon",
		},
		{
			name: "wrong engine",
			edit: func(p *AgenticMTPPacket) { p.Engine = "llama.cpp" },
			want: "engine",
		},
		{
			name: "wrong runtime",
			edit: func(p *AgenticMTPPacket) { p.Runtime = "reference" },
			want: "runtime",
		},
		{
			name: "non-zero fallback count",
			edit: func(p *AgenticMTPPacket) { p.FallbackCount = 1 },
			want: "fallback_count",
		},
		{
			name: "fallback named",
			edit: func(p *AgenticMTPPacket) { p.Fallback = "serial" },
			want: "fallback",
		},
		{
			name: "non-Qwen3.8 model family",
			edit: func(p *AgenticMTPPacket) { p.Model.Family = "Qwen3.6" },
			want: "model.family",
		},
		{
			name: "wrong model ID",
			edit: func(p *AgenticMTPPacket) { p.Model.ID = "llama-3-8b" },
			want: "model.id",
		},
		{
			name: "placeholder model canonical weights",
			edit: func(p *AgenticMTPPacket) { p.Model.CanonicalWeightsSHA256 = strings.Repeat("f", 64) },
			want: "model.canonical_weights_sha256",
		},
		{
			name: "unsupported quantization",
			edit: func(p *AgenticMTPPacket) { p.Model.Quant = "Q8_0" },
			want: "model.quant",
		},
		{
			name: "wrong hardware model",
			edit: func(p *AgenticMTPPacket) { p.Hardware.Model = "MacBookAir10,1" },
			want: "hardware.model",
		},
		{
			name: "wrong chip",
			edit: func(p *AgenticMTPPacket) { p.Hardware.Chip = "Apple M1" },
			want: "hardware.chip",
		},
		{
			name: "insufficient host memory",
			edit: func(p *AgenticMTPPacket) { p.Hardware.MemoryBytes = 16 * 1024 * 1024 * 1024 },
			want: "hardware.memory_bytes",
		},
		{
			name: "non-macOS operating system",
			edit: func(p *AgenticMTPPacket) { p.OS.Name = "Linux" },
			want: "os.name",
		},
		{
			name: "concurrency below 24",
			edit: func(p *AgenticMTPPacket) { p.Workload.Concurrency = 20 },
			want: "workload.concurrency",
		},
		{
			name: "concurrency above 24",
			edit: func(p *AgenticMTPPacket) { p.Workload.Concurrency = 32 },
			want: "workload.concurrency",
		},
		{
			name: "draft depth below 3",
			edit: func(p *AgenticMTPPacket) { p.SpeculativeConfig.DraftDepth = 2 },
			want: "speculative_config.draft_depth",
		},
		{
			name: "draft depth above 4",
			edit: func(p *AgenticMTPPacket) { p.SpeculativeConfig.DraftDepth = 5 },
			want: "speculative_config.draft_depth",
		},
		{
			name: "non-deterministic temperature",
			edit: func(p *AgenticMTPPacket) { p.SpeculativeConfig.Temperature = 0.7 },
			want: "speculative_config.temperature",
		},
		{
			name: "resident memory exceeds 28GB bound",
			edit: func(p *AgenticMTPPacket) { p.Memory.ResidentWorkingSetGB = 29.5 },
			want: "memory.resident_working_set_gb",
		},
		{
			name: "host headroom below 8GB limit",
			edit: func(p *AgenticMTPPacket) { p.Memory.HostHeadroomGB = 6.0 },
			want: "memory.host_headroom_gb",
		},
		{
			name: "shared prefix deduplication false",
			edit: func(p *AgenticMTPPacket) { p.Memory.SharedPrefixDeduplicated = false },
			want: "memory.shared_prefix_deduplicated",
		},
		{
			name: "gdn discount ratio below threshold",
			edit: func(p *AgenticMTPPacket) { p.Memory.GDNDiscountRatio = 1.0 },
			want: "memory.gdn_discount_ratio",
		},
		{
			name: "quality policy placeholder digest",
			edit: func(p *AgenticMTPPacket) { p.QualityPolicy.SHA256 = strings.Repeat("a", 64) },
			want: "quality_policy.sha256",
		},
		{
			name: "quality passed false",
			edit: func(p *AgenticMTPPacket) { p.Quality.Passed = false },
			want: "quality.passed",
		},
		{
			name: "quality score below minimum",
			edit: func(p *AgenticMTPPacket) { p.Quality.Score = 0.5 },
			want: "quality.score",
		},
		{
			name: "streams count mismatch",
			edit: func(p *AgenticMTPPacket) { p.Streams = p.Streams[:20] },
			want: "streams",
		},
		{
			name: "stream acceptance rate math error",
			edit: func(p *AgenticMTPPacket) { p.Streams[0].AcceptanceRate = 0.1 },
			want: "streams[0].acceptance_rate",
		},
		{
			name: "stream ITL p95 below p50",
			edit: func(p *AgenticMTPPacket) { p.Streams[0].P95ITLMS = p.Streams[0].P50ITLMS - 1.0 },
			want: "streams[0].p95_itl_ms",
		},
		{
			name: "aggregate decode throughput below 300 tok/s",
			edit: func(p *AgenticMTPPacket) {
				p.Summary.AggregateDecodeTokS = 290.0
				p.Metrics.Decode.AggregateTokS.P50 = 290.0
			},
			want: "summary.aggregate_decode_tok_s",
		},
		{
			name: "acceptance rate below 0.75 floor",
			edit: func(p *AgenticMTPPacket) {
				p.Summary.AcceptanceRate = 0.72
				p.Metrics.Decode.AcceptanceRate.P50 = 0.72
			},
			want: "summary.acceptance_rate",
		},
		{
			name: "summary draft depth mismatch",
			edit: func(p *AgenticMTPPacket) { p.Summary.DraftDepth = 4; p.SpeculativeConfig.DraftDepth = 3 },
			want: "summary.draft_depth",
		},
		{
			name: "summary unverified",
			edit: func(p *AgenticMTPPacket) { p.Summary.Verified = false },
			want: "summary.verified",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			packet, _, _ := NodeMacOSA24AgentMTPPacket()
			tt.edit(&packet)
			err := ValidateAgenticMTPPacket(packet)
			if err == nil {
				t.Fatalf("expected validation error containing %q, got nil", tt.want)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("expected error containing %q, got: %v", tt.want, err)
			}
		})
	}
}

func TestValidateAgenticMTPPacket_EvidenceVerification(t *testing.T) {
	tempDir := t.TempDir()
	packet, raw, quality := NodeMacOSA24AgentMTPPacket()

	if err := WriteAgenticMTPRun(tempDir, packet, raw, quality); err != nil {
		t.Fatalf("WriteAgenticMTPRun failed: %v", err)
	}

	packetPath := filepath.Join(tempDir, "packet.json")
	rawBytes, err := os.ReadFile(packetPath)
	if err != nil {
		t.Fatal(err)
	}

	var loadedPacket AgenticMTPPacket
	if err := json.Unmarshal(rawBytes, &loadedPacket); err != nil {
		t.Fatalf("unmarshal written packet: %v", err)
	}

	// 1. Valid evidence pass
	if err := ValidateAgenticMTPEvidence(loadedPacket, packetPath); err != nil {
		t.Fatalf("ValidateAgenticMTPEvidence failed on untampered run: %v", err)
	}

	// 2. Tampered raw result file fails closed
	rawPath := filepath.Join(tempDir, loadedPacket.RawResult.Path)
	tamperedRaw := append(rawBytes, []byte("\n// tampering")...)
	if err := os.WriteFile(rawPath, tamperedRaw, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := ValidateAgenticMTPEvidence(loadedPacket, packetPath); err == nil {
		t.Fatal("expected evidence validation error on tampered raw file, got nil")
	}

	// Restore raw file
	rawJsonBytes, _ := json.MarshalIndent(raw, "", "  ")
	_ = os.WriteFile(rawPath, rawJsonBytes, 0o644)

	// 3. Tampered quality result file fails closed
	qualityPath := filepath.Join(tempDir, loadedPacket.Quality.ResultPath)
	tamperedQuality := []byte(`{"invalid": true}`)
	if err := os.WriteFile(qualityPath, tamperedQuality, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := ValidateAgenticMTPEvidence(loadedPacket, packetPath); err == nil {
		t.Fatal("expected evidence validation error on tampered quality file, got nil")
	}
}

func TestValidateAgenticMTPPacket_OnDiskRun(t *testing.T) {
	// Locate packet in repository
	candidatePaths := []string{
		filepath.Join("..", "..", "experiments", "benchmark", "runs", "by-machine", "node-macos-a", "20260908T170000Z-macbench-agentic-mtp", "packet.json"),
		filepath.Join("experiments", "benchmark", "runs", "by-machine", "node-macos-a", "20260908T170000Z-macbench-agentic-mtp", "packet.json"),
	}

	var diskPath string
	for _, p := range candidatePaths {
		if _, err := os.Stat(p); err == nil {
			diskPath = p
			break
		}
	}
	if diskPath == "" {
		t.Skip("on-disk run packet not found, skipping disk witness check")
	}

	raw, err := os.ReadFile(diskPath)
	if err != nil {
		t.Fatalf("read on-disk packet: %v", err)
	}

	var packet AgenticMTPPacket
	if err := decodeStrictAgenticMTPJSON(raw, &packet); err != nil {
		t.Fatalf("decode on-disk packet: %v", err)
	}

	if err := ValidateAgenticMTPEvidence(packet, diskPath); err != nil {
		t.Fatalf("ValidateAgenticMTPEvidence on disk packet failed: %v", err)
	}

	if packet.Summary.AggregateDecodeTokS < 300.0 {
		t.Fatalf("on-disk packet aggregate decode tok/s = %.2f, want >= 300.0", packet.Summary.AggregateDecodeTokS)
	}
	if packet.Summary.Concurrency != 24 {
		t.Fatalf("on-disk packet concurrency = %d, want 24", packet.Summary.Concurrency)
	}
	if packet.Summary.DraftDepth < 3 || packet.Summary.DraftDepth > 4 {
		t.Fatalf("on-disk packet draft depth = %d, want 3 or 4", packet.Summary.DraftDepth)
	}
	if packet.Summary.AcceptanceRate < 0.75 {
		t.Fatalf("on-disk packet acceptance rate = %.3f, want >= 0.75", packet.Summary.AcceptanceRate)
	}
	if !packet.Summary.ZeroFallback {
		t.Fatal("on-disk packet must prove zero fallback")
	}
}

func TestRunAgenticMTP_OptionsValidation(t *testing.T) {
	opts := DefaultAgenticMTPOptions()
	packet, raw, quality, err := RunAgenticMTP(nil, opts)
	if err != nil {
		t.Fatalf("RunAgenticMTP default options failed: %v", err)
	}
	if packet.Summary.Concurrency != 24 {
		t.Fatalf("expected concurrency 24, got %d", packet.Summary.Concurrency)
	}
	if len(raw.Streams) != 24 {
		t.Fatalf("expected 24 raw streams, got %d", len(raw.Streams))
	}
	if quality.Passed != true {
		t.Fatal("expected quality passed")
	}

	// Invalid concurrency
	badOpts := opts
	badOpts.Concurrency = 12
	if _, _, _, err := RunAgenticMTP(nil, badOpts); err == nil {
		t.Fatal("expected error on concurrency != 24, got nil")
	}

	// Invalid draft depth
	badDepthOpts := opts
	badDepthOpts.DraftDepth = 1
	if _, _, _, err := RunAgenticMTP(nil, badDepthOpts); err == nil {
		t.Fatal("expected error on draft depth 1, got nil")
	}
}
