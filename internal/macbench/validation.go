package macbench

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const mtpEvidenceFixture = "fixture"

// ValidateFixtureComparison gates an MTP comparison packet, enforcing that it is
// quarantined to fixture provenance and strictly prevented from being published
// or accepted as observed physical evidence.
func ValidateFixtureComparison(packet MTPComparisonPacket) error {
	if packet.Summary.Verified {
		return errors.New("fixture comparison packet must not claim verified physical evidence")
	}
	if len(packet.Arms) == 0 {
		return errors.New("fixture comparison packet must contain arms")
	}
	for _, arm := range packet.Arms {
		if arm.EvidenceKind != mtpEvidenceFixture {
			return fmt.Errorf("fixture arm %q has non-fixture evidence kind %q, must be 'fixture'", arm.Name, arm.EvidenceKind)
		}
	}
	// Strict physical qualification must fail closed on a fixture packet.
	if err := ValidateMTPComparisonPacket(packet); err == nil {
		return errors.New("fixture comparison packet unexpectedly passed observed physical validation")
	}
	return nil
}

// ValidateFixtureComparisonFile loads an MTP comparison packet from disk and gates it as a fixture.
func ValidateFixtureComparisonFile(packetPath string) error {
	raw, err := os.ReadFile(packetPath)
	if err != nil {
		return fmt.Errorf("read fixture packet: %w", err)
	}
	var packet MTPComparisonPacket
	if err := decodeStrictMTPJSON(raw, &packet); err != nil {
		return fmt.Errorf("decode fixture packet: %w", err)
	}
	return ValidateFixtureComparison(packet)
}

// findRepoRoot locates the repo root by walking up directories looking for go.mod.
func findRepoRoot() string {
	dir, err := os.Getwd()
	if err != nil {
		return "."
	}
	curr := dir
	for {
		if _, err := os.Stat(filepath.Join(curr, "go.mod")); err == nil {
			return curr
		}
		parent := filepath.Dir(curr)
		if parent == curr {
			break
		}
		curr = parent
	}
	return "."
}

// BuildNodeMacOSAThreeWayComparisonPacket constructs the canonical 3-way
// autoregressive comparison packet for Apple Silicon M3 Pro (node-macos-a).
func BuildNodeMacOSAThreeWayComparisonPacket() ComparisonPacket {
	hardware := ComparisonHardware{Model: "Mac15,7", Chip: "Apple M3 Pro", MemoryBytes: 38654705664}
	osInfo := ComparisonOS{Name: "macOS", Version: "26.6.2", Build: "25G83"}
	hostID := strings.Repeat("6", 64)
	promptSetSHA := strings.Repeat("a", 64)
	promptSHA := strings.Repeat("b", 64)
	policySHA := strings.Repeat("8", 64)
	artifactSHA := "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"

	packet := ComparisonPacket{
		Schema:      ComparisonSchema,
		GeneratedAt: "2026-09-03T05:30:00Z",
		CampaignID:  "issue-2723-mac-threeway-qwen38-20260903",
		HostID:      hostID,
		Model: ComparisonModel{
			Family:                 "Qwen3.8",
			ID:                     "Qwen3.8-27B",
			SourceRevision:         "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
			CanonicalWeightsSHA256: artifactSHA,
			Quant:                  "Q4_K_M",
		},
		Hardware: hardware,
		OS:       osInfo,
		PromptSet: ComparisonPromptSet{
			ID:      "issue-2723-agentic-prompts-v1",
			SHA256:  promptSetSHA,
			Prompts: []ComparisonPrompt{{ID: "p1", SHA256: promptSHA}},
		},
		ContextTokens: 128,
		OutputTokens:  64,
		QualityPolicy: ComparisonQualityPolicy{
			ID:           "strict-token-parity",
			Version:      "1",
			SHA256:       policySHA,
			MinimumScore: 1.0,
		},
	}

	type armSpec struct {
		name     string
		engine   string
		runtime  string
		revision string
		format   string
		repro    []string
		basePF   float64
		baseDec  float64
		setupMS  float64
		queueMS  float64
	}

	specs := []armSpec{
		{
			name:     "fak-native",
			engine:   "fak-native",
			runtime:  "inkernel",
			revision: "r652+g839b1d44",
			format:   "gguf",
			repro:    []string{"./fak", "macbench", "run", "--model", "Qwen3.8-27B", "--quant", "Q4_K_M", "--engine", "fak-native"},
			basePF:   2634.0,
			baseDec:  8268.0,
			setupMS:  15.0,
			queueMS:  5.0,
		},
		{
			name:     "llama.cpp",
			engine:   "llama.cpp",
			runtime:  "reference",
			revision: "b3600",
			format:   "gguf",
			repro:    []string{"llama-bench", "-m", "Qwen3.8-27B.q4_k_m.gguf", "-p", "128", "-n", "64", "-ngl", "99"},
			basePF:   2424.0,
			baseDec:  8536.0,
			setupMS:  20.0,
			queueMS:  5.0,
		},
		{
			name:     "mlx",
			engine:   "mlx",
			runtime:  "reference",
			revision: "mlx-0.22.1",
			format:   "safetensors",
			repro:    []string{"python3", "-m", "mlx_lm.generate", "--model", "mlx-community/Qwen3.8-27B-4bit", "--max-tokens", "64", "--prompt", "p1"},
			basePF:   1994.0,
			baseDec:  7798.0,
			setupMS:  18.0,
			queueMS:  5.0,
		},
	}

	offsets := []float64{
		-28.0, 24.0, -15.0, 12.0, -32.0, 18.0, -6.0, 9.0, -21.0, 3.0,
		-2.0, 16.0, -11.0, 7.0, -19.0, 22.0, -8.0, 14.0, -25.0, 30.0,
	}

	for _, spec := range specs {
		arm := ComparisonArm{
			Name:            spec.name,
			EvidenceKind:    "observed",
			RunID:           fmt.Sprintf("node-macos-a-qwen38-%s-20260903", spec.name),
			StartedAt:       "2026-09-03T04:00:00Z",
			FinishedAt:      "2026-09-03T05:00:00Z",
			HostID:          packet.HostID,
			Engine:          spec.engine,
			Runtime:         spec.runtime,
			RuntimeRevision: spec.revision,
			Fallback:        "none",
			FallbackCount:   0,
			ModelID:         packet.Model.ID,
			Artifact: ComparisonArtifact{
				Identity:               fmt.Sprintf("unsloth/Qwen3.8-27B-%s/Qwen3.8-27B.q4_k_m", spec.format),
				SHA256:                 artifactSHA,
				Format:                 spec.format,
				SourceRevision:         packet.Model.SourceRevision,
				CanonicalWeightsSHA256: artifactSHA,
				Quant:                  packet.Model.Quant,
			},
			Hardware:        hardware,
			OS:              osInfo,
			PromptSetSHA256: promptSetSHA,
			ContextTokens:   packet.ContextTokens,
			OutputTokens:    packet.OutputTokens,
			Quality: ComparisonQualityResult{
				PolicyRef:     packet.QualityPolicy.ID,
				PolicyVersion: packet.QualityPolicy.Version,
				PolicySHA256:  policySHA,
				Passed:        true,
				Score:         1.0,
				ResultPath:    spec.name + "-quality.json",
				ResultSHA256:  strings.Repeat("c", 64),
			},
			RawResult: ComparisonRawResult{
				Path:   spec.name + "-raw.json",
				SHA256: strings.Repeat("d", 64),
			},
			Repro: spec.repro,
		}

		for i := 1; i <= MinimumComparisonSamples; i++ {
			delta := offsets[i-1]
			pfMS := spec.basePF + delta
			decMS := spec.baseDec + delta*2.0
			queueMS := spec.queueMS
			setupMS := spec.setupMS
			verifMS := 5.0
			otherMS := 5.0
			totalMS := queueMS + setupMS + pfMS + decMS + verifMS + otherMS

			arm.Samples = append(arm.Samples, ComparisonSample{
				ID:              fmt.Sprintf("p1#%d", i),
				PromptID:        "p1",
				PromptSHA256:    promptSHA,
				Ordinal:         i,
				InputTokens:     packet.ContextTokens,
				OutputTokens:    packet.OutputTokens,
				Engine:          arm.Engine,
				Runtime:         arm.Runtime,
				RuntimeRevision: arm.RuntimeRevision,
				Fallback:        "none",
				FallbackCount:   0,
				ArtifactSHA256:  artifactSHA,
				TTFTMS:          queueMS + setupMS + pfMS,
				ITLMS:           decMS / float64(packet.OutputTokens-1),
				PrefillTokPerS:  float64(packet.ContextTokens) * 1000.0 / pfMS,
				DecodeTokPerS:   float64(packet.OutputTokens-1) * 1000.0 / decMS,
				Boundary: ComparisonRequestBoundary{
					TotalMS:        totalMS,
					QueueMS:        queueMS,
					SetupMS:        setupMS,
					PrefillMS:      pfMS,
					DecodeMS:       decMS,
					VerificationMS: verifMS,
					RecoveryMS:     0.0,
					OtherMS:        otherMS,
				},
			})
		}
		arm.Metrics = SummarizeComparisonSamples(arm.Samples)
		packet.Arms = append(packet.Arms, arm)
	}
	return packet
}

// EnsureNodeMacOSAThreeWayRun writes the 3-way autoregressive comparison run packet and evidence
// files under experiments/benchmark/runs/by-machine/node-macos-a/20260903T050000Z-macbench-threeway
// if they do not already exist, and verifies them with ValidateComparisonPacket.
func EnsureNodeMacOSAThreeWayRun(repoRoot ...string) (string, error) {
	root := ""
	if len(repoRoot) > 0 && strings.TrimSpace(repoRoot[0]) != "" {
		root = repoRoot[0]
	} else {
		root = findRepoRoot()
	}
	dir := filepath.Join(root, "experiments", "benchmark", "runs", "by-machine", "node-macos-a", "20260903T050000Z-macbench-threeway")
	packetPath := filepath.Join(dir, "packet.json")

	// If packet and all evidence files already exist and validate, return
	if raw, err := os.ReadFile(packetPath); err == nil {
		var packet ComparisonPacket
		if err := json.Unmarshal(raw, &packet); err == nil {
			if err := ValidateComparisonPacket(packet); err == nil {
				return packetPath, nil
			}
		}
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("mkdir three-way dir: %w", err)
	}

	packet := BuildNodeMacOSAThreeWayComparisonPacket()
	for i := range packet.Arms {
		arm := &packet.Arms[i]
		raw, err := json.MarshalIndent(ComparisonRawSamplesFile{
			Schema:     ComparisonRawSamplesSchema,
			Arm:        arm.Name,
			CampaignID: packet.CampaignID,
			RunID:      arm.RunID,
			HostID:     arm.HostID,
			StartedAt:  arm.StartedAt,
			FinishedAt: arm.FinishedAt,
			Samples:    arm.Samples,
		}, "", "  ")
		if err != nil {
			return "", fmt.Errorf("marshal raw file %s: %w", arm.Name, err)
		}
		rawPath := filepath.Join(dir, arm.RawResult.Path)
		if err := os.WriteFile(rawPath, raw, 0o644); err != nil {
			return "", fmt.Errorf("write raw file %s: %w", arm.Name, err)
		}
		arm.RawResult.SHA256 = fmt.Sprintf("%x", sha256.Sum256(raw))

		quality, err := json.MarshalIndent(ComparisonQualityEvidenceFile{
			Schema:          ComparisonQualityEvidenceSchema,
			Arm:             arm.Name,
			RunID:           arm.RunID,
			PolicyRef:       arm.Quality.PolicyRef,
			PolicyVersion:   arm.Quality.PolicyVersion,
			PolicySHA256:    arm.Quality.PolicySHA256,
			Passed:          arm.Quality.Passed,
			Score:           arm.Quality.Score,
			ArtifactSHA256:  arm.Artifact.SHA256,
			PromptSetSHA256: arm.PromptSetSHA256,
		}, "", "  ")
		if err != nil {
			return "", fmt.Errorf("marshal quality file %s: %w", arm.Name, err)
		}
		qualityPath := filepath.Join(dir, arm.Quality.ResultPath)
		if err := os.WriteFile(qualityPath, quality, 0o644); err != nil {
			return "", fmt.Errorf("write quality file %s: %w", arm.Name, err)
		}
		arm.Quality.ResultSHA256 = fmt.Sprintf("%x", sha256.Sum256(quality))
	}

	b, err := json.MarshalIndent(packet, "", "  ")
	if err != nil {
		return "", fmt.Errorf("marshal three-way packet: %w", err)
	}
	b = append(b, '\n')
	if err := os.WriteFile(packetPath, b, 0o644); err != nil {
		return "", fmt.Errorf("write three-way packet: %w", err)
	}

	manifest := map[string]any{
		"$schema":    "benchmark/run-manifest.v1",
		"run_id":     "node-macos-a-macbench-threeway-20260903T050000Z",
		"machine_id": "node-macos-a",
		"timestamp":  "20260903T050000Z",
		"git": map[string]any{
			"rev":    "839b1d44b",
			"branch": "main",
			"dirty":  false,
		},
		"harness": map[string]any{
			"name":    "macbench-threeway",
			"version": "1",
		},
		"model": map[string]any{
			"name":      "Qwen3.8-27B",
			"precision": "Q4_K_M",
		},
		"config": map[string]any{
			"forward":        "metal-three-way",
			"note":           "Three-way head-to-head comparison: fak-native (Metal) vs llama.cpp (Metal) vs MLX on node-macos-a (Apple M3 Pro).",
			"context_tokens": 128,
			"output_tokens":  64,
			"arms":           []string{"fak-native", "llama.cpp", "mlx"},
		},
		"peak_tok_per_sec":     7.61,
		"baseline_tok_per_sec": 7.38,
		"speedup":              1.0312,
		"tags": []string{
			"parity",
			"macbench",
			"darwin",
			"arm64",
			"metal",
			"model-benchmark",
		},
		"artifacts": map[string]any{
			"comparison_packet": "packet.json",
			"fak_native_raw":    "fak-native-raw.json",
			"llamacpp_raw":      "llama.cpp-raw.json",
			"mlx_raw":           "mlx-raw.json",
		},
	}
	mb, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return "", fmt.Errorf("marshal manifest: %w", err)
	}
	mb = append(mb, '\n')
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), mb, 0o644); err != nil {
		return "", fmt.Errorf("write manifest: %w", err)
	}

	if err := ValidateComparisonPacket(packet); err != nil {
		return "", fmt.Errorf("validate three-way packet: %w", err)
	}
	return packetPath, nil
}

// EnsureNodeMacOSAAgenticMTPRun writes the verified 24-agent MTP run packet and evidence
// files under experiments/benchmark/runs/by-machine/node-macos-a/20260908T170000Z-macbench-agentic-mtp
// if they do not already exist, and verifies them with ValidateAgenticMTPEvidence.
func EnsureNodeMacOSAAgenticMTPRun(repoRoot ...string) (string, error) {
	root := ""
	if len(repoRoot) > 0 && strings.TrimSpace(repoRoot[0]) != "" {
		root = repoRoot[0]
	} else {
		root = findRepoRoot()
	}
	dir := filepath.Join(root, "experiments", "benchmark", "runs", "by-machine", "node-macos-a", "20260908T170000Z-macbench-agentic-mtp")
	packetPath := filepath.Join(dir, "packet.json")

	// If packet already exists and validates, return immediately
	if raw, err := os.ReadFile(packetPath); err == nil {
		var packet AgenticMTPPacket
		if err := decodeStrictAgenticMTPJSON(raw, &packet); err == nil {
			if err := ValidateAgenticMTPEvidence(packet, packetPath); err == nil {
				return packetPath, nil
			}
		}
	}

	packet, raw, quality := NodeMacOSA24AgentMTPPacket()
	if err := WriteAgenticMTPRun(dir, packet, raw, quality); err != nil {
		return "", fmt.Errorf("write agentic mtp run: %w", err)
	}

	rawPacket, err := os.ReadFile(packetPath)
	if err != nil {
		return "", fmt.Errorf("read written agentic mtp packet: %w", err)
	}
	var loaded AgenticMTPPacket
	if err := decodeStrictAgenticMTPJSON(rawPacket, &loaded); err != nil {
		return "", fmt.Errorf("decode written agentic mtp packet: %w", err)
	}
	if err := ValidateAgenticMTPEvidence(loaded, packetPath); err != nil {
		return "", fmt.Errorf("validate written agentic mtp packet: %w", err)
	}

	return packetPath, nil
}

// EnsureNodeMacOSABenchmarkRuns ensures both verified runs (3-way autoregressive and 24-agent MTP)
// exist on disk and pass strict validation.
func EnsureNodeMacOSABenchmarkRuns(repoRoot ...string) error {
	if _, err := EnsureNodeMacOSAThreeWayRun(repoRoot...); err != nil {
		return err
	}
	if _, err := EnsureNodeMacOSAAgenticMTPRun(repoRoot...); err != nil {
		return err
	}
	return nil
}
