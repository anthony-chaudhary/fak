package macbench

import (
	"context"
	"strconv"
	"strings"
	"testing"
)

// validPrefillMatchedPacket builds a matched, balanced baseline/candidate pair
// with three repeats per arm and a reported spread. The candidate runs 2x the
// baseline, matching the fak#11582 target shape without claiming a physical run
// (evidence_kind stays SW_VERIFIED).
func validPrefillMatchedPacket() PrefillMatchedPacket {
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
	artifact := ComparisonArtifact{
		Identity:               "Qwen3.8-27B-Q4_K_M.gguf",
		SHA256:                 strings.Repeat("a", 64),
		Format:                 "gguf",
		SourceRevision:         "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
		CanonicalWeightsSHA256: "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169",
		Quant:                  "Q4_K_M",
	}
	prompt := PrefillPrompt{ID: "prefill-4096", Tokens: 4096, SHA256: strings.Repeat("b", 64)}
	settings := PrefillSettings{
		SHA256:         strings.Repeat("c", 64),
		Temperature:    0,
		MaxTokens:      1,
		Engine:         "fak-native",
		Fallback:       "none",
		BatchSize:      1,
		NoFallbackPath: true,
	}

	// Three balanced repeats per arm with a small, reported spread.
	baseline := PrefillMatchedArm{
		Name:           PrefillArmBaseline,
		RunID:          "baseline-20260915",
		StartedAt:      "2026-09-15T12:00:00Z",
		FinishedAt:     "2026-09-15T12:00:05Z",
		HostID:         strings.Repeat("6", 64),
		Engine:         "fak-native",
		Runtime:        "inkernel",
		Artifact:       artifact,
		CacheState:     PrefillCacheCold,
		PromptSHA256:   prompt.SHA256,
		SettingsSHA256: settings.SHA256,
		PeakMemoryMB:   18432.0,
		Samples:        prefillSamples(PrefillArmBaseline, artifact.SHA256, prompt.Tokens, PrefillCacheCold, []float64{1000, 1020, 980}),
		RawResult:      ComparisonRawResult{Path: "baseline-raw.json", SHA256: strings.Repeat("d", 64)},
		Repro:          []string{"go run ./cmd/fak macbench prefill-matched --arm baseline --repeats 3"},
	}
	baseline.Metrics = SummarizePrefillSamples(baseline.Samples)

	candidate := PrefillMatchedArm{
		Name:           PrefillArmCandidate,
		RunID:          "candidate-20260915",
		StartedAt:      "2026-09-15T12:00:06Z",
		FinishedAt:     "2026-09-15T12:00:10Z",
		HostID:         strings.Repeat("6", 64),
		Engine:         "fak-native",
		Runtime:        "inkernel",
		Artifact:       artifact,
		CacheState:     PrefillCacheCold,
		PromptSHA256:   prompt.SHA256,
		SettingsSHA256: settings.SHA256,
		PeakMemoryMB:   19456.0,
		Samples:        prefillSamples(PrefillArmCandidate, artifact.SHA256, prompt.Tokens, PrefillCacheCold, []float64{500, 510, 490}),
		RawResult:      ComparisonRawResult{Path: "candidate-raw.json", SHA256: strings.Repeat("e", 64)},
		Repro:          []string{"go run ./cmd/fak macbench prefill-matched --arm candidate --repeats 3"},
	}
	candidate.Metrics = SummarizePrefillSamples(candidate.Samples)

	packet := PrefillMatchedPacket{
		Schema:         PrefillMatchedSchema,
		GeneratedAt:    "2026-09-15T12:00:11Z",
		CampaignID:     "issue-13089-prefill-matched-20260915",
		HostID:         strings.Repeat("6", 64),
		EvidenceKind:   PrefillEvidenceSWVerified,
		BaselineCommit: "d088a6b37",
		Model: ComparisonModel{
			Family:                 "Qwen3.8",
			ID:                     "Qwen3.8-27B",
			SourceRevision:         "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
			CanonicalWeightsSHA256: artifact.CanonicalWeightsSHA256,
			Quant:                  "Q4_K_M",
		},
		Hardware: hardware,
		OS:       osInfo,
		Prompt:   prompt,
		Settings: settings,
		Arms:     []PrefillMatchedArm{baseline, candidate},
		Summary: PrefillMatchedSummary{
			BaselineMeanTokPerS:  baseline.Metrics.MeanTokPerS,
			CandidateMeanTokPerS: candidate.Metrics.MeanTokPerS,
			Ratio:                candidate.Metrics.MeanTokPerS / baseline.Metrics.MeanTokPerS,
			RatioCV:              candidate.Metrics.CoefficientOfVar,
			Verified:             true,
		},
	}
	return packet
}

// prefillSamples builds reconciled samples: prefill_tok_s is derived from
// tokens/prefill_ms so the validator's cross-check holds by construction.
func prefillSamples(arm, artifactSHA string, tokens int, cache string, prefillMS []float64) []PrefillSample {
	samples := make([]PrefillSample, 0, len(prefillMS))
	for i, ms := range prefillMS {
		samples = append(samples, PrefillSample{
			ID:             sampleID(arm, i+1),
			Ordinal:        i + 1,
			InputTokens:    tokens,
			PrefillMS:      ms,
			PrefillTokPerS: float64(tokens) * 1000 / ms,
			CacheState:     cache,
			ArtifactSHA256: artifactSHA,
		})
	}
	return samples
}

func sampleID(arm string, ordinal int) string {
	return arm + "#" + strconv.Itoa(ordinal)
}

// TestPrefillMatchedWitness is the fak#13089 witness: a matched
// baseline/candidate receipt validates with equal-N repeats, a recomputable
// ratio, and a reported spread; a one-sided, missing-data, or unbalanced run
// fails closed.
func TestPrefillMatchedWitness(t *testing.T) {
	packet := validPrefillMatchedPacket()
	if err := ValidatePrefillMatchedPacket(packet); err != nil {
		t.Fatalf("valid matched packet rejected: %v", err)
	}
	if packet.Summary.Ratio <= 0 {
		t.Fatalf("summary ratio must be computed, got %v", packet.Summary.Ratio)
	}

	// Balanced: both arms carry the same repeat count, and that count meets the floor.
	baselineN := len(packet.Arms[0].Samples)
	candidateN := len(packet.Arms[1].Samples)
	if baselineN != candidateN || baselineN < MinPrefillMatchedRepeats {
		t.Fatalf("arms must be balanced with >= %d repeats, got %d vs %d", MinPrefillMatchedRepeats, baselineN, candidateN)
	}

	// Variance is reported, not implied: a tight spread yields a finite, non-negative CV.
	for _, arm := range packet.Arms {
		if arm.Metrics.N != len(arm.Samples) {
			t.Fatalf("%s: metrics.n %d must equal sample count %d", arm.Name, arm.Metrics.N, len(arm.Samples))
		}
		if arm.Metrics.StdDevTokPerS <= 0 || arm.Metrics.CoefficientOfVar <= 0 {
			t.Fatalf("%s: spread must be reported (stddev=%v cv=%v)", arm.Name, arm.Metrics.StdDevTokPerS, arm.Metrics.CoefficientOfVar)
		}
	}

	// The published ratio must reconcile with the arm means.
	expected := packet.Arms[1].Metrics.MeanTokPerS / packet.Arms[0].Metrics.MeanTokPerS
	if diff := packet.Summary.Ratio - expected; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("summary ratio %v does not reconcile with arm means %v", packet.Summary.Ratio, expected)
	}
}

// TestPrefillMatchedWitnessFailsClosed proves the gate rejects the ways a
// receipt can be fabricated: a one-sided run, a missing arm, unbalanced
// repeats, an unreconciled ratio, and a claimed-but-unbacked physical run.
func TestPrefillMatchedWitnessFailsClosed(t *testing.T) {
	t.Run("one-sided run drops the candidate", func(t *testing.T) {
		packet := validPrefillMatchedPacket()
		packet.Arms = packet.Arms[:1]
		assertPrefillMatchesFail(t, packet, "exactly two arms")
	})

	t.Run("missing baseline drops the baseline", func(t *testing.T) {
		packet := validPrefillMatchedPacket()
		candidate := packet.Arms[1]
		packet.Arms = []PrefillMatchedArm{candidate, candidate}
		assertPrefillMatchesFail(t, packet, "baseline arm is required")
	})

	t.Run("unbalanced repeats", func(t *testing.T) {
		packet := validPrefillMatchedPacket()
		packet.Arms[1].Samples = packet.Arms[1].Samples[:2]
		packet.Arms[1].Metrics = SummarizePrefillSamples(packet.Arms[1].Samples)
		assertPrefillMatchesFail(t, packet, "must be balanced")
	})

	t.Run("ratio not reconciled with arms", func(t *testing.T) {
		packet := validPrefillMatchedPacket()
		packet.Summary.Ratio = 9.99
		assertPrefillMatchesFail(t, packet, "summary.ratio")
	})

	t.Run("single repeat has no spread", func(t *testing.T) {
		packet := validPrefillMatchedPacket()
		packet.Arms[0].Samples = packet.Arms[0].Samples[:1]
		packet.Arms[0].Metrics = SummarizePrefillSamples(packet.Arms[0].Samples)
		packet.Arms[1].Samples = packet.Arms[1].Samples[:1]
		packet.Arms[1].Metrics = SummarizePrefillSamples(packet.Arms[1].Samples)
		assertPrefillMatchesFail(t, packet, "balanced repeats")
	})

	t.Run("fallback path left enabled", func(t *testing.T) {
		packet := validPrefillMatchedPacket()
		packet.Settings.Fallback = "host"
		packet.Settings.NoFallbackPath = false
		assertPrefillMatchesFail(t, packet, "settings.fallback")
	})

	t.Run("artifact mismatch between arms and model", func(t *testing.T) {
		packet := validPrefillMatchedPacket()
		packet.Arms[1].Artifact.Quant = "Q8_0"
		assertPrefillMatchesFail(t, packet, "artifact.quant")
	})
}

func assertPrefillMatchesFail(t *testing.T, packet PrefillMatchedPacket, wantSubstring string) {
	t.Helper()
	err := ValidatePrefillMatchedPacket(packet)
	if err == nil {
		t.Fatalf("expected fail-closed rejection containing %q, got nil", wantSubstring)
	}
	if !strings.Contains(err.Error(), wantSubstring) {
		t.Fatalf("expected rejection containing %q, got: %v", wantSubstring, err)
	}
}

// TestRunPrefillMatchedHarness drives the harness with an injected runner: it
// proves the assembled receipt is balanced, validates, and reports spread
// without any Metal device (so it is SW_VERIFIED, not HW_WITNESSED).
func TestRunPrefillMatchedHarness(t *testing.T) {
	base := validPrefillMatchedPacket()
	rates := map[string][]float64{
		PrefillArmBaseline:  {1000, 1020, 980},
		PrefillArmCandidate: {500, 510, 490},
	}
	runner := func(_ context.Context, arm string, repeats int) (PrefillMatchedArm, error) {
		tmpl := base.Arms[0]
		if arm == PrefillArmCandidate {
			tmpl = base.Arms[1]
		}
		ms := make([]float64, 0, repeats)
		for _, rate := range rates[arm] {
			ms = append(ms, float64(base.Prompt.Tokens)*1000/rate)
		}
		tmpl.Samples = prefillSamples(arm, base.Arms[0].Artifact.SHA256, base.Prompt.Tokens, PrefillCacheCold, ms)
		tmpl.Metrics = SummarizePrefillSamples(tmpl.Samples)
		return tmpl, nil
	}

	packet, err := RunPrefillMatched(context.Background(), PrefillMatchedRequest{
		CampaignID:     base.CampaignID,
		HostID:         base.HostID,
		EvidenceKind:   PrefillEvidenceSWVerified,
		BaselineCommit: base.BaselineCommit,
		Model:          base.Model,
		Hardware:       base.Hardware,
		OS:             base.OS,
		Prompt:         base.Prompt,
		Settings:       base.Settings,
		Repeats:        3,
	}, runner)
	if err != nil {
		t.Fatalf("RunPrefillMatched rejected a balanced run: %v", err)
	}
	if len(packet.Arms) != 2 {
		t.Fatalf("expected 2 arms, got %d", len(packet.Arms))
	}
	if len(packet.Arms[0].Samples) != len(packet.Arms[1].Samples) {
		t.Fatalf("harness must produce balanced arms")
	}
	if packet.Summary.Ratio <= 0 || packet.Summary.RatioCV <= 0 {
		t.Fatalf("harness must publish a ratio and a spread, got ratio=%v cv=%v", packet.Summary.Ratio, packet.Summary.RatioCV)
	}
}
