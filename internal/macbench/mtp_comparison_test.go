package macbench

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"
)

// TestValidateMTPComparisonPacketNodeMacOSA verifies that the synthetic builder remains a
// fixture while any on-disk observed packet satisfies strict physical qualification.
func TestValidateMTPComparisonPacketNodeMacOSA(t *testing.T) {
	packet := NodeMacOSAMTPComparisonPacket()
	if err := ValidateMTPComparisonPacket(packet); err == nil {
		t.Fatal("fixture packet unexpectedly passed observed physical validation")
	}
	if packet.Summary.Verified {
		t.Fatal("fixture packet must not claim verified physical evidence")
	}
	for _, arm := range packet.Arms {
		if arm.EvidenceKind != "fixture" {
			t.Errorf("fixture arm %q has evidence kind %q", arm.Name, arm.EvidenceKind)
		}
	}
}

type testMTPContextKey struct{}

func makeValidTestMTPOptions(t *testing.T) MTPRunnerOptions {
	t.Helper()
	return MTPRunnerOptions{
		CampaignID: "campaign-test-12239",
		HostID:     strings.Repeat("6", 64),
		Model: ComparisonModel{
			Family:                 "test-family",
			ID:                     "test-model",
			SourceRevision:         "local-test-rev-12239",
			CanonicalWeightsSHA256: strings.Repeat("e", 64),
			Quant:                  "test-quant",
		},
		Hardware: ComparisonHardware{
			Model:       "test-machine",
			Chip:        "test-chip",
			MemoryBytes: 1024,
		},
		OS: ComparisonOS{
			Name:    "test-os",
			Version: "test-version",
			Build:   "test-build",
		},
		PromptSet: ComparisonPromptSet{
			ID:     "mtp-agentic-prompts-v1",
			SHA256: strings.Repeat("a", 64),
			Prompts: []ComparisonPrompt{
				{ID: "p1", SHA256: strings.Repeat("b", 64)},
			},
		},
		ContextTokens: 128,
		OutputTokens:  64,
		SpeculativeConfig: MTPSpeculativeConfig{
			DraftDepth:             2,
			TargetTokens:           64,
			Temperature:            0.0,
			MinAcceptanceRate:      0.75,
			MinEffectiveDecodeTokS: 14.5,
		},
		QualityPolicy: ComparisonQualityPolicy{
			ID:           "strict-token-parity",
			Version:      "1",
			SHA256:       strings.Repeat("8", 64),
			MinimumScore: 1.0,
		},
		ArmTimeout: 5 * time.Second,
		HTTPClient: &http.Client{Timeout: 3 * time.Second},
		Now: func() time.Time {
			genTime, _ := time.Parse(time.RFC3339, "2026-09-08T16:30:00Z")
			return genTime
		},
		evidenceKind: mtpEvidenceTest,
	}
}

func makeTestMeasuredArm(name string, decodeTokS, acceptanceRate float64, revision string, opts MTPRunnerOptions) MTPComparisonArm {
	engineMap := map[string][3]string{
		"fak-native": {"fak-native", "inkernel", "gguf"},
		"ax-engine":  {"ax-engine", "apple-silicon", "ax"},
		"mtplx":      {"mtplx", "mlx", "safetensors"},
		"llama.cpp":  {"llama.cpp", "reference", "gguf"},
	}
	cfg := engineMap[name]

	arm := MTPComparisonArm{
		Name:            name,
		EvidenceKind:    mtpEvidenceTest,
		RunID:           fmt.Sprintf("local-test-run-%s-12239", name),
		StartedAt:       "2026-09-08T15:00:00Z",
		FinishedAt:      "2026-09-08T16:00:00Z",
		HostID:          opts.HostID,
		Engine:          cfg[0],
		Runtime:         cfg[1],
		RuntimeRevision: revision,
		SpecType:        "mtp-sidecar",
		Fallback:        "none",
		FallbackCount:   0,
		ModelID:         opts.Model.ID,
		Artifact: ComparisonArtifact{
			Identity:               fmt.Sprintf("test-artifact-%s-%s", name, cfg[2]),
			SHA256:                 opts.Model.CanonicalWeightsSHA256,
			Format:                 cfg[2],
			SourceRevision:         opts.Model.SourceRevision,
			CanonicalWeightsSHA256: opts.Model.CanonicalWeightsSHA256,
			Quant:                  opts.Model.Quant,
		},
		Hardware:        opts.Hardware,
		OS:              opts.OS,
		PromptSetSHA256: opts.PromptSet.SHA256,
		ContextTokens:   opts.ContextTokens,
		OutputTokens:    opts.OutputTokens,
		Quality: ComparisonQualityResult{
			PolicyRef:     opts.QualityPolicy.ID,
			PolicyVersion: opts.QualityPolicy.Version,
			PolicySHA256:  opts.QualityPolicy.SHA256,
			Passed:        true,
			Score:         1.0,
			ResultPath:    fmt.Sprintf("%s-quality.json", name),
			ResultSHA256:  strings.Repeat("7", 64),
		},
		DraftDepth:          opts.SpeculativeConfig.DraftDepth,
		AcceptanceRate:      acceptanceRate,
		RollbackCount:       10,
		EffectiveDecodeTokS: decodeTokS,
		RawResult:           ComparisonRawResult{Path: name + "-raw.json", SHA256: strings.Repeat("9", 64)},
		Repro:               []string{"test-runner", "--arm", name},
	}

	tokensToDecode := float64(opts.OutputTokens - 1)
	prefillMS := 2000.0
	decodeMS := tokensToDecode * 1000.0 / decodeTokS

	samples := make([]MTPComparisonSample, MinimumMTPComparisonSamples)
	for i := 1; i <= MinimumMTPComparisonSamples; i++ {
		samples[i-1] = MTPComparisonSample{
			ID:              fmt.Sprintf("p1#%d", i),
			PromptID:        "p1",
			PromptSHA256:    opts.PromptSet.Prompts[0].SHA256,
			Ordinal:         i,
			InputTokens:     opts.ContextTokens,
			OutputTokens:    opts.OutputTokens,
			Engine:          cfg[0],
			Runtime:         cfg[1],
			RuntimeRevision: revision,
			DraftDepth:      opts.SpeculativeConfig.DraftDepth,
			DraftProposed:   64,
			DraftAccepted:   int(math.Round(64.0 * acceptanceRate)),
			AcceptanceRate:  acceptanceRate,
			RollbackCount:   10,
			Fallback:        "none",
			ArtifactSHA256:  opts.Model.CanonicalWeightsSHA256,
			TTFTMS:          15.0 + prefillMS,
			ITLMS:           decodeMS / tokensToDecode,
			PrefillTokPerS:  float64(opts.ContextTokens) * 1000.0 / prefillMS,
			DecodeTokPerS:   decodeTokS,
			Boundary: ComparisonRequestBoundary{
				TotalMS: 25.0 + prefillMS + decodeMS, QueueMS: 5.0, SetupMS: 10.0, PrefillMS: prefillMS,
				DecodeMS: decodeMS, VerificationMS: 5.0, OtherMS: 5.0,
			},
		}
	}
	arm.Samples = samples
	arm.Metrics = SummarizeMTPSamples(samples)
	arm.EffectiveDecodeTokS = arm.Metrics.Decode.ThroughputTokS.P50
	arm.AcceptanceRate = arm.Metrics.Decode.AcceptanceRate.P50
	return arm
}

var testMTPArmSpecs = map[string]struct {
	decodeRate float64
	acceptance float64
	revision   string
}{
	"fak-native": {decodeRate: 16.0, acceptance: 0.8125, revision: "test-rev-fak-101"},
	"ax-engine":  {decodeRate: 15.0, acceptance: 0.78125, revision: "test-rev-ax-202"},
	"mtplx":      {decodeRate: 14.0, acceptance: 0.75, revision: "test-rev-mtplx-303"},
	"llama.cpp":  {decodeRate: 10.0, acceptance: 0.00, revision: "test-rev-llama-404"},
}

func makeTestMTPArms(opts MTPRunnerOptions) map[string]MTPComparisonArm {
	arms := make(map[string]MTPComparisonArm, len(testMTPArmSpecs))
	for name, spec := range testMTPArmSpecs {
		arm := makeTestMeasuredArm(name, spec.decodeRate, spec.acceptance, spec.revision, opts)
		arm.EffectiveDecodeTokS = 999 // Run must replace adapter-level claims with sample metrics.
		arm.AcceptanceRate = 0.99
		arms[name] = arm
	}
	return arms
}

func makeTestMTPAdapters(arms map[string]MTPComparisonArm) map[string]MTPComparisonAdapter {
	adapters := make(map[string]MTPComparisonAdapter, len(canonicalMTPArms))
	for _, name := range canonicalMTPArms {
		arm := arms[name]
		adapters[name] = func(context.Context, MTPArmRequest) (MTPComparisonArm, error) {
			return arm, nil
		}
	}
	return adapters
}

func expectedMTPArmRequest(opts MTPRunnerOptions, name string) MTPArmRequest {
	return MTPArmRequest{
		ArmName: name, CampaignID: opts.CampaignID, HostID: opts.HostID,
		Model: opts.Model, Hardware: opts.Hardware, OS: opts.OS,
		PromptSet: cloneMTPPromptSet(opts.PromptSet), ContextTokens: opts.ContextTokens, OutputTokens: opts.OutputTokens,
		SpeculativeConfig: opts.SpeculativeConfig, QualityPolicy: opts.QualityPolicy,
		ArmTimeout: opts.ArmTimeout, HTTPClient: opts.HTTPClient,
	}
}

// TestMTPRunnerExecutesMeasuredComparison verifies that MTPRunner executes every canonical
// arm exactly once through injected adapters, propagates context and MTPArmRequest, derives
// the summary from test-provenance adapter arms, and fails closed on envelope errors, timeouts,
// adapter failures, or non-observed provenance.
func TestMTPRunnerExecutesMeasuredComparison(t *testing.T) {
	t.Run("ExecutesCanonicalAdaptersAndDerivesSummary", func(t *testing.T) {
		opts := makeValidTestMTPOptions(t)
		testArms := makeTestMTPArms(opts)
		callerPrompt := &opts.PromptSet.Prompts[0]

		type armCallRecord struct {
			count        int
			contextValue any
			hadDeadline  bool
			promptCloned bool
			req          MTPArmRequest
		}
		calls := make(map[string]*armCallRecord)
		for _, name := range canonicalMTPArms {
			calls[name] = &armCallRecord{}
		}

		opts.Adapters = make(map[string]MTPComparisonAdapter)
		for _, name := range canonicalMTPArms {
			armName := name
			opts.Adapters[armName] = func(ctx context.Context, req MTPArmRequest) (MTPComparisonArm, error) {
				rec := calls[armName]
				rec.count++
				rec.contextValue = ctx.Value(testMTPContextKey{})
				_, rec.hadDeadline = ctx.Deadline()
				rec.promptCloned = &req.PromptSet.Prompts[0] != callerPrompt
				rec.req = req
				rec.req.PromptSet = cloneMTPPromptSet(req.PromptSet)
				req.PromptSet.Prompts[0].ID = "adapter-local-mutation"
				return testArms[armName], nil
			}
		}

		runner := NewMTPRunner(opts)
		ctx := context.WithValue(context.Background(), testMTPContextKey{}, "test-marker-12239")

		packet, err := runner.Run(ctx)
		if err != nil {
			t.Fatalf("runner.Run failed: %v", err)
		}

		for _, name := range canonicalMTPArms {
			rec := calls[name]
			if rec.count != 1 {
				t.Errorf("expected canonical arm %q to execute exactly once, got %d", name, rec.count)
			}
			if rec.contextValue != "test-marker-12239" {
				t.Errorf("expected supplied context to reach adapter for arm %q", name)
			}
			if !rec.hadDeadline {
				t.Errorf("expected bounded context for arm %q", name)
			}
			if !rec.promptCloned {
				t.Errorf("expected prompt slice clone for arm %q", name)
			}
			if want := expectedMTPArmRequest(opts, name); !reflect.DeepEqual(rec.req, want) {
				t.Errorf("arm request mismatch for %q:\n got: %#v\nwant: %#v", name, rec.req, want)
			}
			if rec.req.HTTPClient != opts.HTTPClient {
				t.Errorf("expected HTTP client identity to reach arm %q", name)
			}
		}

		if packet.CampaignID != "campaign-test-12239" {
			t.Errorf("expected packet.CampaignID %q, got %q", "campaign-test-12239", packet.CampaignID)
		}
		if len(packet.Arms) != 4 {
			t.Fatalf("expected 4 arms in packet, got %d", len(packet.Arms))
		}
		for _, arm := range packet.Arms {
			expectedArm := testArms[arm.Name]
			if arm.EvidenceKind != mtpEvidenceTest {
				t.Errorf("expected arm %q to have test evidence_kind, got %q", arm.Name, arm.EvidenceKind)
			}
			if arm.RuntimeRevision != expectedArm.RuntimeRevision {
				t.Errorf("expected arm %q revision %q, got %q", arm.Name, expectedArm.RuntimeRevision, arm.RuntimeRevision)
			}
			if arm.RunID != expectedArm.RunID {
				t.Errorf("expected arm %q RunID %q, got %q", arm.Name, expectedArm.RunID, arm.RunID)
			}
			spec := testMTPArmSpecs[arm.Name]
			if math.Abs(arm.EffectiveDecodeTokS-spec.decodeRate) > 0.01 {
				t.Errorf("expected arm %q sample-derived decode %.2f, got %.2f", arm.Name, spec.decodeRate, arm.EffectiveDecodeTokS)
			}
			if math.Abs(arm.AcceptanceRate-spec.acceptance) > 0.001 {
				t.Errorf("expected arm %q sample-derived acceptance %.2f, got %.2f", arm.Name, spec.acceptance, arm.AcceptanceRate)
			}
		}
		if packet.PromptSet.Prompts[0].ID != "p1" {
			t.Errorf("adapter mutated packet prompt set: %q", packet.PromptSet.Prompts[0].ID)
		}

		// Summary speedup ratios: 16.0/10.0=1.60, 16.0/15.0=1.07, 16.0/14.0=1.14
		if math.Abs(packet.Summary.VsLlamaSpeedupRatio-1.60) > 0.01 {
			t.Errorf("expected vs_llama ratio 1.60, got %.2f", packet.Summary.VsLlamaSpeedupRatio)
		}
		if math.Abs(packet.Summary.VsAxEngineRatio-1.07) > 0.01 {
			t.Errorf("expected vs_ax ratio 1.07, got %.2f", packet.Summary.VsAxEngineRatio)
		}
		if math.Abs(packet.Summary.VsMTPLXRatio-1.14) > 0.01 {
			t.Errorf("expected vs_mtplx ratio 1.14, got %.2f", packet.Summary.VsMTPLXRatio)
		}
		if packet.Summary.FakNativeDecodeTokS != 16.0 {
			t.Errorf("expected fak-native decode 16.0, got %.2f", packet.Summary.FakNativeDecodeTokS)
		}
		if packet.Summary.FakNativeAcceptanceRate != testMTPArmSpecs["fak-native"].acceptance {
			t.Errorf("unexpected fak-native acceptance %.5f", packet.Summary.FakNativeAcceptanceRate)
		}
		if packet.Summary.Verified {
			t.Errorf("test provenance must not claim verified physical evidence")
		}
		if err := ValidateMTPComparisonPacket(packet); err == nil {
			t.Error("neutral test packet unexpectedly passed observed physical validation")
		}
	})

	t.Run("EnvelopeValidationFailsClosed", func(t *testing.T) {
		opts := makeValidTestMTPOptions(t)
		opts.Adapters = nil
		runner := NewMTPRunner(opts)
		if _, err := runner.Run(context.Background()); err == nil {
			t.Fatal("expected runner.Run without adapters to fail closed")
		}

		// Invalid host ID fails before any adapter executes
		executed := false
		opts.HostID = "invalid-non-sha256"
		opts.Adapters = map[string]MTPComparisonAdapter{
			"fak-native": func(ctx context.Context, req MTPArmRequest) (MTPComparisonArm, error) {
				executed = true
				return MTPComparisonArm{}, nil
			},
		}
		runnerHost := NewMTPRunner(opts)
		if _, err := runnerHost.Run(context.Background()); err == nil {
			t.Fatal("expected invalid host_id to fail envelope validation")
		}
		if executed {
			t.Fatal("expected no adapter to execute on invalid envelope")
		}
	})

	t.Run("AdapterArmMustMatchRequestEnvelope", func(t *testing.T) {
		opts := makeValidTestMTPOptions(t)
		arms := makeTestMTPArms(opts)
		arm := arms["fak-native"]
		arm.HostID = strings.Repeat("d", 64)
		arm.Artifact.CanonicalWeightsSHA256 = strings.Repeat("c", 64)
		arms["fak-native"] = arm
		opts.Adapters = makeTestMTPAdapters(arms)
		_, err := NewMTPRunner(opts).Run(context.Background())
		if err == nil || !strings.Contains(err.Error(), "request envelope") {
			t.Fatalf("expected mismatched adapter envelope rejection, got %v", err)
		}
	})

	t.Run("ArmTimeoutFailsClosed", func(t *testing.T) {
		opts := makeValidTestMTPOptions(t)
		opts.ArmTimeout = 25 * time.Millisecond
		started := make(chan struct{})
		exited := make(chan struct{})
		unexpected := make(chan string, len(canonicalMTPArms)-1)
		opts.Adapters = makeTestMTPAdapters(makeTestMTPArms(opts))
		opts.Adapters["fak-native"] = func(ctx context.Context, _ MTPArmRequest) (MTPComparisonArm, error) {
			close(started)
			defer close(exited)
			<-ctx.Done()
			return MTPComparisonArm{}, ctx.Err()
		}
		for _, name := range canonicalMTPArms[1:] {
			armName := name
			opts.Adapters[armName] = func(context.Context, MTPArmRequest) (MTPComparisonArm, error) {
				unexpected <- armName
				return MTPComparisonArm{}, nil
			}
		}

		runner := NewMTPRunner(opts)
		_, err := runner.Run(context.Background())
		if err == nil {
			t.Fatal("expected runner.Run to fail closed on timeout")
		}
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("expected errors.Is(err, context.DeadlineExceeded), got: %v", err)
		}

		select {
		case <-started:
		default:
			t.Error("expected fak-native arm to have started")
		}
		select {
		case <-exited:
		case <-time.After(time.Second):
			t.Fatal("timed-out adapter did not exit after context cancellation")
		}
		select {
		case name := <-unexpected:
			t.Errorf("subsequent arm %q executed after timeout", name)
		default:
		}
	})

	t.Run("AdapterErrorFailsClosed", func(t *testing.T) {
		opts := makeValidTestMTPOptions(t)
		opts.Adapters = makeTestMTPAdapters(makeTestMTPArms(opts))
		opts.Adapters["ax-engine"] = func(context.Context, MTPArmRequest) (MTPComparisonArm, error) {
			return MTPComparisonArm{}, errors.New("simulated adapter failure")
		}
		runner := NewMTPRunner(opts)
		_, err := runner.Run(context.Background())
		if err == nil {
			t.Fatal("expected runner.Run to fail closed on adapter error")
		}
		if !strings.Contains(err.Error(), "simulated adapter failure") {
			t.Errorf("expected error to mention adapter failure, got: %v", err)
		}
	})

	t.Run("NonObservedProvenanceFailsClosed", func(t *testing.T) {
		opts := makeValidTestMTPOptions(t)
		testArms := makeTestMTPArms(opts)
		arm := testArms["mtplx"]
		arm.EvidenceKind = "fixture"
		testArms["mtplx"] = arm
		opts.Adapters = makeTestMTPAdapters(testArms)
		runner := NewMTPRunner(opts)
		_, err := runner.Run(context.Background())
		if err == nil {
			t.Fatal("expected runner.Run to fail closed on fixture/non-observed provenance")
		}
	})

	t.Run("DefaultObservedModeRejectsNodeFixture", func(t *testing.T) {
		fixture := NodeMacOSAMTPComparisonPacket()
		opts := MTPRunnerOptions{
			CampaignID: fixture.CampaignID, HostID: fixture.HostID, Model: fixture.Model,
			Hardware: fixture.Hardware, OS: fixture.OS, PromptSet: fixture.PromptSet,
			ContextTokens: fixture.ContextTokens, OutputTokens: fixture.OutputTokens,
			SpeculativeConfig: fixture.SpeculativeConfig, QualityPolicy: fixture.QualityPolicy,
			ArmTimeout: time.Second,
			Now:        func() time.Time { return time.Date(2026, 9, 8, 16, 30, 0, 0, time.UTC) },
		}
		arms := make(map[string]MTPComparisonArm, len(fixture.Arms))
		for _, arm := range fixture.Arms {
			arms[arm.Name] = arm
		}
		opts.Adapters = makeTestMTPAdapters(arms)
		if _, err := NewMTPRunner(opts).Run(context.Background()); err == nil {
			t.Fatal("default observed mode accepted fixture provenance")
		}
	})
}
