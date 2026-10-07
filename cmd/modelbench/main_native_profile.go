package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/macbench"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

func appendNativeProfilePhase(phases []nativeperf.ProfilePhase, name string, duration time.Duration) []nativeperf.ProfilePhase {
	startMilliseconds := 0.0
	if len(phases) > 0 {
		previous := phases[len(phases)-1]
		startMilliseconds = previous.StartMilliseconds + previous.DurationMilliseconds
	}
	return append(phases, nativeperf.ProfilePhase{
		Name:                 name,
		StartMilliseconds:    startMilliseconds,
		DurationMilliseconds: float64(duration.Nanoseconds()) / 1e6,
	})
}

func runNativePerformanceProfile(f *benchFlags, m *model.Model, loadNanos int64, vocab int, controls map[string]string, newSession func() *model.Session) error {
	loadDuration := time.Duration(loadNanos)
	phases := appendNativeProfilePhase(nil, "load-setup", loadDuration)
	var cachePhaseLatency modelperfobs.CachePhaseLatencyRecorder

	s := newSession()
	finishProfile := onceFinishNativeProfile(s.Close)
	defer finishProfile()
	sequenceSelector := controls[nativeProfileSequenceSelector]
	profiler, err := configureNativeProfileSession(s, controls)
	if err != nil {
		return err
	}
	logits, sequenceExecuted, phases, err := runNativeProfileForward(s, vocab, sequenceSelector, phases)
	if err != nil {
		return err
	}

	t := time.Now()
	for _, value := range logits {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("native performance verification: non-finite logits")
		}
	}
	executionReceipt, err := profiler.MetalExecutionReceipt()
	if err != nil {
		return fmt.Errorf("native performance profile unavailable: %w", err)
	}
	if err := metalgemm.ValidateExecutionReceipt(executionReceipt); err != nil {
		return fmt.Errorf("native performance profile unavailable: %w", err)
	}
	counters := executionReceipt.Counters
	fallbackReceipt, err := profiler.MetalFallbackReceipt()
	if err != nil {
		return fmt.Errorf("native performance fallback receipt unavailable: %w", err)
	}
	fallbackCount := fallbackReceipt.PromisedCPUFallbacks
	if err := requireNoMetalFallbacks(fallbackCount); err != nil {
		return err
	}
	resident := m.ResidentReport().TotalResidentBytes
	workingSet, err := peakRSSBytes()
	if err != nil || resident <= 0 || workingSet == 0 {
		return fmt.Errorf("native performance profile unavailable: memory capture resident=%d working_set=%d: %w", resident, workingSet, err)
	}
	d := time.Since(t)
	phases = appendNativeProfilePhase(phases, "verification", d)
	handoffReceipt := s.Qwen35DecodeHandoffReceipt()
	if err := model.ValidateQwen35DecodeHandoffReceipt(handoffReceipt); err != nil {
		return fmt.Errorf("native performance decode handoff receipt: %w", err)
	}

	t = time.Now()
	finishProfile()
	d = time.Since(t)
	phases = appendNativeProfilePhase(phases, "teardown", d)

	identity, err := captureNativeProfileIdentity(f, m, sequenceSelector, sequenceExecuted)
	if err != nil {
		return err
	}

	profile := nativeperf.ProfileBundle{
		Schema:                 nativeperf.ProfileSchema,
		EnvelopeID:             identity.envelope.ID,
		Execution:              nativeperf.ExecutionIdentity{Engine: identity.envelope.Engine, ForwardPath: identity.forwardPath, FallbackCount: fallbackCount},
		Phases:                 phases,
		Metal:                  &nativeperf.MetalCounters{CommandBuffers: counters.CommandBuffers, Encoders: counters.Encoders, DispatchMilliseconds: counters.DispatchMilliseconds, WaitMilliseconds: counters.WaitMilliseconds, ResidentBytes: uint64(resident), WorkingSetBytes: workingSet},
		AttributionUnavailable: &nativeperf.AttributionUnavailable{Reason: nativeperf.AttributionUnavailableCapture, Detail: "fak-native session lifecycle capture does not export per-lever dispatch attribution"},
	}
	if err := validateNativeProfileForControls(profile, controls); err != nil {
		return err
	}
	b, err := json.MarshalIndent(profile, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	profileSum := sha256.Sum256(b)
	q4kResidency := m.Q4KResidencyReceipt()
	cacheLatencyReceipt := cachePhaseLatency.Receipt()
	receipt := nativeProfileReceipt{
		Schema:              nativeProfileReceiptSchema,
		ProfileSHA256:       fmt.Sprintf("%x", profileSum),
		EnvelopeID:          identity.envelope.ID,
		Artifact:            nativeArtifactIdentity{nativeFileIdentity: identity.artifactFile, Model: identity.envelope.Model, ModelRevision: identity.envelope.ModelRevision},
		ModelConfig:         identity.loadedConfig,
		ModelConfigSHA256:   identity.loadedConfigSHA,
		Host:                identity.host,
		Source:              identity.source,
		Binary:              identity.binary,
		Controls:            controls,
		Execution:           executionReceipt,
		Fallbacks:           fallbackReceipt,
		Q4KResidency:        &q4kResidency,
		Qwen35DecodeHandoff: &handoffReceipt,
		CachePhaseLatency:   &cacheLatencyReceipt,
	}
	receipt.BindingSHA256, err = nativeReceiptBinding(receipt)
	if err != nil {
		return err
	}
	if err := validateNativeProfileReceipt(b, profile, receipt); err != nil {
		return fmt.Errorf("native performance receipt self-check: %w", err)
	}
	receiptBytes, err := benchcli.MarshalReport(receipt)
	if err != nil {
		return err
	}
	receiptBytes = append(receiptBytes, '\n')
	if err := os.WriteFile(*f.nativeProfileOut, b, 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(nativeReceiptPath(*f.nativeProfileOut), receiptBytes, 0o644); err != nil {
		return err
	}
	fmt.Fprintln(os.Stderr, "wrote", *f.nativeProfileOut)
	fmt.Fprintln(os.Stderr, "wrote", nativeReceiptPath(*f.nativeProfileOut))
	return nil
}

// configureNativeProfileSession applies the profile's session controls (candidate
// sequence route, decode handoff) and attaches the phase profiler.
func configureNativeProfileSession(s *model.Session, controls map[string]string) (*model.PhaseProfiler, error) {
	if controls[nativeProfileSequenceSelector] == nativeProfileSelectorOn {
		if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
			return nil, fmt.Errorf("native performance candidate route unavailable: %w", err)
		}
	}
	var handoffMode model.Qwen35DecodeHandoffMode
	if err := handoffMode.Set(controls[nativeProfileDecodeHandoffControl]); err != nil {
		return nil, fmt.Errorf("native performance decode handoff: %w", err)
	}
	if err := s.SetQwen35DecodeHandoffMode(handoffMode); err != nil {
		return nil, fmt.Errorf("native performance decode handoff unavailable: %w", err)
	}
	profiler := model.NewPhaseProfiler()
	s.PhaseProfiler = profiler
	return profiler, nil
}

// runNativeProfileForward runs the measured forward pass (prefill with the candidate
// route finalized inside the timed window, first-token, 63 steady steps), appending
// the "prefill"/"first-token"/"steady-decode" phases; returns final logits + executed.
func runNativeProfileForward(s *model.Session, vocab int, sequenceSelector string, phases []nativeperf.ProfilePhase) ([]float32, bool, []nativeperf.ProfilePhase, error) {
	defer s.BeginGPUKeepAlive()()
	prompt := lcgIDs(32, vocab)
	t := time.Now()
	logits := s.Prefill(prompt)
	sequenceExecuted := false
	if sequenceSelector == nativeProfileSelectorOn {
		var err error
		sequenceExecuted, err = s.FinalizeQwen35MetalGDNPreprojectedSequence()
		if err != nil {
			return nil, false, phases, fmt.Errorf("native performance candidate route failed: %w", err)
		}
	} else {
		_, sequenceExecuted = s.Qwen35GDNDecodePath()
	}
	d := time.Since(t)
	phases = appendNativeProfilePhase(phases, "prefill", d)

	id := 7 % vocab
	t = time.Now()
	logits = s.Step(id)
	d = time.Since(t)
	phases = appendNativeProfilePhase(phases, "first-token", d)

	t = time.Now()
	for i := 1; i < 64; i++ {
		id = (id*48271 + 1) % vocab
		logits = s.Step(id)
	}
	d = time.Since(t)
	phases = appendNativeProfilePhase(phases, "steady-decode", d)
	return logits, sequenceExecuted, phases, nil
}

// nativeProfileIdentity is every identity witness the native profile + receipt bind to.
type nativeProfileIdentity struct {
	artifactFile    nativeFileIdentity
	envelope        nativeperf.Envelope
	forwardPath     string
	host            nativeHostIdentity
	source          nativeSourceIdentity
	binary          nativeFileIdentity
	loadedConfig    map[string]any
	loadedConfigSHA string
}

// captureNativeProfileIdentity collects and cross-checks every identity the native
// profile attests: artifact bytes, pinned envelope + P/T controls, executed forward
// path, pinned model name, host + build identities, and loaded-vs-header model config.
func captureNativeProfileIdentity(f *benchFlags, m *model.Model, sequenceSelector string, sequenceExecuted bool) (nativeProfileIdentity, error) {
	artifactFile, err := fileIdentity(*f.gguf)
	if err != nil {
		return nativeProfileIdentity{}, fmt.Errorf("native performance artifact identity: %w", err)
	}
	envelope, err := exactMetalProfileEnvelope(nativeperf.ActiveGraph(), artifactFile)
	if err != nil {
		return nativeProfileIdentity{}, fmt.Errorf("native performance profile unavailable: %w", err)
	}
	if envelope.PromptTokens != 32 || envelope.DecodeTokens != 64 || *f.decodePrompt != envelope.PromptTokens || *f.decodeSteps != envelope.DecodeTokens {
		return nativeProfileIdentity{}, fmt.Errorf("native performance P/T controls do not match envelope: got P=%d T=%d, want P=%d T=%d", *f.decodePrompt, *f.decodeSteps, envelope.PromptTokens, envelope.DecodeTokens)
	}
	forwardPath, err := nativeProfileExecutedForwardPath(envelope.ForwardPath, sequenceSelector, sequenceExecuted)
	if err != nil {
		return nativeProfileIdentity{}, fmt.Errorf("native performance profile unavailable: %w", err)
	}
	if *f.name != "qwen38:27b" {
		return nativeProfileIdentity{}, fmt.Errorf("native performance model name %q is not the pinned qwen38:27b identity", *f.name)
	}
	host, err := captureNativeHost()
	if err != nil {
		return nativeProfileIdentity{}, fmt.Errorf("native performance host identity: %w", err)
	}
	if err := validateNativeHost(envelope, host); err != nil {
		return nativeProfileIdentity{}, fmt.Errorf("native performance profile unavailable: %w", err)
	}
	source, binary, err := captureNativeBuild()
	if err != nil {
		return nativeProfileIdentity{}, fmt.Errorf("native performance build identity: %w", err)
	}
	weights, err := ggufload.OpenWeights(*f.gguf)
	if err != nil {
		return nativeProfileIdentity{}, fmt.Errorf("native performance artifact config: %w", err)
	}
	headerConfig, configErr := weights.File.Config()
	weights.Close()
	if configErr != nil {
		return nativeProfileIdentity{}, fmt.Errorf("native performance artifact config: %w", configErr)
	}
	loadedConfig, err := nativeModelConfigIdentity(m.Cfg)
	if err != nil {
		return nativeProfileIdentity{}, err
	}
	headerConfigReport, err := nativeModelConfigIdentity(headerConfig)
	if err != nil {
		return nativeProfileIdentity{}, err
	}
	loadedConfigSHA, err := sha256JSON(loadedConfig)
	if err != nil {
		return nativeProfileIdentity{}, err
	}
	headerConfigSHA, err := sha256JSON(headerConfigReport)
	if err != nil || headerConfigSHA != loadedConfigSHA {
		return nativeProfileIdentity{}, fmt.Errorf("native performance loaded model config does not match exact artifact header")
	}
	return nativeProfileIdentity{
		artifactFile:    artifactFile,
		envelope:        envelope,
		forwardPath:     forwardPath,
		host:            host,
		source:          source,
		binary:          binary,
		loadedConfig:    loadedConfig,
		loadedConfigSHA: loadedConfigSHA,
	}, nil
}

func maybeRunMTPComparison(f *benchFlags) bool {
	readbackPath := *macbenchMTPReadback
	if readbackPath == "" {
		readbackPath = *macbenchMTPReadbackAlt
	}
	if readbackPath != "" {
		data, err := os.ReadFile(readbackPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "macbench mtp readback: read file: %v\n", err)
			f.exit(1)
		}
		var packet macbench.MTPComparisonPacket
		if err := json.Unmarshal(data, &packet); err != nil {
			fmt.Fprintf(os.Stderr, "macbench mtp readback: decode packet: %v\n", err)
			f.exit(1)
		}
		if err := macbench.ValidateMTPComparisonEvidence(packet, readbackPath); err != nil {
			fmt.Fprintf(os.Stderr, "macbench mtp readback: invalid packet: %v\n", err)
			f.exit(1)
		}
		fmt.Printf("VALID mtp_comparison_packet schema=%s campaign=%s host=%s fak_native_decode=%.2f tok/s\n",
			packet.Schema, packet.CampaignID, packet.HostID, packet.Summary.FakNativeDecodeTokS)
		return true
	}

	dryRun := *macbenchMTPDryRun || *macbenchMTPDryRunAlt
	runMTP := *macbenchMTP || *macbenchMTPAlt || dryRun
	if !runMTP {
		return false
	}

	opts := macbench.DefaultMTPRunnerOptions()
	opts.Adapters = macbench.DefaultMTPAdapters()

	if dryRun {
		if err := macbench.ValidateMTPRunnerEnvelope(opts); err != nil {
			fmt.Fprintf(os.Stderr, "macbench mtp dry-run invalid: %v\n", err)
			f.exit(1)
		}
		fmt.Printf("DRY_RUN_PLAN_VALID campaign=%s host=%s model=%s draft_depth=%d\n",
			opts.CampaignID, opts.HostID, opts.Model.ID, opts.SpeculativeConfig.DraftDepth)
		return true
	}

	runner := macbench.NewMTPRunner(opts)
	packet, err := runner.Run(context.Background())
	if err != nil {
		fmt.Fprintf(os.Stderr, "macbench mtp run: %v\n", err)
		f.exit(1)
	}

	outPath := *macbenchMTPOut
	if outPath == "" {
		outPath = *macbenchMTPOutAlt
	}
	if outPath == "" && f.out != nil && *f.out != "" {
		outPath = *f.out
	}

	b, err := json.MarshalIndent(packet, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "macbench mtp marshal: %v\n", err)
		f.exit(1)
	}

	if outPath != "" {
		if err := os.WriteFile(outPath, b, 0644); err != nil {
			fmt.Fprintf(os.Stderr, "macbench mtp write %s: %v\n", outPath, err)
			f.exit(1)
		}
		fmt.Printf("WROTE %s (%.2f tok/s sustained fak-native decode)\n", outPath, packet.Summary.FakNativeDecodeTokS)
	} else {
		fmt.Println(string(b))
	}
	return true
}
