package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func TestTurnkeyMTPQualificationCatalogSelectsOnlyExactReviewedContext(t *testing.T) {
	now := time.Now().UTC().Truncate(time.Second)
	qualification := turnkeyMTPQualificationKey{
		ModelArtifactSHA256: strings.Repeat("ab", 32),
		ModelFamily:         "Qwen3.8",
		MTPFormat:           fakmodel.Qwen38MTPFormatQ4K,
		DeviceCompatibility: "apple/m3-pro-36gb/v1",
		MetalCompatibility:  "fak-metal/qwen38-mtp/v1",
		EngineCompatibility: "fak-native/qwen38-hybrid/v1",
		NativeConfigSHA256:  strings.Repeat("cd", 32),
		Workload: turnkeyMTPWorkloadEnvelope{
			Greedy: true, ContextTokens: 4096, DraftDepth: 3,
		},
	}
	context := turnkeyMTPQualificationContext{Qualification: qualification, AvailableHeadroomBytes: 4 << 30}
	evidence := fakmodel.Qwen38MTPCanaryEvidence{
		Receipt: fakmodel.Qwen38MTPCanaryReceipt{
			SchemaVersion: fakmodel.Qwen38MTPCanaryReceiptSchema,
			ReceiptID:     "qwen38-m3pro-reviewed-1", DefaultOn: true,
			Engine: fakmodel.Qwen38EngineMTP,
			Envelope: fakmodel.Qwen38CanaryEnvelope{
				ModelFamily: "Qwen3.8", Format: fakmodel.Qwen38MTPFormatQ4K,
				Backend: fakmodel.Qwen38MTPBackendMetal, HeadroomBytes: 3 << 30,
				ArtifactHash: qualification.ModelArtifactSHA256, DraftDepth: 3,
			},
			Speedup: 1.1, TokensProduced: 4, TokensProposed: 3, TokensAccepted: 3,
			DowngradeReason: fakmodel.Qwen38MTPEligible, CircuitStatus: fakmodel.CanaryCircuitClosed,
			LatencyNS:   fakmodel.Qwen38MTPLatencyNS{Setup: 1, Draft: 1, Verify: 1, Total: 3},
			MemoryBytes: fakmodel.Qwen38MTPMemoryBytes{DraftWorkspace: 1, VerifyWorkspace: 1, Peak: 1},
		},
		ObservedAt: now.Add(-time.Hour), ValidUntil: now.Add(time.Hour),
	}
	revision := strings.Repeat("1a", 20)
	catalog := turnkeyMTPTestCatalog(t, evidence, context, revision, false)

	result := selectTurnkeyMTPQualificationCatalog(catalog, now, context)
	if result.Refusal != turnkeyMTPQualificationAvailable || result.Selection == nil {
		t.Fatalf("exact reviewed context result = %+v", result)
	}
	if result.Selection.ReceiptID != evidence.Receipt.ReceiptID || result.Selection.Qualification != qualification || result.Selection.RuntimeContext != context || result.Selection.RequiredHeadroomBytes != 3<<30 {
		t.Fatalf("selection lost bound identity/context: %+v", result.Selection)
	}
	mgr := fakmodel.NewQwen38MTPCanaryManager()
	if err := mgr.RegisterCanaryEvidence(result.Selection.Evidence); err != nil {
		t.Fatal(err)
	}
	decision := mgr.EvaluateCanary(fakmodel.Qwen38CanaryRequest{
		Envelope:          result.Selection.Evidence.Receipt.Envelope,
		EvidenceReceiptID: result.Selection.ReceiptID,
		ModelReady:        true,
	})
	if decision.Engine != fakmodel.Qwen38EngineMTP || !decision.CanaryDefaultOn {
		t.Fatalf("selected evidence did not interoperate with canary evaluator: %+v", decision)
	}

	for name, mutate := range map[string]func(*turnkeyMTPQualificationContext){
		"artifact": func(c *turnkeyMTPQualificationContext) {
			c.Qualification.ModelArtifactSHA256 = strings.Repeat("ef", 32)
		},
		"model family": func(c *turnkeyMTPQualificationContext) { c.Qualification.ModelFamily = "qwen3.8" },
		"device":       func(c *turnkeyMTPQualificationContext) { c.Qualification.DeviceCompatibility = "apple/m3-max-36gb/v1" },
		"metal": func(c *turnkeyMTPQualificationContext) {
			c.Qualification.MetalCompatibility = "fak-metal/qwen38-mtp/v2"
		},
		"engine": func(c *turnkeyMTPQualificationContext) {
			c.Qualification.EngineCompatibility = "fak-native/qwen38-hybrid/v2"
		},
		"native config": func(c *turnkeyMTPQualificationContext) { c.Qualification.NativeConfigSHA256 = strings.Repeat("12", 32) },
		"workload":      func(c *turnkeyMTPQualificationContext) { c.Qualification.Workload.ContextTokens++ },
		"headroom":      func(c *turnkeyMTPQualificationContext) { c.AvailableHeadroomBytes = 2 << 30 },
	} {
		t.Run(name+" mismatch", func(t *testing.T) {
			gotContext := context
			mutate(&gotContext)
			got := selectTurnkeyMTPQualificationCatalog(catalog, now, gotContext)
			if got.Refusal != turnkeyMTPNoEligibleContext || got.Selection != nil {
				t.Fatalf("mismatch result = %+v", got)
			}
		})
	}

	missingRuntime := context
	missingRuntime.Qualification.DeviceCompatibility = ""
	if got := selectTurnkeyMTPQualificationCatalog(catalog, now, missingRuntime); got.Refusal != turnkeyMTPNoEligibleContext || got.Selection != nil {
		t.Fatalf("missing runtime context = %+v", got)
	}
	if got := selectTurnkeyMTPQualificationCatalog(turnkeyMTPTestCatalog(t, evidence, context, revision, true), now, context); got.Refusal != turnkeyMTPNoEligibleContext || got.Selection != nil {
		t.Fatalf("revoked evidence = %+v", got)
	}
	if got := selectTurnkeyMTPQualificationCatalog(catalog, evidence.ValidUntil, context); got.Refusal != turnkeyMTPNoEligibleContext || got.Selection != nil {
		t.Fatalf("expired evidence = %+v", got)
	}
	if got := selectTurnkeyMTPQualificationCatalog(catalog, evidence.ObservedAt.Add(-time.Second), context); got.Refusal != turnkeyMTPNoEligibleContext || got.Selection != nil {
		t.Fatalf("future-observed evidence = %+v", got)
	}
	if got := embeddedTurnkeyMTPQualification(now, context); got.Refusal != turnkeyMTPNoEligibleContext || got.Selection != nil {
		t.Fatalf("empty production catalog = %+v", got)
	}

	var decoded map[string]any
	if err := json.Unmarshal(catalog, &decoded); err != nil {
		t.Fatal(err)
	}
	decoded["unexpected"] = true
	malformed, _ := json.Marshal(decoded)
	if got := selectTurnkeyMTPQualificationCatalog(malformed, now, context); got.Refusal != turnkeyMTPMalformedCatalog || got.Selection != nil {
		t.Fatalf("unknown-field catalog = %+v", got)
	}
	for name, duplicate := range map[string][]byte{
		"top level":  []byte(`{"schema":"fak.up.mtp-evidence-catalog/1","records":[],"records":[],"revocations":[]}`),
		"case alias": []byte(`{"schema":"fak.up.mtp-evidence-catalog/1","records":[],"revocations":[],"Revocations":[]}`),
		"context":    []byte(strings.Replace(string(catalog), `"greedy":true`, `"greedy":true,"greedy":true`, 1)),
		"evidence":   []byte(strings.Replace(string(catalog), `"receipt_id":"qwen38-m3pro-reviewed-1"`, `"receipt_id":"qwen38-m3pro-reviewed-1","receipt_id":"qwen38-m3pro-reviewed-1"`, 1)),
	} {
		t.Run("duplicate key "+name, func(t *testing.T) {
			if got := selectTurnkeyMTPQualificationCatalog(duplicate, now, context); got.Refusal != turnkeyMTPMalformedCatalog || got.Selection != nil {
				t.Fatalf("duplicate-key catalog = %+v", got)
			}
		})
	}
	var validCatalog turnkeyMTPEvidenceCatalog
	if err := json.Unmarshal(catalog, &validCatalog); err != nil {
		t.Fatal(err)
	}
	if valid := selectTurnkeyMTPQualificationCatalog(catalog, now, context); valid.Selection == nil {
		t.Fatalf("valid prefix precondition = %+v", valid)
	}
	validCatalog.Records = append(validCatalog.Records, turnkeyMTPEvidenceRecord{Evidence: json.RawMessage(`{}`)})
	malformed, _ = json.Marshal(validCatalog)
	if got := selectTurnkeyMTPQualificationCatalog(malformed, now, context); got.Refusal != turnkeyMTPMalformedCatalog || got.Selection != nil {
		t.Fatalf("partially malformed catalog was not atomic: %+v", got)
	}
}

func turnkeyMTPTestCatalog(t *testing.T, evidence fakmodel.Qwen38MTPCanaryEvidence, context turnkeyMTPQualificationContext, revision string, revoked bool) []byte {
	t.Helper()
	evidenceRaw, err := json.Marshal(evidence)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(evidenceRaw)
	digest := hex.EncodeToString(sum[:])
	record := turnkeyMTPEvidenceRecord{
		Evidence: evidenceRaw, EvidenceSHA256: digest,
		SourceURL:            "https://github.com/anthony-chaudhary/fak/blob/" + revision + "/experiments/benchmark/mtp/receipt.json",
		SourceRevision:       revision,
		TestedSourceRevision: strings.Repeat("2b", 20),
		Qualification:        context.Qualification,
	}
	catalog := turnkeyMTPEvidenceCatalog{Schema: turnkeyMTPEvidenceCatalogSchema, Records: []turnkeyMTPEvidenceRecord{record}}
	if revoked {
		catalog.Revocations = []turnkeyMTPEvidenceRevocation{{ReceiptID: evidence.Receipt.ReceiptID, EvidenceSHA256: digest}}
	} else {
		catalog.Revocations = []turnkeyMTPEvidenceRevocation{}
	}
	raw, err := json.Marshal(catalog)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// turnkeyMTPStartupProbe records what the startup loader did through its
// qualification seams so each case can witness the side effects, not only the
// reported status.
type turnkeyMTPStartupProbe struct {
	hashCalls    int
	retainCalls  int
	restoreCalls int
	loadCalls    int
	model        *fakmodel.Model
	planner      *agent.InKernelPlanner
}

// retaining reports whether a retainMTPHead seam call is active right now (a
// retain whose restore has not run yet).
func (p *turnkeyMTPStartupProbe) retaining() bool { return p.retainCalls > p.restoreCalls }

// turnkeyMTPStartupFixture is one live-startup scenario: the catalog bytes, the
// Metal decision, the Metal device name, the host headroom, and the model loader.
type turnkeyMTPStartupFixture struct {
	catalog        []byte
	metalLive      bool
	device         string
	availableBytes int64
	newModel       func() *fakmodel.Model
	// totalBytes is the host physical memory (0 = 36 GiB).
	totalBytes int64
	// memoryOverride, when true, reports an operator host-memory override.
	memoryOverride bool
	// loadModel, when set, replaces newModel with a loader that sees the probe
	// (retain state) and the artifact path; nil models a failed load.
	loadModel func(probe *turnkeyMTPStartupProbe, path string) *fakmodel.Model
}

const turnkeyMTPStartupContextTokens = 4096

// turnkeyMTPStartupDeps wires loadTurnkeyNativeResourcesWith to synthetic,
// GPU-free facts. The planner is the real in-kernel planner (no Metal residency,
// no backend) so admission runs the production canary path end to end. The
// planner is built with the loader's context bound, the way the production
// turnkey planner is, so the derived workload envelope is a valid reviewed key.
func turnkeyMTPStartupDeps(probe *turnkeyMTPStartupProbe, fx turnkeyMTPStartupFixture) turnkeyNativeLoadDeps {
	newModel := fx.newModel
	if newModel == nil {
		newModel = fakmodel.NewSyntheticQwen38MTP
	}
	totalBytes := fx.totalBytes
	if totalBytes == 0 {
		totalBytes = 36 << 30
	}
	deps := turnkeyNativeLoadDeps{
		resolveBackend: func() (compute.Backend, error) { return nil, nil },
		resolveMetal:   func() (serveMetalDecision, error) { return serveMetalDecision{live: fx.metalLive}, nil },
		refusePeak:     func(string) error { return nil },
		admitAndLoad: func(_ bool, _ string, load func(), _ *serveFitBudget) (func(), error) {
			load()
			return func() {}, nil
		},
		loadModel: func(path string, _ compute.Backend, _ int, _ *serveFitBudget) (*fakmodel.Model, bool, *gateway.ModelLoadProfile) {
			probe.loadCalls++
			if fx.loadModel != nil {
				probe.model = fx.loadModel(probe, path)
				return probe.model, probe.model != nil, nil
			}
			probe.model = newModel()
			return probe.model, true, nil
		},
		loadTokenizer: func(string) (*tokenizer.Tokenizer, bool) { return &tokenizer.Tokenizer{}, true },
		newPlanner: func(m *fakmodel.Model, _ *tokenizer.Tokenizer, _ string, _ bool, _ compute.Backend, _ bool, contextTokens int) *agent.InKernelPlanner {
			probe.planner = agent.NewInKernelPlannerWithConfig(m, nil, "qwen38", false, nil, false, agent.InKernelPlannerConfig{ContextTokens: contextTokens})
			return probe.planner
		},
		hostMemory:  func() (int64, int64, bool) { return totalBytes, fx.availableBytes, true },
		metalDevice: func() string { return fx.device },
		hashArtifact: func(path string) (string, error) {
			probe.hashCalls++
			return turnkeyMTPArtifactSHA256(path)
		},
		retainMTPHead: func() func() {
			probe.retainCalls++
			return func() { probe.restoreCalls++ }
		},
		mtpCatalog: fx.catalog,
	}
	if fx.memoryOverride {
		deps.memoryOverride = func() bool { return true }
	}
	return deps
}

func runTurnkeyMTPStartup(t *testing.T, artifactPath string, fx turnkeyMTPStartupFixture) (turnkeyNativeStartup, *turnkeyMTPStartupProbe) {
	t.Helper()
	probe := &turnkeyMTPStartupProbe{}
	resources, err := loadTurnkeyNativeResourcesWith(context.Background(), artifactPath, "qwen38", turnkeyMTPStartupContextTokens, turnkeyMTPStartupDeps(probe, fx))
	if err != nil {
		t.Fatalf("startup: %v", err)
	}
	resources.closeModel = func() error { return nil }
	t.Cleanup(func() { _ = resources.Close() })
	if resources.Planner != probe.planner || probe.planner == nil {
		t.Fatalf("startup did not return the constructed planner")
	}
	if probe.restoreCalls != probe.retainCalls {
		t.Fatalf("retainMTPHead restores = %d, retains = %d (the head retention leaked)", probe.restoreCalls, probe.retainCalls)
	}
	// Reported status must never contradict what execution would do.
	if got, want := resources.Startup.MTPActive, probe.planner.MetalMTPAdmitted(); got != want {
		t.Fatalf("startup.MTPActive = %v but planner.MetalMTPAdmitted() = %v", got, want)
	}
	if !resources.Startup.MTPActive {
		if resources.Startup.MTPInactiveReason == "" {
			t.Fatalf("inactive startup carries an empty MTPInactiveReason: %+v", resources.Startup)
		}
		if probe.planner.MetalMTPCoordinator() != nil {
			t.Fatalf("inactive startup left a Metal MTP coordinator installed")
		}
	}
	return resources.Startup, probe
}

func turnkeyMTPWriteArtifact(t *testing.T, path string, content []byte) string {
	t.Helper()
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(content)
	return hex.EncodeToString(sum[:])
}

// turnkeyMTPStartupEvidence is a valid default-on Metal MTP canary witness for
// the exact reviewed key, the artifact digest, and the witnessed headroom.
func turnkeyMTPStartupEvidence(receiptID string, key turnkeyMTPQualificationKey, headroom uint64, observedAt, validUntil time.Time) fakmodel.Qwen38MTPCanaryEvidence {
	return fakmodel.Qwen38MTPCanaryEvidence{
		Receipt: fakmodel.Qwen38MTPCanaryReceipt{
			SchemaVersion: fakmodel.Qwen38MTPCanaryReceiptSchema,
			ReceiptID:     receiptID, DefaultOn: true,
			Engine: fakmodel.Qwen38EngineMTP,
			Envelope: fakmodel.Qwen38CanaryEnvelope{
				ModelFamily: "Qwen3.8", Format: key.MTPFormat,
				Backend: fakmodel.Qwen38MTPBackendMetal, HeadroomBytes: headroom,
				ArtifactHash: key.ModelArtifactSHA256, DraftDepth: key.Workload.DraftDepth,
			},
			Speedup: 1.2, TokensProduced: 4, TokensProposed: 3, TokensAccepted: 3,
			DowngradeReason: fakmodel.Qwen38MTPEligible, CircuitStatus: fakmodel.CanaryCircuitClosed,
			LatencyNS:   fakmodel.Qwen38MTPLatencyNS{Setup: 1, Draft: 1, Verify: 1, Total: 3},
			MemoryBytes: fakmodel.Qwen38MTPMemoryBytes{DraftWorkspace: 1, VerifyWorkspace: 1, Peak: 1},
		},
		ObservedAt: observedAt, ValidUntil: validUntil,
	}
}

func turnkeyMTPExpectInactive(t *testing.T, startup turnkeyNativeStartup, reason turnkeyMTPQualificationRefusal) {
	t.Helper()
	if startup.MTPActive {
		t.Fatalf("startup.MTPActive = true, want false: %+v", startup)
	}
	if got := startup.MTPInactiveReason; got != string(reason) {
		t.Fatalf("startup.MTPInactiveReason = %q, want %q (detail %q)", got, reason, startup.MTPInactiveDetail)
	}
}

// TestTurnkeyMTPStartupActivatesOnlyForExactReviewedRuntime is the #13141
// startup regression: fak up derives the qualification key from live startup
// facts, selects only an exactly matching live reviewed record, and reports
// mtp_active=true only when the planner's Metal MTP coordinator is admitted.
// Every mismatch fails closed with a typed reason and no coordinator.
func TestTurnkeyMTPStartupActivatesOnlyForExactReviewedRuntime(t *testing.T) {
	now := time.Now().UTC().Truncate(time.Second)
	artifact := filepath.Join(t.TempDir(), "qwen38-mtp.gguf")
	digest := turnkeyMTPWriteArtifact(t, artifact, []byte("reviewed qwen3.8 mtp artifact bytes"))
	emptyCatalog := []byte(`{"schema":"fak.up.mtp-evidence-catalog/1","records":[],"revocations":[]}`)
	live := func(catalog []byte) turnkeyMTPStartupFixture {
		return turnkeyMTPStartupFixture{catalog: catalog, metalLive: true, device: "Apple M3 Pro", availableBytes: 8 << 30}
	}

	// (a) Empty reviewed catalog: the live key is derived and reported, nothing
	// is hashed or retained, and the fail-closed token is emitted.
	startup, probe := runTurnkeyMTPStartup(t, artifact, live(emptyCatalog))
	turnkeyMTPExpectInactive(t, startup, turnkeyMTPNoEligibleContext)
	if startup.MTPInactiveDetail == "" {
		t.Fatalf("empty catalog left MTPInactiveDetail empty")
	}
	if probe.hashCalls != 0 || probe.retainCalls != 0 {
		t.Fatalf("empty catalog hashed %d / retained %d times, want 0/0", probe.hashCalls, probe.retainCalls)
	}
	if !startup.MetalLive {
		t.Fatalf("startup.MetalLive = false, want true")
	}
	if startup.MTPQualification == nil {
		t.Fatalf("empty catalog did not report the derived runtime qualification key")
	}
	key := *startup.MTPQualification
	layout, err := probe.model.Qwen38MTPTensorLayout()
	if err != nil {
		t.Fatal(err)
	}
	if layout.Format != fakmodel.Qwen38MTPFormatF32 {
		t.Fatalf("synthetic MTP head format = %q, want %q", layout.Format, fakmodel.Qwen38MTPFormatF32)
	}
	wantWorkload := turnkeyMTPWorkloadEnvelope{
		Greedy: true, ContextTokens: probe.planner.RuntimeConfig().ContextTokens, DraftDepth: fakmodel.DefaultMetalMTPConfig().DraftDepth,
	}
	if key.ModelFamily != "Qwen3.8" || key.MTPFormat != layout.Format ||
		key.DeviceCompatibility != "apple/m3-pro-36gb/v1" ||
		key.MetalCompatibility != turnkeyMTPMetalCompatibility || key.EngineCompatibility != turnkeyMTPForwardCompatibility ||
		!turnkeyMTPIsLowerSHA256(key.NativeConfigSHA256) || key.Workload != wantWorkload || key.ModelArtifactSHA256 != "" {
		t.Fatalf("derived qualification key = %+v, want family Qwen3.8, format %q, device apple/m3-pro-36gb/v1, workload %+v, no artifact digest", key, layout.Format, wantWorkload)
	}
	if wantWorkload.ContextTokens != turnkeyMTPStartupContextTokens {
		t.Fatalf("planner context bound = %d, want %d", wantWorkload.ContextTokens, turnkeyMTPStartupContextTokens)
	}
	wantNative, err := turnkeyMTPNativeConfigSHA256(turnkeyMTPRuntimeFacts{
		ResidentQ4K: true, Planner: probe.planner.RuntimeConfig(), MTP: fakmodel.DefaultMetalMTPConfig(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if key.NativeConfigSHA256 != wantNative {
		t.Fatalf("native config identity = %s, want %s (resident q4k + planner config + default MTP config)", key.NativeConfigSHA256, wantNative)
	}

	reviewedKey := key
	reviewedKey.ModelArtifactSHA256 = digest
	reviewedContext := turnkeyMTPQualificationContext{Qualification: reviewedKey}
	revision := strings.Repeat("3c", 20)
	reviewedEvidence := turnkeyMTPStartupEvidence("qwen38-m3pro-startup-1", reviewedKey, 3<<30, now.Add(-time.Hour), now.Add(time.Hour))
	reviewed := turnkeyMTPTestCatalog(t, reviewedEvidence, reviewedContext, revision, false)

	t.Run("reviewed exact runtime activates MTP", func(t *testing.T) {
		startup, probe := runTurnkeyMTPStartup(t, artifact, live(reviewed))
		if !startup.MTPActive || startup.MTPInactiveReason != "" || startup.MTPInactiveDetail != "" {
			t.Fatalf("reviewed exact runtime: active=%v reason=%q detail=%q, want active with no reason", startup.MTPActive, startup.MTPInactiveReason, startup.MTPInactiveDetail)
		}
		if probe.hashCalls != 1 {
			t.Fatalf("hashArtifact calls = %d, want 1", probe.hashCalls)
		}
		if probe.retainCalls != 1 || probe.restoreCalls != 1 {
			t.Fatalf("retainMTPHead = %d, restore = %d, want 1/1", probe.retainCalls, probe.restoreCalls)
		}
		if !probe.planner.MetalMTPAdmitted() {
			t.Fatalf("planner.MetalMTPAdmitted() = false for a reported-active startup")
		}
		if decision := probe.planner.Qwen38MTPCanaryResult(); decision.Engine != fakmodel.Qwen38EngineMTP {
			t.Fatalf("canary decision = %+v, want engine %q", decision, fakmodel.Qwen38EngineMTP)
		}
		coordinator := probe.planner.MetalMTPCoordinator()
		if coordinator == nil {
			t.Fatalf("admitted startup has no Metal MTP coordinator")
		}
		if got := coordinator.Config().DraftDepth; got != reviewedKey.Workload.DraftDepth {
			t.Fatalf("coordinator draft depth = %d, want reviewed %d", got, reviewedKey.Workload.DraftDepth)
		}
		if startup.MTPQualification == nil || startup.MTPQualification.ModelArtifactSHA256 != digest {
			t.Fatalf("reported qualification = %+v, want artifact digest %s", startup.MTPQualification, digest)
		}
		if *startup.MTPQualification != reviewedKey {
			t.Fatalf("reported qualification = %+v, want reviewed key %+v", *startup.MTPQualification, reviewedKey)
		}
	})

	t.Run("artifact bytes mismatch fails closed after one digest", func(t *testing.T) {
		mutated := filepath.Join(t.TempDir(), "qwen38-mtp.gguf")
		mutatedDigest := turnkeyMTPWriteArtifact(t, mutated, []byte("reviewed qwen3.8 mtp artifact bytes, tampered"))
		if mutatedDigest == digest {
			t.Fatal("mutated artifact digest collides with reviewed digest")
		}
		startup, probe := runTurnkeyMTPStartup(t, mutated, live(reviewed))
		turnkeyMTPExpectInactive(t, startup, turnkeyMTPNoEligibleContext)
		if probe.hashCalls != 1 {
			t.Fatalf("hashArtifact calls = %d, want 1", probe.hashCalls)
		}
		if startup.MTPQualification == nil || startup.MTPQualification.ModelArtifactSHA256 != mutatedDigest {
			t.Fatalf("reported qualification = %+v, want observed digest %s", startup.MTPQualification, mutatedDigest)
		}
	})

	t.Run("device mismatch fails closed without hashing", func(t *testing.T) {
		fx := live(reviewed)
		fx.device = "Apple M3 Max"
		startup, probe := runTurnkeyMTPStartup(t, artifact, fx)
		turnkeyMTPExpectInactive(t, startup, turnkeyMTPNoEligibleContext)
		if probe.hashCalls != 0 {
			t.Fatalf("hashArtifact calls = %d, want 0", probe.hashCalls)
		}
		if startup.MTPQualification == nil || startup.MTPQualification.DeviceCompatibility != "apple/m3-max-36gb/v1" {
			t.Fatalf("reported qualification = %+v, want device apple/m3-max-36gb/v1", startup.MTPQualification)
		}
	})

	t.Run("revoked record neither hashes nor retains the head", func(t *testing.T) {
		startup, probe := runTurnkeyMTPStartup(t, artifact, live(turnkeyMTPTestCatalog(t, reviewedEvidence, reviewedContext, revision, true)))
		turnkeyMTPExpectInactive(t, startup, turnkeyMTPNoEligibleContext)
		if probe.hashCalls != 0 || probe.retainCalls != 0 {
			t.Fatalf("revoked record hashed %d / retained %d times, want 0/0", probe.hashCalls, probe.retainCalls)
		}
	})

	t.Run("expired record neither hashes nor retains the head", func(t *testing.T) {
		expired := turnkeyMTPStartupEvidence("qwen38-m3pro-startup-expired", reviewedKey, 3<<30, now.Add(-2*time.Hour), now.Add(-time.Hour))
		startup, probe := runTurnkeyMTPStartup(t, artifact, live(turnkeyMTPTestCatalog(t, expired, reviewedContext, revision, false)))
		turnkeyMTPExpectInactive(t, startup, turnkeyMTPNoEligibleContext)
		if probe.hashCalls != 0 || probe.retainCalls != 0 {
			t.Fatalf("expired record hashed %d / retained %d times, want 0/0", probe.hashCalls, probe.retainCalls)
		}
	})

	t.Run("metal not live derives no qualification", func(t *testing.T) {
		fx := live(reviewed)
		fx.metalLive = false
		startup, probe := runTurnkeyMTPStartup(t, artifact, fx)
		turnkeyMTPExpectInactive(t, startup, turnkeyMTPNoEligibleContext)
		if startup.MetalLive {
			t.Fatalf("startup.MetalLive = true, want false")
		}
		if startup.MTPQualification != nil {
			t.Fatalf("non-Metal startup reported a qualification key: %+v", startup.MTPQualification)
		}
		if probe.retainCalls != 0 || probe.hashCalls != 0 {
			t.Fatalf("non-Metal startup retained %d / hashed %d times, want 0/0", probe.retainCalls, probe.hashCalls)
		}
	})

	t.Run("model without a resident MTP head fails closed", func(t *testing.T) {
		fx := live(reviewed)
		fx.newModel = func() *fakmodel.Model {
			// The same Qwen3.8 hybrid config, built WITHOUT the MTP head tensors.
			return fakmodel.NewSynthetic(fakmodel.NewSyntheticQwen38MTP().Cfg)
		}
		startup, probe := runTurnkeyMTPStartup(t, artifact, fx)
		turnkeyMTPExpectInactive(t, startup, turnkeyMTPNoEligibleContext)
		if !strings.Contains(startup.MTPInactiveDetail, "MTP head") {
			t.Fatalf("MTPInactiveDetail = %q, want it to name the missing MTP head", startup.MTPInactiveDetail)
		}
		if startup.MTPQualification != nil {
			t.Fatalf("headless model reported a qualification key: %+v", startup.MTPQualification)
		}
		if probe.hashCalls != 0 {
			t.Fatalf("hashArtifact calls = %d, want 0", probe.hashCalls)
		}
	})

	t.Run("canary headroom refusal after a catalog match reports the canary reason", func(t *testing.T) {
		lowEvidence := turnkeyMTPStartupEvidence("qwen38-m3pro-startup-lowheadroom", reviewedKey, 1<<30, now.Add(-time.Hour), now.Add(time.Hour))
		fx := live(turnkeyMTPTestCatalog(t, lowEvidence, reviewedContext, revision, false))
		fx.availableBytes = 3 << 29 // 1.5 GiB: >= the witnessed 1 GiB, <= the canary's 2 GiB floor.
		startup, probe := runTurnkeyMTPStartup(t, artifact, fx)
		if startup.MTPActive {
			t.Fatalf("startup.MTPActive = true under the canary headroom floor")
		}
		decision := probe.planner.Qwen38MTPCanaryResult()
		if decision.Engine == fakmodel.Qwen38EngineMTP || decision.DowngradeReason == "" {
			t.Fatalf("canary decision = %+v, want a typed target-decode downgrade", decision)
		}
		if got := startup.MTPInactiveReason; got == "" || got == string(turnkeyMTPNoEligibleContext) || got != string(decision.DowngradeReason) {
			t.Fatalf("MTPInactiveReason = %q, want the canary downgrade reason %q", got, decision.DowngradeReason)
		}
		if got := startup.MTPInactiveReason; got != string(fakmodel.Qwen38MTPQualityOutsideEnvelope) {
			t.Fatalf("MTPInactiveReason = %q, want %q", got, fakmodel.Qwen38MTPQualityOutsideEnvelope)
		}
		if startup.MTPInactiveDetail == "" {
			t.Fatalf("canary refusal left MTPInactiveDetail empty")
		}
		if probe.hashCalls != 1 {
			t.Fatalf("hashArtifact calls = %d, want 1", probe.hashCalls)
		}
		if startup.MTPQualification == nil || startup.MTPQualification.ModelArtifactSHA256 != digest {
			t.Fatalf("reported qualification = %+v, want matched digest %s", startup.MTPQualification, digest)
		}
	})

	t.Run("insufficient witnessed headroom fails closed at selection", func(t *testing.T) {
		fx := live(reviewed)
		fx.availableBytes = 2 << 30 // below the witnessed 3 GiB.
		startup, probe := runTurnkeyMTPStartup(t, artifact, fx)
		turnkeyMTPExpectInactive(t, startup, turnkeyMTPNoEligibleContext)
		if probe.hashCalls != 1 {
			t.Fatalf("hashArtifact calls = %d, want 1", probe.hashCalls)
		}
	})

	t.Run("unreadable artifact fails closed", func(t *testing.T) {
		missing := filepath.Join(t.TempDir(), "absent.gguf")
		startup, probe := runTurnkeyMTPStartup(t, missing, live(reviewed))
		turnkeyMTPExpectInactive(t, startup, turnkeyMTPNoEligibleContext)
		if probe.hashCalls != 1 {
			t.Fatalf("hashArtifact calls = %d, want 1", probe.hashCalls)
		}
		if !strings.Contains(startup.MTPInactiveDetail, "artifact digest") {
			t.Fatalf("MTPInactiveDetail = %q, want the artifact digest failure", startup.MTPInactiveDetail)
		}
	})

	t.Run("two live records for one runtime are ambiguous", func(t *testing.T) {
		second := turnkeyMTPStartupEvidence("qwen38-m3pro-startup-2", reviewedKey, 3<<30, now.Add(-time.Hour), now.Add(time.Hour))
		var first, other turnkeyMTPEvidenceCatalog
		if err := json.Unmarshal(reviewed, &first); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(turnkeyMTPTestCatalog(t, second, reviewedContext, revision, false), &other); err != nil {
			t.Fatal(err)
		}
		first.Records = append(first.Records, other.Records...)
		both, err := json.Marshal(first)
		if err != nil {
			t.Fatal(err)
		}
		startup, _ := runTurnkeyMTPStartup(t, artifact, live(both))
		turnkeyMTPExpectInactive(t, startup, turnkeyMTPAmbiguousQualification)
	})

	t.Run("malformed catalog fails closed without retaining the head", func(t *testing.T) {
		var decoded map[string]any
		if err := json.Unmarshal(reviewed, &decoded); err != nil {
			t.Fatal(err)
		}
		decoded["unexpected"] = true
		malformed, err := json.Marshal(decoded)
		if err != nil {
			t.Fatal(err)
		}
		startup, probe := runTurnkeyMTPStartup(t, artifact, live(malformed))
		turnkeyMTPExpectInactive(t, startup, turnkeyMTPMalformedCatalog)
		if probe.retainCalls != 0 || probe.hashCalls != 0 {
			t.Fatalf("malformed catalog retained %d / hashed %d times, want 0/0", probe.retainCalls, probe.hashCalls)
		}
	})
}

// TestTurnkeyMTPDeviceCompatibilityIdentity pins the live device identity: an
// Apple Metal device name plus whole-GiB physical memory, and "" (fail closed)
// for anything underivable.
func TestTurnkeyMTPDeviceCompatibilityIdentity(t *testing.T) {
	for _, tc := range []struct {
		name  string
		bytes int64
		want  string
	}{
		{"Apple M3 Pro", 36 << 30, "apple/m3-pro-36gb/v1"},
		{"Apple M4 Max", 128 << 30, "apple/m4-max-128gb/v1"},
		{"  Apple M3 Pro  ", 36 << 30, "apple/m3-pro-36gb/v1"},
		{"", 36 << 30, ""},
		{"Apple", 36 << 30, ""},
		{"Apple ", 36 << 30, ""},
		{"Apple !!", 36 << 30, ""},
		{"AMD Radeon Pro 5500M", 16 << 30, ""},
		{"Intel(R) Iris(TM) Plus Graphics", 16 << 30, ""},
		{"Apple M3 Pro", 0, ""},
		{"Apple M3 Pro", -(36 << 30), ""},
		{"Apple M3 Pro", 36<<30 + 1, ""},
		{"Apple M3 Pro", 36<<30 - 4096, ""},
	} {
		got := turnkeyMTPDeviceCompatibility(tc.name, tc.bytes)
		if got != tc.want {
			t.Errorf("turnkeyMTPDeviceCompatibility(%q, %d) = %q, want %q", tc.name, tc.bytes, got, tc.want)
		}
		if got != "" && !isVersionedCompatibility(got) {
			t.Errorf("device identity %q is not an exact versioned identity", got)
		}
	}
	for _, identity := range []string{turnkeyMTPMetalCompatibility, turnkeyMTPForwardCompatibility} {
		if !isVersionedCompatibility(identity) || identity != strings.TrimSpace(identity) {
			t.Errorf("code-owned identity %q is not an exact versioned identity", identity)
		}
	}
}

// TestTurnkeyMTPNativeConfigSHA256BindsExecutionConfig pins that the native
// config identity is deterministic and moves with every execution-affecting knob.
func TestTurnkeyMTPNativeConfigSHA256BindsExecutionConfig(t *testing.T) {
	base := turnkeyMTPRuntimeFacts{
		ResidentQ4K: true,
		Planner:     agent.InKernelPlannerConfig{ContextTokens: 4096, KVPrecision: fakmodel.KVPrecisionFP32},
		MTP:         fakmodel.DefaultMetalMTPConfig(),
	}
	digest := func(facts turnkeyMTPRuntimeFacts) string {
		t.Helper()
		got, err := turnkeyMTPNativeConfigSHA256(facts)
		if err != nil {
			t.Fatal(err)
		}
		if !turnkeyMTPIsLowerSHA256(got) {
			t.Fatalf("native config identity %q is not lowercase SHA-256", got)
		}
		return got
	}
	want := digest(base)
	if again := digest(base); again != want {
		t.Fatalf("native config identity is not deterministic: %s vs %s", want, again)
	}
	// Fields the identity must ignore: non-execution facts.
	ignored := base
	ignored.ArtifactPath = "/elsewhere/model.gguf"
	ignored.MetalDevice = "Apple M4 Max"
	ignored.HostAvailableBytes = 1 << 30
	if got := digest(ignored); got != want {
		t.Fatalf("native config identity moved with non-execution facts: %s vs %s", got, want)
	}
	for name, mutate := range map[string]func(*turnkeyMTPRuntimeFacts){
		"context tokens": func(f *turnkeyMTPRuntimeFacts) { f.Planner.ContextTokens = 8192 },
		"kv precision":   func(f *turnkeyMTPRuntimeFacts) { f.Planner.KVPrecision = fakmodel.KVPrecisionQ8_0 },
		"resident q4k":   func(f *turnkeyMTPRuntimeFacts) { f.ResidentQ4K = false },
		"mtp draft depth": func(f *turnkeyMTPRuntimeFacts) {
			f.MTP.DraftDepth = 2
		},
	} {
		t.Run(name, func(t *testing.T) {
			facts := base
			mutate(&facts)
			if got := digest(facts); got == want {
				t.Fatalf("native config identity did not change when %s changed", name)
			}
		})
	}
}

// TestTurnkeyMTPArtifactSHA256 pins the streaming artifact digest: exact
// SHA-256 of a regular file, and an error for anything that is not one.
func TestTurnkeyMTPArtifactSHA256(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "artifact.gguf")
	content := []byte(strings.Repeat("qwen3.8 mtp artifact ", 4096))
	want := turnkeyMTPWriteArtifact(t, path, content)
	got, err := turnkeyMTPArtifactSHA256(path)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("turnkeyMTPArtifactSHA256 = %s, want %s", got, want)
	}
	if _, err := turnkeyMTPArtifactSHA256(dir); err == nil {
		t.Fatalf("directory digest succeeded, want an error")
	}
	if _, err := turnkeyMTPArtifactSHA256(filepath.Join(dir, "missing.gguf")); err == nil {
		t.Fatalf("missing-path digest succeeded, want an error")
	}
}

const turnkeyMTPTestEmptyCatalog = `{"schema":"fak.up.mtp-evidence-catalog/1","records":[],"revocations":[]}`

// turnkeyMTPLiveFixture is the default live Metal M3 Pro startup with 8 GiB of
// available headroom against catalog.
func turnkeyMTPLiveFixture(catalog []byte) turnkeyMTPStartupFixture {
	return turnkeyMTPStartupFixture{catalog: catalog, metalLive: true, device: "Apple M3 Pro", availableBytes: 8 << 30}
}

// turnkeyMTPDerivedStartupKey runs the default (always headed) startup against
// the empty reviewed catalog and returns the live-derived qualification key
// (without an artifact digest), so a case can author a record that matches the
// fixture exactly.
func turnkeyMTPDerivedStartupKey(t *testing.T, artifact string) turnkeyMTPQualificationKey {
	t.Helper()
	startup, _ := runTurnkeyMTPStartup(t, artifact, turnkeyMTPLiveFixture([]byte(turnkeyMTPTestEmptyCatalog)))
	if startup.MTPQualification == nil {
		t.Fatalf("empty-catalog startup derived no qualification key: %+v", startup)
	}
	key := *startup.MTPQualification
	if key.ModelArtifactSHA256 != "" {
		t.Fatalf("empty-catalog startup bound an artifact digest: %+v", key)
	}
	return key
}

// turnkeyMTPReviewedCatalog is a well-formed catalog with one live, unrevoked
// reviewed record binding key + digest with 3 GiB of witnessed headroom.
func turnkeyMTPReviewedCatalog(t *testing.T, key turnkeyMTPQualificationKey, digest, receiptID string) []byte {
	t.Helper()
	now := time.Now().UTC().Truncate(time.Second)
	key.ModelArtifactSHA256 = digest
	evidence := turnkeyMTPStartupEvidence(receiptID, key, 3<<30, now.Add(-time.Hour), now.Add(time.Hour))
	return turnkeyMTPTestCatalog(t, evidence, turnkeyMTPQualificationContext{Qualification: key}, strings.Repeat("3c", 20), false)
}

// turnkeyMTPMergeCatalogs concatenates the records of well-formed catalogs.
func turnkeyMTPMergeCatalogs(t *testing.T, catalogs ...[]byte) []byte {
	t.Helper()
	merged := turnkeyMTPEvidenceCatalog{Schema: turnkeyMTPEvidenceCatalogSchema, Revocations: []turnkeyMTPEvidenceRevocation{}}
	for _, raw := range catalogs {
		var one turnkeyMTPEvidenceCatalog
		if err := json.Unmarshal(raw, &one); err != nil {
			t.Fatal(err)
		}
		merged.Records = append(merged.Records, one.Records...)
		merged.Revocations = append(merged.Revocations, one.Revocations...)
	}
	raw, err := json.Marshal(merged)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func turnkeyMTPExpectActive(t *testing.T, startup turnkeyNativeStartup, probe *turnkeyMTPStartupProbe) {
	t.Helper()
	if !startup.MTPActive || startup.MTPInactiveReason != "" {
		t.Fatalf("startup MTPActive=%v reason=%q detail=%q, want active", startup.MTPActive, startup.MTPInactiveReason, startup.MTPInactiveDetail)
	}
	if !probe.planner.MetalMTPAdmitted() || probe.planner.MetalMTPCoordinator() == nil {
		t.Fatalf("reported-active startup has no admitted Metal MTP coordinator")
	}
}

func turnkeyMTPExpectNoCoordinator(t *testing.T, probe *turnkeyMTPStartupProbe) {
	t.Helper()
	if probe.planner.MetalMTPAdmitted() || probe.planner.MetalMTPCoordinator() != nil {
		t.Fatalf("refused startup admitted/installed a Metal MTP coordinator")
	}
}

// TestTurnkeyMTPStartupMemoryOverrideRefusesBeforeRetainOrHash pins that an
// operator host-memory override makes the device/headroom facts non-live: even
// an exactly matching reviewed catalog neither retains the MTP head, hashes the
// artifact, nor installs a coordinator.
func TestTurnkeyMTPStartupMemoryOverrideRefusesBeforeRetainOrHash(t *testing.T) {
	artifact := filepath.Join(t.TempDir(), "qwen38-mtp.gguf")
	digest := turnkeyMTPWriteArtifact(t, artifact, []byte("reviewed qwen3.8 mtp artifact bytes"))
	reviewed := turnkeyMTPReviewedCatalog(t, turnkeyMTPDerivedStartupKey(t, artifact), digest, "qwen38-m3pro-override-1")

	// Control: without the override the same catalog activates MTP, so the
	// override is the only cause of the refusal below.
	control, controlProbe := runTurnkeyMTPStartup(t, artifact, turnkeyMTPLiveFixture(reviewed))
	turnkeyMTPExpectActive(t, control, controlProbe)

	fx := turnkeyMTPLiveFixture(reviewed)
	fx.memoryOverride = true
	startup, probe := runTurnkeyMTPStartup(t, artifact, fx)
	turnkeyMTPExpectInactive(t, startup, turnkeyMTPNoEligibleContext)
	if !strings.Contains(startup.MTPInactiveDetail, "memory override") {
		t.Fatalf("MTPInactiveDetail = %q, want it to name the host memory override", startup.MTPInactiveDetail)
	}
	if probe.retainCalls != 0 || probe.restoreCalls != 0 {
		t.Fatalf("memory override retained the MTP head %d times (restores %d), want 0", probe.retainCalls, probe.restoreCalls)
	}
	if probe.hashCalls != 0 {
		t.Fatalf("memory override hashed the artifact %d times, want 0", probe.hashCalls)
	}
	turnkeyMTPExpectNoCoordinator(t, probe)
	if startup.MTPQualification != nil {
		t.Fatalf("memory override reported a derived qualification key: %+v", startup.MTPQualification)
	}
}

// TestTurnkeyMTPStartupSplitGGUFShardRefusesWithoutHashing pins that a split
// GGUF shard is never qualified (one digest cannot bind the other shards), even
// when a reviewed record binds the named shard's exact bytes.
func TestTurnkeyMTPStartupSplitGGUFShardRefusesWithoutHashing(t *testing.T) {
	dir := t.TempDir()
	base := filepath.Join(dir, "qwen38-mtp.gguf")
	turnkeyMTPWriteArtifact(t, base, []byte("reviewed qwen3.8 mtp artifact bytes"))
	key := turnkeyMTPDerivedStartupKey(t, base)
	shardBytes := []byte("qwen3.8 mtp split shard 1 of 2 bytes")

	for _, name := range []string{
		"qwen38-00001-of-00002.gguf",
		// The GGUF loader (internal/ggufload shardSuffixRe `-(\d+)-of-(\d+)\.gguf$`)
		// assembles a split set for ANY zero-padding width, so these load as
		// multi-file checkpoints too and must be refused the same way.
		"qwen38-1-of-2.gguf",
		"qwen38-001-of-002.gguf",
	} {
		t.Run(name, func(t *testing.T) {
			shard := filepath.Join(t.TempDir(), name)
			shardDigest := turnkeyMTPWriteArtifact(t, shard, shardBytes)
			reviewed := turnkeyMTPReviewedCatalog(t, key, shardDigest, "qwen38-m3pro-split-1")
			startup, probe := runTurnkeyMTPStartup(t, shard, turnkeyMTPLiveFixture(reviewed))
			turnkeyMTPExpectInactive(t, startup, turnkeyMTPNoEligibleContext)
			if !strings.Contains(strings.ToLower(startup.MTPInactiveDetail), "split") {
				t.Fatalf("MTPInactiveDetail = %q, want it to name the split artifact", startup.MTPInactiveDetail)
			}
			if probe.hashCalls != 0 {
				t.Fatalf("split shard hashed %d times, want 0", probe.hashCalls)
			}
			turnkeyMTPExpectNoCoordinator(t, probe)
		})
	}

	t.Run("control: the same bytes under a single-file name activate", func(t *testing.T) {
		plain := filepath.Join(t.TempDir(), "qwen38-mtp-single.gguf")
		plainDigest := turnkeyMTPWriteArtifact(t, plain, shardBytes)
		reviewed := turnkeyMTPReviewedCatalog(t, key, plainDigest, "qwen38-m3pro-split-1")
		startup, probe := runTurnkeyMTPStartup(t, plain, turnkeyMTPLiveFixture(reviewed))
		turnkeyMTPExpectActive(t, startup, probe)
	})
}

// TestTurnkeyMTPStartupRefusesArtifactReplacedDuringLoad pins that the digest
// is bound to the file observed BEFORE the load: an artifact atomically
// replaced at the same path while loading refuses, even when the reviewed
// record binds the replacement's bytes.
func TestTurnkeyMTPStartupRefusesArtifactReplacedDuringLoad(t *testing.T) {
	keyArtifact := filepath.Join(t.TempDir(), "qwen38-mtp.gguf")
	turnkeyMTPWriteArtifact(t, keyArtifact, []byte("reviewed qwen3.8 mtp artifact bytes"))
	key := turnkeyMTPDerivedStartupKey(t, keyArtifact)
	loadedBytes := []byte("qwen3.8 mtp artifact bytes that were actually loaded")
	replacementBytes := []byte("qwen3.8 mtp replacement bytes renamed in during the load")
	replacementSum := sha256.Sum256(replacementBytes)
	replacementDigest := hex.EncodeToString(replacementSum[:])
	reviewed := turnkeyMTPReviewedCatalog(t, key, replacementDigest, "qwen38-m3pro-swap-1")

	t.Run("control: the reviewed bytes in place activate", func(t *testing.T) {
		artifact := filepath.Join(t.TempDir(), "qwen38-mtp.gguf")
		turnkeyMTPWriteArtifact(t, artifact, replacementBytes)
		startup, probe := runTurnkeyMTPStartup(t, artifact, turnkeyMTPLiveFixture(reviewed))
		turnkeyMTPExpectActive(t, startup, probe)
	})

	t.Run("replacement renamed over the artifact during load refuses", func(t *testing.T) {
		dir := t.TempDir()
		artifact := filepath.Join(dir, "qwen38-mtp.gguf")
		if turnkeyMTPWriteArtifact(t, artifact, loadedBytes) == replacementDigest {
			t.Fatal("loaded bytes collide with the replacement digest")
		}
		var swapErr error
		swaps := 0
		fx := turnkeyMTPLiveFixture(reviewed)
		fx.loadModel = func(_ *turnkeyMTPStartupProbe, path string) *fakmodel.Model {
			if swaps == 0 {
				swaps++
				tmp := filepath.Join(dir, "replacement.gguf.tmp")
				if swapErr = os.WriteFile(tmp, replacementBytes, 0o600); swapErr == nil {
					swapErr = os.Rename(tmp, path)
				}
			}
			return fakmodel.NewSyntheticQwen38MTP()
		}
		startup, probe := runTurnkeyMTPStartup(t, artifact, fx)
		if swapErr != nil || swaps != 1 {
			t.Fatalf("artifact swap during load: swaps=%d err=%v", swaps, swapErr)
		}
		onDisk, err := turnkeyMTPArtifactSHA256(artifact)
		if err != nil || onDisk != replacementDigest {
			t.Fatalf("post-load artifact digest = %s (%v), want the reviewed replacement %s", onDisk, err, replacementDigest)
		}
		turnkeyMTPExpectInactive(t, startup, turnkeyMTPNoEligibleContext)
		if !strings.Contains(startup.MTPInactiveDetail, "not the file observed before the load") {
			t.Fatalf("MTPInactiveDetail = %q, want it to say the artifact is not the file observed before the load", startup.MTPInactiveDetail)
		}
		turnkeyMTPExpectNoCoordinator(t, probe)
		if startup.MTPQualification != nil && startup.MTPQualification.ModelArtifactSHA256 == replacementDigest {
			t.Fatalf("startup reported the swapped-in digest %s as the loaded artifact identity", replacementDigest)
		}
	})
}

// TestTurnkeyMTPStartupRetainsHeadOnlyForTheReviewedDevice pins that head
// retention is scoped to a live record for THIS device identity (Metal device
// name + physical memory), not to any live record.
func TestTurnkeyMTPStartupRetainsHeadOnlyForTheReviewedDevice(t *testing.T) {
	artifact := filepath.Join(t.TempDir(), "qwen38-mtp.gguf")
	digest := turnkeyMTPWriteArtifact(t, artifact, []byte("reviewed qwen3.8 mtp artifact bytes"))
	key := turnkeyMTPDerivedStartupKey(t, artifact)
	if key.DeviceCompatibility != "apple/m3-pro-36gb/v1" {
		t.Fatalf("derived device = %q, want apple/m3-pro-36gb/v1", key.DeviceCompatibility)
	}
	reviewed := turnkeyMTPReviewedCatalog(t, key, digest, "qwen38-m3pro-device-1")

	t.Run("another chip does not retain", func(t *testing.T) {
		fx := turnkeyMTPLiveFixture(reviewed)
		fx.device = "Apple M3 Max"
		startup, probe := runTurnkeyMTPStartup(t, artifact, fx)
		turnkeyMTPExpectInactive(t, startup, turnkeyMTPNoEligibleContext)
		if probe.retainCalls != 0 || probe.restoreCalls != 0 || probe.hashCalls != 0 {
			t.Fatalf("M3 Max retained %d / restored %d / hashed %d times, want 0/0/0", probe.retainCalls, probe.restoreCalls, probe.hashCalls)
		}
	})

	t.Run("the same chip with different memory does not retain", func(t *testing.T) {
		fx := turnkeyMTPLiveFixture(reviewed)
		fx.totalBytes = 18 << 30
		startup, probe := runTurnkeyMTPStartup(t, artifact, fx)
		turnkeyMTPExpectInactive(t, startup, turnkeyMTPNoEligibleContext)
		if probe.retainCalls != 0 || probe.hashCalls != 0 {
			t.Fatalf("18 GiB M3 Pro retained %d / hashed %d times, want 0/0", probe.retainCalls, probe.hashCalls)
		}
	})

	t.Run("the reviewed device retains exactly once", func(t *testing.T) {
		startup, probe := runTurnkeyMTPStartup(t, artifact, turnkeyMTPLiveFixture(reviewed))
		turnkeyMTPExpectActive(t, startup, probe)
		if probe.retainCalls != 1 || probe.restoreCalls != 1 {
			t.Fatalf("M3 Pro retained %d / restored %d times, want 1/1", probe.retainCalls, probe.restoreCalls)
		}
		if probe.loadCalls != 1 {
			t.Fatalf("loadModel calls = %d, want 1", probe.loadCalls)
		}
	})

	t.Run("an expired record for this device beside a live one for another does not retain", func(t *testing.T) {
		now := time.Now().UTC().Truncate(time.Second)
		expiredKey := key
		expiredKey.ModelArtifactSHA256 = digest
		expired := turnkeyMTPTestCatalog(t,
			turnkeyMTPStartupEvidence("qwen38-m3pro-device-expired", expiredKey, 3<<30, now.Add(-2*time.Hour), now.Add(-time.Hour)),
			turnkeyMTPQualificationContext{Qualification: expiredKey}, strings.Repeat("3c", 20), false)
		maxKey := key
		maxKey.DeviceCompatibility = "apple/m3-max-36gb/v1"
		liveOther := turnkeyMTPReviewedCatalog(t, maxKey, digest, "qwen38-m3max-device-1")
		startup, probe := runTurnkeyMTPStartup(t, artifact, turnkeyMTPLiveFixture(turnkeyMTPMergeCatalogs(t, expired, liveOther)))
		turnkeyMTPExpectInactive(t, startup, turnkeyMTPNoEligibleContext)
		if probe.retainCalls != 0 || probe.hashCalls != 0 {
			t.Fatalf("M3 Pro retained %d / hashed %d times for an expired own record, want 0/0", probe.retainCalls, probe.hashCalls)
		}
	})
}

// TestTurnkeyMTPStartupRetainedLoadFailureFallsBackWithoutHead pins that a load
// that fails only because the MTP head was retained retries without it: startup
// succeeds and MTP refuses on the missing head instead of failing `fak up`.
func TestTurnkeyMTPStartupRetainedLoadFailureFallsBackWithoutHead(t *testing.T) {
	artifact := filepath.Join(t.TempDir(), "qwen38-mtp.gguf")
	digest := turnkeyMTPWriteArtifact(t, artifact, []byte("reviewed qwen3.8 mtp artifact bytes"))
	reviewed := turnkeyMTPReviewedCatalog(t, turnkeyMTPDerivedStartupKey(t, artifact), digest, "qwen38-m3pro-fallback-1")

	var retainedAtLoad []bool
	fx := turnkeyMTPLiveFixture(reviewed)
	fx.loadModel = func(probe *turnkeyMTPStartupProbe, _ string) *fakmodel.Model {
		retainedAtLoad = append(retainedAtLoad, probe.retaining())
		if probe.retaining() {
			return nil // the loader refuses the retained MTP layout
		}
		return fakmodel.NewSynthetic(fakmodel.NewSyntheticQwen38MTP().Cfg)
	}
	startup, probe := runTurnkeyMTPStartup(t, artifact, fx) // fails the test on a startup error
	if probe.loadCalls != 2 {
		t.Fatalf("loadModel calls = %d, want 2 (retained attempt + headless retry)", probe.loadCalls)
	}
	if len(retainedAtLoad) != 2 || !retainedAtLoad[0] || retainedAtLoad[1] {
		t.Fatalf("retain state per load = %v, want [true false]", retainedAtLoad)
	}
	if probe.retainCalls != 1 || probe.restoreCalls != 1 {
		t.Fatalf("retained %d / restored %d times, want 1/1", probe.retainCalls, probe.restoreCalls)
	}
	if probe.model == nil {
		t.Fatalf("startup kept no model after the headless retry")
	}
	turnkeyMTPExpectInactive(t, startup, turnkeyMTPNoEligibleContext)
	if !strings.Contains(startup.MTPInactiveDetail, "MTP head") {
		t.Fatalf("MTPInactiveDetail = %q, want it to name the missing MTP head", startup.MTPInactiveDetail)
	}
	if probe.hashCalls != 0 {
		t.Fatalf("headless retry hashed the artifact %d times, want 0", probe.hashCalls)
	}
	turnkeyMTPExpectNoCoordinator(t, probe)
}

// TestTurnkeyMTPStartupRealisticHeadRetention models the production loader: the
// MTP head is resident ONLY when startup retained it. Retention must follow the
// reviewed catalog, and each refusal must keep its own typed reason.
func TestTurnkeyMTPStartupRealisticHeadRetention(t *testing.T) {
	artifact := filepath.Join(t.TempDir(), "qwen38-mtp.gguf")
	digest := turnkeyMTPWriteArtifact(t, artifact, []byte("reviewed qwen3.8 mtp artifact bytes"))
	reviewed := turnkeyMTPReviewedCatalog(t, turnkeyMTPDerivedStartupKey(t, artifact), digest, "qwen38-m3pro-realistic-1")
	headedOnlyWhenRetained := func(fx turnkeyMTPStartupFixture) turnkeyMTPStartupFixture {
		fx.loadModel = func(probe *turnkeyMTPStartupProbe, _ string) *fakmodel.Model {
			if probe.retaining() {
				return fakmodel.NewSyntheticQwen38MTP()
			}
			return fakmodel.NewSynthetic(fakmodel.NewSyntheticQwen38MTP().Cfg)
		}
		return fx
	}

	t.Run("reviewed exact catalog retains the head and activates", func(t *testing.T) {
		startup, probe := runTurnkeyMTPStartup(t, artifact, headedOnlyWhenRetained(turnkeyMTPLiveFixture(reviewed)))
		turnkeyMTPExpectActive(t, startup, probe)
		if probe.retainCalls != 1 || probe.loadCalls != 1 || probe.hashCalls != 1 {
			t.Fatalf("retained %d / loaded %d / hashed %d times, want 1/1/1", probe.retainCalls, probe.loadCalls, probe.hashCalls)
		}
	})

	t.Run("malformed catalog reports MALFORMED without retention", func(t *testing.T) {
		var decoded map[string]any
		if err := json.Unmarshal(reviewed, &decoded); err != nil {
			t.Fatal(err)
		}
		decoded["unexpected"] = true
		malformed, err := json.Marshal(decoded)
		if err != nil {
			t.Fatal(err)
		}
		startup, probe := runTurnkeyMTPStartup(t, artifact, headedOnlyWhenRetained(turnkeyMTPLiveFixture(malformed)))
		turnkeyMTPExpectInactive(t, startup, turnkeyMTPMalformedCatalog)
		if probe.retainCalls != 0 || probe.hashCalls != 0 {
			t.Fatalf("malformed catalog retained %d / hashed %d times, want 0/0", probe.retainCalls, probe.hashCalls)
		}
	})

	t.Run("empty catalog reports no live record", func(t *testing.T) {
		startup, probe := runTurnkeyMTPStartup(t, artifact, headedOnlyWhenRetained(turnkeyMTPLiveFixture([]byte(turnkeyMTPTestEmptyCatalog))))
		turnkeyMTPExpectInactive(t, startup, turnkeyMTPNoEligibleContext)
		if !strings.Contains(startup.MTPInactiveDetail, "no live record") {
			t.Fatalf("MTPInactiveDetail = %q, want it to say the catalog has no live record", startup.MTPInactiveDetail)
		}
		if probe.retainCalls != 0 || probe.hashCalls != 0 {
			t.Fatalf("empty catalog retained %d / hashed %d times, want 0/0", probe.retainCalls, probe.hashCalls)
		}
	})
}

// TestTurnkeyMTPStartupDigestMismatchDetailNamesArtifactDigest pins that a
// digest mismatch is self-explaining: the detail names the observed artifact
// digest prefix.
func TestTurnkeyMTPStartupDigestMismatchDetailNamesArtifactDigest(t *testing.T) {
	artifact := filepath.Join(t.TempDir(), "qwen38-mtp.gguf")
	digest := turnkeyMTPWriteArtifact(t, artifact, []byte("reviewed qwen3.8 mtp artifact bytes"))
	reviewed := turnkeyMTPReviewedCatalog(t, turnkeyMTPDerivedStartupKey(t, artifact), digest, "qwen38-m3pro-tamper-1")
	tampered := filepath.Join(t.TempDir(), "qwen38-mtp.gguf")
	tamperedDigest := turnkeyMTPWriteArtifact(t, tampered, []byte("reviewed qwen3.8 mtp artifact bytes, tampered"))
	if tamperedDigest[:12] == digest[:12] {
		t.Fatal("tampered digest prefix collides with the reviewed digest prefix")
	}
	startup, probe := runTurnkeyMTPStartup(t, tampered, turnkeyMTPLiveFixture(reviewed))
	turnkeyMTPExpectInactive(t, startup, turnkeyMTPNoEligibleContext)
	if startup.MTPInactiveDetail == "" {
		t.Fatalf("digest mismatch left MTPInactiveDetail empty")
	}
	if !strings.Contains(startup.MTPInactiveDetail, tamperedDigest[:12]) {
		t.Fatalf("MTPInactiveDetail = %q, want it to name the artifact digest prefix %s", startup.MTPInactiveDetail, tamperedDigest[:12])
	}
	if probe.hashCalls != 1 {
		t.Fatalf("hashArtifact calls = %d, want 1", probe.hashCalls)
	}
	turnkeyMTPExpectNoCoordinator(t, probe)
}

// TestTurnkeyMTPBoundedDetail pins the 256-byte refusal-detail cap and that it
// never splits a UTF-8 sequence.
func TestTurnkeyMTPBoundedDetail(t *testing.T) {
	ascii := func(n int) string { return strings.Repeat("a", n) }
	for _, tc := range []struct {
		name  string
		in    string
		wantN int // expected output length in bytes
	}{
		{"empty", "", 0},
		{"short", "no live record", len("no live record")},
		{"exactly 256 ASCII", ascii(256), 256},
		{"257 ASCII", ascii(257), 256},
		{"long ASCII", ascii(1000), 256},
		{"2-byte rune straddles byte 256", ascii(255) + "é" + ascii(40), 255},
		{"3-byte rune straddles byte 256", ascii(254) + "€" + ascii(40), 254},
		{"4-byte rune ends at byte 257", ascii(253) + "😀" + ascii(40), 253},
		{"rune starts exactly at byte 256", ascii(256) + "é" + ascii(40), 256},
		{"rune ends exactly at byte 256", ascii(254) + "é" + ascii(40), 256},
		{"short multi-byte unchanged", "détail " + "é", len("détail " + "é")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := turnkeyMTPBoundedDetail(tc.in)
			if len(got) > 256 {
				t.Fatalf("len = %d, want <= 256", len(got))
			}
			if len(got) != tc.wantN {
				t.Fatalf("len = %d, want %d", len(got), tc.wantN)
			}
			if !strings.HasPrefix(tc.in, got) {
				t.Fatalf("output is not a prefix of the input")
			}
			if utf8.ValidString(tc.in) && !utf8.ValidString(got) {
				t.Fatalf("output %q is not valid UTF-8", got)
			}
			if len(tc.in) <= 256 && got != tc.in {
				t.Fatalf("short detail changed: %q -> %q", tc.in, got)
			}
		})
	}
	// The refusal constructor applies the same cap.
	long := errors.New(strings.Repeat("é", 200))
	if got := turnkeyMTPRefusal(turnkeyMTPNoEligibleContext, long).Detail; len(got) > 256 || !utf8.ValidString(got) {
		t.Fatalf("turnkeyMTPRefusal detail len %d valid %v, want <= 256 valid UTF-8", len(got), utf8.ValidString(got))
	}
}

// TestTurnkeyMTPNativeConfigSHA256KVPrecisionSpelling pins that the zero KV
// precision and the explicit f32 spelling are one identity, while a different
// cache precision is a different identity.
func TestTurnkeyMTPNativeConfigSHA256KVPrecisionSpelling(t *testing.T) {
	digest := func(precision fakmodel.KVPrecision) string {
		t.Helper()
		got, err := turnkeyMTPNativeConfigSHA256(turnkeyMTPRuntimeFacts{
			ResidentQ4K: true,
			Planner:     agent.InKernelPlannerConfig{ContextTokens: 4096, KVPrecision: precision},
			MTP:         fakmodel.DefaultMetalMTPConfig(),
		})
		if err != nil {
			t.Fatal(err)
		}
		return got
	}
	zero, fp32 := digest(""), digest(fakmodel.KVPrecisionFP32)
	if zero != fp32 {
		t.Fatalf("KVPrecision \"\" = %s but %q = %s, want one identity for the f32 cache", zero, fakmodel.KVPrecisionFP32, fp32)
	}
	for _, other := range []fakmodel.KVPrecision{fakmodel.KVPrecisionQ8_0, fakmodel.KVPrecisionFP16} {
		if got := digest(other); got == fp32 {
			t.Fatalf("KVPrecision %q shares the f32 identity %s", other, fp32)
		}
	}
}

// TestTurnkeyMTPRuntimeContextRequiresPlannerContextBound pins that an unknown
// planner context bound is a refusal, never a defaulted workload envelope.
func TestTurnkeyMTPRuntimeContextRequiresPlannerContextBound(t *testing.T) {
	facts := turnkeyMTPRuntimeFacts{
		Model:        fakmodel.NewSyntheticQwen38MTP(),
		ArtifactPath: "/models/qwen38-mtp.gguf",
		MetalLive:    true, MetalDevice: "Apple M3 Pro",
		HostTotalBytes: 36 << 30, HostAvailableBytes: 8 << 30,
		ResidentQ4K: true,
		Planner:     agent.InKernelPlannerConfig{ContextTokens: 4096},
		MTP:         fakmodel.DefaultMetalMTPConfig(),
	}
	// Control: the same facts with a known bound derive a valid context.
	runtime, err := turnkeyMTPLiveQualification(facts)
	if err != nil {
		t.Fatalf("control facts refused: %v", err)
	}
	if runtime.Qualification.DeviceCompatibility != "apple/m3-pro-36gb/v1" || runtime.Qualification.Workload.ContextTokens != 4096 {
		t.Fatalf("control runtime = %+v, want device apple/m3-pro-36gb/v1 and context 4096", runtime)
	}
	for _, bound := range []int{0, -1} {
		facts.Planner.ContextTokens = bound
		if runtime, err := turnkeyMTPLiveQualification(facts); err == nil {
			t.Fatalf("ContextTokens=%d derived %+v, want an error", bound, runtime)
		}
	}
}
