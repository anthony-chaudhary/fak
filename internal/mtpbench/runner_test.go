package mtpbench

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type fakeExecutor struct{ prepared *fakePrepared }

func (f fakeExecutor) Preflight(context.Context, Config) (Preflight, error) {
	return Preflight{Schema: Schema + ".preflight", LoadPayload: false}, nil
}
func (f fakeExecutor) Prepare(context.Context, Config) (Prepared, error) {
	if f.prepared != nil {
		f.prepared.prepares++
		if f.prepared.onPrepare != nil {
			f.prepared.onPrepare()
		}
	}
	return f.prepared, nil
}
func (f fakeExecutor) Memory() (MemorySnapshot, error) {
	if f.prepared != nil {
		return f.prepared.Memory()
	}
	return MemorySnapshot{CurrentRSSBytes: 10, SwapUsedBytes: 2, RSSAvailable: true, SwapAvailable: true}, nil
}

func TestRunnerRejectsStartupSwapGrowth(t *testing.T) {
	before := MemorySnapshot{CurrentRSSBytes: 10, SwapUsedBytes: 2, RSSAvailable: true, SwapAvailable: true}
	wantBefore := before
	after := MemorySnapshot{CurrentRSSBytes: 20, SwapUsedBytes: 9, RSSAvailable: true, SwapAvailable: true}
	p := &fakePrepared{cleanup: goodCleanup(), memory: &before}
	p.onPrepare = func() {
		*p.memory = after
	}
	ex := fakeExecutor{prepared: p}
	report, err := (Runner{Executor: ex}).Run(context.Background(), validConfig())
	if err == nil || !strings.Contains(err.Error(), "swap grew during model preparation") {
		t.Fatalf("err=%v, want startup swap rejection", err)
	}
	if p.prepares != 1 || p.closes != 1 || p.warms != 0 || p.runs != 0 {
		t.Fatalf("prepare/cleanup/warm/runs=(%d,%d,%d,%d), want (1,1,0,0)", p.prepares, p.closes, p.warms, p.runs)
	}
	if report.MemoryBefore != wantBefore {
		t.Fatalf("memory before=%+v, want %+v", report.MemoryBefore, wantBefore)
	}
	raw, marshalErr := json.Marshal(report)
	if marshalErr != nil {
		t.Fatal(marshalErr)
	}
	if !bytes.Contains(raw, []byte(`"memory_after_prepare":{"current_rss_bytes":20,"swap_used_bytes":9`)) {
		t.Fatalf("failed-startup report omitted post-prepare memory evidence: %s", raw)
	}
}

func TestRunnerAllowsRSSAbovePrePrepareWhenItRestoresBelowPostPrepare(t *testing.T) {
	pre := MemorySnapshot{CurrentRSSBytes: 10, SwapUsedBytes: 2, RSSAvailable: true, SwapAvailable: true}
	wantPre := pre
	post := MemorySnapshot{CurrentRSSBytes: 20, SwapUsedBytes: 2, RSSAvailable: true, SwapAvailable: true}
	final := MemorySnapshot{CurrentRSSBytes: 15, SwapUsedBytes: 2, RSSAvailable: true, SwapAvailable: true}
	p := &fakePrepared{cleanup: goodCleanup(), memory: &pre, memoryAfterClose: &final}
	p.onPrepare = func() { *p.memory = post }
	report, err := (Runner{Executor: fakeExecutor{prepared: p}}).Run(context.Background(), validConfig())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if report.MemoryBefore != wantPre || report.MemoryAfter != final {
		t.Fatalf("memory evidence before/final = %+v/%+v", report.MemoryBefore, report.MemoryAfter)
	}
	raw, marshalErr := json.Marshal(report)
	if marshalErr != nil {
		t.Fatal(marshalErr)
	}
	if !bytes.Contains(raw, []byte(`"memory_after_prepare":{"current_rss_bytes":20,"swap_used_bytes":2`)) {
		t.Fatalf("successful report omitted post-prepare memory evidence: %s", raw)
	}
}

type fakePrepared struct {
	runs             int
	prepares         int
	warms            int
	closes           int
	mutate           func(int, string, *Observation)
	cleanup          CleanupReceipt
	identity         *Identity
	memory           *MemorySnapshot
	memoryAfterClose *MemorySnapshot
	memoryErr        error
	onPrepare        func()
}

func (f *fakePrepared) Warm(context.Context, string, []int, int) error { f.warms++; return nil }
func (f *fakePrepared) Run(_ context.Context, arm string, _ []int, generated int) (Observation, error) {
	f.runs++
	d := 2 * time.Second
	var receipt *NativeReceipt
	if arm == CandidateArm {
		d = time.Second
		receipt = validReceipt()
	}
	metal := goodMetal()
	if arm == BaselineArm {
		clearDraftProfile(metal)
	}
	obs := Observation{Tokens: make([]int, generated), Elapsed: d, Receipt: receipt, Metal: metal}
	for i := range obs.Tokens {
		obs.Tokens[i] = i + 7
	}
	if f.mutate != nil {
		f.mutate(f.runs, arm, &obs)
	}
	return obs, nil
}
func (f *fakePrepared) Memory() (MemorySnapshot, error) {
	if f.memoryErr != nil {
		return MemorySnapshot{}, f.memoryErr
	}
	if f.memory != nil {
		if f.closes > 0 && f.memoryAfterClose != nil {
			return *f.memoryAfterClose, nil
		}
		return *f.memory, nil
	}
	return MemorySnapshot{CurrentRSSBytes: 10, SwapUsedBytes: 2, RSSAvailable: true, SwapAvailable: true}, nil
}
func (f *fakePrepared) Identity() Identity {
	if f.identity != nil {
		return *f.identity
	}
	return fakeIdentity()
}

func fakeIdentity() Identity {
	h := strings.Repeat("a", 64)
	hw := HardwareIdentity{Hostname: "test", MetalDevice: "Apple Test", OSVersion: "1", OSBuild: "1A", Arch: "arm64", CPUs: 8, PhysicalRAMBytes: 1, MetalMemoryBytes: 1}
	shards := []ArtifactShard{{Path: "model.gguf", Size: 1, RawFileSHA256: h}}
	artifact, _ := digestJSON(shards)
	hostHash, _ := digestJSON(hw)
	hostRaw, _ := json.Marshal(hw)
	return Identity{ArtifactShardSetSHA256: artifact, TokenizerSHA256: h, RuntimeSHA256: h, WorkloadSHA256: h, EnvelopeSHA256: h, HostSHA256: hostHash, Host: string(hostRaw), ArtifactShards: shards, Hardware: hw}
}
func (f *fakePrepared) Close() (CleanupReceipt, error) { f.closes++; return f.cleanup, nil }

func validReceipt() *NativeReceipt {
	h := strings.Repeat("b", 64)
	return &NativeReceipt{Engine: "fak-native", Rounds: 1, Paths: []string{ExpectedPanelPath}, OneOperationRounds: 1,
		TargetVerificationOperations: 1, CommandBuffers: 1, Q6KDownProjectionOperations: 64,
		Q6KHeadOperations: 1, StateSHA256: []string{h}, TransactionSHA256: []string{h}}
}

func validConfig() Config {
	h := strings.Repeat("a", 64)
	id := fakeIdentity()
	index := &BaselineIndex{Schema: Schema + ".baseline-index", CapturedAt: time.Now(), ArtifactShardSetSHA256: id.ArtifactShardSetSHA256, TokenizerSHA256: h, WorkloadSHA256: h, EnvelopeSHA256: id.EnvelopeSHA256, HostSHA256: id.HostSHA256, Entries: []ExternalBaseline{{Engine: "fak-recent", P50TokensSec: 5}, {Engine: "llama.cpp", P50TokensSec: 6}, {Engine: "MLX", P50TokensSec: 7}}}
	raw, _ := json.Marshal(index)
	sum := sha256.Sum256(raw)
	return Config{ArtifactPath: "model.gguf", ExpectedArtifactShardSetSHA256: h, TokenizerPath: "tokenizer.json",
		ExpectedTokenizerSHA256: h, PromptIDs: []int{1, 2}, GeneratedTokens: 8, SamplesPerArm: 20,
		RequiredSpeedup: 2, MaximumCoefficientVariation: .05, BaselineIndex: index, ExpectedBaselineIndexSHA256: hex.EncodeToString(sum[:])}
}

func goodMetal() *MetalProfile {
	return &MetalProfile{Available: true, DraftObserved: true, Q4Operations: 1, Q6Operations: 1, ExecutionSHA256: strings.Repeat("d", 64), FallbackSHA256: strings.Repeat("e", 64), DraftQ4Operations: 1, DraftQ6Operations: 1, DraftQ8Operations: 1, DraftExecutionSHA256: strings.Repeat("1", 64), DraftFallbackSHA256: strings.Repeat("2", 64)}
}

func clearDraftProfile(p *MetalProfile) {
	p.DraftObserved = false
	p.DraftFallbacks = 0
	p.DraftQ4Operations = 0
	p.DraftQ6Operations = 0
	p.DraftQ8Operations = 0
	p.DraftExecutionSHA256 = ""
	p.DraftFallbackSHA256 = ""
}

func TestRunnerABBAAndPassMath(t *testing.T) {
	p := &fakePrepared{cleanup: CleanupReceipt{RetainMTPRestored: true, ModelWeightsClosed: true, Q6LiveBefore: 3, Q6LiveAfter: 3}}
	report, err := (Runner{Executor: fakeExecutor{p}}).Run(context.Background(), validConfig())
	if err != nil {
		t.Fatal(err)
	}
	if report.Status != "PASS" || report.SpeedupRatio != 2 || !report.TokenVectorsEqual {
		t.Fatalf("report=%+v", report)
	}
	if want := []string{CandidateArm, BaselineArm, BaselineArm, CandidateArm}; !slices.Equal(report.Schedule[:4], want) {
		t.Fatalf("schedule=%v", report.Schedule[:4])
	}
	if len(report.Samples) != 40 || report.Arms[0].Samples != 20 || report.Arms[1].Rank != 1 {
		t.Fatalf("samples/arms=%d %+v", len(report.Samples), report.Arms)
	}
}

func TestRunnerFailureGates(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(int, string, *Observation)
		cleanup CleanupReceipt
		want    string
	}{
		{"token-parity", func(n int, arm string, o *Observation) {
			if n == 40 {
				o.Tokens[0]++
			}
		}, goodCleanup(), "token vectors differ"},
		{"receipt-absent", func(_ int, arm string, o *Observation) {
			if arm == CandidateArm {
				o.Receipt = nil
			}
		}, goodCleanup(), "absent target"},
		{"receipt-path", func(_ int, arm string, o *Observation) {
			if arm == CandidateArm {
				o.Receipt.Paths[0] = "wrong"
			}
		}, goodCleanup(), "unexpected verification path"},
		{"receipt-one-op", func(_ int, arm string, o *Observation) {
			if arm == CandidateArm {
				o.Receipt.OneOperationRounds = 0
			}
		}, goodCleanup(), "not one target operation"},
		{"receipt-q6", func(_ int, arm string, o *Observation) {
			if arm == CandidateArm {
				o.Receipt.Q6KDownProjectionOperations = 0
			}
		}, goodCleanup(), "native/Q6/fallback"},
		{"fallback", func(_ int, arm string, o *Observation) {
			if arm == CandidateArm {
				o.Receipt.FallbackCount = 1
			}
		}, goodCleanup(), "native/Q6/fallback"},
		{"cv", func(n int, _ string, o *Observation) {
			if n == 40 {
				o.Elapsed = 10 * time.Second
			}
		}, goodCleanup(), "CV"},
		{"speedup", func(_ int, arm string, o *Observation) {
			if arm == CandidateArm {
				o.Elapsed = 1100 * time.Millisecond
			}
		}, goodCleanup(), "below 2.000000x"},
		{"cleanup", nil, CleanupReceipt{RetainMTPRestored: false, Q6LiveBefore: 1, Q6LiveAfter: 2}, "cleanup gate"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := &fakePrepared{mutate: tc.mutate, cleanup: tc.cleanup}
			_, err := (Runner{Executor: fakeExecutor{p}}).Run(context.Background(), validConfig())
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v, want %q", err, tc.want)
			}
		})
	}
}

func TestConfigAndPreflightFailClosed(t *testing.T) {
	cfg := validConfig()
	cfg.SamplesPerArm = 18
	if _, err := (Runner{Executor: fakeExecutor{}}).Run(context.Background(), cfg); err == nil {
		t.Fatal("accepted <20 samples")
	}
	cfg = validConfig()
	cfg.ExpectedArtifactShardSetSHA256 = ""
	if _, err := (Runner{Executor: fakeExecutor{}}).Run(context.Background(), cfg); err == nil {
		t.Fatal("accepted absent artifact pin")
	}
	if _, err := (Runner{Executor: fakeExecutor{}}).Preflight(context.Background(), Config{}); err == nil {
		t.Fatal("accepted absent preflight artifact")
	}
	if _, err := (Runner{}).Preflight(context.Background(), Config{ArtifactPath: "x"}); err == nil {
		t.Fatal("accepted nil executor")
	}
	cfg = validConfig()
	cfg.RequiredSpeedup = math.NaN()
	if _, err := (Runner{Executor: fakeExecutor{}}).Run(context.Background(), cfg); err == nil {
		t.Fatal("accepted NaN threshold")
	}
}

func TestNewFailClosedGates(t *testing.T) {
	t.Run("invalid identity still cleans", func(t *testing.T) {
		p := &fakePrepared{cleanup: goodCleanup()}
		id := p.Identity()
		id.Hardware.MetalDevice = ""
		p.identity = &id
		_, err := (Runner{Executor: fakeExecutor{p}}).Run(context.Background(), validConfig())
		if err == nil || p.closes == 0 {
			t.Fatalf("err=%v closes=%d", err, p.closes)
		}
	})
	t.Run("primary and cleanup errors preserved", func(t *testing.T) {
		p := &fakePrepared{cleanup: CleanupReceipt{}}
		id := p.Identity()
		id.Hardware.MetalDevice = ""
		p.identity = &id
		_, err := (Runner{Executor: fakeExecutor{p}}).Run(context.Background(), validConfig())
		if err == nil || !strings.Contains(err.Error(), "identity") || !strings.Contains(err.Error(), "cleanup") {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("memory unavailable", func(t *testing.T) {
		m := MemorySnapshot{CurrentRSSBytes: 10, RSSAvailable: true}
		p := &fakePrepared{cleanup: goodCleanup(), memory: &m}
		if _, err := (Runner{Executor: fakeExecutor{p}}).Run(context.Background(), validConfig()); err == nil {
			t.Fatal("accepted unavailable swap")
		}
		if p.prepares != 0 || p.closes != 0 || p.warms != 0 || p.runs != 0 {
			t.Fatalf("work after unavailable pre-observation: prepares=%d closes=%d warms=%d runs=%d", p.prepares, p.closes, p.warms, p.runs)
		}
	})
	t.Run("memory not restored", func(t *testing.T) {
		before := MemorySnapshot{CurrentRSSBytes: 10, SwapUsedBytes: 2, RSSAvailable: true, SwapAvailable: true}
		after := before
		after.CurrentRSSBytes = 11
		p := &fakePrepared{cleanup: goodCleanup(), memory: &before, memoryAfterClose: &after}
		if _, err := (Runner{Executor: fakeExecutor{p}}).Run(context.Background(), validConfig()); err == nil || !strings.Contains(err.Error(), "did not restore") {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("draft profiler unavailable", func(t *testing.T) {
		p := &fakePrepared{cleanup: goodCleanup(), mutate: func(_ int, arm string, o *Observation) {
			if arm == CandidateArm {
				o.Metal.DraftObserved = false
			}
		}}
		if _, err := (Runner{Executor: fakeExecutor{p}}).Run(context.Background(), validConfig()); err == nil || !strings.Contains(err.Error(), "profile incomplete") {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("actual Metal fallback", func(t *testing.T) {
		p := &fakePrepared{cleanup: goodCleanup(), mutate: func(_ int, _ string, o *Observation) { o.Metal.TargetFallbacks = 1 }}
		if _, err := (Runner{Executor: fakeExecutor{p}}).Run(context.Background(), validConfig()); err == nil {
			t.Fatal("accepted actual CPU fallback")
		}
	})
	t.Run("actual draft Metal fallback", func(t *testing.T) {
		p := &fakePrepared{cleanup: goodCleanup(), mutate: func(_ int, arm string, o *Observation) {
			if arm == CandidateArm {
				o.Metal.DraftFallbacks = 1
			}
		}}
		if _, err := (Runner{Executor: fakeExecutor{p}}).Run(context.Background(), validConfig()); err == nil {
			t.Fatal("accepted actual draft CPU fallback")
		}
	})
	t.Run("baseline draft contamination", func(t *testing.T) {
		p := &fakePrepared{cleanup: goodCleanup(), mutate: func(_ int, arm string, o *Observation) {
			if arm == BaselineArm {
				o.Metal.DraftObserved = true
				o.Metal.DraftQ4Operations = 1
				o.Metal.DraftExecutionSHA256 = strings.Repeat("1", 64)
			}
		}}
		if _, err := (Runner{Executor: fakeExecutor{p}}).Run(context.Background(), validConfig()); err == nil || !strings.Contains(err.Error(), "draft contamination") {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("stale baseline index", func(t *testing.T) {
		cfg := validConfig()
		cfg.BaselineIndex.CapturedAt = time.Now().Add(-31 * 24 * time.Hour)
		pinIndex(&cfg)
		if _, err := (Runner{Executor: fakeExecutor{&fakePrepared{cleanup: goodCleanup()}}}).Run(context.Background(), cfg); err == nil {
			t.Fatal("accepted stale index")
		}
	})
	t.Run("baseline digest mismatch", func(t *testing.T) {
		cfg := validConfig()
		cfg.ExpectedBaselineIndexSHA256 = strings.Repeat("f", 64)
		if _, err := (Runner{Executor: fakeExecutor{&fakePrepared{cleanup: goodCleanup()}}}).Run(context.Background(), cfg); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("indexed engine faster", func(t *testing.T) {
		cfg := validConfig()
		cfg.BaselineIndex.Entries[2].P50TokensSec = 9
		pinIndex(&cfg)
		if _, err := (Runner{Executor: fakeExecutor{&fakePrepared{cleanup: goodCleanup()}}}).Run(context.Background(), cfg); err == nil || !strings.Contains(err.Error(), "does not exceed") {
			t.Fatalf("err=%v", err)
		}
	})
	if _, err := digestJSON(math.NaN()); err == nil {
		t.Fatal("digest silently accepted NaN")
	}
}

func TestArtifactSetBindsOrderedShards(t *testing.T) {
	dir := t.TempDir()
	one := filepath.Join(dir, "model-00001-of-00002.gguf")
	two := filepath.Join(dir, "model-00002-of-00002.gguf")
	if err := os.WriteFile(one, []byte("one"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(two, []byte("two"), 0o600); err != nil {
		t.Fatal(err)
	}
	shards, before, err := artifactSet(two)
	if err != nil {
		t.Fatal(err)
	}
	if len(shards) != 2 || shards[0].Path != one || shards[1].Path != two {
		t.Fatalf("shards=%+v", shards)
	}
	if err := os.WriteFile(two, []byte("changed"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, after, err := artifactSet(one)
	if err != nil {
		t.Fatal(err)
	}
	if before == after {
		t.Fatal("shard-set digest ignored content/size mutation")
	}
}

func TestArtifactDigestJSONDistinguishesSetFromRawFiles(t *testing.T) {
	raw, err := json.Marshal(fakeIdentity())
	if err != nil {
		t.Fatal(err)
	}
	text := string(raw)
	if !strings.Contains(text, `"artifact_shard_set_sha256"`) || !strings.Contains(text, `"raw_file_sha256"`) {
		t.Fatalf("ambiguous artifact digest report: %s", text)
	}
	if strings.Contains(text, `"artifact_sha256"`) || strings.Contains(text, `"sha256"`) {
		t.Fatalf("legacy ambiguous artifact digest key remains: %s", text)
	}
}

type fakeDraftCoordinator struct {
	attachErr, receiptErr error
	attached              *model.PhaseProfiler
	receipt               model.MetalMTPDraftPhaseProfilerReceipt
}

func (f *fakeDraftCoordinator) SetDraftPhaseProfiler(p *model.PhaseProfiler) error {
	f.attached = p
	return f.attachErr
}
func (f *fakeDraftCoordinator) DraftPhaseProfilerReceipt() (model.MetalMTPDraftPhaseProfilerReceipt, error) {
	return f.receipt, f.receiptErr
}

func TestDraftProfilerWiringFailClosed(t *testing.T) {
	target := model.NewPhaseProfiler()
	good := &fakeDraftCoordinator{}
	draft, err := installDraftProfiler(good, target)
	if err != nil || draft == nil || draft == target || good.attached != draft {
		t.Fatalf("draft=%p target=%p attached=%p err=%v", draft, target, good.attached, err)
	}
	for _, tc := range []struct {
		name   string
		coord  draftProfilerCoordinator
		target *model.PhaseProfiler
	}{
		{"unavailable", nil, target},
		{"nil-target", good, nil},
		{"alias-error", &fakeDraftCoordinator{attachErr: model.ErrMetalMTPDraftProfilerTargetAlias}, target},
		{"attach-error", &fakeDraftCoordinator{attachErr: errors.New("boom")}, target},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := installDraftProfiler(tc.coord, tc.target); err == nil {
				t.Fatal("accepted invalid draft profiler wiring")
			}
		})
	}

	profile := goodMetal()
	receipt := validDraftProfilerReceipt()
	good.receipt = receipt
	if err := collectDraftProfile(good, profile); err != nil {
		t.Fatal(err)
	}
	if err := collectDraftProfile(&fakeDraftCoordinator{receiptErr: errors.New("read boom")}, goodMetal()); err == nil {
		t.Fatal("accepted draft receipt error")
	}
	bad := receipt
	bad.MetalFallback.PromisedCPUFallbacks = 1
	if err := mergeDraftProfile(goodMetal(), bad); err == nil {
		t.Fatal("accepted nonzero/invalid draft fallback")
	}
	bad = receipt
	bad.MetalExecution.Events = bad.MetalExecution.Events[:1]
	raw, _ := json.Marshal(bad.MetalExecution.Events)
	sum := sha256.Sum256(raw)
	bad.MetalExecution.EventsSHA256 = hex.EncodeToString(sum[:])
	bad.MetalExecution.Counters = metalgemm.ExecutionCounters{CommandBuffers: 1, Encoders: 1, DispatchMilliseconds: 1}
	if err := mergeDraftProfile(goodMetal(), bad); err == nil {
		t.Fatal("accepted missing draft mechanism counters")
	}
	if err := mergeDraftProfile(nil, receipt); err == nil {
		t.Fatal("accepted unavailable target profile")
	}
}

func TestBaselineMetalProfileRejectsEveryDraftField(t *testing.T) {
	clean := goodMetal()
	clearDraftProfile(clean)
	if err := validateMetalProfile(clean, false); err != nil {
		t.Fatal(err)
	}
	mutations := []func(*MetalProfile){
		func(p *MetalProfile) { p.DraftObserved = true }, func(p *MetalProfile) { p.DraftFallbacks = 1 },
		func(p *MetalProfile) { p.DraftQ4Operations = 1 }, func(p *MetalProfile) { p.DraftQ6Operations = 1 }, func(p *MetalProfile) { p.DraftQ8Operations = 1 },
		func(p *MetalProfile) { p.DraftExecutionSHA256 = strings.Repeat("1", 64) }, func(p *MetalProfile) { p.DraftFallbackSHA256 = strings.Repeat("2", 64) },
	}
	for i, mutate := range mutations {
		p := *clean
		mutate(&p)
		if err := validateMetalProfile(&p, false); err == nil {
			t.Fatalf("mutation %d accepted", i)
		}
	}
}

func TestMetalProfileMechanismsMatchArtifactRoles(t *testing.T) {
	baseline := goodMetal()
	clearDraftProfile(baseline)
	if baseline.Q8Operations != 0 {
		t.Fatal("test fixture must model target inventory without Q8_0")
	}
	if err := validateMetalProfile(baseline, false); err != nil {
		t.Fatalf("target Q4/Q6 with no Q8 rejected: %v", err)
	}
	if err := validateMetalProfile(goodMetal(), true); err != nil {
		t.Fatalf("candidate target Q4/Q6 plus draft Q4/Q6/Q8 rejected: %v", err)
	}
	for _, tc := range []struct {
		name         string
		requireDraft bool
		mutate       func(*MetalProfile)
	}{
		{"baseline-target-q4", false, func(p *MetalProfile) { p.Q4Operations = 0 }},
		{"baseline-target-q6", false, func(p *MetalProfile) { p.Q6Operations = 0 }},
		{"candidate-target-q4", true, func(p *MetalProfile) { p.Q4Operations = 0 }},
		{"candidate-target-q6", true, func(p *MetalProfile) { p.Q6Operations = 0 }},
		{"candidate-target-q8-negative", true, func(p *MetalProfile) { p.Q8Operations = -1 }},
		{"candidate-draft-q4", true, func(p *MetalProfile) { p.DraftQ4Operations = 0 }},
		{"candidate-draft-q6", true, func(p *MetalProfile) { p.DraftQ6Operations = 0 }},
		{"candidate-draft-q8", true, func(p *MetalProfile) { p.DraftQ8Operations = 0 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			profile := goodMetal()
			if !tc.requireDraft {
				clearDraftProfile(profile)
			}
			tc.mutate(profile)
			if err := validateMetalProfile(profile, tc.requireDraft); err == nil {
				t.Fatal("accepted missing role-required Metal operation")
			}
		})
	}
}

func validDraftProfilerReceipt() model.MetalMTPDraftPhaseProfilerReceipt {
	events := []metalgemm.ExecutionEvent{
		{Operation: metalgemm.ExecutionQ4KGEMV, CommandBufferID: 1, Committed: true, CompletedWait: true, HostReadback: true, Encoders: 1, GPUMilliseconds: 1, TimingAvailable: true},
		{Operation: metalgemm.ExecutionQ6KGEMV, CommandBufferID: 1, Committed: true, CompletedWait: true, HostReadback: true, Encoders: 1, GPUMilliseconds: 1, TimingAvailable: true},
		{Operation: metalgemm.ExecutionQ8GEMV, CommandBufferID: 1, Committed: true, CompletedWait: true, HostReadback: true, Encoders: 1, GPUMilliseconds: 1, TimingAvailable: true},
	}
	raw, _ := json.Marshal(events)
	sum := sha256.Sum256(raw)
	var empty []model.MetalFallbackEvent
	emptyRaw, _ := json.Marshal(empty)
	emptySum := sha256.Sum256(emptyRaw)
	return model.MetalMTPDraftPhaseProfilerReceipt{MetalExecution: metalgemm.ExecutionReceipt{Schema: "fak-metal-execution-receipt/v1", Events: events, EventsSHA256: hex.EncodeToString(sum[:]), Counters: metalgemm.ExecutionCounters{CommandBuffers: 3, Encoders: 3, DispatchMilliseconds: 3}}, MetalFallback: model.MetalFallbackReceipt{Schema: "fak-metal-fallback-receipt/v1", Events: empty, EventsSHA256: hex.EncodeToString(emptySum[:])}}
}

func pinIndex(cfg *Config) {
	raw, _ := json.Marshal(cfg.BaselineIndex)
	sum := sha256.Sum256(raw)
	cfg.ExpectedBaselineIndexSHA256 = hex.EncodeToString(sum[:])
}

func TestPrepareFailurePropagates(t *testing.T) {
	ex := failingExecutor{}
	if _, err := (Runner{Executor: ex}).Run(context.Background(), validConfig()); !errors.Is(err, errPrepare) {
		t.Fatalf("err=%v", err)
	}
}

func TestPhysicalReceiptGate(t *testing.T) {
	h := strings.Repeat("c", 64)
	valid := func() model.MetalMTPTargetVerificationReceipt {
		return model.MetalMTPTargetVerificationReceipt{
			TargetVerificationReceipt: model.TargetVerificationReceipt{Engine: "fak-native", Path: ExpectedPanelPath, OneOperation: true, TargetVerificationOperations: 1},
			Panel:                     &model.Qwen35MetalMTPVerifyPanelReceipt{Path: ExpectedPanelPath, Committed: true, CompletedWait: true, CommandBuffers: 1, Q6KDownProjectionOperations: 64, Q6KHeadOperations: 1, StateSHA256: h, TransactionSHA256: h},
		}
	}
	if err := accumulateReceipt(&NativeReceipt{}, valid(), 64); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*model.MetalMTPTargetVerificationReceipt){
		func(r *model.MetalMTPTargetVerificationReceipt) { r.TargetDecodeSteps = 1 },
		func(r *model.MetalMTPTargetVerificationReceipt) { r.TargetVerificationOperations = 2 },
		func(r *model.MetalMTPTargetVerificationReceipt) { r.OneOperation = false },
		func(r *model.MetalMTPTargetVerificationReceipt) { r.Engine = "other" },
		func(r *model.MetalMTPTargetVerificationReceipt) { r.Panel.Path = "other" },
		func(r *model.MetalMTPTargetVerificationReceipt) { r.Panel.CommandBuffers = 2 },
		func(r *model.MetalMTPTargetVerificationReceipt) { r.Panel.Q6KDownProjectionOperations = 63 },
		func(r *model.MetalMTPTargetVerificationReceipt) { r.Panel.Q6KHeadOperations = 0 },
		func(r *model.MetalMTPTargetVerificationReceipt) { r.Panel.Committed = false },
		func(r *model.MetalMTPTargetVerificationReceipt) { r.Panel.StateSHA256 = "bad" },
	} {
		r := valid()
		mutate(&r)
		if err := accumulateReceipt(&NativeReceipt{}, r, 64); err == nil {
			t.Fatal("accepted invalid physical receipt")
		}
	}
}

func TestArtifactIdentityGate(t *testing.T) {
	layers := make([]string, 64)
	for i := range layers {
		layers[i] = "linear_attention"
	}
	cfg := model.Config{Name: "Qwen3.8-27B-Q4_K_M", ModelType: "qwen3_5_text", NumLayers: 64, HiddenSize: 5120, MTPNumHiddenLayers: 1, LayerTypes: layers}
	quant := ggufload.ArtifactQuant{Recipe: "Q4_K_M", Q4KResident: true}
	if err := validateArtifact(cfg, quant); err != nil {
		t.Fatal(err)
	}
	bad := cfg
	bad.HiddenSize = 4096
	if err := validateArtifact(bad, quant); err == nil {
		t.Fatal("accepted wrong model shape")
	}
	if err := validateArtifact(cfg, ggufload.ArtifactQuant{Recipe: "Q5_K_M"}); err == nil {
		t.Fatal("accepted wrong quant recipe")
	}
}

var errPrepare = errors.New("prepare failed")

type failingExecutor struct{}

func (failingExecutor) Preflight(context.Context, Config) (Preflight, error) {
	return Preflight{}, errPrepare
}
func (failingExecutor) Memory() (MemorySnapshot, error) {
	return MemorySnapshot{CurrentRSSBytes: 10, SwapUsedBytes: 2, RSSAvailable: true, SwapAvailable: true}, nil
}
func (failingExecutor) Prepare(context.Context, Config) (Prepared, error) { return nil, errPrepare }
func goodCleanup() CleanupReceipt {
	return CleanupReceipt{RetainMTPRestored: true, ModelWeightsClosed: true}
}
