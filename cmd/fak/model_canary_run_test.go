package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestModelCanaryRunStrictConfig(t *testing.T) {
	cfg := validModelCanaryTestConfig()
	body, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var decoded modelCanaryRunConfig
	if err := decodeModelCanaryStrict(body, &decoded); err != nil {
		t.Fatalf("valid strict config: %v", err)
	}
	if _, err := validateModelCanaryConfig(decoded); err != nil {
		t.Fatalf("validate config: %v", err)
	}

	unknown := append(append([]byte(nil), body[:len(body)-1]...), []byte(`,"candidate_runtime":"llama.cpp"}`)...)
	if err := decodeModelCanaryStrict(unknown, &decoded); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("unknown field err=%v", err)
	}
	trailing := append(append([]byte(nil), body...), []byte("\n{}")...)
	if err := decodeModelCanaryStrict(trailing, &decoded); err == nil || !strings.Contains(err.Error(), "multiple JSON values") {
		t.Fatalf("trailing JSON err=%v", err)
	}

	invalid := []struct {
		name   string
		mutate func(*modelCanaryRunConfig)
		want   string
	}{
		{"non-native engine", func(v *modelCanaryRunConfig) { v.Candidate.Engine = "reference-runtime" }, "fak-native"},
		{"port mismatch", func(v *modelCanaryRunConfig) { v.Candidate.ListenerPort++ }, "exact handoff"},
		{"remote endpoint", func(v *modelCanaryRunConfig) {
			v.Candidate.ReadinessEndpoints = []string{"http://example.com:18090/health"}
		}, "not loopback"},
		{"empty launchd identity", func(v *modelCanaryRunConfig) { v.Incumbent.LaunchdTarget = "" }, "launchd_target"},
		{"zero stability", func(v *modelCanaryRunConfig) { v.Cleanup.StabilityDuration = "0s" }, "positive Go duration"},
		{"zero consecutive", func(v *modelCanaryRunConfig) { v.Watcher.ConsecutiveCrossings = 0 }, "at least 1"},
	}
	for _, tc := range invalid {
		t.Run(tc.name, func(t *testing.T) {
			copy := cfg
			tc.mutate(&copy)
			if _, err := validateModelCanaryConfig(copy); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v want substring %q", err, tc.want)
			}
		})
	}
}

func TestModelCanaryRunReceiptHashAndTamperReadback(t *testing.T) {
	start := time.Date(2026, 8, 26, 16, 0, 0, 0, time.UTC)
	receipt := newModelCanaryReceipt(start, digestBytes([]byte("config")), "darwin", "arm64")
	receipt.Outcome = "complete"
	receipt.TerminalPhase = modelCanaryPhaseComplete
	receipt.LeaseReleased = true
	addModelCanaryEvent(&receipt, start, modelCanaryPhaseComplete, "complete", "ok", "")
	finishModelCanaryReceipt(&receipt, start.Add(time.Second))

	path := filepath.Join(t.TempDir(), "receipt.json")
	if err := writeModelCanaryReceiptAtomic(path, receipt); err != nil {
		t.Fatalf("write receipt: %v", err)
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var readback modelCanaryRunReceipt
	if err := decodeModelCanaryStrict(body, &readback); err != nil {
		t.Fatalf("strict receipt readback: %v", err)
	}
	got, err := recomputeModelCanaryEvidenceSHA256(readback)
	if err != nil || got != readback.EvidenceSHA256 {
		t.Fatalf("readback evidence=(%s,%v) want %s", got, err, readback.EvidenceSHA256)
	}

	readback.Events[0].Action = "tampered"
	if err := writeModelCanaryReceiptAtomic(filepath.Join(t.TempDir(), "tampered.json"), readback); err == nil || !strings.Contains(err.Error(), "stale") {
		t.Fatalf("tampered receipt write err=%v", err)
	}
}

func TestModelCanaryRunKConsecutiveResetAndTrip(t *testing.T) {
	cfg := validModelCanaryTestConfig().Watcher
	cfg.ConsecutiveCrossings = 3
	cfg.MaximumRSSBytes = 100
	cfg.MaximumFootprintBytes = 100
	cfg.MaximumSwapGrowthBytes = 100
	cfg.MinimumSystemFreePercent = 10
	cfg.MinimumMemorystatusPercent = 10

	state := modelCanaryGuardState{}
	values := []int64{101, 100, 101, 101, 101}
	for i, rss := range values {
		state = foldModelCanaryGuard(state, cfg, modelCanarySample{
			Sequence: i + 1, RSSBytes: rss, FootprintBytes: 50,
			SwapGrowthBytes: 0, SystemFreePercent: 50, MemorystatusPercent: 50,
		})
		if i < len(values)-1 && state.TrippedMetric != "" {
			t.Fatalf("guard tripped early at sample %d: %+v", i+1, state)
		}
	}
	if state.RSSCrossings != 3 || state.TrippedMetric != "rss_bytes" || state.TripSequence != 5 {
		t.Fatalf("reset/trip state=%+v", state)
	}

	cfg.ConsecutiveCrossings = 2
	state = foldModelCanaryGuard(modelCanaryGuardState{}, cfg, modelCanarySample{
		Sequence: 1, RSSBytes: 101, FootprintBytes: 101, SystemFreePercent: 50, MemorystatusPercent: 50,
	})
	state = foldModelCanaryGuard(state, cfg, modelCanarySample{
		Sequence: 2, RSSBytes: 50, FootprintBytes: 101, SystemFreePercent: 50, MemorystatusPercent: 50,
	})
	if state.RSSCrossings != 0 || state.FootprintCrossings != 2 || state.TrippedMetric != "footprint_bytes" || state.TripSequence != 2 {
		t.Fatalf("independent metric counters=%+v", state)
	}
}

func TestModelCanaryRunRawSamplesRecomputeGuardTripAndRejectTamper(t *testing.T) {
	t.Run("trip from raw", func(t *testing.T) {
		h := newModelCanaryTestHarness("")
		h.requestDoneAfter = -1
		h.sampleRSSKiB = 2
		cfg := validModelCanaryTestConfig()
		cfg.Request.Deadline = "10s"
		cfg.Watcher.MaximumRSSBytes = 1024
		cfg.Watcher.ConsecutiveCrossings = 3
		durations, err := validateModelCanaryConfig(cfg)
		if err != nil {
			t.Fatal(err)
		}
		receipt := executeModelCanaryRun(context.Background(), cfg, durations, digestBytes([]byte("config")), h.dependencies())
		if receipt.TerminalPhase != modelCanaryPhaseTerminalDecision || receipt.Reason != modelCanaryReasonGuardTripped ||
			receipt.Guard.TrippedMetric != "rss_bytes" || receipt.Guard.TripSequence != 3 || len(receipt.Samples) != 3 {
			t.Fatalf("raw guard receipt=%+v actions=%v", receipt, h.actions)
		}
		for i, sample := range receipt.Samples {
			parsed, err := parseModelCanaryRSS([]byte(sample.Raw["ps"]), h.candidate.PID)
			if err != nil || parsed != sample.RSSBytes || sample.Sequence != i+1 {
				t.Fatalf("sample[%d] parsed=%d err=%v row=%+v", i, parsed, err, sample)
			}
		}
	})

	for _, failAt := range []string{"raw_value_tamper", "raw_hash_tamper"} {
		t.Run(failAt, func(t *testing.T) {
			h := newModelCanaryTestHarness(failAt)
			h.requestDoneAfter = -1
			cfg := validModelCanaryTestConfig()
			durations, err := validateModelCanaryConfig(cfg)
			if err != nil {
				t.Fatal(err)
			}
			receipt := executeModelCanaryRun(context.Background(), cfg, durations, digestBytes([]byte("config")), h.dependencies())
			if receipt.TerminalPhase != modelCanaryPhaseMonitoring || receipt.Reason != modelCanaryReasonObservationUnavailable || len(receipt.Samples) != 0 {
				t.Fatalf("tampered raw receipt=%+v actions=%v", receipt, h.actions)
			}
		})
	}
}

func TestModelCanaryRunPhaseAndCleanupOrderingAcrossFailures(t *testing.T) {
	tests := []struct {
		name             string
		failAt           string
		deadline         bool
		wantPhase        modelCanaryPhase
		wantReason       string
		leaseAcquired    bool
		postBootout      bool
		candidateStarted bool
		wantActions      []string
	}{
		{"before lease", "preflight", false, modelCanaryPhasePreflightComplete, modelCanaryReasonPreflightFailed, false, false, false,
			[]string{"preflight"}},
		{"after lease", "verify", false, modelCanaryPhaseIncumbentVerified, modelCanaryReasonIncumbentIdentityMismatch, true, false, false,
			[]string{"preflight", "acquire_gpu_lease", "verify_incumbent", "release_gpu_lease"}},
		{"after incumbent verification", "bootout", false, modelCanaryPhaseIncumbentStopped, modelCanaryReasonBootoutFailed, true, true, false,
			[]string{"preflight", "acquire_gpu_lease", "verify_incumbent", "bootout_incumbent", "restore_incumbent", "endpoints_stable", "release_gpu_lease"}},
		{"after bootout", "start_candidate", false, modelCanaryPhaseCandidateStarted, modelCanaryReasonCandidateStartFailed, true, true, false,
			[]string{"preflight", "acquire_gpu_lease", "verify_incumbent", "bootout_incumbent", "start_candidate", "restore_incumbent", "endpoints_stable", "release_gpu_lease"}},
		{"after candidate start", "candidate_ready", false, modelCanaryPhaseCandidateReady, modelCanaryReasonCandidateReadinessFailed, true, true, true,
			[]string{"preflight", "acquire_gpu_lease", "verify_incumbent", "bootout_incumbent", "start_candidate", "candidate_ready", "signal:TERM", "restore_incumbent", "endpoints_stable", "release_gpu_lease"}},
		{"during sampling", "sample", false, modelCanaryPhaseMonitoring, modelCanaryReasonObservationUnavailable, true, true, true,
			[]string{"preflight", "acquire_gpu_lease", "verify_incumbent", "bootout_incumbent", "start_candidate", "candidate_ready", "start_request", "poll_request", "sample", "stop_request", "signal:TERM", "restore_incumbent", "endpoints_stable", "release_gpu_lease"}},
		{"at request deadline", "", true, modelCanaryPhaseTerminalDecision, modelCanaryReasonRequestDeadline, true, true, true,
			[]string{"preflight", "acquire_gpu_lease", "verify_incumbent", "bootout_incumbent", "start_candidate", "candidate_ready", "start_request", "poll_request", "sample", "sleep", "poll_request", "sample", "sleep", "poll_request", "stop_request", "signal:TERM", "restore_incumbent", "endpoints_stable", "release_gpu_lease"}},
		{"during candidate TERM wait", "term_candidate", false, modelCanaryPhaseCandidateTerminated, modelCanaryReasonCandidateTERMFailed, true, true, true,
			[]string{"preflight", "acquire_gpu_lease", "verify_incumbent", "bootout_incumbent", "start_candidate", "candidate_ready", "start_request", "poll_request", "sample", "sleep", "poll_request", "signal:TERM", "restore_incumbent", "endpoints_stable", "release_gpu_lease"}},
		{"during restore", "restore", false, modelCanaryPhaseIncumbentRestored, modelCanaryReasonRestoreFailed, true, true, true,
			[]string{"preflight", "acquire_gpu_lease", "verify_incumbent", "bootout_incumbent", "start_candidate", "candidate_ready", "start_request", "poll_request", "sample", "sleep", "poll_request", "signal:TERM", "restore_incumbent", "release_gpu_lease"}},
		{"during stability proof", "stable", false, modelCanaryPhaseEndpointsStable, modelCanaryReasonStabilityFailed, true, true, true,
			[]string{"preflight", "acquire_gpu_lease", "verify_incumbent", "bootout_incumbent", "start_candidate", "candidate_ready", "start_request", "poll_request", "sample", "sleep", "poll_request", "signal:TERM", "restore_incumbent", "endpoints_stable", "release_gpu_lease"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			h := newModelCanaryTestHarness(tc.failAt)
			if tc.deadline {
				h.requestDoneAfter = -1
			}
			cfg := validModelCanaryTestConfig()
			durations, err := validateModelCanaryConfig(cfg)
			if err != nil {
				t.Fatal(err)
			}
			receipt := executeModelCanaryRun(context.Background(), cfg, durations, digestBytes([]byte("config")), h.dependencies())
			if receipt.Outcome != "refused" || receipt.TerminalPhase != tc.wantPhase || receipt.Reason != tc.wantReason {
				t.Fatalf("terminal=%s/%s/%s want refused/%s/%s\nactions=%v\nevents=%+v", receipt.Outcome, receipt.TerminalPhase, receipt.Reason, tc.wantPhase, tc.wantReason, h.actions, receipt.Events)
			}
			if !reflect.DeepEqual(h.actions, tc.wantActions) {
				t.Fatalf("actions=%v\nwant=%v", h.actions, tc.wantActions)
			}
			assertModelCanaryEventSequence(t, receipt.Events)
			if tc.leaseAcquired != receipt.LeaseReleased {
				t.Fatalf("lease released=%v want %v actions=%v", receipt.LeaseReleased, tc.leaseAcquired, h.actions)
			}
			if tc.postBootout != receipt.RestorationAttempted {
				t.Fatalf("restoration attempted=%v want %v actions=%v", receipt.RestorationAttempted, tc.postBootout, h.actions)
			}
			if tc.postBootout {
				assertModelCanaryActionBefore(t, h.actions, "restore_incumbent", "release_gpu_lease")
			}
			if !tc.leaseAcquired && !reflect.DeepEqual(h.actions, []string{"preflight"}) {
				t.Fatalf("preflight refusal mutated state: %v", h.actions)
			}
			for _, action := range h.actions {
				if strings.HasPrefix(action, "signal:") && action != "signal:TERM" {
					t.Fatalf("non-TERM signal observed: %v", h.actions)
				}
			}
			if tc.candidateStarted && tc.failAt != "start_candidate" {
				if !containsModelCanaryAction(h.actions, "signal:TERM") {
					t.Fatalf("started candidate received no TERM cleanup: %v", h.actions)
				}
			}
		})
	}
}

func TestModelCanaryRunSuccessCompletesAfterRestoreAndLeaseRelease(t *testing.T) {
	h := newModelCanaryTestHarness("")
	cfg := validModelCanaryTestConfig()
	durations, err := validateModelCanaryConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	receipt := executeModelCanaryRun(context.Background(), cfg, durations, digestBytes([]byte("config")), h.dependencies())
	if receipt.Outcome != "complete" || receipt.TerminalPhase != modelCanaryPhaseComplete || !receipt.LeaseReleased || receipt.RestoredIncumbent == nil {
		t.Fatalf("success receipt=%+v actions=%v", receipt, h.actions)
	}
	if receipt.Engine != "fak-native" || receipt.Request == nil || !receipt.Request.equal(h.request) {
		t.Fatalf("receipt did not bind native engine and request identity: engine=%q request=%+v", receipt.Engine, receipt.Request)
	}
	assertModelCanaryActionBefore(t, h.actions, "signal:TERM", "restore_incumbent")
	assertModelCanaryActionBefore(t, h.actions, "restore_incumbent", "endpoints_stable")
	assertModelCanaryActionBefore(t, h.actions, "endpoints_stable", "release_gpu_lease")
	assertModelCanaryEventSequence(t, receipt.Events)
	if got, err := recomputeModelCanaryEvidenceSHA256(receipt); err != nil || got != receipt.EvidenceSHA256 {
		t.Fatalf("receipt evidence=(%s,%v) want %s", got, err, receipt.EvidenceSHA256)
	}
}

func TestModelCanaryRunMissingRequestEvidenceRefuses(t *testing.T) {
	h := newModelCanaryTestHarness("")
	cfg := validModelCanaryTestConfig()
	durations, err := validateModelCanaryConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	deps := h.dependencies()
	deps.RequestEvidence = nil
	receipt := executeModelCanaryRun(context.Background(), cfg, durations, digestBytes([]byte("config")), deps)
	if receipt.Outcome != "refused" || receipt.TerminalPhase != modelCanaryPhaseTerminalDecision || receipt.Reason != modelCanaryReasonRequestEvidenceUnavailable {
		t.Fatalf("missing request evidence terminal=%s/%s/%s", receipt.Outcome, receipt.TerminalPhase, receipt.Reason)
	}
	assertModelCanaryActionBefore(t, h.actions, "signal:TERM", "restore_incumbent")
	assertModelCanaryActionBefore(t, h.actions, "restore_incumbent", "release_gpu_lease")
}

func TestModelCanaryRunLateTerminalEvidenceRefuses(t *testing.T) {
	h := newModelCanaryTestHarness("")
	h.requestDoneAfter = 1
	cfg := validModelCanaryTestConfig()
	durations, err := validateModelCanaryConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	deps := h.dependencies()
	deps.RequestEvidence = func(modelCanaryProcess) (modelCanaryRequestEvidence, error) {
		body := "late response\n"
		return modelCanaryRequestEvidence{
			CompletedAt: h.now.Add(durations.RequestDeadline + time.Nanosecond).UTC().Format(time.RFC3339Nano),
			Stdout:      body, StdoutBytes: len(body), StdoutSHA256: digestBytes([]byte(body)),
			Stderr: "", StderrBytes: 0, StderrSHA256: digestBytes(nil),
		}, nil
	}
	receipt := executeModelCanaryRun(context.Background(), cfg, durations, digestBytes([]byte("config")), deps)
	if receipt.Outcome != "refused" || receipt.TerminalPhase != modelCanaryPhaseTerminalDecision || receipt.Reason != modelCanaryReasonRequestDeadline {
		t.Fatalf("late request terminal=%s/%s/%s detail=%q", receipt.Outcome, receipt.TerminalPhase, receipt.Reason, receipt.Detail)
	}
	assertModelCanaryActionBefore(t, h.actions, "signal:TERM", "restore_incumbent")
}

func TestModelCanaryRunEnvironmentAndDarwinObservationHelpers(t *testing.T) {
	if got, want := modelCanaryEnvironmentRows(map[string]string{"Z": "last", "A": "first"}), []string{"A=first", "Z=last"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("exact environment=%v want %v", got, want)
	}
	if present, err := classifyModelCanaryLsofExit([]byte("p12\ncfak\nf9\ntIPv6\nn*:8090\n"), nil, 0); err != nil || !present {
		t.Fatalf("present lsof classification=(%v,%v)", present, err)
	}
	if present, err := classifyModelCanaryLsofExit(nil, nil, 1); err != nil || present {
		t.Fatalf("absent lsof classification=(%v,%v)", present, err)
	}
	for _, tc := range []struct {
		name     string
		stdout   []byte
		stderr   []byte
		exitCode int
	}{
		{"success without evidence", nil, nil, 0},
		{"diagnostic is not absence", nil, []byte("permission denied"), 1},
		{"tool failure", nil, nil, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := classifyModelCanaryLsofExit(tc.stdout, tc.stderr, tc.exitCode); err == nil {
				t.Fatal("unavailable lsof observation was accepted")
			}
		})
	}

	target := "gui/501/com.fak.model"
	raw := []byte(target + " = {\n\tpath = /Library/LaunchAgents/com.fak.model.plist\n\tpid = 50123\n}\n")
	pid, plist, err := parseDarwinModelCanaryLaunchctl(raw, target)
	if err != nil || pid != 50123 || plist != filepath.Clean("/Library/LaunchAgents/com.fak.model.plist") {
		t.Fatalf("launchctl parse=(%d,%q,%v)", pid, plist, err)
	}
	if _, _, err := parseDarwinModelCanaryLaunchctl([]byte(target+" = {\n\tpath = relative.plist\n\tpid = 50123\n}\n"), target); err == nil {
		t.Fatal("relative launchctl plist was accepted")
	}
}

func TestModelCanaryRunReusedPIDSentinelRemainsAlive(t *testing.T) {
	h := newModelCanaryTestHarness("reused_pid")
	cfg := validModelCanaryTestConfig()
	durations, err := validateModelCanaryConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	receipt := executeModelCanaryRun(context.Background(), cfg, durations, digestBytes([]byte("config")), h.dependencies())
	if receipt.TerminalPhase != modelCanaryPhaseCandidateTerminated || receipt.Reason != modelCanaryReasonCandidateIdentityMismatch {
		t.Fatalf("terminal=%s/%s actions=%v", receipt.TerminalPhase, receipt.Reason, h.actions)
	}
	if !h.sentinelAlive {
		t.Fatal("reused-PID sentinel was signaled")
	}
	if containsModelCanaryAction(h.actions, "signal:TERM") || containsModelCanaryAction(h.actions, "signal:KILL") {
		t.Fatalf("unmatched PID received a signal: %v", h.actions)
	}
	assertModelCanaryActionBefore(t, h.actions, "identity_refused", "restore_incumbent")
	assertModelCanaryActionBefore(t, h.actions, "restore_incumbent", "release_gpu_lease")
}

func TestModelCanaryRunCancellationDoesNotCancelCleanup(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	h := newModelCanaryTestHarness("")
	h.requestDoneAfter = -1
	h.cancelOnSleep = cancel
	cfg := validModelCanaryTestConfig()
	durations, err := validateModelCanaryConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	receipt := executeModelCanaryRun(ctx, cfg, durations, digestBytes([]byte("config")), h.dependencies())
	if receipt.TerminalPhase != modelCanaryPhaseTerminalDecision || receipt.Reason != modelCanaryReasonCanceled {
		t.Fatalf("canceled terminal=%s/%s actions=%v", receipt.TerminalPhase, receipt.Reason, h.actions)
	}
	if h.cleanupSawCanceled {
		t.Fatalf("request cancellation leaked into cleanup context: %v", h.actions)
	}
	if !receipt.RestorationAttempted || !receipt.LeaseReleased {
		t.Fatalf("canceled run skipped terminal cleanup: %+v actions=%v", receipt, h.actions)
	}
	assertModelCanaryActionBefore(t, h.actions, "stop_request", "signal:TERM")
	assertModelCanaryActionBefore(t, h.actions, "signal:TERM", "restore_incumbent")
	assertModelCanaryActionBefore(t, h.actions, "restore_incumbent", "release_gpu_lease")
}

func validModelCanaryTestConfig() modelCanaryRunConfig {
	root := filepath.VolumeName(os.TempDir()) + string(filepath.Separator)
	return modelCanaryRunConfig{
		Schema: modelCanaryRunConfigSchema,
		Lease:  modelCanaryLeaseConfig{Path: filepath.Join(os.TempDir(), "model-canary-test.lease"), Timeout: "1s"},
		Incumbent: modelCanaryIncumbentConfig{
			ListenerPort: 18090, LaunchdTarget: "gui/501/com.fak.model", ExpectedArgvSHA256: strings.Repeat("a", 64),
			RestorePlist: filepath.Join(root, "Library", "LaunchAgents", "com.fak.model.plist"), RestorePlistSHA256: strings.Repeat("b", 64),
			RestoreCommand:  []string{"launchctl", "bootstrap", "gui/501", "<restore-plist>"},
			StableEndpoints: []string{"http://127.0.0.1:18090/health", "http://127.0.0.1:18090/v1/models"},
		},
		Candidate: modelCanaryCandidateConfig{
			Engine: "fak-native", Command: []string{"fak", "serve", "--port", "18090"}, Environment: map[string]string{"FAK_MODEL": "test"},
			ListenerPort: 18090, ReadinessEndpoints: []string{"http://127.0.0.1:18090/health"}, ReadinessTimeout: "1s",
		},
		Request: modelCanaryRequestConfig{Command: []string{"request-fixture", "--once"}, Deadline: "2s"},
		Watcher: modelCanaryWatcherConfig{
			Interval: "1s", ConsecutiveCrossings: 3, MaximumRSSBytes: 1 << 30, MaximumFootprintBytes: 2 << 30,
			MaximumSwapGrowthBytes: 1 << 30, MinimumSystemFreePercent: 10, MinimumMemorystatusPercent: 10,
		},
		Cleanup: modelCanaryCleanupConfig{CandidateTERMTimeout: "1s", RestoreTimeout: "1s", StabilityDuration: "1s", ProbeInterval: "1s"},
	}
}

type modelCanaryTestLease struct {
	release func() error
}

func (l modelCanaryTestLease) Release() error { return l.release() }

type modelCanaryTestHarness struct {
	failAt             string
	actions            []string
	now                time.Time
	polls              int
	requestDoneAfter   int
	sampleRSSKiB       int64
	sentinelAlive      bool
	cancelOnSleep      func()
	cleanupSawCanceled bool
	incumbent          modelCanaryProcessIdentity
	candidate          modelCanaryProcessIdentity
	request            modelCanaryProcessIdentity
}

func newModelCanaryTestHarness(failAt string) *modelCanaryTestHarness {
	return &modelCanaryTestHarness{
		failAt: failAt, now: time.Date(2026, 8, 26, 16, 0, 0, 0, time.UTC), requestDoneAfter: 2, sampleRSSKiB: 1, sentinelAlive: true,
		incumbent: modelCanaryProcessIdentity{PID: 41001, StartedAt: "2026-08-26T15:00:00Z", ArgvSHA256: strings.Repeat("a", 64)},
		candidate: modelCanaryProcessIdentity{PID: 41002, StartedAt: "2026-08-26T16:00:00Z", ArgvSHA256: strings.Repeat("c", 64)},
		request:   modelCanaryProcessIdentity{PID: 41003, StartedAt: "2026-08-26T16:00:01Z", ArgvSHA256: strings.Repeat("d", 64)},
	}
}

func (h *modelCanaryTestHarness) record(action string) { h.actions = append(h.actions, action) }

func (h *modelCanaryTestHarness) failure(at string) error {
	if h.failAt == at {
		return errors.New("injected " + at + " failure")
	}
	return nil
}

func (h *modelCanaryTestHarness) dependencies() modelCanaryRunDeps {
	return modelCanaryRunDeps{
		Platform: "darwin", Architecture: "arm64", Now: func() time.Time { return h.now },
		Preflight: func(context.Context, modelCanaryRunConfig) (modelCanaryPreflight, error) {
			h.record("preflight")
			if err := h.failure("preflight"); err != nil {
				return modelCanaryPreflight{}, err
			}
			tools := make(map[string]string)
			executableSHA := make(map[string]string)
			toolRoot := filepath.Join(filepath.VolumeName(os.TempDir())+string(filepath.Separator), "test", "bin")
			for _, name := range []string{"lsof", "ps", "footprint", "sysctl", "memory_pressure", "launchctl", "candidate", "request", "restore"} {
				tools[name] = filepath.Join(toolRoot, name)
				executableSHA[name] = strings.Repeat("e", 64)
			}
			return modelCanaryPreflight{
				Incumbent: h.incumbent, BaselineSwapBytes: 1024, RestorePlistSHA256: strings.Repeat("b", 64),
				Tools: tools, ExecutableSHA256: executableSHA,
			}, nil
		},
		AcquireLease: func(context.Context, modelCanaryLeaseConfig, time.Duration) (modelCanaryLease, error) {
			h.record("acquire_gpu_lease")
			if err := h.failure("acquire"); err != nil {
				return nil, err
			}
			return modelCanaryTestLease{release: func() error {
				h.record("release_gpu_lease")
				return h.failure("release")
			}}, nil
		},
		VerifyIncumbent: func(context.Context, modelCanaryRunConfig, modelCanaryProcessIdentity) (modelCanaryProcessIdentity, error) {
			h.record("verify_incumbent")
			if err := h.failure("verify"); err != nil {
				return modelCanaryProcessIdentity{}, err
			}
			return h.incumbent, nil
		},
		BootoutIncumbent: func(context.Context, modelCanaryRunConfig, modelCanaryProcessIdentity, time.Duration) error {
			h.record("bootout_incumbent")
			return h.failure("bootout")
		},
		StartCandidate: func(context.Context, modelCanaryCandidateConfig) (modelCanaryProcess, error) {
			h.record("start_candidate")
			if err := h.failure("start_candidate"); err != nil {
				return modelCanaryProcess{}, err
			}
			return modelCanaryProcess{Identity: h.candidate, Handle: "candidate"}, nil
		},
		WaitCandidateReady: func(context.Context, modelCanaryRunConfig, modelCanaryProcess, time.Duration) error {
			h.record("candidate_ready")
			return h.failure("candidate_ready")
		},
		StartRequest: func(context.Context, modelCanaryRequestConfig) (modelCanaryProcess, error) {
			h.record("start_request")
			if err := h.failure("start_request"); err != nil {
				return modelCanaryProcess{}, err
			}
			return modelCanaryProcess{Identity: h.request, Handle: "request"}, nil
		},
		PollRequest: func(modelCanaryProcess) (bool, int, error) {
			h.record("poll_request")
			h.polls++
			return h.requestDoneAfter >= 0 && h.polls >= h.requestDoneAfter, 0, nil
		},
		RequestEvidence: func(modelCanaryProcess) (modelCanaryRequestEvidence, error) {
			body := "fixture response\n"
			return modelCanaryRequestEvidence{
				CompletedAt: h.now.UTC().Format(time.RFC3339Nano),
				Stdout:      body, StdoutBytes: len(body), StdoutSHA256: digestBytes([]byte(body)),
				Stderr: "", StderrBytes: 0, StderrSHA256: digestBytes(nil),
			}, nil
		},
		StopRequest: func(ctx context.Context, _ modelCanaryProcess, _ time.Duration) error {
			if ctx.Err() != nil {
				h.cleanupSawCanceled = true
			}
			h.record("stop_request")
			return h.failure("stop_request")
		},
		Sample: func(context.Context, modelCanaryProcess, int64) (modelCanarySample, error) {
			h.record("sample")
			if err := h.failure("sample"); err != nil {
				return modelCanarySample{}, err
			}
			raw := map[string]string{
				"ps":              "41002 " + strconv.FormatInt(h.sampleRSSKiB, 10) + "\n",
				"footprint":       "phys_footprint: 2 KB\n",
				"swap":            "vm.swapusage: total = 2.00K used = 1.00K free = 1.00K\n",
				"memory_pressure": "System-wide memory free percentage: 50%\n",
				"memorystatus":    "50\n",
			}
			digests := make(map[string]string, len(raw))
			for source, text := range raw {
				digests[source] = digestBytes([]byte(text))
			}
			if h.failAt == "raw_hash_tamper" {
				digests["ps"] = strings.Repeat("0", 64)
			}
			rssBytes := h.sampleRSSKiB * 1024
			if h.failAt == "raw_value_tamper" {
				rssBytes++
			}
			return modelCanarySample{Candidate: h.candidate, RSSBytes: rssBytes, FootprintBytes: 2048, SwapUsedBytes: 1024, SwapGrowthBytes: 0, SystemFreePercent: 50, MemorystatusPercent: 50, Raw: raw, RawSHA256: digests}, nil
		},
		Sleep: func(ctx context.Context, duration time.Duration) error {
			h.record("sleep")
			h.now = h.now.Add(duration)
			if h.cancelOnSleep != nil {
				h.cancelOnSleep()
			}
			return ctx.Err()
		},
		TermCandidate: func(ctx context.Context, _ modelCanaryProcess, _ time.Duration) error {
			if ctx.Err() != nil {
				h.cleanupSawCanceled = true
			}
			if h.failAt == "reused_pid" {
				h.record("identity_refused")
				return &modelCanaryRefusal{Reason: modelCanaryReasonCandidateIdentityMismatch, Phase: modelCanaryPhaseCandidateTerminated, Detail: "PID start identity changed"}
			}
			h.record("signal:TERM")
			h.sentinelAlive = false
			return h.failure("term_candidate")
		},
		RestoreIncumbent: func(ctx context.Context, _ modelCanaryRunConfig, _ time.Duration) error {
			if ctx.Err() != nil {
				h.cleanupSawCanceled = true
			}
			h.record("restore_incumbent")
			return h.failure("restore")
		},
		EndpointsStable: func(ctx context.Context, _ modelCanaryRunConfig, _, _ time.Duration) (modelCanaryProcessIdentity, error) {
			if ctx.Err() != nil {
				h.cleanupSawCanceled = true
			}
			h.record("endpoints_stable")
			if err := h.failure("stable"); err != nil {
				return modelCanaryProcessIdentity{}, err
			}
			return modelCanaryProcessIdentity{PID: 51001, StartedAt: "2026-08-26T16:01:00Z", ArgvSHA256: h.incumbent.ArgvSHA256}, nil
		},
	}
}

func assertModelCanaryEventSequence(t *testing.T, events []modelCanaryEvent) {
	t.Helper()
	for i, event := range events {
		if event.Sequence != i+1 {
			t.Fatalf("event[%d].sequence=%d want %d", i, event.Sequence, i+1)
		}
	}
}

func assertModelCanaryActionBefore(t *testing.T, actions []string, first, second string) {
	t.Helper()
	firstIndex, secondIndex := -1, -1
	for i, action := range actions {
		if action == first && firstIndex < 0 {
			firstIndex = i
		}
		if action == second && secondIndex < 0 {
			secondIndex = i
		}
	}
	if firstIndex < 0 || secondIndex < 0 || firstIndex >= secondIndex {
		t.Fatalf("want %q before %q, actions=%v", first, second, actions)
	}
}

func containsModelCanaryAction(actions []string, want string) bool {
	for _, action := range actions {
		if action == want {
			return true
		}
	}
	return false
}

func TestModelCanaryNoIncumbentConfig(t *testing.T) {
	cfg := validModelCanaryTestConfig()
	cfg.NoIncumbent = true
	if _, err := validateModelCanaryConfig(cfg); err == nil {
		t.Fatal("accepted ambiguous incumbent configuration")
	}
	cfg.Incumbent = modelCanaryIncumbentConfig{}
	if _, err := validateModelCanaryConfig(cfg); err != nil {
		t.Fatal(err)
	}
	cfg.NoIncumbent = false
	if _, err := validateModelCanaryConfig(cfg); err == nil {
		t.Fatal("default no longer requires exact handoff")
	}
}

// The test binary is a bounded, model-free child; the live runner still owns identity,
// listener readiness, lease, pressure sampling, request capture and TERM-only cleanup.
func TestModelCanaryBoundedChild(t *testing.T) {
	switch os.Getenv("FAK_CANARY_TEST_CHILD") {
	case "candidate":
		listener, err := net.Listen("tcp", "127.0.0.1:"+os.Getenv("FAK_CANARY_TEST_PORT"))
		if err != nil {
			os.Exit(2)
		}
		server := &http.Server{ReadHeaderTimeout: time.Second, Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { fmt.Fprint(w, "bounded-child") })}
		go server.Serve(listener)
		time.Sleep(20 * time.Second)
		server.Close()
		os.Exit(0)
	case "request":
		time.Sleep(300 * time.Millisecond)
		client := &http.Client{Timeout: time.Second}
		response, err := client.Get("http://127.0.0.1:" + os.Getenv("FAK_CANARY_TEST_PORT") + "/health")
		if err != nil {
			os.Exit(3)
		}
		response.Body.Close()
		fmt.Println("bounded-request-complete")
		os.Exit(0)
	}
}

func TestModelCanaryNoIncumbentLiveLifecycle(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("live safety runtime requires Darwin arm64")
	}
	deps, err := modelCanaryLiveDependencies()
	if err != nil {
		t.Fatal(err)
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	port := listener.Addr().(*net.TCPAddr).Port
	cfg := validModelCanaryTestConfig()
	cfg.NoIncumbent = true
	cfg.Incumbent = modelCanaryIncumbentConfig{}
	cfg.Lease.Path = filepath.Join(t.TempDir(), "gpu.lease")
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cfg.Candidate.Command = []string{executable, "-test.run=^TestModelCanaryBoundedChild$"}
	cfg.Candidate.Environment = map[string]string{"FAK_CANARY_TEST_CHILD": "candidate", "FAK_CANARY_TEST_PORT": strconv.Itoa(port)}
	cfg.Candidate.ListenerPort = port
	cfg.Candidate.ReadinessEndpoints = []string{fmt.Sprintf("http://127.0.0.1:%d/health", port)}
	cfg.Candidate.ReadinessTimeout = "5s"
	cfg.Request.Command = cfg.Candidate.Command
	cfg.Request.Environment = map[string]string{"FAK_CANARY_TEST_CHILD": "request", "FAK_CANARY_TEST_PORT": strconv.Itoa(port)}
	cfg.Request.Deadline = "10s"
	cfg.Watcher.Interval = "100ms"
	cfg.Watcher.MaximumRSSBytes, cfg.Watcher.MaximumFootprintBytes, cfg.Watcher.MaximumSwapGrowthBytes = 1<<40, 1<<40, 1<<40
	cfg.Watcher.MinimumSystemFreePercent, cfg.Watcher.MinimumMemorystatusPercent = 1, 1
	durations, err := validateModelCanaryConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	occupied := executeModelCanaryRun(ctx, cfg, durations, "fixture", deps)
	if !occupied.NoIncumbent || occupied.Outcome == "complete" || occupied.Candidate != nil || occupied.RestorationAttempted {
		t.Fatalf("occupied listener admitted: %+v", occupied)
	}
	// The incumbent owner is this real test process; it must still accept connections.
	conn, err := net.DialTimeout("tcp", listener.Addr().String(), time.Second)
	if err != nil {
		t.Fatalf("occupied listener owner was disturbed: %v", err)
	}
	conn.Close()
	listener.Close()
	receipt := executeModelCanaryRun(ctx, cfg, durations, "fixture", deps)
	if receipt.Outcome != "complete" {
		t.Fatalf("live lifecycle: %s: %s; events=%+v", receipt.Reason, receipt.Detail, receipt.Events)
	}
	if !receipt.NoIncumbent || !receipt.LeaseReleased || receipt.RestorationAttempted || receipt.RestoredIncumbent != nil || receipt.Candidate == nil || receipt.RequestEvidence == nil || !strings.Contains(receipt.RequestEvidence.Stdout, "bounded-request-complete") {
		t.Fatalf("incomplete live receipt: %+v", receipt)
	}
	if err := deps.VerifyUnusedListener(ctx, port); err != nil {
		t.Fatalf("child listener not cleaned: %v", err)
	}
	// A second cleanup must see the original child as gone, never signal a reused PID.
	if err := deps.TermCandidate(ctx, modelCanaryProcess{Identity: *receipt.Candidate}, time.Second); err == nil {
		t.Fatal("cleanup accepted an unowned process handle")
	}
	assertModelCanaryEventSequence(t, receipt.Events)
	for _, event := range receipt.Events {
		if event.Action == "bootout_incumbent" || event.Action == "restore_incumbent" {
			t.Fatal("no-incumbent run mutated a service")
		}
	}
}
