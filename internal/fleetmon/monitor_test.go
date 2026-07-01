package fleetmon

import (
	"testing"
	"time"
)

func TestClassifyDeadOnlyWhenPIDGone(t *testing.T) {
	now := time.Now()
	// Registry-active (LIVE) but the PID is not alive => dead.
	got := Classify(WorkerEvidence{
		Session: "w", RegistryDisp: "LIVE", HasPID: true, PID: 4242, PIDAlive: false,
		Transcript: TranscriptSignal{Exists: true, HasTimestamp: true, LastTimestamp: now.Add(-2 * time.Minute)},
	}, now, DefaultThresholds())
	if got.Class != ClassDead {
		t.Fatalf("registry-active worker with a dead PID must be dead, got %s", got.Class)
	}
}

func TestClassifyNeverDeadWhenPIDAlive(t *testing.T) {
	now := time.Now()
	// PID alive, transcript very stale, no final report — stale, NOT dead.
	got := Classify(WorkerEvidence{
		Session: "w", RegistryDisp: "LIVE", HasPID: true, PID: 10, PIDAlive: true,
		Transcript: TranscriptSignal{Exists: true, HasTimestamp: true, LastTimestamp: now.Add(-40 * time.Minute)},
	}, now, DefaultThresholds())
	if got.Class == ClassDead {
		t.Fatalf("a worker whose PID is alive must never be classified dead, got %s", got.Class)
	}
	if got.Class != ClassStaleTranscript {
		t.Fatalf("want stale-transcript, got %s", got.Class)
	}
}

func TestClassifyBlocked(t *testing.T) {
	now := time.Now()
	got := Classify(WorkerEvidence{
		Session: "w", HasPID: true, PID: 10, PIDAlive: true,
		Transcript: TranscriptSignal{Exists: true, HasTimestamp: true, LastTimestamp: now.Add(-1 * time.Minute), Blocker: "rate/usage limit hit", BlockerKind: "rate"},
	}, now, DefaultThresholds())
	if got.Class != ClassAuthRateBlocked {
		t.Fatalf("a current transcript blocker must classify auth-or-rate-blocked, got %s", got.Class)
	}
	if got.Blocker == "" {
		t.Fatal("blocker reason should be set")
	}
}

func TestClassifyBlockedFromRegistryAuth(t *testing.T) {
	now := time.Now()
	got := Classify(WorkerEvidence{
		Session: "w", RegistryDisp: "INFRA_AUTH", RegistryAction: "BLOCKED_AUTH", HasPID: true, PID: 10, PIDAlive: true,
		Transcript: TranscriptSignal{Exists: true, HasTimestamp: true, LastTimestamp: now.Add(-1 * time.Minute)},
	}, now, DefaultThresholds())
	if got.Class != ClassAuthRateBlocked {
		t.Fatalf("registry auth block must classify auth-or-rate-blocked, got %s", got.Class)
	}
}

func TestClassifyCompletedFinalIdle(t *testing.T) {
	now := time.Now()
	lines := iptr(100)
	got := Classify(WorkerEvidence{
		Session: "w", HasPID: true, PID: 10, PIDAlive: true, PrevLines: lines,
		Transcript: TranscriptSignal{Exists: true, Lines: 100, HasTimestamp: true, LastTimestamp: now.Add(-2 * time.Minute), FinalReport: true, FinalReportText: "done"},
	}, now, DefaultThresholds())
	if got.Class != ClassCompletedFinal {
		t.Fatalf("idle worker with a final report must be completed-final, got %s", got.Class)
	}
}

func TestClassifyFinalReportButStillAdvancingIsHealthy(t *testing.T) {
	now := time.Now()
	prev := iptr(90)
	// A final report was seen, but the transcript is still growing (+10 lines) and
	// CPU is burning: the worker resumed — treat as healthy, not completed.
	got := Classify(WorkerEvidence{
		Session: "w", HasPID: true, PID: 10, PIDAlive: true, PrevLines: prev, CPUDeltaSec: fptr(5),
		Transcript: TranscriptSignal{Exists: true, Lines: 100, HasTimestamp: true, LastTimestamp: now.Add(-30 * time.Second), FinalReport: true},
	}, now, DefaultThresholds())
	if got.Class != ClassHealthy {
		t.Fatalf("a final report that is still advancing must be healthy, got %s", got.Class)
	}
}

func TestClassifyStaleChildBeatsStaleTranscript(t *testing.T) {
	now := time.Now()
	got := Classify(WorkerEvidence{
		Session: "w", HasPID: true, PID: 10, PIDAlive: true,
		Transcript:    TranscriptSignal{Exists: true, HasTimestamp: true, LastTimestamp: now.Add(-40 * time.Minute)},
		StaleChildren: []ChildCommand{{RootPID: 55, Name: "go", Class: CmdTest, AgeSec: 900}},
	}, now, DefaultThresholds())
	if got.Class != ClassStaleChild {
		t.Fatalf("a wedged stale child is the more specific/actionable class, got %s", got.Class)
	}
	if got.ChildSummary == "" {
		t.Fatal("child summary should be populated")
	}
}

func TestClassifyHealthyAdvancing(t *testing.T) {
	now := time.Now()
	prev := iptr(40)
	got := Classify(WorkerEvidence{
		Session: "w", HasPID: true, PID: 10, PIDAlive: true, PrevLines: prev,
		Transcript: TranscriptSignal{Exists: true, Lines: 55, HasTimestamp: true, LastTimestamp: now.Add(-10 * time.Second)},
	}, now, DefaultThresholds())
	if got.Class != ClassHealthy {
		t.Fatalf("advancing transcript must be healthy, got %s", got.Class)
	}
	if got.LineDelta == nil || *got.LineDelta != 15 {
		t.Fatalf("line delta should be +15, got %v", got.LineDelta)
	}
}

func TestClassifyAttentionOnThinEvidence(t *testing.T) {
	now := time.Now()
	got := Classify(WorkerEvidence{Session: "w"}, now, DefaultThresholds())
	if got.Class != ClassAttention {
		t.Fatalf("no PID + no transcript => attention, got %s", got.Class)
	}
}

// TestWitnessThirtyWorkerRun reproduces the issue's witness fixture: 30 workers,
// all registry-busy, all PIDs alive, only a subset advancing transcripts, one
// idle final report, several stale child commands. The monitor must classify
// them WITHOUT a single false dead-worker alert.
func TestWitnessThirtyWorkerRun(t *testing.T) {
	now := time.Now()
	th := DefaultThresholds()
	var samples []WorkerSample

	// 20 healthy: advancing transcripts (fresh + growing line count).
	for i := 0; i < 20; i++ {
		prev := iptr(10)
		samples = append(samples, Classify(WorkerEvidence{
			Issue: 1000 + i, Session: sess(i), RegistryDisp: "LIVE", HasPID: true, PID: 1000 + i, PIDAlive: true, PrevLines: prev,
			Transcript: TranscriptSignal{Exists: true, Lines: 25, HasTimestamp: true, LastTimestamp: now.Add(-15 * time.Second)},
		}, now, th))
	}
	// 5 stale-transcript: alive, idle 30m, no final report, no stale child.
	for i := 20; i < 25; i++ {
		prev := iptr(30)
		samples = append(samples, Classify(WorkerEvidence{
			Issue: 1000 + i, Session: sess(i), RegistryDisp: "LIVE", HasPID: true, PID: 1000 + i, PIDAlive: true, PrevLines: prev,
			Transcript: TranscriptSignal{Exists: true, Lines: 30, HasTimestamp: true, LastTimestamp: now.Add(-30 * time.Minute)},
		}, now, th))
	}
	// 4 stale-child-command: alive, idle, wedged on a stale simple child.
	for i := 25; i < 29; i++ {
		samples = append(samples, Classify(WorkerEvidence{
			Issue: 1000 + i, Session: sess(i), RegistryDisp: "LIVE", HasPID: true, PID: 1000 + i, PIDAlive: true,
			Transcript:    TranscriptSignal{Exists: true, HasTimestamp: true, LastTimestamp: now.Add(-8 * time.Minute)},
			StaleChildren: []ChildCommand{{RootPID: 5000 + i, Name: "ls", Class: CmdSimpleShell, AgeSec: 400}},
		}, now, th))
	}
	// 1 completed-final: idle with a final report.
	samples = append(samples, Classify(WorkerEvidence{
		Issue: 1029, Session: sess(29), RegistryDisp: "LIVE", HasPID: true, PID: 1029, PIDAlive: true,
		Transcript: TranscriptSignal{Exists: true, HasTimestamp: true, LastTimestamp: now.Add(-3 * time.Minute), FinalReport: true, FinalReportText: "done"},
	}, now, th))

	counts := map[Classification]int{}
	for _, s := range samples {
		counts[s.Class]++
	}
	if counts[ClassDead] != 0 {
		t.Fatalf("FALSE DEAD ALERT: %d workers classified dead though all PIDs are alive", counts[ClassDead])
	}
	if counts[ClassHealthy] != 20 {
		t.Errorf("want 20 healthy, got %d", counts[ClassHealthy])
	}
	if counts[ClassStaleTranscript] != 5 {
		t.Errorf("want 5 stale-transcript, got %d", counts[ClassStaleTranscript])
	}
	if counts[ClassStaleChild] != 4 {
		t.Errorf("want 4 stale-child-command, got %d", counts[ClassStaleChild])
	}
	if counts[ClassCompletedFinal] != 1 {
		t.Errorf("want 1 completed-final, got %d", counts[ClassCompletedFinal])
	}

	// The payload histogram should agree and total 30.
	payload := NewMonitorPayload("run-x", samples, now)
	if payload.Total != 30 {
		t.Fatalf("want 30 workers total, got %d", payload.Total)
	}
}

func sess(i int) string { return "issue-" + itoa(1000+i) }
