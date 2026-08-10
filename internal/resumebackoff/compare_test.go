package resumebackoff

import (
	"testing"
	"time"
)

func TestCompareLocalKeepsSchedulerAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := map[string]struct {
		kind      string
		available bool
	}{
		"fak native resume backoff":   {"native", true},
		"immediate resume":            {"baseline", true},
		"Kubernetes CrashLoopBackOff": {"external", false},
		"systemd RestartSec":          {"external", false},
		"AWS Step Functions retry":    {"external", false},
	}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d want %d: %#v", len(got.Arms), len(want), got.Arms)
	}
	for _, arm := range got.Arms {
		expected, ok := want[arm.Name]
		if !ok {
			t.Fatalf("unexpected arm %q", arm.Name)
		}
		if arm.Kind != expected.kind || arm.Available != expected.available {
			t.Errorf("arm %q=%q available=%v want %q/%v", arm.Name, arm.Kind, arm.Available, expected.kind, expected.available)
		}
		if !arm.Available && (arm.Correct || arm.Latency != 0 || arm.Delay != 0 || arm.Bytes != 0 || arm.CostUSD != 0) {
			t.Errorf("unavailable arm %q claims a result: %#v", arm.Name, arm)
		}
	}
	if !got.Arms[0].Correct || got.Arms[0].Delay != 2*60*1e9 {
		t.Fatalf("native=%#v", got.Arms[0])
	}
}

func BenchmarkDecideResumeBackoff(b *testing.B) {
	now := time.Unix(10_000, 0)
	in := Input{Session: "session", Signature: "same-failure", Now: now, History: []Event{{Session: "session", Signature: "same-failure", At: now.Add(-90 * time.Second)}, {Session: "session", Signature: "same-failure", At: now.Add(-30 * time.Second)}}, Base: time.Minute, Ceiling: time.Hour}
	var got Decision
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		got = Decide(in)
	}
	if got.Reason != ReasonBackoff {
		b.Fatal(got)
	}
}
