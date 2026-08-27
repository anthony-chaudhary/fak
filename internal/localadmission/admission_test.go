package localadmission

import (
	"testing"
	"time"
)

func fixture() (Hardware, TaskEnvelope, Limits, Signals) {
	return Hardware{Chip: "M3", MemoryBytes: 16 << 30, OSVersion: 15}, TaskEnvelope{Task: "summarize", MinOS: 14, Chips: []string{"M2", "M3"}, PeakMemoryBytes: 4 << 30, DiskBytes: 6 << 30, Quality: .92, MinQuality: .9, TTFT: 300 * time.Millisecond, MaxTTFT: time.Second, DecodeTPS: 30, MinDecodeTPS: 20}, Limits{MemoryHighWater: .8, MaxQueue: 3, MaxConcurrent: 1, MaxForegroundLatency: 100 * time.Millisecond, AllowLowPower: false, MaxCrashes: 3, CrashWindow: time.Minute}, Signals{DiskFreeBytes: 20 << 30}
}
func TestMeasuredMatrixAndTransitions(t *testing.T) {
	h, task, limits, s := fixture()
	g := New()
	if got := g.Admit(h, task, limits, s); got.Readiness != Ready {
		t.Fatal(got)
	}
	tests := []struct {
		name   string
		mut    func(*Hardware, *TaskEnvelope, *Limits, *Signals)
		want   Readiness
		reason string
	}{{"chip", func(h *Hardware, _ *TaskEnvelope, _ *Limits, _ *Signals) { h.Chip = "Intel" }, Unsupported, "chip"}, {"quality", func(_ *Hardware, t *TaskEnvelope, _ *Limits, _ *Signals) { t.Quality = .5 }, Unsupported, "fixture_quality"}, {"memory pressure", func(_ *Hardware, _ *TaskEnvelope, _ *Limits, s *Signals) { s.MemoryUsedBytes = 13 << 30 }, TemporarilyUnavailable, "memory_high_water"}, {"disk", func(_ *Hardware, _ *TaskEnvelope, _ *Limits, s *Signals) { s.DiskFreeBytes = 1 }, TemporarilyUnavailable, "disk_reservation"}, {"queue", func(_ *Hardware, _ *TaskEnvelope, l *Limits, s *Signals) { s.Queue = l.MaxQueue }, TemporarilyUnavailable, "queue_full"}, {"foreground", func(_ *Hardware, _ *TaskEnvelope, _ *Limits, s *Signals) { s.ForegroundLatency = time.Second }, TemporarilyUnavailable, "foreground_latency"}, {"battery", func(_ *Hardware, _ *TaskEnvelope, l *Limits, s *Signals) { l.RequireAC = true; s.OnBattery = true }, TemporarilyUnavailable, "ac_required"}, {"low power", func(_ *Hardware, _ *TaskEnvelope, _ *Limits, s *Signals) { s.LowPower = true }, TemporarilyUnavailable, "low_power_mode"}, {"thermal", func(_ *Hardware, _ *TaskEnvelope, _ *Limits, s *Signals) { s.Thermal = "serious" }, ReadyDegraded, "resource_downshift"}, {"concurrency", func(_ *Hardware, _ *TaskEnvelope, _ *Limits, s *Signals) { s.Concurrent = 1 }, ReadyDegraded, "queued"}, {"cancel", func(_ *Hardware, _ *TaskEnvelope, _ *Limits, s *Signals) { s.Cancelled = true }, TemporarilyUnavailable, "cancelled"}}
	for _, tc := range tests {
		hh, tt, ll, ss := fixture()
		tc.mut(&hh, &tt, &ll, &ss)
		got := g.Admit(hh, tt, ll, ss)
		if got.Readiness != tc.want || got.Reason != tc.reason {
			t.Errorf("%s: %+v", tc.name, got)
		}
	}
}
func TestCrashLoopCircuitBreakerRecovers(t *testing.T) {
	h, task, l, s := fixture()
	g := New()
	now := time.Unix(1000, 0)
	g.now = func() time.Time { return now }
	for range 3 {
		g.RecordCrash(task.Task)
	}
	if got := g.Admit(h, task, l, s); got.Reason != "crash_loop" {
		t.Fatal(got)
	}
	now = now.Add(2 * time.Minute)
	if got := g.Admit(h, task, l, s); got.Readiness != Ready {
		t.Fatal(got)
	}
}
