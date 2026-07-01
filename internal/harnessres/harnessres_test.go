package harnessres

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestCPUPercentAvg(t *testing.T) {
	h := Half{CPUUser: 3 * time.Second, CPUSys: time.Second, HaveCPU: true}
	pct, ok := h.CPUPercentAvg(2 * time.Second)
	if !ok {
		t.Fatal("want ok")
	}
	if pct != 200 { // 4 CPU-seconds over 2 wall-seconds == 200% (two busy cores)
		t.Fatalf("pct = %v, want 200", pct)
	}
	if _, ok := (Half{}).CPUPercentAvg(time.Second); ok {
		t.Fatal("no CPU observed: want ok=false")
	}
	if _, ok := h.CPUPercentAvg(0); ok {
		t.Fatal("zero elapsed: want ok=false")
	}
}

func TestFoldProcTracksPeaks(t *testing.T) {
	clock := time.Unix(1000, 0)
	now := func() time.Time { return clock }
	s := newSampler(now)

	// t=0: 0 CPU, 100 MiB RSS
	s.foldProc(procSample{haveCPU: true, cpuUser: 0, haveRSS: true, rss: 100 << 20}, clock, 10, 1<<20)
	// t=1s: +1.5 CPU-seconds (150% busy), RSS drops to 80 MiB
	clock = clock.Add(time.Second)
	s.foldProc(procSample{haveCPU: true, cpuUser: 1500 * time.Millisecond, haveRSS: true, rss: 80 << 20}, clock, 42, 5<<20)
	// t=2s: +0.5 CPU-seconds (50% busy), RSS climbs to 120 MiB (new peak)
	clock = clock.Add(time.Second)
	s.foldProc(procSample{haveCPU: true, cpuUser: 2 * time.Second, haveRSS: true, rss: 120 << 20}, clock, 30, 3<<20)

	snap := s.Snapshot()
	if snap.Samples != 3 {
		t.Fatalf("samples = %d, want 3", snap.Samples)
	}
	if snap.Kernel.CPUUser != 2*time.Second {
		t.Fatalf("cpuUser = %v, want 2s (latest cumulative)", snap.Kernel.CPUUser)
	}
	if snap.Kernel.RSSBytes != 120<<20 {
		t.Fatalf("rss = %d, want 120MiB (latest)", snap.Kernel.RSSBytes)
	}
	if snap.Kernel.PeakRSSBytes != 120<<20 {
		t.Fatalf("peak rss = %d, want 120MiB", snap.Kernel.PeakRSSBytes)
	}
	if !snap.HaveKernelCPUPeak || snap.KernelCPUPercentPeak != 150 {
		t.Fatalf("peak cpu%% = %v (have=%v), want 150", snap.KernelCPUPercentPeak, snap.HaveKernelCPUPeak)
	}
	if snap.GoroutinesPeak != 42 {
		t.Fatalf("goroutines peak = %d, want 42", snap.GoroutinesPeak)
	}
	if snap.GoHeapSysBytes != 5<<20 {
		t.Fatalf("go heap peak = %d, want 5MiB", snap.GoHeapSysBytes)
	}
	if snap.Elapsed != 2*time.Second {
		t.Fatalf("elapsed = %v, want 2s", snap.Elapsed)
	}
}

func TestFoldProcPeakRSSFromReportedPeak(t *testing.T) {
	clock := time.Unix(0, 0)
	s := newSampler(func() time.Time { return clock })
	// A platform that reports only a peak (e.g. darwin Maxrss) with no current RSS.
	s.foldProc(procSample{havePeakRSS: true, peakRSS: 512 << 20}, clock, 1, 0)
	snap := s.Snapshot()
	if snap.Kernel.HaveRSS {
		t.Fatal("no current RSS reported: HaveRSS should be false")
	}
	if !snap.Kernel.HavePeakRSS || snap.Kernel.PeakRSSBytes != 512<<20 {
		t.Fatalf("peak rss = %d (have=%v), want 512MiB", snap.Kernel.PeakRSSBytes, snap.Kernel.HavePeakRSS)
	}
}

func TestReportPresenceBits(t *testing.T) {
	// A snapshot with a fully-populated kernel half but an empty agent half: the agent
	// side must render n/a, never a fake 0.
	snap := Snapshot{
		Elapsed: 64 * time.Second,
		Samples: 32,
		Kernel: Half{
			CPUUser: 10 * time.Second, CPUSys: 2 * time.Second, HaveCPU: true,
			RSSBytes: 142 << 20, HaveRSS: true, PeakRSSBytes: 210 << 20, HavePeakRSS: true,
			IOReadBytes: 3 << 20, IOWriteBytes: 1 << 20, HaveIO: true,
		},
		KernelCPUPercentPeak: 74, HaveKernelCPUPeak: true,
		GoroutinesPeak: 41, GoHeapSysBytes: 96 << 20, NumCPU: 16, GOMAXPROCS: 16,
	}
	out := snap.Report()
	for _, want := range []string{
		"harness resources —",
		"kernel(guard+gateway) cpu 12.0s (19% avg, 74% peak)",
		"rss 142.0 MiB (peak 210.0 MiB)",
		"io r/w 3.0 MiB/1.0 MiB",
		"agent(child) cpu n/a, rss n/a",
		"16 cores",
		"sampled 32x over 1m4s",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("Report() missing %q\ngot: %s", want, out)
		}
	}
}

func TestPrometheusTextOmitsAbsentAxes(t *testing.T) {
	snap := Snapshot{
		Kernel: Half{CPUUser: time.Second, CPUSys: time.Second, HaveCPU: true, RSSBytes: 1 << 20, HaveRSS: true},
		NumCPU: 8,
	}
	out := snap.PrometheusText()
	if !strings.Contains(out, `fak_harness_cpu_seconds_total{half="kernel",mode="user"} 1`) {
		t.Errorf("missing kernel user cpu line\n%s", out)
	}
	if !strings.Contains(out, `fak_harness_rss_bytes{half="kernel"}`) {
		t.Errorf("missing kernel rss line\n%s", out)
	}
	if strings.Contains(out, `fak_harness_rss_bytes{half="agent"}`) {
		t.Errorf("agent rss absent but emitted\n%s", out)
	}
	if strings.Contains(out, "fak_harness_io_bytes_total{half=") {
		t.Errorf("io absent but emitted\n%s", out)
	}
	if !strings.Contains(out, "# TYPE fak_harness_cpu_seconds_total gauge") {
		t.Errorf("missing HELP/TYPE header\n%s", out)
	}
}

func TestMarshalLedgerRowOmitsAbsentAxes(t *testing.T) {
	snap := Snapshot{
		Elapsed: 5 * time.Second, Samples: 3,
		Kernel:         Half{CPUUser: time.Second, HaveCPU: true, PeakRSSBytes: 200 << 20, HavePeakRSS: true},
		Agent:          Half{}, // nothing observed
		NumCPU:         4,
		GoroutinesPeak: 7,
	}
	b, err := snap.MarshalLedgerRow("guard", "anthropic", "claude", time.Unix(1700000000, 0))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("row is not valid JSON: %v\n%s", err, b)
	}
	if m["schema"] != LedgerSchema {
		t.Errorf("schema = %v, want %s", m["schema"], LedgerSchema)
	}
	kernel := m["kernel"].(map[string]any)
	if _, ok := kernel["peak_rss_bytes"]; !ok {
		t.Errorf("kernel.peak_rss_bytes missing: %s", b)
	}
	if _, ok := kernel["io_read_bytes"]; ok {
		t.Errorf("kernel.io_read_bytes present but I/O was absent: %s", b)
	}
	agent := m["agent"].(map[string]any)
	if len(agent) != 0 {
		t.Errorf("agent half should be empty (no axes observed), got %v", agent)
	}
}

func TestHumanBytes(t *testing.T) {
	cases := map[uint64]string{
		512:     "512 B",
		1 << 10: "1.0 KiB",
		1536:    "1.5 KiB",
		1 << 20: "1.0 MiB",
		3 << 30: "3.0 GiB",
	}
	for in, want := range cases {
		if got := humanBytes(in); got != want {
			t.Errorf("humanBytes(%d) = %q, want %q", in, got, want)
		}
	}
}

func TestStopIsIdempotent(t *testing.T) {
	s := New()
	s.Start(time.Hour) // long interval: only the immediate + final samples fire
	first := s.Stop()
	second := s.Stop() // must not panic (double close guarded by sync.Once)
	if second.Samples < first.Samples {
		t.Fatalf("second Stop lost samples: %d < %d", second.Samples, first.Samples)
	}
	if first.NumCPU == 0 {
		t.Fatal("NumCPU should be populated from runtime")
	}
}
