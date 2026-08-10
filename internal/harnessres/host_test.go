package harnessres

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// hostFixture is the #2053 shape: a harness slice sampled against a known box, so the
// expected fractions are hand-checkable. 2 GiB of 16 GiB RAM == 12.5%; 32 CPU-seconds
// against 8 cores x 100s == 800 core-seconds == 4%.
func hostFixture() Snapshot {
	return Snapshot{
		Elapsed: 100 * time.Second,
		Samples: 50,
		Kernel: Half{
			CPUUser: 20 * time.Second, CPUSys: 4 * time.Second, HaveCPU: true,
			RSSBytes: 1536 << 20, HaveRSS: true,
		},
		Agent: Half{
			CPUUser: 7 * time.Second, CPUSys: time.Second, HaveCPU: true,
			RSSBytes: 512 << 20, HaveRSS: true,
		},
		NumCPU: 8, GOMAXPROCS: 8,
		Host: Host{TotalRAMBytes: 16 << 30, AvailRAMBytes: 4 << 30, HaveRAM: true},
	}
}

func TestHostFractions(t *testing.T) {
	s := hostFixture()

	rss, ok := s.HarnessRSSBytes()
	if !ok || rss != 2<<30 {
		t.Fatalf("HarnessRSSBytes = %d (ok=%v), want %d (both halves summed)", rss, ok, uint64(2)<<30)
	}
	pct, ok := s.RSSPercentOfHostRAM()
	if !ok || pct != 12.5 {
		t.Fatalf("RSSPercentOfHostRAM = %v (ok=%v), want 12.5 (2 GiB of 16 GiB)", pct, ok)
	}
	coreS, ok := s.HostCoreSeconds()
	if !ok || coreS != 800 {
		t.Fatalf("HostCoreSeconds = %v (ok=%v), want 800 (8 cores x 100s)", coreS, ok)
	}
	cpu, ok := s.HarnessCPUSeconds()
	if !ok || cpu != 32 {
		t.Fatalf("HarnessCPUSeconds = %v (ok=%v), want 32 (24 kernel + 8 agent)", cpu, ok)
	}
	cpuPct, ok := s.CPUPercentOfHost()
	if !ok || cpuPct != 4 {
		t.Fatalf("CPUPercentOfHost = %v (ok=%v), want 4 (32 of 800 core-s)", cpuPct, ok)
	}
}

// A host axis the platform cannot read must render n/a, never a fabricated 0 — the same
// presence-bit rule the hardware axes already follow.
func TestHostFractionsAbsentWhenUnread(t *testing.T) {
	noRAM := hostFixture()
	noRAM.Host = Host{} // no host-memory reader on this platform
	if _, ok := noRAM.RSSPercentOfHostRAM(); ok {
		t.Error("host RAM unread: RSSPercentOfHostRAM must report ok=false")
	}
	if _, ok := noRAM.CPUPercentOfHost(); !ok {
		t.Error("CPU fraction needs only cores+elapsed: it must survive an unread RAM axis")
	}

	noCPU := hostFixture()
	noCPU.Kernel.HaveCPU, noCPU.Agent.HaveCPU = false, false
	if _, ok := noCPU.HarnessCPUSeconds(); ok {
		t.Error("neither half reported CPU: HarnessCPUSeconds must report ok=false")
	}
	if _, ok := noCPU.CPUPercentOfHost(); ok {
		t.Error("neither half reported CPU: CPUPercentOfHost must report ok=false")
	}

	noElapsed := hostFixture()
	noElapsed.Elapsed = 0
	if _, ok := noElapsed.HostCoreSeconds(); ok {
		t.Error("zero elapsed: HostCoreSeconds must report ok=false")
	}
}

// The Done-when of #2053: the exit summary reports harness RSS as both bytes AND % of
// host RAM, and CPU-seconds against host core-seconds elapsed.
func TestReportRendersHarnessAsFractionOfHost(t *testing.T) {
	out := hostFixture().Report()
	for _, want := range []string{
		"host 8 cores",
		"16.0 GiB ram",
		"(4.0 GiB avail)",
		"harness/host rss 2.0 GiB (12.5% of host ram)",
		"cpu 32.0s of 800.0 core-s (4.0%)",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("Report() missing %q\ngot: %s", want, out)
		}
	}
}

// With no host reader the summary still names the box's cores and the CPU share, and
// says the RAM fraction is unavailable rather than printing a 0%.
func TestReportHostAxisAbsentReadsNA(t *testing.T) {
	s := hostFixture()
	s.Host = Host{}
	out := s.Report()
	if !strings.Contains(out, "host 8 cores;") {
		t.Errorf("Report() should still name the core count\ngot: %s", out)
	}
	if !strings.Contains(out, "rss 2.0 GiB (host ram n/a)") {
		t.Errorf("unread host RAM must render n/a, not a fabricated percent\ngot: %s", out)
	}
	if strings.Contains(out, "of host ram)") {
		t.Errorf("unread host RAM must not render a fraction\ngot: %s", out)
	}
}

func TestReportRendersHostLoadWhenSupplied(t *testing.T) {
	s := hostFixture()
	s.Host.Load1, s.Host.HaveLoad = 3.5, true
	if !strings.Contains(s.Report(), "load 3.50") {
		t.Errorf("Report() missing host load\ngot: %s", s.Report())
	}
	if strings.Contains(hostFixture().Report(), "load ") {
		t.Error("load unread: Report() must not render a load figure")
	}
}

// humanPercent must keep a small-but-real fraction legible: a harness using a sliver of
// a build server's RAM must not round to "0%", which reads as "unmeasured".
func TestHumanPercentKeepsSmallFractionsVisible(t *testing.T) {
	cases := map[float64]string{
		42.4:  "42.4%",
		12.5:  "12.5%",
		1.25:  "1.2%",
		0.031: "0.03%",
	}
	for in, want := range cases {
		if got := humanPercent(in); got != want {
			t.Errorf("humanPercent(%v) = %q, want %q", in, got, want)
		}
	}
}

func TestPrometheusHostGauges(t *testing.T) {
	out := hostFixture().PrometheusText()
	for _, want := range []string{
		`fak_harness_host_ram_bytes{kind="total"} 17179869184`,
		`fak_harness_host_ram_bytes{kind="available"} 4294967296`,
		"fak_harness_host_core_seconds 800",
		"# TYPE fak_harness_host_ram_bytes gauge",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("PrometheusText() missing %q\n%s", want, out)
		}
	}
	if strings.Contains(out, "fak_harness_host_load1") {
		t.Errorf("load unread but emitted\n%s", out)
	}

	bare := Snapshot{NumCPU: 4} // no host reader, no elapsed
	if got := bare.PrometheusText(); strings.Contains(got, "fak_harness_host_ram_bytes") {
		t.Errorf("host RAM unread but emitted\n%s", got)
	}
}

func TestMarshalLedgerRowCarriesHostBlock(t *testing.T) {
	b, err := hostFixture().MarshalLedgerRow("guard", "anthropic", "claude", time.Unix(1700000000, 0))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("row is not valid JSON: %v\n%s", err, b)
	}
	host, ok := m["host"].(map[string]any)
	if !ok {
		t.Fatalf("row has no host block: %s", b)
	}
	for k, want := range map[string]float64{
		"cores":                          8,
		"ram_total_bytes":                17179869184,
		"ram_avail_bytes":                4294967296,
		"core_seconds":                   800,
		"harness_rss_bytes":              2147483648,
		"harness_rss_pct_of_host_ram":    12.5,
		"harness_cpu_s":                  32,
		"harness_cpu_pct_of_host_core_s": 4,
	} {
		got, present := host[k].(float64)
		if !present {
			t.Errorf("host.%s missing: %s", k, b)
			continue
		}
		if got != want {
			t.Errorf("host.%s = %v, want %v", k, got, want)
		}
	}
	if _, present := host["load1"]; present {
		t.Errorf("load unread but recorded: %s", b)
	}
}

// An unread host axis must stay OUT of the durable row rather than banking a
// misleading 0 that a later reader would take as a measurement.
func TestMarshalLedgerRowOmitsUnreadHostAxes(t *testing.T) {
	s := Snapshot{NumCPU: 4} // no host reader, no elapsed, no halves
	b, err := s.MarshalLedgerRow("guard", "anthropic", "claude", time.Unix(1700000000, 0))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	host := m["host"].(map[string]any)
	if host["cores"].(float64) != 4 {
		t.Errorf("cores is always known, want 4: %s", b)
	}
	for _, k := range []string{"ram_total_bytes", "ram_avail_bytes", "load1", "core_seconds", "harness_rss_bytes", "harness_rss_pct_of_host_ram", "harness_cpu_s", "harness_cpu_pct_of_host_core_s"} {
		if _, present := host[k]; present {
			t.Errorf("host.%s unread but recorded: %s", k, b)
		}
	}
}

// The provider seam keeps the leaf stdlib-only: the caller (fak guard) supplies the
// host reading, exactly as it already does for the network and GPU axes.
func TestHostProviderFeedsSnapshot(t *testing.T) {
	clock := time.Unix(1000, 0)
	s := newSampler(func() time.Time { return clock })
	calls := 0
	s.SetHostProvider(func() (Host, bool) {
		calls++
		return Host{TotalRAMBytes: 32 << 30, AvailRAMBytes: 8 << 30, HaveRAM: true, Load1: 1.5, HaveLoad: true}, true
	})
	s.foldProc(procSample{haveRSS: true, rss: 1 << 30}, clock, 1, 0)
	if calls != 1 {
		t.Fatalf("host provider called %d times, want 1 per sample", calls)
	}
	snap := s.Snapshot()
	if !snap.Host.HaveRAM || snap.Host.TotalRAMBytes != 32<<30 {
		t.Fatalf("host RAM not folded: %+v", snap.Host)
	}
	if !snap.Host.HaveLoad || snap.Host.Load1 != 1.5 {
		t.Fatalf("host load not folded: %+v", snap.Host)
	}

	// A provider that cannot read (ok=false) must leave the block absent, not zero it.
	s2 := newSampler(func() time.Time { return clock })
	s2.SetHostProvider(func() (Host, bool) { return Host{}, false })
	s2.foldProc(procSample{}, clock, 1, 0)
	if s2.Snapshot().Host.HaveRAM {
		t.Error("provider reported ok=false: host block must stay absent")
	}

	// No provider at all (the default) is a no-op, not a panic.
	s3 := newSampler(func() time.Time { return clock })
	s3.foldProc(procSample{}, clock, 1, 0)
	if s3.Snapshot().Host.HaveRAM {
		t.Error("no host provider: host block must stay absent")
	}
	(*Sampler)(nil).SetHostProvider(nil) // nil receiver is a no-op
}
