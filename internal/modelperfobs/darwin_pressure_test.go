package modelperfobs

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestParseDarwinPressureSignals(t *testing.T) {
	vmStat := `Mach Virtual Memory Statistics: (page size of 4096 bytes)
Pages free:                               100.
Pages inactive:                           200.
Pages wired down:                         300.
Pages occupied by compressor:             400.
Pageins:                                  1000.
Pageouts:                                   10.
Swapins:                                    20.
Swapouts:                                   30.
`
	var host HostSignals
	if !parseDarwinVMStat(vmStat, &host) {
		t.Fatal("valid vm_stat input reported unavailable")
	}
	assertUint64Pointer(t, "free bytes", host.PhysicalFreeBytes, 100*4096)
	assertUint64Pointer(t, "available approximation", host.PhysicalAvailableBytes, 300*4096)
	if host.PhysicalAvailableSemantics != darwinAvailableSemantics {
		t.Fatalf("available semantics=%q", host.PhysicalAvailableSemantics)
	}
	assertUint64Pointer(t, "wired resident", host.MemoryWiredResidentBytes, 300*4096)
	assertUint64Pointer(t, "compressed resident", host.MemoryCompressedResidentBytes, 400*4096)
	assertUint64Pointer(t, "page ins", host.MemoryPageInPagesTotal, 1000)
	assertUint64Pointer(t, "page outs", host.MemoryPageOutPagesTotal, 10)
	assertUint64Pointer(t, "swap ins", host.MemorySwapInPagesTotal, 20)
	assertUint64Pointer(t, "swap outs", host.MemorySwapOutPagesTotal, 30)

	if !parseDarwinPhysicalTotal("34359738368\n", &host) {
		t.Fatal("valid hw.memsize input reported unavailable")
	}
	if !parseDarwinSwapUsage("total = 4096.00M  used = 512.50M  free = 3583.50M  (encrypted)\n", &host) {
		t.Fatal("valid vm.swapusage input reported unavailable")
	}
	if !parseDarwinPSRSS("12345\n", &host) {
		t.Fatal("valid ps RSS input reported unavailable")
	}
	assertUint64Pointer(t, "physical total", host.PhysicalTotalBytes, 34359738368)
	assertUint64Pointer(t, "swap total", host.SwapTotalBytes, 4096*1024*1024)
	assertUint64Pointer(t, "swap used", host.SwapUsedBytes, 512*1024*1024+512*1024)
	assertUint64Pointer(t, "process RSS", host.ProcessResidentBytes, 12345*1024)
}

func TestParseDarwinPressureMalformedRowsStayOmitted(t *testing.T) {
	vmStat := `Mach Virtual Memory Statistics: (page size of nope bytes)
Pages free: 100.
Pages inactive: 200.
Pages wired down: not-a-number.
Pages occupied by compressor: -1.
Pageins: broken.
Pageouts: 7.
`
	var host HostSignals
	if !parseDarwinVMStat(vmStat, &host) {
		t.Fatal("independently valid page-out counter should survive malformed rows")
	}
	if host.PhysicalFreeBytes != nil || host.PhysicalAvailableBytes != nil ||
		host.MemoryWiredResidentBytes != nil || host.MemoryCompressedResidentBytes != nil ||
		host.MemoryPageInPagesTotal != nil {
		t.Fatalf("malformed fields must remain omitted: %+v", host)
	}
	if host.PhysicalAvailableSemantics != "" {
		t.Fatalf("semantics without an approximation: %q", host.PhysicalAvailableSemantics)
	}
	assertUint64Pointer(t, "independent page outs", host.MemoryPageOutPagesTotal, 7)

	if parseDarwinPhysicalTotal("0\n", &host) || parseDarwinPhysicalTotal("1 extra\n", &host) {
		t.Fatal("malformed physical total reported available")
	}
	var partial HostSignals
	if !parseDarwinVMStat("Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 5.\n", &partial) {
		t.Fatal("valid free-page row reported unavailable")
	}
	assertUint64Pointer(t, "partial free bytes", partial.PhysicalFreeBytes, 5*4096)
	if partial.PhysicalAvailableBytes != nil || partial.PhysicalAvailableSemantics != "" {
		t.Fatalf("missing inactive pages must not become an available-memory zero: %+v", partial)
	}
	var malformedSwap HostSignals
	if parseDarwinSwapUsage("total = 1/2M used = 2.00Q", &malformedSwap) {
		t.Fatal("malformed swap values reported available")
	}
	if parseDarwinPSRSS("header 12345", &malformedSwap) {
		t.Fatal("malformed RSS reported available")
	}
	encoded, err := json.Marshal(malformedSwap)
	if err != nil {
		t.Fatal(err)
	}
	if string(encoded) != "{}" {
		t.Fatalf("malformed values encoded as observations: %s", encoded)
	}
}

func TestParseDarwinPressureOmitsOverflowAndImpossibleUsage(t *testing.T) {
	vmStat := `Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 18446744073709551615.
Pages inactive: 1.
Pages wired down: 18446744073709551615.
`
	var host HostSignals
	parseDarwinVMStat(vmStat, &host)
	if host.PhysicalFreeBytes != nil || host.PhysicalAvailableBytes != nil || host.MemoryWiredResidentBytes != nil {
		t.Fatalf("overflowed byte values must remain omitted: %+v", host)
	}

	if !parseDarwinSwapUsage("total = 1.00G used = 2.00G", &host) {
		t.Fatal("valid swap capacity should survive impossible used value")
	}
	assertUint64Pointer(t, "swap total", host.SwapTotalBytes, 1<<30)
	if host.SwapUsedBytes != nil {
		t.Fatalf("swap used above capacity must remain omitted: %+v", host)
	}
}

func TestDeriveDarwinPagingRatesAndReset(t *testing.T) {
	pageIn1, pageIn2 := uint64(100), uint64(130)
	pageOut1, pageOut2 := uint64(20), uint64(24)
	prev := BandwidthSample{
		Provenance: BandwidthProvenance{SampledAt: "2026-08-27T00:00:00Z"},
		Host: HostSignals{
			MemoryPageInPagesTotal:  &pageIn1,
			MemoryPageOutPagesTotal: &pageOut1,
		},
	}
	curr := BandwidthSample{
		Provenance: BandwidthProvenance{SampledAt: "2026-08-27T00:00:02Z"},
		Host: HostSignals{
			MemoryPageInPagesTotal:  &pageIn2,
			MemoryPageOutPagesTotal: &pageOut2,
		},
	}
	deriveHostPressureRates(&prev, &curr)
	assertRate(t, "darwin page in", curr.Host.MemoryPageInPagesPerSecond, 15)
	assertRate(t, "darwin page out", curr.Host.MemoryPageOutPagesPerSecond, 2)
	if prev.Host.MemoryPageInPagesPerSecond != nil || prev.Host.MemoryPageOutPagesPerSecond != nil {
		t.Fatal("first sample must not fabricate Darwin paging rates")
	}

	reset := BandwidthSample{
		Provenance: BandwidthProvenance{SampledAt: "2026-08-27T00:00:03Z"},
		Host: HostSignals{
			MemoryPageInPagesTotal:  new(uint64),
			MemoryPageOutPagesTotal: new(uint64),
		},
	}
	deriveHostPressureRates(&curr, &reset)
	if reset.Host.MemoryPageInPagesPerSecond != nil || reset.Host.MemoryPageOutPagesPerSecond != nil {
		t.Fatalf("counter reset must remain omitted: %+v", reset.Host)
	}
}

func TestDarwinPagingRatesUseVMStatObservationMidpoints(t *testing.T) {
	clock := time.Date(2026, 8, 27, 0, 0, 0, 0, time.UTC)
	vmStatRuns := 0
	run := func(_ context.Context, name string, args ...string) (string, error) {
		if name == "/usr/bin/vm_stat" {
			vmStatRuns++
			clock = clock.Add(2 * time.Second)
			pageIns := 100
			swapIns := 10
			if vmStatRuns == 2 {
				pageIns = 132
				swapIns = 42
			}
			return fmt.Sprintf("Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPageins: %d.\nSwapins: %d.\n", pageIns, swapIns), nil
		}
		unrelatedDuration := 10 * time.Second
		if vmStatRuns == 2 {
			unrelatedDuration = 100 * time.Second
		}
		clock = clock.Add(unrelatedDuration)
		switch name + " " + strings.Join(args, " ") {
		case "/usr/sbin/sysctl -n hw.memsize":
			return "34359738368\n", nil
		case "/usr/sbin/sysctl -n vm.swapusage":
			return "total = 4096.00M used = 512.00M", nil
		case "/bin/ps -o rss= -p 42":
			return "12345\n", nil
		default:
			return "", fmt.Errorf("unexpected command %s %v", name, args)
		}
	}
	now := func() time.Time { return clock }

	first := collectDarwinHostSnapshot(context.Background(), 42, run, now)
	second := collectDarwinHostSnapshot(context.Background(), 42, run, now)
	if want := time.Date(2026, 8, 27, 0, 0, 1, 0, time.UTC); !first.at.Equal(want) {
		t.Fatalf("first observation time=%s want vm_stat midpoint %s", first.at, want)
	}
	if want := time.Date(2026, 8, 27, 0, 0, 33, 0, time.UTC); !second.at.Equal(want) {
		t.Fatalf("second observation time=%s want vm_stat midpoint %s", second.at, want)
	}

	previous := BandwidthSample{Provenance: BandwidthProvenance{SampledAt: first.at.Format(time.RFC3339Nano)}, Host: first.host}
	current := BandwidthSample{Provenance: BandwidthProvenance{SampledAt: second.at.Format(time.RFC3339Nano)}, Host: second.host}
	deriveHostPressureRates(&previous, &current)
	assertRate(t, "page-in midpoint rate", current.Host.MemoryPageInPagesPerSecond, 1)
	assertRate(t, "swap-in midpoint rate", current.Host.MemorySwapInPagesPerSecond, 1)
}

func TestDarwinApproximateAvailabilityStaysHostOnly(t *testing.T) {
	total, available := uint64(1000), uint64(1)
	latency := 2.0
	host := HostSignals{
		PhysicalTotalBytes:         &total,
		PhysicalAvailableBytes:     &available,
		PhysicalAvailableSemantics: darwinAvailableSemantics,
	}
	capacity := capacityFromExactHostAvailability(host)
	if capacity.TotalBytes != nil || capacity.UsedBytes != nil || capacity.Utilization != nil {
		t.Fatalf("approximate availability promoted to generic capacity: %+v", capacity)
	}

	report, err := AnalyzeBandwidth(BandwidthCapture{
		Engine: "fak-native",
		Trigger: TriggerConfig{
			SymptomWindow:       1,
			ResourceWindow:      1,
			LatencyThresholdMS:  1,
			ResourceUtilization: .9,
		},
		Samples: []BandwidthSample{{
			Phase:      PhaseOther,
			Shape:      ShapeSmall,
			Provenance: BandwidthProvenance{Source: "live-host"},
			Request:    RequestSignals{LatencyMS: &latency},
			Host:       host,
			Capacity:   capacity,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	observation := report.Observations[0]
	if observation.Bottleneck != BottleneckUnknown {
		t.Fatalf("approximate availability classified as %q", observation.Bottleneck)
	}
	if observation.DeepCapture.ResourceStreak != 0 || observation.DeepCapture.Triggered {
		t.Fatalf("approximate availability became actionable resource pressure: %+v", observation.DeepCapture)
	}
	if observation.Host.PhysicalAvailableSemantics != darwinAvailableSemantics ||
		observation.Host.PhysicalAvailableBytes == nil {
		t.Fatalf("approximation lost as Host evidence: %+v", observation.Host)
	}

	exact := host
	exact.PhysicalAvailableSemantics = ""
	exactCapacity := capacityFromExactHostAvailability(exact)
	assertUint64Pointer(t, "exact total capacity", exactCapacity.TotalBytes, total)
	assertUint64Pointer(t, "exact used capacity", exactCapacity.UsedBytes, total-available)
}

func TestDarwinPressureJSONIsExplicitlyNonBandwidth(t *testing.T) {
	total, available, free := uint64(32<<30), uint64(8<<30), uint64(2<<30)
	wired, compressed := uint64(10<<30), uint64(3<<30)
	swapTotal, rss := uint64(4<<30), uint64(512<<20)
	pageIns, pageOutRate := uint64(123), 4.5
	host := HostSignals{
		PhysicalTotalBytes:            &total,
		PhysicalAvailableBytes:        &available,
		PhysicalAvailableSemantics:    darwinAvailableSemantics,
		PhysicalFreeBytes:             &free,
		MemoryWiredResidentBytes:      &wired,
		MemoryCompressedResidentBytes: &compressed,
		SwapTotalBytes:                &swapTotal,
		ProcessResidentBytes:          &rss,
		MemoryPageInPagesTotal:        &pageIns,
		MemoryPageOutPagesPerSecond:   &pageOutRate,
	}
	encoded, err := json.Marshal(host)
	if err != nil {
		t.Fatal(err)
	}
	got := string(encoded)
	for _, field := range []string{
		`"physical_total_bytes":`,
		`"physical_available_semantics":"darwin-vm-stat-free-plus-inactive-pages"`,
		`"memory_wired_resident_bytes":`,
		`"memory_compressed_resident_bytes":`,
		`"swap_total_bytes":`,
		`"process_resident_bytes":`,
		`"memory_page_in_pages_total":123`,
		`"memory_page_out_pages_per_second":4.5`,
	} {
		if !strings.Contains(got, field) {
			t.Fatalf("missing explicit capacity/pressure/activity field %q: %s", field, got)
		}
	}
	if strings.Contains(got, "gb_s") || strings.Contains(got, "bandwidth") ||
		strings.Contains(got, "dram") || strings.Contains(got, "unified") {
		t.Fatalf("Darwin pressure mislabeled as memory bandwidth: %s", got)
	}
	if strings.Contains(got, "swap_used_bytes") {
		t.Fatalf("unsupported nil field must be omitted, not synthesized: %s", got)
	}
}

func assertUint64Pointer(t *testing.T, name string, got *uint64, want uint64) {
	t.Helper()
	if got == nil || *got != want {
		t.Fatalf("%s=%v want %d", name, got, want)
	}
}
