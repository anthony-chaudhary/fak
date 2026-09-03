//go:build linux

package modelperfobs

import (
	"reflect"
	"testing"
)

func TestParseProcSelfStatFaultsHandlesSpacesInComm(t *testing.T) {
	line := "123 (fak worker) R 1 2 3 4 5 6 17 8 19 10 0 0"
	minor, major, ok := parseProcSelfStatFaults(line)
	if !ok || minor != 17 || major != 19 {
		t.Fatalf("minor=%d major=%d ok=%v", minor, major, ok)
	}
}

func TestParseProcSelfStatFaultsRejectsMalformed(t *testing.T) {
	if _, _, ok := parseProcSelfStatFaults("123 malformed"); ok {
		t.Fatal("expected malformed row rejection")
	}
}

func TestParseProcPressureMemoryDeterministic(t *testing.T) {
	const input = `some avg10=1.25 avg60=2.50 avg300=3.75 total=123456 extra=ignored
full avg10=0.10 avg60=0.20 avg300=0.30 total=654321
cpu avg10=99.00 avg60=99.00 avg300=99.00 total=999
`
	var first, second HostSignals
	if !parseProcPressureMemory(input, &first) || !parseProcPressureMemory(input, &second) {
		t.Fatal("expected supported PSI fields")
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("non-deterministic parse:\nfirst=%+v\nsecond=%+v", first, second)
	}
	assertFloat64(t, "some avg10", first.MemoryPressureSomeAvg10Percent, 1.25)
	assertFloat64(t, "some avg60", first.MemoryPressureSomeAvg60Percent, 2.50)
	assertFloat64(t, "some avg300", first.MemoryPressureSomeAvg300Percent, 3.75)
	assertUint64(t, "some total stall", first.MemoryPressureSomeTotalStallMicroseconds, 123456)
	assertFloat64(t, "full avg10", first.MemoryPressureFullAvg10Percent, 0.10)
	assertFloat64(t, "full avg60", first.MemoryPressureFullAvg60Percent, 0.20)
	assertFloat64(t, "full avg300", first.MemoryPressureFullAvg300Percent, 0.30)
	assertUint64(t, "full total stall", first.MemoryPressureFullTotalStallMicroseconds, 654321)
}

func TestParseProcPressureMemoryOmitsMalformedFields(t *testing.T) {
	const input = `some avg10=NaN avg60=-1 avg300=101 total=broken
full avg10=0 avg60=1.5 avg300=broken total=42
`
	var host HostSignals
	if !parseProcPressureMemory(input, &host) {
		t.Fatal("expected valid full-pressure fields")
	}
	if host.MemoryPressureSomeAvg10Percent != nil ||
		host.MemoryPressureSomeAvg60Percent != nil ||
		host.MemoryPressureSomeAvg300Percent != nil ||
		host.MemoryPressureSomeTotalStallMicroseconds != nil ||
		host.MemoryPressureFullAvg300Percent != nil {
		t.Fatalf("malformed fields were not omitted: %+v", host)
	}
	assertFloat64(t, "full avg10", host.MemoryPressureFullAvg10Percent, 0)
	assertFloat64(t, "full avg60", host.MemoryPressureFullAvg60Percent, 1.5)
	assertUint64(t, "full total stall", host.MemoryPressureFullTotalStallMicroseconds, 42)

	var unavailable HostSignals
	if parseProcPressureMemory("some avg10=bad total=-1\n", &unavailable) {
		t.Fatalf("malformed-only input reported available: %+v", unavailable)
	}
}

func TestParseProcVMStatSelectedCounters(t *testing.T) {
	const input = `nr_free_pages 900
pgscan_kswapd 100
pgscan_direct malformed
pgsteal_kswapd 80
pgsteal_direct 50 extra
pswpin 7
pswpout 9
`
	var first, second HostSignals
	if !parseProcVMStat(input, &first) || !parseProcVMStat(input, &second) {
		t.Fatal("expected supported vmstat counters")
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("non-deterministic parse:\nfirst=%+v\nsecond=%+v", first, second)
	}
	assertUint64(t, "kswapd scanned", first.MemoryReclaimKswapdScannedPagesTotal, 100)
	assertUint64(t, "kswapd reclaimed", first.MemoryReclaimKswapdReclaimedPagesTotal, 80)
	assertUint64(t, "swap in", first.MemorySwapInPagesTotal, 7)
	assertUint64(t, "swap out", first.MemorySwapOutPagesTotal, 9)
	if first.MemoryReclaimDirectScannedPagesTotal != nil || first.MemoryReclaimDirectReclaimedPagesTotal != nil {
		t.Fatalf("malformed vmstat counters were not omitted: %+v", first)
	}

	var unavailable HostSignals
	if parseProcVMStat("pgscan_kswapd malformed\npswpout -1\n", &unavailable) {
		t.Fatalf("malformed-only input reported available: %+v", unavailable)
	}
}

func TestParseProcVMStatLegacyZoneCounters(t *testing.T) {
	const input = `pgscan_kswapd_dma 1
pgscan_kswapd_dma32 2
pgscan_kswapd_normal 3
pgscan_kswapd_high 4
pgscan_kswapd_movable 5
pgscan_direct_dma 6
pgscan_direct_normal 7
pgsteal_kswapd_dma32 8
pgsteal_kswapd_high 9
pgsteal_direct_normal 10
pgsteal_direct_movable 11
`
	var first, second HostSignals
	if !parseProcVMStat(input, &first) || !parseProcVMStat(input, &second) {
		t.Fatal("expected supported legacy vmstat counters")
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("non-deterministic legacy parse:\nfirst=%+v\nsecond=%+v", first, second)
	}
	assertUint64(t, "legacy kswapd scanned", first.MemoryReclaimKswapdScannedPagesTotal, 15)
	assertUint64(t, "legacy direct scanned", first.MemoryReclaimDirectScannedPagesTotal, 13)
	assertUint64(t, "legacy kswapd reclaimed", first.MemoryReclaimKswapdReclaimedPagesTotal, 17)
	assertUint64(t, "legacy direct reclaimed", first.MemoryReclaimDirectReclaimedPagesTotal, 21)
}

func TestParseProcVMStatAggregatePrecedenceOnMixedInput(t *testing.T) {
	const input = `pgscan_kswapd_normal 40
pgscan_kswapd 100
pgscan_kswapd_dma 60
pgscan_direct_normal 7
pgscan_direct_movable 8
pgsteal_kswapd 80
pgsteal_kswapd_normal 50
pgsteal_direct_dma32 9
pswpin 3
`
	var host HostSignals
	if !parseProcVMStat(input, &host) {
		t.Fatal("expected mixed vmstat counters")
	}
	assertUint64(t, "aggregate kswapd scanned", host.MemoryReclaimKswapdScannedPagesTotal, 100)
	assertUint64(t, "legacy direct scanned", host.MemoryReclaimDirectScannedPagesTotal, 15)
	assertUint64(t, "aggregate kswapd reclaimed", host.MemoryReclaimKswapdReclaimedPagesTotal, 80)
	assertUint64(t, "legacy direct reclaimed", host.MemoryReclaimDirectReclaimedPagesTotal, 9)
	assertUint64(t, "swap in", host.MemorySwapInPagesTotal, 3)
}

func TestParseProcVMStatMalformedLegacyCountersAreOmitted(t *testing.T) {
	const input = `pgscan_kswapd malformed
pgscan_kswapd_normal 12
pgscan_direct_dma malformed
pgscan_direct_normal 7
pgscan_direct_device 99
pgsteal_kswapd_high -1
pgsteal_direct_normal 7 extra
`
	var host HostSignals
	if !parseProcVMStat(input, &host) {
		t.Fatal("expected representable legacy vmstat counter")
	}
	if host.MemoryReclaimKswapdScannedPagesTotal != nil ||
		host.MemoryReclaimKswapdReclaimedPagesTotal != nil ||
		host.MemoryReclaimDirectReclaimedPagesTotal != nil {
		t.Fatalf("malformed or unrecognized legacy counters were not omitted: %+v", host)
	}
	assertUint64(t, "valid legacy counter", host.MemoryReclaimDirectScannedPagesTotal, 7)

	var unavailable HostSignals
	if parseProcVMStat("pgscan_direct_dma malformed\npgsteal_kswapd_high -1\n", &unavailable) {
		t.Fatalf("malformed-only input reported available: %+v", unavailable)
	}
}

func TestParseProcVMStatLegacyCounterOverflowIsOmitted(t *testing.T) {
	const input = `pgscan_kswapd_dma 18446744073709551615
pgscan_kswapd_normal 1
pgscan_direct_normal 7
`
	var host HostSignals
	if !parseProcVMStat(input, &host) {
		t.Fatal("expected representable legacy vmstat counter")
	}
	if host.MemoryReclaimKswapdScannedPagesTotal != nil {
		t.Fatalf("overflowing legacy sum must be omitted, got %d", *host.MemoryReclaimKswapdScannedPagesTotal)
	}
	assertUint64(t, "unrelated legacy counter", host.MemoryReclaimDirectScannedPagesTotal, 7)
}

func assertUint64(t *testing.T, name string, got *uint64, want uint64) {
	t.Helper()
	if got == nil || *got != want {
		t.Fatalf("%s=%v want %d", name, got, want)
	}
}

func assertFloat64(t *testing.T, name string, got *float64, want float64) {
	t.Helper()
	if got == nil || *got != want {
		t.Fatalf("%s=%v want %v", name, got, want)
	}
}

func TestReadAndParseProcHostSignals(t *testing.T) {
	var host HostSignals
	called := false
	parser := func(content string, h *HostSignals) bool {
		called = true
		return true
	}
	if readAndParseProc("/nonexistent/proc/test", parser, &host) {
		t.Fatal("expected false for nonexistent path")
	}
	if called {
		t.Fatal("parser should not be called for nonexistent path")
	}
}
