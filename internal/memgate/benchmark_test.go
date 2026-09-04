package memgate

import "testing"

const (
	benchLinuxMeminfo = `MemTotal: 36000000 kB
MemFree: 4000000 kB
MemAvailable: 28000000 kB
Cached: 12000000 kB
`
	benchDarwinVMStat = `Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                    10279.
Pages active:                                1012697.
Pages inactive:                              1012324.
Pages speculative:                              1215.
Pages throttled:                                   0.
Pages wired down:                             154812.
Pages purgeable:                               13107.
`
	benchHoldersOutput = `PID RSS COMM
123 2500000 llama-server
456 100 shell
789 1500000 python worker
1011 3200000 fak serve
1213 1800000 runner
`
)

// BenchmarkMemGate measures end-to-end memory admission and snapshot evaluation in a b.N loop.
func BenchmarkMemGate(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mem := ParseLinux(benchLinuxMeminfo)
		holders := ParseHolders(benchHoldersOutput)
		snap := BuildSnapshot("linux", mem, holders)
		eval := Evaluate(snap, 8.0)
		sample := AdmissionSampleFor(mem)
		if eval.Admit == nil || sample.Pressure == PressureUnknown {
			b.Fatal("unexpected rejection during benchmark")
		}
	}
}

// BenchmarkParseLinux measures procfs meminfo parsing throughput in a b.N loop.
func BenchmarkParseLinux(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mem := ParseLinux(benchLinuxMeminfo)
		if mem.TotalBytes == 0 {
			b.Fatal("failed parsing meminfo")
		}
	}
}

// BenchmarkParseDarwin measures macOS vm_stat parsing throughput in a b.N loop.
func BenchmarkParseDarwin(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mem := ParseDarwin(benchDarwinVMStat, 16384, 38_654_705_664)
		if mem.AvailableBytes == 0 {
			b.Fatal("failed parsing vm_stat")
		}
	}
}

// BenchmarkParseHolders measures ps process listing filtering and sorting in a b.N loop.
func BenchmarkParseHolders(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		holders := ParseHolders(benchHoldersOutput)
		if len(holders) == 0 {
			b.Fatal("failed parsing holders")
		}
	}
}

// BenchmarkBuildSnapshot measures snapshot aggregation and threshold calculation in a b.N loop.
func BenchmarkBuildSnapshot(b *testing.B) {
	mem := Memory{
		TotalBytes:      36_000_000_000,
		FreeBytes:       4_000_000_000,
		AvailableBytes:  28_000_000_000,
		PurgeableBytes:  12_000_000_000,
		WiredBytes:      2_000_000_000,
		CompressedBytes: 500_000_000,
	}
	holders := []Holder{
		{PID: 123, RSSGB: 2.5, Comm: "llama-server"},
		{PID: 1011, RSSGB: 3.2, Comm: "fak serve"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		snap := BuildSnapshot("linux", mem, holders)
		if snap.HighWired {
			b.Fatal("unexpected high wired flag")
		}
	}
}

// BenchmarkEvaluate measures admission comparison and shortfall calculations in a b.N loop.
func BenchmarkEvaluate(b *testing.B) {
	snap := Snapshot{
		Platform:    "linux",
		TotalGB:     36.0,
		FreeGB:      4.0,
		AvailableGB: 28.0,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := Evaluate(snap, 14.0)
		if result.Admit == nil || !*result.Admit {
			b.Fatal("expected admission")
		}
	}
}

// BenchmarkAdmissionSampleFor measures pressure classification and sample construction in a b.N loop.
func BenchmarkAdmissionSampleFor(b *testing.B) {
	mem := Memory{
		TotalBytes:      16 << 30,
		AvailableBytes:  8 << 30,
		WiredBytes:      2 << 30,
		CompressedBytes: 1 << 30,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sample := AdmissionSampleFor(mem)
		if sample.Pressure != PressureNormal {
			b.Fatalf("unexpected pressure: %s", sample.Pressure)
		}
	}
}
