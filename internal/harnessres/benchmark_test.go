package harnessres

import (
	"fmt"
	"testing"
	"time"
)

var (
	benchSnapshotSink Snapshot
	benchStringSink   string
	benchBytesSink    []byte
	benchIntSink      int
	benchBoolSink     bool
	benchProcsSink    []ProcRef
	benchRollupSink   FleetRollup
)

// BenchmarkSamplerFoldProc measures folding raw process readings into the Sampler,
// updating peaks and CPU deltas under the sampler mutex.
func BenchmarkSamplerFoldProc(b *testing.B) {
	clock := time.Unix(1000, 0)
	s := newSampler(func() time.Time { return clock })
	samples := []procSample{
		{haveCPU: true, cpuUser: 100 * time.Millisecond, cpuSys: 20 * time.Millisecond, haveRSS: true, rss: 120 << 20},
		{haveCPU: true, cpuUser: 250 * time.Millisecond, cpuSys: 40 * time.Millisecond, haveRSS: true, rss: 150 << 20, havePeakRSS: true, peakRSS: 160 << 20},
		{haveCPU: true, cpuUser: 400 * time.Millisecond, cpuSys: 60 * time.Millisecond, haveRSS: true, rss: 110 << 20, haveIO: true, ioRead: 5 << 20, ioWrite: 1 << 20},
		{haveCPU: true, cpuUser: 600 * time.Millisecond, cpuSys: 80 * time.Millisecond, haveRSS: true, rss: 180 << 20, havePeakRSS: true, peakRSS: 180 << 20},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ps := samples[i%len(samples)]
		clock = clock.Add(time.Second)
		s.foldProc(ps, clock, 25+i%10, uint64(50+i%20)<<20)
	}
}

// BenchmarkSamplerSnapshot measures the cost of extracting a point-in-time Snapshot
// under mutex lock from an active Sampler.
func BenchmarkSamplerSnapshot(b *testing.B) {
	clock := time.Unix(1000, 0)
	s := newSampler(func() time.Time { return clock })
	s.foldProc(procSample{
		haveCPU: true, cpuUser: 500 * time.Millisecond, cpuSys: 100 * time.Millisecond,
		haveRSS: true, rss: 128 << 20, havePeakRSS: true, peakRSS: 256 << 20,
		haveIO: true, ioRead: 10 << 20, ioWrite: 2 << 20,
	}, clock, 32, 64<<20)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSnapshotSink = s.Snapshot()
	}
}

// BenchmarkSnapshotReport measures formatting the human-readable exit summary string
// from a folded Snapshot, including kernel/agent halves, host context, and GPU.
func BenchmarkSnapshotReport(b *testing.B) {
	snap := Snapshot{
		Elapsed: 120 * time.Second,
		Samples: 60,
		Kernel: Half{
			CPUUser: 25 * time.Second, CPUSys: 5 * time.Second, HaveCPU: true,
			RSSBytes: 256 << 20, HaveRSS: true, PeakRSSBytes: 384 << 20, HavePeakRSS: true,
			IOReadBytes: 20 << 20, IOWriteBytes: 5 << 20, HaveIO: true,
			NetRxBytes: 15 << 20, NetTxBytes: 3 << 20, HaveNet: true,
		},
		Agent: Half{
			CPUUser: 12 * time.Second, CPUSys: 2 * time.Second, HaveCPU: true,
			RSSBytes: 128 << 20, HaveRSS: true, PeakRSSBytes: 192 << 20, HavePeakRSS: true,
		},
		KernelCPUPercentPeak: 85.0, HaveKernelCPUPeak: true,
		GoroutinesPeak: 48,
		GoHeapSysBytes: 96 << 20,
		NumCPU:         16,
		GOMAXPROCS:     16,
		Host: Host{
			TotalRAMBytes: 64 << 30,
			AvailRAMBytes: 42 << 30,
			HaveRAM:       true,
			Load1:         1.45,
			HaveLoad:      true,
		},
		GPUVRAMUsedBytes:  4 << 30,
		GPUVRAMTotalBytes: 16 << 30,
		HaveGPU:           true,
		GPUUtilPercent:    62.0,
		HaveGPUUtil:       true,
		GuardStops: GuardStopCounts{
			OperatorDirected: 2,
			FailOpen:         0,
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = snap.Report()
	}
}

// BenchmarkSnapshotPrometheusText measures rendering the Prometheus /metrics exposition
// format lines for all observed harness axes.
func BenchmarkSnapshotPrometheusText(b *testing.B) {
	snap := Snapshot{
		Elapsed: 120 * time.Second,
		Samples: 60,
		Kernel: Half{
			CPUUser: 25 * time.Second, CPUSys: 5 * time.Second, HaveCPU: true,
			RSSBytes: 256 << 20, HaveRSS: true, PeakRSSBytes: 384 << 20, HavePeakRSS: true,
			IOReadBytes: 20 << 20, IOWriteBytes: 5 << 20, HaveIO: true,
			NetRxBytes: 15 << 20, NetTxBytes: 3 << 20, HaveNet: true,
		},
		Agent: Half{
			CPUUser: 12 * time.Second, CPUSys: 2 * time.Second, HaveCPU: true,
			RSSBytes: 128 << 20, HaveRSS: true, PeakRSSBytes: 192 << 20, HavePeakRSS: true,
		},
		KernelCPUPercentPeak: 85.0, HaveKernelCPUPeak: true,
		GoroutinesPeak: 48,
		GoHeapSysBytes: 96 << 20,
		NumCPU:         16,
		Host: Host{
			TotalRAMBytes: 64 << 30,
			AvailRAMBytes: 42 << 30,
			HaveRAM:       true,
			Load1:         1.45,
			HaveLoad:      true,
		},
		GPUVRAMUsedBytes:  4 << 30,
		GPUVRAMTotalBytes: 16 << 30,
		HaveGPU:           true,
		GPUUtilPercent:    62.0,
		HaveGPUUtil:       true,
		GuardStops: GuardStopCounts{
			OperatorDirected: 1,
			FailOpen:         0,
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = snap.PrometheusText()
	}
}

// BenchmarkSnapshotMarshalLedgerRow measures JSON encoding of a durable session ledger row.
func BenchmarkSnapshotMarshalLedgerRow(b *testing.B) {
	snap := Snapshot{
		Elapsed: 120 * time.Second,
		Samples: 60,
		Kernel: Half{
			CPUUser: 25 * time.Second, CPUSys: 5 * time.Second, HaveCPU: true,
			RSSBytes: 256 << 20, HaveRSS: true, PeakRSSBytes: 384 << 20, HavePeakRSS: true,
			IOReadBytes: 20 << 20, IOWriteBytes: 5 << 20, HaveIO: true,
			NetRxBytes: 15 << 20, NetTxBytes: 3 << 20, HaveNet: true,
		},
		Agent: Half{
			CPUUser: 12 * time.Second, CPUSys: 2 * time.Second, HaveCPU: true,
			RSSBytes: 128 << 20, HaveRSS: true, PeakRSSBytes: 192 << 20, HavePeakRSS: true,
		},
		KernelCPUPercentPeak: 85.0, HaveKernelCPUPeak: true,
		GoroutinesPeak: 48,
		GoHeapSysBytes: 96 << 20,
		NumCPU:         16,
		GOMAXPROCS:     16,
		Host: Host{
			TotalRAMBytes: 64 << 30,
			AvailRAMBytes: 42 << 30,
			HaveRAM:       true,
			Load1:         1.45,
			HaveLoad:      true,
		},
		GPUVRAMUsedBytes:  4 << 30,
		GPUVRAMTotalBytes: 16 << 30,
		HaveGPU:           true,
		GPUUtilPercent:    62.0,
		HaveGPUUtil:       true,
		GuardStops: GuardStopCounts{
			OperatorDirected: 1,
			FailOpen:         0,
		},
	}
	now := time.Unix(1700000000, 0)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		raw, err := snap.MarshalLedgerRow("guard", "anthropic", "claude-code", now)
		if err != nil {
			b.Fatal(err)
		}
		benchBytesSink = raw
	}
}

// BenchmarkTurnFragment measures the compact per-turn resource fragment rendering
// executed on every turn for the guard's debug-stats line.
func BenchmarkTurnFragment(b *testing.B) {
	snap := Snapshot{
		Elapsed: 45 * time.Second,
		Kernel: Half{
			CPUUser: 4 * time.Second, CPUSys: time.Second, HaveCPU: true,
			RSSBytes: 192 << 20, HaveRSS: true,
			IOReadBytes: 12 << 20, IOWriteBytes: 3 << 20, HaveIO: true,
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = snap.TurnFragment()
	}
}

// BenchmarkDensityGovernorUpdate measures the Vegas/TCP BBR feedback loop update
// step evaluating host memory headroom, swap activity, and PSI stalls.
func BenchmarkDensityGovernorUpdate(b *testing.B) {
	gov := NewDensityGovernor(DefaultGovernorConfig())
	samples := []HostSample{
		{
			TotalRAMBytes: 64 * 1024 * 1024 * 1024,
			AvailRAMBytes: 48 * 1024 * 1024 * 1024,
			HaveRAM:       true,
			HavePSI:       true,
			PSIAvg10:      0.5,
			PSISomeTotal:  10 * time.Millisecond,
		},
		{
			TotalRAMBytes: 64 * 1024 * 1024 * 1024,
			AvailRAMBytes: 32 * 1024 * 1024 * 1024,
			HaveRAM:       true,
			HavePSI:       true,
			PSIAvg10:      2.0,
			PSISomeTotal:  25 * time.Millisecond,
		},
		{
			TotalRAMBytes: 64 * 1024 * 1024 * 1024,
			AvailRAMBytes: 12 * 1024 * 1024 * 1024,
			HaveRAM:       true,
			HavePSI:       true,
			PSIAvg10:      8.0,
			PSISomeTotal:  80 * time.Millisecond,
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := samples[i%len(samples)]
		s.Timestamp = time.Now()
		benchIntSink = gov.Update(s)
	}
}

// BenchmarkDensityGovernorShouldAdmit measures the worker admission gate decision.
func BenchmarkDensityGovernorShouldAdmit(b *testing.B) {
	gov := NewDensityGovernor(DefaultGovernorConfig())
	gov.Update(HostSample{
		TotalRAMBytes: 32 * 1024 * 1024 * 1024,
		AvailRAMBytes: 20 * 1024 * 1024 * 1024,
		HaveRAM:       true,
	})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		admit, _ := gov.ShouldAdmit()
		benchBoolSink = admit
	}
}

// BenchmarkDensityGovernorResolveConcurrency measures concurrency flag/argument resolution.
func BenchmarkDensityGovernorResolveConcurrency(b *testing.B) {
	gov := NewDensityGovernor(DefaultGovernorConfig())
	args := []string{"auto", "dynamic", "16", "32", "invalid", ""}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchIntSink = gov.ResolveConcurrency(args[i%len(args)])
	}
}

// BenchmarkWalkFleet measures the tree traversal over a whole-host census, isolating
// the fak-owned subtree while filtering unrelated processes and impossible recycled PIDs.
func BenchmarkWalkFleet(b *testing.B) {
	census := makeBenchmarkCensus(120)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchProcsSink = WalkFleet(census)
	}
}

// BenchmarkFoldFleet measures folding sampled fleet processes into class rollups
// and computing host-capacity ratios.
func BenchmarkFoldFleet(b *testing.B) {
	procs := makeBenchmarkFleetProcs(60)
	host := Host{
		TotalRAMBytes: 64 << 30,
		AvailRAMBytes: 36 << 30,
		HaveRAM:       true,
		Load1:         2.1,
		HaveLoad:      true,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchRollupSink = FoldFleet(procs, host, 16)
	}
}

// BenchmarkFleetReport measures formatting the human multi-line fleet rollup report.
func BenchmarkFleetReport(b *testing.B) {
	procs := makeBenchmarkFleetProcs(60)
	host := Host{
		TotalRAMBytes: 64 << 30,
		AvailRAMBytes: 36 << 30,
		HaveRAM:       true,
		Load1:         2.1,
		HaveLoad:      true,
	}
	rollup := FoldFleet(procs, host, 16)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = rollup.Report()
	}
}

// BenchmarkFleetMarshalLedgerRow measures JSON encoding of the durable fleet rollup row.
func BenchmarkFleetMarshalLedgerRow(b *testing.B) {
	procs := makeBenchmarkFleetProcs(60)
	host := Host{
		TotalRAMBytes: 64 << 30,
		AvailRAMBytes: 36 << 30,
		HaveRAM:       true,
		Load1:         2.1,
		HaveLoad:      true,
	}
	rollup := FoldFleet(procs, host, 16)
	now := time.Unix(1700000000, 0)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		raw, err := rollup.MarshalLedgerRow(now)
		if err != nil {
			b.Fatal(err)
		}
		benchBytesSink = raw
	}
}

func makeBenchmarkCensus(total int) []ProcRef {
	census := make([]ProcRef, 0, total)
	pid := 1000
	for seat := 0; seat < total/6; seat++ {
		guardPID := pid
		pid++
		census = append(census, ProcRef{
			PID: guardPID, PPID: 1, Name: "fak.exe", Cmdline: fmt.Sprintf("fak guard --seat s%d", seat),
			AgeSec: 1800, HaveAge: true,
		})
		seatPID := pid
		pid++
		census = append(census, ProcRef{
			PID: seatPID, PPID: guardPID, Name: "claude", Cmdline: "claude --print",
			AgeSec: 1750, HaveAge: true,
		})
		brokerPID := pid
		pid++
		census = append(census, ProcRef{
			PID: brokerPID, PPID: seatPID, Name: "node.exe", Cmdline: "node mcp-server.js",
			AgeSec: 1700, HaveAge: true,
		})
		toolPID := pid
		pid++
		census = append(census, ProcRef{
			PID: toolPID, PPID: seatPID, Name: "git.exe", Cmdline: "git status",
			AgeSec: 10, HaveAge: true,
		})
	}
	for len(census) < total {
		curPID := pid
		pid++
		census = append(census, ProcRef{
			PID: curPID, PPID: 1, Name: "chrome.exe", Cmdline: "chrome.exe",
			AgeSec: 7200, HaveAge: true,
		})
	}
	return census
}

func makeBenchmarkFleetProcs(total int) []FleetProc {
	procs := make([]FleetProc, 0, total)
	classes := []Class{ClassSeat, ClassFak, ClassBroker, ClassOther}
	for i := 0; i < total; i++ {
		cls := classes[i%len(classes)]
		procs = append(procs, FleetProc{
			ProcRef: ProcRef{
				PID:     1000 + i,
				PPID:    1000 + (i / 4),
				Name:    fmt.Sprintf("proc-%d", i),
				AgeSec:  600 + i*10,
				HaveAge: true,
			},
			Class:        cls,
			Read:         true,
			CPUSeconds:   float64(5 + i%20),
			HaveCPU:      true,
			RSSBytes:     uint64(64+i%128) << 20,
			HaveRSS:      true,
			PrivateBytes: uint64(48+i%96) << 20,
			HavePrivate:  true,
		})
	}
	return procs
}

// TestBenchmarksRun verifies that benchmark functions execute properly in a standard
// test run when not in -short mode.
func TestBenchmarksRun(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping benchmark runner in -short mode")
	}

	res := testing.Benchmark(BenchmarkSamplerFoldProc)
	if res.N == 0 {
		t.Fatal("BenchmarkSamplerFoldProc produced 0 iterations")
	}
	resGov := testing.Benchmark(BenchmarkDensityGovernorUpdate)
	if resGov.N == 0 {
		t.Fatal("BenchmarkDensityGovernorUpdate produced 0 iterations")
	}
}
