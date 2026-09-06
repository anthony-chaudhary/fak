package stallscan

import (
	"fmt"
	"testing"
	"time"
)

var (
	benchVerdict Verdict
	benchAdvice  RebootAdvice
	benchArming  Arming
	benchTopIO   []ProcIO
)

func BenchmarkClassify_Calm(b *testing.B) {
	s := Sample{
		TotalFaultsPerSec:     22000,
		HardFaultsPerSec:      10,
		ContextSwitchesPerSec: 20000,
		SystemCallsPerSec:     45000,
		AvailableMB:           16384,
		DiskQueueLen:          0.1,
		ProcessorCount:        16,
		CPUPercent:            15.0,
		ProcessorQueueLength:  1.0,
		TopCPU: []ProcCPU{
			{PID: 1001, Name: "fak.exe", Percent: 8.5},
			{PID: 1002, Name: "go.exe", Percent: 4.2},
		},
		TopIO: []ProcIO{
			{PID: 1001, Name: "fak.exe", Ops: 120.0},
			{PID: 1002, Name: "go.exe", Ops: 45.0},
		},
		TopHandles: []ProcHandles{
			{PID: 1001, Name: "fak.exe", Handles: 450},
			{PID: 1002, Name: "pwsh.exe", Handles: 800},
		},
		TopThreads: []ProcThreads{
			{PID: 1001, Name: "fak.exe", Threads: 24},
			{PID: 1002, Name: "pwsh.exe", Threads: 18},
		},
	}
	t := DefaultThresholds()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchVerdict = Classify(s, t)
	}
}

func BenchmarkClassify_SoftFaultChurn(b *testing.B) {
	s := Sample{
		TotalFaultsPerSec:      489505,
		HardFaultsPerSec:       508,
		DemandZeroFaultsPerSec: 192374,
		TransitionFaultsPerSec: 136422,
		ContextSwitchesPerSec:  24000,
		AvailableMB:            229257,
		DiskQueueLen:           0,
		ProcessorCount:         16,
	}
	t := DefaultThresholds()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchVerdict = Classify(s, t)
	}
}

func BenchmarkClassify_SpawnStormRate(b *testing.B) {
	s := Sample{
		TotalFaultsPerSec:     120000,
		HardFaultsPerSec:      50,
		ContextSwitchesPerSec: 106000,
		SystemCallsPerSec:     733000,
		SpawnBurst:            400,
		SpawnWindowSeconds:    2.0,
		AvailableMB:           229000,
		ProcessorCount:        16,
	}
	t := DefaultThresholds()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchVerdict = Classify(s, t)
	}
}

func BenchmarkClassify_CPUSaturation(b *testing.B) {
	s := Sample{
		CPUPercent:           96.5,
		ProcessorQueueLength: 16.0,
		ProcessorCount:       16,
		AvailableMB:          16384,
		DiskQueueLen:         0.5,
		TopCPU: []ProcCPU{
			{PID: 1201, Name: "compiler.exe", Percent: 52.0},
			{PID: 1202, Name: "linker.exe", Percent: 35.0},
		},
	}
	t := DefaultThresholds()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchVerdict = Classify(s, t)
	}
}

func BenchmarkClassify_ScalingCensus(b *testing.B) {
	for _, count := range []int{10, 50, 250, 1000} {
		b.Run(fmt.Sprintf("%d_procs", count), func(b *testing.B) {
			handles := make([]ProcHandles, count)
			threads := make([]ProcThreads, count)
			io := make([]ProcIO, count)
			cpu := make([]ProcCPU, count)
			for j := 0; j < count; j++ {
				pid := 1000 + j
				name := fmt.Sprintf("worker_%d.exe", j%20)
				handles[j] = ProcHandles{PID: pid, Name: name, Handles: 200 + (j*7)%15000}
				threads[j] = ProcThreads{PID: pid, Name: name, Threads: 10 + (j*3)%800}
				io[j] = ProcIO{PID: pid, Name: name, Ops: float64(10 + (j*17)%500)}
				cpu[j] = ProcCPU{PID: pid, Name: name, Percent: float64((j * 5) % 80)}
			}
			s := Sample{
				TotalFaultsPerSec:    35000,
				AvailableMB:          32768,
				ProcessorCount:       16,
				CPUPercent:           45.0,
				ProcessorQueueLength: 2.0,
				TopHandles:           handles,
				TopThreads:           threads,
				TopIO:                io,
				TopCPU:               cpu,
			}
			t := DefaultThresholds()

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchVerdict = Classify(s, t)
			}
		})
	}
}

func BenchmarkClassifyWithBaseline(b *testing.B) {
	for _, count := range []int{20, 100, 500} {
		b.Run(fmt.Sprintf("%d_procs", count), func(b *testing.B) {
			baseHandles := make([]ProcHandles, count)
			curHandles := make([]ProcHandles, count)
			baseThreads := make([]ProcThreads, count)
			curThreads := make([]ProcThreads, count)

			for j := 0; j < count; j++ {
				pid := 2000 + j
				name := fmt.Sprintf("proc_%d.exe", j)
				baseHandles[j] = ProcHandles{PID: pid, Name: name, Handles: 1000 + j*10}
				baseThreads[j] = ProcThreads{PID: pid, Name: name, Threads: 50 + j*2}

				climbH := 50
				climbT := 10
				if j == count-1 {
					climbH = 2500
					climbT = 300
				}
				curHandles[j] = ProcHandles{PID: pid, Name: name, Handles: baseHandles[j].Handles + climbH}
				curThreads[j] = ProcThreads{PID: pid, Name: name, Threads: baseThreads[j].Threads + climbT}
			}

			baseSample := Sample{TopHandles: baseHandles, TopThreads: baseThreads}
			curSample := Sample{
				TotalFaultsPerSec: 15000,
				AvailableMB:       16384,
				ProcessorCount:    16,
				TopHandles:        curHandles,
				TopThreads:        curThreads,
			}
			t := DefaultThresholds()

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchVerdict = ClassifyWithBaseline(baseSample, curSample, t)
			}
		})
	}
}

func BenchmarkAdviseReboot(b *testing.B) {
	b.Run("Calm", func(b *testing.B) {
		s := Sample{
			TopHandles: []ProcHandles{
				{PID: 3001, Name: "WindowsTerminal.exe", Handles: 12000},
				{PID: 3002, Name: "node.exe", Handles: 4000},
			},
			TopThreads: []ProcThreads{
				{PID: 3001, Name: "WindowsTerminal.exe", Threads: 800},
				{PID: 3002, Name: "node.exe", Threads: 120},
			},
		}
		t := DefaultRebootThresholds()

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchAdvice = AdviseReboot(s, t)
		}
	})

	b.Run("MultipleCrossers", func(b *testing.B) {
		s := Sample{
			TopHandles: []ProcHandles{
				{PID: 3101, Name: "WindowsTerminal.exe", Handles: 35000},
				{PID: 3102, Name: "svchost.exe", Handles: 32000},
				{PID: 3103, Name: "fak.exe", Handles: 5000},
			},
			TopThreads: []ProcThreads{
				{PID: 3101, Name: "WindowsTerminal.exe", Threads: 2800},
				{PID: 3104, Name: "conhost.exe", Threads: 2400},
				{PID: 3102, Name: "svchost.exe", Threads: 1200},
			},
		}
		t := DefaultRebootThresholds()

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchAdvice = AdviseReboot(s, t)
		}
	})
}

func BenchmarkClassifyArming(b *testing.B) {
	now := time.Date(2026, 9, 6, 12, 0, 0, 0, time.UTC)
	freshness := 30 * time.Second

	b.Run("ArmedWithRate", func(b *testing.B) {
		r := LedgerRead{
			Found:              true,
			Parsed:             true,
			Timestamp:          now.Add(-5 * time.Second),
			SpawnBurst:         42,
			SpawnWindowSeconds: 2.0,
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchArming = ClassifyArming(r, now, freshness, true)
		}
	})

	b.Run("Stale", func(b *testing.B) {
		r := LedgerRead{
			Found:              true,
			Parsed:             true,
			Timestamp:          now.Add(-60 * time.Second),
			SpawnBurst:         120,
			SpawnWindowSeconds: 2.0,
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchArming = ClassifyArming(r, now, freshness, true)
		}
	})
}

func BenchmarkSortTopIO(b *testing.B) {
	for _, count := range []int{20, 100, 500} {
		b.Run(fmt.Sprintf("%d_ops", count), func(b *testing.B) {
			in := make([]ProcIO, count)
			for j := 0; j < count; j++ {
				in[j] = ProcIO{
					PID:  4000 + j,
					Name: fmt.Sprintf("io_proc_%d.exe", j),
					Ops:  float64((j * 31) % 10000),
				}
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchTopIO = SortTopIO(in, 10)
			}
		})
	}
}
