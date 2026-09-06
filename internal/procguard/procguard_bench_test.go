package procguard

import (
	"fmt"
	"strconv"
	"strings"
	"testing"
)

// Global sinks to prevent compiler dead-code elimination.
var (
	benchProcSink       []Proc
	benchTopologySink   RelationTopology
	benchSprawlSink     ObservationSummary
	benchHeadroomSink   SystemCommitHeadroom
	benchFindingSink    []Finding
	benchPayloadSink    Payload
	benchCPUMapSink     map[int]float64
	benchDurationSink   float64
	benchChildCountSink map[int]int
	benchChildNameSink  map[int][]string
	benchStringSink     string
	benchStreakSink     map[string]int
)

func benchFloatPtr(f float64) *float64 { return &f }

func makeSyntheticCensusText(n int, dialect string) string {
	var sb strings.Builder
	for i := 1; i <= n; i++ {
		pid := 1000 + i
		rssKB := 1024 * (i%64 + 1)
		if dialect == "darwin" {
			comm := fmt.Sprintf("/usr/local/bin/worker_%d", i%16)
			dur := fmt.Sprintf("%02d:%02d.%02d", (i/60)%60, i%60, (i*17)%100)
			if i%10 == 0 {
				dur = fmt.Sprintf("1-%02d:%02d:%02d", (i/3600)%24, (i/60)%60, i%60)
			}
			sb.WriteString(fmt.Sprintf(" %6d %8d %12s %s\n", pid, rssKB, dur, comm))
		} else {
			name := fmt.Sprintf("worker_%d", i%16)
			threads := (i % 32) + 1
			if i%25 == 0 {
				threads = 120000 // simulate runaway thread count
			}
			cpuSec := i * 3
			sb.WriteString(fmt.Sprintf(" %6d %6d %8d %8d %s\n", pid, threads, rssKB, cpuSec, name))
		}
	}
	return sb.String()
}

func makeSyntheticRelationsText(n int, dialect string) string {
	var sb strings.Builder
	for i := 1; i <= n; i++ {
		pid := 1000 + i
		ppid := 1000 + (i % 20)
		if i <= 20 {
			ppid = 1 // root ancestors
		}
		if dialect == "darwin" {
			comm := fmt.Sprintf("/usr/local/bin/worker_%d", i%16)
			dur := fmt.Sprintf("%02d:%02d", (i/60)%60, i%60)
			cmdline := fmt.Sprintf("worker_%d --session s-%d --lane l-%d", i%16, i, i%8)
			sb.WriteString(fmt.Sprintf(" %6d %6d %10s %s %s\n", pid, ppid, dur, comm, cmdline))
		} else {
			name := fmt.Sprintf("worker_%d", i%16)
			ageSec := i * 15
			cmdline := fmt.Sprintf("worker_%d --session s-%d --lane l-%d", i%16, i, i%8)
			sb.WriteString(fmt.Sprintf(" %6d %6d %8d %s %s\n", pid, ppid, ageSec, name, cmdline))
		}
	}
	return sb.String()
}

func makeSyntheticProcSlice(n int) []Proc {
	procs := make([]Proc, n)
	names := []string{"pwsh", "bash", "sh", "cmd", "conhost", "python", "node", "git", "go", "llama-cli", "System"}
	for i := 0; i < n; i++ {
		pid := 2000 + i
		ppid := 2000 + (i % 25)
		if i < 25 {
			ppid = 1
		}
		name := names[i%len(names)]
		threads := 4 + (i % 16)
		handles := 100 + (i * 10)
		wsmb := 50 + (i * 2)
		var cpuPct *float64

		if i%30 == 0 {
			// runaway process
			threads = 99999
			handles = 150000
			wsmb = 45000
		}
		if i%20 == 0 {
			c := 98.5
			cpuPct = &c
		}

		cmdline := fmt.Sprintf("%s --run run-%04d --task %d", name, i%50, i)
		if i%15 == 0 {
			cmdline = fmt.Sprintf("fak loop drive --run run-%04d", i%50)
		} else if i%12 == 0 {
			cmdline = "python -m dos_mcp.server"
		}

		age := i * 20
		procs[i] = Proc{
			PID:     pid,
			PPID:    IntPtr(ppid),
			Name:    name,
			Threads: IntPtr(threads),
			Handles: IntPtr(handles),
			WSMB:    IntPtr(wsmb),
			CPUPct:  cpuPct,
			Cmdline: cmdline,
			AgeSec:  IntPtr(age),
			Start:   fmt.Sprintf("2026-09-05T12:%02d:%02dZ", (i/60)%60, i%60),
		}
	}
	return procs
}

// ============================================================================
// 1. Process Scanning Benchmarks
// ============================================================================

func BenchmarkProcessScanning(b *testing.B) {
	b.Run("CensusParse", func(b *testing.B) {
		dialects := []string{"linux", "darwin"}
		sizes := []int{10, 100, 1000}
		for _, d := range dialects {
			spec := psCensusSpec(d)
			for _, n := range sizes {
				text := makeSyntheticCensusText(n, d)
				b.Run(fmt.Sprintf("%s/%d_procs", d, n), func(b *testing.B) {
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						procs, secs := parsePSCensus(text, spec)
						benchProcSink = procs
						benchCPUMapSink = secs
					}
					b.StopTimer()
					procsPerSec := float64(n*b.N) / b.Elapsed().Seconds()
					b.ReportMetric(procsPerSec, "procs/s")
				})
			}
		}
	})

	b.Run("RelationsParse", func(b *testing.B) {
		dialects := []string{"linux", "darwin"}
		sizes := []int{10, 100, 500}
		for _, d := range dialects {
			spec := psRelationSpec(d)
			for _, n := range sizes {
				text := makeSyntheticRelationsText(n, d)
				b.Run(fmt.Sprintf("%s/%d_procs", d, n), func(b *testing.B) {
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						procs := parsePSRelations(text, spec)
						benchProcSink = procs
					}
					b.StopTimer()
					procsPerSec := float64(n*b.N) / b.Elapsed().Seconds()
					b.ReportMetric(procsPerSec, "procs/s")
				})
			}
		}
	})

	b.Run("DurationParse", func(b *testing.B) {
		samples := []string{
			"0", "93784", "0:04", "12:31", "0:04.21", "01:02:03", "1-02:03:04", "10-00:00:00",
			" 3:20 ", "", "-", "?", "ELAPSED", "NaN", "Inf", "12:",
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s := samples[i%len(samples)]
			dur, _ := parsePSDuration(s)
			benchDurationSink = dur
		}
	})

	b.Run("TopologyBuild", func(b *testing.B) {
		sizes := []int{50, 250, 1000}
		for _, n := range sizes {
			procs := makeSyntheticProcSlice(n)
			b.Run(fmt.Sprintf("%d_procs", n), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					top := NewRelationTopology(procs)
					benchTopologySink = top
				}
				b.StopTimer()
				procsPerSec := float64(n*b.N) / b.Elapsed().Seconds()
				b.ReportMetric(procsPerSec, "procs/s")
			})
		}
	})

	b.Run("ObserveSprawl", func(b *testing.B) {
		sizes := []int{50, 250, 1000}
		opt := DefaultOrphanOptions()
		for _, n := range sizes {
			procs := makeSyntheticProcSlice(n)
			top := NewRelationTopology(procs)
			b.Run(fmt.Sprintf("%d_procs", n), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					summary := ObserveSprawl(procs, top, opt)
					benchSprawlSink = summary
				}
				b.StopTimer()
				procsPerSec := float64(n*b.N) / b.Elapsed().Seconds()
				b.ReportMetric(procsPerSec, "procs/s")
			})
		}
	})
}

// ============================================================================
// 2. Boundary Checks Benchmarks
// ============================================================================

func BenchmarkBoundaryChecks(b *testing.B) {
	b.Run("EvaluateCommitHeadroom", func(b *testing.B) {
		scenarios := []struct {
			name     string
			snapshot MemorySnapshot
			required uint64
		}{
			{
				name: "admitted_above_boundary",
				snapshot: MemorySnapshot{
					Metric:                     MemoryMetricCommit,
					SystemBytes:                20 << 30,
					SystemLimit:                64 << 30,
					HostPhysicalBytes:          32 << 30,
					HostPhysicalAvailableBytes: 16 << 30,
				},
				required: 16 << 30,
			},
			{
				name: "exact_boundary_refusal",
				snapshot: MemorySnapshot{
					Metric:                     MemoryMetricCommit,
					SystemBytes:                48 << 30,
					SystemLimit:                64 << 30,
					HostPhysicalBytes:          32 << 30,
					HostPhysicalAvailableBytes: 16 << 30,
				},
				required: 16 << 30,
			},
			{
				name: "below_boundary_refusal",
				snapshot: MemorySnapshot{
					Metric:                     MemoryMetricCommit,
					SystemBytes:                55 << 30,
					SystemLimit:                64 << 30,
					HostPhysicalBytes:          32 << 30,
					HostPhysicalAvailableBytes: 8 << 30,
				},
				required: 16 << 30,
			},
			{
				name: "unsupported_metric_abstain",
				snapshot: MemorySnapshot{
					Metric:      MemoryMetricRSS,
					SystemBytes: 55 << 30,
					SystemLimit: 64 << 30,
				},
				required: 16 << 30,
			},
			{
				name: "zero_system_limit_abstain",
				snapshot: MemorySnapshot{
					Metric:      MemoryMetricCommit,
					SystemBytes: 20 << 30,
					SystemLimit: 0,
				},
				required: 16 << 30,
			},
		}

		for _, sc := range scenarios {
			b.Run(sc.name, func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					res := EvaluateSystemCommitHeadroom(sc.snapshot, sc.required)
					benchHeadroomSink = res
				}
			})
		}
	})

	b.Run("RequiredCommitHeadroom", func(b *testing.B) {
		getters := []struct {
			name   string
			getenv func(string) string
		}{
			{"default_fallback", func(string) string { return "" }},
			{"valid_override", func(string) string { return "8192" }},
			{"malformed_fallback", func(string) string { return "invalid-123mb" }},
			{"overflow_fallback", func(string) string { return "18446744073709551615" }},
		}
		for _, g := range getters {
			b.Run(g.name, func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					req := RequiredSystemCommitHeadroom(g.getenv)
					_ = req
				}
			})
		}
	})

	b.Run("CPUPctDelta", func(b *testing.B) {
		b.Run("valid_delta", func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				pct, ok := CPUPctDelta(1.0, 3.0, 2.0, true, true)
				if !ok || pct != 100.0 {
					b.Fatalf("unexpected delta: %v, %v", pct, ok)
				}
			}
		})

		b.Run("backward_counter_skipped", func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, ok := CPUPctDelta(10.0, 2.0, 1.0, true, true)
				if ok {
					b.Fatal("backward counter must be rejected")
				}
			}
		})
	})

	b.Run("CPUPctSustained", func(b *testing.B) {
		sizes := []int{50, 250, 1000}
		for _, n := range sizes {
			s1 := make(map[int]float64, n)
			s2 := make(map[int]float64, n)
			s3 := make(map[int]float64, n)
			for i := 1; i <= n; i++ {
				base := float64(i * 10)
				s1[i] = base
				if i%10 == 0 {
					s2[i] = base + 1.0 // 100% core pin
					s3[i] = base + 2.0 // 100% core pin
				} else {
					s2[i] = base + 0.1 // 10%
					s3[i] = base + 0.1 // 0%
				}
			}
			samples := []map[int]float64{s1, s2, s3}
			b.Run(fmt.Sprintf("%d_procs", n), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					sustained := CPUPctSustained(samples, 1.0)
					benchCPUMapSink = sustained
				}
			})
		}
	})

	b.Run("StreakKeyAndBumping", func(b *testing.B) {
		keys := make([]string, 100)
		prev := make(map[string]int, 100)
		for i := 0; i < 100; i++ {
			k := StreakKey(5000+i, "2026-09-05T10:00:00Z")
			keys[i] = k
			if i%2 == 0 {
				prev[k] = i
			}
		}

		b.Run("StreakKeyFormat", func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				k := StreakKey(5000+(i%100), "2026-09-05T10:00:00Z")
				benchStringSink = k
			}
		})

		b.Run("BumpStreaks", func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				bumped := bumpStreaks(prev, keys)
				benchStreakSink = bumped
			}
		})
	})
}

// ============================================================================
// 3. Tree Audits Benchmarks
// ============================================================================

func BenchmarkTreeAudits(b *testing.B) {
	sizes := []int{50, 250, 1000}

	b.Run("ChildCounts", func(b *testing.B) {
		for _, n := range sizes {
			procs := makeSyntheticProcSlice(n)
			b.Run(fmt.Sprintf("%d_nodes", n), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					counts := ChildCounts(procs)
					benchChildCountSink = counts
				}
				b.StopTimer()
				nodesPerSec := float64(n*b.N) / b.Elapsed().Seconds()
				b.ReportMetric(nodesPerSec, "nodes/s")
			})
		}
	})

	b.Run("ChildNames", func(b *testing.B) {
		for _, n := range sizes {
			procs := makeSyntheticProcSlice(n)
			b.Run(fmt.Sprintf("%d_nodes", n), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					names := ChildNames(procs)
					benchChildNameSink = names
				}
				b.StopTimer()
				nodesPerSec := float64(n*b.N) / b.Elapsed().Seconds()
				b.ReportMetric(nodesPerSec, "nodes/s")
			})
		}
	})

	b.Run("AttendedParentAndStemAudits", func(b *testing.B) {
		procs := makeSyntheticProcSlice(250)
		top := NewRelationTopology(procs)
		interactive := DefaultInteractiveParentNames

		b.Run("attendedParent", func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				p := procs[i%len(procs)]
				att := attendedParent(p, top, interactive)
				_ = att
			}
		})

		b.Run("childNameStems", func(b *testing.B) {
			rawNames := []string{"cmd.exe", "pwsh.exe", "CONHOST.EXE", "bash", "python.EXE"}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				stems := childNameStems(rawNames)
				_ = stems
			}
		})

		b.Run("onlyConsoleChildren", func(b *testing.B) {
			conHosts := DefaultConsoleHostChildNames
			kidsAllConsole := []string{"conhost", "openconsole"}
			kidsMixed := []string{"conhost", "cmd", "pwsh"}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = onlyConsoleChildren(kidsAllConsole, conHosts)
				_ = onlyConsoleChildren(kidsMixed, conHosts)
			}
		})
	})
}

// ============================================================================
// 4. Rule Adjudication Benchmarks
// ============================================================================

func BenchmarkRuleAdjudication(b *testing.B) {
	b.Run("ClassifyResourceLevel", func(b *testing.B) {
		sizes := []int{50, 250, 1000}
		th := Thresholds{
			MaxThreads: 2000,
			MaxHandles: 50000,
			MaxWSMB:    20000,
			MaxCPUPct:  90.0,
		}
		protPIDs := []int{2001, 2004, 2010}
		allowNames := []string{"git", "pwsh"}

		for _, n := range sizes {
			procs := makeSyntheticProcSlice(n)
			b.Run(fmt.Sprintf("%d_procs", n), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					findings := Classify(procs, th, protPIDs, allowNames)
					benchFindingSink = findings
				}
				b.StopTimer()
				procsPerSec := float64(n*b.N) / b.Elapsed().Seconds()
				b.ReportMetric(procsPerSec, "procs/s")
			})
		}
	})

	b.Run("ClassifyOrphanSprawl", func(b *testing.B) {
		sizes := []int{50, 250, 1000}
		opt := DefaultOrphanOptions()
		opt.ReapIdleShells = true

		for _, n := range sizes {
			procs := makeSyntheticProcSlice(n)
			top := NewRelationTopology(procs)
			b.Run(fmt.Sprintf("%d_procs", n), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					flagged := ClassifyOrphanSprawl(procs, top, opt)
					benchFindingSink = flagged
				}
				b.StopTimer()
				procsPerSec := float64(n*b.N) / b.Elapsed().Seconds()
				b.ReportMetric(procsPerSec, "procs/s")
			})
		}
	})

	b.Run("ClassifyDeadOwnerOrphans", func(b *testing.B) {
		sizes := []int{50, 250, 1000}
		leaseAlive := func(runID string) bool {
			// Even run IDs are alive, odd are dead
			if strings.HasPrefix(runID, "run-") {
				num, _ := strconv.Atoi(runID[4:])
				return num%2 == 0
			}
			return false
		}
		opt := DeadOwnerOptions{
			LeaseAlive: leaseAlive,
		}

		for _, n := range sizes {
			procs := makeSyntheticProcSlice(n)
			top := NewRelationTopology(procs)
			b.Run(fmt.Sprintf("%d_procs", n), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					flagged := ClassifyDeadOwnerOrphans(procs, top, opt)
					benchFindingSink = flagged
				}
				b.StopTimer()
				procsPerSec := float64(n*b.N) / b.Elapsed().Seconds()
				b.ReportMetric(procsPerSec, "procs/s")
			})
		}
	})

	b.Run("BuildPayload", func(b *testing.B) {
		procs := makeSyntheticProcSlice(150)
		th := Thresholds{MaxThreads: 2000, MaxCPUPct: 90}
		killer := func(int) (bool, string) { return true, "reaped" }

		b.Run("report_only", func(b *testing.B) {
			opt := Options{
				Thresholds:     th,
				Platform:       "linux",
				CPUReapConfirm: 2,
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				payload := Build(procs, opt)
				benchPayloadSink = payload
			}
		})

		b.Run("enact_reaper", func(b *testing.B) {
			prevStreaks := map[string]int{}
			for _, p := range procs {
				if p.CPUPct != nil && *p.CPUPct > 90 {
					prevStreaks[StreakKey(p.PID, p.Start)] = 2
				}
			}
			opt := Options{
				Thresholds:     th,
				Enact:          true,
				Killer:         killer,
				CPUReapConfirm: 2,
				CPUStreaksPrev: prevStreaks,
				Platform:       "linux",
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				payload := Build(procs, opt)
				benchPayloadSink = payload
			}
		})
	})

	b.Run("NextAction", func(b *testing.B) {
		flaggedRunaway := []Finding{
			{PID: 101, Name: "llama-cli", Reasons: []string{"threads 120000 > 2000"}},
		}
		flaggedSprawl := []Finding{
			{PID: 202, Name: "python", Kind: "orphan-helper", Reasons: []string{"orphaned helper"}},
			{PID: 203, Name: "pwsh", Kind: "idle-shell", Reasons: []string{"idle launcher shell"}},
		}
		flaggedReaped := []Finding{
			{PID: 301, Name: "llama-cli", Action: "killed", Reasons: []string{"threads 120000 > 2000"}},
			{PID: 302, Name: "spin", Action: "cpu-unconfirmed", Reasons: []string{"cpu 100% sustained"}},
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = NextAction(nil, false, "")
			_ = NextAction(nil, false, "collector error")
			_ = NextAction(flaggedRunaway, false, "")
			_ = NextAction(flaggedSprawl, false, "")
			_ = NextAction(flaggedReaped, true, "")
		}
	})
}
