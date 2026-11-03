package perfledger

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// benchWindow builds a deterministic 500-row serving window covering every
// summary axis: served turns with varying token counts (some above
// MinPrefillRateTokens so PrefillTPS is set), synthetic probes, stall
// failures, native rows across all three decode paths with kernel tables,
// proxied upstream-spec rows, and clients/identities from the closed
// vocabularies.
func benchWindow() []Record {
	const total = 500
	now := time.Date(2026, 8, 14, 12, 0, 0, 0, time.UTC)
	recs := make([]Record, 0, total)
	for i := 0; i < total; i++ {
		ts := now.Add(time.Duration(i) * time.Minute)
		switch {
		case i%50 == 7:
			probe := NewRecord(ts, "stop", LocalitySelfHosted, 16, 1, 0, 40*time.Millisecond, 30*time.Millisecond)
			probe.Synthetic = true
			recs = append(recs, probe)
		case i%50 == 13:
			recs = append(recs, NewFailureRecord(ts, LocalitySelfHosted, ErrorStall, 504, 500*time.Millisecond, 0))
		case i%50 == 19:
			upstream := NewRecord(ts, "stop", LocalityVendor, 512, 128, 256, 2000*time.Millisecond, 300*time.Millisecond)
			upstream.UpstreamSpecDraftTokens = 64
			upstream.UpstreamSpecAcceptedTokens = 48
			recs = append(recs, upstream)
		default:
			prompt := 64 + (i%7)*96
			compl := 32 + (i%11)*24
			cached := 0
			if i%3 == 0 {
				cached = prompt / 2
			}
			dur := time.Duration(500+i*13) * time.Millisecond
			ttft := time.Duration(80+i%50) * time.Millisecond
			r := NewRecord(ts, "stop", LocalitySelfHosted, prompt, compl, cached, dur, ttft)
			if i%25 == 0 {
				path := Paths[(i/25)%len(Paths)]
				eng := &Engine{Path: path, CohortSize: 1 + i%4}
				if path == PathSpeculative {
					eng.SpecRounds = 8
					eng.SpecDraftTokens = 40
					eng.SpecAcceptedTokens = 30
				}
				nt := NewNativeTiming(prompt, compl-1, float64(ttft.Milliseconds()), float64((dur - ttft).Milliseconds()))
				if k := NewNativeKernels(KernelWindow{
					Backend:      "metal",
					Steps:        compl - 1,
					GPUNS:        2500 * float64(compl-1),
					WindowWallNS: 4000 * float64(compl-1),
					Samples: []KernelSample{
						{Kernel: "matmul_q8", Op: "gemm", GX: 1, GY: 1, Calls: uint64(compl - 1), NS: 1200},
						{Kernel: "attn_flash", Op: "attention", GX: 1, GY: 1, Calls: uint64(compl - 1), NS: 900},
					},
				}); k != nil {
					nt = nt.WithKernels(k)
				}
				r.Model = "qwen38-bench"
				r.Engine = eng
				r.NativeTiming = nt
			}
			if i%5 == 0 {
				r.Client = Clients[(i/5)%len(Clients)]
				r.Identity = Identity{Planner: "proxy", Backend: "native", Host: "bench-host", Version: "v0.0.0-bench"}
			}
			recs = append(recs, r)
		}
	}
	return recs
}

// The quantile fold behind `fak perf`'s summary line.
func BenchmarkSummarize(b *testing.B) {
	recs := benchWindow()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := Summarize(recs)
		if s.Count <= 0 {
			b.Fatalf("count = %d, want > 0", s.Count)
		}
	}
}

// What `fak perf` assembles before rendering: the recent-window trim plus the
// summary fold.
func BenchmarkBuildReport(b *testing.B) {
	recs := benchWindow()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep := BuildReport(recs, 0, true, 0)
		if rep.Window.Requests <= 0 {
			b.Fatalf("requests = %d, want > 0", rep.Window.Requests)
		}
	}
}

// The one-line agent-facing readout printed by `fak perf`.
func BenchmarkRenderCompact(b *testing.B) {
	rep := BuildReport(benchWindow(), 0, false, 0)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		line := RenderCompact(rep)
		if line == "" {
			b.Fatal("empty compact render")
		}
	}
}

// The hot read path: `fak perf` seeds its window from the ledger tail, with
// the server up or down.
func BenchmarkReadTail(b *testing.B) {
	dir := b.TempDir()
	var buf strings.Builder
	for i := 0; i < RingCap; i++ {
		line, err := json.Marshal(Record{Schema: Schema, UnixMS: int64(i), E2EMS: 1})
		if err != nil {
			b.Fatal(err)
		}
		buf.Write(line)
		buf.WriteByte('\n')
	}
	path := filepath.Join(dir, "perf.jsonl")
	if err := os.WriteFile(path, []byte(buf.String()), 0o644); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		recs, _ := ReadTail(path)
		if len(recs) != RingCap {
			b.Fatalf("got %d rows, want %d", len(recs), RingCap)
		}
	}
}
