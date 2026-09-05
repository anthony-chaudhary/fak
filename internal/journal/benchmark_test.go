package journal

import (
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func BenchmarkEmitMemory(b *testing.B) {
	j := OpenMemory()
	ev := abi.Event{
		Kind: abi.EvDecide,
		Call: &abi.ToolCall{
			Tool:    "bash",
			TraceID: "trace-bench-memory",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"ls -la"}`)},
		},
		Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "rule"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j.Emit(ev)
	}
}

func BenchmarkEmitMemoryWithSubscriber(b *testing.B) {
	j := OpenMemory()
	ch, cancel := j.Subscribe()
	defer cancel()

	ev := abi.Event{
		Kind: abi.EvDecide,
		Call: &abi.ToolCall{
			Tool:    "bash",
			TraceID: "trace-bench-sub",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"git status"}`)},
		},
		Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "rule"},
	}

	go func() {
		for range ch {
		}
	}()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j.Emit(ev)
	}
}

func BenchmarkEmitFile(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "journal.jsonl")
	j, err := Open(path)
	if err != nil {
		b.Fatalf("Open failed: %v", err)
	}
	defer j.Close()

	ev := abi.Event{
		Kind: abi.EvDecide,
		Call: &abi.ToolCall{
			Tool:    "bash",
			TraceID: "trace-bench-file",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"go test"}`)},
		},
		Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "rule"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j.Emit(ev)
	}
}

func BenchmarkChainHash(b *testing.B) {
	row := Row{
		Seq:          42,
		TSUnixNano:   1700000000000000000,
		Kind:         "DECIDE",
		Tool:         "bash",
		TraceID:      "trace-bench-hash",
		Verdict:      "ALLOW",
		Reason:       "POLICY_ALLOW",
		By:           "adjudicator",
		Taint:        "clean",
		ArgsDigest:   "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
		ResultDigest: "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
	}
	prev := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = chainHash(prev, row)
	}
}

func BenchmarkVerifyRows(b *testing.B) {
	for _, count := range []int{50, 200, 1000} {
		b.Run(fmt.Sprintf("%d_rows", count), func(b *testing.B) {
			j := OpenMemory()
			ev := abi.Event{
				Kind: abi.EvDecide,
				Call: &abi.ToolCall{
					Tool:    "bash",
					TraceID: "trace-bench-verify",
					Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"uptime"}`)},
				},
				Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "rule"},
			}
			for i := 0; i < count; i++ {
				j.Emit(ev)
			}
			rows := j.Recent(count)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				n, err := VerifyRows(rows)
				if err != nil || n != count {
					b.Fatalf("VerifyRows failed: n=%d err=%v", n, err)
				}
			}
		})
	}
}

func BenchmarkVerifyFile(b *testing.B) {
	for _, count := range []int{50, 200, 1000} {
		b.Run(fmt.Sprintf("%d_rows", count), func(b *testing.B) {
			dir := b.TempDir()
			path := filepath.Join(dir, "journal.jsonl")
			j, err := Open(path)
			if err != nil {
				b.Fatalf("Open failed: %v", err)
			}
			ev := abi.Event{
				Kind: abi.EvDecide,
				Call: &abi.ToolCall{
					Tool:    "bash",
					TraceID: "trace-bench-disk",
					Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"date"}`)},
				},
				Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "rule"},
			}
			for i := 0; i < count; i++ {
				j.Emit(ev)
			}
			if err := j.Close(); err != nil {
				b.Fatalf("Close failed: %v", err)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				n, err := Verify(path)
				if err != nil || n != count {
					b.Fatalf("Verify failed: n=%d err=%v", n, err)
				}
			}
		})
	}
}

func BenchmarkReadRows(b *testing.B) {
	for _, count := range []int{50, 200, 1000} {
		b.Run(fmt.Sprintf("%d_rows", count), func(b *testing.B) {
			dir := b.TempDir()
			path := filepath.Join(dir, "journal.jsonl")
			j, err := Open(path)
			if err != nil {
				b.Fatalf("Open failed: %v", err)
			}
			ev := abi.Event{
				Kind: abi.EvDecide,
				Call: &abi.ToolCall{
					Tool:    "bash",
					TraceID: "trace-bench-read",
					Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"ls"}`)},
				},
				Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "rule"},
			}
			for i := 0; i < count; i++ {
				j.Emit(ev)
			}
			if err := j.Close(); err != nil {
				b.Fatalf("Close failed: %v", err)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				rows, err := ReadRows(path)
				if err != nil || len(rows) != count {
					b.Fatalf("ReadRows failed: len=%d err=%v", len(rows), err)
				}
			}
		})
	}
}

func BenchmarkReadTail(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "journal.jsonl")
	j, err := Open(path)
	if err != nil {
		b.Fatalf("Open failed: %v", err)
	}
	ev := abi.Event{
		Kind: abi.EvDecide,
		Call: &abi.ToolCall{
			Tool:    "bash",
			TraceID: "trace-bench-tail",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"ps"}`)},
		},
		Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "rule"},
	}
	for i := 0; i < 50; i++ {
		j.Emit(ev)
	}
	if _, err := j.Cut(); err != nil {
		b.Fatalf("Cut failed: %v", err)
	}
	for i := 0; i < 50; i++ {
		j.Emit(ev)
	}
	if err := j.Close(); err != nil {
		b.Fatalf("Close failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rows, om, err := ReadTail(path)
		if err != nil || len(rows) != 51 || om.SealedSegments != 1 {
			b.Fatalf("ReadTail failed: len=%d om=%+v err=%v", len(rows), om, err)
		}
	}
}

func BenchmarkVerifyForest(b *testing.B) {
	for _, branches := range []int{2, 5, 10} {
		b.Run(fmt.Sprintf("%d_branches", branches), func(b *testing.B) {
			var rows []Row
			for br := 0; br < branches; br++ {
				j := OpenMemory()
				ev := abi.Event{
					Kind: abi.EvDecide,
					Call: &abi.ToolCall{
						Tool:    "bash",
						TraceID: fmt.Sprintf("trace-branch-%d", br),
						Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"env"}`)},
					},
					Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "rule"},
				}
				for i := 0; i < 50; i++ {
					j.Emit(ev)
				}
				rows = append(rows, j.Recent(0)...)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				forest, err := VerifyForest(rows)
				if err != nil || forest.Tips != branches {
					b.Fatalf("VerifyForest failed: forest=%+v err=%v", forest, err)
				}
			}
		})
	}
}

func BenchmarkAppendSupervision(b *testing.B) {
	b.Run("crash", func(b *testing.B) {
		j := OpenMemory()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			j.AppendCrash("agent-worker", "trace-bench-crash", "OOM", 137)
		}
	})

	b.Run("child_exit", func(b *testing.B) {
		j := OpenMemory()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			j.AppendChildExit("agent-worker", "trace-bench-exit", "CLEAN", 0, time.Second, "on_finish")
		}
	})

	b.Run("config_swap", func(b *testing.B) {
		j := OpenMemory()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			j.AppendConfigSwap("policy", "overlay.json", "sha256:abc", "APPLIED", "operator_reload")
		}
	})

	b.Run("capability_grant", func(b *testing.B) {
		j := OpenMemory()
		grant := CapabilityGrantRow{
			Schema:    CapabilityGrantSchema,
			Knob:      "AllowPrefix",
			Direction: GrantDirectionWiden,
			Class:     "GATED_WIDEN",
			Old:       "deny",
			New:       "admit_and_log",
			Channel:   GrantChannelOperatorOverlay,
			Actor:     "operator",
			Source:    "manifest.json",
			Reason:    "maintenance",
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			j.AppendCapabilityGrant(grant)
		}
	})

	b.Run("restart_hop", func(b *testing.B) {
		j := OpenMemory()
		hop := RestartHop{
			Schema:     RestartChainSchema,
			Hop:        1,
			FromTrace:  "trace-1",
			ToTrace:    "trace-2",
			SeedTokens: 512,
			Handback:   "continue",
			Status:     RestartHopOK,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			j.AppendRestartHop("agent-worker", "trace-parent", hop)
		}
	})

	b.Run("quality_quarantine", func(b *testing.B) {
		j := OpenMemory()
		row := QualityQuarantineRow{
			Schema:  QualityQuarantineSchema,
			Case:    "case-eval-01",
			Owner:   "qa-team",
			Tier:    QualityTierPR,
			CostMS:  120,
			Verdict: QualityPass,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			j.AppendQualityQuarantine(row)
		}
	})
}

func BenchmarkCut(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "journal.jsonl")
	j, err := Open(path)
	if err != nil {
		b.Fatalf("Open failed: %v", err)
	}
	defer j.Close()

	ev := abi.Event{
		Kind: abi.EvDecide,
		Call: &abi.ToolCall{
			Tool:    "bash",
			TraceID: "trace-bench-cut",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"echo test"}`)},
		},
		Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "rule"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j.Emit(ev)
		if _, err := j.Cut(); err != nil {
			b.Fatalf("Cut failed: %v", err)
		}
	}
}
