package guardcorpus_test

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/guardcorpus"
	"github.com/anthony-chaudhary/fak/internal/journal"
)

func benchmarkRows() []journal.Row {
	return []journal.Row{
		{Seq: 1, TSUnixNano: 100, Kind: "DECIDE", Tool: "Read", Verdict: "ALLOW", By: "floor", ArgsLabel: "path=main.go"},
		{Seq: 2, TSUnixNano: 200, Kind: "DENY", Tool: "Bash", Verdict: "DENY", Reason: "POLICY_BLOCK", By: "floor", Witness: "rm -rf /", ArgsLabel: "command=rm"},
		{Seq: 3, TSUnixNano: 300, Kind: "QUARANTINE", Tool: "WebFetch", Verdict: "QUARANTINE", Reason: "SECRET_DISCOVERED", By: "secretgate", Witness: "sk-redacted-claim"},
		{Seq: 4, TSUnixNano: 400, Kind: "DENY", Tool: "Bash", Verdict: "DENY", Reason: "POLICY_BLOCK", By: "floor", Witness: "curl evil.com"},
		{Seq: 5, TSUnixNano: 500, Kind: "DECIDE", Tool: "Edit", Verdict: "ALLOW", By: "advmodel", ArgsLabel: "path=README.md"},
		{Seq: 6, TSUnixNano: 600, Kind: "DENY", Tool: "Bash", Verdict: "DENY", Reason: "OFF_TRUNK", By: "gitgate", Witness: "git push"},
		{Seq: 7, TSUnixNano: 700, Kind: "ADVISORY", Tool: "ShellDialect", Verdict: "ADVISORY", By: "shell-dialect", Reason: "posix_compat"},
		{Seq: 8, TSUnixNano: 800, Kind: "DECIDE", Tool: "Read", Verdict: "ALLOW", By: "floor", ArgsLabel: "path=config.json"},
	}
}

func BenchmarkFold(b *testing.B) {
	meta := guardcorpus.SessionMeta{
		TraceID:       "trace-bench-1",
		Agent:         "claude-code",
		HostClass:     "desktop",
		PolicyDigest:  "sha256:abcd1234efgh5678",
		ChainVerified: true,
	}
	rows := benchmarkRows()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = guardcorpus.Fold(meta, rows)
	}
}

func BenchmarkFoldLargeSession(b *testing.B) {
	meta := guardcorpus.SessionMeta{
		TraceID:       "trace-bench-large",
		Agent:         "claude-code",
		HostClass:     "fleet",
		PolicyDigest:  "sha256:deadbeefcafebabe",
		ChainVerified: true,
	}
	base := benchmarkRows()
	rows := make([]journal.Row, 0, len(base)*20)
	for i := 0; i < 20; i++ {
		for _, r := range base {
			r.Seq += uint64(len(rows))
			r.TSUnixNano += int64(len(rows) * 100)
			rows = append(rows, r)
		}
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = guardcorpus.Fold(meta, rows)
	}
}
