package versionskew

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/binstamp"
)

func BenchmarkClassify(b *testing.B) {
	const tip = "1111111111111111111111111111111111111111"
	cases := []struct {
		running binstamp.Stamp
		tip     string
		rel     Relation
	}{
		{binstamp.Stamp{Revision: tip, HasVCS: true}, tip, RelEqual},
		{binstamp.Stamp{Revision: "abc1234", HasVCS: true}, tip, RelBehind},
		{binstamp.Stamp{Revision: "abc1234", HasVCS: true}, tip, RelAhead},
		{binstamp.Stamp{Revision: "abc1234", HasVCS: true}, tip, RelDiverged},
		{binstamp.Stamp{}, tip, RelEqual},
		{binstamp.Stamp{Revision: "abc1234", HasVCS: true, Dirty: true}, tip, RelBehind},
		{binstamp.Stamp{Revision: "abc1234", HasVCS: true}, "", RelUndetermined},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c := cases[i%len(cases)]
		v := Classify(c.running, c.tip, c.rel)
		if v == Unknown && c.tip != "" && c.rel != RelUndetermined {
			b.Fatalf("unexpected Unknown verdict for case: %+v", c)
		}
	}
}

func BenchmarkVerdictString(b *testing.B) {
	verdicts := []Verdict{Unknown, Fresh, Skewed, Ahead, Diverged, Unstamped, Dirty}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v := verdicts[i%len(verdicts)]
		s := v.String()
		if len(s) == 0 {
			b.Fatal("unexpected empty verdict string")
		}
	}
}

func BenchmarkVerdictRefusable(b *testing.B) {
	verdicts := []Verdict{Unknown, Fresh, Skewed, Ahead, Diverged, Unstamped, Dirty}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v := verdicts[i%len(verdicts)]
		if v.Refusable() && (v == Fresh || v == Ahead || v == Unknown) {
			b.Fatalf("unexpected refusable verdict for %v", v)
		}
	}
}

func BenchmarkAssessStampExactSHA(b *testing.B) {
	ctx := context.Background()
	const sha = "0123456789abcdef0123456789abcdef01234567"
	runner := func(_ context.Context, _ string, _ string, _ ...string) (string, bool) {
		return sha + "\n", true
	}
	stamp := binstamp.Stamp{Revision: sha, HasVCS: true}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		got := AssessStamp(ctx, runner, ".", "origin/main", stamp)
		if got.Verdict != Fresh || got.Relation != RelEqual {
			b.Fatalf("unexpected assessment: %+v", got)
		}
	}
}

func BenchmarkAssessStampAncestry(b *testing.B) {
	ctx := context.Background()
	const (
		tip     = "1111111111111111111111111111111111111111"
		running = "2222222222222222222222222222222222222222"
	)
	runner := func(_ context.Context, _ string, _ string, args ...string) (string, bool) {
		if len(args) >= 2 && args[0] == "rev-parse" {
			return tip + "\n", true
		}
		if len(args) >= 4 && args[0] == "merge-base" {
			if args[2] == running && args[3] == tip {
				return "", true
			}
			return "", false
		}
		return "", false
	}
	stamp := binstamp.Stamp{Revision: running, HasVCS: true}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		got := AssessStamp(ctx, runner, ".", "origin/main", stamp)
		if got.Verdict != Skewed || got.Relation != RelBehind {
			b.Fatalf("unexpected assessment: %+v", got)
		}
	}
}

func BenchmarkAssessStampUnstamped(b *testing.B) {
	ctx := context.Background()
	runner := func(_ context.Context, _ string, _ string, _ ...string) (string, bool) {
		return "", false
	}
	stamp := binstamp.Stamp{}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		got := AssessStamp(ctx, runner, ".", "origin/main", stamp)
		if got.Verdict != Unstamped {
			b.Fatalf("unexpected assessment: %+v", got)
		}
	}
}

func BenchmarkVersioned(b *testing.B) {
	row := map[string]any{
		"node_id":   "node-1",
		"status":    "healthy",
		"workload":  "eval",
		"instances": 4,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := Versioned(row, "1.0.0")
		if out["version"] != "1.0.0" {
			b.Fatalf("unexpected version: %v", out["version"])
		}
	}
}

func BenchmarkVersionedRows(b *testing.B) {
	rows := []map[string]any{
		{"node_id": "node-1", "load": 0.25},
		{"node_id": "node-2", "load": 0.50},
		{"node_id": "node-3", "load": 0.75},
		{"node_id": "node-4", "load": 1.00},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := VersionedRows(rows, "1.0.0")
		if len(out) != 4 {
			b.Fatalf("unexpected rows length: %d", len(out))
		}
	}
}
