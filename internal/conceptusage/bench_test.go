package conceptusage

import (
	"testing"
	"time"
)

func BenchmarkConceptUsage(b *testing.B) {
	root := b.TempDir()
	writeJournal(b, root, "verdict-journal.jsonl",
		`{"syscall":"verify","verdict":"SHIPPED"}`,
		`{"syscall":"improve","verdict":"KEEP"}`,
		`{"syscall":"memory_recall","verdict":"RECALL_FRESH"}`,
	)
	writeJournal(b, root, "lane-journal.jsonl",
		`{"op":"ACQUIRE","lane":"model"}`,
		`{"op":"RELEASE","lane":"model"}`,
	)
	opts := Options{
		Root:   root,
		Now:    time.Unix(1_700_000_000, 0).UTC(),
		gitLog: disciplinedCommits(20),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		payload := Build(opts)
		if !payload.OK {
			b.Fatalf("expected payload.OK, got reason %s", payload.Reason)
		}
	}
}

func TestBenchmarkConceptUsage(t *testing.T) {
	root := t.TempDir()
	writeJournal(t, root, "verdict-journal.jsonl",
		`{"syscall":"verify","verdict":"SHIPPED"}`,
	)
	opts := Options{
		Root:   root,
		Now:    time.Unix(1_700_000_000, 0).UTC(),
		gitLog: disciplinedCommits(5),
	}
	payload := Build(opts)
	if payload.Workspace == "" {
		t.Fatal("expected non-empty workspace")
	}
}
