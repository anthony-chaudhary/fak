package sessionsearch

import (
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

func sampleDocs(n int) []Doc {
	docs := make([]Doc, n)
	tools := []string{"git_diff", "file_read", "file_edit", "shell_exec", "test_runner", "build_check"}
	reasons := []string{"TOOL_DEADLINE_EXCEEDED", "PROCESS_EXIT_ERROR", "TIMEOUT", "PERMISSION_DENIED", "SIGKILL"}
	for i := 0; i < n; i++ {
		src := SourceInteractive
		session := fmt.Sprintf("interactive-%d", i%10)
		if i%3 == 0 {
			src = SourceCron
			session = fmt.Sprintf("cron-nightly-%d", i%5)
		}
		tool := tools[i%len(tools)]
		reason := reasons[i%len(reasons)]
		text := fmt.Sprintf("spawn %s %s at unix ms %d status running", tool, session, 1700000000000+i*1000)
		if i%2 == 1 {
			text = fmt.Sprintf("kill %s %s reason %s deadline exceeded", tool, session, reason)
		}
		docs[i] = Doc{
			ID:      fmt.Sprintf("ev:%d", i),
			Ordinal: i,
			Source:  src,
			Text:    text,
		}
	}
	return docs
}

func BenchmarkDocsFromJournal(b *testing.B) {
	journalData := strings.Join([]string{
		`{"kind":"spawn","call_id":"c1","session":"interactive-42","tool":"slow_fetch","at_unix_ms":1700000000000,"deadline_ms":30000}`,
		`{"kind":"kill","call_id":"c1","session":"interactive-42","reason":"TOOL_DEADLINE_EXCEEDED","at_unix_ms":1700000032000}`,
		`{"kind":"spawn","call_id":"c2","session":"cron-nightly-1","tool":"bg_tail","at_unix_ms":1700000001000}`,
		`{"kind":"exit","call_id":"c2","status":"ok","at_unix_ms":1700000002000}`,
		`{"kind":"spawn","call_id":"c3","session":"interactive-43","tool":"git_diff","at_unix_ms":1700000003000}`,
		`{"kind":"exit","call_id":"c3","status":"ok","at_unix_ms":1700000004000}`,
		`{"kind":"spawn","call_id":"c4","session":"cron-cleanup","tool":"clean_temp","at_unix_ms":1700000005000}`,
		`{"kind":"kill","call_id":"c4","session":"cron-cleanup","reason":"SIGKILL","at_unix_ms":1700000006000}`,
	}, "\n")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := strings.NewReader(journalData)
		docs, err := DocsFromJournal(r)
		if err != nil {
			b.Fatalf("DocsFromJournal failed: %v", err)
		}
		if len(docs) != 8 {
			b.Fatalf("unexpected doc count: %d", len(docs))
		}
	}
}

func BenchmarkIndexAdd(b *testing.B) {
	docs := sampleDocs(50)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ix := NewIndex()
		for _, d := range docs {
			ix.Add(d)
		}
	}
}

func BenchmarkIndexSearch(b *testing.B) {
	docs := sampleDocs(200)
	ix := NewIndex()
	for _, d := range docs {
		ix.Add(d)
	}

	b.Run("SingleTerm", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			hits := ix.Search("deadline", 5, 2)
			if len(hits) == 0 {
				b.Fatal("expected hits for query")
			}
		}
	})

	b.Run("MultiTermScored", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			hits := ix.Search("tool deadline exceeded git diff", 5, 2)
			if len(hits) == 0 {
				b.Fatal("expected hits for query")
			}
		}
	})
}

func BenchmarkInject(b *testing.B) {
	prefix := []cachemeta.PromptSegment{
		{Kind: cachemeta.SegStable, Tokens: 100, Content: []byte("SYSTEM PROMPT: you are a careful kernel agent.")},
		{Kind: cachemeta.SegToolSchema, Tokens: 50, Content: []byte("TOOLS: search_repo, read_file, run_tests")},
		{Kind: cachemeta.SegMessage, Tokens: 20, Content: []byte("user: what killed the fetch tool last night?")},
	}
	recalled := "recalled from prior sessions:\n  #1 [interactive] kill TOOL_DEADLINE_EXCEEDED\n      ~ spawn slow_fetch"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, proof := Inject(prefix, recalled)
		if !proof.PrefixStable {
			b.Fatal("expected prefix stable")
		}
	}
}

func BenchmarkWitnessUsefulness(b *testing.B) {
	hits := []Hit{
		{
			Doc: Doc{Text: "kerberos ticket renewal runbook auth procedure"},
			Window: []Doc{
				{Text: "spawn fetch_ticket auth interactive"},
				{Text: "exit ok ticket stored"},
			},
		},
		{
			Doc: Doc{Text: "database connection pool exhaustion cleanup"},
			Window: []Doc{
				{Text: "spawn pool_cleaner db maintenance"},
			},
		},
	}
	prior := "user asked about morning authentication failures and connection timeout"
	outcome := "I followed the kerberos renewal runbook procedure and verified auth"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		u := WitnessUsefulness(hits, prior, outcome)
		if u.Referenced != 1 {
			b.Fatalf("expected 1 referenced, got %d", u.Referenced)
		}
	}
}

func BenchmarkRecalledSpan(b *testing.B) {
	hits := []Hit{
		{
			Doc: Doc{Source: SourceInteractive, Text: "kill TOOL_DEADLINE_EXCEEDED interactive-42"},
			Window: []Doc{
				{Ordinal: 0, Text: "spawn slow_fetch interactive-42"},
				{Ordinal: 1, Text: "kill TOOL_DEADLINE_EXCEEDED interactive-42"},
			},
		},
		{
			Doc: Doc{Source: SourceCron, Text: "exit ok cron-nightly-1"},
			Window: []Doc{
				{Ordinal: 2, Text: "spawn bg_tail cron-nightly-1"},
				{Ordinal: 3, Text: "exit ok cron-nightly-1"},
			},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		span := RecalledSpan(hits)
		if span == "" {
			b.Fatal("unexpected empty span")
		}
	}
}
