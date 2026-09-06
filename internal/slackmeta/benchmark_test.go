package slackmeta

import "testing"

var (
	benchScoreSink  Score
	benchStringSink string
	benchBlocksSink []any
	benchMapSink    map[string]any
	benchIntSink    int
)

func sampleScore() Score {
	return New(6, 2, "findings and action items vs source metadata")
}

func samplePost() Post {
	return Post{
		Emoji: ":white_check_mark:",
		Title: "Maturity Debt Sweep",
		Lead:  "All targeted lanes advanced to production readiness.",
		Lines: []string{
			"lane internal/slackmeta: benchmarked badge secured",
			"lane internal/cachemeta: zero regression across prefix reuse",
			"lane internal/blockerpost: triage paths verified",
		},
		Source: "ci",
	}
}

func TestBenchmarkSanity(t *testing.T) {
	post := samplePost()
	score := sampleScore()

	txt := post.Text(score)
	if txt == "" {
		t.Fatal("expected non-empty rendered post text")
	}
	blks := post.Blocks(score)
	if len(blks) == 0 {
		t.Fatal("expected non-empty rendered post blocks")
	}
	if score.Line() == "" {
		t.Fatal("expected non-empty score line")
	}
	if got := NonEmpty("alpha", " ", "", "beta"); got != 2 {
		t.Fatalf("expected 2 non-empty strings, got %d", got)
	}
}

func BenchmarkNew(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchScoreSink = New(8, 2, "production signal vs noise")
	}
}

func BenchmarkScoreLine(b *testing.B) {
	score := sampleScore()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = score.Line()
	}
}

func BenchmarkPostText(b *testing.B) {
	post := samplePost()
	score := sampleScore()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = post.Text(score)
	}
}

func BenchmarkPostBlocks(b *testing.B) {
	post := samplePost()
	score := sampleScore()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBlocksSink = post.Blocks(score)
	}
}

func BenchmarkAppendText(b *testing.B) {
	score := sampleScore()
	rawText := "Status update: 4 tasks finished cleanly\nDetails follow below"
	withSN := AppendText(rawText, score)

	b.Run("Fresh", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = AppendText(rawText, score)
		}
	})

	b.Run("Idempotent", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = AppendText(withSN, score)
		}
	})
}

func BenchmarkAppendContext(b *testing.B) {
	score := sampleScore()
	blocks := []any{
		map[string]any{"type": "header", "text": map[string]any{"type": "plain_text", "text": "Report"}},
		map[string]any{"type": "section", "text": map[string]any{"type": "mrkdwn", "text": "All checks green"}},
	}
	appended := AppendContext(blocks, score)

	b.Run("Fresh", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBlocksSink = AppendContext(blocks, score)
		}
	})

	b.Run("Idempotent", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBlocksSink = AppendContext(appended, score)
		}
	})
}

func BenchmarkContextBlock(b *testing.B) {
	score := sampleScore()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMapSink = ContextBlock(score)
	}
}

func BenchmarkNonEmpty(b *testing.B) {
	values := []string{"finding", "", "current", "   ", "trend", "action", "", "fence", "fleet"}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchIntSink = NonEmpty(values...)
	}
}
