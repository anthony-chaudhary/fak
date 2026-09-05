package toolrollup

import (
	"bytes"
	"encoding/json"
	"io"
	"testing"
)

var (
	benchSinkStats []ToolStat
	benchSinkCalls []ToolCall
)

func generateBenchCorpus(n int) []ToolCall {
	tools := []string{"read", "write", "edit", "bash", "glob", "grep", "fak_read", "fak_syscall"}
	records := make([]ToolCall, n)
	for i := 0; i < n; i++ {
		tool := tools[i%len(tools)]
		records[i] = ToolCall{
			Tool:       tool,
			TokensIn:   50 + (i*17)%500,
			TokensOut:  10 + (i*13)%200,
			DurationMS: int64(5 + (i*7)%100),
			OK:         (i % 9) != 0,
		}
	}
	return records
}

func generateBenchJSONL(n int) []byte {
	records := generateBenchCorpus(n)
	var buf bytes.Buffer
	for _, r := range records {
		b, _ := json.Marshal(r)
		buf.Write(b)
		buf.WriteByte('\n')
	}
	return buf.Bytes()
}

func TestBenchmarkFixtures(t *testing.T) {
	corpus := generateBenchCorpus(100)
	if len(corpus) != 100 {
		t.Fatalf("expected 100 calls, got %d", len(corpus))
	}
	data := generateBenchJSONL(100)
	records, err := ReadCorpus(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("ReadCorpus failed: %v", err)
	}
	if len(records) != 100 {
		t.Fatalf("expected 100 records, got %d", len(records))
	}
}

func BenchmarkRollup(b *testing.B) {
	corpus := generateBenchCorpus(100)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchSinkStats = Rollup(corpus)
	}
}

func BenchmarkReadCorpus(b *testing.B) {
	data := generateBenchJSONL(100)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		records, err := ReadCorpus(bytes.NewReader(data))
		if err != nil {
			b.Fatal(err)
		}
		benchSinkCalls = records
	}
}

func BenchmarkRender(b *testing.B) {
	stats := Rollup(generateBenchCorpus(100))
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		Render(io.Discard, stats)
	}
}

func BenchmarkEndToEnd(b *testing.B) {
	data := generateBenchJSONL(100)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		records, err := ReadCorpus(bytes.NewReader(data))
		if err != nil {
			b.Fatal(err)
		}
		stats := Rollup(records)
		Render(io.Discard, stats)
	}
}
