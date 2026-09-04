package nodecompare

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

var (
	benchRenderSink string
	benchNodesSink  []Node
	benchQ8Sink     Q8Kernel
	benchBatchSink  Batchbench
)

func setupBenchEnvironment(tb testing.TB, nodeCount int) string {
	tb.Helper()
	root := tb.TempDir()

	q8Sample := `fak q8 kernel microbench
GOMAXPROCS=32
  f32             3.40 ms     12.0 weight-GB/s   1.00x vs f32
  int8xf32(WO)    1.00 ms     28.5 weight-GB/s   0.39x vs f32
  int8xf32(ACT)   1.20 ms     24.1 weight-GB/s   0.45x vs f32
  q4xf32(WO)      0.85 ms     35.2 weight-GB/s   0.31x vs f32
`
	batchTextSample := `B=1 step=0.100 agg= 14.5 tok/s
B=2 step=0.180 agg= 26.2 tok/s
B=4 step=0.320 agg= 48.0 tok/s
B=8 step=0.550 agg= 82.4 tok/s
B=16 step=0.920 agg= 124.0 tok/s
`

	for i := 0; i < nodeCount; i++ {
		host := fmt.Sprintf("node-%02d", i)
		nodeDir := filepath.Join(root, host)
		if err := os.MkdirAll(nodeDir, 0o755); err != nil {
			tb.Fatal(err)
		}

		info := map[string]any{
			"host":  host,
			"os":    "linux",
			"arch":  "amd64",
			"cores": 32,
			"cpu":   "AMD EPYC 7763",
			"go":    "go1.26",
			"git":   "v0.37.0",
		}
		rawInfo, err := json.Marshal(info)
		if err != nil {
			tb.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(nodeDir, "node-info.json"), rawInfo, 0o644); err != nil {
			tb.Fatal(err)
		}

		if err := os.WriteFile(filepath.Join(nodeDir, "q8kernel.txt"), []byte(q8Sample), 0o644); err != nil {
			tb.Fatal(err)
		}

		if i%2 == 0 {
			batchJSON := map[string]any{
				"baseline_b1_tok_per_sec": 14.5,
				"peak": map[string]any{
					"batch":           16,
					"agg_tok_per_sec": 124.0,
				},
			}
			rawBatch, err := json.Marshal(batchJSON)
			if err != nil {
				tb.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(nodeDir, "batchbench-q8.json"), rawBatch, 0o644); err != nil {
				tb.Fatal(err)
			}
		} else {
			if err := os.WriteFile(filepath.Join(nodeDir, "batchbench.txt"), []byte(batchTextSample), 0o644); err != nil {
				tb.Fatal(err)
			}
		}
	}

	return root
}

func BenchmarkNodeCompare(b *testing.B) {
	dir := setupBenchEnvironment(b, 4)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nodes, err := LoadNodes(dir)
		if err != nil {
			b.Fatal(err)
		}
		out := Render(nodes)
		if len(out) == 0 {
			b.Fatal("empty render output")
		}
		benchRenderSink = out
	}
}

func BenchmarkLoadNodes(b *testing.B) {
	dir := setupBenchEnvironment(b, 8)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nodes, err := LoadNodes(dir)
		if err != nil {
			b.Fatal(err)
		}
		benchNodesSink = nodes
	}
}

func BenchmarkRender(b *testing.B) {
	dir := setupBenchEnvironment(b, 8)
	nodes, err := LoadNodes(dir)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := Render(nodes)
		benchRenderSink = out
	}
}

func BenchmarkParseQ8Kernel(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "q8kernel.txt")
	sample := `fak q8 kernel microbench
GOMAXPROCS=64
  f32             3.40 ms     12.0 weight-GB/s   1.00x vs f32
  int8xf32(WO)    1.00 ms     28.5 weight-GB/s   0.39x vs f32
  int8xf32(ACT)   1.20 ms     24.1 weight-GB/s   0.45x vs f32
  q4xf32(WO)      0.85 ms     35.2 weight-GB/s   0.31x vs f32
`
	if err := os.WriteFile(path, []byte(sample), 0o644); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchQ8Sink = ParseQ8Kernel(path)
	}
}

func BenchmarkParseBatchbench(b *testing.B) {
	dir := b.TempDir()
	txtPath := filepath.Join(dir, "batchbench.txt")
	jsonPath := filepath.Join(dir, "batchbench-q8.json")

	rawJSON := []byte(`{"baseline_b1_tok_per_sec": 14.5, "peak": {"batch": 16, "agg_tok_per_sec": 124.0}}`)
	if err := os.WriteFile(jsonPath, rawJSON, 0o644); err != nil {
		b.Fatal(err)
	}
	if err := os.WriteFile(txtPath, []byte("B=1 step=0.1 agg= 14.5 tok/s\nB=16 step=0.9 agg= 124.0 tok/s\n"), 0o644); err != nil {
		b.Fatal(err)
	}

	b.Run("JSON", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBatchSink = ParseBatchbench(txtPath)
		}
	})

	textDir := b.TempDir()
	textOnlyPath := filepath.Join(textDir, "batchbench.txt")
	if err := os.WriteFile(textOnlyPath, []byte("B=1 step=0.1 agg= 14.5 tok/s\nB=4 step=0.3 agg= 48.0 tok/s\nB=16 step=0.9 agg= 124.0 tok/s\n"), 0o644); err != nil {
		b.Fatal(err)
	}

	b.Run("TextFallback", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBatchSink = ParseBatchbench(textOnlyPath)
		}
	})
}
