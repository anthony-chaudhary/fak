package nodecompare

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

const q8Txt = `fak q8 kernel microbench
GOMAXPROCS=32
  f32             3.40 ms     12.0 weight-GB/s   1.00x vs f32
  int8xf32(WO)    1.00 ms     28.5 weight-GB/s   0.39x vs f32
`

func TestParseQ8Kernel(t *testing.T) {
	p := filepath.Join(t.TempDir(), "q8kernel.txt")
	mustWriteNode(t, p, q8Txt)
	out := ParseQ8Kernel(p)
	if out.GOMAXPROCS == nil || *out.GOMAXPROCS != 32 {
		t.Fatalf("gomaxprocs=%v", out.GOMAXPROCS)
	}
	i8 := out.Rows["int8xf32(WO)"]
	if i8.MS != 1.00 || i8.GBS != 28.5 || i8.XF32 != 0.39 {
		t.Fatalf("i8 row=%+v", i8)
	}
	if !(i8.XF32 < out.Rows["f32"].XF32) {
		t.Fatalf("quantized row should beat f32: %+v vs %+v", i8, out.Rows["f32"])
	}
}

func TestParseQ8KernelMissing(t *testing.T) {
	out := ParseQ8Kernel(filepath.Join(t.TempDir(), "missing.txt"))
	if out.GOMAXPROCS != nil || len(out.Rows) != 0 {
		t.Fatalf("missing q8=%+v", out)
	}
}

func TestParseBatchbenchPrefersJSON(t *testing.T) {
	dir := t.TempDir()
	mustWriteNodeJSON(t, filepath.Join(dir, "batchbench-q8.json"), map[string]any{
		"baseline_b1_tok_per_sec": 11.0,
		"peak":                    map[string]any{"batch": 16, "agg_tok_per_sec": 95.0},
	})
	out := ParseBatchbench(filepath.Join(dir, "batchbench.txt"))
	if out.B1TokS == nil || *out.B1TokS != 11.0 || out.BMax == nil || *out.BMax != 16 || out.BMaxTokS == nil || *out.BMaxTokS != 95.0 {
		t.Fatalf("batch=%+v", out)
	}
}

func TestParseBatchbenchText(t *testing.T) {
	p := filepath.Join(t.TempDir(), "batchbench.txt")
	mustWriteNode(t, p, "B=1 step=0.1 agg=  9.0 tok/s\nB=4 step=0.4 agg= 30.0 tok/s\nB=8 step=0.8 agg= 48.0 tok/s\n")
	out := ParseBatchbench(p)
	if out.B1TokS == nil || *out.B1TokS != 9.0 || out.BMax == nil || *out.BMax != 8 || out.BMaxTokS == nil || *out.BMaxTokS != 48.0 {
		t.Fatalf("batch=%+v", out)
	}
}

func TestLoadNodes(t *testing.T) {
	root := t.TempDir()
	host := filepath.Join(root, "node-a")
	if err := os.MkdirAll(host, 0o755); err != nil {
		t.Fatal(err)
	}
	mustWriteNodeJSON(t, filepath.Join(host, "node-info.json"), map[string]any{"host": "node-a", "arch": "arm64"})
	mustWriteNode(t, filepath.Join(host, "q8kernel.txt"), q8Txt)
	nodes, err := LoadNodes(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(nodes) != 1 || nodes[0]["host"] != "node-a" {
		t.Fatalf("nodes=%+v", nodes)
	}
	q8 := nodes[0]["q8"].(Q8Kernel)
	if q8.GOMAXPROCS == nil || *q8.GOMAXPROCS != 32 {
		t.Fatalf("q8=%+v", q8)
	}
}

func mustWriteNode(t *testing.T, path, body string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func mustWriteNodeJSON(t *testing.T, path string, body any) {
	t.Helper()
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	mustWriteNode(t, path, string(b))
}
