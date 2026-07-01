// Package nodecompare folds per-node benchmark outputs into a cross-hardware table.
package nodecompare

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

type Q8Kernel struct {
	GOMAXPROCS *int                   `json:"gomaxprocs"`
	Rows       map[string]Q8KernelRow `json:"rows"`
}

type Q8KernelRow struct {
	MS   float64 `json:"ms"`
	GBS  float64 `json:"gbs"`
	XF32 float64 `json:"xf32"`
}

type Batchbench struct {
	B1TokS   *float64 `json:"b1_tok_s"`
	BMax     *int     `json:"bmax"`
	BMaxTokS *float64 `json:"bmax_tok_s"`
}

type Node map[string]any

var (
	gomaxRE = regexp.MustCompile(`GOMAXPROCS=(\d+)`)
	q8RowRE = regexp.MustCompile(`(?m)^\s*(\S+)\s+([\d.]+)\s*ms\s+([\d.]+)\s*weight-GB/s\s+([\d.]+)x`)
	batchRE = regexp.MustCompile(`B=(\d+)\s+step=.*?agg=\s*([\d.]+)\s*tok/s`)
)

func ParseQ8Kernel(path string) Q8Kernel {
	out := Q8Kernel{Rows: map[string]Q8KernelRow{}}
	b, err := os.ReadFile(path)
	if err != nil {
		return out
	}
	txt := string(b)
	if m := gomaxRE.FindStringSubmatch(txt); m != nil {
		if n, err := strconv.Atoi(m[1]); err == nil {
			out.GOMAXPROCS = &n
		}
	}
	for _, m := range q8RowRE.FindAllStringSubmatch(txt, -1) {
		ms, _ := strconv.ParseFloat(m[2], 64)
		gbs, _ := strconv.ParseFloat(m[3], 64)
		xf32, _ := strconv.ParseFloat(m[4], 64)
		out.Rows[m[1]] = Q8KernelRow{MS: ms, GBS: gbs, XF32: xf32}
	}
	return out
}

func ParseBatchbench(path string) Batchbench {
	jsonPath := filepath.Join(filepath.Dir(path), "batchbench-q8.json")
	if b, err := os.ReadFile(jsonPath); err == nil {
		var rep map[string]any
		if json.Unmarshal(b, &rep) == nil {
			out := Batchbench{}
			if v, ok := numberPtr(rep["baseline_b1_tok_per_sec"]); ok {
				out.B1TokS = &v
			}
			if peak, ok := rep["peak"].(map[string]any); ok {
				if v, ok := intPtr(peak["batch"]); ok {
					out.BMax = &v
				}
				if v, ok := numberPtr(peak["agg_tok_per_sec"]); ok {
					out.BMaxTokS = &v
				}
			}
			return out
		}
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return Batchbench{}
	}
	rows := batchRE.FindAllStringSubmatch(string(b), -1)
	if len(rows) == 0 {
		return Batchbench{}
	}
	out := Batchbench{}
	for _, row := range rows {
		if row[1] == "1" {
			if v, err := strconv.ParseFloat(row[2], 64); err == nil {
				out.B1TokS = &v
			}
		}
	}
	maxB := -1
	var maxTok float64
	for _, row := range rows {
		batch, err := strconv.Atoi(row[1])
		if err != nil {
			continue
		}
		if batch > maxB {
			maxB = batch
			maxTok, _ = strconv.ParseFloat(row[2], 64)
		}
	}
	if maxB >= 0 {
		out.BMax = &maxB
		out.BMaxTokS = &maxTok
	}
	return out
}

func LoadNodes(nodesDir string) ([]Node, error) {
	entries, err := os.ReadDir(nodesDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var dirs []string
	for _, e := range entries {
		if e.IsDir() {
			dirs = append(dirs, filepath.Join(nodesDir, e.Name()))
		}
	}
	sort.Strings(dirs)
	var nodes []Node
	for _, dir := range dirs {
		infoPath := filepath.Join(dir, "node-info.json")
		if _, err := os.Stat(infoPath); err != nil {
			continue
		}
		node := Node{}
		if b, err := os.ReadFile(infoPath); err == nil {
			_ = json.Unmarshal(b, &node)
		}
		if len(node) == 0 {
			node["host"] = filepath.Base(dir)
		}
		node["q8"] = ParseQ8Kernel(filepath.Join(dir, "q8kernel.txt"))
		node["batch"] = ParseBatchbench(filepath.Join(dir, "batchbench.txt"))
		nodes = append(nodes, node)
	}
	return nodes, nil
}

func Render(nodes []Node) string {
	var b strings.Builder
	fmt.Fprintf(&b, "==== fak cross-node kernel comparison (%d node(s)) ====\n", len(nodes))
	fmt.Fprintf(&b, "%-16s %-14s %5s %10s %12s %13s %12s %15s\n", "HOST", "OS/ARCH", "CORES", "q8 f32 ms", "q8 i8xf32 ms", "i8xf32 vs f32", "batch B1 t/s", "batch Bmax t/s")
	for _, n := range nodes {
		q8 := q8Rows(n)
		f32 := q8["f32"]
		i8 := q8["int8xf32(WO)"]
		batch := batchObj(n)
		bmax := "-"
		if batch.BMaxTokS != nil && batch.BMax != nil {
			bmax = fmt.Sprintf("%v (B%d)", trimFloat(*batch.BMaxTokS), *batch.BMax)
		}
		fmt.Fprintf(&b, "%-16s %-14s %5s %10s %12s %13s %12s %15s\n",
			trunc(fmt.Sprint(n["host"]), 16),
			trunc(fmt.Sprintf("%v/%v", valueOr(n["os"], "?"), valueOr(n["arch"], "?")), 14),
			fmt.Sprint(valueOr(n["cores"], "?")),
			q8MS(f32),
			q8MS(i8),
			q8XF32(i8),
			floatPtrString(batch.B1TokS),
			bmax,
		)
	}
	b.WriteString("\nlower ms = faster kernel; i8xf32 vs f32 <1.0 = quantized GEMV beats f32;\n")
	b.WriteString("batch t/s = aggregate decode throughput (continuous batching).\n")
	b.WriteString("cpu per node:\n")
	for _, n := range nodes {
		fmt.Fprintf(&b, "  %-16s %v  (%v, git %v)\n", valueOr(n["host"], "?"), valueOr(n["cpu"], "?"), valueOr(n["go"], "?"), valueOr(n["git"], "?"))
	}
	return strings.TrimRight(b.String(), "\n")
}

func q8Rows(n Node) map[string]Q8KernelRow {
	switch q := n["q8"].(type) {
	case Q8Kernel:
		return q.Rows
	case map[string]any:
		rows, _ := q["rows"].(map[string]Q8KernelRow)
		return rows
	default:
		return map[string]Q8KernelRow{}
	}
}

func batchObj(n Node) Batchbench {
	if b, ok := n["batch"].(Batchbench); ok {
		return b
	}
	return Batchbench{}
}

func q8MS(row Q8KernelRow) string {
	if row == (Q8KernelRow{}) {
		return "-"
	}
	return trimFloat(row.MS)
}

func q8XF32(row Q8KernelRow) string {
	if row == (Q8KernelRow{}) {
		return "-"
	}
	return trimFloat(row.XF32) + "x"
}

func numberPtr(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case int:
		return float64(x), true
	default:
		return 0, false
	}
}

func intPtr(v any) (int, bool) {
	switch x := v.(type) {
	case float64:
		return int(x), true
	case int:
		return x, true
	default:
		return 0, false
	}
}

func floatPtrString(v *float64) string {
	if v == nil {
		return "-"
	}
	return trimFloat(*v)
}

func trimFloat(v float64) string {
	return strconv.FormatFloat(v, 'f', -1, 64)
}

func valueOr(v any, fallback string) any {
	if v == nil || fmt.Sprint(v) == "" {
		return fallback
	}
	return v
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
