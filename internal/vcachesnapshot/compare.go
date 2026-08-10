package vcachesnapshot

import (
	"bufio"
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
	"os"
	"path/filepath"
	"time"
)

type ComparisonArm struct {
	Name         string
	Kind         string
	Available    bool
	Correct      bool
	WriteLatency time.Duration
	ReadLatency  time.Duration
	InputRows    int
	RetainedRows int
	DroppedRows  int
	InputTokens  int64
	StorageBytes int64
	CostUSD      float64
	Note         string
}
type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}

func snapshotFixture() []vcacheobserve.Turn {
	return []vcacheobserve.Turn{{Family: "a", UnixMillis: 1, InputTokens: 100}, {Family: "a", UnixMillis: 2, InputTokens: 200, CacheCreation: 100}, {Family: "a", UnixMillis: 3, InputTokens: 300, CacheRead: 100}, {Family: "b", UnixMillis: 4, InputTokens: 400}, {Family: "b", UnixMillis: 5, InputTokens: 500, CacheRead: 200}}
}
func tokenTotal(ts []vcacheobserve.Turn) int64 {
	var n int64
	for _, t := range ts {
		n += t.InputTokens
	}
	return n
}
func appendJSONL(path string, turns []vcacheobserve.Turn) error {
	f, e := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if e != nil {
		return e
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	enc := json.NewEncoder(w)
	for _, t := range turns {
		if e = enc.Encode(t); e != nil {
			return e
		}
	}
	if e = w.Flush(); e != nil {
		return e
	}
	return f.Sync()
}
func CompareLocal() ComparisonResult {
	turns := snapshotFixture()
	dir, e := os.MkdirTemp("", "fak-vcache-snapshot-compare-")
	if e != nil {
		panic(e)
	}
	defer os.RemoveAll(dir)
	nativePath := filepath.Join(dir, "native.jsonl")
	retained := boundWindow(turns, 3)
	start := time.Now()
	e = Write(nativePath, retained)
	nw := time.Since(start)
	start = time.Now()
	read, ok, re := Read(nativePath)
	nr := time.Since(start)
	info, _ := os.Stat(nativePath)
	correct := e == nil && re == nil && ok && len(read) == 3 && read[0].UnixMillis == 3 && read[2].UnixMillis == 5
	rawPath := filepath.Join(dir, "raw.jsonl")
	start = time.Now()
	ae := appendJSONL(rawPath, turns)
	rw := time.Since(start)
	start = time.Now()
	raw, rok, rerr := Read(rawPath)
	rr := time.Since(start)
	rawInfo, _ := os.Stat(rawPath)
	rawCorrect := ae == nil && rerr == nil && rok && len(raw) == 3
	return ComparisonResult{Workload: "persist five ordered provider-cache turns with a three-turn retention window and replay the retained rows", Arms: []ComparisonArm{{Name: "fak native bounded fsynced JSONL snapshot", Kind: "native", Available: true, Correct: correct, WriteLatency: nw, ReadLatency: nr, InputRows: 5, RetainedRows: len(read), DroppedRows: 2, InputTokens: tokenTotal(turns), StorageBytes: info.Size(), Note: "truncate/write/flush/fsync; tolerant replay; not atomic replacement"}, {Name: "unbounded append-only JSONL", Kind: "baseline", Available: true, Correct: rawCorrect, WriteLatency: rw, ReadLatency: rr, InputRows: 5, RetainedRows: len(raw), InputTokens: tokenTotal(turns), StorageBytes: rawInfo.Size(), Note: "tuned baseline preserves all rows but violates the three-row retention contract"}, {Name: "fak + Prometheus", Kind: "integration", Note: "requires real remote-write/scrape and retention"}, {Name: "fak + OpenTelemetry", Kind: "integration", Note: "requires real collector/exporter and retention"}, {Name: "SQLite WAL", Kind: "external", Note: "requires real WAL durability and bounded-row policy"}, {Name: "Prometheus TSDB", Kind: "external", Note: "requires real TSDB ingestion and retention"}, {Name: "ClickHouse", Kind: "external", Note: "requires real MergeTree ingestion and TTL"}}}
}
