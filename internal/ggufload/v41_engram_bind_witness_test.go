package ggufload

import (
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestV41EngramBindLoadWitness times the load-time Engram bind over a real V4.1
// checkpoint without serving it: open the shards mapped (the serve load's
// reader), open every declared layer's table, and bind all layers concurrently
// as attachV41EngramServing does, then exit. FAK_V41_ENGRAM_BIND_MODE=flat-f32
// times the flat sha256 over the dequantized rows that acquisition bound before,
// over the same readers, so before and after share one reader.
// Opt-in: FAK_V41_ENGRAM_BIND_GGUF=<first shard>.
// fak-test:runtime integration est=180s lane=optin
func TestV41EngramBindLoadWitness(t *testing.T) {
	path := os.Getenv("FAK_V41_ENGRAM_BIND_GGUF")
	if path == "" {
		t.Skip("set FAK_V41_ENGRAM_BIND_GGUF to a V4.1 GGUF shard to time the Engram bind")
	}
	mode := os.Getenv("FAK_V41_ENGRAM_BIND_MODE")
	if mode == "" {
		mode = "acquire"
	}
	if mode != "acquire" && mode != "flat-f32" {
		t.Fatalf("FAK_V41_ENGRAM_BIND_MODE=%q, want acquire or flat-f32", mode)
	}
	opened := time.Now()
	ws, err := OpenWeightsMapped(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	if _, err := ws.File.Config(); err != nil {
		t.Fatal(err)
	}
	eng := ws.File.DeepSeek41Engram
	if eng == nil || len(eng.LayerIDs) == 0 {
		t.Fatal("checkpoint declares no Engram layers")
	}
	type table struct {
		reader  io.ReaderAt
		shard   model.V41EngramShard
		bind    model.V41EngramArtifactBinding
		encoded int64
	}
	tables := make([]table, len(eng.LayerIDs))
	var served, encoded int64
	for i, id := range eng.LayerIDs {
		inner, rows, rowBytes, err := openV41EngramRowSource(ws, id)
		if err != nil {
			t.Fatalf("layer %d: %v", id, err)
		}
		name := fmt.Sprintf("blk.%d.%s", id, v41EngramEmbdSuffix)
		size := int64(rows) * int64(rowBytes)
		tb := table{
			reader:  newV41EngramRowReaderAt(inner, rows, rowBytes),
			shard:   model.V41EngramShard{ID: name, ExpectedID: name, Size: size, Rows: rows},
			bind:    model.V41EngramArtifactBinding{Shard: name, Size: size, Rows: rows, RowBytes: rowBytes},
			encoded: size,
		}
		if enc, ok := inner.(model.V41EngramEncodedRows); ok {
			if _, _, n, _, err := enc.V41EngramEncodedRows(0, rows); err == nil {
				tb.encoded = n
			}
		}
		tables[i] = tb
		served += size
		encoded += tb.encoded
	}
	setup := time.Since(opened)

	digests := make([]string, len(tables))
	errs := make([]error, len(tables))
	start := time.Now()
	var wg sync.WaitGroup
	for i := range tables {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			tb := tables[i]
			if mode == "flat-f32" {
				digests[i], errs[i] = model.V41EngramDigestExtent(tb.reader, 0, tb.bind.Size)
				return
			}
			src, err := model.NewV41AcquiredEngramRowSource(tb.reader, tb.shard, tb.bind)
			if err != nil {
				errs[i] = err
				return
			}
			b, _ := model.V41EngramBinding(src)
			digests[i] = b.Digest
		}(i)
	}
	wg.Wait()
	wall := time.Since(start)
	for i, err := range errs {
		if err != nil {
			t.Fatalf("layer %d: bind: %v", eng.LayerIDs[i], err)
		}
	}
	// VmHWM counts the touched mapped checkpoint pages; RssAnon is the heap.
	mem := map[string]string{"VmHWM": "unknown", "RssAnon": "unknown"}
	if status, err := os.ReadFile("/proc/self/status"); err == nil {
		for _, line := range strings.Split(string(status), "\n") {
			for key := range mem {
				if v, ok := strings.CutPrefix(line, key+":"); ok {
					mem[key] = strings.Join(strings.Fields(v), "")
				}
			}
		}
	}
	t.Logf("V41_ENGRAM_BIND mode=%s layers=%v served_bytes=%d encoded_bytes=%d setup_s=%.3f bind_wall_s=%.3f rss_anon=%s vm_hwm_incl_mapped=%s digests=%v",
		mode, eng.LayerIDs, served, encoded, setup.Seconds(), wall.Seconds(), mem["RssAnon"], mem["VmHWM"], digests)
}
