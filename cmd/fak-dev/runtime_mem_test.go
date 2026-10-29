package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/runtimeobs"
)

// runtimeMemServer serves /debug/vars with a fresh real receipt per scrape,
// nested under runtime.go_runtime as the gateway publishes it.
func runtimeMemServer(t *testing.T, rss uint64) *httptest.Server {
	t.Helper()
	c := runtimeobs.New(runtimeobs.WithRSSReader(func() (uint64, bool) { return rss, true }))
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/debug/vars" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"cmdline": []string{"fak", "serve"},
			"runtime": map[string]any{"go_runtime": c.Sample()},
		})
	}))
}

func noSleepDeps() runtimeMemDeps {
	return runtimeMemDeps{client: http.DefaultClient, sleep: func(time.Duration) {}}
}

// fak-test:runtime fast est=50ms lane=default
func TestRuntimeMemJSONDecodesReceiptAndDiff(t *testing.T) {
	const rss = 3 << 30
	ts := runtimeMemServer(t, rss)
	defer ts.Close()
	var out, errb bytes.Buffer
	code := runRuntimeMemWith(&out, &errb, []string{"--url", ts.URL, "--samples", "3", "--interval", "1ms", "--json"}, noSleepDeps())
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	var doc runtimeMemDoc
	if err := json.Unmarshal(out.Bytes(), &doc); err != nil {
		t.Fatalf("decode doc: %v", err)
	}
	if doc.Schema != runtimeMemSchema || doc.Samples != 3 || doc.URL != ts.URL+"/debug/vars" {
		t.Fatalf("header schema=%q samples=%d url=%q", doc.Schema, doc.Samples, doc.URL)
	}
	if doc.Last.Schema != runtimeobs.Schema || doc.Last.Seq <= doc.First.Seq {
		t.Fatalf("receipts schema=%q seq %d..%d", doc.Last.Schema, doc.First.Seq, doc.Last.Seq)
	}
	if doc.Delta == nil || doc.DeltaError != "" || doc.Delta.Schema != runtimeobs.DeltaSchema || !doc.Delta.GCCycles.OK {
		t.Fatalf("delta=%+v err=%q", doc.Delta, doc.DeltaError)
	}
	want := []string{
		"memory.process_rss_bytes", "memory.total_mapped_bytes", "memory.heap_objects_bytes",
		"gc.heap_live_bytes", "gc.heap_goal_bytes", "memory.stacks_bytes", "memory.heap_released_bytes",
	}
	if len(doc.Gauges) != len(want) {
		t.Fatalf("gauges=%d want %d", len(doc.Gauges), len(want))
	}
	for i, g := range doc.Gauges {
		if g.Field != want[i] {
			t.Fatalf("gauge[%d]=%q want %q", i, g.Field, want[i])
		}
	}
	if g := doc.Gauges[0]; !g.First.OK || g.First.V != rss || !g.Last.OK || g.Last.V != rss {
		t.Fatalf("rss gauge=%+v want %d", g, uint64(rss))
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestRuntimeMemTableIsBounded(t *testing.T) {
	ts := runtimeMemServer(t, 1<<30)
	defer ts.Close()
	var out, errb bytes.Buffer
	if code := runRuntimeMemWith(&out, &errb, []string{"--url", ts.URL + "/debug/vars", "--samples", "2"}, noSleepDeps()); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	lines := strings.Split(strings.TrimRight(out.String(), "\n"), "\n")
	if len(lines) > 12 {
		t.Fatalf("table has %d lines, cap 12", len(lines))
	}
	for _, field := range []string{"memory.process_rss_bytes", "gc.heap_goal_bytes", "delta"} {
		if !strings.Contains(out.String(), field) {
			t.Fatalf("table lacks %s row", field)
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestRuntimeMemRefusals(t *testing.T) {
	bare := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"runtime":{}}`))
	}))
	defer bare.Close()
	if _, err := scrapeGoRuntime(http.DefaultClient, bare.URL, time.Second); !errors.Is(err, errRuntimeMemNoReceipt) {
		t.Fatalf("missing receipt err=%v", err)
	}
	wrong := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"go_runtime":{"schema":"other.v9"}}`))
	}))
	defer wrong.Close()
	if _, err := scrapeGoRuntime(http.DefaultClient, wrong.URL, time.Second); !errors.Is(err, errRuntimeMemSchema) {
		t.Fatalf("schema err=%v", err)
	}
	var out, errb bytes.Buffer
	if code := runRuntimeMemWith(&out, &errb, []string{"--samples", "0"}, noSleepDeps()); code != 2 {
		t.Fatalf("samples=0 code=%d", code)
	}
	if code := runRuntimeMemWith(&out, &errb, []string{"--url", bare.URL}, noSleepDeps()); code != 1 {
		t.Fatalf("missing receipt code=%d", code)
	}
}
