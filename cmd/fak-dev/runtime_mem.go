package main

// runtime-mem — bounded reader for a serving process's Go runtime receipt.
// It scrapes /debug/vars, decodes only runtime.go_runtime (runtimeobs.Receipt),
// and prints a compact table plus the runtimeobs.Diff between the first and
// last sample. Exit codes: 0 read, 1 typed scrape/decode failure, 2 usage.

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/runtimeobs"
)

const (
	runtimeMemUsage = "usage: fak-dev runtime-mem [--url URL] [--samples N] [--interval D] [--timeout D] [--json]"
	// RuntimeMemSchema versions the --json document.
	runtimeMemSchema = "fak.dev.runtime-mem.v1"
	// runtimeMemMaxBody bounds one /debug/vars body; the receipt is a few KiB.
	runtimeMemMaxBody = 8 << 20
)

var (
	errRuntimeMemNoReceipt = errors.New("runtime-mem: /debug/vars has no go_runtime receipt")
	errRuntimeMemSchema    = errors.New("runtime-mem: go_runtime schema mismatch")
)

// runtimeMemDoc is the --json output: first/last receipts and their delta.
type runtimeMemDoc struct {
	Schema     string               `json:"schema"`
	URL        string               `json:"url"`
	Samples    int                  `json:"samples"`
	First      runtimeobs.Receipt   `json:"first"`
	Last       runtimeobs.Receipt   `json:"last"`
	Delta      *runtimeobs.Delta    `json:"delta,omitempty"`
	DeltaError string               `json:"delta_error,omitempty"`
	Gauges     []runtimeMemGaugeRow `json:"gauges"`
}

// runtimeMemGaugeRow is one point-in-time memory gauge, first and last.
type runtimeMemGaugeRow struct {
	Field string         `json:"field"`
	First runtimeobs.U64 `json:"first"`
	Last  runtimeobs.U64 `json:"last"`
}

type runtimeMemDeps struct {
	client *http.Client
	sleep  func(time.Duration)
}

// runtimeMemHTTPClient is the hard outer bound on a scrape: the per-request context
// (--timeout, default 10s) stays authoritative for realistic use, while the client
// deadline guarantees a stalled peer can never hold the reader open indefinitely no
// matter how large --timeout is set.
var runtimeMemHTTPClient = &http.Client{Timeout: 10 * time.Minute}

func runRuntimeMem(stdout, stderr io.Writer, argv []string) int {
	return runRuntimeMemWith(stdout, stderr, argv, runtimeMemDeps{client: runtimeMemHTTPClient, sleep: time.Sleep})
}

func runRuntimeMemWith(stdout, stderr io.Writer, argv []string, deps runtimeMemDeps) int {
	fs := flag.NewFlagSet("runtime-mem", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	url := fs.String("url", "http://127.0.0.1:8080", "serve base URL (or the full /debug/vars URL)")
	samples := fs.Int("samples", 2, "number of scrapes (>=1; 2+ prints a delta)")
	interval := fs.Duration("interval", 5*time.Second, "wait between scrapes")
	timeout := fs.Duration("timeout", 10*time.Second, "per-scrape HTTP timeout")
	asJSON := fs.Bool("json", false, "emit the fak.dev.runtime-mem.v1 document")
	if err := fs.Parse(argv); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			fmt.Fprintln(stdout, runtimeMemUsage)
			return 0
		}
		fmt.Fprintln(stderr, runtimeMemUsage)
		return 2
	}
	if fs.NArg() != 0 || *samples < 1 || *interval < 0 || *timeout <= 0 {
		fmt.Fprintln(stderr, runtimeMemUsage)
		return 2
	}
	target := runtimeMemVarsURL(*url)
	var got []runtimeobs.Receipt
	for i := 0; i < *samples; i++ {
		if i > 0 && *interval > 0 {
			deps.sleep(*interval)
		}
		r, err := scrapeGoRuntime(deps.client, target, *timeout)
		if err != nil {
			fmt.Fprintf(stderr, "runtime-mem: sample %d: %v\n", i+1, err)
			return 1
		}
		got = append(got, r)
	}
	doc := buildRuntimeMemDoc(target, got)
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(doc); err != nil {
			fmt.Fprintf(stderr, "runtime-mem: encode: %v\n", err)
			return 1
		}
		return 0
	}
	writeRuntimeMemTable(stdout, doc)
	return 0
}

func runtimeMemVarsURL(base string) string {
	base = strings.TrimRight(base, "/")
	if strings.HasSuffix(base, "/debug/vars") {
		return base
	}
	return base + "/debug/vars"
}

// scrapeGoRuntime reads one bounded /debug/vars body and decodes only its
// go_runtime receipt (nested under runtime, or top-level for a bare receipt).
func scrapeGoRuntime(client *http.Client, url string, timeout time.Duration) (runtimeobs.Receipt, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return runtimeobs.Receipt{}, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return runtimeobs.Receipt{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return runtimeobs.Receipt{}, fmt.Errorf("GET %s: HTTP %d", url, resp.StatusCode)
	}
	var body struct {
		Runtime struct {
			GoRuntime json.RawMessage `json:"go_runtime"`
		} `json:"runtime"`
		GoRuntime json.RawMessage `json:"go_runtime"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, runtimeMemMaxBody)).Decode(&body); err != nil {
		return runtimeobs.Receipt{}, fmt.Errorf("decode %s: %w", url, err)
	}
	raw := body.Runtime.GoRuntime
	if len(raw) == 0 || string(raw) == "null" {
		raw = body.GoRuntime
	}
	if len(raw) == 0 || string(raw) == "null" {
		return runtimeobs.Receipt{}, errRuntimeMemNoReceipt
	}
	var r runtimeobs.Receipt
	if err := json.Unmarshal(raw, &r); err != nil {
		return runtimeobs.Receipt{}, fmt.Errorf("decode go_runtime: %w", err)
	}
	if r.Schema != runtimeobs.Schema {
		return runtimeobs.Receipt{}, fmt.Errorf("%w: got %q want %q", errRuntimeMemSchema, r.Schema, runtimeobs.Schema)
	}
	return r, nil
}

func runtimeMemStacks(m runtimeobs.Memory) runtimeobs.U64 {
	if !m.HeapStacksBytes.OK || !m.OSStacksBytes.OK {
		return runtimeobs.U64{}
	}
	return runtimeobs.U64{V: m.HeapStacksBytes.V + m.OSStacksBytes.V, OK: true}
}

// runtimeMemGauges is the closed gauge set, in table order.
func runtimeMemGauges(r runtimeobs.Receipt) []runtimeMemGaugeRow {
	return []runtimeMemGaugeRow{
		{Field: "memory.process_rss_bytes", Last: r.Memory.ProcessRSSBytes},
		{Field: "memory.total_mapped_bytes", Last: r.Memory.TotalMappedBytes},
		{Field: "memory.heap_objects_bytes", Last: r.Memory.HeapObjectsBytes},
		{Field: "gc.heap_live_bytes", Last: r.GC.HeapLiveBytes},
		{Field: "gc.heap_goal_bytes", Last: r.GC.HeapGoalBytes},
		{Field: "memory.stacks_bytes", Last: runtimeMemStacks(r.Memory)},
		{Field: "memory.heap_released_bytes", Last: r.Memory.HeapReleasedBytes},
	}
}

func buildRuntimeMemDoc(url string, rs []runtimeobs.Receipt) runtimeMemDoc {
	first, last := rs[0], rs[len(rs)-1]
	doc := runtimeMemDoc{Schema: runtimeMemSchema, URL: url, Samples: len(rs), First: first, Last: last}
	firstG, lastG := runtimeMemGauges(first), runtimeMemGauges(last)
	for i := range lastG {
		lastG[i].First = firstG[i].Last
	}
	doc.Gauges = lastG
	if len(rs) > 1 {
		d, err := runtimeobs.Diff(first, last)
		if err != nil {
			doc.DeltaError = err.Error()
		} else {
			doc.Delta = &d
		}
	}
	return doc
}

func fmtMiB(u runtimeobs.U64) string {
	if !u.OK {
		return "n/a"
	}
	return fmt.Sprintf("%.1f MiB", float64(u.V)/(1<<20))
}

func fmtU64(u runtimeobs.U64) string {
	if !u.OK {
		return "n/a"
	}
	return fmt.Sprintf("%d", u.V)
}

func fmtF64(f runtimeobs.F64, format string) string {
	if !f.OK {
		return "n/a"
	}
	return fmt.Sprintf(format, f.V)
}

// writeRuntimeMemTable prints at most 12 lines: header, gauges, limits, delta.
func writeRuntimeMemTable(w io.Writer, doc runtimeMemDoc) {
	l := doc.Last
	fmt.Fprintf(w, "go_runtime %s pid=%d seq=%d..%d samples=%d\n", l.GoVersion, l.PID, doc.First.Seq, l.Seq, doc.Samples)
	for _, g := range doc.Gauges {
		fmt.Fprintf(w, "  %-28s %12s -> %s\n", g.Field, fmtMiB(g.First), fmtMiB(g.Last))
	}
	limit := fmtMiB(l.GC.MemoryLimitBytes)
	if l.GC.MemoryLimitUnlimited {
		limit = "unlimited"
	}
	gogc := fmtU64(l.GC.GOGCPercent)
	if l.GC.GOGCOff {
		gogc = "off"
	}
	fmt.Fprintf(w, "  gc.memory_limit=%s gc.gogc=%s gc.cycles=%s goroutines=%s threads=%s\n",
		limit, gogc, fmtU64(l.GC.Cycles), fmtU64(l.Sched.Goroutines), fmtU64(l.Sched.Threads))
	switch {
	case doc.Delta != nil:
		d := doc.Delta
		fmt.Fprintf(w, "  delta %.1fs: gc_cycles=%s heap_allocs=%s gc_cpu_fraction=%s gc_pause_p99=%s\n",
			d.IntervalSeconds, fmtU64(d.GCCycles), fmtMiB(d.HeapAllocsBytes),
			fmtF64(d.GCCPUFraction, "%.4f"), fmtF64(d.GCPausesSummary.P99, "%.6fs"))
	case doc.DeltaError != "":
		fmt.Fprintf(w, "  delta: refused: %s\n", doc.DeltaError)
	}
	if n := len(l.Unavailable); n > 0 {
		fmt.Fprintf(w, "  unavailable: %d field(s), first %s (%s)\n", n, l.Unavailable[0].Field, l.Unavailable[0].Reason)
	}
}
