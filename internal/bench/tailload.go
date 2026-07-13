// tailload.go is the G5 tail-under-load benchmark arm (issue #2223, risk R6 of
// docs/notes/DYNAMIC-RANGE-ONE-BINARY-2026-07-01.md).
//
// Every B0 anchor in BENCHMARK-AUTHORITY is a quiet-process, single-flow number.
// In production ONE process runs the µs decide fold + SSE streaming + session
// bookkeeping on one Go GC and scheduler. This arm measures the SAME in-process
// kernel.Decide fold the 100 µs distribution gate covers
// (internal/gateway/adjudication_latency_test.go), twice in one process:
//
//	quiet  — nothing else on the scheduler (the existing anchors' shape)
//	loaded — the SAME process concurrently drives synthetic B2/B3-shaped work:
//	         SSE-style streaming churn (allocation + encoding pressure) and
//	         session-table churn (lock + map + budget-fold pressure), per the
//	         issue's definition ("the same process drives the synthetic B2/B3
//	         work, not a separate external load generator").
//
// The readout is the per-call latency DISTRIBUTION (exact percentiles over raw
// samples, not histogram buckets — a bucketed p99 would quantize exactly the
// tail this arm exists to see). Per-call time.Now timing adds ~2 clock reads
// (~50 ns on Linux vDSO) per sample; that overhead is identical in both arms,
// so the quiet-vs-loaded delta is clean even though absolute numbers carry it.
//
// Research-grade fence: results land OBSERVED with hardware named; this arm
// asserts and flips NO gate (out of scope for #2223 — no promotion until two
// runs agree).
package bench

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/session"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

// TailLoadSchema versions the tail-under-load artifact.
const TailLoadSchema = "fak.tailload.v1"

// TailArm is one measured arm's latency distribution, exact over raw samples.
type TailArm struct {
	Label   string  `json:"label"`
	Samples int     `json:"samples"`
	P50NS   int64   `json:"p50_ns"`
	P90NS   int64   `json:"p90_ns"`
	P99NS   int64   `json:"p99_ns"`
	P999NS  int64   `json:"p999_ns"`
	MaxNS   int64   `json:"max_ns"`
	MeanNS  float64 `json:"mean_ns"`
	// Over100us counts samples over the existing gate's 100 µs bar — context
	// against the standing single-flow gate, NOT a gate (no flip here).
	Over100us int `json:"over_100us"`
}

// TailLoadConfig sizes the run. Zero values take the defaults below.
type TailLoadConfig struct {
	Samples       int `json:"samples"`
	Warmup        int `json:"warmup"`
	StreamWorkers int `json:"stream_workers"`
	ChurnWorkers  int `json:"churn_workers"`
	// SessionIDs bounds the per-churn-worker session-id set, so the shared
	// table exercises steady-state update churn, not unbounded growth.
	SessionIDs int `json:"session_ids"`
}

func (c TailLoadConfig) withDefaults() TailLoadConfig {
	if c.Samples <= 0 {
		c.Samples = 200_000
	}
	if c.Warmup <= 0 {
		c.Warmup = 5_000
	}
	if c.StreamWorkers <= 0 {
		c.StreamWorkers = mathx.MaxInt(2, runtime.NumCPU()/4)
	}
	if c.ChurnWorkers <= 0 {
		c.ChurnWorkers = mathx.MaxInt(2, runtime.NumCPU()/4)
	}
	if c.SessionIDs <= 0 {
		c.SessionIDs = 512
	}
	return c
}

// TailLoadReport is the committed artifact: quiet vs loaded distributions of
// the same decide fold in the same process, provenance-labeled.
type TailLoadReport struct {
	Schema      string         `json:"schema"`
	Issue       string         `json:"issue"`
	GeneratedAt string         `json:"generated_at"`
	Hardware    string         `json:"hardware"`
	GoOS        string         `json:"goos"`
	GoArch      string         `json:"goarch"`
	NumCPU      int            `json:"num_cpu"`
	GoMaxProcs  int            `json:"gomaxprocs"`
	GoVersion   string         `json:"go_version"`
	Tool        string         `json:"tool"` // the decided call (canonical allow)
	Config      TailLoadConfig `json:"config"`
	Fence       string         `json:"fence"`
	Quiet       TailArm        `json:"quiet"`
	Loaded      TailArm        `json:"loaded"`
}

const tailFence = "OBSERVED, research-grade (#2223): per-call kernel.Decide latency, single measuring flow, quiet vs same-process synthetic B2/B3 load; no gate asserted or flipped; timer overhead identical in both arms"

// RunTailUnderLoad measures the decide fold quiet, then under same-process
// synthetic load, and returns the folded report. Hermetic: no network, no
// model, no filesystem writes.
func RunTailUnderLoad(ctx context.Context, cfg TailLoadConfig) (*TailLoadReport, error) {
	cfg = cfg.withDefaults()
	rep := &TailLoadReport{
		Schema:      TailLoadSchema,
		Issue:       "#2223 (epic #2218, gap G5, risk R6)",
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		Hardware:    hardwareLabel(),
		GoOS:        runtime.GOOS,
		GoArch:      runtime.GOARCH,
		NumCPU:      runtime.NumCPU(),
		GoMaxProcs:  runtime.GOMAXPROCS(0),
		GoVersion:   runtime.Version(),
		Tool:        "get_user_details",
		Config:      cfg,
		Fence:       tailFence,
	}

	k := kernel.New("mock")
	res := abi.ActiveResolver()
	ref, err := res.Put(ctx, []byte(`{"user_id":"mia_li_3668"}`))
	if err != nil {
		return nil, fmt.Errorf("tailload: resolver put: %w", err)
	}
	tc := &abi.ToolCall{Tool: rep.Tool, Args: ref, Meta: readHints}
	if v := k.Decide(ctx, tc); v.Kind != abi.VerdictAllow {
		// Measuring a silent deny would time the wrong (cheaper) branch.
		return nil, fmt.Errorf("tailload: canonical call verdict = %s, want ALLOW", verdictName(v.Kind))
	}

	measure := func(label string) TailArm {
		for i := 0; i < cfg.Warmup; i++ {
			_ = k.Decide(ctx, tc)
		}
		samples := make([]int64, cfg.Samples)
		for i := range samples {
			t0 := time.Now()
			_ = k.Decide(ctx, tc)
			samples[i] = time.Since(t0).Nanoseconds()
		}
		return foldTailArm(label, samples)
	}

	rep.Quiet = measure("quiet")

	stop := make(chan struct{})
	var wg sync.WaitGroup
	for w := 0; w < cfg.StreamWorkers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			streamChurn(stop, w)
		}(w)
	}
	tbl := session.NewTable()
	for w := 0; w < cfg.ChurnWorkers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			sessionChurn(stop, tbl, w, cfg.SessionIDs)
		}(w)
	}
	rep.Loaded = measure("loaded")
	close(stop)
	wg.Wait()

	return rep, nil
}

// streamChurn is the synthetic B2-shaped load: SSE-style event encoding with
// realistic allocation pressure (marshal + frame + buffer growth), reset
// periodically so it exercises the GC, not the heap ceiling.
func streamChurn(stop <-chan struct{}, worker int) {
	type chunk struct {
		Seq   int    `json:"seq"`
		Delta string `json:"delta"`
	}
	var buf bytes.Buffer
	payload := "streamed token delta payload for synthetic B2 load — worker " + strconv.Itoa(worker)
	for seq := 0; ; seq++ {
		select {
		case <-stop:
			return
		default:
		}
		b, _ := json.Marshal(chunk{Seq: seq, Delta: payload})
		buf.WriteString("event: message\ndata: ")
		buf.Write(b)
		buf.WriteString("\n\n")
		if buf.Len() > 1<<18 {
			buf.Reset()
		}
	}
}

// sessionChurn is the synthetic B3-shaped load: live session-table bookkeeping
// (decide + token debit) over a bounded id set on ONE shared table, so the
// churn contends the same lock the serving path would.
func sessionChurn(stop <-chan struct{}, tbl *session.Table, worker, ids int) {
	prefix := "tailload-" + strconv.Itoa(worker) + "-"
	for i := 0; ; i++ {
		select {
		case <-stop:
			return
		default:
		}
		id := prefix + strconv.Itoa(i%ids)
		_ = tbl.Decide(id)
		_ = tbl.Debit(id, 37)
	}
}

func foldTailArm(label string, samples []int64) TailArm {
	sorted := make([]int64, len(samples))
	copy(sorted, samples)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	var sum, over int64
	for _, s := range sorted {
		sum += s
		if s > 100_000 { // the existing gate's 100 µs bar, context only
			over++
		}
	}
	pct := func(p float64) int64 {
		idx := int(float64(len(sorted)) * p)
		if idx >= len(sorted) {
			idx = len(sorted) - 1
		}
		return sorted[idx]
	}
	return TailArm{
		Label:     label,
		Samples:   len(sorted),
		P50NS:     pct(0.50),
		P90NS:     pct(0.90),
		P99NS:     pct(0.99),
		P999NS:    pct(0.999),
		MaxNS:     sorted[len(sorted)-1],
		MeanNS:    float64(sum) / float64(len(sorted)),
		Over100us: int(over),
	}
}

// hardwareLabel names the run host for the research-grade OBSERVED label. The
// operator sets FAK_BENCH_HW (e.g. "AMD Ryzen 9 9950X, WSL2 Ubuntu-24.04");
// an unset label is surfaced honestly rather than guessed.
func hardwareLabel() string {
	if hw := os.Getenv("FAK_BENCH_HW"); hw != "" {
		return hw
	}
	return "unspecified (set FAK_BENCH_HW to name the host)"
}
