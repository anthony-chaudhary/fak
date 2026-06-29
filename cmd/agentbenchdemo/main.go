// Command agentbenchdemo is the performance micro-benchmark of fak's agentic spine:
// how much does the kernel's per-tool-call adjudication actually COST? It folds a
// fixed plan of tool calls through the REAL kernel (the same internal/agentdemo path
// cmd/timewolfdemo and `fak preflight` use — adjudicator.Default.SetPolicy + a live
// kernel.Fold per call), times the loop, and reports the per-call adjudication cost
// — the "self-tax" the safety floor adds to an agent's critical path.
//
// The headline is a net-value number, not a micro-optimisation: a single LLM
// tool-calling round-trip is ~1–3 s, while the kernel adjudicates a tool call in a
// handful of microseconds — so the floor is, in practice, free. The benchmark proves
// that on the box you run it on, with no model, no key, no network.
//
// The WORKLOAD is deterministic (the same plan, the same allow/deny outcome every
// run — 1 allow + 2 deny per iteration); only the measured wall-clock varies by box.
//
// Headless, three ways (no browser, no model):
//
//	go run ./cmd/agentbenchdemo            # the self-tax table in the terminal
//	go run ./cmd/agentbenchdemo -json      # the measured result as JSON
//	go run ./cmd/agentbenchdemo -selfcheck # assert the self-tax invariants (CI gate)
//	go run ./cmd/agentbenchdemo -n 20000   # more iterations for a tighter mean
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentdemo"
	"github.com/anthony-chaudhary/fak/internal/demoui"

	// Wire the full ABI (resolver, vDSO, adjudicator, ctx-MMU, normgate, IFC,
	// witness, engines) before kernel.Fold runs inside agentdemo.Run.
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

// llmRoundTripNs is the conservative reference for one LLM tool-calling round-trip
// (1.5 s) used only to frame the per-call adjudication cost as a ratio. It is a
// deliberately low end of the ~1–3 s range, so the "N× cheaper" claim is an
// under-statement, never an over-statement.
const llmRoundTripNs = int64(1_500_000_000)

// benchToolset is the fixed capability floor the benchmark folds against: the
// read-only get_ family is allowed, delete_calendar is an explicit deny
// (POLICY_BLOCK), and wipe_disk is left off the floor so it falls to fail-closed
// DEFAULT_DENY — one allow + two denies per iteration, the same shape the safety
// demos use, so the measured cost is the real mixed allow/deny path.
func benchToolset() *agentdemo.Toolset {
	return agentdemo.NewToolset(
		agentdemo.Floor{
			AllowPrefix: []string{"get_"},
			Deny:        []string{"delete_calendar"},
		},
		agentdemo.Tool{Name: "get_time", Summary: "read-only call — ALLOWed", Handler: func(json.RawMessage) string { return "11:58 AM" }},
		agentdemo.Tool{Name: "delete_calendar", Summary: "explicit floor deny — POLICY_BLOCK"},
		agentdemo.Tool{Name: "wipe_disk", Summary: "off-floor sink — DEFAULT_DENY"},
	)
}

// plan is the fixed per-iteration workload: one allowed read, one explicitly denied
// destructive call, one off-floor denied call. Deterministic by construction.
func plan() []agentdemo.Step {
	return []agentdemo.Step{
		{Tool: "get_time"},
		{Tool: "delete_calendar"},
		{Tool: "wipe_disk"},
	}
}

// result is the measured benchmark outcome. The counts are deterministic; the timing
// fields are measured on the running box.
type result struct {
	Iterations  int     `json:"iterations"`
	Calls       int     `json:"calls"`         // adjudicated tool calls = iterations × len(plan)
	Allowed     int     `json:"allowed"`       // ALLOWed calls (1 per iteration)
	Denied      int     `json:"denied"`        // refused calls (2 per iteration)
	NsPerCall   int64   `json:"ns_per_call"`   // mean wall-clock per adjudicated call
	CallsPerSec int64   `json:"calls_per_sec"` // throughput
	TotalMs     float64 `json:"total_ms"`      // total measured wall-clock
	CheaperBy   int64   `json:"cheaper_than_llm_roundtrip_x"`
}

// runBench folds the fixed plan `iters` times through the real kernel and returns the
// measured per-call adjudication cost. A single warmup iteration primes the
// process-global adjudicator policy and resolver so the timed loop measures steady
// state, not first-call wiring.
func runBench(ctx context.Context, ts *agentdemo.Toolset, iters int) (result, error) {
	p := plan()
	if _, err := ts.Run(ctx, "warmup", "warm", p); err != nil {
		return result{}, fmt.Errorf("warmup: %w", err)
	}
	var allowed, denied int
	start := time.Now()
	for i := 0; i < iters; i++ {
		tr, err := ts.Run(ctx, "bench", "what time is it?", p)
		if err != nil {
			return result{}, fmt.Errorf("iteration %d: %w", i, err)
		}
		allowed += tr.Allowed
		denied += tr.Denied
	}
	elapsed := time.Since(start)
	calls := iters * len(p)
	nsPerCall := int64(1)
	if calls > 0 && elapsed > 0 {
		nsPerCall = elapsed.Nanoseconds() / int64(calls)
	}
	cps := int64(0)
	if elapsed > 0 {
		cps = int64(float64(calls) / elapsed.Seconds())
	}
	cheaper := int64(0)
	if nsPerCall > 0 {
		cheaper = llmRoundTripNs / nsPerCall
	}
	return result{
		Iterations:  iters,
		Calls:       calls,
		Allowed:     allowed,
		Denied:      denied,
		NsPerCall:   nsPerCall,
		CallsPerSec: cps,
		TotalMs:     float64(elapsed.Microseconds()) / 1000.0,
		CheaperBy:   cheaper,
	}, nil
}

// render writes the human-facing self-tax table.
func render(r result) {
	fmt.Println("agentbenchdemo · the self-tax: what the kernel costs per tool call")
	fmt.Printf("hardware: %s\n\n", demoui.Probe().Summary)
	fmt.Printf("  iterations:        %d   (×%d calls = %d adjudicated tool calls)\n", r.Iterations, len(plan()), r.Calls)
	fmt.Printf("  allowed / denied:  %d / %d   (1 allow + 2 deny per iteration)\n", r.Allowed, r.Denied)
	fmt.Printf("  per call:          %s   (mean, this box)\n", humanNs(r.NsPerCall))
	fmt.Printf("  throughput:        %s adjudicated calls/sec\n", commas(r.CallsPerSec))
	fmt.Printf("  total wall-clock:  %.1f ms\n\n", r.TotalMs)
	fmt.Printf("  net: a single LLM tool-calling round-trip is ~1–3 s. The kernel adjudicates a\n")
	fmt.Printf("       tool call in %s — about %s× cheaper, so the safety floor is effectively\n", humanNs(r.NsPerCall), commas(r.CheaperBy))
	fmt.Printf("       free on the agent's critical path (the floor never gates the LLM, only the call).\n")
}

// humanNs renders a nanosecond duration as µs/ms/ns, whichever reads cleanest.
func humanNs(ns int64) string {
	switch {
	case ns >= 1_000_000:
		return fmt.Sprintf("~%.2f ms", float64(ns)/1_000_000)
	case ns >= 1_000:
		return fmt.Sprintf("~%.2f µs", float64(ns)/1_000)
	default:
		return fmt.Sprintf("~%d ns", ns)
	}
}

// commas renders an integer with thousands separators (no locale dependency).
func commas(n int64) string {
	s := fmt.Sprintf("%d", n)
	if n < 0 {
		return s
	}
	out := make([]byte, 0, len(s)+len(s)/3)
	for i := 0; i < len(s); i++ {
		if i > 0 && (len(s)-i)%3 == 0 {
			out = append(out, ',')
		}
		out = append(out, s[i])
	}
	return string(out)
}

func main() {
	fs := flag.NewFlagSet("agentbenchdemo", flag.ExitOnError)
	n := fs.Int("n", 4000, "iterations to fold (each is one allowed + two denied tool calls)")
	doJSON := fs.Bool("json", false, "emit the measured result as JSON and exit")
	doSelfcheck := fs.Bool("selfcheck", false, "assert the self-tax invariants (deterministic counts, sane per-call bound) and exit non-zero on drift")
	_ = fs.Parse(os.Args[1:])

	ctx := context.Background()
	ts := benchToolset()

	if *doSelfcheck {
		os.Exit(selfcheck(ctx, ts))
	}

	iters := *n
	if iters < 1 {
		iters = 1
	}
	r, err := runBench(ctx, ts, iters)
	if err != nil {
		fmt.Fprintln(os.Stderr, "agentbenchdemo:", err)
		os.Exit(1)
	}
	if *doJSON {
		b, _ := json.MarshalIndent(r, "", "  ")
		fmt.Println(string(b))
		return
	}
	render(r)
}

// selfcheck runs a small bench and asserts the SELF-TAX invariants that hold on any
// box: the deterministic call accounting (1 allow + 2 deny per iteration) and a sane,
// non-flaky per-call ceiling (the loop produced real, bounded timing — it neither
// hung nor returned a zero/absurd cost). It deliberately does NOT assert an absolute
// latency, which is box-dependent; the structural invariants are what gate.
func selfcheck(ctx context.Context, ts *agentdemo.Toolset) int {
	const iters = 300
	r, err := runBench(ctx, ts, iters)
	if err != nil {
		fmt.Fprintln(os.Stderr, "FAIL:", err)
		return 1
	}
	var c demoui.SelfcheckChecker
	c.Check("calls", r.Calls, iters*3)
	c.Check("allowed", r.Allowed, iters)
	c.Check("denied", r.Denied, iters*2)
	if r.NsPerCall <= 0 {
		c.Notef("ns_per_call=%d, want > 0 (the timed loop produced no measurable cost)", r.NsPerCall)
	}
	// 50 ms/call is a generous anti-hang ceiling — orders of magnitude above any real
	// adjudication cost, so it never flakes on a loaded CI box but still catches a
	// pathological regression or a stalled fold.
	if r.NsPerCall >= 50_000_000 {
		c.Notef("ns_per_call=%d ns (>= 50 ms) — adjudication is implausibly slow, likely a regression", r.NsPerCall)
	}
	if c.Failed() {
		fmt.Fprintf(os.Stderr, "agentbenchdemo -selfcheck: FAIL: %v\n", c.Mismatches())
		return 1
	}
	fmt.Printf("agentbenchdemo -selfcheck: the self-tax invariants hold "+
		"(%d calls · %d allowed · %d denied · %s/call on this box)\n",
		r.Calls, r.Allowed, r.Denied, humanNs(r.NsPerCall))
	return 0
}
