package main

// deepseekbench.go — `fak deepseekbench`, the thin WIRE over internal/deepseekbench:
// the DeepSeek V4 Pro/Flash TTFT/TPOT/context-scaling SCORECARD (#3014, program
// #3006; complements the self-host runbook #3013).
//
// The schema, dry-run fixture, speedup-refusal gate, and live measurement all live
// in the pure, isolation-testable internal/deepseekbench package. This file is only
// the flag/I-O shim: default is a NO-KEY DRY-RUN FIXTURE (JSONL rows → stdout,
// coverage + honesty → stderr); a live run is opt-in and doubly gated behind
// DEEPSEEK_API_KEY + an explicit --spend acknowledgement.
//
//	fak deepseekbench                             # no-key dry-run fixture → JSONL
//	fak deepseekbench --out rows.jsonl            # write the JSONL rows to a file
//	DEEPSEEK_API_KEY=… fak deepseekbench --live --spend --model deepseek-v4-pro
//	DEEPSEEK_API_KEY=… fak deepseekbench --live --spend \
//	  --base-url http://host:8000/v1 --model deepseek-ai/DeepSeek-V4-Pro   # self-hosted arm

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/deepseekbench"
)

func cmdDeepSeekBench(argv []string) { os.Exit(runDeepSeekBench(os.Stdout, os.Stderr, argv)) }

// runDeepSeekBench is the testable core (exit code, no os.Exit): 0 ok, 1 a run error,
// 2 a usage/gating error.
func runDeepSeekBench(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("deepseekbench", flag.ContinueOnError)
	fs.SetOutput(stderr)
	live := fs.Bool("live", false, "opt-in LIVE run (requires DEEPSEEK_API_KEY and --spend); default is a no-key dry-run fixture")
	spend := fs.Bool("spend", false, "explicit acknowledgement that a --live run costs money")
	baseURL := fs.String("base-url", "https://api.deepseek.com", "OpenAI-compatible root for a --live run (point at a self-hosted vLLM/SGLang endpoint for the self-hosted arm)")
	model := fs.String("model", deepseekbench.ModelV4Flash, "model id to route for a --live run")
	outPath := fs.String("out", "", "write the JSONL rows to this file (default: stdout)")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}

	var rows []deepseekbench.Row
	if *live {
		key := strings.TrimSpace(os.Getenv("DEEPSEEK_API_KEY"))
		if msg, ok := deepseekbench.LiveGate(key != "", *spend); !ok {
			fmt.Fprintln(stderr, "fak deepseekbench:", msg)
			return 2
		}
		row, err := deepseekbench.MeasureStreamed(nil, *baseURL, key, *model)
		if err != nil {
			fmt.Fprintln(stderr, "fak deepseekbench: live run failed:", err)
			return 1
		}
		rows = []deepseekbench.Row{row}
	} else {
		rows = deepseekbench.DryRunRows()
	}

	sink := stdout
	if *outPath != "" {
		f, err := os.Create(*outPath)
		if err != nil {
			fmt.Fprintln(stderr, "fak deepseekbench:", err)
			return 1
		}
		defer f.Close()
		sink = f
	}
	if err := writeDeepSeekJSONL(sink, rows); err != nil {
		fmt.Fprintln(stderr, "fak deepseekbench:", err)
		return 1
	}

	printDeepSeekSummary(stderr, rows, *live)
	return 0
}

// writeDeepSeekJSONL emits one compact JSON object per line — the locked scorecard format.
func writeDeepSeekJSONL(w io.Writer, rows []deepseekbench.Row) error {
	bw := bufio.NewWriter(w)
	enc := json.NewEncoder(bw)
	for _, r := range rows {
		if err := enc.Encode(r); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// printDeepSeekSummary writes the human-facing coverage + honesty summary to stderr
// (the JSONL rows themselves are the machine artifact on stdout/--out).
func printDeepSeekSummary(w io.Writer, rows []deepseekbench.Row, live bool) {
	models := map[string]int{}
	for _, r := range rows {
		models[r.ModelID]++
	}
	ids := make([]string, 0, len(models))
	for id := range models {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	fmt.Fprintln(w, "== fak deepseekbench — DeepSeek V4 Pro/Flash TTFT/TPOT/context scorecard (#3014) ==")
	if live {
		fmt.Fprintln(w, "mode: LIVE (measurement=\"live\"; latency is provider-observed)")
	} else {
		fmt.Fprintln(w, "mode: DRY-RUN FIXTURE (measurement=\"dry-run-fixture\"; every latency is a labelled placeholder, NOT measured)")
	}
	fmt.Fprintf(w, "rows: %d across models:\n", len(rows))
	for _, id := range ids {
		fmt.Fprintf(w, "  %-22s %d rows\n", id, models[id])
	}
	// Demonstrate the speedup-refusal gate on a DeepSeek row vs a non-DeepSeek baseline.
	fmt.Fprintf(w, "scorecard speedup gate: %s\n", deepSeekGateExample(rows))
	fmt.Fprintln(w, "OBSERVED provider speed is never a fak-authored saving — see docs/benchmarks/DEEPSEEK-V4-PERF-SCORECARD.md")
	if !live {
		fmt.Fprintln(w, "live recipe: DEEPSEEK_API_KEY=… fak deepseekbench --live --spend --model deepseek-v4-pro")
	}
}

// deepSeekGateExample runs the refusal gate over the first DeepSeek row and the first
// non-DeepSeek baseline row, so the summary always shows what the gate decides.
func deepSeekGateExample(rows []deepseekbench.Row) string {
	var subject, baseline *deepseekbench.Row
	for i := range rows {
		if rows[i].ProviderRoute == "deepseek" && subject == nil {
			subject = &rows[i]
		}
		if rows[i].ProviderRoute != "deepseek" && baseline == nil {
			baseline = &rows[i]
		}
	}
	if subject == nil || baseline == nil {
		return "[NOT COMPARABLE: need one DeepSeek row and one baseline row]"
	}
	line, _ := deepseekbench.CompareSpeedup(*subject, *baseline)
	return line
}
