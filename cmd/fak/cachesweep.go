package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/cachesweep"
)

// runCachesweep implements `fak cachesweep` — the cache-budget→reuse sweep filed as #3952
// (the tair-kvcache borrow; see docs/notes/tair-kvcache-borrow-study-2026-07-10.md). It
// replays ONE recorded prefix-access trace across the requested cached-token budgets PLUS
// one unbounded pass, then reports the reuse-vs-budget curve, the infinite-cache ceiling,
// and the smallest budget reaching 99% of it (the ROI knee) — so the radixkv LRU budget
// and --compact-history-budget can be sized from evidence instead of intuition.
//
//	fak cachesweep --trace trace.jsonl --budgets 64,128,256,512,1024
//	fak cachesweep --trace - --budgets 128,256 --json           # read the trace from stdin
//	fak cachesweep --trace trace.jsonl --budgets 128,256 --write-delay-ns 4
//
// The trace is JSONL: each non-empty line is either a bare token-id array ("[1,2,3]") or
// an object ("{\"tokens\":[1,2,3],\"t_ns\":10}"). A line with no t_ns is stamped with its
// 0-based ordinal, so --write-delay-ns is then measured in access-steps.
func runCachesweep(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak cachesweep", flag.ContinueOnError)
	fs.SetOutput(stderr)
	tracePath := fs.String("trace", "", "recorded prefix-access trace, JSONL (- for stdin)")
	budgetsCSV := fs.String("budgets", "", "comma-separated cached-token budgets to sweep, e.g. 64,128,256,512")
	writeDelay := fs.Int64("write-delay-ns", 0, "optional KV write-delay window: a re-request inside it counts as a miss (0 = off)")
	kneeFrac := fs.Float64("knee-fraction", cachesweep.DefaultKneeFraction, "knee threshold as a fraction of the infinite-cache ceiling")
	asJSON := fs.Bool("json", false, "emit the sweep Result as JSON instead of a table")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak cachesweep: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *tracePath == "" {
		fmt.Fprintln(stderr, "fak cachesweep: --trace is required (a JSONL prefix-access trace; - for stdin)")
		return 2
	}
	budgets, err := parseIntListCSV(*budgetsCSV)
	if err != nil {
		fmt.Fprintf(stderr, "fak cachesweep: --budgets: %v\n", err)
		return 2
	}
	if len(budgets) == 0 {
		fmt.Fprintln(stderr, "fak cachesweep: --budgets is required (e.g. --budgets 64,128,256,512)")
		return 2
	}

	tr, err := loadCachesweepTrace(*tracePath)
	if err != nil {
		fmt.Fprintf(stderr, "fak cachesweep: %v\n", err)
		return 2
	}
	if len(tr.Accesses) == 0 {
		fmt.Fprintf(stderr, "fak cachesweep: trace %q is empty\n", *tracePath)
		return 2
	}

	res := cachesweep.Sweep(tr, cachesweep.Options{
		Budgets:      budgets,
		KneeFraction: *kneeFrac,
		WriteDelayNs: *writeDelay,
	})

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(res); err != nil {
			fmt.Fprintf(stderr, "fak cachesweep: encode: %v\n", err)
			return 1
		}
		return 0
	}
	printCachesweepReport(stdout, res)
	return 0
}

// printCachesweepReport renders the human-readable curve + ceiling + knee table.
func printCachesweepReport(w io.Writer, res cachesweep.Result) {
	delay := ""
	if res.WriteDelayNs > 0 {
		delay = fmt.Sprintf(", write-delay %d", res.WriteDelayNs)
	}
	fmt.Fprintf(w, "== fak cachesweep: %d accesses, %d tokens%s ==\n",
		res.Accesses, res.TotalTokens, delay)
	fmt.Fprintf(w, "infinite-cache ceiling: reuse %.4f (%d/%d, unbounded)\n\n",
		res.Ceiling.ReuseRatio, res.Ceiling.ReusedTokens, res.Ceiling.TotalTokens)

	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "budget\treuse\treused/total\tevictions\tpct-of-ceiling\t")
	for _, p := range res.Curve {
		pct := 0.0
		if res.Ceiling.ReuseRatio > 0 {
			pct = 100 * p.ReuseRatio / res.Ceiling.ReuseRatio
		}
		fmt.Fprintf(tw, "%d\t%.4f\t%d/%d\t%d\t%.1f%%\t\n",
			p.Budget, p.ReuseRatio, p.ReusedTokens, p.TotalTokens, p.Evictions, pct)
	}
	tw.Flush()

	fmt.Fprintln(w)
	if res.KneeReached {
		fmt.Fprintf(w, "knee (>=%.0f%% of ceiling): budget %d  reuse %.4f\n",
			res.KneeFraction*100, res.Knee.Budget, res.Knee.ReuseRatio)
	} else if res.Ceiling.ReuseRatio <= 0 {
		fmt.Fprintln(w, "knee: n/a (ceiling reuse is 0 — the trace has no reuse to size a budget against)")
	} else {
		fmt.Fprintf(w, "knee: not reached — no swept budget hit %.0f%% of the ceiling; try larger budgets\n",
			res.KneeFraction*100)
	}
}

// loadCachesweepTrace reads a JSONL prefix-access trace from a file (or stdin for "-").
// Each non-empty, non-"#" line is a bare token-id array or a {tokens,t_ns} object; a line
// without t_ns is stamped with its 0-based ordinal (so a delay is measured in access-steps).
func loadCachesweepTrace(path string) (cachesweep.Trace, error) {
	var r io.Reader
	if path == "-" {
		r = os.Stdin
	} else {
		f, err := os.Open(path)
		if err != nil {
			return cachesweep.Trace{}, err
		}
		defer f.Close()
		r = f
	}

	var tr cachesweep.Trace
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 32*1024*1024) // tolerate long token-id lines
	ordinal := int64(0)
	for lineNo := 1; sc.Scan(); lineNo++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		acc, hasTime, err := parseTraceLine(line)
		if err != nil {
			return cachesweep.Trace{}, fmt.Errorf("trace line %d: %w", lineNo, err)
		}
		if !hasTime {
			acc.TimeNs = ordinal
		}
		tr.Accesses = append(tr.Accesses, acc)
		ordinal++
	}
	if err := sc.Err(); err != nil {
		return cachesweep.Trace{}, err
	}
	return tr, nil
}

// parseTraceLine parses one JSONL trace line. It accepts a bare token-id array
// ("[1,2,3]") and an object ("{\"tokens\":[...],\"t_ns\":N}"); hasTime reports whether the
// line carried an explicit t_ns (so the caller can default it to the access ordinal).
func parseTraceLine(line string) (acc cachesweep.Access, hasTime bool, err error) {
	if strings.HasPrefix(line, "[") {
		var toks []int
		if err := json.Unmarshal([]byte(line), &toks); err != nil {
			return acc, false, fmt.Errorf("token array: %w", err)
		}
		return cachesweep.Access{Tokens: toks}, false, nil
	}
	var wire struct {
		Tokens []int  `json:"tokens"`
		TNs    *int64 `json:"t_ns"`
	}
	if err := json.Unmarshal([]byte(line), &wire); err != nil {
		return acc, false, fmt.Errorf("access object: %w", err)
	}
	acc = cachesweep.Access{Tokens: wire.Tokens}
	if wire.TNs != nil {
		acc.TimeNs = *wire.TNs
		hasTime = true
	}
	return acc, hasTime, nil
}

// parseIntListCSV parses "64,128,256" into []int, tolerating spaces and empty fields.
func parseIntListCSV(s string) ([]int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	var out []int
	for _, field := range strings.Split(s, ",") {
		field = strings.TrimSpace(field)
		if field == "" {
			continue
		}
		n, err := strconv.Atoi(field)
		if err != nil {
			return nil, fmt.Errorf("%q is not an integer", field)
		}
		out = append(out, n)
	}
	return out, nil
}
