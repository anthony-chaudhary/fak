package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// fak routebench — the OFFLINE ROUTING BENCHMARK. It is to "does per-aspect +
// ensemble routing actually pay off?" what `fak route` is to "which model routes
// this aspect?": run a CORPUS of recorded cases through TWO manifests — a routed
// policy (per-aspect + ensemble) and a single-model baseline (the SOTA shape) —
// and print the delta on three axes: COST (rough usage saved), LATENCY (rough
// total compute), and QUALITY (fraction matching the expected answer).
//
// OFFLINE means no model in the loop: each case carries the stand-in OUTPUT every
// candidate model produces, exactly like `fak route --simulate`. So it reuses the
// pure Route + Combine halves and is deterministic end to end — no key, no GPU.
//
//	fak routebench                                  (the built-in 8-case demo corpus + manifests)
//	fak routebench --corpus FILE [--routed F] [--single F]
//	fak routebench --json                           (machine-readable comparison)
//	fak routebench --prices small=0.25/1.25 --latencies small=20,large=120
//
// Every figure is a ROUGH lens over a recorded corpus, never a bill or a measured
// SLA — see the cost lens honesty fences in `fak route`.
func cmdRoutebench(argv []string) { os.Exit(runRoutebench(os.Stdout, os.Stderr, argv)) }

// runRoutebench is the testable core: it returns the process exit code (0 ok,
// 1 a load error, 2 a usage error) instead of calling os.Exit.
func runRoutebench(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("routebench", flag.ContinueOnError)
	fs.SetOutput(stderr)
	corpusPath := fs.String("corpus", "", "JSON corpus of cases (subject + per-model stand-in outputs + expected); default: built-in 8-case demo")
	routedPath := fs.String("routed", "", "the per-aspect + ensemble manifest under test (default: built-in DefaultManifest)")
	singlePath := fs.String("single", "", "the single-model baseline manifest (default: a one-frontier-model manifest)")
	frontier := fs.String("frontier", "frontier", "the baseline model id for the rough cost/latency lenses")
	prices := fs.String("prices", "", "override the rough price book: model=in/out[,model=N,...] (same grammar as 'fak route')")
	latencies := fs.String("latencies", "", "override the rough latency book: model=ms[,model=ms,...]")
	asJSON := fs.Bool("json", false, "emit the comparison as JSON")
	dumpCorpus := fs.Bool("dump-corpus", false, "write the built-in demo corpus as JSON to stdout (to edit into your own)")
	if err := fs.Parse(argv); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}

	if *dumpCorpus {
		stdout.Write(modelroute.DemoCorpus().JSON())
		return 0
	}

	// The two arms: routed (per-aspect + ensemble) vs single (one frontier model).
	routed := modelroute.DefaultManifest()
	if *routedPath != "" {
		m, err := modelroute.LoadManifest(*routedPath)
		if err != nil {
			fmt.Fprintln(stderr, "fak routebench:", err)
			return 1
		}
		routed = m
		fmt.Fprintf(stderr, "fak: loaded routed manifest from %s\n", *routedPath)
	}
	single := modelroute.SingleModelManifest(*frontier)
	if *singlePath != "" {
		m, err := modelroute.LoadManifest(*singlePath)
		if err != nil {
			fmt.Fprintln(stderr, "fak routebench:", err)
			return 1
		}
		single = m
		fmt.Fprintf(stderr, "fak: loaded single-model manifest from %s\n", *singlePath)
	}

	// The corpus: an explicit file, else the built-in demo.
	var corpus modelroute.Corpus
	corpusSrc := "built-in 8-case demo"
	if *corpusPath != "" {
		c, err := modelroute.LoadCorpus(*corpusPath)
		if err != nil {
			fmt.Fprintln(stderr, "fak routebench:", err)
			return 1
		}
		corpus = c
		corpusSrc = *corpusPath
	} else {
		corpus = modelroute.DemoCorpus()
	}

	// The two rough lenses: the default books overlaid with any operator overrides.
	book := modelroute.DefaultPrices()
	if *prices != "" {
		over, err := modelroute.ParsePrices(*prices)
		if err != nil {
			fmt.Fprintln(stderr, "fak routebench:", err)
			return 2
		}
		book = book.Overlay(over)
	}
	lat := modelroute.DefaultLatencies()
	if *latencies != "" {
		over, err := modelroute.ParseLatencies(*latencies)
		if err != nil {
			fmt.Fprintln(stderr, "fak routebench:", err)
			return 2
		}
		lat = lat.Overlay(over)
	}

	cmp := corpus.Compare(routed, single, book, lat, *frontier)
	if *asJSON {
		fmt.Fprintln(stdout, routebenchJSON(cmp, corpusSrc))
		return 0
	}
	printRoutebench(stdout, cmp, corpusSrc)
	return 0
}

// printRoutebench renders the three-axis comparison for a human. ASCII only.
func printRoutebench(w io.Writer, cmp modelroute.Comparison, corpusSrc string) {
	fmt.Fprintln(w, "== fak routebench — offline routing benchmark (no model in the loop) ==")
	fmt.Fprintf(w, "corpus: %s    frontier baseline: %s\n\n", corpusSrc, cmp.Frontier)
	r, s := cmp.Routed, cmp.Single
	col := func(label, rv, sv, delta string) {
		fmt.Fprintf(w, "  %-32s %-34s %-34s %s\n", label, rv, sv, delta)
	}
	col("", "routed (per-aspect + ensemble)", "single (one model)", "delta")
	fmt.Fprintln(w, strings.Repeat("-", 118))
	col("cases", fmt.Sprintf("%d", r.Cases), fmt.Sprintf("%d", s.Cases), "")
	col("ensembles run", fmt.Sprintf("%d", r.Ensembles), fmt.Sprintf("%d", s.Ensembles), "")
	col("cost (rough $/Mtok-out, sum)", fmt.Sprintf("%.2f", r.Cost), fmt.Sprintf("%.2f", s.Cost), deltaFrac(cmp.CostSavingFrac(), true))
	col("latency (rough ms, sum)", fmt.Sprintf("%.0f", r.Latency), fmt.Sprintf("%.0f", s.Latency), deltaFrac(cmp.LatencySavingFrac(), true))
	col("quality (fraction == expected)", fmt.Sprintf("%.3f (%d/%d)", r.Quality, r.Hits, r.Cases), fmt.Sprintf("%.3f (%d/%d)", s.Quality, s.Hits, s.Cases), deltaQuality(cmp.QualityDelta()))
	fmt.Fprintln(w)
	if a := r.Assumed; len(a) > 0 {
		fmt.Fprintf(w, "  [routed] unpriced models charged at the %s rate: %s (pass --prices)\n", cmp.Frontier, strings.Join(a, ", "))
	}
	fmt.Fprintf(w, "  %s\n", routebenchVerdict(cmp))
	fmt.Fprintln(w, "  (rough offline lenses over a recorded corpus; not a bill, not a measured SLA)")
}

// deltaFrac renders a saving/premium fraction: + means cheaper/faster for routed.
func deltaFrac(frac float64, positiveIsWin bool) string {
	switch {
	case frac > 0.005:
		return fmt.Sprintf("routed %.0f%% %s", frac*100, winWord(positiveIsWin))
	case frac < -0.005:
		return fmt.Sprintf("routed %.0f%% %s", -frac*100, loseWord(positiveIsWin))
	default:
		return "~ tied"
	}
}

func winWord(positiveIsWin bool) string {
	if positiveIsWin {
		return "better"
	}
	return "worse"
}
func loseWord(positiveIsWin bool) string {
	if positiveIsWin {
		return "worse"
	}
	return "better"
}

// deltaQuality renders the accuracy delta: + means routed answered more correctly.
func deltaQuality(d float64) string {
	switch {
	case d > 0.005:
		return fmt.Sprintf("routed +%.1f%% accuracy", d*100)
	case d < -0.005:
		return fmt.Sprintf("routed %.1f%% accuracy", d*100)
	default:
		return "~ tied"
	}
}

// routebenchVerdict is the one-line honest summary of the three-axis trade.
func routebenchVerdict(cmp modelroute.Comparison) string {
	c, l, q := cmp.CostSavingFrac(), cmp.LatencySavingFrac(), cmp.QualityDelta()
	var parts []string
	switch {
	case c > 0.005:
		parts = append(parts, fmt.Sprintf("~%.0f%% cheaper", c*100))
	case c < -0.005:
		parts = append(parts, fmt.Sprintf("%.0f%% costlier (ensemble premium)", -c*100))
	}
	switch {
	case l > 0.005:
		parts = append(parts, fmt.Sprintf("~%.0f%% less total compute", l*100))
	case l < -0.005:
		parts = append(parts, fmt.Sprintf("%.0f%% more total compute", -l*100))
	}
	switch {
	case q > 0.005:
		parts = append(parts, fmt.Sprintf("+%.0f%% accuracy (ensemble rescues)", q*100))
	case q < -0.005:
		parts = append(parts, fmt.Sprintf("%.0f%% accuracy (downgrade losses)", -q*100))
	default:
		parts = append(parts, "quality tied")
	}
	if len(parts) == 0 {
		return "no measurable difference on any axis"
	}
	return "verdict: routed is " + strings.Join(parts, ", ") + "."
}

// routebenchJSON renders the comparison as a stable JSON object (with the raw
// Metrics + the computed deltas a consumer reads directly).
func routebenchJSON(cmp modelroute.Comparison, corpusSrc string) string {
	obj := map[string]any{
		"corpus":              corpusSrc,
		"frontier":            cmp.Frontier,
		"cases":               cmp.Cases,
		"routed":              cmp.Routed,
		"single":              cmp.Single,
		"cost_saving_frac":    cmp.CostSavingFrac(),
		"latency_saving_frac": cmp.LatencySavingFrac(),
		"quality_delta":       cmp.QualityDelta(),
		"verdict":             routebenchVerdict(cmp),
		"note":                "rough offline lenses over a recorded corpus; not a bill, not a measured SLA",
	}
	b, _ := json.MarshalIndent(obj, "", "  ")
	return string(b)
}
