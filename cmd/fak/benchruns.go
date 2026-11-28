package main

import (
	"flag"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/anthony-chaudhary/fak/internal/benchruns"
)

func cmdBenchRuns(argv []string) { os.Exit(runBenchRuns(os.Stdout, os.Stderr, argv)) }

func runBenchRuns(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		benchRunsUsage(stderr)
		return 2
	}
	sub, rest := argv[0], argv[1:]
	switch sub {
	case "list":
		return benchRunsList(stdout, stderr, rest)
	case "show":
		return benchRunsShow(stdout, stderr, rest)
	case "compare":
		return benchRunsCompare(stdout, stderr, rest)
	case "best":
		return benchRunsBest(stdout, stderr, rest)
	case "table":
		return benchRunsTable(stdout, stderr, rest)
	case "summary":
		return benchRunsSummary(stdout, stderr, rest)
	case "pareto":
		return benchRunsParetoCmd(stdout, stderr, rest)
	case "-h", "--help", "help":
		benchRunsUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak bench-runs: unknown subcommand %q\n", sub)
		benchRunsUsage(stderr)
		return 2
	}
}

func benchRunsList(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak bench-runs list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	machine := fs.String("machine", "", "filter by machine ID")
	model := fs.String("model", "", "filter by model substring")
	precision := fs.String("precision", "", "filter by precision")
	since := fs.String("since", "", "filter by start date/string")
	until := fs.String("until", "", "filter by end date/string")
	format := fs.String("format", "table", "output format: table|json")
	if !parseFlags(fs, argv) {
		return 2
	}
	cat, code := benchRunsCatalog(stderr, *workspace)
	if code != 0 {
		return code
	}
	runs := benchruns.FilterRuns(cat.Runs, benchruns.Filter{Machine: *machine, Model: *model, Precision: *precision, Since: *since, Until: *until})
	if *format == "json" {
		_ = writeIndentedJSON(stdout, runs)
		return 0
	}
	if *format != "table" {
		fmt.Fprintf(stderr, "fak bench-runs list: unknown --format %q\n", *format)
		return 2
	}
	fmt.Fprint(stdout, benchruns.RenderList(runs))
	return 0
}

func benchRunsShow(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak bench-runs show", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "usage: fak bench-runs show <run-id>")
		return 2
	}
	root := benchRunsRoot(*workspace)
	cat, code := benchRunsCatalog(stderr, root)
	if code != 0 {
		return code
	}
	d, err := benchruns.LoadRun(root, cat, fs.Arg(0))
	if err != nil {
		fmt.Fprintf(stderr, "fak bench-runs show: %v\n", err)
		return 1
	}
	fmt.Fprint(stdout, benchruns.RenderShow(d))
	return 0
}

func benchRunsCompare(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak bench-runs compare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 2 {
		fmt.Fprintln(stderr, "usage: fak bench-runs compare <run-id-1> <run-id-2>")
		return 2
	}
	root := benchRunsRoot(*workspace)
	cat, code := benchRunsCatalog(stderr, root)
	if code != 0 {
		return code
	}
	a, err := benchruns.LoadRun(root, cat, fs.Arg(0))
	if err != nil {
		fmt.Fprintf(stderr, "fak bench-runs compare: %v\n", err)
		return 1
	}
	b, err := benchruns.LoadRun(root, cat, fs.Arg(1))
	if err != nil {
		fmt.Fprintf(stderr, "fak bench-runs compare: %v\n", err)
		return 1
	}
	fmt.Fprint(stdout, benchruns.RenderCompare(a, b))
	return 0
}

func benchRunsBest(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak bench-runs best", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	model := fs.String("model", "", "filter by model")
	metric := fs.String("metric", "peak_tok_per_sec", "metric: peak_tok_per_sec|speedup")
	if !parseFlags(fs, argv) {
		return 2
	}
	cat, code := benchRunsCatalog(stderr, *workspace)
	if code != 0 {
		return code
	}
	best, err := benchruns.Best(cat.Runs, *model, *metric)
	if err != nil {
		fmt.Fprintf(stderr, "fak bench-runs best: %v\n", err)
		return 1
	}
	fmt.Fprint(stdout, benchruns.RenderBest(best, *metric))
	return 0
}

func benchRunsTable(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak bench-runs table", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	model := fs.String("model", "", "filter by model substring")
	precision := fs.String("precision", "", "filter by precision")
	format := fs.String("format", "markdown", "output format: markdown|json")
	if !parseFlags(fs, argv) {
		return 2
	}
	cat, code := benchRunsCatalog(stderr, *workspace)
	if code != 0 {
		return code
	}
	runs := benchruns.FilterRuns(cat.Runs, benchruns.Filter{Model: *model, Precision: *precision})
	if *format == "json" {
		_ = writeIndentedJSON(stdout, runs)
		return 0
	}
	if *format != "markdown" {
		fmt.Fprintf(stderr, "fak bench-runs table: unknown --format %q\n", *format)
		return 2
	}
	if len(runs) == 0 {
		fmt.Fprintln(stderr, "No runs found")
		return 0
	}
	fmt.Fprint(stdout, benchruns.RenderMarkdownTable(runs))
	return 0
}

func benchRunsSummary(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak bench-runs summary", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	groupBy := fs.String("group-by", "machine", "group by: machine|model")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *groupBy != "machine" && *groupBy != "model" {
		fmt.Fprintf(stderr, "fak bench-runs summary: unknown --group-by %q\n", *groupBy)
		return 2
	}
	cat, code := benchRunsCatalog(stderr, *workspace)
	if code != 0 {
		return code
	}
	fmt.Fprint(stdout, benchruns.RenderSummary(cat.Runs, *groupBy))
	return 0
}

func benchRunsCatalog(stderr io.Writer, workspace string) (benchruns.Catalog, int) {
	root := benchRunsRoot(workspace)
	cat, err := benchruns.LoadCatalog(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak bench-runs: %v\n", err)
		fmt.Fprintln(stderr, "fak bench-runs: run `fak bench-catalog build` or refresh experiments/benchmark/catalog.json first")
		return benchruns.Catalog{}, 1
	}
	return cat, 0
}

func benchRunsRoot(workspace string) string {
	if workspace != "" {
		return workspace
	}
	return repoRoot()
}

func benchRunsUsage(w io.Writer) {
	fmt.Fprint(w, `fak bench-runs - query and compare recorded benchmark runs

usage:
  fak bench-runs list [--machine M] [--model M] [--precision P] [--format table|json]
  fak bench-runs show <run-id>
  fak bench-runs compare <run-id-1> <run-id-2>
  fak bench-runs best [--model M] [--metric peak_tok_per_sec|speedup]
  fak bench-runs table [--model M] [--precision P] [--format markdown|json]
  fak bench-runs summary [--group-by machine|model]
  fak bench-runs pareto [--model M] [--throughput F] [--latency F] [--format table|json]
`)
}

// ---------------------------------------------------------------------------
// CI-width convergence (ported from ai-dynamo/aiperf)
//
// NOTICE: Ported from ai-dynamo/aiperf @ ab610ad
// (src/aiperf/orchestrator/convergence/ci_width.py), Apache-2.0.
// ---------------------------------------------------------------------------

// CIWidthConvergence converges when the Student-t confidence-interval width
// relative to the mean of the measured values falls below Threshold.
type CIWidthConvergence struct {
	Threshold       float64
	ConfidenceLevel float64
	MinRuns         int
}

// NewCIWidthConvergence builds a CIWidthConvergence, applying the source
// defaults when the caller leaves a parameter unset: threshold 0 -> 0.10,
// confidence level 0 -> 0.95, min runs <= 0 -> 3.
func NewCIWidthConvergence(threshold, confidenceLevel float64, minRuns int) *CIWidthConvergence {
	c := &CIWidthConvergence{Threshold: threshold, ConfidenceLevel: confidenceLevel, MinRuns: minRuns}
	if c.Threshold == 0 {
		c.Threshold = 0.10
	}
	if c.ConfidenceLevel == 0 {
		c.ConfidenceLevel = 0.95
	}
	if c.MinRuns <= 0 {
		c.MinRuns = 3
	}
	return c
}

// IsConverged reports whether the Student-t CI width relative to the mean of
// values falls below Threshold, mirroring the source is_converged exactly.
// It returns false when fewer than max(MinRuns, 2) values are present or when
// the mean is zero.
func (c *CIWidthConvergence) IsConverged(values []float64) bool {
	if len(values) < max(c.MinRuns, 2) {
		return false
	}
	mean := 0.0
	for _, v := range values {
		mean += v
	}
	mean /= float64(len(values))
	if mean == 0 {
		return false
	}
	n := len(values)
	sumSquares := 0.0
	for _, v := range values {
		delta := v - mean
		sumSquares += delta * delta
	}
	std := math.Sqrt(sumSquares / float64(n-1)) // sample standard deviation, ddof=1
	se := std / math.Sqrt(float64(n))
	tCrit := studentTQuantile((1+c.ConfidenceLevel)/2, float64(n-1))
	ciHalf := tCrit * se
	ciWidthRatio := (2 * ciHalf) / math.Abs(mean)
	return ciWidthRatio < c.Threshold
}

// studentTQuantile returns the p-quantile of the Student-t distribution with
// df degrees of freedom by bisecting the monotone Student-t CDF on [0, 100].
// It returns 0 for non-finite or out-of-domain inputs.
func studentTQuantile(p, df float64) float64 {
	if math.IsNaN(p) || math.IsInf(p, 0) || math.IsNaN(df) || math.IsInf(df, 0) || df <= 0 || p <= 0 || p >= 1 {
		return 0
	}
	lo, hi := 0.0, 100.0
	for i := 0; i < 200; i++ {
		mid := (lo + hi) / 2
		cdf, ok := studentTCDF(mid, df)
		if !ok {
			return 0
		}
		if cdf < p {
			lo = mid
		} else {
			hi = mid
		}
		if hi-lo < 1e-12 {
			break
		}
	}
	return (lo + hi) / 2
}

// studentTCDF returns the Student-t CDF at t >= 0 with df degrees of freedom:
// 1 - 0.5*I_x(df/2, 0.5), where x = df/(df + t*t) and I_x is the regularized
// incomplete beta function. ok is false when the inputs are non-finite or the
// bounded continued fraction cannot produce finite evidence.
func studentTCDF(t, df float64) (float64, bool) {
	if t < 0 || df <= 0 || math.IsNaN(t) || math.IsInf(t, 0) || math.IsNaN(df) || math.IsInf(df, 0) {
		return 0, false
	}
	x := df / (df + t*t)
	ibeta, ok := regularizedIncompleteBeta(x, df/2, 0.5)
	if !ok {
		return 0, false
	}
	return 1 - 0.5*ibeta, true
}

// regularizedIncompleteBeta evaluates the regularized incomplete beta function
// I_x(a, b) via the standard continued fraction (Lentz). Ported from
// cmd/modelbench/profile_compare.go.
func regularizedIncompleteBeta(x, a, b float64) (float64, bool) {
	if x < 0 || x > 1 || !positiveFinite(a) || !positiveFinite(b) {
		return 0, false
	}
	if x == 0 {
		return 0, true
	}
	if x == 1 {
		return 1, true
	}
	logGammaAB, _ := math.Lgamma(a + b)
	logGammaA, _ := math.Lgamma(a)
	logGammaB, _ := math.Lgamma(b)
	factor := math.Exp(logGammaAB - logGammaA - logGammaB + a*math.Log(x) + b*math.Log1p(-x))
	if math.IsNaN(factor) || math.IsInf(factor, 0) {
		return 0, false
	}
	if x < (a+1)/(a+b+2) {
		fraction, ok := incompleteBetaContinuedFraction(a, b, x)
		return factor * fraction / a, ok
	}
	fraction, ok := incompleteBetaContinuedFraction(b, a, 1-x)
	return 1 - factor*fraction/b, ok
}

// incompleteBetaContinuedFraction evaluates the continued fraction for the
// incomplete beta function using the Lentz algorithm. Ported from
// cmd/modelbench/profile_compare.go.
func incompleteBetaContinuedFraction(a, b, x float64) (float64, bool) {
	const (
		maxIterations = 200
		epsilon       = 3e-14
		minimum       = 1e-300
	)
	qab := a + b
	qap := a + 1
	qam := a - 1
	c := 1.0
	d := 1 - qab*x/qap
	if math.Abs(d) < minimum {
		d = minimum
	}
	d = 1 / d
	h := d
	for iteration := 1; iteration <= maxIterations; iteration++ {
		m := float64(iteration)
		m2 := 2 * m
		aa := m * (b - m) * x / ((qam + m2) * (a + m2))
		d = 1 + aa*d
		if math.Abs(d) < minimum {
			d = minimum
		}
		c = 1 + aa/c
		if math.Abs(c) < minimum {
			c = minimum
		}
		d = 1 / d
		h *= d * c

		aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
		d = 1 + aa*d
		if math.Abs(d) < minimum {
			d = minimum
		}
		c = 1 + aa/c
		if math.Abs(c) < minimum {
			c = minimum
		}
		d = 1 / d
		delta := d * c
		h *= delta
		if math.Abs(delta-1) < epsilon {
			return h, !math.IsNaN(h) && !math.IsInf(h, 0)
		}
	}
	return 0, false
}

// positiveFinite reports whether value is a finite positive number.
func positiveFinite(value float64) bool {
	return value > 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}

// NOTICE: The monotonic search planner below is an Apache-2.0 licensed reuse
// ported from NVIDIA's aiperf project.
//
//	File:      src/aiperf/orchestrator/search_planner/monotonic.py
//	License:   Apache-2.0 (SPDX-License-Identifier: Apache-2.0)
//	Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// The source implements a 1D "max-concurrency-under-SLA" search planner: an
// exponential probe from the low end of a monotonically degrading sweep axis,
// followed by bisection between the last passing point (lo') and the first
// failing point (hi'). This port preserves the source's boundary semantics and
// edge cases (all-true, all-false, single element) over a sorted candidate
// list, and keeps the search monotonic: it never revisits a probed value.

// MonotonicSearchReason mirrors the convergence reasons of the source planner.
type MonotonicSearchReason string

const (
	// MonotonicPrecisionReached: bisection narrowed the bracket to two adjacent
	// candidates (the discrete analogue of the source's 5% relative precision).
	MonotonicPrecisionReached MonotonicSearchReason = "monotonic_precision_reached"
	// MonotonicNoFailureInRange: every probed candidate passed; the boundary
	// lies above the last candidate (the source's "SLA saturates above hi").
	MonotonicNoFailureInRange MonotonicSearchReason = "monotonic_no_failure_in_range"
	// MonotonicNoPassInRange: every probed candidate failed; the boundary lies
	// below the first candidate (the source's "SLA saturates below lo").
	MonotonicNoPassInRange MonotonicSearchReason = "monotonic_no_pass_in_range"
)

// MonotonicSearchResult is the boundary found by MonotonicSearch.
type MonotonicSearchResult struct {
	// SplitIndex is the index of the first failing candidate: every candidate
	// before it passes the predicate, every candidate at or after it fails.
	// SplitIndex == len(candidates) when all candidates pass; 0 when all fail.
	SplitIndex int
	// FeasibleMax is the highest passing candidate (the source's lo'), or -1
	// when no candidate passes.
	FeasibleMax int
	// InfeasibleMin is the lowest failing candidate (the source's hi'), or -1
	// when no candidate fails.
	InfeasibleMin int
	// Reason is the convergence reason.
	Reason MonotonicSearchReason
	// Probed is the sequence of candidate values probed, in order. No value is
	// probed more than once: the search never revisits a value.
	Probed []int
}

// MonotonicSearch finds the feasibility boundary over a sorted candidate list
// using the source's monotonic (never-decreasing) parameter search: an
// exponential probe from the low end, then bisection between the last passing
// and first failing candidate.
//
// candidates must be sorted in ascending order. feasible reports whether a
// candidate value satisfies the (monotonically degrading) constraint: it must
// return true for low values and false for high values, and once it returns
// false at some value it must return false at every higher value.
//
// The search never revisits a value: each candidate is probed at most once, and
// the probe sequence never re-enters the already-established passing region
// (at or below the last passing candidate) or failing region (at or above the
// first failing candidate).
func MonotonicSearch(candidates []int, feasible func(int) bool) MonotonicSearchResult {
	n := len(candidates)
	res := MonotonicSearchResult{FeasibleMax: -1, InfeasibleMin: -1}
	if n == 0 {
		res.Reason = MonotonicNoPassInRange
		return res
	}

	probed := make([]int, 0, n)
	probe := func(idx int) bool {
		probed = append(probed, candidates[idx])
		return feasible(candidates[idx])
	}

	// Phase 1: exponential probe from the low end (the source's "probe" phase).
	// Probe index 0 first; on a pass, double the index (1, 2, 4, 8, ...) until
	// a failure or the top of the list.
	if !probe(0) {
		// First candidate fails; monotonicity implies every candidate fails.
		res.SplitIndex = 0
		res.InfeasibleMin = candidates[0]
		res.Reason = MonotonicNoPassInRange
		res.Probed = probed
		return res
	}
	lastPass := 0
	firstFail := -1
	idx := 1
	for idx < n {
		if probe(idx) {
			lastPass = idx
			if idx == n-1 {
				// Reached the top and it passed: every candidate passes.
				res.SplitIndex = n
				res.FeasibleMax = candidates[n-1]
				res.Reason = MonotonicNoFailureInRange
				res.Probed = probed
				return res
			}
			idx *= 2
			if idx > n-1 {
				idx = n - 1
			}
		} else {
			// First failure during probing: bracket found.
			firstFail = idx
			break
		}
	}
	if firstFail == -1 {
		// No failure found (single-element list): every candidate passes.
		res.SplitIndex = n
		res.FeasibleMax = candidates[n-1]
		res.Reason = MonotonicNoFailureInRange
		res.Probed = probed
		return res
	}

	// Phase 2: bisection between the last passing and first failing candidate
	// (the source's "bisect" phase) until the bracket is two adjacent
	// candidates.
	lo, hi := lastPass, firstFail
	for hi-lo > 1 {
		mid := lo + (hi-lo)/2
		if probe(mid) {
			lo = mid
		} else {
			hi = mid
		}
	}
	res.SplitIndex = hi
	res.FeasibleMax = candidates[lo]
	res.InfeasibleMin = candidates[hi]
	res.Reason = MonotonicPrecisionReached
	res.Probed = probed
	return res
}

// credential redaction at the single plaintext-holding point.
func redactCredentialArgs(args []string) []string {
	out := make([]string, len(args))
	copy(out, args)
	secretFlags := map[string]bool{"--api-key": true, "--token": true, "--secret": true}
	for i := 0; i < len(out); i++ {
		if secretFlags[out[i]] && i+1 < len(out) {
			out[i+1] = "[REDACTED]"
			i++
			continue
		}
		for _, prefix := range []string{"--api-key=", "--token=", "--secret=", "OPENAI_API_KEY=", "HF_TOKEN="} {
			if len(out[i]) >= len(prefix) && out[i][:len(prefix)] == prefix {
				out[i] = prefix + "[REDACTED]"
				break
			}
		}
	}
	return out
}
