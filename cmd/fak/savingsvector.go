package main

// fak savings-vector -- decompose a turn-tax saving into the FOUR orthogonal
// accounts (local_cpu MEASURED, gpu_prefill MODELED, context_window MEASURED-as-rate
// or MODELED, wall_clock MODELED) named by
// docs/explainers/compounding-benefits-of-a-saved-call.md. It reads a turnbench
// Report JSON (the artifact `fak turntax --out report.json` writes) and re-projects
// its already-measured/already-computed fields, labeled per axis. The Go port of
// tools/savings_vector.py; its selfcheck invariants are the anti-overclaim contract.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/savingsvector"
)

func cmdSavingsVector(argv []string) { os.Exit(runSavingsVector(os.Stdout, os.Stderr, argv)) }

func runSavingsVector(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "selfcheck" {
		return runReportSelfcheck(stdout, stderr, argv[1:], "savings-vector", savingsvector.Selfcheck,
			"SELFCHECK OK -- vector decomposes Net (does not inflate); local_cpu "+
				"surfaced and measured; gpu_prefill/wall_clock honestly modeled; happy control saves 0; "+
				"binding account follows the profile.")
	}

	fs := flag.NewFlagSet("fak savings-vector", flag.ContinueOnError)
	fs.SetOutput(stderr)
	reportPath := fs.String("report", "", "path to a turnbench Report JSON (fak turntax --out); or pass it positionally")
	profile := fs.String("profile", "laptop", "host shape that sets the binding (scarce) account: "+
		"laptop|gpu-fleet|ci-hook")
	pollutionRate := fs.Float64("pollution-rate", math.NaN(),
		"ctxmmu paged-out fraction; promotes the context axis to measured-rate")
	asJSON := fs.Bool("json", false, "emit the vector as JSON")
	// Go's flag package stops at the first non-flag token, so a positional path
	// before the flags (savings-vector PATH --profile X) would strand the flags as
	// stray args. Reorder so a single leading positional moves after the flags,
	// matching the original tool's "report <path> [--profile ...]" surface.
	if err := fs.Parse(reorderLeadingPositional(argv)); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}

	path, ok := resolveReportPath(fs, stderr, "savings-vector",
		"fak savings-vector: give a turnbench Report JSON path "+
			"(positional or --report), or run 'fak savings-vector selfcheck'",
		*reportPath)
	if !ok {
		return 2
	}

	if !savingsvector.KnownProfile(*profile) {
		fmt.Fprintf(stderr, "fak savings-vector: unknown profile %q; pick one of %v\n",
			*profile, savingsvector.Profiles())
		return 2
	}

	report, ok := readReportJSON[savingsvector.Report](path, "savings-vector", stderr)
	if !ok {
		return 2
	}

	var ratePtr *float64
	if !math.IsNaN(*pollutionRate) {
		ratePtr = pollutionRate
	}

	v, err := savingsvector.BuildVector(report, *profile, ratePtr)
	if err != nil {
		fmt.Fprintf(stderr, "fak savings-vector: %v\n", err)
		return 2
	}

	if *asJSON {
		if err := writeIndentedJSON(stdout, v); err != nil {
			fmt.Fprintf(stderr, "fak savings-vector: encode json: %v\n", err)
			return 1
		}
	} else {
		writeSavingsVectorHuman(stdout, v)
	}

	// The invariant the doc rests on: the vector re-projects Net, never exceeds it.
	if math.Abs(v.NetDollarsSaved-v.VectorDollarsSaved) >= 1e-9 {
		fmt.Fprintln(stderr, "FAIL: vector dollar saving != Net dollar saving (inflation bug)")
		return 1
	}
	return 0
}

func writeSavingsVectorHuman(w io.Writer, v savingsvector.Vector) {
	fmt.Fprintf(w, "savings vector (profile=%s) -- %d turns saved (forced=%d, elision=%d)\n",
		v.Profile, v.TurnsSaved, v.Forced, v.Elision)
	fmt.Fprintf(w, "  binding account: %s\n\n", v.Binding)
	fmt.Fprintf(w, "  %-16s%16s  %-14s%-14s\n", "account", "saving", "unit", "provenance")
	fmt.Fprintf(w, "  %-16s%16s  %-14s%-14s\n", dashes(16), dashes(16), dashes(14), dashes(14))
	for _, a := range v.Axes {
		fmt.Fprintf(w, "  %-16s%16.4g  %-14s%-14s\n", a.Account, a.Amount, a.Unit, a.Provenance)
	}
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "  cross-check (re-projection must equal the flat Net, not exceed it):")
	fmt.Fprintf(w, "    Net.dollars_saved    = %.6f\n", v.NetDollarsSaved)
	fmt.Fprintf(w, "    vector.dollars_saved = %.6f\n", v.VectorDollarsSaved)
	ok := math.Abs(v.NetDollarsSaved-v.VectorDollarsSaved) < 1e-9
	verdict := "NO -- BUG"
	if ok {
		verdict = "YES (decomposes, does not inflate)"
	}
	fmt.Fprintf(w, "    equal: %s\n", verdict)
}

// resolveReportPath resolves the report-JSON path shared by the report-replay
// commands (horizon-recovery, savings-vector): prefer an explicit --report
// (reportFlag), else a single positional argument. cmdName labels the
// command-specific error messages and missingHint is the command-specific
// "give a ... path" sentence shown when no path is supplied. ok is false when
// the path is missing or an extra argument was given (caller should return 2).
func resolveReportPath(fs *flag.FlagSet, stderr io.Writer, cmdName, missingHint, reportFlag string) (path string, ok bool) {
	path = reportFlag
	if path == "" {
		if fs.NArg() == 0 {
			fmt.Fprintln(stderr, missingHint)
			return "", false
		}
		path = fs.Arg(0)
		if fs.NArg() > 1 {
			fmt.Fprintf(stderr, "fak %s: unexpected argument %q\n", cmdName, fs.Arg(1))
			return "", false
		}
	} else if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak %s: unexpected argument %q\n", cmdName, fs.Arg(0))
		return "", false
	}
	return path, true
}

// readReportJSON reads path and unmarshals it into a fresh T, emitting the shared
// read/parse error messages (labeled "fak <cmdName>: ...") on failure and returning
// ok=false (the report-replay callers return 2 on a false). It is the shared
// report-load step behind horizon-recovery and savings-vector.
func readReportJSON[T any](path, cmdName string, stderr io.Writer) (report T, ok bool) {
	raw, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak %s: read report: %v\n", cmdName, err)
		return report, false
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		fmt.Fprintf(stderr, "fak %s: parse report JSON: %v\n", cmdName, err)
		return report, false
	}
	return report, true
}

// runReportSelfcheck implements the shared `selfcheck` subcommand of the
// report-replay commands (horizon-recovery, savings-vector): it accepts no flags,
// rejects any extra argument, runs the package selfcheck, and prints okMsg on
// success. cmdName labels the flagset and the error ("fak <cmdName> selfcheck ...").
func runReportSelfcheck(stdout, stderr io.Writer, argv []string, cmdName string, selfcheck func() error, okMsg string) int {
	fs := flag.NewFlagSet("fak "+cmdName+" selfcheck", flag.ContinueOnError)
	fs.SetOutput(stderr)
	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak %s selfcheck: unexpected argument %q\n", cmdName, fs.Arg(0))
		return 2
	}
	if err := selfcheck(); err != nil {
		fmt.Fprintf(stderr, "SELFCHECK FAIL:\n  - %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, okMsg)
	return 0
}

// reorderLeadingPositional moves a single leading non-flag token (the Report path)
// to the end so flags that follow it are still parsed. Only the FIRST token is
// considered positional; once a flag is seen the order is left untouched (so a value
// like --report=x or --profile laptop is never mistaken for the positional). If the
// first token is already a flag, argv is returned unchanged.
func reorderLeadingPositional(argv []string) []string {
	if len(argv) == 0 || strings.HasPrefix(argv[0], "-") {
		return argv
	}
	out := make([]string, 0, len(argv))
	out = append(out, argv[1:]...)
	out = append(out, argv[0])
	return out
}

func dashes(n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = '-'
	}
	return string(b)
}
