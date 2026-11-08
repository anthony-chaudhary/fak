package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
	"github.com/anthony-chaudhary/fak/internal/optimize"
)

// fak optimize — the dedicated inference-optimization driver. It projects the
// committed native-performance hill-climb graph into a bounded OptimizePlan -> measure
// -> verify loop (internal/optimize): OptimizePlan selects the next dependency-ready
// unwitnessed lever and emits the exact follow-on commands; measure replays one
// frozen trace through the ablate sweep under the lever's mapped runtime arm;
// verify folds two receipts into an accept/reject OptimizeVerdict under either the
// preregistered nativeperf comparison criterion or the OptimizePlan's expected band;
// loop drives the three, stopping after the first reject so gains are only ever
// stacked when witnessed. Unrunnable paths refuse with a closed-vocabulary
// class instead of fabricating numbers.
//
//	fak optimize plan [--envelope ID] [--profile FILE] [--json]
//	fak optimize measure --lever ID (--trace FILE | --receipt FILE) [--envelope ID] [--out FILE] [--json]
//	fak optimize verify --baseline FILE --candidate FILE [--quality FILE] [--envelope ID] [--json]
//	fak optimize loop --max-iters N [--envelope ID] [--dry-run] [--json]
//
// Exit codes: 0 success (verify: accept), 1 failure (plan: no ready lever;
// verify: reject; any internal error), 2 typed OptimizeRefusal / usage error.
func cmdOptimize(args []string) { os.Exit(runOptimize(os.Stdout, os.Stderr, args)) }

// runOptimize is the testable core: it returns the process exit code instead of
// calling os.Exit and takes its streams explicitly (mirrors runAblate).
func runOptimize(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 {
		printOptimizeUsage(stderr)
		return 2
	}
	switch args[0] {
	case "plan":
		return runOptimizePlan(stdout, stderr, args[1:])
	case "measure":
		return runOptimizeMeasure(stdout, stderr, args[1:])
	case "verify":
		return runOptimizeVerify(stdout, stderr, args[1:])
	case "loop":
		return runOptimizeLoop(stdout, stderr, args[1:])
	default:
		fmt.Fprintf(stderr, "fak optimize: unknown subcommand %q\n", args[0])
		printOptimizeUsage(stderr)
		return 2
	}
}

func printOptimizeUsage(w io.Writer) {
	fmt.Fprintln(w, `usage: fak optimize <sub> [flags]

  plan      [--envelope ID] [--profile FILE] [--json]
          select the next dependency-ready unwitnessed lever (optionally
          restricted to pinned envelope ID; or the profile-classified one)
          and emit the OptimizePlan JSON
  measure --lever ID (--trace FILE | --receipt FILE) [--envelope ID] [--out FILE] [--json]
          replay the frozen trace A/B under the lever's mapped ablate arm, or
          fold a nativeperf baseline/candidate receipt pair into the receipt
  verify  --baseline FILE --candidate FILE [--quality FILE] [--envelope ID] [--json]
          fold two receipts into an accept/reject OptimizeVerdict
  loop    --max-iters N [--envelope ID] [--dry-run] [--json]
          bounded OptimizePlan->measure->verify hill-climb over the default suite trace`)
}

// emitOptimizeRefusal renders a typed OptimizeRefusal as the structured {"error": ...}
// JSON on stdout and yields the sub-verb's OptimizeRefusal exit code.
func emitOptimizeRefusal(stdout io.Writer, r *optimize.OptimizeRefusal, code int) int {
	body := struct {
		Error  string `json:"error"`
		Detail string `json:"detail,omitempty"`
	}{Error: r.Class, Detail: r.Detail}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(body); err != nil {
		fmt.Fprintf(stdout, "{\"error\":%q}\n", r.Class)
	}
	return code
}

func encodeOptimizeJSON(stdout, stderr io.Writer, value any) int {
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(value); err != nil {
		fmt.Fprintf(stderr, "fak optimize: encode JSON: %v\n", err)
		return 1
	}
	return 0
}

func runOptimizePlan(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("optimize OptimizePlan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	envelope := fs.String("envelope", "", "restrict selection to the pinned envelope ID")
	profilePath := fs.String("profile", "", "select the next lever from native profile FILE (the --profile-next classifier path)")
	_ = fs.Bool("json", false, "emit the OptimizePlan JSON (the default rendering)")
	if rc, ok := parseFlagsOrHelp(fs, args); !ok {
		return rc
	}
	cfg := optimize.Config{Graph: nativeperf.ActiveGraph(), EnvelopeID: *envelope}
	var OptimizePlan *optimize.OptimizePlan
	var err error
	if *profilePath != "" {
		data, readErr := os.ReadFile(*profilePath)
		if readErr != nil {
			return emitOptimizeRefusal(stdout, optimize.NewOptimizeRefusal(optimize.ClassInvalidInput, fmt.Sprintf("read profile: %v", readErr)), 2)
		}
		profile, decodeErr := nativeperf.DecodeProfile(data)
		if decodeErr != nil {
			return emitOptimizeRefusal(stdout, optimize.NewOptimizeRefusal(optimize.ClassInvalidInput, fmt.Sprintf("decode profile: %v", decodeErr)), 2)
		}
		OptimizePlan, err = optimize.BuildPlanFromProfile(cfg, profile)
	} else {
		OptimizePlan, err = optimize.BuildPlan(cfg)
	}
	if err != nil {
		if refuse := optimizeRefusalOf(err); refuse != nil {
			return emitOptimizeRefusal(stdout, refuse, 1)
		}
		fmt.Fprintf(stderr, "fak optimize plan: %v\n", err)
		return 1
	}
	return encodeOptimizeJSON(stdout, stderr, OptimizePlan)
}

func runOptimizeMeasure(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("optimize measure", flag.ContinueOnError)
	fs.SetOutput(stderr)
	lever := fs.String("lever", "", "lever id to measure (required)")
	tracePath := fs.String("trace", "", "frozen trace FILE to replay (exactly one of --trace/--receipt)")
	receiptPath := fs.String("receipt", "", "nativeperf baseline/candidate receipt-pair FILE to fold (exactly one of --trace/--receipt)")
	envelope := fs.String("envelope", "", "pinned envelope ID the lever must belong to")
	out := fs.String("out", "", "write the OptimizeReceipt JSON to this path")
	asJSON := fs.Bool("json", false, "emit the OptimizeReceipt JSON to stdout")
	if rc, ok := parseFlagsOrHelp(fs, args); !ok {
		return rc
	}
	if *lever == "" {
		fmt.Fprintln(stderr, "fak optimize measure: --lever is required")
		return 2
	}
	if *tracePath != "" && *receiptPath != "" {
		return emitOptimizeRefusal(stdout, optimize.NewOptimizeRefusal(optimize.ClassInvalidInput, "--trace and --receipt are mutually exclusive; supply exactly one workload source"), 2)
	}
	if *tracePath == "" && *receiptPath == "" {
		fmt.Fprintln(stderr, "fak optimize measure: --trace or --receipt is required")
		return 2
	}
	rec, err := optimize.Measure(ctx(), optimize.Config{Graph: nativeperf.ActiveGraph(), EnvelopeID: *envelope}, *lever, *tracePath, *receiptPath)
	if err != nil {
		if refuse := optimizeRefusalOf(err); refuse != nil {
			return emitOptimizeRefusal(stdout, refuse, 2)
		}
		fmt.Fprintf(stderr, "fak optimize measure: %v\n", err)
		return 1
	}
	raw, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "fak optimize measure: encode receipt: %v\n", err)
		return 1
	}
	raw = append(raw, '\n')
	if *out != "" {
		if err := os.WriteFile(*out, raw, 0o644); err != nil {
			fmt.Fprintf(stderr, "fak optimize measure: write %s: %v\n", *out, err)
			return 1
		}
		fmt.Fprintf(stdout, "OptimizeReceipt written : %s\n", *out)
		return 0
	}
	if *asJSON {
		_, _ = stdout.Write(raw)
		return 0
	}
	fmt.Fprintf(stdout, "== fak optimize measure: %s ==\n", rec.LeverID)
	fmt.Fprintf(stdout, "trace            : %s (%s...)\n", rec.TracePath, rec.TraceSHA256[:12])
	fmt.Fprintf(stdout, "engine           : %s\n", rec.EngineID)
	fmt.Fprintf(stdout, "baseline (off)   : %.4f tok/s\n", rec.BaselineTokS)
	fmt.Fprintf(stdout, "candidate (on)   : %.4f tok/s\n", rec.CandidateTokS)
	fmt.Fprintf(stdout, "delta            : %+.2f%%\n", rec.DeltaPct)
	fmt.Fprintf(stdout, "reps             : %d\n", rec.Reps)
	fmt.Fprintf(stdout, "measured_at      : %s\n", rec.MeasuredAt)
	return 0
}

func runOptimizeVerify(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("optimize verify", flag.ContinueOnError)
	fs.SetOutput(stderr)
	baseline := fs.String("baseline", "", "baseline OptimizeReceipt FILE (required)")
	candidate := fs.String("candidate", "", "candidate OptimizeReceipt FILE (required)")
	quality := fs.String("quality", "", "fak quality run --json result FILE; a failing gate forces reject")
	envelope := fs.String("envelope", "", "pinned envelope ID both receipts must target")
	_ = fs.Bool("json", false, "emit the OptimizeVerdict JSON (the default rendering)")
	if rc, ok := parseFlagsOrHelp(fs, args); !ok {
		return rc
	}
	if *baseline == "" || *candidate == "" {
		fmt.Fprintln(stderr, "fak optimize verify: --baseline and --candidate are required")
		return 2
	}
	verdict, err := optimize.Verify(optimize.Config{Graph: nativeperf.ActiveGraph(), EnvelopeID: *envelope}, *baseline, *candidate, *quality)
	if err != nil {
		if refuse := optimizeRefusalOf(err); refuse != nil {
			return emitOptimizeRefusal(stdout, refuse, 2)
		}
		fmt.Fprintf(stderr, "fak optimize verify: %v\n", err)
		return 1
	}
	if code := encodeOptimizeJSON(stdout, stderr, verdict); code != 0 {
		return code
	}
	if verdict.Verdict == optimize.VerdictAccept {
		return 0
	}
	return 1
}

func runOptimizeLoop(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("optimize loop", flag.ContinueOnError)
	fs.SetOutput(stderr)
	maxIters := fs.Int("max-iters", 0, "bounded iteration count (required, positive)")
	dryRun := fs.Bool("dry-run", false, "stop after the first OptimizePlan; never measure")
	envelope := fs.String("envelope", "", "restrict every iteration to the pinned envelope ID")
	_ = fs.Bool("json", false, "ledger lines are JSONL regardless; accepted for surface parity")
	if rc, ok := parseFlagsOrHelp(fs, args); !ok {
		return rc
	}
	lines, err := optimize.RunLoop(ctx(), optimize.Config{Graph: nativeperf.ActiveGraph(), EnvelopeID: *envelope}, *maxIters, *dryRun, optimize.DefaultTracePath)
	if err != nil {
		if refuse := optimizeRefusalOf(err); refuse != nil {
			return emitOptimizeRefusal(stdout, refuse, 2)
		}
		fmt.Fprintf(stderr, "fak optimize loop: %v\n", err)
		return 1
	}
	for _, line := range lines {
		raw, err := json.Marshal(line)
		if err != nil {
			fmt.Fprintf(stderr, "fak optimize loop: encode ledger line: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(raw))
	}
	return 0
}

func optimizeRefusalOf(err error) *optimize.OptimizeRefusal {
	var r *optimize.OptimizeRefusal
	if errors.As(err, &r) {
		return r
	}
	return nil
}
