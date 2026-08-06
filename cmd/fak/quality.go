package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/quality"
)

// cmdQuality is the `fak quality` operator surface for the missing-middle quality
// ladder (epic #4509): `fak quality run` executes one versioned case through a
// reference path and an engine path and emits a machine-readable result with a
// pass/fail verdict and a replayable failure bundle; `fak quality explain` renders
// a result as first-failure localization (#4520) — the failing oracle, the exact
// divergent step, and the stage of the serving path the evidence attributes it to
// (normalization, tokenization, logits, sampling, stops, cache, transport, rubric)
// or an explicit abstention when the evidence names none of them; `fak quality
// replay` is the consuming half of the portable failure bundle (#4515) — it
// reproduces a recorded failure from the stored artifact ALONE, so the ladder's
// replay promise is exercised rather than asserted.
func cmdQuality(argv []string) {
	if len(argv) == 0 {
		fmt.Fprintln(os.Stderr, "usage: fak quality <run|explain|replay> [flags]")
		os.Exit(2)
	}
	switch argv[0] {
	case "run":
		os.Exit(runQualityRun(os.Stdout, os.Stderr, argv[1:]))
	case "explain":
		os.Exit(runQualityExplain(os.Stdout, os.Stderr, argv[1:]))
	case "replay":
		os.Exit(runQualityReplay(os.Stdout, os.Stderr, os.Stdin, argv[1:]))
	default:
		fmt.Fprintf(os.Stderr, "fak quality: unknown subcommand %q (want run|explain|replay)\n", argv[0])
		os.Exit(2)
	}
}

// runQualityRun runs a case and emits its Result. It exits 0 on pass, 1 on a
// quality failure (so a CI gate reads $?), and 2 on a usage/IO error — a quality
// failure and an infrastructure error are never conflated.
func runQualityRun(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak quality run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	casePath := fs.String("case", "", "path to a quality-case JSON (default: built-in demo case)")
	enginePath := fs.String("engine-trace", "", "path to an engine Trace JSON to judge against the reference (default: demo engine)")
	inject := fs.String("inject", "", "demo defect to inject into the engine path: decode|stop|report (default: none/clean)")
	asJSON := fs.Bool("json", false, "emit the machine-readable result JSON to stdout")
	if !parseFlags(fs, argv) {
		return 2
	}

	c := quality.DemoCase()
	if *casePath != "" {
		loaded, err := loadCase(*casePath)
		if err != nil {
			fmt.Fprintf(stderr, "fak quality run: %v\n", err)
			return 2
		}
		c = loaded
	}

	oracles, err := quality.Lookup(c.Oracles)
	if err != nil {
		fmt.Fprintf(stderr, "fak quality run: %v\n", err)
		return 2
	}

	var eng quality.Runner
	switch {
	case *enginePath != "":
		tr, err := loadTrace(*enginePath)
		if err != nil {
			fmt.Fprintf(stderr, "fak quality run: %v\n", err)
			return 2
		}
		eng = quality.ScriptedRunner{Label: "engine", Trace: tr}
	default:
		eng = quality.DemoEngine(*inject)
	}

	res, err := quality.RunCase(c, quality.ReferenceRunner{}, eng, oracles)
	if err != nil {
		fmt.Fprintf(stderr, "fak quality run: %v\n", err)
		return 2
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(res); err != nil {
			fmt.Fprintf(stderr, "fak quality run: encode result: %v\n", err)
			return 2
		}
	} else {
		fmt.Fprint(stdout, quality.Explain(res))
	}
	if !res.Pass {
		return 1
	}
	return 0
}

// runQualityExplain renders a stored result JSON (or a freshly run demo) as
// human-readable first-failure localization.
func runQualityExplain(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak quality explain", flag.ContinueOnError)
	fs.SetOutput(stderr)
	resultPath := fs.String("result", "", "path to a result JSON emitted by `fak quality run --json` (default: run the demo)")
	inject := fs.String("inject", "", "when running the demo, inject a defect: decode|stop|report")
	if !parseFlags(fs, argv) {
		return 2
	}

	var res quality.Result
	if *resultPath != "" {
		blob, err := os.ReadFile(*resultPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak quality explain: %v\n", err)
			return 2
		}
		if err := json.Unmarshal(blob, &res); err != nil {
			fmt.Fprintf(stderr, "fak quality explain: parse result: %v\n", err)
			return 2
		}
	} else {
		c := quality.DemoCase()
		oracles, err := quality.Lookup(c.Oracles)
		if err != nil {
			fmt.Fprintf(stderr, "fak quality explain: %v\n", err)
			return 2
		}
		res, err = quality.RunCase(c, quality.ReferenceRunner{}, quality.DemoEngine(*inject), oracles)
		if err != nil {
			fmt.Fprintf(stderr, "fak quality explain: %v\n", err)
			return 2
		}
	}
	fmt.Fprint(stdout, quality.Explain(res))
	if !res.Pass {
		return 1
	}
	return 0
}

// runQualityReplay is the ONE command that replays an injected failure from its
// bundle (#4515). Its sole input is the stored artifact — a failure bundle, or the
// whole result `fak quality run --json` emitted — and it reproduces the recorded
// failure from that artifact's own contents: no case file, no environment, no live
// engine. Exit codes follow `run`'s convention, read through the replay lens: 0 the
// bundle reproduced its recorded failure, 1 it did not (it replayed clean, replayed
// to a DIFFERENT failure, or was too incomplete to replay — inconclusive is never a
// pass), 2 a usage/IO error, which is never conflated with either.
func runQualityReplay(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	fs := flag.NewFlagSet("fak quality replay", flag.ContinueOnError)
	fs.SetOutput(stderr)
	bundlePath := fs.String("bundle", "", "path to a failure-bundle JSON or a `fak quality run --json` result (required; \"-\" reads stdin)")
	asJSON := fs.Bool("json", false, "emit the machine-readable replay verdict JSON to stdout")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *bundlePath == "" {
		fmt.Fprintln(stderr, "fak quality replay: -bundle is required (a replay's only input is the bundle)")
		return 2
	}

	var (
		blob []byte
		err  error
	)
	if *bundlePath == "-" {
		blob, err = io.ReadAll(stdin)
	} else {
		blob, err = os.ReadFile(*bundlePath)
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak quality replay: %v\n", err)
		return 2
	}

	b, err := quality.LoadBundle(blob)
	if err != nil {
		fmt.Fprintf(stderr, "fak quality replay: %v\n", err)
		return 2
	}

	v := quality.Replay(b)
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(v); err != nil {
			fmt.Fprintf(stderr, "fak quality replay: encode verdict: %v\n", err)
			return 2
		}
	} else {
		fmt.Fprint(stdout, quality.ExplainReplay(v))
	}
	if !v.Reproduced {
		return 1
	}
	return 0
}

func loadCase(path string) (quality.QualityCase, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return quality.QualityCase{}, err
	}
	var c quality.QualityCase
	if err := json.Unmarshal(blob, &c); err != nil {
		return quality.QualityCase{}, fmt.Errorf("parse case %s: %w", path, err)
	}
	if ok, why := c.Valid(); !ok {
		return quality.QualityCase{}, fmt.Errorf("invalid case %s: %s", path, why)
	}
	return c, nil
}

func loadTrace(path string) (quality.Trace, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return quality.Trace{}, err
	}
	var t quality.Trace
	if err := json.Unmarshal(blob, &t); err != nil {
		return quality.Trace{}, fmt.Errorf("parse trace %s: %w", path, err)
	}
	return t, nil
}
