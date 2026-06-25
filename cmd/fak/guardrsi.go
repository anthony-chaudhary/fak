package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/guardrsi"
)

func cmdGuardVerdictRSI(argv []string) { os.Exit(runGuardVerdictRSI(os.Stdout, os.Stderr, argv)) }

func runGuardVerdictRSI(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak guard-verdict-rsi", flag.ContinueOnError)
	fs.SetOutput(stderr)
	checkPath := fs.String("check", "", "honesty-gate an emitted iteration JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *checkPath != "" {
		if fs.NArg() != 0 {
			fmt.Fprintf(stderr, "fak guard-verdict-rsi: unexpected argument %q\n", fs.Arg(0))
			return 2
		}
		return runGuardVerdictRSICheck(stdout, stderr, *checkPath)
	}
	args := fs.Args()
	cmd := "run"
	if len(args) > 0 {
		cmd = args[0]
		args = args[1:]
	}
	switch cmd {
	case "fold":
		return runGuardVerdictRSIFold(stdout, stderr, args)
	case "run":
		return runGuardVerdictRSIRun(stdout, stderr, args)
	case "-h", "--help", "help":
		guardVerdictRSIUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak guard-verdict-rsi: unknown command %q\n", cmd)
		guardVerdictRSIUsage(stderr)
		return 2
	}
}

func runGuardVerdictRSIFold(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak guard-verdict-rsi fold", flag.ContinueOnError)
	fs.SetOutput(stderr)
	audit := fs.String("audit", "", "explicit guard-audit.jsonl to fold")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak guard-verdict-rsi fold: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	payload := guardrsi.BuildFold(root, *audit)
	if *asJSON {
		return encodeGuardRSIJSON(stdout, stderr, "fak guard-verdict-rsi fold", payload)
	}
	fmt.Fprintf(stdout, "guard-verdict-rsi fold: rows %d  quality %.3g  by_verdict %v\n",
		payload.Fold.TotalRows, payload.VerdictQuality, payload.Fold.ByVerdict)
	if len(payload.JournalPaths) == 0 {
		fmt.Fprintf(stdout, "  (no journal: %s)\n", guardrsi.DiagnoseAuditGap(root))
	}
	return 0
}

func runGuardVerdictRSIRun(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak guard-verdict-rsi run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	audit := fs.String("audit", "", "explicit guard-audit.jsonl to fold")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	witnessJSON := fs.String("witness", "", "JSON witness object not authored by the loop, e.g. {\"ok\":true}")
	outPath := fs.String("out", "", "write iteration JSON to this file")
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak guard-verdict-rsi run: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	var witness map[string]any
	if *witnessJSON != "" {
		if err := json.Unmarshal([]byte(*witnessJSON), &witness); err != nil {
			fmt.Fprintf(stderr, "fak guard-verdict-rsi run: parse --witness: %v\n", err)
			return 2
		}
	}
	it := guardrsi.RunIteration(root, *audit, witness)
	if *outPath != "" {
		b, _ := json.MarshalIndent(it, "", "  ")
		if err := os.WriteFile(*outPath, append(b, '\n'), 0o644); err != nil {
			fmt.Fprintf(stderr, "fak guard-verdict-rsi run: write --out: %v\n", err)
			return 1
		}
	}
	if *asJSON {
		return encodeGuardRSIJSON(stdout, stderr, "fak guard-verdict-rsi run", it)
	}
	fmt.Fprintln(stdout, guardrsi.RenderIteration(it))
	return 0
}

func runGuardVerdictRSICheck(stdout, stderr io.Writer, path string) int {
	b, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard-verdict-rsi --check: read: %v\n", err)
		return 2
	}
	var it guardrsi.Iteration
	if err := json.Unmarshal(b, &it); err != nil {
		fmt.Fprintf(stderr, "fak guard-verdict-rsi --check: parse: %v\n", err)
		return 2
	}
	violations := guardrsi.CheckIteration(it)
	if len(violations) > 0 {
		fmt.Fprintln(stdout, "guard-verdict-rsi --check: FAIL")
		for _, v := range violations {
			fmt.Fprintf(stdout, "  - %s\n", v)
		}
		return 1
	}
	fmt.Fprintln(stdout, "guard-verdict-rsi --check: OK (iteration is honest)")
	return 0
}

func cmdGuardRSIScorecard(argv []string) { os.Exit(runGuardRSIScorecard(os.Stdout, os.Stderr, argv)) }

func runGuardRSIScorecard(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak guard-rsi-scorecard", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	asMarkdown := fs.Bool("markdown", false, "emit scorecard markdown")
	comparePath := fs.String("compare", "", "compare against a prior --json payload")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak guard-rsi-scorecard: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	payload := guardrsi.BuildScorecard(root)
	if *comparePath != "" {
		b, err := os.ReadFile(*comparePath)
		if err != nil {
			fmt.Fprintf(stderr, "fak guard-rsi-scorecard: read --compare: %v\n", err)
			return 2
		}
		var base map[string]any
		if err := json.Unmarshal(b, &base); err != nil {
			fmt.Fprintf(stderr, "fak guard-rsi-scorecard: parse --compare: %v\n", err)
			return 2
		}
		fmt.Fprintln(stdout, guardrsi.Compare(payload, base))
		if payload.OK {
			return 0
		}
		return 1
	}
	if *asJSON {
		_ = encodeGuardRSIJSON(stdout, stderr, "fak guard-rsi-scorecard", payload)
	} else if *asMarkdown {
		fmt.Fprint(stdout, guardrsi.Markdown(payload))
	} else {
		fmt.Fprintln(stdout, guardrsi.RenderScorecard(payload))
	}
	if payload.OK {
		return 0
	}
	return 1
}

func encodeGuardRSIJSON(stdout, stderr io.Writer, label string, v any) int {
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		fmt.Fprintf(stderr, "%s: encode json: %v\n", label, err)
		return 1
	}
	return 0
}

func guardVerdictRSIUsage(w io.Writer) {
	fmt.Fprint(w, `fak guard-verdict-rsi - RSI loop over the real guard decision journal

  fak guard-verdict-rsi fold [--audit FILE] [--json]
  fak guard-verdict-rsi run  [--audit FILE] [--witness JSON] [--json] [--out FILE]
  fak guard-verdict-rsi --check ITER.json
`)
}
