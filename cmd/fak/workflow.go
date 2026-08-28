package main

// fak workflow — keep ultracode-generated Workflow scripts fak-native (epic #1494 / C4 #1502).
//
//	fak workflow lint [<script>|-]   refute a fak-blind workflow: exit 1 unless it
//	                                 references self-index + memory + shared-path
//	fak workflow seed                emit the fak-native Workflow seed template
//	fak workflow resume <run-dir>    skip only the steps whose completion is witnessed,
//	                                 re-execute everything else (#2444, workflow_resume.go)
//
// `lint` is a real gate — exit 0 = FAK-NATIVE, exit 1 = FAK-BLIND — so it can sit on
// the ultracode generation path and a fak-guarded session cannot emit a workflow that
// orchestrates generic agents instead of fak-native ones (self-index understand,
// per-agent memory recall/compact, arbitrated shared-path leases). Thin shell over
// internal/workflowlint; --json makes the verdict MCP- and CI-consumable.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/interspersedflags"
	"github.com/anthony-chaudhary/fak/internal/workflowlint"
)

func cmdWorkflow(argv []string) { os.Exit(runWorkflow(os.Stdout, os.Stderr, os.Stdin, argv)) }

func runWorkflow(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	if len(argv) == 0 {
		writeWorkflowUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "lint":
		return workflowLint(stdout, stderr, stdin, argv[1:])
	case "check":
		return workflowCheck(stdout, stderr, argv[1:])
	case "seed":
		return workflowSeed(stdout, stderr, argv[1:])
	case "resume":
		return workflowResume(stdout, stderr, argv[1:], nil)
	default:
		fmt.Fprintf(stderr, "fak workflow: unknown subcommand %q\n", argv[0])
		writeWorkflowUsage(stderr)
		return 2
	}
}

// workflowCheck exposes the GitHub Actions structural checker. With --base it is the
// push-path gate: first prove the immutable base...tip range touches .github/workflows,
// then archive and inspect tip rather than trusting the checkout. Without --base it checks
// the named current tree, which is useful for local/CI witnesses.
func workflowCheck(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("workflow check", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	base := fs.String("base", "", "base object for a pushed range (enables immutable-tip mode)")
	tip := fs.String("tip", "", "immutable pushed tip (default: HEAD)")
	asJSON := fs.Bool("json", false, "emit workflowStructureResult as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	r := resolveRoot(*root)
	if r == "" {
		fmt.Fprintln(stderr, "fak workflow check: not in a git repo (or git unavailable)")
		return 2
	}

	var res workflowStructureResult
	var code int
	if strings.TrimSpace(*base) == "" {
		findings, err := workflowlint.CheckTree(r)
		res = workflowStructureResult{Schema: "fak.workflow_structure.v1", Verdict: "OK", OK: err == nil && len(findings) == 0, Root: r, Findings: findings}
		if err != nil {
			res.Verdict, res.Detail, code = "COULD_NOT_RUN", err.Error(), 2
		} else if len(findings) > 0 {
			res.Verdict, code = "WORKFLOW_STRUCTURE", 1
		}
	} else {
		res, code = evaluatePrePushWorkflow(r, *base, *tip)
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(res); err != nil {
			fmt.Fprintf(stderr, "fak workflow check: %v\n", err)
			return 2
		}
		return code
	}
	if res.Verdict == "NOOP" {
		fmt.Fprintln(stdout, "WORKFLOW_STRUCTURE NOOP: pushed range does not touch .github/workflows/**")
		return code
	}
	if res.Touched {
		fmt.Fprintln(stderr, "WORKFLOW_SCOPE warning: pushing .github/workflows/** requires GitHub OAuth/PAT `workflow` scope; verify the credential before the server-side refusal.")
	}
	for _, finding := range res.Findings {
		fmt.Fprintf(stderr, "WORKFLOW_STRUCTURE %s:%d [%s] %s\n", finding.Path, finding.Line, finding.Code, finding.Message)
	}
	if code == 0 {
		fmt.Fprintf(stdout, "WORKFLOW_STRUCTURE OK: %s has zero findings\n", res.Ref)
	} else if code == 2 {
		fmt.Fprintf(stderr, "WORKFLOW_STRUCTURE could not run: %s\n", res.Detail)
	}
	return code
}

// workflowLint reads a Workflow script (from a path arg, or "-"/no arg = stdin) and
// adjudicates it. Exit 1 on FAK-BLIND so it gates a generation pipeline; exit 0 on
// FAK-NATIVE. The --json form emits the full per-class workflowlint.Report.
func workflowLint(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	fs := flag.NewFlagSet("workflow lint", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the verdict as JSON (workflowlint.Report)")
	// Accept --json before OR after the positional script path. Go's flag package
	// stops at the first non-flag arg, so interleave Parse with positional collection
	// (the same ergonomics as `fak index`).
	pos, parseErr := interspersedflags.Parse(fs, argv)
	if parseErr != nil {
		return 2
	}

	src := "-"
	if len(pos) > 0 {
		src = pos[0]
	}
	var (
		data []byte
		err  error
	)
	if src == "-" {
		data, err = io.ReadAll(stdin)
	} else {
		data, err = os.ReadFile(src)
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak workflow lint: %v\n", err)
		return 2
	}

	rep := workflowlint.Lint(string(data))
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintf(stderr, "fak workflow lint: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "%s\n", rep.Verdict)
		for _, c := range rep.Classes {
			mark := "MISS"
			if c.Present {
				mark = " ok "
			}
			fmt.Fprintf(stdout, "  [%s] %-12s %s\n", mark, c.Key, c.Why)
		}
		if !rep.Native {
			fmt.Fprintf(stdout, "refused: missing fak concept(s): %v\n", rep.Missing)
		}
	}
	if !rep.Native {
		return 1
	}
	return 0
}

// workflowSeed prints the canonical fak-native seed template. Pipe it into lint to
// see the verdict it earns by construction: `fak workflow seed | fak workflow lint -`.
func workflowSeed(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("workflow seed", flag.ContinueOnError)
	fs.SetOutput(stderr)
	if !parseFlags(fs, argv) {
		return 2
	}
	fmt.Fprint(stdout, workflowlint.SeedTemplate)
	return 0
}

func writeWorkflowUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak workflow <lint|check|seed|resume> [args]")
	fmt.Fprintln(w, "  lint [<script>|-]   refute a fak-blind workflow (exit 1 unless self-index + memory + shared-path)")
	fmt.Fprintln(w, "  check               structurally check .github/workflows (use --base/--tip for a pushed range)")
	fmt.Fprintln(w, "  seed                emit the fak-native Workflow seed template")
	fmt.Fprintln(w, "  resume <run-dir>    replay only witnessed steps, re-execute the rest (skipped=N executed=M)")
}
