package main

// fak workflow — keep ultracode-generated Workflow scripts fak-native (epic #1494 / C4 #1502).
//
//	fak workflow lint [<script>|-]   refute a fak-blind workflow: exit 1 unless it
//	                                 references self-index + memory + shared-path
//	fak workflow seed                emit the fak-native Workflow seed template
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
	case "seed":
		return workflowSeed(stdout, stderr, argv[1:])
	default:
		fmt.Fprintf(stderr, "fak workflow: unknown subcommand %q\n", argv[0])
		writeWorkflowUsage(stderr)
		return 2
	}
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
	var pos []string
	for rest := argv; ; {
		if err := fs.Parse(rest); err != nil {
			return 2
		}
		rest = fs.Args()
		if len(rest) == 0 {
			break
		}
		pos = append(pos, rest[0])
		rest = rest[1:]
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
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	fmt.Fprint(stdout, workflowlint.SeedTemplate)
	return 0
}

func writeWorkflowUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak workflow <lint|seed> [args]")
	fmt.Fprintln(w, "  lint [<script>|-]   refute a fak-blind workflow (exit 1 unless self-index + memory + shared-path)")
	fmt.Fprintln(w, "  seed                emit the fak-native Workflow seed template")
}
