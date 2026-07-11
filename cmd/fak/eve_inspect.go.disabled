package main

// fak eve inspect -- the impure shell over evebridge.InspectFS (#2601): reads
// an Eve app root (its documented agent/ layout) or a compiled .eve/ directory
// and emits the stable JSON inspect manifest — authored surfaces with their
// path-derived runtime names, typed diagnostics with source-evidence paths,
// and the fak policy/mount view — before the app ever serves traffic.
//
//	fak eve inspect [--json] [--policy-draft] [ROOT]
//	    ROOT defaults to ".". Output is JSON-first: the manifest (or, with
//	    --policy-draft, the derived fak-policy/v1 draft) always goes to
//	    stdout; a one-line verdict goes to stderr. --json is accepted for
//	    explicitness — the manifest is always JSON. Exit 0 = green manifest,
//	    3 = typed diagnostics gate the manifest closed (the JSON still
//	    carries them), 1 = root unreadable, 2 = usage.
//
// All impurity lives here (os.DirFS, flags, exit codes); the fold itself is
// pure in internal/evebridge.

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/evebridge"
)

// eveInspectFailed is the fail-closed exit: the root was readable, the fold
// ran, and at least one typed diagnostic gates the manifest.
const eveInspectFailed = 3

func runEveInspect(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("eve inspect", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonFlag := fs.Bool("json", false, "emit the manifest as JSON (accepted for explicitness; output is always JSON)")
	draftFlag := fs.Bool("policy-draft", false, "emit the fak-policy/v1 draft derived from a GREEN manifest instead of the manifest itself")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	_ = *jsonFlag // JSON-first: there is no non-JSON rendering to switch away from
	root := "."
	switch fs.NArg() {
	case 0:
	case 1:
		root = fs.Arg(0)
	default:
		fmt.Fprintf(stderr, "fak eve inspect: expected at most one ROOT argument, got %d\n", fs.NArg())
		return 2
	}
	if info, err := os.Stat(root); err != nil || !info.IsDir() {
		fmt.Fprintf(stderr, "fak eve inspect: root %q is not a readable directory (%v)\n", root, err)
		return 1
	}

	manifest, err := evebridge.InspectFS(os.DirFS(root))
	if err != nil {
		fmt.Fprintf(stderr, "fak eve inspect: %v\n", err)
		return 1
	}

	if *draftFlag {
		draft, err := manifest.PolicyDraft()
		if err != nil {
			fmt.Fprintf(stderr, "fak eve inspect: %v\n", err)
			return eveInspectFailed
		}
		if _, err := stdout.Write(draft.JSON()); err != nil {
			return 1
		}
		fmt.Fprintf(stderr, "eve inspect: policy draft — %d tool(s) allowed, %d write-region rule(s), unknown connection operations default-deny\n",
			len(draft.Allow), len(draft.ArgRules))
		return 0
	}

	if _, err := stdout.Write(manifest.JSON()); err != nil {
		return 1
	}
	if !manifest.OK {
		fails := 0
		for _, d := range manifest.Diagnostics {
			if d.Severity == "fail" {
				fails++
			}
		}
		fmt.Fprintf(stderr, "eve inspect: FAILED closed — %d gating diagnostic(s); see the JSON report's typed codes and evidence paths\n", fails)
		return eveInspectFailed
	}
	fmt.Fprintf(stderr, "eve inspect: ok — %d tool(s), %d connection(s), %d subagent(s), %d schedule(s), %d sandbox mount(s)\n",
		len(manifest.Tools), len(manifest.Connections), len(manifest.Subagents), len(manifest.Schedules), len(manifest.SandboxMounts))
	return 0
}
