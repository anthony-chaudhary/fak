package main

// agentreadinessscore.go wires the `fak score agent-readiness` verb over
// internal/agentreadinessscore: the agent-experience measuring stick. It re-derives, from
// the git-tracked tree, whether an autonomous coding agent (Claude Code, OpenAI Codex,
// Cursor, an MCP client) can DISCOVER fak, ADOPT it, and BUILD on it — folding each HARD
// mechanical defect into a friction-debt gate (lower = better, floor 0) and every real,
// working affordance into the unbounded experience-frontier (higher = better).
//
// This is the Go port of the former tools/agent_readiness_scorecard.py; the surfaces
// (default / --json / --markdown / --stamp / --compare) are preserved byte-for-byte in
// behavior so the committed snapshot and the control-pane fold are unchanged.

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/agentreadinessscore"
)

func emitScorecardComparison(stdout, stderr io.Writer, label, path string, exitCode int, compare func(map[string]any) string) (int, bool) {
	if path == "" {
		return 0, false
	}
	base, ok := readCompareBase(stderr, label, path)
	if !ok {
		return 2, true
	}
	fmt.Fprintln(stdout, compare(base))
	return exitCode, true
}

func cmdAgentReadinessScore(argv []string) {
	os.Exit(runAgentReadinessScore(os.Stdout, os.Stderr, argv))
}

// runAgentReadinessScore scores whether an agent can discover, adopt, and build on fak,
// re-derived from the git-tracked tree. Exit codes: 0 no friction-debt, 1 carries
// friction-debt, 2 usage/IO error.
func runAgentReadinessScore(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak score agent-readiness", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	asMarkdown := fs.Bool("markdown", false, "emit the snapshot markdown body")
	stamp := fs.String("stamp", "", "date stamp for the markdown header")
	comparePath := fs.String("compare", "", "compare the experience-frontier (+35% goal) and friction-debt delta vs a prior --json baseline")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak score agent-readiness: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	payload := agentreadinessscore.Build(root)
	if code, done := emitScorecardComparison(stdout, stderr, "fak score agent-readiness", *comparePath, okExit(payload.OK), func(base map[string]any) string {
		return agentreadinessscore.Compare(payload, base)
	}); done {
		return code
	}
	if code := emitJSONOrRender(stdout, stderr, "fak score agent-readiness", *asJSON, payload, func(w io.Writer) {
		if *asMarkdown {
			fmt.Fprint(w, agentreadinessscore.Markdown(payload, *stamp))
		} else {
			fmt.Fprintln(w, agentreadinessscore.Render(payload))
		}
	}); code != 0 {
		return code
	}
	if payload.OK {
		return 0
	}
	return 1
}
