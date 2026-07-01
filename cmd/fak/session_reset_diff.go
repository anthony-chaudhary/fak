package main

// session_reset_diff.go -- issue #1575: `fak session reset-diff`, the CLI front end
// over internal/sessionreset.DiffReset. It reads a JSON description of a drained
// transcript (the same shape resetServedSessionOnBudget already builds from a live
// gateway's messages: an old/new trace id, the Msg list, and the fresh context-token
// budget), folds it through the SAME BuildSeed/BuildResetTransaction path the live
// reset hook uses, and renders the before/after diff -- what survived, what was
// summarized, what expired, and what needs an explicit follow-up query -- either as
// human-readable text (Explain), Markdown (--md), or JSON (--json).
//
// This performs NO new capture: DiffReset (internal/sessionreset/diff.go) classifies
// exactly the Seed/ResetTransaction fields the reset path already computes. The CLI
// is offline and gateway-free by design (json in, json/text/markdown out) -- unlike
// `fak session ls/status/...`, it never dials a live gateway, mirroring the
// `fak callavoid`/`fak dispatch order` --in-FILE-or-stdin convention already used
// elsewhere in this file's siblings for a pure computation over a JSON payload.
//
//	fak session reset-diff [--in FILE] [--json] [--md]

import (
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/sessionreset"
)

// resetDiffRequest is the CLI's JSON input contract: everything DiffReset needs to
// fold BuildSeed/BuildResetTransaction over a drained transcript. OldTrace/NewTrace
// name the lineage the same way BuildResetTransaction's parent/newTrace args do;
// Messages/FreshBudgetTok map straight onto sessionreset.Input.
type resetDiffRequest struct {
	OldTrace       string             `json:"old_trace"`
	NewTrace       string             `json:"new_trace"`
	Messages       []sessionreset.Msg `json:"messages"`
	FreshBudgetTok int                `json:"fresh_budget_tok"`
}

// runSessionResetDiff is the testable shell: it returns the process exit code (0 ok,
// 2 malformed input/usage) and takes its streams explicitly.
func runSessionResetDiff(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak session reset-diff", flag.ContinueOnError)
	fs.SetOutput(stderr)
	inPath := fs.String("in", "", "read resetDiffRequest JSON from this file (default: stdin)")
	asJSON := fs.Bool("json", false, "emit the ResetDiff as JSON")
	asMD := fs.Bool("md", false, "emit the ResetDiff as Markdown")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	raw, code := readCallavoidInput(stdin, stderr, *inPath, "reset-diff")
	if code != 0 {
		return code
	}
	var req resetDiffRequest
	if err := strictUnmarshal(raw, &req); err != nil {
		fmt.Fprintf(stderr, "fak session reset-diff: invalid input JSON: %v\n", err)
		return 2
	}
	if req.NewTrace == "" {
		fmt.Fprintln(stderr, "fak session reset-diff: new_trace is required")
		return 2
	}

	in := sessionreset.Input{Trace: req.OldTrace, Messages: req.Messages, FreshBudgetTok: req.FreshBudgetTok}
	seed := sessionreset.BuildSeed(in)
	tx := sessionreset.BuildResetTransaction(in, req.NewTrace, seed)
	diff := sessionreset.DiffReset(in, seed, tx)

	switch {
	case *asJSON:
		return encodeJSONOrFail(stdout, stderr, diff, "fak session reset-diff")
	case *asMD:
		fmt.Fprint(stdout, diff.Markdown())
	default:
		fmt.Fprint(stdout, diff.Explain())
	}
	return 0
}
