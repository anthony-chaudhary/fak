package main

import (
	"fmt"
	"io"
	"path/filepath"
)

// announceAgentReport tells the caller, in plain words, that `fak agent` just
// left a file behind and exactly where it landed (#5473).
//
// `fak agent --offline` is the first command the README hands a stranger, and
// the --out default is the bare name "agent-report.json" — a cwd-relative path —
// so the proof drops a file into whatever directory the user's shell happened to
// be in. The run summary already ends with "report written: <the --out value>",
// but a bare filename does not say WHICH directory, which is the part a
// first-time evaluator needs in order to read the evidence or delete it again.
//
// The notice resolves the path to an absolute one and names the flag that
// redirects it. It goes to stderr on purpose: the report on stdout is a
// published transcript (GETTING-STARTED.md, docs/fak/tutorial.md,
// examples/agent-ab/EXAMPLE-OUTPUT.md) and is what a script would parse, so this
// is additive and cannot break either. Nothing about the file's location,
// contents, or schema changes — an explicit --out is still honoured verbatim.
func announceAgentReport(w io.Writer, out string) {
	shown := out
	if abs, err := filepath.Abs(out); err == nil {
		shown = abs
	}
	fmt.Fprintf(w, "fak agent: wrote the run report to %s (pass --out PATH to write it elsewhere)\n", shown)
}
