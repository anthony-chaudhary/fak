package main

// fak milestone status-doc -- generate / freshness-check the durable milestone
// CLIMB status doc (docs/milestones/STATUS.md), the milestone twin of
// `fak support-maturity-scorecard --write-doc|--check-doc` (#1441, child of #1436).
//
//	fak milestone status-doc --write-doc    # regenerate the generated block in place
//	fak milestone status-doc --check-doc    # CI gate: red when the committed block is stale
//	fak milestone status-doc --block        # emit the generated block to stdout (no file)
//
// Only the maturity CLIMB is embedded (deterministic from covmatrix.Grid()); the epic
// ROADMAP is `gh`-fed and non-deterministic, so it stays on the Slack card + JSONL
// ledger and never the committed doc. The fold lives in internal/milestonereport; this
// file owns only the file I/O + the freshness verdict.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/milestonedoc"
)

// milestoneStatusDocRel is the committed status doc, under docs/milestones/ next to the
// durable trend ledger so the milestone state is one reviewable directory.
const milestoneStatusDocRel = "docs/milestones/STATUS.md"

// runMilestoneStatusDoc regenerates (--write-doc) or freshness-checks (--check-doc) the
// generated milestone-climb block inside docs/milestones/STATUS.md. The witness for
// #1441: the doc regenerates deterministically from the committed grid, and a stale
// cell reds --check-doc (and the cmd/fak freshness test that drives it).
func runMilestoneStatusDoc(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak milestone status-doc", flag.ContinueOnError)
	fs.SetOutput(stderr)
	writeDoc := fs.Bool("write-doc", false, "regenerate the milestone-climb block in "+milestoneStatusDocRel+" in place")
	checkDoc := fs.Bool("check-doc", false, "CI gate: red when the committed "+milestoneStatusDocRel+" block is stale vs the live grid")
	emitBlock := fs.Bool("block", false, "emit the generated milestone-climb block to stdout; do not touch the file")
	asJSON := fs.Bool("json", false, "emit the freshness verdict as JSON")
	workspace := fs.String("workspace", "", "workspace root (default: repo root) for --write-doc / --check-doc")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak milestone status-doc: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *emitBlock {
		fmt.Fprintln(stdout, milestonedoc.Block())
		return 0
	}
	if !*writeDoc && !*checkDoc {
		fmt.Fprintln(stderr, "fak milestone status-doc: pass --write-doc, --check-doc, or --block")
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	docPath := filepath.Join(root, filepath.FromSlash(milestoneStatusDocRel))

	if *writeDoc {
		return milestoneStatusDocWrite(stdout, stderr, docPath)
	}
	return milestoneStatusDocCheck(stdout, docPath, *asJSON)
}

// milestoneStatusDocWrite regenerates the generated block in place, scaffolding the
// doc on first write (so a missing file is created with the markers, not an error).
func milestoneStatusDocWrite(stdout, stderr io.Writer, docPath string) int {
	raw, err := os.ReadFile(docPath)
	doc := string(raw)
	if err != nil {
		if !os.IsNotExist(err) {
			fmt.Fprintf(stderr, "fak milestone status-doc --write-doc: read %s: %v\n", docPath, err)
			return 1
		}
		doc = milestonedoc.Scaffold()
		if mkErr := os.MkdirAll(filepath.Dir(docPath), 0o755); mkErr != nil {
			fmt.Fprintf(stderr, "fak milestone status-doc --write-doc: mkdir: %v\n", mkErr)
			return 1
		}
	}

	next, err := milestonedoc.Splice(doc)
	if err != nil {
		fmt.Fprintf(stderr, "fak milestone status-doc --write-doc: %v\n", err)
		return 1
	}
	if next == doc {
		fmt.Fprintf(stdout, "%s milestone-climb block already fresh; no change\n", milestoneStatusDocRel)
		return 0
	}
	if err := os.WriteFile(docPath, []byte(next), 0o644); err != nil {
		fmt.Fprintf(stderr, "fak milestone status-doc --write-doc: write %s: %v\n", docPath, err)
		return 1
	}
	fmt.Fprintf(stdout, "wrote %s milestone-climb block\n", milestoneStatusDocRel)
	return 0
}

// milestoneStatusDocCheck reds (exit 1) when the committed block drifted from the live
// fold or the markers are missing. It is advisory in spirit but exits non-zero so CI
// and the cmd/fak freshness test can gate on it, like the support-maturity matrix.
func milestoneStatusDocCheck(stdout io.Writer, docPath string, asJSON bool) int {
	raw, err := os.ReadFile(docPath)
	emit := func(fresh bool, msg string) int {
		if asJSON {
			fmt.Fprintf(stdout, "{\"doc\":%q,\"fresh\":%t,\"message\":%q}\n", milestoneStatusDocRel, fresh, msg)
		} else {
			fmt.Fprintln(stdout, msg)
		}
		if fresh {
			return 0
		}
		return 1
	}
	if err != nil {
		return emit(false, fmt.Sprintf("STALE  %s: not found; run `fak milestone status-doc --write-doc`", milestoneStatusDocRel))
	}
	doc := string(raw)
	if _, ok := milestonedoc.Extract(doc); !ok {
		return emit(false, fmt.Sprintf("STALE  %s: milestone-climb markers not found; run `fak milestone status-doc --write-doc`", milestoneStatusDocRel))
	}
	if !milestonedoc.Fresh(doc) {
		return emit(false, fmt.Sprintf("STALE  %s: milestone-climb block drifted from the live grid; run `fak milestone status-doc --write-doc`", milestoneStatusDocRel))
	}
	return emit(true, fmt.Sprintf("OK  %s milestone-climb block is fresh", milestoneStatusDocRel))
}
