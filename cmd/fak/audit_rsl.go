package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/rsl"
)

// cmdAuditRSL is the REFERENCE STATE LOG side of `fak audit` (#3190, borrowed
// from gittuf's RSL). Where `fak audit verify` re-checks a guard DECISION
// journal's hash chain, `fak audit rsl` replays a git Reference State Log — the
// append-only, hash-chained record of observed trunk ref transitions — and
// asserts BOTH invariants an offline auditor needs: the chain is un-tampered AND
// every ref moved fast-forward-only (no rewrite / force-push). It is the
// forge-independent no-force-push proof: it fails, naming the offending ref, on
// the exact transition where the trunk was rewound, without trusting GitHub's
// server-side ruleset.
//
// Exit: 0 the chain + fast-forward invariant hold; 1 a broken chain or a
// non-fast-forward gap (naming the ref); 2 a read/setup error.
func cmdAuditRSL(args []string) {
	fs := flag.NewFlagSet("audit rsl", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: fak audit rsl <rsl.jsonl>")
		fmt.Fprintln(os.Stderr, "  (replay a git Reference State Log; exit 1 on a tampered chain or a non-fast-forward gap)")
	}
	_ = fs.Parse(args)
	if fs.NArg() != 1 {
		fs.Usage()
		os.Exit(2)
	}
	os.Exit(runAuditRSL(os.Stdout, os.Stderr, fs.Arg(0)))
}

// runAuditRSL replays the RSL at path and returns the process exit code, so the
// verdict logic is unit-tested without spawning a process (the runAuditReplay
// pattern).
func runAuditRSL(stdout, stderr io.Writer, path string) int {
	rows, err := rsl.ReadRows(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak audit rsl: %v\n", err)
		return 2
	}
	n, verr := rsl.Verify(rows)
	if verr != nil {
		fmt.Fprintf(stderr, "fak audit rsl: %s — REWRITTEN/TAMPERED after %d sound row(s): %v\n", path, n, verr)
		return 1
	}
	fmt.Fprintf(stdout, "fak audit rsl: %s — OK: %d hash-chained ref transition(s), chain intact and fast-forward-only (no force-push)\n", path, n)
	return 0
}
