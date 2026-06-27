package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/hooks"
)

// cmd/fak/hygiene.go — `fak hygiene`: run the repo's whole-tree (`--audit-tree`) hygiene gates IN
// ONE PROCESS instead of spawning a Python interpreter per checker. `make hygiene` (part of the
// mandated pre-push `make ci`, mirrored in CI) historically spawned 11 Python interpreters, each
// doing sub-millisecond regex/path work over `git ls-files`; on Windows that is ~15-20s of pure
// process-create + Defender-scan tax before any checking happens. This reads `git ls-files` ONCE
// and runs every ported gate over it — the same collapse `fak hooks` made for the pre-commit hook.
//
// Exit codes mirror the gate contract so the Makefile / CI wrapper can fall back to Python:
// 0 = clean, 1 = a hygiene gate fired, 2 = could-not-run (the wrapper then runs the Python path).
//
// `--gates A,B,...` runs only the named gates (so `make index-sync` can call this for INDEX_SYNC
// while `make hygiene` runs the rest); the default is every gate HygieneGates() returns. The
// remaining make-hygiene checkers (demo_* x3, brand_consistency, scrub_hardware_names,
// guard_mcp_status_audit) stay on the Python path until ported (#928 A3/A4/A5).

func cmdHygiene(argv []string) { os.Exit(runHygiene(os.Stdout, os.Stderr, argv)) }

func runHygiene(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("hygiene", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	asJSON := fs.Bool("json", false, "emit findings as JSON")
	gatesCSV := fs.String("gates", "", "comma-separated gate names to run (default: all)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	r := resolveRoot(*root)
	if r == "" {
		return 2 // not in a repo => could-not-run => fall through to python
	}

	d, err := hooks.ReadTrackedTree(r)
	if err != nil {
		// could-not-run: never wedge the gate. The wrapper treats exit 2 as "fall back to python".
		return 2
	}

	want := gateFilter(*gatesCSV)
	var allFindings []hooks.Finding
	blocked := false
	for _, g := range hooks.HygieneGates() {
		if want != nil && !want[g.Name] {
			continue
		}
		findings, gerr := g.Check(d)
		if gerr != nil {
			// a single gate that could-not-run is skipped (fail-open); the others still run.
			continue
		}
		if len(findings) == 0 {
			continue
		}
		allFindings = append(allFindings, findings...)
		blocked = true
		if !*asJSON {
			fmt.Fprintf(stderr, "%s: %d finding(s):\n", g.Name, len(findings))
			for _, f := range findings {
				fmt.Fprintf(stderr, "  %s\n", formatFinding(f))
			}
		}
	}

	if *asJSON {
		emitFindingsJSON(stdout, stderr, allFindings)
	}
	if blocked {
		if !*asJSON {
			fmt.Fprintln(stderr, "")
			fmt.Fprintln(stderr, "hygiene refused by a tree gate (above) — the --audit-tree backstop make ci runs HARD.")
		}
		return 1
	}
	if !*asJSON {
		fmt.Fprintln(stdout, "hygiene OK")
	}
	return 0
}

// gateFilter parses --gates into a set, or nil to mean "all". Names are upper-cased and trimmed so
// `--gates index_sync` and `--gates INDEX_SYNC` both resolve.
func gateFilter(csv string) map[string]bool {
	csv = strings.TrimSpace(csv)
	if csv == "" {
		return nil
	}
	want := map[string]bool{}
	for _, n := range strings.Split(csv, ",") {
		n = strings.ToUpper(strings.TrimSpace(n))
		if n != "" {
			want[n] = true
		}
	}
	return want
}
