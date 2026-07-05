package main

// fak knownbad -- the impure shell over the internal/knownbad fold core: the two
// verbs the blast-radius containment epic (#2712) spine exposes.
//
//   fak knownbad record --tree internal/foo/** --reason build --note "..."
//       appends one fak.known-bad.v1 JSONL row to a fleet-visible ledger.
//   fak knownbad match  --tree internal/foo/bar.go [--json]
//       reports whether the requested tree intersects any LIVE (open, unexpired)
//       known-bad signature, printing the matching record(s). Exit is non-zero
//       (with --json, matched:false) when nothing matches, so a worker OR the
//       dispatcher can short-circuit before burning a cycle.
//
// All impurity lives here: the ledger read/write, the clock (Unix seconds,
// injected as `nowUnix` so runKnownBad is deterministic under test), and flag
// parsing. The signature derivation, tree intersection, and liveness are the pure
// core in internal/knownbad.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/knownbad"
)

func cmdKnownBad(argv []string) {
	os.Exit(runKnownBad(os.Stdout, os.Stderr, argv, time.Now().Unix()))
}

func runKnownBad(stdout, stderr io.Writer, argv []string, nowUnix int64) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak knownbad: expected a subcommand (record|match)")
		return 2
	}
	switch argv[0] {
	case "record":
		return runKnownBadRecord(stdout, stderr, argv[1:], nowUnix)
	case "match":
		return runKnownBadMatch(stdout, stderr, argv[1:], nowUnix)
	case "-h", "--help", "help":
		fmt.Fprintln(stderr, "fak knownbad: record | match  (fleet-wide known-bad signature ledger)")
		return 0
	default:
		fmt.Fprintf(stderr, "fak knownbad: unknown subcommand %q (want record|match)\n", argv[0])
		return 2
	}
}

// knownBadLedgerPath resolves the ledger to write/read: an explicit --ledger wins,
// otherwise the repo-root-relative fleet default.
func knownBadLedgerPath(override string) string {
	if strings.TrimSpace(override) != "" {
		return override
	}
	return filepath.Join(repoRoot(), knownbad.DefaultLedgerRel)
}

// knownBadDiscoverer resolves the discovered_by id when --by is not given: the
// FAK_AGENT_ID env if present, else the hostname, else "unknown".
func knownBadDiscoverer(by string) string {
	if s := strings.TrimSpace(by); s != "" {
		return s
	}
	if s := strings.TrimSpace(os.Getenv("FAK_AGENT_ID")); s != "" {
		return s
	}
	if h, err := os.Hostname(); err == nil && strings.TrimSpace(h) != "" {
		return h
	}
	return "unknown"
}

func runKnownBadRecord(stdout, stderr io.Writer, argv []string, nowUnix int64) int {
	fs := flag.NewFlagSet("knownbad record", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var trees stringList
	fs.Var(&trees, "tree", "repo-relative tree glob the failure covers (repeatable), e.g. internal/foo/**")
	reason := fs.String("reason", "", "failure class (required), e.g. build, test, lint")
	note := fs.String("note", "", "free-text note describing the known-bad")
	by := fs.String("by", "", "discoverer id (default: $FAK_AGENT_ID, else hostname)")
	failureHash := fs.String("failure-hash", "", "optional guardrsi failure hash to fold into the signature")
	ttl := fs.Int64("ttl", 0, "time-to-live in seconds (0 = no expiry)")
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+knownbad.DefaultLedgerRel+")")
	asJSON := fs.Bool("json", false, "emit the recorded row as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if len(trees) == 0 {
		fmt.Fprintln(stderr, "fak knownbad record: at least one --tree is required")
		return 2
	}
	if strings.TrimSpace(*reason) == "" {
		fmt.Fprintln(stderr, "fak knownbad record: --reason is required")
		return 2
	}
	rec := knownbad.NewRecord(*reason, []string(trees), *note, knownBadDiscoverer(*by), *failureHash, nowUnix, *ttl)
	if len(rec.TreeGlobs) == 0 {
		fmt.Fprintln(stderr, "fak knownbad record: --tree produced no valid repo-relative globs")
		return 2
	}
	path := knownBadLedgerPath(*ledger)
	if err := appendKnownBadRow(path, rec); err != nil {
		fmt.Fprintf(stderr, "fak knownbad record: %v\n", err)
		return 1
	}
	if *asJSON {
		return knownBadEmitJSON(stdout, stderr, rec)
	}
	fmt.Fprintf(stdout, "recorded known-bad %s reason=%s trees=%s -> %s\n",
		rec.Signature, rec.ReasonClass, strings.Join(rec.TreeGlobs, ","), path)
	return 0
}

func runKnownBadMatch(stdout, stderr io.Writer, argv []string, nowUnix int64) int {
	fs := flag.NewFlagSet("knownbad match", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var trees stringList
	fs.Var(&trees, "tree", "repo-relative tree glob (or file) to check (repeatable)")
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+knownbad.DefaultLedgerRel+")")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if len(trees) == 0 {
		fmt.Fprintln(stderr, "fak knownbad match: at least one --tree is required")
		return 2
	}
	path := knownBadLedgerPath(*ledger)
	records, err := readKnownBadLedger(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak knownbad match: %v\n", err)
		return 1
	}
	matches := knownbad.Match(records, knownbad.Query{TreeGlobs: []string(trees)}, nowUnix)

	if *asJSON {
		out := struct {
			Schema  string            `json:"schema"`
			Matched bool              `json:"matched"`
			Query   []string          `json:"query"`
			Count   int               `json:"count"`
			Records []knownbad.Record `json:"records"`
		}{
			Schema:  knownbad.Schema,
			Matched: len(matches) > 0,
			Query:   []string(trees),
			Count:   len(matches),
			Records: matches,
		}
		if code := knownBadEmitJSON(stdout, stderr, out); code != 0 {
			return code
		}
	} else if len(matches) == 0 {
		fmt.Fprintf(stdout, "matched:false  no live known-bad signature intersects %s\n", strings.Join([]string(trees), ","))
	} else {
		fmt.Fprintf(stdout, "matched:true  %d live known-bad signature(s) intersect %s:\n", len(matches), strings.Join([]string(trees), ","))
		for _, m := range matches {
			fmt.Fprintf(stdout, "  %s reason=%s trees=%s by=%s note=%q\n",
				m.Signature, m.ReasonClass, strings.Join(m.TreeGlobs, ","), m.DiscoveredBy, m.Note)
		}
	}

	// Exit non-zero on a match so a worker/dispatcher can short-circuit in shell.
	if len(matches) > 0 {
		return 3
	}
	return 0
}

// appendKnownBadRow appends one record as a JSONL line, creating the ledger's
// parent directory on first write — the same append idiom the other fak ledgers
// use (see appendLedgerRow in cadence.go).
func appendKnownBadRow(path string, rec knownbad.Record) error {
	line, err := knownbad.MarshalLine(rec)
	if err != nil {
		return err
	}
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString(line + "\n")
	return err
}

// readKnownBadLedger reads the ledger file and folds it to records. A missing
// ledger is not an error — it means nothing has been recorded yet (no match).
func readKnownBadLedger(path string) ([]knownbad.Record, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	return knownbad.ParseLedger(data), nil
}

// knownBadEmitJSON marshals v as indented JSON to stdout; a marshal failure is a
// rc-1 error on stderr.
func knownBadEmitJSON(stdout, stderr io.Writer, v any) int {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "fak knownbad: json: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, string(b))
	return 0
}
