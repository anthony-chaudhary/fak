package main

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// route_coverage.go — the `--accounts-cover` helper for `fak route`. It answers the
// whole-policy question the per-decision `--accounts` binding never does: does a
// user's OWN account roster bind EVERY model id the built-in routing manifest can
// emit? It leans on the pure modelroute cross-check (Manifest.ModelIDs enumerates the
// routed-id manifest; Roster.Cover classifies each id as bound / via-default /
// UNBOUND) and only adds the CLI shell: load the roster, render the report, and map
// the coverage tally onto a process exit code (0 covered, 1 a hole that would FAIL at
// dispatch). Kept out of route.go so runRoute stays under the god-function gate.

// runAccountsCover loads the roster at path, cross-checks it against the built-in
// routing manifest's routed-id set, prints a report (human or --json), and returns the
// exit code: 0 when every routed id is covered, 1 when the roster fails to load OR
// leaves any id UNBOUND (no binding and no default). The routed-id manifest is exactly
// modelroute.DefaultManifest().ModelIDs() — every member + scout across the rules and
// the fail-closed default plan — so coverage is measured against the real policy, not
// a guess.
func runAccountsCover(stdout, stderr io.Writer, path string, asJSON bool) int {
	r, err := modelroute.LoadRoster(path)
	if err != nil {
		fmt.Fprintln(stderr, "fak route:", err)
		return 1
	}
	cov := r.Cover(modelroute.DefaultManifest().ModelIDs())
	if asJSON {
		fmt.Fprintln(stdout, coverageJSON(cov, path))
	} else {
		fmt.Fprint(stdout, coverageReport(cov, path))
	}
	if cov.Unbound > 0 {
		return 1
	}
	return 0
}

// coverageReport renders the whole-policy coverage as an operator-readable table: one
// row per routed id with its disposition (BOUND / VIA-DEFAULT / UNBOUND) and, when
// covered, the serving account. It ends with a COMPLETE verdict when nothing is
// unbound, else an INCOMPLETE verdict that names the count of fail-at-dispatch holes.
func coverageReport(cov modelroute.Coverage, path string) string {
	var sb strings.Builder
	sb.WriteString("== fak route accounts coverage ==\n")
	sb.WriteString(fmt.Sprintf("roster: %s\n", path))
	sb.WriteString(fmt.Sprintf("routed ids: %d  (bound=%d via-default=%d unbound=%d)\n\n",
		len(cov.Rows), cov.Bound, cov.Default, cov.Unbound))
	for _, row := range cov.Rows {
		acct := row.Account
		if acct == "" {
			acct = "-"
		}
		sb.WriteString(fmt.Sprintf("  %-20s %-12s account=%-16s upstream=%s\n",
			row.Model, strings.ToUpper(string(row.Status)), acct, orDash(row.UpstreamModel)))
	}
	sb.WriteString("\n")
	if cov.Unbound == 0 {
		sb.WriteString(fmt.Sprintf("verdict: COMPLETE -- all %d routed id(s) are covered by the roster.\n", len(cov.Rows)))
	} else {
		sb.WriteString(fmt.Sprintf("verdict: INCOMPLETE -- %d routed id(s) UNBOUND (no binding and no default); these routes FAIL at dispatch.\n", cov.Unbound))
	}
	return sb.String()
}

// coverageJSON renders the coverage as a stable machine object. It carries the
// per-id rows plus the tallies; the load-bearing field for callers is the exact
// "unbound": <N> count (0 == the roster fully covers the manifest).
func coverageJSON(cov modelroute.Coverage, path string) string {
	obj := map[string]any{
		"roster":      path,
		"complete":    cov.Unbound == 0,
		"rows":        cov.Rows,
		"bound":       cov.Bound,
		"via_default": cov.Default,
		"unbound":     cov.Unbound,
	}
	b, _ := json.MarshalIndent(obj, "", "  ")
	return string(b)
}
