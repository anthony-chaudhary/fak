package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// accountsCheckTwins is the audit gate for Regression A: RED (exit 1) when two config homes
// logged into DIFFERENT accounts share one setup-token fingerprint (the cross-account smear that
// surfaces as "subscription disabled"). Homes that share a token but resolve to ONE account pass.
func accountsCheckTwins(stdout, stderr io.Writer, homeDir string, asJSON bool) int {
	findings, err := accounts.AuditTokenTwins(homeDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	if asJSON {
		stdout.Write(mustJSON(map[string]any{"clean": len(findings) == 0, "findings": findings}))
		fmt.Fprintln(stdout)
	}
	if len(findings) == 0 {
		if !asJSON {
			fmt.Fprintln(stdout, "ok: no cross-account token-twins — every shared setup token is one account")
		}
		return 0
	}
	if !asJSON {
		for _, f := range findings {
			fmt.Fprintf(stdout, "TOKEN-TWIN: homes [%s] share one setup token but log into %d accounts [%s]\n",
				strings.Join(f.Homes, ", "), len(f.Accounts), strings.Join(f.Accounts, ", "))
		}
		fmt.Fprintf(stderr, "fak accounts: %d cross-account token-twin(s) — a foreign token will surface as "+
			"\"subscription disabled\". Give each account its OWN setup token in its OWN dir.\n", len(findings))
	}
	return 1
}

// accountsGateWrite is the pre-write gate: decide whether writing a setup token (stdin) into
// gateDir is safe BEFORE any flow persists it. Exit 0 = safe; exit 1 = refused (would create a
// cross-account token-twin). The token is read from stdin only, never argv, and is fingerprinted.
func accountsGateWrite(stdout, stderr io.Writer, gateDir, homeDir string, asJSON bool) int {
	if gateDir == "" {
		fmt.Fprintln(stderr, "usage: fak accounts gate-write --dir <config-dir> < token")
		return 2
	}
	tokBytes, _ := io.ReadAll(os.Stdin)
	verdict := accounts.GateTokenWrite(gateDir, string(tokBytes), homeDir)
	if asJSON {
		stdout.Write(mustJSON(verdict))
		fmt.Fprintln(stdout)
	} else if verdict.Allow {
		fmt.Fprintf(stdout, "ok: safe to write into %s (login: %s)\n", gateDir, verdict.DirAccount)
	} else {
		fmt.Fprintf(stderr, "REFUSED (%s): %s\n", verdict.Reason, verdict.Detail)
	}
	if verdict.Allow {
		return 0
	}
	return 1
}

// accountsCheck is the drift detector: RED (exit 1) if any on-disk view differs from a
// freshly-rendered projection of the registry. The ratchet that keeps the generated views from
// silently diverging from the canonical source. Each drift line carries a (+N/-M) magnitude —
// N projection lines the view is missing, M on-disk lines the generator never emits — and a
// hand-edit warning when M>0 (the on-disk file carries lines `sync` would overwrite). --diff
// prints the offending lines (`- ` removed, `+ ` added).
func accountsCheck(stdout, stderr io.Writer, registryPath, dosView, jobView string, diff bool) int {
	reg, err := accounts.LoadRegistry(registryPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	reg = reg.Refresh()
	fixes := accountFixSummary(registryPath, reg)
	drift := 0
	for _, t := range viewTargets(dosView, jobView) {
		want, err := reg.RenderView(t.view)
		if err != nil {
			fmt.Fprintf(stderr, "fak accounts: %v\n", err)
			return 1
		}
		got, err := os.ReadFile(t.path)
		if err != nil {
			fmt.Fprintf(stdout, "DRIFT %s: cannot read %s%s (%v)\n", t.view, t.path, accountsViewConsumerHint(t.view), err)
			drift++
			continue
		}
		if string(got) != want {
			d := accountsViewDrift(want, string(got))
			fmt.Fprintf(stdout, "DRIFT %s: %s differs from registry projection%s (+%d/-%d) — run `fak accounts sync`\n",
				t.view, t.path, accountsViewConsumerHint(t.view), d.add, d.remove)
			if d.handEdited {
				fmt.Fprintf(stdout, "  %s appears hand-edited: %d line(s) the generator never emits — `sync` will overwrite those edits\n",
					t.path, d.remove)
			}
			if diff {
				for _, l := range d.removed {
					fmt.Fprintf(stdout, "  - %s\n", l)
				}
				for _, l := range d.added {
					fmt.Fprintf(stdout, "  + %s\n", l)
				}
			}
			drift++
			continue
		}
		fmt.Fprintf(stdout, "ok %s: %s matches registry\n", t.view, t.path)
	}
	printAccountFixSummary(stdout, fixes, "account fixes")
	if drift > 0 {
		return 1
	}
	return 0
}

// viewDrift quantifies how an on-disk roster view diverges from its registry projection.
type viewDrift struct {
	add        int      // projection lines missing on disk — `sync` would add them back
	remove     int      // on-disk lines absent from the projection — `sync` would overwrite them
	added      []string // the missing projection lines, in projection order (for --diff `+ `)
	removed    []string // the foreign on-disk lines, in on-disk order (for --diff `- `)
	handEdited bool     // true when remove>0: the file carries lines the generator never emits
}

// accountsViewDrift computes a line-multiset diff between the projection (want) and the on-disk
// view (got). A view that is a strict subset of generator-shaped lines drifts (add>0) but is NOT
// flagged hand-edited; a view carrying lines the projection never emits is (remove>0, hand-edited).
func accountsViewDrift(want, got string) viewDrift {
	wantLines := splitViewLines(want)
	gotLines := splitViewLines(got)
	wantCount := map[string]int{}
	for _, l := range wantLines {
		wantCount[l]++
	}
	gotCount := map[string]int{}
	for _, l := range gotLines {
		gotCount[l]++
	}
	var d viewDrift
	// removed: on-disk lines beyond what the projection accounts for (`sync` overwrites them).
	seen := map[string]int{}
	for _, l := range gotLines {
		seen[l]++
		if seen[l] > wantCount[l] {
			d.removed = append(d.removed, l)
		}
	}
	// added: projection lines the on-disk view is missing (`sync` adds them back).
	seenW := map[string]int{}
	for _, l := range wantLines {
		seenW[l]++
		if seenW[l] > gotCount[l] {
			d.added = append(d.added, l)
		}
	}
	d.remove = len(d.removed)
	d.add = len(d.added)
	d.handEdited = d.remove > 0
	return d
}

// splitViewLines splits a rendered view into lines, ignoring a single trailing newline so a
// generator-shaped file and its subset compare on real content rather than a phantom empty line.
func splitViewLines(s string) []string {
	s = strings.TrimRight(s, "\n")
	if s == "" {
		return nil
	}
	return strings.Split(s, "\n")
}

func accountsViewConsumerHint(view accounts.ViewName) string {
	if view == accounts.ViewJob {
		return " (this is what `u` reads)"
	}
	return ""
}
