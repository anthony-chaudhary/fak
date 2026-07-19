package main

// The roster shell for `fak superloop roster` (issue #4955):
// the ONE canonical loop-fleet roster, the source of truth for what the operator
// supervises. The shell reads the three sources — the cross-ledger loop-health fold
// (loopfleet.Fold), the loopmgr job registry, and the registered super loops — and
// hands them to the pure builder (superloop.BuildRoster), which dedupes on each
// loop's stable identity so everything is counted exactly once. Read-only: it
// mutates nothing; a missing/unreadable source surfaces as a KNOWN gap, never a
// silent drop. The worst-first meta-walk over this roster is #4958.

import (
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"sort"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopfleet"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/superloop"
)

func runRoster(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("superloop roster", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the roster as JSON")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	if !parseFlags(fs, argv) {
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}

	roster := collectRoster(root)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, roster, "fak superloop roster")
	}
	printRoster(stdout, roster)
	return 0
}

// collectRoster reads the three sources and folds them through the pure
// builder. Source reads are skip-and-surface: a loopfleet ledger that cannot be
// folded arrives as a Skipped row; a loopmgr registry that cannot be loaded is
// surfaced as its own gap (a missing registry file is fine — it loads empty).
func collectRoster(root string) superloop.Roster {
	fleet := loopfleet.Fold(root, time.Now(), loopmgr.HealthThresholds{})
	folded := make([]superloop.RosterLoop, 0, len(fleet.Loops))
	for _, l := range fleet.Loops {
		folded = append(folded, superloop.RosterLoop{Kind: l.Kind, State: string(l.State), Dark: l.Dark})
	}
	gaps := make([]superloop.RosterGap, 0, len(fleet.Skipped))
	for _, sk := range fleet.Skipped {
		gaps = append(gaps, superloop.RosterGap{Ledger: sk.Ledger, Path: sk.Path, Reason: sk.Reason})
	}

	regRel := defaultLoopRegistry()
	regPath := regRel
	if !filepath.IsAbs(regPath) {
		regPath = filepath.Join(root, regRel)
	}
	var registryIDs []string
	if reg, err := loopmgr.LoadRegistry(regPath); err != nil {
		gaps = append(gaps, superloop.RosterGap{Ledger: "loop-registry", Path: regRel, Reason: err.Error()})
	} else {
		for id := range reg.Jobs {
			registryIDs = append(registryIDs, id)
		}
		sort.Strings(registryIDs)
	}

	return superloop.BuildRoster(folded, registryIDs, superloop.Registry(), gaps)
}

func printRoster(w io.Writer, r superloop.Roster) {
	fmt.Fprintf(w, "superloop roster — the canonical loop-fleet roster (what the operator supervises)\n")
	fmt.Fprintf(w, "  %d entries: %d loop(s) + %d super loop(s); %d measured, %d unmeasured, %d unnamed loop(s)\n\n",
		r.Total, r.Loops, r.Supers, r.Measured, r.Unmeasured, r.Unnamed)

	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "  ID\tKIND\tSTATE\tNAMED\tSOURCES\tDETAIL")
	for _, e := range r.Entries {
		state := e.State
		if !e.Measured {
			state = "UNMEASURED"
		} else if state == "" {
			state = "-"
		}
		if e.Dark {
			state = "DARK"
		}
		named := "-"
		if e.Named {
			named = "named"
		}
		fmt.Fprintf(tw, "  %s\t%s\t%s\t%s\t%s\t%s\n", e.ID, e.Kind, state, named, joinRosterSources(e.Sources), e.Detail)
	}
	_ = tw.Flush()

	if len(r.Gaps) > 0 {
		fmt.Fprintf(w, "\n  known gaps — ledgers that could not be folded (surfaced, never a healthy zero):\n")
		for _, g := range r.Gaps {
			fmt.Fprintf(w, "    %-14s %s (%s)\n", g.Ledger, g.Reason, g.Path)
		}
	}
	fmt.Fprintf(w, "\n  every loop above is counted exactly once; the worst-first meta-walk over this roster is #4958\n")
}

func joinRosterSources(sources []string) string {
	out := ""
	for i, s := range sources {
		if i > 0 {
			out += "+"
		}
		out += s
	}
	if out == "" {
		return "-"
	}
	return out
}
