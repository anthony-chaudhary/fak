package hooks

import (
	"sort"
	"strconv"
	"strings"
)

// repairorder.go — the DISPOSITION and ORDER of the multi-item fix list a gate run hands back
// (#5972).
//
// PreCommitGates() is registered in the order tools/githooks/pre-commit invoked its Python
// oracles, and that order is load-bearing for parity — it is deliberately NOT an impact
// ranking. A finding's disposition (binding vs advisory) comes from the gate's resolved MODE
// for THIS run, not from where the gate sits in the registry, so the two can disagree the
// moment an operator softens an early gate or hardens a late one: soften PUBLIC_LEAK to warn
// and set FLEET_DUP_GUARD=block, and registry order leads the hand-off with the item that does
// not block. An agent applying a truncated or skimmed fix list then spends its budget on
// cosmetics while the refusing finding survives.
//
// These helpers keep marking and ordering in ONE place so every consumer of a gate run — the
// --json payload and the model-facing stderr block alike — sees the same list in the same
// order. They only ORDER and MARK: no verdict, no exit code, and no gate's own decision is
// re-decided here. Ordering is STABLE, so within a disposition the registry/file/line order the
// gates produced is preserved exactly and two identical runs emit identical lists.
//
// internal/codelint's Summary (errors first, one line per finding, fed straight back to a
// coding agent) is the in-repo template; this is the same technique on the commit-boundary path.

// BindingMode is the one gate mode that can refuse a commit. Every other resolved mode ("warn",
// and anything unrecognized) is visible-but-non-blocking for that run, which is why the test is
// written as an equality against the blocking mode rather than as a not-equals against "warn" —
// an unknown mode must degrade to advisory, never silently to binding.
const BindingMode = "block"

// ModeIsBinding reports whether findings produced under this resolved gate mode can refuse the
// commit. Surrounding whitespace is tolerated because the mode arrives from an env var.
func ModeIsBinding(mode string) bool { return strings.TrimSpace(mode) == BindingMode }

// MarkDisposition returns a COPY of findings with Advisory set from the gate mode this run
// resolved, so each item in the hand-off says whether fixing it is required or optional. It
// copies rather than mutating in place because a gate's Check owns the slice it returned and
// may reuse it; the caller appends the copy to the run-wide list.
//
// A gate that already graded a finding is not overridden downward or upward — the MODE is the
// authority on whether this run can refuse, and Severity (the one gate-specific 0-100 magnitude)
// is untouched here.
func MarkDisposition(findings []Finding, mode string) []Finding {
	if len(findings) == 0 {
		return nil
	}
	advisory := !ModeIsBinding(mode)
	out := make([]Finding, len(findings))
	copy(out, findings)
	for i := range out {
		out[i].Advisory = advisory
	}
	return out
}

// OrderForRepair sorts findings IN PLACE so every binding item precedes every advisory one,
// without disturbing the order inside either disposition. The sort is stable and the comparison
// is a strict weak ordering on a single boolean, so the result is a partition: the relative
// order the gate registry produced survives within each half, and the ordering is deterministic
// across runs with no dependence on map iteration or timing.
func OrderForRepair(findings []Finding) {
	sort.SliceStable(findings, func(i, j int) bool {
		return !findings[i].Advisory && findings[j].Advisory
	})
}

// RepairSummary renders findings as a compact, model-facing fix list: binding work first, one
// line per finding, each line stating its disposition explicitly so a truncated read still knows
// which items are required. Returns "" for an empty list so a clean run prints nothing.
//
// It ORDERS a copy — the caller's slice is left alone — so calling it does not depend on
// OrderForRepair having run first, and calling both is idempotent.
func RepairSummary(findings []Finding) string {
	if len(findings) == 0 {
		return ""
	}
	ordered := make([]Finding, len(findings))
	copy(ordered, findings)
	OrderForRepair(ordered)

	var b strings.Builder
	b.WriteString("pre-commit fix list (" + strconv.Itoa(countBinding(ordered)) + " binding of " +
		strconv.Itoa(len(ordered)) + "), binding work first:\n")
	for _, f := range ordered {
		b.WriteString("  ")
		if f.Advisory {
			b.WriteString("[advisory] ")
		} else {
			b.WriteString("[binding]  ")
		}
		b.WriteString(f.Gate)
		if f.File != "" {
			b.WriteString(" " + f.File)
			if f.Line > 0 {
				b.WriteString(":" + strconv.Itoa(f.Line))
			}
		}
		if f.Detail != "" {
			b.WriteString(" — " + f.Detail)
		}
		b.WriteByte('\n')
	}
	return strings.TrimRight(b.String(), "\n")
}

// countBinding is the numerator of the summary header: how many of the listed items can refuse
// this commit. It is the one number that tells an agent under context pressure where the
// required work stops.
func countBinding(findings []Finding) int {
	n := 0
	for _, f := range findings {
		if !f.Advisory {
			n++
		}
	}
	return n
}
