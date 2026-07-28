// guard_policy_diff.go implements `fak guard policy diff` (#5173, epic #5170,
// Track A sibling A3): the widen-drift report. Where `policy explain` (#5172)
// lists the effective floor by amendment class, `diff` answers the sharper
// operator question — "how does the floor I am RUNNING differ from the floor
// the binary SHIPPED, and did any of that drift LOOSEN the guard?" It diffs the
// shipped compiled-in floor against the effective floor (shipped + operator
// allow/reload widenings) through the canonical policy.DiffAmendment engine, so
// every reported change is classified by the same registry that governs
// admission: a WIDENED bucket (drift that loosened the floor — the gate-able
// signal), a TIGHTENED bucket (operator hardening, always welcome), and a
// FROZEN bucket that can only be non-empty if a compiled-in floor element was
// somehow moved (a bug — reported loudly, fail-closed).
package main

import (
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// policyDiffReport renders the amendment-classified drift from the shipped
// `floor` to the `effective` running floor. It returns the human-readable lines
// (without the header) and whether any WIDENING or FROZEN-violation drift is
// present — the condition an operator or CI gate acts on. Pure over its inputs
// so it is unit-testable without touching disk or env.
func policyDiffReport(floor, effective adjudicator.Policy) (lines []string, gateable bool) {
	d := policy.DiffAmendment(floor, effective)
	if d.Empty() {
		return []string{"  none — the effective floor matches the shipped floor"}, false
	}
	if len(d.Frozen) > 0 {
		lines = append(lines, "  FROZEN-VIOLATION  "+policy.FormatAmendmentChanges(d.Frozen))
	}
	if len(d.Widen) > 0 {
		lines = append(lines, "  WIDENED (drift)   "+policy.FormatAmendmentChanges(d.Widen))
	}
	if len(d.Tighten) > 0 {
		lines = append(lines, "  TIGHTENED (hard)  "+policy.FormatAmendmentChanges(d.Tighten))
	}
	// Widening drift, or any frozen-floor movement, is the gate-able condition.
	// A purely-tightened floor is drift too, but it never loosens the guard, so
	// it does not trip the exit-1 signal.
	gateable = len(d.Widen) > 0 || len(d.Frozen) > 0
	return lines, gateable
}

// runGuardPolicyDiff loads the shipped and effective floors and prints the drift
// report. Exit 0 = no widen-drift (a purely-tightened or identical floor), exit
// 1 = widening drift or a frozen violation is present (a lint signal an operator
// can gate CI on), exit 2 = the floor could not be read.
func runGuardPolicyDiff(stdout, stderr io.Writer, policyPath string) int {
	floorManifest, err := policy.ParseManifest(guardDefaultPolicyJSON)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard policy diff: embedded floor: %v\n", err)
		return 2
	}
	floor, err := floorManifest.ToPolicy()
	if err != nil {
		fmt.Fprintf(stderr, "fak guard policy diff: embedded floor: %v\n", err)
		return 2
	}
	// The EFFECTIVE floor is whatever a launch with these same arguments would
	// enforce: the named --policy manifest (or the embedded floor), unioned with the
	// launch-time operator allow/deny overlays and the config self-protection rules.
	// Reusing the launch loader rather than re-deriving the floor here is the point —
	// a drift report assembled from its own copy of the layering rules could disagree
	// with the guard it claims to describe.
	rt, _, _, _ := loadGuardCapabilityFloor(policyPath)

	lines, gateable := policyDiffReport(floor, rt.Adjudicator)
	fmt.Fprintln(stdout, "fak guard policy diff — widen-drift from the shipped floor")
	for _, ln := range lines {
		fmt.Fprintln(stdout, ln)
	}
	if gateable {
		return 1
	}
	return 0
}
