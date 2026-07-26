package main

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/policy"
)

// formatAmendmentPosture folds the policy amendment-class registry (#5171, epic
// #5170) into the guard exit summary (#5184): one glance tells the operator who
// could have moved which policy surface this session. The counts come straight
// from policy.KnobsByClass over the compiled-in PolicyKnobRegistry — never from
// session state — so the line is a property of the binary, not a self-report.
// The load-bearing row is the last one: the SELF_AMENDABLE (agent-writable)
// frontier is empty, i.e. the agent could not widen anything on its own this
// session. Pure string builder in the shared summary grammar
// (guard_format_layout.go); color is layered on at print time like every other
// section.
func formatAmendmentPosture() string {
	frozen := len(policy.KnobsByClass(policy.AmendFrozen))
	ratchet := len(policy.KnobsByClass(policy.AmendRatchet))
	gated := len(policy.KnobsByClass(policy.AmendGatedWiden))
	selfAmendable := policy.KnobsByClass(policy.AmendSelfAmendable)

	var b strings.Builder
	b.WriteString(guardSection("amendment posture"))
	b.WriteString(guardRow("policy knobs",
		fmt.Sprintf("%d FROZEN, %d RATCHET (tighten-only), %d GATED_WIDEN (operator-gated)",
			frozen, ratchet, gated)))
	if len(selfAmendable) == 0 {
		b.WriteString(guardRow("self-amendable", "0 — empty frontier"))
		b.WriteString(guardNote("no knob is agent-writable: the agent could not widen anything on its own this session"))
	} else {
		// Declared-but-unexpected: the frontier exists so it is enumerated, not
		// implied by omission — if a knob ever lands here, name it loudly.
		names := make([]string, 0, len(selfAmendable))
		for _, k := range selfAmendable {
			names = append(names, k.Field)
		}
		b.WriteString(guardRow("⚠ self-amendable",
			fmt.Sprintf("%d — agent-writable frontier is NOT empty: %s",
				len(selfAmendable), strings.Join(names, ", "))))
	}
	return b.String()
}
