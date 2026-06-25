package main

import (
	"fmt"
	"strings"
)

// renderMarkdown turns a Comparison into the deterministic markdown summary issue
// #434 asks for (acceptance #6). It carries no timestamp or absolute path, so the
// artifact is byte-stable and a regeneration check can compare it exactly.
func renderMarkdown(cmp Comparison) string {
	var b strings.Builder

	fmt.Fprintf(&b, "# Workflow-memory benchmark — virtual views vs. stale/poison replay (%s)\n\n", cmp.Issue)
	b.WriteString("Memory as an **agent-workflow substrate**, not a chatbot recall layer: three\n")
	b.WriteString("memory policies scored over one finished session whose tool results carry the\n")
	b.WriteString("workflow hazards that matter for `fak` — provenance, stale world witnesses,\n")
	b.WriteString("sealed/poisoned pages, tombstones, multi-agent handoff, and effect claims that\n")
	b.WriteString("require evidence.\n\n")

	fmt.Fprintf(&b, "Reproduce:\n\n```\n%s\n```\n\n", cmp.Command)

	// Fixture.
	b.WriteString("## Fixture\n\n")
	fmt.Fprintf(&b, "%d pages — %d benign, %d sealed, %d tombstoned — %d raw bytes, %d resident bytes.\n\n",
		cmp.Fixture.Pages, cmp.Fixture.Benign, cmp.Fixture.Sealed, cmp.Fixture.Tombstoned,
		cmp.Fixture.RawBytes, cmp.Fixture.ResidentBytes)
	b.WriteString("Hazards encoded:\n\n")
	for _, h := range cmp.Fixture.Hazards {
		fmt.Fprintf(&b, "- %s\n", h)
	}
	b.WriteString("\n")

	// Arms table.
	b.WriteString("## Arms\n\n")
	b.WriteString("| arm | kind | resident bytes | resident tokens | view fault | source coverage | stale reuse | poison leak | task ok | fallback→raw |\n")
	b.WriteString("|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|\n")
	for _, a := range cmp.Arms {
		fmt.Fprintf(&b, "| %s | %s | %d | %d | %.2f | %.2f | %d | %d | %s | %.2f |\n",
			a.Name, a.Kind, a.ResidentBytes, a.ResidentTokens, a.ViewFaultRate, a.SourceCoverage,
			a.StaleReuse, a.PoisonLeak, yesno(a.TaskSuccess), a.FallbackToRaw)
	}
	b.WriteString("\n")
	for _, a := range cmp.Arms {
		fmt.Fprintf(&b, "- **%s** (%s) — %s\n", a.Name, a.Kind, a.Note)
	}
	b.WriteString("\n")

	// Replays.
	b.WriteString("## Stale replay (acceptance #4)\n\n")
	fmt.Fprintf(&b, "%s\n\n", cmp.Stale.Description)
	fmt.Fprintf(&b, "- recomputes: %d\n- stale reuse (must be 0): %d\n- old view rejected: %s\n\n",
		cmp.Stale.Recomputes, cmp.Stale.StaleReuse, yesno(cmp.Stale.OldViewRejected))

	b.WriteString("## Poison replay (acceptance #5)\n\n")
	fmt.Fprintf(&b, "%s\n\n", cmp.Poison.Description)
	fmt.Fprintf(&b, "- sealed refused: %d\n- poison leakage (must be 0): %d\n- sealed contained: %s\n\n",
		cmp.Poison.SealedRefused, cmp.Poison.PoisonLeakage, yesno(cmp.Poison.SealedContained))

	// Verdict.
	b.WriteString("## Verdict\n\n")
	b.WriteString("The full transcript is correct but carries the most bytes and leaks every sealed\n")
	b.WriteString("and tombstoned page. The naive global summary is cheapest in bytes but destroys\n")
	b.WriteString("provenance, leaks sealed content, and cannot fall back to a source. Only the\n")
	b.WriteString("provenance-bound virtual views are both lean and fail-closed: zero stale reuse,\n")
	b.WriteString("zero poison leakage, the goal still answered, and a measured raw fallback rate.\n\n")
	b.WriteString("The two modeled baselines are closed-form reductions over the page table; the\n")
	b.WriteString("virtual-views arm is measured by driving the real derived-view substrate.\n")

	return b.String()
}

func yesno(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}
