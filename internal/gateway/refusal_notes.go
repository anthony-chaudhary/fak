package gateway

import (
	"fmt"
	"strconv"
	"strings"
)

// refusalNote is one "actionable half of a refusal" renderer: given a refused
// ToolAdjudication it emits the in-band remedy string for that refusal facet
// (preview-confirm recipe, sanctioned alternative, livelock nudge, ...), or ""
// when the facet does not apply to this adjudication.
//
// The registry below is the single seam every content-channel wire folds over
// (#2750). A NEW refusal kind that carries actionable Meta is surfaced by
// appending its renderer here - NOT by hand-stitching another call at each
// denySummary / adjudicationNote site, which is exactly how reversibilityGateNote,
// remedyNote, and livelockInBandNote each grew a bespoke retrofit.
type refusalNote struct {
	render func(ToolAdjudication) string
	// confirmRecipe marks a non-empty render as a preview-confirm PAUSE rather
	// than a denial, so the trailer that sanctions re-proposing the same call
	// (only reversibilityGateNote today) can be scoped correctly.
	confirmRecipe bool
}

// refusalNotes is the ordered registry of actionable-half renderers. Order is
// load-bearing: it fixes the in-band rendering order (reversibility recipe, then
// the sanctioned alternative, then the livelock nudge) that the wire tests pin.
var refusalNotes = []refusalNote{
	{render: reversibilityGateNote, confirmRecipe: true},
	{render: remedyNote},
	{render: livelockInBandNote},
}

// renderRefusalNotes folds every registered renderer over one refused
// adjudication and returns the space-joined actionable notes plus whether any of
// them was a preview-confirm recipe. Both content-channel call sites (denySummary
// and adjudicationNote) consume this instead of stitching each note by hand.
func renderRefusalNotes(a ToolAdjudication) (notes string, confirmRecipe bool) {
	parts := make([]string, 0, len(refusalNotes))
	for _, n := range refusalNotes {
		if s := n.render(a); s != "" {
			parts = append(parts, s)
			if n.confirmRecipe {
				confirmRecipe = true
			}
		}
	}
	return strings.Join(parts, " "), confirmRecipe
}

// denySummary renders a short human-readable note when every proposed tool_call
// was refused, so a client that ignores the `fak` extension still adapts.
func denySummary(adjs []ToolAdjudication) string {
	parts := make([]string, 0, len(adjs))
	for _, a := range adjs {
		part := fmt.Sprintf("%s: %s (%s/%s)", a.Tool, a.Verdict.Kind, reasonOrKind(a.Verdict), a.Verdict.Disposition)
		if notes, _ := renderRefusalNotes(a); notes != "" {
			part += " " + notes
		}
		parts = append(parts, part)
	}
	return "All proposed tool calls were refused by the fak kernel: " + strings.Join(parts, "; ")
}

// adjudicationNote renders a short, agent-readable summary of the kernel's
// non-trivial decisions (drops + repairs) on a turn, for clients that read only
// the in-band content channel and never the `fak` extension — Claude Code on the
// Anthropic wire is exactly that client. It is the difference between a denied
// call SILENTLY VANISHING (the agent re-proposes it, or proceeds on a false
// premise) and the agent being told "fak refused rm -rf for POLICY_BLOCK" so it
// can adapt. Returns "" when every call was a clean ALLOW (nothing worth saying).
func adjudicationNote(adjs []ToolAdjudication) string {
	denied := make([]string, 0, len(adjs))
	repaired := make([]string, 0, len(adjs))
	allowedLoops := make([]string, 0, len(adjs))
	hasConfirmRecipe := false
	for _, a := range adjs {
		switch {
		case !a.Admitted:
			entry := fmt.Sprintf("%s (%s/%s)", a.Tool, reasonOrKind(a.Verdict), a.Verdict.Disposition)
			if notes, recipe := renderRefusalNotes(a); notes != "" {
				entry += " " + notes
				if recipe {
					hasConfirmRecipe = true
				}
			}
			denied = append(denied, entry)
		case a.Admitted && a.Livelock != nil:
			allowedLoops = append(allowedLoops, livelockInBandNote(a))
		case a.Verdict.Kind == "TRANSFORM":
			repaired = append(repaired, a.Tool)
		}
	}
	if len(denied) == 0 && len(repaired) == 0 && len(allowedLoops) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("[fak] ")
	if len(denied) > 0 {
		b.WriteString("refused ")
		b.WriteString(strconv.Itoa(len(denied)))
		b.WriteString(" tool call(s): ")
		joined := strings.Join(denied, "; ")
		b.WriteString(joined)
		if !strings.HasSuffix(joined, ".") {
			b.WriteString(".")
		}
		// The blanket "do not re-propose" trailer contradicts the preview-confirm
		// recipe, whose sanctioned recovery IS re-proposing the same call (plus the
		// confirm key). Witnessed wedging a fleet session for 1h+: the agent obeyed
		// the trailer, never echoed the token, and the push never happened. When any
		// denied call carries a confirm recipe, the trailer must except it.
		if hasConfirmRecipe {
			b.WriteString(" A preview-confirm refusal is a pause, not a denial: the sanctioned next step is to re-propose that same call with only the _fak_confirm key added. This is per-tool feedback, not a session stop. Do not re-propose any other refused call unchanged; choose an allowed alternative. A session stop only comes from a declared stop policy.")
		} else {
			b.WriteString(" This is per-tool feedback, not a session stop. Do not re-propose a refused call unchanged; fix the arguments/tool choice or choose an allowed alternative. A session stop only comes from a declared stop policy.")
		}
	}
	if len(repaired) > 0 {
		if len(denied) > 0 {
			b.WriteString(" ")
		}
		b.WriteString("repaired arguments for: ")
		b.WriteString(strings.Join(repaired, ", "))
		b.WriteString(".")
	}
	if len(allowedLoops) > 0 {
		if len(denied) > 0 || len(repaired) > 0 {
			b.WriteString(" ")
		}
		b.WriteString("observed repeated admitted tool call(s): ")
		b.WriteString(strings.Join(allowedLoops, "; "))
		b.WriteString(". This is advisory: do not repeat a successful identical call unchanged unless you can name the new evidence it will produce.")
	}
	return b.String()
}

func prependAdjudicationContentNote(content string, adjs []ToolAdjudication) string {
	note := adjudicationNote(adjs)
	if note == "" {
		return content
	}
	if strings.TrimSpace(content) == "" {
		return note
	}
	return note + "\n" + content
}

func livelockInBandNote(a ToolAdjudication) string {
	if a.Livelock == nil {
		return ""
	}
	note := fmt.Sprintf("LIVELOCK_DETECTED repeat=%d repeated_call=%s approach=%s",
		a.Livelock.RepeatCount,
		livelockCallLabel(*a.Livelock),
		a.Livelock.SuggestedChange)
	if a.Livelock.Escalate {
		// The retryable fuse itself was ignored turn after turn. This refusal is now
		// TERMINAL: the turn escalates to a deny-all and the session's bounded give-up
		// policy will end it. Tell the model the loop is over so it stops and reports
		// the blocker instead of burning more tokens re-proposing the same call.
		note += " ABORT=terminal (this identical call has been refused too many times; it will NOT be admitted — stop retrying, end the turn, and report the blocker with a witness)"
	} else if a.Livelock.Fuse {
		// The advisory note repeated for several turns and the loop kept going, so the
		// fuse converted this call into a refusal. Say so plainly: the model must change
		// approach, not re-propose — re-proposing the identical call fuses again.
		note += " fuse=armed (this repeated call was refused; changing approach is required, not optional)"
	}
	return note
}
