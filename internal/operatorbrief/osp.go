package operatorbrief

// osp.go folds the operator-steerability overlay (internal/steerpr) into the
// operator brief as one more optional source, exactly like heaviness and fleet.
// The overlay bundles continuous-merge trunk commits into PR-sized units and
// bands each unit by where operator attention is owed — RESIDUAL (a member
// claimed something the machine could not confirm), UNVERIFIABLE (no checkable
// claim), or CLEARED (every member witnessed). Without this fold an operator would
// run `fak steer prs` as a second pane and reconcile it against the brief by
// hand; the brief already knows how to timebox and order, so the overlay should
// feed it, not compete with it.
//
// The bucket mapping honours the HUMAN_RESIDUAL doctrine the overlay is built on.
// A RESIDUAL band means "an oracle could not confirm", which is NOT automatically
// "a human must decide now" — so a RESIDUAL unit reaches the human bucket (the
// pager) ONLY when choicetriage judges it a genuine authority decision. Every
// other RESIDUAL unit is a watch, UNVERIFIABLE units are watch, and CLEARED units
// are background. The overlay therefore cannot spam the pager: only a unit that
// names authority a person holds ever pages.
//
// Because OSP items land in the standard buckets, the brief's existing attention
// weighting counts them for free — a large residual pile lengthens the human /
// watch timebox honestly through boundedAttentionMinutes rather than silently.
//
// An unreadable or absent overlay is marked UNMEASURED, never folded as a clean
// zero: "not measured" and "measured, nothing owed" are different operator facts,
// and collapsing the first into the second would let a broken overlay read as a
// clean bill of health.

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
	"github.com/anthony-chaudhary/fak/internal/steerpr"
	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

// OSP is the operator-steerability overlay folded as one optional brief source.
// It carries the banded units plus an Unreadable marker so the fold can tell
// "measured: nothing owes attention" from "the payload could not be read" — the
// latter must never render as a clean zero. A nil *OSP on Inputs is an absent
// source (the overlay was not collected); a non-nil OSP with Unreadable set is a
// present-but-unreadable payload.
type OSP struct {
	Schema     string         // overlay payload schema, when known
	Units      []steerpr.Unit // banded overlay units
	Unreadable bool           // payload was provided but could not be parsed
	Note       string         // optional human note behind an unreadable read
}

// ospState records the overlay's presence and derived verdict as a source stamp.
// Like heaviness/fleet it carries no date/commit, so it never perturbs snapshot
// coherence. An unreadable payload reads "unmeasured"; a readable one reads "ok"
// (measured), regardless of how many units owe attention — the residual pile
// shapes the buckets, not the source's measured/unmeasured status.
func ospState(o *OSP) SourceState {
	if o == nil {
		return SourceState{Name: "osp", Status: "missing"}
	}
	if o.Unreadable {
		return SourceState{Name: "osp", Schema: o.Schema, Status: "unmeasured", Verdict: "UNMEASURED", Finding: "osp_unmeasured"}
	}
	finding := "osp_measured"
	if residual := steerpr.Residual(o.Units); residual > 0 {
		finding = fmt.Sprintf("osp_residual_%d", residual)
	}
	return SourceState{
		Name:    "osp",
		Schema:  strmatch.FirstNonBlank(o.Schema, steerpr.Schema),
		Status:  "ok",
		Verdict: "OK",
		Finding: finding,
	}
}

// addOSP folds the overlay's units into the brief's buckets. An unreadable
// overlay is surfaced as a single watch note (visible, never a clean zero and
// never a human page). A readable overlay partitions its units by band:
//
//   - RESIDUAL -> human ONLY when choicetriage judges the unit a genuine
//     authority decision; otherwise watch. This is the second filter the epic's
//     HUMAN_RESIDUAL doctrine requires — a RESIDUAL band alone is "an oracle
//     could not confirm", not "a person must decide now", so it must clear the
//     authority test before it may page.
//   - UNVERIFIABLE -> watch (reviewable; it made no checkable claim).
//   - CLEARED      -> background (every member witnessed; buys no attention).
func addOSP(r *Report, o OSP) {
	if o.Unreadable {
		r.addWatch("osp", "overlay unmeasured",
			strmatch.FirstNonBlank(o.Note, "the operator-steerability overlay payload could not be read"),
			"restore `fak steer prs --json` before trusting the brief's attention pile")
		return
	}
	for _, u := range o.Units {
		band := u.Band
		if band == "" {
			band = steerpr.FoldBand(u.Commits)
		}
		switch band {
		case steerpr.BandResidual:
			title, detail, action := ospTitle(u), ospDetail(u), ospAction(u)
			// The second filter: only a genuine authority decision may page.
			v := choicetriage.Triage(choicetriage.Signal{
				Severity: "decision",
				Source:   "osp",
				Question: title,
				Detail:   detail,
				Action:   action,
			})
			if v.NeedsHuman {
				r.addHuman("osp", "decision", title, detail, strmatch.FirstNonBlank(v.Resolve, action))
			} else {
				r.addWatch("osp", title, detail, strmatch.FirstNonBlank(v.Resolve, "review the unwitnessed unit; it owes attention but not a human decision"))
			}
		case steerpr.BandUnverifiable:
			r.addWatch("osp", ospTitle(u)+" (unverifiable)", ospDetail(u),
				"review the unit; no member made a checkable claim to witness")
		case steerpr.BandCleared:
			r.addBackground("osp", ospTitle(u)+" (cleared)", ospDetail(u),
				"every member was witnessed; no operator attention is owed here")
		default:
			// An unrecognised band is "not yet graded": a watch, never a clean zero.
			r.addWatch("osp", ospTitle(u)+" (ungraded)", ospDetail(u),
				"grade the unit's members before trusting the overlay's band")
		}
	}
}

// ospTitle is the human-legible name of an overlay unit: its rendered title, or a
// leaf-derived fallback when the fold produced none.
func ospTitle(u steerpr.Unit) string {
	if title := strings.TrimSpace(u.Title); title != "" {
		return title
	}
	return strmatch.FirstNonBlank(u.Leaf, "unstamped") + " overlay unit"
}

// ospDetail summarizes the unit's size and closure refs so an operator sees what
// the unit is without re-running the overlay.
func ospDetail(u steerpr.Unit) string {
	parts := []string{fmt.Sprintf("leaf %s: %d commit(s)", strmatch.DashIfBlank(u.Leaf), len(u.Commits))}
	if len(u.Resolves) > 0 {
		parts = append(parts, "closes "+strings.Join(u.Resolves, ", "))
	}
	if len(u.Files) > 0 {
		parts = append(parts, fmt.Sprintf("%d file(s)", len(u.Files)))
	}
	return strings.Join(parts, "; ")
}

// ospAction phrases the operator move for a RESIDUAL unit. It deliberately names
// no authority token of its own, so the authority test in choicetriage keys on
// the unit's real content (leaf/title/resolves), not on this generic prompt.
func ospAction(u steerpr.Unit) string {
	if len(u.Resolves) > 0 {
		return "decide whether the unwitnessed claims behind " + strings.Join(u.Resolves, ", ") + " may stand, or hold the unit for a witness"
	}
	return "decide whether the unit's unwitnessed claim may stand, or hold it for a witness"
}
