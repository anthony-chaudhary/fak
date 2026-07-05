package blockerpost

import (
	"fmt"
	"sort"
	"strings"
)

// This file is the W7 operator surface of the blast-radius containment epic (#2712):
// the fold from the fleet's LIVE known-bad signatures into ONE blast-framed Blocker —
// "1 root cause -> N affected, 1 fixing, N-1 parked, witness pending". It is the twin
// of feed.go's FoldIssues: a pure fold from a producer's state (there the blocked
// backlog, here the known-bad ledger + blast estimate) into the single Blocker shape
// render.go turns into a Slack card, reusing the same {status, operator, clear}
// severity tiers so no new transport is needed.
//
// The load-bearing thing this fold owns is turning N coincidences into ONE event: a
// shared bug taxing the fleet reads today as N unrelated stuck workers; this collapses
// it to a single legible line whose severity says whether a human is needed. Recognition
// (W2), estimation (W3), holding (W4), election (W5) all run upstream; this only RENDERS
// the state they produced — it never decides a hold or elects a fixer.

// Signature is one LIVE known-bad signature projected into the shape the blast card
// reasons over. The shell (cmd/fak/knownbad.go) builds one per record returned by
// knownbad.LiveRecords, joining the affected count from the W3 blast estimate (or the
// signature's own declared tree when no estimate is run). The fold itself stays pure —
// no ledger, no estimate code, no clock — so it is unit-testable, exactly like Issue.
type Signature struct {
	// ID is the root-cause signature id (e.g. "sha256:abc…"); Reason its failure class.
	ID     string
	Reason string
	// Trees are the signature's declared repo-relative tree globs (the broken tree).
	Trees []string
	// Affected is the count of in-flight leases + queued issues inside the blast radius
	// (from the W3 estimate). Parked is Affected minus the one elected fixer when a fixer
	// is claimed — the "N-1 parked" the card frames.
	Affected int
	// Fixer names the single elected fixer (knownbad Record.ClaimedBy), empty when the
	// signature is UNCLAIMED — the case that escalates to an operator page once overdue.
	Fixer string
	// WitnessPending is true while the signature is still open (no witnessed resolve yet)
	// — the card's "witness: pending". A resolved signature is not live, so it never
	// reaches this fold; this stays true for every live row and reads as the honest
	// "the fix is not yet proven" state.
	WitnessPending bool
	// NoFixerOverdue is set by the shell when an UNCLAIMED signature has gone longer than
	// the operator threshold without a fixer — the trigger that flips the card from a
	// muted status line to a surfaced operator page. An unclaimed-but-not-yet-overdue
	// signature stays status (recorded, no page) so a just-discovered bug does not page
	// before the fleet has had a tick to elect a fixer.
	NoFixerOverdue bool
}

// claimed reports whether a signature has an elected fixer.
func (s Signature) claimed() bool { return strings.TrimSpace(s.Fixer) != "" }

// parked is the affected count minus the one elected fixer (never negative). With no
// fixer every affected agent is parked-or-colliding; with a fixer, one is fixing and the
// rest are parked behind them.
func (s Signature) parked() int {
	if s.claimed() && s.Affected > 0 {
		return s.Affected - 1
	}
	return s.Affected
}

// maxBlastLines caps how many signatures a single roll-up lists so a fleet-wide storm of
// known-bad signatures does not flood the channel; the overflow is summarized, worst
// (unclaimed / overdue) first.
const maxBlastLines = 12

// FoldBlast folds the fleet's LIVE known-bad signatures into ONE blast-framed Blocker.
// Severity is the honest mapping of "muted while a fixer is progressing, surfaced when a
// human is needed":
//
//	0 live signatures                       -> SeverityClear    (all-clear, no page)
//	>=1, at least one unclaimed AND overdue  -> SeverityOperator (no fixer — a human must
//	                                            elect/step in — paged)
//	>=1, otherwise (every one has a fixer,   -> SeverityStatus   (contained, in progress —
//	    or the unclaimed ones aren't overdue)   recorded, no page)
//
// repoURL, when set, links the operator card's "do this next" to the known-bad ledger
// note in the repo (the affordance an operator follows to see the raw rows). Unclaimed /
// overdue signatures are listed first (worst-first), each row carrying the root cause,
// the affected/parked frame, the fixer (or *NO FIXER*), and the witness state.
func FoldBlast(sigs []Signature, repoURL string) Blocker {
	if len(sigs) == 0 {
		return Blocker{
			Severity: SeverityClear,
			Title:    "no shared blockers",
			Detail:   "0 live known-bad signatures — no shared bug is taxing the fleet.",
			Ref:      "known-bad",
		}
	}

	// Stable worst-first order: overdue-unclaimed, then any-unclaimed, then by most
	// affected, then by signature id so the render is deterministic.
	ordered := append([]Signature(nil), sigs...)
	sort.SliceStable(ordered, func(a, c int) bool {
		la, lc := ordered[a], ordered[c]
		if la.NoFixerOverdue != lc.NoFixerOverdue {
			return la.NoFixerOverdue // overdue first
		}
		if la.claimed() != lc.claimed() {
			return !la.claimed() // unclaimed before claimed
		}
		if la.Affected != lc.Affected {
			return la.Affected > lc.Affected // widest blast first
		}
		return la.ID < lc.ID
	})

	var unclaimed, overdue, totalAffected, totalParked int
	for _, s := range ordered {
		if !s.claimed() {
			unclaimed++
		}
		if s.NoFixerOverdue {
			overdue++
		}
		totalAffected += s.Affected
		totalParked += s.parked()
	}

	b := Blocker{Ref: fmt.Sprintf("known-bad · %d live", len(ordered))}
	if overdue > 0 {
		b.Severity = SeverityOperator
		b.Title = fmt.Sprintf("%d shared blocker(s) need a fixer", overdue)
		b.Detail = fmt.Sprintf("%d live known-bad signature(s) taxing the fleet; %d have NO elected fixer and %d agent(s)/issue(s) are parked. A human must step in or elect one.", len(ordered), overdue, totalParked)
		b.Action = "elect a fixer or resolve the known-bad"
		b.ActionURL = knownBadURL(repoURL)
	} else {
		b.Severity = SeverityStatus
		b.Title = fmt.Sprintf("%d shared blocker(s) contained", len(ordered))
		b.Detail = fmt.Sprintf("%d live known-bad signature(s); %d affected, %d parked behind an elected fixer — contained, witness pending.", len(ordered), totalAffected, totalParked)
	}

	shown := ordered
	if len(shown) > maxBlastLines {
		shown = shown[:maxBlastLines]
	}
	for _, s := range shown {
		b.Lines = append(b.Lines, signatureLine(s))
	}
	if len(ordered) > len(shown) {
		b.Lines = append(b.Lines, fmt.Sprintf("…and %d more (unclaimed-first)", len(ordered)-len(shown)))
	}
	return b
}

// signatureLine renders one signature row: the blast frame (root cause -> N affected, 1
// fixing / NO fixer, N-1 parked) plus the witness state, so an operator reads the whole
// containment state of one shared bug at a glance.
func signatureLine(s Signature) string {
	cause := shortSig(s.ID)
	if r := strings.TrimSpace(s.Reason); r != "" {
		cause = fmt.Sprintf("%s `%s`", cause, r)
	}
	var fixing string
	if s.claimed() {
		fixing = fmt.Sprintf("1 fixing (@%s)", strings.TrimPrefix(strings.TrimSpace(s.Fixer), "@"))
	} else {
		fixing = "*NO FIXER*"
	}
	witness := "witness: pending"
	if !s.WitnessPending {
		witness = "witness: resolved"
	}
	return fmt.Sprintf("%s → %d affected, %s, %d parked · %s",
		cause, s.Affected, fixing, s.parked(), witness)
}

// shortSig trims a "sha256:"-scheme signature to its scheme tag plus a short hex prefix
// so a card row stays legible; a non-hashed or short id is returned as-is (trimmed).
func shortSig(id string) string {
	s := strings.TrimSpace(id)
	if s == "" {
		return "(unknown)"
	}
	if i := strings.IndexByte(s, ':'); i >= 0 && i+1 < len(s) {
		scheme, body := s[:i], s[i+1:]
		if len(body) > 12 {
			body = body[:12] + "…"
		}
		return scheme + ":" + body
	}
	return s
}

// knownBadURL builds the operator card's "do this next" link to the known-bad ledger in
// the repo. Returns "" when no repo base is known (the button is then omitted; the
// fallback line still names the action).
func knownBadURL(repoURL string) string {
	repoURL = strings.TrimRight(strings.TrimSpace(repoURL), "/")
	if repoURL == "" {
		return ""
	}
	return repoURL + "/blob/main/docs/nightrun/known-bad.jsonl"
}
