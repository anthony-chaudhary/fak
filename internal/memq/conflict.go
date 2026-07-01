package memq

import (
	"fmt"
	"strings"
)

// conflict.go — issue #1597: managed-context needs an EXPLICIT resolution the moment
// two durable facts about the same subject disagree, rather than letting the newer
// promotion silently overwrite the older one (a caller reading PromotionLedger.For
// today gets back BOTH records with no signal that they conflict at all) or letting
// both sit in the ledger as if they were compatible history.
//
// This mirrors recall.stalefact.go's shape exactly (#1594): a pure function of its
// inputs, closed-vocabulary outcome, fail-closed default, no model call, no I/O. It
// lives in memq (not recall) because PromotionRecord/PromotionLedger — the storage this
// issue says to reuse rather than inventing a second ledger — are memq types, and memq
// already imports recall (recallbackend.go): recall importing memq back for this would
// be an import cycle, so the resolver is native to the package that owns the record
// type it resolves over, exactly like PromotionLedger.Explain/Timeline already are.
//
// WHAT COUNTS AS "THE SAME THING". Two PromotionRecords are a CANDIDATE conflict when
// they share a subject — same CellID (a cell re-promoted/reclassified over its life,
// exactly the multi-record history PromotionLedger.For already returns) OR the same
// non-empty SourceSpan.Descriptor (two different cells asserting a fact about the same
// named subject, e.g. two "user's preferred name" observations under different cell
// IDs). They are an ACTUAL conflict only when their content actually disagrees —
// different SourceSpan.Digest (the CAS address of the underlying bytes). Two records
// with the same digest are the same fact re-affirmed, not a conflict; DetectFactConflict
// returns ConflictNone for that case rather than flagging agreement as disagreement.

// ConflictOutcome is the CLOSED vocabulary of resolutions for two durable facts that
// disagree — mirroring recall.StaleFactOutcome's shape (closed string type, membership
// set, fail-closed String()). Exactly one outcome is produced per detection; there is
// no "no decision" state once DetectFactConflict runs on an actual conflict.
type ConflictOutcome string

const (
	// ConflictNone: the two records are not in conflict — either they do not share a
	// subject, or they share a subject but carry identical content (same digest, a
	// re-affirmation rather than a disagreement). Nothing to resolve.
	ConflictNone ConflictOutcome = "none"
	// ConflictPreferNewer: both records carry ConsentExplicit and the disagreement is
	// within the same scope — safe to let the later promotion supersede the earlier
	// one, but EXPLICITLY (the decision travels with a reason and names the winner),
	// never a silent overwrite.
	ConflictPreferNewer ConflictOutcome = "prefer_newer"
	// ConflictAskUser: the disagreement cannot be resolved from the records' own
	// metadata alone — e.g. either side's consent is ConsentUnknown, or both sides are
	// merely ConsentInferred with no explicit affirmation to break the tie. The safe
	// move is to ask before silently picking a winner, mirroring
	// recall.StaleFactExpiredMustQuery / ctxplan.PageFaultQueryUser's "ask rather than
	// assume" posture. A caller wiring an interactive surface may route this outcome
	// through selfquery's clarification broker (selfquery.ClarificationQuestion),
	// exactly as selfquery.PromotionClarification (#1596) already routes an ambiguous
	// CONSENT verdict through that broker — this package does not import selfquery
	// itself (selfquery is the composer that imports memq, not the reverse), it only
	// mints the same shape of typed, ask-worthy decision selfquery already knows how to
	// wrap.
	ConflictAskUser ConflictOutcome = "ask_user"
	// ConflictKeepBothScoped: both records carry independent explicit consent (the
	// user affirmatively asserted each one) and their source spans differ enough
	// (different Step AND Role) that they plausibly describe two SCOPES of the same
	// subject rather than one subject with two contradictory values (e.g. "prefers
	// tea" from a home-context turn and "prefers coffee" from a work-context turn).
	// Both are retained; neither is discarded; the decision records that they are
	// scoped rather than merged.
	ConflictKeepBothScoped ConflictOutcome = "keep_both_scoped"
)

// validConflictOutcomes is the membership set every ConflictDecision.Outcome belongs
// to — used by tests and any (de)serializing caller to fail closed on a corrupt or
// foreign value.
var validConflictOutcomes = map[ConflictOutcome]bool{
	ConflictNone:           true,
	ConflictPreferNewer:    true,
	ConflictAskUser:        true,
	ConflictKeepBothScoped: true,
}

// ValidConflictOutcome reports whether o is a member of the closed vocabulary.
func ValidConflictOutcome(o ConflictOutcome) bool { return validConflictOutcomes[o] }

func (o ConflictOutcome) String() string {
	if ValidConflictOutcome(o) {
		return string(o)
	}
	if o == "" {
		return "(unset)"
	}
	return "unknown(" + string(o) + ")"
}

// ConflictDecision is the typed verdict DetectFactConflict produces: the outcome, an
// operator-readable reason (mirroring recall.StaleFactDecision.Reason), and the two
// records it was computed over so a caller/audit surface never has to re-derive which
// promotion "won" from prose.
type ConflictDecision struct {
	Outcome ConflictOutcome `json:"outcome"`
	Reason  string          `json:"reason"`
	// Winner is the record DetectFactConflict identifies as the one to keep acting on
	// when Outcome is ConflictPreferNewer (the zero value otherwise — ConflictAskUser
	// and ConflictKeepBothScoped never pick a single winner by construction).
	Winner PromotionRecord `json:"winner,omitempty"`
	// A and B are the two records the decision was computed over, always in the same
	// order they were passed to DetectFactConflict — never reordered, so a caller can
	// tell which side of the call produced Winner.
	A PromotionRecord `json:"a"`
	B PromotionRecord `json:"b"`
}

// SameSubject reports whether two PromotionRecords describe the same durable subject —
// the precondition for a conflict to even be possible. Two records are the same
// subject when they share a CellID (the same cell re-promoted over its life) or share a
// non-empty SourceSpan.Descriptor (two cells asserting a fact about the same named
// thing). An empty descriptor never matches another empty descriptor — two records that
// both failed to record a descriptor are not thereby "the same subject," they are two
// unlabeled facts, and treating them as a match would manufacture false conflicts out
// of missing data.
func SameSubject(a, b PromotionRecord) bool {
	if a.CellID != "" && a.CellID == b.CellID {
		return true
	}
	d1, d2 := strings.TrimSpace(a.SourceSpan.Descriptor), strings.TrimSpace(b.SourceSpan.Descriptor)
	return d1 != "" && d1 == d2
}

// DetectFactConflict is the PURE detection function (#1597): given two PromotionRecords
// already identified as SameSubject, it returns EXACTLY ONE closed-vocabulary
// ConflictDecision. No clock read, no I/O, no hidden state — the same (a, b) always
// reproduces the same decision, and swapping the argument order never changes the
// resolved Outcome or Winner (only which of A/B the caller sees the inputs echoed back
// as), so a caller cannot get a different answer by calling it "the other way around."
//
// Decision order (first match wins, most conservative first — mirrors
// recall.DetectStaleFact's structure):
//
//  1. Not SameSubject -> ConflictNone. Nothing to resolve; the two records are about
//     different things.
//  2. SameSubject but identical SourceSpan.Digest (or both empty) -> ConflictNone. A
//     re-affirmation of the same content is agreement, not disagreement.
//  3. SameSubject, differing digest, BOTH sides carry ConsentExplicit -> compare source
//     spans: if the spans differ in Step AND Role (distinct scopes: different turns
//     produced by different roles/contexts), the disagreement is plausibly two scoped
//     facts rather than one fact with two contradictory values ->
//     ConflictKeepBothScoped. Otherwise (both explicit, same/overlapping scope, genuinely
//     contradictory) -> ConflictPreferNewer, with Winner set to the higher-Seq record
//     (later promotion wins because both sides already cleared the strongest consent
//     bar, so recency is a safe, explicit tiebreaker — never a silent one, since the
//     decision and its reason travel with the resolved value).
//  4. Either side is ConsentUnknown, or neither side is ConsentExplicit (both merely
//     ConsentInferred with no explicit affirmation to break the tie) -> ConflictAskUser.
//     The records disagree and nothing in their own metadata is strong enough to settle
//     it without a human — the same "ask rather than assume" posture
//     recall.StaleFactExpiredMustQuery already takes.
func DetectFactConflict(a, b PromotionRecord) ConflictDecision {
	dec := ConflictDecision{A: a, B: b}

	if !SameSubject(a, b) {
		dec.Outcome = ConflictNone
		dec.Reason = "records do not share a subject (different cell and no matching descriptor): nothing to resolve"
		return dec
	}

	da, db := strings.TrimSpace(a.SourceSpan.Digest), strings.TrimSpace(b.SourceSpan.Digest)
	if da == db {
		dec.Outcome = ConflictNone
		dec.Reason = "same subject but identical content digest: a re-affirmation, not a disagreement"
		return dec
	}

	ca, cb := NormConsent(a.Consent), NormConsent(b.Consent)

	if ca == ConsentExplicit && cb == ConsentExplicit {
		if scopesDiffer(a, b) {
			dec.Outcome = ConflictKeepBothScoped
			dec.Reason = fmt.Sprintf(
				"both records carry explicit consent but their source spans differ (step %d/role %q vs step %d/role %q): retaining both as distinct scopes rather than discarding either",
				a.SourceSpan.Step, a.SourceSpan.Role, b.SourceSpan.Step, b.SourceSpan.Role,
			)
			return dec
		}
		winner := a
		if b.Seq > a.Seq {
			winner = b
		}
		dec.Outcome = ConflictPreferNewer
		dec.Winner = winner
		dec.Reason = fmt.Sprintf(
			"both records carry explicit consent for the same scope with contradictory content (digests %q vs %q): preferring the newer promotion (seq=%d) explicitly, not silently",
			da, db, winner.Seq,
		)
		return dec
	}

	dec.Outcome = ConflictAskUser
	dec.Reason = fmt.Sprintf(
		"records disagree (digests %q vs %q) and consent alone cannot settle it (a=%s, b=%s): asking rather than assuming a winner",
		da, db, ca, cb,
	)
	return dec
}

// scopesDiffer reports whether two source spans plausibly describe different SCOPES of
// the same subject (as opposed to two contradictory readings of the identical scope).
// It requires BOTH the step and the role to differ — a change in only one (e.g. the
// same role speaking again at a later step) is ordinary re-affirmation-or-correction
// traffic within the same scope, not evidence of a new scope; the bar for
// ConflictKeepBothScoped is deliberately high so it is not used as an escape hatch that
// avoids ever having to prefer-newer or ask.
func scopesDiffer(a, b PromotionRecord) bool {
	return a.SourceSpan.Step != b.SourceSpan.Step && a.SourceSpan.Role != "" && b.SourceSpan.Role != "" &&
		a.SourceSpan.Role != b.SourceSpan.Role
}

// DetectFactConflicts scans every SameSubject pair within a slice of PromotionRecords
// (e.g. PromotionLedger.For(cellID), or a caller-assembled cross-cell candidate set for
// a shared descriptor) and returns a ConflictDecision for each pair whose Outcome is
// not ConflictNone — the caller-facing entry point that spares every site from
// re-deriving pairwise iteration, mirroring the "detect over the whole set, act only on
// the actionable subset" shape this package's other multi-record scans already use.
// Pairs are compared in slice order (i<j); a record is never compared against itself.
func DetectFactConflicts(recs []PromotionRecord) []ConflictDecision {
	var out []ConflictDecision
	for i := 0; i < len(recs); i++ {
		for j := i + 1; j < len(recs); j++ {
			d := DetectFactConflict(recs[i], recs[j])
			if d.Outcome != ConflictNone {
				out = append(out, d)
			}
		}
	}
	return out
}

// Conflicts scans this ledger's OWN history for cellID — PromotionLedger.For(cellID) —
// for internal disagreements (a cell that was re-promoted with contradictory content
// over its life). This is the direct "reuse PromotionLedger's existing storage/lookup"
// entry point the issue asks for: a caller never needs a second ledger or to
// re-assemble the record slice itself to ask "does this cell's own promotion history
// conflict with itself." Cross-cell conflicts (same descriptor, different CellID) are
// still reachable via DetectFactConflicts on a caller-assembled slice; this method
// covers the common single-cell case for free.
func (l *PromotionLedger) Conflicts(cellID string) []ConflictDecision {
	recs, ok := l.For(cellID)
	if !ok {
		return nil
	}
	return DetectFactConflicts(recs)
}
