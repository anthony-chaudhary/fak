package main

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// The placement ladder's ACTUATOR (epic #5416 track D).
//
// dispatch_rung_resolve.go starts a slot on the cheapest rung its measured evidence supports.
// This is the other direction: when that rung TRIED the work and could not do it, something
// has to re-dispatch the item one rung up, or the cheap-by-default posture is just a cheaper
// way to fail. modelroute.AfterAttempt is the rule that decides; dispatchtick.AttemptResultFor
// is the classifier that feeds it; both have been live and neither was wired to anything.
//
// Escalation is an automatic SPEND, so every part of this file is built to make that spend
// bounded, attributable and refusable:
//
//  1. Only an UNDERPOWERED failure earns a rung — AfterAttempt's rule 2, and the reason the
//     classifier refuses to call an unwitnessed claim underpowered. A guard refusal never
//     escalates (re-aiming a refusal at another rung is a bypass, not a retry) and a
//     transport wall retries where it stood.
//  2. The authority must be DECLARED: a ceiling zone and a per-item attempt budget, both
//     read from the operator's environment, both fail-closed. Nothing here defaults.
//  3. The budget is counted from the durable ledger (modelroute.AppendEscalation), not from
//     memory, because a budget that resets every process restart is not a budget.
//  4. The debit is written BEFORE the pin is returned. If the ledger cannot be appended, the
//     escalation does not happen. A rung bought without a recorded debit is a rung outside
//     the budget.
//  5. One debit per ATTEMPT, not per tick. The ledger row is keyed by the finished slot being
//     escalated away from, so a witness record that appears in two consecutive sweeps is
//     charged once and re-applies the SAME decision — otherwise a long-lived record would
//     drain an item's budget by being looked at repeatedly.
//
// It shares the launchability gate with the placement half: a rung this backend cannot dial
// is not a rung to escalate onto, and the vendor rung is not exempt from that. Like the
// placement half, this is a launchability boundary and not a residency one — it answers "can
// this worker reach that rung", never "may this payload leave the box" (#5421).
//
// Known and stated rather than hidden: the ledger is a file, so an operator who deletes it
// resets every item's budget. That is a property of any on-disk budget and the alternative
// (refusing to run without one) would wedge a fleet on a missing file. What the ledger does
// guarantee is that nothing SILENTLY forgets — a reset is an act, not a restart.

const (
	// dispatchRungCeilingEnv is the highest rung escalation may reach (device / fleet /
	// vendor). Unset authorises nothing: AfterAttempt's rule 4 stops on an invalid ceiling.
	dispatchRungCeilingEnv = "FLEET_DISPATCH_RUNG_CEILING"
	// dispatchRungBudgetEnv is how many rungs ONE work item may buy. Unset authorises
	// nothing; it is never read as "unlimited".
	dispatchRungBudgetEnv = "FLEET_DISPATCH_RUNG_BUDGET"

	// dispatchEscalationLedgerName is the append-only spend ledger, beside the turn journal
	// it is the mirror image of.
	dispatchEscalationLedgerName = "escalations.jsonl"
)

// The closed vocabulary of reasons THIS file did not escalate, reported under the payload key
// rung_escalation_skipped. It is disjoint from AfterAttempt's reasons, which are reported
// under the same key verbatim: those say the RULE declined, these say the actuator could not
// put the rule's verdict into effect.
const (
	escSkipOutranked    = "escalation-outranked"          // somebody other than the ladder chose this model
	escSkipNotLive      = "escalation-not-live"           // a preview tick must not spend a budget
	escSkipNoAttempt    = "no-finished-attempt"           // nothing in this sweep finished for the target
	escSkipUnidentified = "unidentifiable-attempt"        // a finished slot with no log to key a debit to
	escSkipBadCeiling   = "bad-escalation-ceiling"        // the declared ceiling is not a rung
	escSkipBadBudget    = "bad-escalation-budget"         // the declared budget is not a number
	escSkipNoBudgetDecl = "no-declared-escalation-budget" // nobody granted an attempt budget
	escSkipLedgerBig    = "ledger-too-large"              // past the per-launch read cap
	escSkipLedgerBad    = "ledger-unreadable"             // present and it does not read
	escSkipLedgerWrite  = "ledger-unwritable"             // the debit would not append; nothing is bought
	escSkipNoRungAbove  = "no-reachable-rung-above"       // the authorised rungs bind nothing placeable
	escSkipUnmeasured   = "escalation-unmeasured"         // the rung above is bound but ungraded
)

// modelSourceEscalated is the placement ladder acting on an underpowered outcome: the rung
// ABOVE the one that just failed, bought against a declared budget and recorded as spent.
//
// Distinct from modelSourceRung because the two answer different questions. That one is "what
// is the cheapest rung the evidence supports for this class of work"; this one is "the rung
// that answer picked has now tried and failed, and the operator authorised paying more". A
// payload that conflated them would hide every automatic spend inside the seam whose entire
// purpose is to avoid spending.
const modelSourceEscalated = "placement-escalated"

// rungEscalation is what the actuator did, for the tick payload. Nil when nothing escalated.
type rungEscalation struct {
	Issue int
	From  modelroute.PlacementZone
	To    modelroute.PlacementZone
	Model string
	// Reason is AfterAttempt's closed token for WHY the rung was earned.
	Reason string
	// Replayed reports that the debit for this attempt already existed, so the same decision
	// was re-applied without charging again.
	Replayed bool
}

// Map renders the escalation for the tick payload — closed tokens only, no free text.
func (e rungEscalation) Map() map[string]any {
	out := map[string]any{
		"issue":  e.Issue,
		"from":   string(e.From),
		"to":     string(e.To),
		"model":  e.Model,
		"reason": e.Reason,
	}
	if e.Replayed {
		out["replayed"] = true
	}
	return out
}

// dispatchRungBounds reads the operator's declared escalation authority. The reason is
// non-empty when a declaration is present but unusable, which is deliberately distinct from
// an absent one: a typo'd ceiling and an undeclared ceiling have different cures, and
// reporting the first as the second sends an operator to write a variable they already wrote.
//
// An ABSENT ceiling is not a reason here — it flows through to AfterAttempt, whose rule 4
// already names it (no-declared-escalation-ceiling) as part of the closed vocabulary.
func dispatchRungBounds() (modelroute.EscalationBounds, string) {
	var b modelroute.EscalationBounds
	if raw := strings.TrimSpace(os.Getenv(dispatchRungCeilingEnv)); raw != "" {
		z := modelroute.PlacementZone(strings.ToLower(raw))
		if !z.Valid() {
			return b, escSkipBadCeiling
		}
		b.Ceiling = z
	}
	raw, ok := os.LookupEnv(dispatchRungBudgetEnv)
	if !ok || strings.TrimSpace(raw) == "" {
		// AfterAttempt would call this "budget spent", which is true of the arithmetic and
		// wrong about the operator: nothing was ever granted, and "spent" reads as "this item
		// has had its rungs" to the person reading the payload.
		return b, escSkipNoBudgetDecl
	}
	n, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil {
		return b, escSkipBadBudget
	}
	b.MaxAttempts = n
	return b, ""
}

// dispatchEscalationLedgerPath is where the debits live.
func dispatchEscalationLedgerPath(root string) string {
	return filepath.Join(root, dispatchtick.RunsDirName, dispatchEscalationLedgerName)
}

// readEscalationLedger loads the durable spend ledger. A MISSING ledger is not a failure —
// nothing has been bought yet — but an unreadable or oversized one is, because a budget
// counted from a file the reader could not finish is not a bound.
func readEscalationLedger(root string) ([]modelroute.EscalationRecord, modelroute.EscalationTally, string) {
	path := dispatchEscalationLedgerPath(root)
	st, err := os.Stat(path)
	if err != nil {
		return nil, modelroute.TallyEscalations(nil, modelroute.JournalStats{}), ""
	}
	if st.Size() > dispatchRungJournalCap {
		return nil, modelroute.EscalationTally{}, escSkipLedgerBig
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, modelroute.EscalationTally{}, escSkipLedgerBad
	}
	defer f.Close()
	records, stats, err := modelroute.ReadEscalations(f)
	if err != nil {
		return nil, modelroute.EscalationTally{}, escSkipLedgerBad
	}
	return records, modelroute.TallyEscalations(records, stats), ""
}

// appendEscalationDebit records one bought rung. It is the last thing that can refuse an
// escalation, and it is allowed to: the caller does not pin a model if this returns an error.
func appendEscalationDebit(root string, rec modelroute.EscalationRecord) error {
	dir := filepath.Join(root, dispatchtick.RunsDirName)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	// O_APPEND, one write per row: concurrent lanes interleave whole lines rather than
	// corrupting each other's, and the one thing that can still tear — a partial final write
	// — is charged rather than dropped by the reader. See modelroute/escalatelog.go.
	f, err := os.OpenFile(filepath.Join(dir, dispatchEscalationLedgerName), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	if err := modelroute.AppendEscalation(f, rec); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// escalationDebitID keys a debit to the ATTEMPT it escalated away from, so the same finished
// slot seen in two sweeps is charged once.
//
// The key is the slot's own log path — the same thing the evidence journal dedupes turns on,
// so a slot recognised as one attempt over there is one attempt over here. The issue number
// is deliberately NOT a fallback: one issue is dispatched many times, and keying on it would
// collapse an item's whole history into a single debit, which is a silent under-charge. An
// unidentifiable slot returns empty and the caller refuses, because the alternative is
// charging the same attempt again on every sweep.
func escalationDebitID(rec dispatchtick.WitnessRecord) string {
	log := strings.TrimSpace(rec.Log)
	if log == "" {
		return ""
	}
	return "slot:" + log
}

// lastFinishedFor returns the target's last finished slot in this tick's sweep.
func lastFinishedFor(records []dispatchtick.WitnessRecord, target int) (dispatchtick.WitnessRecord, bool) {
	var out dispatchtick.WitnessRecord
	found := false
	for _, rec := range records {
		if rec.Issue == target {
			out, found = rec, true
		}
	}
	return out, found
}

// applyRungEscalation raises this slot one rung when the target's last finished attempt was
// underpowered and the operator authorised the spend. It returns the (possibly unchanged)
// policy, what it did, and the reason it did nothing — exactly one of the last two is set.
//
// live is load-bearing: a preview tick must not append a debit, because the attempt it would
// be charging for is never launched. Pinning without charging would be worse still, so a
// non-live tick reports itself and changes nothing.
func applyRungEscalation(root string, live bool, target int, labels []string, records []dispatchtick.WitnessRecord, p workerModelPolicy) (workerModelPolicy, *rungEscalation, string) {
	if !dispatchRungPlacementEnabled() {
		return p, nil, ""
	}
	// Only the ladder's own decisions may be raised. An operator pin, a lane pin and the
	// bench gate are deliberate human intent and outrank an automatic spend — the same
	// doctrine the preventive placement gate follows. A tier/work-class table pin is left
	// alone too: raising it would be this seam second-guessing a different subsystem's
	// declared choice using evidence about rungs that subsystem never consulted.
	switch p.Source {
	case modelSourceSeatDefault, modelSourceRung:
	default:
		return p, nil, escSkipOutranked
	}
	if !live {
		return p, nil, escSkipNotLive
	}
	bounds, reason := dispatchRungBounds()
	if reason != "" {
		return p, nil, reason
	}
	reach, ok := dispatchRungReach()
	if !ok {
		return p, nil, rungSkipNoReachDecl
	}
	class, _ := dispatchtick.WorkClassForIssue(labels)
	if class == "" {
		return p, nil, rungSkipNoWorkClass
	}
	rec, ok := lastFinishedFor(records, target)
	if !ok {
		return p, nil, escSkipNoAttempt
	}
	debitID := escalationDebitID(rec)
	if debitID == "" {
		// Charging an attempt that cannot be recognised again would re-charge it on every
		// sweep, which drains the budget rather than bounding it.
		return p, nil, escSkipUnidentified
	}

	ledger, tally, reason := readEscalationLedger(root)
	if reason != "" {
		return p, nil, reason
	}

	from := modelroute.PlacementZone(strings.TrimSpace(rec.Zone))
	to, why, replayed := escalationTarget(ledger, debitID, from, rec, bounds, tally)
	if to == "" {
		return p, nil, why
	}

	model, landed, reason := resolveEscalatedModel(root, class, reach, to, bounds.Ceiling)
	if reason != "" {
		return p, nil, reason
	}
	// The rung RECORDED is the one the work will actually run on, which is not always the one
	// it earned: when the rung above binds nothing, Place walks on up to the ceiling. Writing
	// the earned rung instead would put a debit in the ledger saying the fleet handled work the
	// vendor is about to be billed for — and on replay it would re-run that same walk under
	// whatever ceiling is current, so a ceiling lowered onto the skipped rung would read as
	// "nothing to escalate onto" rather than "past your ceiling".
	to = landed

	if !replayed {
		// Rule 4 of this file's header: the debit lands before the rung does.
		debit := modelroute.EscalationRecord{
			ID: debitID, Item: strconv.Itoa(target),
			From: from, To: to, Reason: why, At: time.Now().UTC(),
		}
		if err := appendEscalationDebit(root, debit); err != nil {
			return p, nil, escSkipLedgerWrite
		}
	}

	out := workerModelPolicy{
		Model:  model,
		Chain:  dropModel(p.Chain, model),
		Source: modelSourceEscalated,
		// A rung wall is model-scoped, not reasoning-scoped, so the tier's posture crosses the
		// switch — the same rule the Layer-2 downgrade follows.
		Effort:    p.Effort,
		Ultracode: p.Ultracode,
	}
	return out, &rungEscalation{Issue: target, From: from, To: to, Model: model, Reason: why, Replayed: replayed}, ""
}

// escalationTarget decides which rung this attempt is authorised to move to.
//
// A debit already recorded against this attempt is REPLAYED rather than re-decided: the rung
// was bought once and re-applying it costs nothing, while re-running the verdict would charge
// the same failure again every time its record turns up in a sweep. The replay is still
// checked against the ceiling, because an operator who LOWERED the ceiling since is entitled
// to have that apply to the next launch and not only to the next purchase.
//
// to is empty when nothing may move; why then holds the closed reason, from AfterAttempt's
// vocabulary or this file's.
func escalationTarget(ledger []modelroute.EscalationRecord, debitID string, from modelroute.PlacementZone, rec dispatchtick.WitnessRecord, bounds modelroute.EscalationBounds, tally modelroute.EscalationTally) (to modelroute.PlacementZone, why string, replayed bool) {
	for _, e := range ledger {
		if strings.TrimSpace(e.ID) != debitID {
			continue
		}
		if !e.To.Valid() {
			// A row that names no rung cannot be replayed into a placement. It was still a
			// debit and is still counted; the item simply has to earn its next rung afresh.
			break
		}
		if !bounds.Ceiling.Valid() {
			return "", modelroute.ReasonNoCeiling, false
		}
		if e.To.Rank() > bounds.Ceiling.Rank() {
			return "", modelroute.ReasonAtCeiling, false
		}
		return e.To, e.Reason, true
	}
	v := modelroute.AfterAttempt(modelroute.Placement{Zone: from}, dispatchtick.AttemptResultFor(rec), bounds, tally.Spent(strconv.Itoa(rec.Issue)))
	if !v.Escalates() {
		return "", v.Reason, false
	}
	return v.To, v.Reason, false
}

// resolveEscalatedModel picks the model to run on, from the rungs the operator authorised:
// at or above the earned rung `to`, at or below the declared ceiling. It returns the model and
// the rung that model actually sits on, which is the one the ledger records.
//
// It reuses Place rather than reaching into a zone directly, so an escalation obeys the same
// two rules a placement does — the class's tier floor, and no descent onto an unmeasured
// capability. Bounding the pool from ABOVE is what keeps Place's walk from stepping past the
// ceiling when the earned rung binds nothing: without it, an operator's fleet ceiling would
// be silently upgraded to vendor by an empty fleet rung.
func resolveEscalatedModel(root string, class modelroute.WorkClass, reach rungReach, to, ceiling modelroute.PlacementZone) (string, modelroute.PlacementZone, string) {
	rosterPath := dispatchAccountsRosterPath(root)
	if rosterPath == "" {
		return "", "", rungSkipNoRoster
	}
	roster, err := modelroute.LoadRoster(rosterPath)
	if err != nil {
		return "", "", rungSkipNoRoster
	}
	evidence, reason := dispatchRungEvidence(root)
	if reason != "" {
		return "", "", reason
	}
	base := placementCandidates(roster, nil)
	if len(base) == 0 {
		return "", "", rungSkipRosterEmpty
	}
	if base = reach.filter(roster, base); len(base) == 0 {
		return "", "", rungSkipUnreachable
	}
	if base = candidatesWithinRungs(roster, base, to, ceiling); len(base) == 0 {
		return "", "", escSkipNoRungAbove
	}
	pool, _, _ := gradedPool(base, evidence, modelroute.DefaultGradeFloor())
	rung, err := roster.Place(class, pool)
	if err != nil {
		return "", "", rungSkipRefused
	}
	if !rung.Measured {
		// The same refusal the placement half makes, and for a stronger reason: this one is
		// about to spend money on the strength of the grade.
		return "", "", escSkipUnmeasured
	}
	return rung.Model, rung.Zone, ""
}

// candidatesWithinRungs keeps the candidates whose serving account sits in [low, high].
//
// This IS the ceiling — not a pre-filter in front of some later check. Place walks the static
// ladder and returns the first rung that holds an admissible candidate, so the only way to stop
// it stepping past the authorised ceiling is for no candidate above the ceiling to be in the
// pool. There is no second bound on the outcome and there should not be one: Place returns a
// zone only for a rung that had a pool candidate, and errors outright on a candidate it cannot
// resolve, so it never hands back a rung this list did not offer it. An extra check on the
// returned zone would be unreachable, and unreachable code on a spend boundary reads to the
// next person like the thing doing the bounding.
//
// A candidate Resolve rejects is KEPT, exactly as the reach filter keeps it: a dangling
// binding is a misconfigured roster that Place is already fail-loud about, and swallowing it
// here would turn a broken roster into a quiet "no rung above".
func candidatesWithinRungs(roster modelroute.Roster, in []modelroute.Candidate, low, high modelroute.PlacementZone) []modelroute.Candidate {
	out := make([]modelroute.Candidate, 0, len(in))
	for _, c := range in {
		t, err := roster.Resolve(c.Model)
		if err != nil {
			out = append(out, c)
			continue
		}
		z := t.Zone()
		if !z.Valid() || z.Rank() < low.Rank() || z.Rank() > high.Rank() {
			continue
		}
		out = append(out, c)
	}
	return out
}
