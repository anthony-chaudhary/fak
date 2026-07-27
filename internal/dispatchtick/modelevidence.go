package dispatchtick

import (
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// Capability evidence from finished dispatch slots (#5428, epic #5416 tracks D and F).
//
// `modelroute` can grade what a model is capable of from observed turns, and until
// something PRODUCES those turns the grader has nothing to grade — which is not a cosmetic
// gap: rule 2 of `Roster.Place` says an unmeasured capability may not descend the ladder,
// so with no evidence anywhere every placement resolves to the vendor rung and the whole
// three-stratum design moves no traffic at all.
//
// The witness sweep is the strongest evidence this repo already computes. A finished
// worker's claim is graded against git by `dos commit-audit` and, since #3838, against a
// real run of the resolving commit's affected tests — neither of which is the worker's own
// report of how it did. This turns that record into `modelroute.TurnOutcome` rows.
//
// The rule that shapes all of it: **provenance must not correlate with outcome.**
// `FoldTurnOutcomes` keys evidence by (model, class, verification) and deliberately never
// merges rows across provenance, so a producer that filed its successes as *witnessed* and
// its failures as *judged* would hand the grader a witness row that is 100% successes — a
// perfect record manufactured out of bookkeeping alone, with every individual row honest.
// So provenance here is a property of WHAT CHECKED THE SLOT, and both outcomes of the same
// check carry it:
//
//	a test run happened (GREEN or RED)  ->  VerifyWitness, success = GREEN
//	no test run happened               ->  VerifyJudge,   success = the diff-shape claim
//
// That split is this file's own vocabulary read straight: `WitnessOK` proves the diff did
// the KIND of thing the subject claimed and says nothing about whether it works, which is
// a graded claim rather than an executed witness. `CLAIM_TEST_UNRUN` is a valid, surfaced
// state precisely because the rung never fabricates a pass it did not see, and grading a
// model off it as if a test had run would fabricate one here instead.

// WitnessEvidenceOptions carries the three things a witness record cannot know about
// itself. Each is a function rather than a value because one sweep spans many slots.
type WitnessEvidenceOptions struct {
	// Class answers what KIND of work the slot was, and there is deliberately no default
	// and no inference from the issue's title. `PolicyFor` maps an unknown class to the
	// T0 floor, which is the right conservatism when picking a floor for WORK and a
	// capability-MINTING hole when read backwards to grade a MODEL — so a resolver that
	// cannot say drops the record instead of guessing.
	Class func(WitnessRecord) modelroute.WorkClass
	// At stamps the row, normally with the slot's finish time. Evidence with no timestamp
	// cannot be shown to be inside a freshness window, and capability is a property of a
	// model AS DEPLOYED, so an unstamped corpus quietly caps what an operator can ask for.
	At func(WitnessRecord) time.Time
	// Zone records which rung actually served the slot, when the caller knows. Left nil it
	// stays empty, which reads as "not recorded" rather than as the device rung.
	Zone func(WitnessRecord) modelroute.PlacementZone
}

// WitnessEvidenceStats accounts for the records that did NOT become evidence. Each count
// is a different missing wire with a different fix, so they are never summed into one
// "skipped" number.
type WitnessEvidenceStats struct {
	// Produced is the number of outcome rows emitted.
	Produced int
	// Unattributed slots ran on the seat default with no --model pin, so nothing here can
	// say which model earned the result. The fix is Layer-5b pinning, not a guess.
	Unattributed int
	// Unclassified slots had no work class the caller was willing to declare.
	Unclassified int
	// Undated slots produced a row the caller could not stamp; it still counts, but it
	// cannot satisfy a `--since` window later.
	Undated int
	// Unidentified slots produced a row with no stable id, so a second sweep of the same
	// runs directory would count it twice. Reported rather than papered over with a
	// synthetic id: an id that is not stable across sweeps is worse than none, because
	// `FoldTurnOutcomes` would treat each sweep's copy as a fresh attempt.
	Unidentified int
}

// TurnOutcomesFromWitness folds a witness sweep into capability evidence.
//
// Nothing is inferred that the record does not carry: a slot with no pinned model is not
// evidence about any model, and a slot whose class the caller will not declare is not
// evidence about any kind of work. Both are counted, because "the fleet ran 400 slots and
// graded nobody" has an actionable cause and a silent zero does not.
func TurnOutcomesFromWitness(records []WitnessRecord, opt WitnessEvidenceOptions) ([]modelroute.TurnOutcome, WitnessEvidenceStats) {
	var (
		out   []modelroute.TurnOutcome
		stats WitnessEvidenceStats
	)
	for _, r := range records {
		model := strings.TrimSpace(r.Model)
		if model == "" {
			stats.Unattributed++
			continue
		}
		var class modelroute.WorkClass
		if opt.Class != nil {
			class = modelroute.WorkClass(strings.TrimSpace(string(opt.Class(r))))
		}
		if class == "" {
			stats.Unclassified++
			continue
		}
		o := modelroute.TurnOutcome{
			ID:    witnessOutcomeID(r),
			Model: model,
			Class: class,
		}
		o.Success, o.Verify = witnessOutcome(r)
		if opt.At != nil {
			o.At = opt.At(r)
		}
		if o.At.IsZero() {
			stats.Undated++
		}
		if o.ID == "" {
			stats.Unidentified++
		}
		if opt.Zone != nil {
			o.Zone = opt.Zone(r)
		}
		out = append(out, o)
		stats.Produced++
	}
	return out, stats
}

// witnessOutcome grades one finished slot into (success, provenance).
//
// The test rung decides the PROVENANCE, and the claim decides the outcome only where no
// test ran. Written as one switch on purpose: any arrangement that reaches the witness
// provenance on a success path but not on the matching failure path re-opens the
// bookkeeping hole this file exists to close.
func witnessOutcome(r WitnessRecord) (bool, modelroute.Verification) {
	switch strings.TrimSpace(r.TestClaim) {
	case ClaimTestGreen:
		// The affected tests ran and passed. This is the only path to witnessed success,
		// and the only rung in the sweep that observed the work actually WORKING — but it
		// still requires the diff witness, because a commit that passes its tests while
		// doing something other than what the subject claimed is not evidence that the
		// model can do the asked-for work.
		return r.Claim == ClaimWitnessed, modelroute.VerifyWitness
	case ClaimTestRed:
		// They ran and failed. A diff-witnessed commit can still be RED, and that failure
		// is exactly as independently observed as the green one — same bucket, so a model
		// cannot bank its greens at a provenance its reds never reach.
		return false, modelroute.VerifyWitness
	}
	// No test ran (UNRUN, or a no-commit slot that never had a test rung at all). The only
	// thing that graded this slot is the diff-shape audit, which checked the SHAPE of the
	// claim rather than its effect — so both the success and the failure land at judge.
	return r.Claim == ClaimWitnessed, modelroute.VerifyJudge
}

// witnessOutcomeID returns an id that is stable across re-sweeps of the same runs
// directory and distinct per slot, or "" when the record carries neither.
//
// The log path is the slot's own file and survives a re-sweep, which is what makes replay
// refusable. The issue number deliberately is NOT a fallback: one issue is dispatched many
// times, so keying on it would collapse a model's whole history with that issue into a
// single attempt — a silent LOSS of evidence, where an empty id merely costs the replay
// check and says so.
func witnessOutcomeID(r WitnessRecord) string {
	if log := strings.TrimSpace(r.Log); log != "" {
		return "slot:" + log
	}
	if sha := strings.TrimSpace(r.SHA); sha != "" {
		return "sha:" + sha
	}
	return ""
}
