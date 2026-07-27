package modelroute

import "sort"

// CAPABILITY GRADING: turning observed outcomes into a capability a candidate may
// carry down the ladder (epic #5416, track F).
//
// Place() refuses to let an UNMEASURED capability win a cheap rung. That refusal is what
// keeps the placer from "saving" money on evidence nobody has, and it is also, today, the
// reason nothing moves: every Candidate arrives ungraded, so the only rung allowed to
// serve is the one that already served. This file is the other half of that seam — the
// pure fold that turns OBSERVED outcomes into a grade, or honestly declines to.
//
// The whole design is one rule, applied four ways: A GRADE IS A CLAIM, SO IT MUST COST
// EVIDENCE.
//
//  1. Untrusted evidence does not count. A model's own report that it succeeded is
//     forgeable (verify.go's fail-closed Trusted()), and a forgeable success is exactly
//     how a cheap rung would come to serve work it cannot do. Self-reported attempts are
//     DROPPED, and the count of what was dropped is reported so a caller can see that the
//     evidence existed and was refused rather than never arriving.
//  2. Evidence tagged with an unrecognised class does not count. PolicyFor maps an unknown
//     class to the T0 floor (conservative when choosing a floor for WORK), so grading
//     through it would let a typo in a class label mint the STRONGEST possible capability
//     claim. The same conservatism, read in the other direction, is a hole.
//  3. A grade is the floor of the work the model was observed to do, never the optimal
//     tier of it. Succeeding at security work (floor T1, optimal T0) evidences T1. The
//     optimal tier is what an operator would PREFER to spend on that class, which is not
//     something the model's success rate says anything about.
//  4. Failing every bar is not a grade of "worst tier". A model that missed the success
//     floor everywhere comes back UNMEASURED with the reason, not T2 — because T2 is a
//     positive claim that it can serve routine work, and the evidence says it could not.
//
// Grading is DOWNWARD-monotone by construction and never upward: qualifying at a
// demanding class carries the easier ones with it (a capability IS "the most demanding
// tier this model can serve"), while qualifying only at routine work says nothing about
// harder work and grades T2.
//
// Pure and stdlib-only, like the rest of this leaf: the caller above this seam owns
// producing the evidence (a judge score, a DOS git witness) exactly as it owns producing
// a Verification.

// Closed reason vocabulary for a grade. A surface renders these verbatim so "why is my
// local model still not eligible for the device rung?" has an answer that is not free text.
const (
	ReasonGradedFromEvidence  = "graded-from-verified-evidence" // a class met both bars
	ReasonNoTrustedEvidence   = "no-trusted-evidence"           // nothing survived the provenance/class filter
	ReasonInsufficientSamples = "insufficient-samples"          // trusted evidence, but no class reached MinAttempts
	ReasonBelowSuccessFloor   = "below-success-floor"           // enough samples, no class met the rate
)

// ClassEvidence is what ONE model was observed to do on work of ONE class: how many
// attempts, how many of them succeeded, and HOW that success was established.
//
// Verify is not decoration. It is the difference between "the model says it did the
// work" and "DOS confirmed from git that the work shipped", and only the second kind of
// fact may move traffic onto cheaper hardware.
type ClassEvidence struct {
	Class     WorkClass    `json:"class"`
	Attempts  int64        `json:"attempts"`
	Successes int64        `json:"successes"`
	Verify    Verification `json:"verify"`
}

// GradeFloor is the operator's evidentiary bar — the two numbers plus the provenance
// knob that decide how much observation buys a grade.
//
// There is deliberately no knob for "let self-report count". An operator who wants to
// assert a capability without evidence can do so explicitly and be held to it (that is
// what a hand-declared Candidate.Capability IS); what must not exist is a setting that
// makes an assertion look like a measurement.
type GradeFloor struct {
	// MinAttempts is the per-class sample floor. A class below it is not evidence.
	MinAttempts int64 `json:"min_attempts"`
	// MinSuccessRate is the fraction of attempts that must have succeeded, in [0,1].
	MinSuccessRate float64 `json:"min_success_rate"`
	// RequireWitness narrows trusted provenance to DOS git-witnessed outcomes only,
	// dropping judge scores. A fleet whose cheap rung touches anything it would mind
	// being wrong about sets this; the default accepts a judge.
	RequireWitness bool `json:"require_witness,omitempty"`
}

// DefaultGradeFloor is the bar a caller gets by asking for nothing: 20 independently
// verified attempts at a class, 80% of them successful. The numbers are a starting
// posture, not a discovery — they are stated here so a fleet that changes them changes
// something visible rather than inheriting a hidden constant.
func DefaultGradeFloor() GradeFloor {
	return GradeFloor{MinAttempts: 20, MinSuccessRate: 0.8}
}

// counts reports whether one piece of evidence may be graded through at all: its
// provenance must be independent (and witnessed, when the floor demands it) and its
// class must be one the policy table actually knows.
func (f GradeFloor) counts(e ClassEvidence) bool {
	if !knownWorkClass(e.Class) {
		return false
	}
	if f.RequireWitness {
		return e.Verify.Witnessed()
	}
	return e.Verify.Trusted()
}

// knownWorkClass reports whether a class is one of the four the policy table names.
// PolicyFor deliberately maps everything else to the strictest floor, which is the right
// answer when choosing a floor for WORK and the wrong one when grading a MODEL.
func knownWorkClass(c WorkClass) bool {
	switch c {
	case ClassUltraHard, ClassNormalImpl, ClassRoutine, ClassSecurityRelease:
		return true
	}
	return false
}

// Grade is a model's graded capability plus the evidence trail behind it.
//
// Measured is the field Place() reads, and it is false in every case where the evidence
// did not reach the bar — including the case where plenty of evidence arrived and all of
// it was self-reported. Capability is meaningless when Measured is false; it is left at
// the zero value rather than at some "lowest" tier so that a caller ignoring Measured
// fails toward the most demanding rung instead of the cheapest.
type Grade struct {
	Model      string       `json:"model"`
	Capability WorkTier     `json:"capability"`
	Measured   bool         `json:"measured"`
	Reason     string       `json:"reason"`
	Class      WorkClass    `json:"class,omitempty"`   // the class whose floor produced the grade
	Attempts   int64        `json:"attempts"`          // counted attempts at that class
	Successes  int64        `json:"successes"`         // of which succeeded
	Verify     Verification `json:"verify,omitempty"`  // the WEAKEST provenance behind the grade
	Dropped    int64        `json:"dropped,omitempty"` // attempts refused by the provenance/class filter
}

// Candidate turns a grade into a placement candidate. An ungraded model still becomes a
// candidate — it can serve the top rung — it simply cannot descend.
func (g Grade) Candidate() Candidate {
	return Candidate{Model: g.Model, Capability: g.Capability, Measured: g.Measured}
}

// GradeCapability folds one model's observed outcomes into a capability grade.
//
// The result is deterministic in the evidence: the same rows in any order yield the same
// grade, because rows are merged per class before anything is compared.
func GradeCapability(model string, evidence []ClassEvidence, floor GradeFloor) Grade {
	g := Grade{Model: model, Reason: ReasonNoTrustedEvidence}

	type tally struct {
		attempts, successes int64
		weakest             Verification
		haveVerify          bool
	}
	byClass := map[WorkClass]*tally{}
	for _, e := range evidence {
		attempts, successes := e.Attempts, e.Successes
		if attempts <= 0 {
			continue
		}
		// Malformed evidence is clamped DOWN, never up: more successes than attempts is
		// a producer bug, and the safe reading of a bug is the pessimistic one.
		if successes < 0 {
			successes = 0
		}
		if successes > attempts {
			successes = attempts
		}
		if !floor.counts(e) {
			g.Dropped += attempts
			continue
		}
		t := byClass[e.Class]
		if t == nil {
			t = &tally{}
			byClass[e.Class] = t
		}
		t.attempts += attempts
		t.successes += successes
		if !t.haveVerify || e.Verify.Rank() < t.weakest.Rank() {
			t.weakest, t.haveVerify = e.Verify, true
		}
	}
	if len(byClass) == 0 {
		return g
	}

	// Walk the surviving classes in a fixed order, then keep the most demanding floor
	// that cleared both bars. Order is fixed for determinism only — the comparison below
	// is what decides, and it goes through MoreDemandingThan so the T0<T1<T2 numbering
	// inversion cannot leak in as a raw `<`.
	classes := make([]WorkClass, 0, len(byClass))
	for c := range byClass {
		classes = append(classes, c)
	}
	sort.Slice(classes, func(i, j int) bool { return classes[i] < classes[j] })

	sawEnoughSamples := false
	for _, c := range classes {
		t := byClass[c]
		if t.attempts < floor.MinAttempts {
			continue
		}
		sawEnoughSamples = true
		if successRate(t.successes, t.attempts) < floor.MinSuccessRate {
			continue
		}
		tier := PolicyFor(c).RequiredTier
		if g.Measured && !tier.MoreDemandingThan(g.Capability) {
			continue
		}
		g.Capability, g.Measured, g.Reason = tier, true, ReasonGradedFromEvidence
		g.Class, g.Attempts, g.Successes, g.Verify = c, t.attempts, t.successes, t.weakest
	}
	if !g.Measured {
		g.Reason = ReasonInsufficientSamples
		if sawEnoughSamples {
			g.Reason = ReasonBelowSuccessFloor
		}
	}
	return g
}

// GradeCandidates grades every model in a fixed order, so a placement built from evidence
// is as reproducible as one built from a hand-written declaration. A model with no
// evidence at all still appears — ungraded, and therefore top-rung only.
func GradeCandidates(models []string, evidence map[string][]ClassEvidence, floor GradeFloor) []Grade {
	out := make([]Grade, 0, len(models))
	seen := map[string]bool{}
	for _, m := range models {
		if m == "" || seen[m] {
			continue
		}
		seen[m] = true
		out = append(out, GradeCapability(m, evidence[m], floor))
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Model < out[j].Model })
	return out
}

// successRate is the observed success fraction. Zero attempts yields 0, which never
// clears a floor above 0 — a class with no attempts cannot pass by dividing by nothing.
func successRate(successes, attempts int64) float64 {
	if attempts <= 0 {
		return 0
	}
	return float64(successes) / float64(attempts)
}
