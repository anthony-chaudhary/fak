package modelroute

// Verification provenance — HOW a served route's quality was checked, and the
// coverage rollup that makes LOWER-tier routing observably SAFE (#600 follow-on,
// epic #595).
//
// THE GAP IT FILLS. outcome.go records a served route's Quality as a bare 0..1
// number, but a number alone hides its own trustworthiness: a 1.0 the serving
// agent asserted about itself is not the same fact as a 1.0 a larger judge model
// gave it, and neither is the same fact as a 1.0 DOS WITNESSED from git evidence
// (the task actually shipped/passed). When the whole point is to hand a LOWER-tier
// agent a lower-tier task and still sleep at night, the load-bearing question is
// not "what score?" but "who checked, and could they have forged it?". This file
// adds that provenance axis and the pure coverage fold over it, so the safety of a
// cheap route is OBSERVABLE, not assumed.
//
// THE THREE TIERS (ordered least- to most-trusted, forgeable to non-forgeable):
//   - VerifyNone    self-report: the serving agent's own claim, no independent
//                   check. Fully forgeable; the fail-closed default.
//   - VerifyJudge   a larger/judge model scored the answer (the Scorer seam in
//                   judge.go). An OBSERVED model opinion — better than self-report,
//                   but still a relayed judgement, not git-truth.
//   - VerifyWitness DOS confirmed the work from git evidence (dos verify /
//                   commit-audit). WITNESSED, non-forgeable; the only tier fak
//                   controls end-to-end, and the safest.
//
// THE HONESTY BOUNDARY (fail-closed, load-bearing). Coverage math counts ONLY the
// two known independent tiers as trusted; an absent OR unrecognised provenance is
// treated as unverified and NEVER inflates coverage. So a route can only be
// reported as "checked" when it carries a provenance token that names a real
// check — the same discipline outcome.go uses to keep an unmeasured route from
// counting as a zero.
//
// LANE PURITY (same rule as judge.go / observe.go). This leaf stays stdlib-only.
// It owns the pure TAXONOMY + the pure coverage FOLD; the live binding that
// PRODUCES a provenance — calling a judge model to set VerifyJudge, or dos verify
// to set VerifyWitness — is the caller's job ABOVE this seam, exactly as Scorer is
// a bound closure and the gateway exporter is a copy-out of Counts().

import "sort"

// Verification names the provenance of an Outcome.Quality — HOW that quality
// signal was established, and therefore how much a reader may trust it when a
// lower tier served the route. It is a closed vocabulary; an unrecognised value is
// conservatively treated as unverified by every predicate below (fail-closed).
type Verification string

const (
	// VerifyNone is self-report: the serving agent's own quality claim with no
	// independent check. Forgeable; the least safe. It is the empty string so the
	// zero value of Outcome.Verify is VerifyNone — an outcome recorded without a
	// stated provenance is conservatively unchecked, never silently promoted.
	VerifyNone Verification = ""
	// VerifyJudge marks a quality a larger/judge model scored (the Scorer path,
	// judge.go). OBSERVED — a model opinion, more trusted than self-report but not
	// git-witnessed.
	VerifyJudge Verification = "judge"
	// VerifyWitness marks a quality DOS confirmed from git evidence (dos verify /
	// commit-audit): the task actually shipped/passed. WITNESSED, non-forgeable;
	// the safest tier and the only one fak controls end-to-end.
	VerifyWitness Verification = "witness"
)

// Trusted reports whether the provenance names a real INDEPENDENT check (a judge
// model or a DOS witness). Self-report and any unrecognised token are untrusted —
// the fail-closed rule that keeps an unchecked route out of the "verified" count.
func (v Verification) Trusted() bool { return v == VerifyJudge || v == VerifyWitness }

// Witnessed reports whether the provenance is the non-forgeable git-evidence tier
// (VerifyWitness). Only DOS-witnessed outcomes satisfy it; a judge score does not.
func (v Verification) Witnessed() bool { return v == VerifyWitness }

// Rank orders the provenance least- to most-trusted: 0 self-report (also every
// unrecognised token, fail-closed), 1 judge, 2 witness. A caller comparing two
// provenances (e.g. "did this route's check get stronger?") uses Rank so the
// ordering is explicit and unknowns sink to the bottom.
func (v Verification) Rank() int {
	switch v {
	case VerifyWitness:
		return 2
	case VerifyJudge:
		return 1
	default:
		return 0
	}
}

// Label is the human/metric label for the provenance: "self-reported", "judge",
// or "witness". An unrecognised token is passed through verbatim so a dump never
// hides it (the coverage math still treats it as untrusted); the empty default
// renders as "self-reported".
func (v Verification) Label() string {
	switch v {
	case VerifyNone:
		return "self-reported"
	case VerifyJudge:
		return "judge"
	case VerifyWitness:
		return "witness"
	default:
		return string(v)
	}
}

// VerificationCounts is the per-provenance rollup of served outcomes — the safety
// headline for a set of routes: how many outcomes were self-reported vs
// judge-scored vs DOS-witnessed, plus the derived trusted / witnessed totals. It
// answers "of the routes a lower tier served, how many were actually checked, and
// how many by non-forgeable evidence?". ByProvenance keys on the raw Verification
// so an unrecognised token stays visible; Trusted/Witnessed are computed via the
// fail-closed predicates, so an unknown token is counted but never trusted.
type VerificationCounts struct {
	Total        int                  `json:"total"`
	ByProvenance map[Verification]int `json:"by_provenance"`
	Trusted      int                  `json:"trusted"`   // judge or witness
	Witnessed    int                  `json:"witnessed"` // witness only
}

// SelfReported is the count with no independent check — Total minus the trusted
// count (so an unrecognised token folds in here, fail-closed).
func (c VerificationCounts) SelfReported() int { return c.Total - c.Trusted }

// Coverage is the trusted fraction in [0,1] — the share of outcomes an
// independent check (judge or witness) covered. It is the observability headline
// for "is this route served safely?". Zero when Total is zero (no outcomes ⇒ no
// coverage to report, never a divide-by-zero).
func (c VerificationCounts) Coverage() float64 {
	if c.Total == 0 {
		return 0
	}
	return float64(c.Trusted) / float64(c.Total)
}

// WitnessCoverage is the non-forgeable fraction in [0,1] — the share DOS
// git-evidence witnessed, ignoring judge opinions. It is always <= Coverage; the
// gap between them is the share resting on a model's word rather than git-truth.
func (c VerificationCounts) WitnessCoverage() float64 {
	if c.Total == 0 {
		return 0
	}
	return float64(c.Witnessed) / float64(c.Total)
}

// SortedProvenance returns the provenance keys in a stable order — most-trusted
// first (Rank descending), ties broken by label — so a CLI dump or a metric
// export over the rollup is deterministic across runs regardless of map order.
func (c VerificationCounts) SortedProvenance() []Verification {
	out := make([]Verification, 0, len(c.ByProvenance))
	for k := range c.ByProvenance {
		out = append(out, k)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Rank() != out[j].Rank() {
			return out[i].Rank() > out[j].Rank()
		}
		return out[i].Label() < out[j].Label()
	})
	return out
}

// foldVerification folds a set of outcome records into a VerificationCounts. Pure
// over the records: same inputs always yield the same counts. The trusted /
// witnessed totals are derived through the fail-closed predicates, so an absent or
// unrecognised provenance is counted in Total and ByProvenance but never in
// Trusted/Witnessed.
func foldVerification(records []OutcomeRecord) VerificationCounts {
	c := VerificationCounts{ByProvenance: map[Verification]int{}}
	for _, r := range records {
		v := r.Outcome.Verify
		c.Total++
		c.ByProvenance[v]++
		if v.Trusted() {
			c.Trusted++
		}
		if v.Witnessed() {
			c.Witnessed++
		}
	}
	return c
}

// VerificationCoverage folds the whole journal into one VerificationCounts — the
// overall safety headline across every served route. Pure fold, mirrors
// Aggregate: same recorded outcomes always yield the same counts.
func (j *OutcomeJournal) VerificationCoverage() VerificationCounts {
	return foldVerification(j.records)
}

// VerificationByKey folds the journal into per-(aspect,rule) VerificationCounts —
// so an observer can localise the risk: WHICH route buckets a lower tier serves
// with weak verification coverage, rather than one blended number that hides them.
// Pure fold, keyed exactly like Aggregate (an empty rule == the fail-closed
// default; an empty aspect == the un-aspected route).
func (j *OutcomeJournal) VerificationByKey() map[AspectRuleKey]VerificationCounts {
	byKey := make(map[AspectRuleKey][]OutcomeRecord)
	for _, r := range j.records {
		byKey[r.Key] = append(byKey[r.Key], r)
	}
	out := make(map[AspectRuleKey]VerificationCounts, len(byKey))
	for k, recs := range byKey {
		out[k] = foldVerification(recs)
	}
	return out
}

// RiskyBucket is one (aspect,rule) route bucket whose verification coverage falls
// below the caller's safety threshold — a place a lower tier is serving work that
// is mostly self-reported, where DOS/judge verification is not yet wired in. It
// carries the full VerificationCounts so a caller can filter further (e.g. by
// sample count) or render the shortfall.
type RiskyBucket struct {
	Key    AspectRuleKey      `json:"key"`
	Counts VerificationCounts `json:"counts"`
}

// RiskyBuckets returns the (aspect,rule) buckets whose Coverage() is STRICTLY
// below minCoverage — the lower-tier routes running mostly on self-report, the
// ones that need a DOS witness or a judge model wired in before they can be called
// safe. This is the "in certain cases" selector: it names exactly the buckets a
// verification policy must cover, instead of forcing verification on every route.
//
// minCoverage is clamped to [0,1]. The result is sorted worst-coverage-first, ties
// broken by aspect then rule, so the report is deterministic across runs. A bucket
// with no outcomes cannot appear (it is not in the journal); a caller that wants to
// ignore thin buckets filters the returned Counts.Total itself.
func (j *OutcomeJournal) RiskyBuckets(minCoverage float64) []RiskyBucket {
	if minCoverage < 0 {
		minCoverage = 0
	}
	if minCoverage > 1 {
		minCoverage = 1
	}
	byKey := j.VerificationByKey()
	out := make([]RiskyBucket, 0, len(byKey))
	for k, c := range byKey {
		if c.Coverage() < minCoverage {
			out = append(out, RiskyBucket{Key: k, Counts: c})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		ci, cj := out[i].Counts.Coverage(), out[j].Counts.Coverage()
		if ci != cj {
			return ci < cj
		}
		if out[i].Key.Aspect != out[j].Key.Aspect {
			return out[i].Key.Aspect < out[j].Key.Aspect
		}
		return out[i].Key.Rule < out[j].Key.Rule
	})
	return out
}
