package modelroute

// Scout-model live classification (#599, epic #595).
//
// THE RESERVED PATTERN, MADE LIVE. The spine already reserves the classify-first
// shape in three places — AspectScout (the cheap probe's granularity), Plan.Scout
// (the model id a matched plan names as its scout), and DecisionRecord.ScoutCalls
// (the per-decision counter). What was missing is the LIVE step that closes the
// loop: run a cheap classifier over the subject, parse its label into the
// Subject's complexity/domain signals, then re-Route on the now-populated subject
// so a rule the static signals alone could not match can fire.
//
// THE SEAM IS AN INTERFACE, NOT A HARDCODED MODEL. The scout classifier is
// injected exactly the way the best_of judge is (judge.go's Scorer): a Classifier
// interface plus a ClassifierFunc adapter, so the live wiring binds it to a real
// cheap-model call (a gateway turn, a `fak run` of a small model) while a test
// binds a deterministic stand-in. This leaf never runs an engine itself — it owns
// the WIRING (scout-call -> parse label -> populate Subject -> re-Route), and the
// classifier owns the model call.
//
// HONESTY (load-bearing, from the issue): a scout is an EXTRA model call with
// cost. Three invariants keep that honest:
//   - The scout decision is OBSERVABLE. ScoutRoute returns a ScoutOutcome that
//     carries BOTH the route the static signals alone would have taken (Before)
//     and the route taken after the scout populated the subject (After), plus the
//     label and whether the scout actually changed the matched rule — so the
//     overhead is never an unbenchmarked "near-zero latency" headline; the caller
//     can price it and prove the scout earned its call.
//   - The NO-SCOUT path is unchanged. Routing without a scout is still
//     Manifest.Route over the caller's subject; ScoutRoute only ADDS a step, and
//     a manifest/plan that names no scout never invokes the classifier (the
//     classifier call is gated on a non-empty scout model, never unconditional).
//   - A scout NEVER fabricates a signal it cannot justify. The classifier returns
//     a ScoutLabel whose Complexity must be a closed-vocabulary value (or "" to
//     leave it unset); an out-of-vocabulary complexity is a fail-loud error, not a
//     silent guess — the same fail-closed discipline the manifest validator uses.
//
// PURE BY CONSTRUCTION except the injected call: ApplyScoutLabel (label ->
// Subject) and the Before/After framing are deterministic; the only impurity is
// the Classifier the caller binds, which carries its own context for
// cancellation/deadline.

import (
	"context"
	"fmt"
	"time"
)

// ScoutLabel is the parsed verdict a scout classifier returns for a subject: the
// coarse Complexity it assigns and the OPEN Labels (domain, language, taint, …) it
// observed. Both are OPTIONAL — a scout that can only judge complexity leaves
// Labels nil, and one that can only tag a domain leaves Complexity "". The label
// is the structured form of the cheap model's answer; ApplyScoutLabel folds it
// into a Subject, and the Subject's static fields are only ever ADDED to, never
// overwritten with a worse signal (see ApplyScoutLabel).
type ScoutLabel struct {
	Complexity Complexity        `json:"complexity,omitempty"`
	Labels     map[string]string `json:"labels,omitempty"`
}

// Valid reports whether the label is well-formed: its Complexity must be one of
// the closed vocabulary (or "" to leave the signal unset). Labels are OPEN — any
// key/value is allowed, mirroring Subject.Labels — so only Complexity is checked.
// An invalid label is refused at the boundary so a scout can never inject an
// out-of-vocabulary complexity that would silently never match a MinComplexity
// floor.
func (l ScoutLabel) Valid() bool { return validComplexity(l.Complexity) }

// Classifier is the bound scout-model call — the injected seam that keeps any
// engine or provider client out of this leaf's import graph, exactly as judge.go's
// Scorer does for best_of. Given the subject to route it returns the cheap model's
// ScoutLabel; the context carries cancellation/deadline to the live call. The
// caller binds this to a real cheap-model invocation (a gateway turn, a `fak run`
// of a small model); a test binds a fixed-label stand-in. ScoutRoute never runs an
// engine itself — the Classifier owns the model call, this leaf owns the wiring.
type Classifier interface {
	Classify(ctx context.Context, s Subject) (ScoutLabel, error)
}

// ClassifierFunc adapts a plain func to the Classifier interface, so a caller can
// bind a closure (a gateway turn, a `fak run` call, a test stub) without declaring
// a named type — the same adapter idiom as ScorerFunc.
type ClassifierFunc func(ctx context.Context, s Subject) (ScoutLabel, error)

// Classify satisfies Classifier.
func (f ClassifierFunc) Classify(ctx context.Context, s Subject) (ScoutLabel, error) {
	return f(ctx, s)
}

// ApplyScoutLabel folds a scout's label into a subject and returns the ENRICHED
// subject, leaving the input untouched (pure — the caller keeps its original for
// the Before/After comparison). It is deliberately ADDITIVE and never destructive:
//   - Complexity is filled ONLY when the subject left it unset (ComplexityAny);
//     a subject that already carries a complexity keeps it, so a scout cannot
//     downgrade a caller's explicit signal.
//   - Each scout Label is set ONLY when the subject has no value for that key, so
//     a scout enriches the open signals without clobbering an existing taint/tenant
//     label the caller set deliberately.
//
// This additive rule is what makes the no-scout path observably unchanged: a fully
// pre-classified subject is returned byte-for-byte equal, so routing it with or
// without the scout produces the identical decision.
func ApplyScoutLabel(s Subject, l ScoutLabel) Subject {
	out := s
	if out.Complexity == ComplexityAny && l.Complexity != ComplexityAny {
		out.Complexity = l.Complexity
	}
	if len(l.Labels) > 0 {
		merged := make(map[string]string, len(s.Labels)+len(l.Labels))
		for k, v := range s.Labels {
			merged[k] = v
		}
		for k, v := range l.Labels {
			if _, ok := merged[k]; !ok {
				merged[k] = v
			}
		}
		out.Labels = merged
	}
	return out
}

// ScoutOutcome is the OBSERVABLE result of a scouted route: the original subject,
// the route the STATIC signals alone would have taken (Before), the scout's label,
// the enriched subject, the route taken AFTER the scout populated it (After), and
// the ScoutCalls counter (1 — the scout is an extra model call, priced here, never
// hidden). Changed reports whether the scout actually moved the route to a
// different rule — the witness that the extra call earned its cost. A caller folds
// After.Decision into the normal pipeline; the Before/Changed fields let it price
// and audit the scout, so a scout is never an unbenchmarked latency headline.
type ScoutOutcome struct {
	Subject    Subject    // the caller's original subject
	Before     Decision   // route without the scout (static signals only)
	Label      ScoutLabel // the scout classifier's verdict
	Enriched   Subject    // the subject after ApplyScoutLabel
	After      Decision   // route after the scout populated the subject
	ScoutCalls int        // model calls the scout made (1) — the priced overhead
}

// Changed reports whether the scout moved the route: the matched rule (or
// default-vs-matched) differs between the static route and the scouted route. A
// false here means the extra classifier call did NOT change the decision — exactly
// the over-cost a caller wants to see and tune away, never a hidden cost.
func (o ScoutOutcome) Changed() bool {
	return o.Before.RuleName != o.After.RuleName || o.Before.Matched != o.After.Matched
}

// Record builds the after-the-fact DecisionRecord for the SCOUTED decision under a
// manifest version, with the scout call counted: it reuses RecordDecision for the
// After decision and then forces ScoutCalls to reflect the live scout step (the
// After plan may not itself name a Scout, but a live scout call WAS made, so the
// observable overhead must show it). overhead is the routing+scout time the caller
// measured; pass 0 when not measured.
func (o ScoutOutcome) Record(version string, overhead time.Duration) DecisionRecord {
	rec := RecordDecision(version, o.After, overhead)
	if o.ScoutCalls > rec.ScoutCalls {
		rec.ScoutCalls = o.ScoutCalls
	}
	return rec
}

// ScoutRoute runs the live classify-first step: it classifies the subject with the
// injected scout, folds the label into the subject, and re-Routes the enriched
// subject — returning a ScoutOutcome that carries both the static (Before) and
// scouted (After) decisions so the scout's effect is observable and priceable.
//
// The classifier is REQUIRED and its label must be Valid — a nil classifier, a
// classifier error, or an out-of-vocabulary complexity is a fail-loud error, never
// a silent fall-through to the static route. (A caller that wants the no-scout
// path simply calls Manifest.Route directly; ScoutRoute is the path that PAYS for
// a scout, so it must prove the scout ran cleanly.)
//
// The no-scout invariant is structural: ScoutRoute always charges exactly one
// scout call (ScoutCalls == 1) and the Before decision is the unmodified
// Manifest.Route over the caller's subject — so a caller can diff Before vs After
// to confirm the scout, and only the scout, changed the route.
func (m Manifest) ScoutRoute(ctx context.Context, scout Classifier, s Subject) (ScoutOutcome, error) {
	if scout == nil {
		return ScoutOutcome{}, fmt.Errorf("modelroute: ScoutRoute needs a bound Classifier (the scout model call)")
	}
	before := m.Route(s)
	label, err := scout.Classify(ctx, s)
	if err != nil {
		return ScoutOutcome{}, fmt.Errorf("modelroute: scout classify: %w", err)
	}
	if !label.Valid() {
		return ScoutOutcome{}, fmt.Errorf("modelroute: scout returned out-of-vocabulary complexity %q", label.Complexity)
	}
	enriched := ApplyScoutLabel(s, label)
	after := m.Route(enriched)
	return ScoutOutcome{
		Subject:    s,
		Before:     before,
		Label:      label,
		Enriched:   enriched,
		After:      after,
		ScoutCalls: 1,
	}, nil
}
