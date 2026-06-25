package taskmgr

import "strings"

// The default concept vocabulary. A step's Concept is free-form, but these are the
// canonical buckets so process snapshots compare cleanly across serve, guard,
// benches, and demos. Each maps to an existing fak concept:
//
//	observe    - admission / session decision, status sampling
//	adjudicate - guard adjudication of a tool call or request
//	tool       - tool-call execution and result admission
//	model      - an upstream model / provider request
//	cache      - context-cache reuse and prefix accounting
//	io         - disk, network, or other blocking I/O
//	verify     - verification of a claim or effect (a trust rung)
//	wait       - intentional waiting on an external signal
//	other      - anything that does not fit a named bucket
//
// The vocabulary is runtime accounting only. It carries no security trust and no
// completion truth: a step tagged verify has not been verified, it merely spent
// time in verification. Callers may still pass a custom concept for local
// experiments; NormalizeConcept lets such values pass through unchanged.
const (
	ConceptObserve    = "observe"
	ConceptAdjudicate = "adjudicate"
	ConceptTool       = "tool"
	ConceptModel      = "model"
	ConceptCache      = "cache"
	ConceptIO         = "io"
	ConceptVerify     = "verify"
	ConceptWait       = "wait"
	ConceptOther      = "other"
)

// defaultConcepts is the canonical vocabulary in stable display order.
var defaultConcepts = []string{
	ConceptObserve,
	ConceptAdjudicate,
	ConceptTool,
	ConceptModel,
	ConceptCache,
	ConceptIO,
	ConceptVerify,
	ConceptWait,
	ConceptOther,
}

var defaultConceptSet = func() map[string]struct{} {
	m := make(map[string]struct{}, len(defaultConcepts))
	for _, c := range defaultConcepts {
		m[c] = struct{}{}
	}
	return m
}()

// DefaultConcepts returns a copy of the canonical concept vocabulary in stable
// order. The returned slice is the caller's to mutate; it does not alias package
// state.
func DefaultConcepts() []string {
	out := make([]string, len(defaultConcepts))
	copy(out, defaultConcepts)
	return out
}

// IsDefaultConcept reports whether concept is a member of the canonical
// vocabulary. The check is exact: pass a value already lowercased and trimmed, or
// run it through NormalizeConcept first.
func IsDefaultConcept(concept string) bool {
	_, ok := defaultConceptSet[concept]
	return ok
}

// NormalizeConcept maps a caller-supplied concept onto the value that should be
// recorded so snapshots aggregate consistently:
//
//   - whitespace is trimmed;
//   - an empty concept becomes "other" rather than a blank bucket;
//   - a value that matches the default vocabulary case-insensitively is folded to
//     its canonical lowercase form (so "Verify" and "verify" aggregate together);
//   - any other non-empty value passes through trimmed, preserving its case, so
//     custom concepts remain usable for local experiments.
func NormalizeConcept(concept string) string {
	trimmed := strings.TrimSpace(concept)
	if trimmed == "" {
		return ConceptOther
	}
	if lower := strings.ToLower(trimmed); IsDefaultConcept(lower) {
		return lower
	}
	return trimmed
}

// StartConceptStep starts a step after normalizing its Concept through
// NormalizeConcept, so callers get a consistent bucket without hand-typing the
// canonical string. All other StepSpec fields are passed through unchanged.
func (t *Task) StartConceptStep(spec StepSpec) (*Step, error) {
	spec.Concept = NormalizeConcept(spec.Concept)
	return t.StartStep(spec)
}

// StartObserveStep, StartModelStep, StartToolStep, and StartVerifyStep are
// convenience constructors for the most common served-request phases. Each applies
// the matching default concept; reach for StartConceptStep when you need to set
// Total, Unit, or Labels as well.
func (t *Task) StartObserveStep(stepID, title string) (*Step, error) {
	return t.StartConceptStep(StepSpec{StepID: stepID, Title: title, Concept: ConceptObserve})
}

func (t *Task) StartModelStep(stepID, title string) (*Step, error) {
	return t.StartConceptStep(StepSpec{StepID: stepID, Title: title, Concept: ConceptModel})
}

func (t *Task) StartToolStep(stepID, title string) (*Step, error) {
	return t.StartConceptStep(StepSpec{StepID: stepID, Title: title, Concept: ConceptTool})
}

func (t *Task) StartVerifyStep(stepID, title string) (*Step, error) {
	return t.StartConceptStep(StepSpec{StepID: stepID, Title: title, Concept: ConceptVerify})
}
