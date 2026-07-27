package modelroute

import "strings"

// CLASSIFYING A SUBJECT INTO A WORK CLASS (epic #5416, track A).
//
// Route() answers "which model", Place() answers "which rung" — but Place() needs a
// WorkClass and Route() consumes a Subject, and until now nothing joined the two. This
// is that joint, and it is the piece where an automatic placer is most tempted to
// cheat: it would be easy to look at a tool name, decide "grep is harmless", and route
// it to a laptop. That guess is exactly the judgment fak already refuses to make from a
// name — internal/adjudicator owns reversibility, sits a tier above this leaf, and is
// deliberately NOT duplicated here.
//
// So the rule is: a work class is DECLARED, or it is conservative. There is one
// exception, and it earns its place by being a declaration in disguise (AspectScout is
// defined as the cheap classify-first probe — the aspect names the work's nature).
// Complexity may only RATCHET the floor UP, never down.

// ClassLabel is the Subject label key an operator declares a work class in, e.g.
// Subject{Labels: {"work_class": "routine"}}. Manifest matches already carry labels
// (examples/model-routing.example.json matches {"domain": "legal"}), so this needs no
// schema change — a deployment declares placement intent the same way it declares
// anything else about a subject.
const ClassLabel = "work_class"

// Closed reason vocabulary for classification, rendered verbatim by a status surface.
const (
	ReasonClassDeclared           = "class-declared"                // an operator declared it
	ReasonClassFromAspect         = "class-from-aspect"             // the aspect IS the declaration (scout)
	ReasonClassUndeclared         = "class-undeclared-conservative" // nothing said; stay at the strictest floor
	ReasonClassLabelUnrecognized  = "class-label-unrecognized"      // a label was set, but not to a known class
	ReasonClassRaisedByComplexity = "class-raised-by-complexity"    // a high-complexity subject ratcheted the floor up
)

// Classification is the work class a Subject carries, whether an operator actually
// SAID so, and the closed reasons that produced it.
//
// Declared is reported rather than folded away because "we placed this conservatively
// because nobody told us what it was" and "we placed this conservatively because it is
// genuinely hard" are different operator problems with different fixes.
type Classification struct {
	Class    WorkClass `json:"class"`
	Declared bool      `json:"declared"`
	Reasons  []string  `json:"reasons"`
}

// ClassOf maps a routing Subject onto the work class whose risk floor governs it.
//
// The precedence is:
//
//  1. An explicit Labels["work_class"] naming a known class WINS. This is the operator
//     saying what the work is, and it is the only input allowed to place work on a
//     cheap rung.
//  2. AspectScout, absent a declaration, is ClassRoutine — the scout aspect is defined
//     as a cheap classify-first probe, so the aspect is itself the claim.
//  3. Everything else is UNDECLARED. ClassOf returns the empty class, which PolicyFor
//     already maps to the strictest floor (T0) with ReasonUnknownClass. No second
//     conservatism mechanism is added here: one gate, in one place, is why the
//     behaviour is predictable.
//
// A label set to something unrecognized is NOT treated as a declaration — a typo in
// "routine" must not read as permission to use a laptop. It reports
// ReasonClassLabelUnrecognized so the operator can see their config did nothing.
//
// ComplexityHigh then ratchets a routine classification up to normal-impl. The ratchet
// is one-way by construction: it can only ever make the floor stricter, so a subject
// that lies about its complexity in the cheap direction gains nothing.
func ClassOf(s Subject) Classification {
	c := Classification{}

	if raw, ok := s.Labels[ClassLabel]; ok {
		if cls, known := parseWorkClass(raw); known {
			c.Class, c.Declared = cls, true
			c.Reasons = append(c.Reasons, ReasonClassDeclared)
		} else {
			c.Reasons = append(c.Reasons, ReasonClassLabelUnrecognized)
		}
	}

	if !c.Declared {
		if s.Aspect == AspectScout {
			c.Class, c.Declared = ClassRoutine, true
			c.Reasons = append(c.Reasons, ReasonClassFromAspect)
		} else {
			c.Reasons = append(c.Reasons, ReasonClassUndeclared)
			return c
		}
	}

	// The one-way ratchet. Raising the floor is always safe; lowering it never is, so
	// only the raising direction exists.
	if s.Complexity == ComplexityHigh && c.Class == ClassRoutine {
		c.Class = ClassNormalImpl
		c.Reasons = append(c.Reasons, ReasonClassRaisedByComplexity)
	}
	return c
}

// parseWorkClass parses a declared class token against the closed vocabulary,
// tolerating surrounding whitespace and case. It is the one place the string form of
// the class vocabulary is read.
func parseWorkClass(raw string) (WorkClass, bool) {
	switch WorkClass(strings.ToLower(strings.TrimSpace(raw))) {
	case ClassUltraHard:
		return ClassUltraHard, true
	case ClassNormalImpl:
		return ClassNormalImpl, true
	case ClassRoutine:
		return ClassRoutine, true
	case ClassSecurityRelease:
		return ClassSecurityRelease, true
	}
	return "", false
}

// PlaceSubject is the composed answer to "where should THIS run": classify the subject,
// then walk the zone ladder for the class it carries.
//
// It is the call a dispatch seam makes. The classification rides back on the Placement
// so the operator sees both halves of the decision — an undeclared subject that lands
// on a vendor reports ReasonClassUndeclared next to escalated-past-cheaper-zone, which
// tells them the fix is a label, not more hardware.
func (r Roster) PlaceSubject(s Subject, candidates []Candidate) (Placement, Classification, error) {
	cls := ClassOf(s)
	p, err := r.Place(cls.Class, candidates)
	return p, cls, err
}
