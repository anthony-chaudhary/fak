package dispatchtick

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// WHAT KIND OF WORK DID THIS SLOT DO? (#5416 track F — the producer's missing half.)
//
// TurnOutcomesFromWitness deliberately takes its Class hook with NO DEFAULT: a turn
// outcome filed under the wrong work class is not a small error, because
// modelroute.PolicyFor maps an unrecognized class to the T0 floor. That mapping is the
// right conservatism when picking a floor for WORK ("we don't know what this is, so
// assume the worst") and a capability-MINTING hole when read backwards to grade a MODEL
// ("this model succeeded at T0 work"). So the producer refuses to guess and this file is
// the one place that answers the question — from what an operator actually DECLARED.
//
// The signal is the issue's own tier labels, read through the SAME parser that chose the
// slot's launch profile (LaunchBucketForIssue). Sharing the parser is the point: if the
// grading path re-derived the class from labels independently, a model could be launched
// as one class and graded as another, and the drift would be invisible in both payloads.
//
// Two answers here are deliberately EMPTY, and the fold drops an empty class:
//
//   - An untagged issue grades nothing. It is tempting to call untagged work "routine"
//     — it is usually small — but "usually" is how a 4B laptop model earns a rung from a
//     backlog nobody triaged. The tick-wide work kind (LaunchProfileForDispatch's PM
//     fallback) is deliberately NOT consulted either: it declares what the LOOP is, not
//     what this ISSUE was.
//   - A coordination slot grades nothing. BucketPM work is triage, planning, and
//     gardening — real work, and not evidence about implementing anything. Mapping it to
//     the cheapest class would let a model earn the routine IMPLEMENTATION rung by
//     labelling issues, which is the same minting hole in a friendlier costume. The
//     vocabulary has no coordination class to grade it into, so the refusal is named
//     rather than silent, and an operator can see how much evidence it costs.
//
// Nothing here ever mints modelroute.ClassSecurityRelease: no dispatch label declares
// security/release/destructive work, so no slot can be graded into the class whose floor
// exists to stop a cheap model from serving it.

// ClassAttribution is the closed vocabulary for why a slot's work class is or is not
// known. The two failures are different missing pieces with different fixes — one is a
// triage gap, the other is a vocabulary gap — so they are never summed into one
// "unclassified".
type ClassAttribution string

const (
	// ClassFromTierLabel: the issue carried a tier/ultra/T<N> label and it named a class.
	ClassFromTierLabel ClassAttribution = "tier-label"
	// ClassNoTierLabel: the issue carried no trusted tier signal, so nothing declared what
	// kind of work it was. The fix is triage — label the backlog.
	ClassNoTierLabel ClassAttribution = "no-tier-label"
	// ClassCoordinationBucket: the issue resolved to the project-management bucket, which
	// the work-class vocabulary cannot express. The fix would be a coordination class, not
	// a cheaper mapping of this one.
	ClassCoordinationBucket ClassAttribution = "coordination-bucket"
)

// bucketWorkClass maps the closed launch-bucket vocabulary onto the closed work-class
// vocabulary. Both T0 buckets land on ultra-hard: tier/ultra is a promotion WITHIN T0
// (the tier vocabulary has no level beyond it), not a fifth class.
//
// BucketPM is absent BY DESIGN — see the file header. Its absence is what makes a
// coordination slot fall through to a named refusal instead of a class.
var bucketWorkClass = map[LaunchBucket]modelroute.WorkClass{
	BucketRoutine: modelroute.ClassRoutine,
	BucketNormal:  modelroute.ClassNormalImpl,
	BucketHard:    modelroute.ClassUltraHard,
	BucketUltra:   modelroute.ClassUltraHard,
}

// WorkClassForIssue names the work class an issue's outcome may be graded under, or
// returns the empty class with the reason it cannot be graded.
//
// An empty class is never an error and never a default — it is the answer "this slot
// produced no capability evidence", which the fold counts and drops.
func WorkClassForIssue(labels []string) (modelroute.WorkClass, ClassAttribution) {
	bucket, ok := LaunchBucketForIssue(labels)
	if !ok {
		return "", ClassNoTierLabel
	}
	class, mapped := bucketWorkClass[bucket]
	if !mapped {
		return "", ClassCoordinationBucket
	}
	return class, ClassFromTierLabel
}

// ClassResolver adapts a per-issue label lookup into the Class hook
// TurnOutcomesFromWitness takes.
//
// A nil lookup yields a resolver that classes every record empty rather than panicking or
// falling back to a class: a caller that has no label source has no declaration to read,
// and the honest result of that is zero evidence, not evidence at the strictest floor.
func ClassResolver(labels func(issue int) []string) func(WitnessRecord) modelroute.WorkClass {
	return func(r WitnessRecord) modelroute.WorkClass {
		if labels == nil {
			return ""
		}
		class, _ := WorkClassForIssue(labels(r.Issue))
		return class
	}
}

// ClassTally is how much of a sweep could be graded at all, and why the rest could
// not. It is the class-side twin of ZoneShare: a capability grade assembled from 8 of 200
// finished slots is not wrong, but an operator who cannot see the 192 will read it as if
// it described the fleet.
type ClassTally struct {
	// Total is every record folded.
	Total int
	// Classified is the number of records that named a work class.
	Classified int
	// ByClass counts the classified records per class.
	ByClass map[modelroute.WorkClass]int
	// Unclassified counts the rest by which declaration was missing. It never contains
	// ClassFromTierLabel.
	Unclassified map[ClassAttribution]int
}

// FoldClassTally counts a witness sweep by work class using a per-issue label lookup.
// A nil lookup reports every record unclassified for want of a tier label, which is the
// truth about a caller that supplied no labels — the sweep saw no declaration.
func FoldClassTally(records []WitnessRecord, labels func(issue int) []string) ClassTally {
	c := ClassTally{
		Total:        len(records),
		ByClass:      map[modelroute.WorkClass]int{},
		Unclassified: map[ClassAttribution]int{},
	}
	for _, r := range records {
		var (
			class modelroute.WorkClass
			why   = ClassNoTierLabel
		)
		if labels != nil {
			class, why = WorkClassForIssue(labels(r.Issue))
		}
		if class == "" {
			c.Unclassified[why]++
			continue
		}
		c.ByClass[class]++
		c.Classified++
	}
	return c
}

// Reasons lists the distinct reasons records went unclassified, in a stable order, so a
// status line renders identically across runs.
func (c ClassTally) Reasons() []ClassAttribution {
	out := make([]string, 0, len(c.Unclassified))
	for k := range c.Unclassified {
		out = append(out, string(k))
	}
	sort.Strings(out)
	reasons := make([]ClassAttribution, 0, len(out))
	for _, k := range out {
		reasons = append(reasons, ClassAttribution(k))
	}
	return reasons
}
