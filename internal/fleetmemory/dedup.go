package fleetmemory

import "fmt"

// Publish outcomes. A write is either PUBLISHED (new canonical fact) or refused
// with DUP_LESSON — refuse-with-merge: it names the existing canonical entry and
// offers to strengthen it, never a silent drop. DUP_LESSON is the closed-
// vocabulary refusal reason this leaf adds (#2142).
const (
	OutcomePublished = "PUBLISHED"
	OutcomeDupLesson = "DUP_LESSON"
)

// PublishResult is the outcome of a write against the ledger.
type PublishResult struct {
	Outcome  string  // OutcomePublished | OutcomeDupLesson
	Lesson   Lesson  // the newly published lesson (PUBLISHED) or the candidate (DUP_LESSON)
	Existing *Lesson // on DUP_LESSON: the canonical entry to strengthen instead
	Reason   string  // human-readable refuse-with-merge / publish note
}

// Refused reports whether the write was refused as a duplicate.
func (r PublishResult) Refused() bool { return r.Outcome == OutcomeDupLesson }

// Publish writes a candidate lesson to the ledger, or refuses it with DUP_LESSON
// when a peer already covers the same fact (#2142). This is the write-time dedup
// that keeps the ledger to one canonical entry per fact instead of letting N
// agents each store the same scar.
//
// On a match it does NOT mutate the ledger and returns the existing canonical
// entry so the caller can strengthen it (add a witness, widen the trigger)
// rather than duplicate. On no match the candidate is appended and indexed, and
// a subsequent equivalent Publish will refuse against it.
func (l *Ledger) Publish(candidate Lesson) PublishResult {
	k := factKey(candidate.Fact)
	if k == "" {
		return PublishResult{
			Outcome: OutcomeDupLesson,
			Lesson:  candidate,
			Reason:  "DUP_LESSON: empty fact carries no dedup key — nothing to publish",
		}
	}
	if existing, ok := l.Match(candidate.Fact); ok {
		e := existing
		return PublishResult{
			Outcome:  OutcomeDupLesson,
			Lesson:   candidate,
			Existing: &e,
			Reason: fmt.Sprintf(
				"DUP_LESSON: fact already covered by lesson %q — strengthen it (add a witness / widen its trigger) instead of writing a duplicate",
				existing.ID),
		}
	}
	idx := len(l.lessons)
	l.lessons = append(l.lessons, candidate)
	l.byKey[k] = idx
	return PublishResult{
		Outcome: OutcomePublished,
		Lesson:  candidate,
		Reason:  "PUBLISHED: new canonical lesson",
	}
}
