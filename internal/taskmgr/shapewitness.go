package taskmgr

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/answershape"
)

// OutputRefKind is the EvidenceRef.Kind a ShapeWitness reads back: an EvidenceRef
// whose Kind is "output" carries, in its Ref field, the OUTPUT TEXT of a beat to be
// graded for shape. It is the semantic-failure analogue of PathWitness's "path"
// refs — but where a "path" ref POINTS at proof on disk, an "output" ref carries the
// proof inline (the bytes the process actually emitted), because the failure mode
// being witnessed is in the bytes themselves, not in some external artifact.
const OutputRefKind = "output"

// shapeWitnessSource is the WitnessRecord.Source stamp for a ShapeWitness verdict.
const shapeWitnessSource = "answershape"

// shapeRefusedVerdict is the fixed, payload-free headline a ShapeWitness writes into
// Verdict when it refuses a degenerate beat. The WHY (which sub-signal tripped, by
// how much) goes into Detail from the answershape reasons; the Verdict stays a short
// constant so a snapshot reader can group refusals without parsing the detail.
const shapeRefusedVerdict = "degenerate output (silent semantic failure)"

// maxEchoChars bounds the only beat-derived text echoed back into the record — the
// reason Detail — so a multi-megabyte degenerate payload (the very thing being
// refused) is never copied wholesale into the snapshot it taints. The answershape
// reasons are already short (each names one sub-signal with a ≤16-rune unit preview),
// but a runaway top-n-gram could still be long; this is the belt.
const maxEchoChars = 160

// ShapeWitness grades the CLAIMED OUTPUT TEXT of a beat — carried on the Claim as one
// or more EvidenceRef{Kind:"output"} refs — through internal/answershape, and refuses
// a degenerate (looping / runaway) beat as a silent semantic failure. It is the
// "healthy process, degraded output" witness: a task can be alive and beating while
// emitting garbage, and a liveness rung that only watches heartbeats cannot tell the
// two apart. ShapeWitness reads the actual emitted bytes back — a source the
// reporting process's own "I'm fine" claim does not author — exactly as PathWitness
// reads the filesystem, and returns:
//
//   - VerifiedRefused when a graded output is DEGENERATE (answershape Report.Degenerate
//     true). The answershape reason(s) go into Verdict/Detail so the refusal explains
//     WHY (which repetition/verbosity sub-signal tripped) without dumping the payload.
//   - VerifiedDone when every graded output is present and in-shape.
//   - VerifiedUnavailable when the Claim carries NO "output" ref to read back — there
//     is nothing to grade, so the claim is neither confirmed nor refused. It must
//     NEVER silently downgrade to VerifiedDone (the PathWitness "no path evidence"
//     rule): "I saw no output" is not "I saw good output".
//
// Limits are caller-tunable. A zero-value ShapeWitness is usable: effectiveLimits
// fills the repeat threshold and n-gram width from answershape's defaults so the
// witness grades meaningfully out of the box; MaxChars stays 0 (the verbosity check
// is opt-in, since a long-but-coherent answer is not a semantic failure).
//
// Fence: ShapeWitness only RECORDS a graded verdict beside the claimed State; it
// never overwrites the claim and never gates admission. Manager.WitnessTask preserves
// that separation — a refused beat leaves the task's claimed State standing, visible
// next to the verified_refused rung.
type ShapeWitness struct {
	// Limits are the answershape thresholds this witness grades against. The zero
	// value is valid; effectiveLimits supplies sane defaults for any unset knob.
	Limits answershape.Limits
}

// WitnessClaim grades every EvidenceRef{Kind:"output"} on c through answershape.
// Mirrors PathWitness.WitnessClaim: it counts the refs it can actually read back,
// refuses on the first degenerate one (most-restrictive-wins — a single garbage beat
// taints the claim), confirms when all are in-shape, and reports unavailable when
// there is nothing of the right kind to grade.
func (w ShapeWitness) WitnessClaim(c Claim) WitnessRecord {
	lim := w.effectiveLimits()
	rec := WitnessRecord{Source: shapeWitnessSource, EvidenceRefs: gradedRefs(c.Refs)}
	checked := 0
	for _, ref := range c.Refs {
		if ref.Kind != OutputRefKind {
			continue
		}
		checked++
		report := answershape.Measure([]byte(ref.Ref), lim)
		if report.Degenerate {
			rec.VerifiedState = VerifiedRefused
			rec.Verdict = shapeRefusedVerdict
			rec.Detail = boundText(strings.Join(report.Reasons, "; "))
			return rec
		}
	}
	switch {
	case checked == 0:
		rec.VerifiedState = VerifiedUnavailable
		rec.Detail = "no output evidence to grade"
	default:
		rec.VerifiedState = VerifiedDone
		rec.Detail = "output in shape (not degenerate)"
	}
	return rec
}

// effectiveLimits fills any unset knob on the witness's Limits from answershape's
// defaults, so a zero-value ShapeWitness still grades meaningfully. MaxRepeat <= 0
// would DISABLE the repetition check (answershape's contract), which would make the
// whole witness a no-op that confirms any text — so it defaults to DefaultMaxRepeat.
// NGram <= 0 defaults to DefaultNGram. MaxChars is left untouched: 0 keeps the
// verbosity check OFF unless the caller opts in, because verbose-but-coherent is not
// a semantic failure.
func (w ShapeWitness) effectiveLimits() answershape.Limits {
	lim := w.Limits
	if lim.MaxRepeat <= 0 {
		lim.MaxRepeat = answershape.DefaultMaxRepeat
	}
	if lim.NGram <= 0 {
		lim.NGram = answershape.DefaultNGram
	}
	return lim
}

// gradedRefs returns a record-safe copy of the claim's refs. An "output" ref carries
// the graded payload INLINE in its Ref, so copying it verbatim into the stored record
// would dump the (possibly huge, possibly degenerate) beat text into every snapshot
// that renders the witness. Instead the payload is dropped and replaced by its byte
// length alone — preserving the audit trail ("this much text was graded") without
// echoing a single byte of the payload back. The SHAPE that was found lives in the
// record's Detail (answershape's reasons, each already bounded to a ≤16-rune unit
// preview), so nothing is lost by not re-emitting the bytes here. Non-output refs (a
// "path", a "commit") point AT external proof and are copied through unchanged.
func gradedRefs(refs []EvidenceRef) []EvidenceRef {
	if len(refs) == 0 {
		return nil
	}
	out := make([]EvidenceRef, len(refs))
	for i, ref := range refs {
		if ref.Kind == OutputRefKind {
			ref.Note = fmt.Sprintf("graded %d bytes", len(ref.Ref))
			ref.Ref = ""
		}
		out[i] = ref
	}
	return out
}

// boundText renders s for an in-record reason/preview, capped at maxEchoChars runes
// (a trailing ellipsis marks truncation) and flattened of newlines so a multi-line
// degenerate block stays one readable line. It is the single chokepoint that keeps a
// graded payload from leaking unbounded into a WitnessRecord.
func boundText(s string) string {
	s = strings.ToValidUTF8(s, "")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\r", " ")
	if utf8.RuneCountInString(s) <= maxEchoChars {
		return s
	}
	r := []rune(s)
	return string(r[:maxEchoChars]) + "…"
}

// VerifiedProgressing reports whether the task is making REAL progress, as opposed to
// "advancing on garbage". It is a read-only DERIVED view over the claimed State and
// witness rungs beside it: it never mutates either. The answer is true UNLESS a
// task or child-step witness ran and refused.
//
// That single refused case is the "alive but emitting garbage" signal: a task can be
// StateRunning and beating (liveness says live) while a ShapeWitness has graded its
// output degenerate. Heartbeat liveness alone cannot distinguish a healthy task from
// a looping one; pairing running-and-beating with VerifiedProgressing()==false is
// what surfaces the silent semantic failure. A claimed-only task (no witness) and a
// task whose witness confirmed or could-not-read (done / unavailable / unknown) are
// all reported as progressing — only an affirmative refusal flips the bit, so the
// derived view never invents a problem the witness did not attest.
func (t TaskSnapshot) VerifiedProgressing() bool {
	if witnessRefused(t.Witness) {
		return false
	}
	for _, step := range t.Steps {
		if !step.VerifiedProgressing() {
			return false
		}
	}
	return true
}

// VerifiedProgressing reports whether the step is making REAL progress by the
// same refused-witness rule TaskSnapshot uses.
func (s StepSnapshot) VerifiedProgressing() bool {
	return !witnessRefused(s.Witness)
}

func witnessRefused(w *WitnessRecord) bool {
	return w != nil && w.VerifiedState == VerifiedRefused
}
