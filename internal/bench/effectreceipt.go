// effectreceipt.go binds preflight, mutation, and rollback to ONE immutable
// subject-set receipt (issue #5683, a leaf of the #4509 "missing middle" quality
// ladder). It answers a question a benchmark harness cannot otherwise answer: did
// this arm change exactly the resources whose prior state it observed, and did it
// put every one of them back?
//
// The defect it closes is scope widening between observation and effect. A harness
// that captures the "before" of one subject and then mutates a wider set has
// authorized an unscoped change from a scoped observation: the compensating
// rollback then restores resources it never examined, using assumptions derived
// from the subjects it did. So the pre-state capture emits an opaque SubjectReceipt
// and EVERY later stage consumes it — the mutation walks the receipt's subject set,
// not the caller's plan, and a plan that no longer binds to the receipt is refused
// BEFORE anything is applied.
//
// Five typed stages are managed distinctly (never collapsed into "it worked"):
//
//	BEFORE            — pre-state, independently read
//	REQUESTED         — what was asked for; ACCEPTANCE only, never proof of effect
//	EFFECTIVE         — independent readback proving the change actually took
//	RESTORE_ATTEMPTED — the compensation request; acceptance only
//	RESTORED          — independent post-action readback proving the state came back
//
// Two rules keep it honest. First, a benchmark is not complete when measurements
// finish: compensation runs after success, refusal, measurement error, and
// cancellation alike, and ineffective application, failed restoration, and an
// unreadable post-state are first-class outcomes that primary success cannot hide
// (Success() is true for exactly ONE outcome). Second, an adapter is never believed
// about its own effect — every write is followed by an independent read, so an
// "accepted" request that silently did nothing is still caught.
//
// Restoration compares under a DECLARED per-subject rule, because dynamic fields
// need not return byte-for-byte: a subject is either `exact` (must come back
// identical) or `semantic` (compared after a named normalizer). The rule is
// published in the receipt so an auditor reads the policy, not the implementation.
//
// A sequence of arms carries the dirt forward: an arm whose subjects a previous arm
// left changed or UNKNOWN is refused rather than run, so a later comparison rests on
// verified clean state rather than on cleanup convention.
//
// The receipt is scrubbed by construction — values appear only as subject-salted
// digests, faults only as a closed typed vocabulary, never as free adapter text — so
// it is publishable without leaking the environment it describes. It is also
// deterministic (no clock, no randomness, no map iteration in output):
//
//	go test ./internal/bench -run TestEffectReceipt -count=1
//
// (the golden artifact regenerates into testdata/effect_receipt.json with
// UPDATE_GOLDEN=1).
package bench

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// The five typed stages of one effect transaction (done-when 7). Each is recorded
// separately so acceptance, observed effect, compensation attempt, and observed
// restored state can never collapse into a single "it worked".
const (
	StageBefore           = "BEFORE"
	StageRequested        = "REQUESTED"
	StageEffective        = "EFFECTIVE"
	StageRestoreAttempted = "RESTORE_ATTEMPTED"
	StageRestored         = "RESTORED"
)

// Restoration equivalence rules. A subject declares which one governs it, and the
// declaration is published in the receipt: dynamic fields need not return
// byte-for-byte, but which fields those are is policy, not a runtime accident.
const (
	EquivalenceExact    = "exact"    // must come back byte-for-byte identical
	EquivalenceSemantic = "semantic" // compared after the subject's declared normalizer
)

// The closed outcome vocabulary. Exactly one is assigned per arm, by the precedence
// in effectOutcome: a definite state fault outranks an indefinite one, a pre-mutation
// refusal outranks a measurement result, and OutcomeRestored is reachable only when
// the post-action readback proved every subject came back.
const (
	// OutcomeRestored is the ONLY success: the measurement ran and every subject was
	// independently observed back at its captured value under its declared rule.
	OutcomeRestored = "restored"
	// OutcomeMeasurementFailed: the benchmark body failed, state verified clean.
	OutcomeMeasurementFailed = "measurement_failed"
	// OutcomeCancelled: the context was cancelled; compensation still ran.
	OutcomeCancelled = "cancelled"
	// OutcomeApplyRejected: the adapter refused the requested change (no effect).
	OutcomeApplyRejected = "apply_rejected"
	// OutcomeIneffective: the request was ACCEPTED but the readback proves the change
	// never took — the measurement would have benchmarked the unmutated state.
	OutcomeIneffective = "ineffective_application"
	// OutcomeRestoreFailed: post-action readback proves a subject did NOT come back.
	OutcomeRestoreFailed = "restore_failed"
	// OutcomePostStateUnreadable: the post-action state could not be read, so
	// restoration is UNKNOWN — never assumed clean.
	OutcomePostStateUnreadable = "post_state_unreadable"
	// OutcomePreStateUnreadable: a subject's pre-state could not be captured, so no
	// mutation is authorized (there would be nothing to roll back to).
	OutcomePreStateUnreadable = "pre_state_unreadable"
	// OutcomeSubjectSetWidened: the arm's subjects do not bind to the receipt the
	// pre-state capture produced — a scoped observation may not authorize an
	// unscoped change. Refused before anything is applied.
	OutcomeSubjectSetWidened = "subject_set_widened"
	// OutcomePredecessorUnclean: a previous arm left one of this arm's subjects
	// changed or unknown, so this arm's measurement would be contaminated.
	OutcomePredecessorUnclean = "predecessor_state_unclean"
)

// Per-subject dispositions after the independent post-action readback. UNKNOWN is
// never folded into "clean": an unreadable post-state is a real outcome.
const (
	SubjectRestored = "restored"
	SubjectChanged  = "changed"
	SubjectUnknown  = "unknown"
)

// The closed typed fault vocabulary. The receipt records fault CLASSES, never the
// adapter's free-text error, so it cannot leak an environmental value through a
// message that happens to quote one.
const (
	FaultReadFailed       = "read_failed"
	FaultApplyRejected    = "apply_rejected"
	FaultRestoreRejected  = "restore_rejected"
	FaultPostReadFailed   = "post_read_failed"
	FaultEffectReadFailed = "effect_read_failed"
)

// Sequence verdicts.
const (
	VerdictEffectClean   = "effect_transaction_clean"
	VerdictEffectFlagged = "effect_transaction_flagged"
)

// Adapter is the port a harness implements to mutate environmental or runtime
// state. NONE of its returns are trusted as proof of effect: the driver reads back
// independently after every write, so an adapter that accepts a request and does
// nothing is caught rather than believed.
type Adapter interface {
	// Read observes a subject's current value. An error means the value is
	// UNREADABLE — it never means "absent" and never means "unchanged".
	Read(subject string) (string, error)
	// Apply REQUESTS a change. A nil error is request acceptance only.
	Apply(subject, value string) error
	// Restore REQUESTS the compensating change back to the captured value. A nil
	// error is acceptance only; the driver reads back before emitting RESTORED.
	Restore(subject, value string) error
}

// SubjectMutation declares one subject an arm intends to mutate and the rule its
// restoration is judged under.
type SubjectMutation struct {
	// Subject names the resource (an env var, a runtime knob, a lease key).
	Subject string
	// Requested is the value the arm asks for.
	Requested string
	// Equivalence is EquivalenceExact or EquivalenceSemantic ("" means exact).
	Equivalence string
	// Normalizer is the published label for the semantic rule ("" means identity).
	Normalizer string
	// Normalize projects a value onto its semantically significant part. It is
	// consulted only for EquivalenceSemantic subjects; nil means identity.
	Normalize func(string) string
}

func (p SubjectMutation) rule() string {
	if p.Equivalence == EquivalenceSemantic {
		return EquivalenceSemantic
	}
	return EquivalenceExact
}

func (p SubjectMutation) normalizerLabel() string {
	if p.rule() != EquivalenceSemantic {
		return "identity"
	}
	if strings.TrimSpace(p.Normalizer) == "" {
		return "identity"
	}
	return p.Normalizer
}

// equivalent reports whether got may stand in for want under this subject's
// declared rule.
func (p SubjectMutation) equivalent(want, got string) bool {
	if p.rule() == EquivalenceSemantic && p.Normalize != nil {
		return p.Normalize(want) == p.Normalize(got)
	}
	return want == got
}

// EquivalenceDecl is the published restoration policy for one subject (done-when
// 10): which fields must return exactly, and which are judged semantically.
type EquivalenceDecl struct {
	Subject    string `json:"subject"`
	Rule       string `json:"rule"`
	Normalizer string `json:"normalizer"`
}

// EffectArm is one benchmark arm that temporarily mutates state.
type EffectArm struct {
	Name     string
	Subjects []SubjectMutation
	// Measure is the benchmark body. It runs ONLY after every subject's change has
	// been independently observed effective, and its failure never suppresses
	// compensation.
	Measure func(context.Context) error
}

// SubjectReceipt is the opaque, immutable subject-set receipt produced by the
// pre-state capture. It is the ONLY authorization the mutation and the compensation
// consume: both walk its subject set, so neither can reach a resource the capture
// did not observe.
//
// Its identity (id) is derived from the subject set AND the captured values and is
// deliberately never published — the artifact carries only the non-reversible
// subject-set binding, which proves the five stages agree on scope without
// disclosing what the receipt holds.
type SubjectReceipt struct {
	id       string
	binding  string
	order    []string
	plans    map[string]SubjectMutation
	before   map[string]string
	readable map[string]bool
}

// Binding is the published subject-set binding every stage carries.
func (r *SubjectReceipt) Binding() string {
	if r == nil {
		return effectBinding(nil)
	}
	return r.binding
}

// covers reports whether the receipt authorizes touching this subject at all.
func (r *SubjectReceipt) covers(subject string) bool {
	if r == nil {
		return false
	}
	_, ok := r.plans[subject]
	return ok
}

// complete reports whether every subject's pre-state was actually captured. A
// subject with no readable "before" has nothing to roll back to, so no mutation of
// the set is authorized.
func (r *SubjectReceipt) complete() bool {
	if r == nil {
		return false
	}
	for _, s := range r.order {
		if !r.readable[s] {
			return false
		}
	}
	return len(r.order) > 0
}

// SubjectState is one subject's record inside one stage. Values appear only as
// subject-salted digests; faults only as the closed typed vocabulary.
type SubjectState struct {
	Subject          string `json:"subject"`
	Readable         bool   `json:"readable"`
	ValueDigest      string `json:"value_digest"`
	NormalizedDigest string `json:"normalized_digest,omitempty"`
	// Request records adapter ACCEPTANCE ("accepted"/"rejected") — never effect.
	Request string `json:"request,omitempty"`
	// Effect records the INDEPENDENTLY OBSERVED result of the request.
	Effect string `json:"effect,omitempty"`
	// Restoration is the post-action disposition: restored, changed, or unknown.
	Restoration string `json:"restoration,omitempty"`
	// Fault is a typed class from the closed vocabulary, never adapter text.
	Fault string `json:"fault,omitempty"`
}

// StageRecord is one typed stage of the transaction, bound to the subject-set
// receipt and flagged with whether its values were independently observed.
type StageRecord struct {
	Stage   string `json:"stage"`
	Binding string `json:"subject_binding"`
	// Observed distinguishes request acceptance from independently observed effect
	// (done-when 9): REQUESTED and RESTORE_ATTEMPTED are what the harness asked for
	// and was told; BEFORE, EFFECTIVE, and RESTORED are what it read back itself.
	Observed bool           `json:"independently_observed"`
	Subjects []SubjectState `json:"subjects"`
	Refusal  string         `json:"refusal,omitempty"`
}

// EffectReceipt is the typed, scrubbed result of one arm: the five stages, the
// declared restoration policy, and the single dominant outcome.
type EffectReceipt struct {
	Arm     string        `json:"arm"`
	Binding string        `json:"subject_binding"`
	Stages  []StageRecord `json:"stages"`
	// Policy publishes which subjects must return exactly and which are judged
	// under a named semantic rule.
	Policy []EquivalenceDecl `json:"restoration_rule"`
	// MeasurementRan says whether the benchmark body executed at all — a refusal
	// or an ineffective mutation means the measurement was never taken.
	MeasurementRan bool `json:"measurement_ran"`
	// CompensationAccepted says whether every compensation REQUEST was accepted.
	// It is acceptance, not proof: StateVerifiedClean is the observed half.
	CompensationAccepted bool `json:"compensation_accepted"`
	// StateVerifiedClean is true only when the independent post-action readback put
	// every subject back at its captured value under its declared rule.
	StateVerifiedClean bool   `json:"state_verified_clean"`
	Outcome            string `json:"outcome"`
	Finding            string `json:"finding"`
	// dirty carries this arm's non-clean subjects forward to the sequence runner.
	dirty map[string]string
}

// Success is true for exactly one outcome. A benchmark is not complete when the
// measurements finish: failed restoration, an ineffective mutation, and an
// unreadable post-state are all non-success, and none of them can be hidden by the
// primary operation having succeeded.
func (r EffectReceipt) Success() bool { return r.Outcome == OutcomeRestored }

// SequenceReceipt is a run of arms plus the contamination guard between them.
type SequenceReceipt struct {
	Schema     string          `json:"schema"`
	Provenance Provenance      `json:"provenance"`
	Arms       []EffectReceipt `json:"arms"`
	Verdict    string          `json:"verdict"`
	Finding    string          `json:"finding"`
}

// JSON renders the sequence as stable, indented JSON — a re-derivable, publishable
// witness artifact.
func (r SequenceReceipt) JSON() ([]byte, error) { return json.MarshalIndent(r, "", "  ") }

// effectDigest is the subject-salted value digest. Salting by subject means the same
// value under two subjects does not produce the same digest, so the artifact cannot
// be used to correlate values across subjects, and no raw value ever appears.
func effectDigest(subject, value string) string {
	sum := sha256.Sum256([]byte("fak-effect-value/1\x00" + subject + "\x00" + value))
	return "sha256:" + hex.EncodeToString(sum[:])[:16]
}

// effectBinding is the non-reversible subject-set binding: a digest over the sorted
// subject NAMES only. Every stage carries it, so a stage that saw a different set
// cannot masquerade as part of the same transaction.
func effectBinding(subjects []string) string {
	h := sha256.New()
	h.Write([]byte("fak-effect-subject-set/1"))
	for _, s := range subjects {
		h.Write([]byte{0})
		h.Write([]byte(s))
	}
	return "bind:" + hex.EncodeToString(h.Sum(nil))[:16]
}

// effectSubjectNames returns the deduped, sorted subject names of a plan set — the
// canonical form both the binding and the receipt walk.
func effectSubjectNames(plans []SubjectMutation) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(plans))
	for _, p := range plans {
		if seen[p.Subject] {
			continue
		}
		seen[p.Subject] = true
		out = append(out, p.Subject)
	}
	sort.Strings(out)
	return out
}

// CaptureSubjects runs the BEFORE stage: it independently reads every declared
// subject and returns the immutable subject-set receipt that authorizes the rest of
// the transaction. A subject it cannot read is recorded as UNREADABLE, which makes
// the receipt incomplete and blocks mutation — capturing nothing to roll back to is
// not a licence to mutate.
func CaptureSubjects(ad Adapter, subjects []SubjectMutation) (*SubjectReceipt, StageRecord) {
	names := effectSubjectNames(subjects)
	rc := &SubjectReceipt{
		binding:  effectBinding(names),
		order:    names,
		plans:    make(map[string]SubjectMutation, len(names)),
		before:   make(map[string]string, len(names)),
		readable: make(map[string]bool, len(names)),
	}
	for _, p := range subjects {
		if _, dup := rc.plans[p.Subject]; !dup {
			rc.plans[p.Subject] = p
		}
	}

	rec := StageRecord{Stage: StageBefore, Binding: rc.binding, Observed: true}
	idh := sha256.New()
	idh.Write([]byte("fak-effect-receipt/1\x00" + rc.binding))
	for _, s := range names {
		p := rc.plans[s]
		st, value, err := effectReadSubjectState(ad, p, s, true)
		if err != nil {
			st.Fault = FaultReadFailed
			st.ValueDigest = "unreadable"
		} else {
			rc.before[s] = value
			rc.readable[s] = true
		}
		idh.Write([]byte{0})
		idh.Write([]byte(s + "=" + st.ValueDigest))
		rec.Subjects = append(rec.Subjects, st)
	}
	rc.id = "rcpt:" + hex.EncodeToString(idh.Sum(nil))
	return rc, rec
}

func effectNormalize(p SubjectMutation, v string) string {
	if p.rule() == EquivalenceSemantic && p.Normalize != nil {
		return p.Normalize(v)
	}
	return v
}

func effectObservedSubjectState(p SubjectMutation, subject, value string, normalize bool) SubjectState {
	st := SubjectState{Subject: subject, Readable: true, ValueDigest: effectDigest(subject, value)}
	if normalize && p.rule() == EquivalenceSemantic {
		st.NormalizedDigest = effectDigest(subject, effectNormalize(p, value))
	}
	return st
}

func effectReadSubjectState(ad Adapter, p SubjectMutation, subject string, normalize bool) (SubjectState, string, error) {
	value, err := ad.Read(subject)
	if err != nil {
		return SubjectState{Subject: subject}, "", err
	}
	return effectObservedSubjectState(p, subject, value, normalize), value, nil
}

func effectReadEffectiveState(ad Adapter, p SubjectMutation, subject string) (SubjectState, string, bool) {
	st, value, err := effectReadSubjectState(ad, p, subject, true)
	if err != nil {
		st.Effect = "unreadable"
		st.Fault = FaultEffectReadFailed
		st.ValueDigest = "unreadable"
		return st, "", false
	}
	return st, value, true
}

func effectReadRestoredState(ad Adapter, p SubjectMutation, subject string) (SubjectState, string, bool) {
	st, value, err := effectReadSubjectState(ad, p, subject, true)
	if err != nil {
		st.Restoration = SubjectUnknown
		st.Fault = FaultPostReadFailed
		st.ValueDigest = "unreadable"
		return st, "", false
	}
	return st, value, true
}

// RunEffectArm drives one arm through all five stages against the receipt captured
// earlier. It never consults the arm's plan for scope: after the binding check it
// walks the receipt's subject set, so scope widening is refused rather than applied.
// Compensation runs on every exit path — success, refusal, measurement error, and
// cancellation alike.
func RunEffectArm(ctx context.Context, rc *SubjectReceipt, arm EffectArm, ad Adapter) EffectReceipt {
	out := EffectReceipt{
		Arm:                  arm.Name,
		Binding:              rc.Binding(),
		Policy:               restorationRule(rc, arm.Subjects),
		CompensationAccepted: true,
		dirty:                map[string]string{},
	}

	// Stage BEFORE is the capture the receipt already made; replay it so the arm's
	// stage list is complete and self-contained.
	out.Stages = append(out.Stages, effectBeforeStage(rc))

	// Scope check FIRST: a plan that does not bind to the receipt is refused before
	// a single Apply. A nil receipt observed nothing, so any mutation widens it.
	if effectBinding(effectSubjectNames(arm.Subjects)) != rc.Binding() {
		return effectRefuse(out, rc, ad, OutcomeSubjectSetWidened,
			"arm subject set does not bind to the captured receipt: a scoped observation may not authorize an unscoped change")
	}
	for _, p := range arm.Subjects {
		if !rc.covers(p.Subject) {
			return effectRefuse(out, rc, ad, OutcomeSubjectSetWidened,
				fmt.Sprintf("subject %q is outside the captured receipt", p.Subject))
		}
	}
	if !rc.complete() {
		return effectRefuse(out, rc, ad, OutcomePreStateUnreadable,
			"pre-state could not be captured for every subject: nothing to roll back to, so no mutation is authorized")
	}

	// Stage REQUESTED — acceptance only, never proof of effect. Every subject we
	// call Apply on is compensated later, even one that rejected the request: a
	// rejection is not proof that nothing moved.
	reqStage := StageRecord{Stage: StageRequested, Binding: rc.binding}
	var applied []string
	rejected := false
	for _, s := range rc.order {
		p := rc.plans[s]
		st := effectObservedSubjectState(p, s, p.Requested, true)
		applied = append(applied, s)
		if err := ad.Apply(s, p.Requested); err != nil {
			st.Request = "rejected"
			st.Fault = FaultApplyRejected
			rejected = true
		} else {
			st.Request = "accepted"
		}
		reqStage.Subjects = append(reqStage.Subjects, st)
		if rejected {
			break // fail closed: stop widening the blast radius, go compensate
		}
	}
	out.Stages = append(out.Stages, reqStage)

	// Stage EFFECTIVE — the independent readback. An accepted request that changed
	// nothing is caught here rather than benchmarked.
	effStage := StageRecord{Stage: StageEffective, Binding: rc.binding, Observed: true}
	ineffective := false
	for _, s := range applied {
		p := rc.plans[s]
		st, v, readable := effectReadEffectiveState(ad, p, s)
		if !readable {
			ineffective = true
		} else {
			if p.equivalent(p.Requested, v) {
				st.Effect = "effective"
			} else {
				st.Effect = "ineffective"
				ineffective = true
			}
		}
		effStage.Subjects = append(effStage.Subjects, st)
	}
	out.Stages = append(out.Stages, effStage)

	// Measure only when the mutation is proven in place and the context is live.
	measureFailed := false
	cancelled := ctx != nil && ctx.Err() != nil
	if !rejected && !ineffective && !cancelled {
		if arm.Measure != nil {
			out.MeasurementRan = true
			if err := arm.Measure(ctx); err != nil {
				measureFailed = true
			}
		} else {
			out.MeasurementRan = true
		}
		if ctx != nil && ctx.Err() != nil {
			cancelled = true
		}
	}

	// Stages RESTORE_ATTEMPTED + RESTORED always run, on every path above.
	out = effectCompensate(out, rc, ad, applied)

	out.Outcome, out.Finding = effectOutcome(effectVerdictInput{
		receipt:     out,
		rejected:    rejected,
		ineffective: ineffective,
		cancelled:   cancelled,
		measureFail: measureFailed,
	})
	return out
}

// effectBeforeStage re-renders the capture's BEFORE record from the receipt so each
// arm's stage list stands alone.
func effectBeforeStage(rc *SubjectReceipt) StageRecord {
	rec := StageRecord{Stage: StageBefore, Binding: rc.Binding(), Observed: true}
	if rc == nil {
		return rec
	}
	for _, s := range rc.order {
		p := rc.plans[s]
		st := SubjectState{Subject: s, ValueDigest: "unreadable"}
		if rc.readable[s] {
			st.Readable = true
			st.ValueDigest = effectDigest(s, rc.before[s])
			if p.rule() == EquivalenceSemantic {
				st.NormalizedDigest = effectDigest(s, effectNormalize(p, rc.before[s]))
			}
		} else {
			st.Fault = FaultReadFailed
		}
		rec.Subjects = append(rec.Subjects, st)
	}
	return rec
}

func restorationRule(rc *SubjectReceipt, plans []SubjectMutation) []EquivalenceDecl {
	byName := map[string]SubjectMutation{}
	var names []string
	add := func(p SubjectMutation) {
		if _, ok := byName[p.Subject]; ok {
			return
		}
		byName[p.Subject] = p
		names = append(names, p.Subject)
	}
	if rc != nil {
		for _, s := range rc.order {
			add(rc.plans[s])
		}
	}
	for _, p := range plans {
		add(p)
	}
	sort.Strings(names)
	out := make([]EquivalenceDecl, 0, len(names))
	for _, n := range names {
		p := byName[n]
		out = append(out, EquivalenceDecl{Subject: n, Rule: p.rule(), Normalizer: p.normalizerLabel()})
	}
	return out
}

// effectRefuse takes a pre-mutation refusal path. Compensation still runs (over an
// empty applied set) and the post-action readback still executes, so the receipt
// PROVES the refusal left the observed state untouched rather than merely asserting
// it.
func effectRefuse(out EffectReceipt, rc *SubjectReceipt, ad Adapter, outcome, why string) EffectReceipt {
	out.Stages = append(out.Stages, StageRecord{Stage: StageRequested, Binding: rc.Binding(), Refusal: outcome})
	out.Stages = append(out.Stages, StageRecord{Stage: StageEffective, Binding: rc.Binding(), Observed: true, Refusal: outcome})
	out = effectCompensate(out, rc, ad, nil)
	// A refusal that nonetheless found the state moved or unknown must not report the
	// refusal as the whole story: the state fault outranks it.
	switch {
	case effectHas(out.dirty, SubjectChanged):
		out.Outcome = OutcomeRestoreFailed
	case effectHas(out.dirty, SubjectUnknown):
		out.Outcome = OutcomePostStateUnreadable
	default:
		out.Outcome = outcome
	}
	out.Finding = why
	if out.Outcome != outcome {
		out.Finding = why + "; additionally the post-action readback did not confirm clean state"
	}
	return out
}

// effectCompensate runs RESTORE_ATTEMPTED then the independent RESTORED readback.
// The compensation unwinds in reverse application order; the readback covers every
// subject in the receipt, not just the applied ones, so a subject the arm reached
// out of order is still checked. RESTORED is emitted only from the readback —
// never from the adapter's acceptance of the compensation.
func effectCompensate(out EffectReceipt, rc *SubjectReceipt, ad Adapter, applied []string) EffectReceipt {
	attempt := StageRecord{Stage: StageRestoreAttempted, Binding: rc.Binding()}
	compensated := map[string]bool{}
	for i := len(applied) - 1; i >= 0; i-- {
		s := applied[i]
		if compensated[s] {
			continue
		}
		compensated[s] = true
		st := SubjectState{Subject: s, Readable: true, ValueDigest: effectDigest(s, rc.before[s])}
		if err := ad.Restore(s, rc.before[s]); err != nil {
			st.Request = "rejected"
			st.Fault = FaultRestoreRejected
			out.CompensationAccepted = false
		} else {
			st.Request = "accepted"
		}
		attempt.Subjects = append(attempt.Subjects, st)
	}
	sort.Slice(attempt.Subjects, func(i, j int) bool {
		return attempt.Subjects[i].Subject < attempt.Subjects[j].Subject
	})
	out.Stages = append(out.Stages, attempt)

	restored := StageRecord{Stage: StageRestored, Binding: rc.Binding(), Observed: true}
	clean := true
	if rc != nil {
		for _, s := range rc.order {
			p := rc.plans[s]
			st, v, readable := effectReadRestoredState(ad, p, s)
			switch {
			case !readable:
			case !rc.readable[s]:
				// No captured "before" to compare against: the subject's restoration
				// is unknowable, not clean.
				st.Restoration = SubjectUnknown
			default:
				if p.equivalent(rc.before[s], v) {
					st.Restoration = SubjectRestored
				} else {
					st.Restoration = SubjectChanged
				}
			}
			if st.Restoration != SubjectRestored {
				clean = false
				// Only a subject whose pre-state WAS captured can have been left
				// dirty by this arm. One that was already unreadable at capture was
				// never mutated (the arm refused), so it is not this arm's residue
				// and must not block a successor that would refuse on its own.
				if rc.readable[s] {
					out.dirty[s] = st.Restoration
				}
			}
			restored.Subjects = append(restored.Subjects, st)
		}
	}
	if rc == nil || len(rc.order) == 0 {
		clean = false
	}
	out.Stages = append(out.Stages, restored)
	out.StateVerifiedClean = clean
	return out
}

func effectHas(dirty map[string]string, want string) bool {
	for _, v := range dirty {
		if v == want {
			return true
		}
	}
	return false
}

type effectVerdictInput struct {
	receipt     EffectReceipt
	rejected    bool
	ineffective bool
	cancelled   bool
	measureFail bool
}

// effectOutcome folds the run into ONE dominant outcome. The precedence is the
// contract's fail-closed rule: a definite state fault (something did not come back)
// outranks an indefinite one (we could not tell), both outrank any result of the
// primary operation, and an ineffective mutation outranks a measurement verdict
// computed over state that never changed.
func effectOutcome(in effectVerdictInput) (string, string) {
	r := in.receipt
	names := effectDirtyNames(r.dirty, SubjectChanged)
	unknown := effectDirtyNames(r.dirty, SubjectUnknown)
	switch {
	case len(names) > 0:
		return OutcomeRestoreFailed, fmt.Sprintf(
			"restoration FAILED: %s did not return to the captured value under the declared rule; "+
				"state is left changed and no measurement from this arm may be compared.",
			strings.Join(names, ", "))
	case len(unknown) > 0:
		return OutcomePostStateUnreadable, fmt.Sprintf(
			"post-action state UNKNOWN for %s: restoration could not be verified, so clean state is not assumed.",
			strings.Join(unknown, ", "))
	case in.rejected:
		return OutcomeApplyRejected, "the adapter REJECTED the requested change; state verified back at the captured value."
	case in.ineffective:
		return OutcomeIneffective, "the change was ACCEPTED but the independent readback proves it never took effect; " +
			"the measurement was not taken over the requested state."
	case in.cancelled:
		return OutcomeCancelled, "the run was cancelled; compensation still ran and the state was verified restored."
	case in.measureFail:
		return OutcomeMeasurementFailed, "the measurement failed; compensation still ran and the state was verified restored."
	}
	return OutcomeRestored, fmt.Sprintf(
		"applied, independently observed effective, measured, compensated, and independently observed back at "+
			"the captured value for all %d subject(s).", effectStageCount(r, StageRestored))
}

func effectStageCount(r EffectReceipt, stage string) int {
	for _, s := range r.Stages {
		if s.Stage == stage {
			return len(s.Subjects)
		}
	}
	return 0
}

func effectDirtyNames(dirty map[string]string, want string) []string {
	var out []string
	for k, v := range dirty {
		if v == want {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}

// RunEffectSequence runs arms in order and refuses any arm whose subjects a
// previous arm left changed or UNKNOWN. That is what makes a later comparison rest
// on verified clean state rather than on cleanup convention: a leaked run-scoped
// setting stops the next arm instead of silently contaminating it.
func RunEffectSequence(ctx context.Context, arms []EffectArm, ad Adapter) SequenceReceipt {
	unclean := map[string]string{}
	out := SequenceReceipt{
		Schema: "effect-receipt.v1",
		Provenance: simulatedProvenance(
			"go test ./internal/bench -run TestEffectReceipt -count=1",
			"fak/internal/bench.RunEffectSequence",
			"Arms run against a labeled in-memory adapter fixture (SyntheticEnv) with injected faults: the run "+
				"witnesses the stage machine, the subject-set binding, and the typed refusals. A real harness feeds "+
				"the same receipt shape by implementing Adapter over its own environment.",
		),
	}

	for _, arm := range arms {
		if blocked := effectBlockedBy(arm, unclean); len(blocked) > 0 {
			rc, _ := CaptureSubjects(effectNullAdapter{}, arm.Subjects)
			r := EffectReceipt{
				Arm:                  arm.Name,
				Binding:              rc.Binding(),
				Policy:               restorationRule(nil, arm.Subjects),
				CompensationAccepted: true,
				Outcome:              OutcomePredecessorUnclean,
				dirty:                map[string]string{},
				Finding: fmt.Sprintf(
					"refused before any observation: a preceding arm left %s in a state this arm depends on; "+
						"running it would compare against unverified state.", strings.Join(blocked, ", ")),
			}
			for _, stage := range []string{StageBefore, StageRequested, StageEffective, StageRestoreAttempted, StageRestored} {
				r.Stages = append(r.Stages, StageRecord{
					Stage:    stage,
					Binding:  rc.Binding(),
					Observed: stage != StageRequested && stage != StageRestoreAttempted,
					Refusal:  OutcomePredecessorUnclean,
				})
			}
			out.Arms = append(out.Arms, r)
			continue
		}

		rc, _ := CaptureSubjects(ad, arm.Subjects)
		r := RunEffectArm(ctx, rc, arm, ad)
		for s, d := range r.dirty {
			unclean[s] = d
		}
		out.Arms = append(out.Arms, r)
	}

	out.Verdict, out.Finding = effectSequenceVerdict(out.Arms)
	return out
}

// effectBlockedBy names the subjects this arm needs that a previous arm left dirty.
func effectBlockedBy(arm EffectArm, unclean map[string]string) []string {
	var blocked []string
	for _, p := range arm.Subjects {
		if d, bad := unclean[p.Subject]; bad {
			blocked = append(blocked, p.Subject+" ("+d+")")
		}
	}
	sort.Strings(blocked)
	return blocked
}

func effectSequenceVerdict(arms []EffectReceipt) (string, string) {
	ok := 0
	var flagged []string
	for _, a := range arms {
		if a.Success() {
			ok++
			continue
		}
		flagged = append(flagged, a.Arm+"="+a.Outcome)
	}
	if len(flagged) == 0 {
		return VerdictEffectClean, fmt.Sprintf(
			"all %d arm(s) applied, measured, compensated, and were independently observed back at the captured state.", ok)
	}
	return VerdictEffectFlagged, fmt.Sprintf(
		"%d of %d arm(s) reached a non-success outcome (%s); %d arm(s) completed with verified clean state.",
		len(flagged), len(arms), strings.Join(flagged, ", "), ok)
}

// effectNullAdapter reads nothing: it backs the pre-observation refusal path, where
// an arm is turned away before its environment is touched at all.
type effectNullAdapter struct{}

func (effectNullAdapter) Read(string) (string, error)  { return "", fmt.Errorf("not observed") }
func (effectNullAdapter) Apply(string, string) error   { return fmt.Errorf("not observed") }
func (effectNullAdapter) Restore(string, string) error { return fmt.Errorf("not observed") }

// ---------------------------------------------------------------------------
// SyntheticEnv — the reference adapter and the adversarial fixture surface.
// ---------------------------------------------------------------------------

// Injectable adapter faults. Each is a real failure mode a live harness hits: a
// setting that cannot be read, a write the runtime refuses, a write it accepts and
// ignores, a value that becomes unreadable the moment it is changed, and a cleanup
// that reports success without restoring anything.
const (
	FaultInjectNone             = ""
	FaultInjectReadBefore       = "read_before_fails"
	FaultInjectApplyRejects     = "apply_rejects"
	FaultInjectApplyIgnored     = "apply_ignored"
	FaultInjectReadAfterApply   = "read_after_apply_fails"
	FaultInjectRestoreRejects   = "restore_rejects"
	FaultInjectRestoreIgnored   = "restore_ignored"
	FaultInjectReadAfterRestore = "read_after_restore_fails"
)

type syntheticSubject struct {
	value    string
	fault    string
	volatile bool
	applied  bool
	restored bool
}

// SyntheticEnv is the in-memory reference Adapter: a labeled fixture that models an
// environment with per-subject injectable faults and "volatile" subjects whose every
// write gains a changing marker (the dynamic fields that motivate semantic
// equivalence). Its write marker is a monotonic counter, never a clock or a random
// value, so a run over it is byte-identical every time.
type SyntheticEnv struct {
	subjects map[string]*syntheticSubject
	writes   int
}

// NewSyntheticEnv builds the fixture from an initial subject/value map.
func NewSyntheticEnv(initial map[string]string) *SyntheticEnv {
	e := &SyntheticEnv{subjects: make(map[string]*syntheticSubject, len(initial))}
	for k, v := range initial {
		e.subjects[k] = &syntheticSubject{value: v}
	}
	return e
}

// Inject arms a fault on one subject and returns the env for chaining.
func (e *SyntheticEnv) Inject(subject, fault string) *SyntheticEnv {
	if s := e.subjects[subject]; s != nil {
		s.fault = fault
	}
	return e
}

// Volatile marks a subject whose writes gain a changing marker — the dynamic field
// that must be judged by semantic equivalence rather than byte comparison.
func (e *SyntheticEnv) Volatile(subject string) *SyntheticEnv {
	if s := e.subjects[subject]; s != nil {
		s.volatile = true
	}
	return e
}

// Snapshot returns the current values, for asserting that ambient state the
// transaction never declared was preserved untouched.
func (e *SyntheticEnv) Snapshot() map[string]string {
	out := make(map[string]string, len(e.subjects))
	for k, s := range e.subjects {
		out[k] = s.value
	}
	return out
}

// write stamps a volatile subject with a FRESH write marker, replacing any marker
// the incoming value already carried. Markers replace rather than accumulate, which
// is what makes the dynamic field genuinely non-byte-restorable while still
// semantically equal to what was captured.
func (e *SyntheticEnv) write(s *syntheticSubject, v string) {
	e.writes++
	if s.volatile {
		v = StripVolatileMarker(v) + volatileMarker + strconv.Itoa(e.writes)
	}
	s.value = v
}

// volatileMarker separates a volatile subject's significant value from the changing
// write marker. StripVolatileMarker is the matching declared normalizer.
const volatileMarker = "#w"

// StripVolatileMarker is the semantic-equivalence normalizer for volatile subjects:
// it drops the changing write marker so a value that came back semantically intact
// is judged restored even though it is not byte-for-byte identical.
func StripVolatileMarker(v string) string {
	if i := strings.LastIndex(v, volatileMarker); i >= 0 {
		return v[:i]
	}
	return v
}

func (e *SyntheticEnv) Read(subject string) (string, error) {
	s := e.subjects[subject]
	if s == nil {
		return "", fmt.Errorf("bench: subject %q is not present in the environment", subject)
	}
	switch {
	case s.fault == FaultInjectReadBefore,
		s.fault == FaultInjectReadAfterApply && s.applied,
		s.fault == FaultInjectReadAfterRestore && s.restored:
		return "", fmt.Errorf("bench: subject %q is unreadable", subject)
	}
	return s.value, nil
}

func (e *SyntheticEnv) Apply(subject, value string) error {
	s := e.subjects[subject]
	if s == nil {
		return fmt.Errorf("bench: subject %q is not present in the environment", subject)
	}
	s.applied = true
	switch s.fault {
	case FaultInjectApplyRejects:
		return fmt.Errorf("bench: subject %q refused the change", subject)
	case FaultInjectApplyIgnored:
		return nil // accepted, and silently ineffective — the readback must catch it
	}
	e.write(s, value)
	return nil
}

func (e *SyntheticEnv) Restore(subject, value string) error {
	s := e.subjects[subject]
	if s == nil {
		return fmt.Errorf("bench: subject %q is not present in the environment", subject)
	}
	s.restored = true
	switch s.fault {
	case FaultInjectRestoreRejects:
		return fmt.Errorf("bench: subject %q refused the compensating change", subject)
	case FaultInjectRestoreIgnored:
		return nil // cleanup claims success without restoring — the readback must catch it
	}
	e.write(s, value)
	return nil
}

// ---------------------------------------------------------------------------
// The known-positive fixture.
// ---------------------------------------------------------------------------

// DefaultEffectEnv is the fixture environment: two declared subjects the arms
// mutate (one exact, one volatile/semantic) plus one AMBIENT subject no arm ever
// declares, which exists to prove the transaction leaves undeclared state alone.
func DefaultEffectEnv() *SyntheticEnv {
	e := NewSyntheticEnv(map[string]string{
		"bench.threads":        "8",
		"bench.run_label":      "baseline",
		"operator.log_channel": "ambient-untouched",
	})
	return e.Volatile("bench.run_label")
}

// AmbientSubject is the fixture subject no arm declares — the untouched control.
const AmbientSubject = "operator.log_channel"

// DefaultEffectSubjects is the known-positive subject plan: one subject that must
// come back byte-for-byte and one dynamic subject judged under a named semantic
// rule, so the fixture exercises BOTH declared restoration rules.
func DefaultEffectSubjects() []SubjectMutation {
	return []SubjectMutation{
		{Subject: "bench.threads", Requested: "1", Equivalence: EquivalenceExact},
		{
			Subject: "bench.run_label", Requested: "arm-under-test",
			Equivalence: EquivalenceSemantic,
			Normalizer:  "drop-volatile-write-marker",
			Normalize:   StripVolatileMarker,
		},
	}
}

// DefaultEffectArms is the reference sequence: a clean arm that round-trips, an arm
// whose own body puts the environment into a state where cleanup reports success
// without restoring anything, and a successor arm that must be refused because the
// leaked subject is exactly the one it needs.
//
// The middle arm arms the cleanup fault from inside its own measurement, which is
// how the failure actually presents in a live harness — the run itself changes the
// conditions its cleanup depends on, so the leak is not visible until the
// post-action readback looks.
func DefaultEffectArms(env *SyntheticEnv) []EffectArm {
	nop := func(context.Context) error { return nil }
	return []EffectArm{
		{Name: "baseline", Subjects: DefaultEffectSubjects(), Measure: nop},
		{
			Name: "leaky-cleanup", Subjects: DefaultEffectSubjects(),
			Measure: func(context.Context) error {
				env.Inject("bench.threads", FaultInjectRestoreIgnored)
				return nil
			},
		},
		{Name: "successor", Subjects: DefaultEffectSubjects(), Measure: nop},
	}
}

// DefaultEffectSequence is the committed witness: one run demonstrating the accepted
// case (arm 1 round-trips and is observed clean), the failed-restoration refusal
// (arm 2's cleanup claims success while the readback proves the subject did not come
// back), and the contamination refusal (arm 3 is turned away before it observes
// anything). It is deterministic, so it re-derives byte-for-byte.
func DefaultEffectSequence() SequenceReceipt {
	env := DefaultEffectEnv()
	return RunEffectSequence(context.Background(), DefaultEffectArms(env), env)
}
