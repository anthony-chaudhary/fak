package taskmgr

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/answershape"
)

// coherentOutput is a multi-sentence answer with natural, non-repetitive shape — the
// in-shape control fixture. It sits well below answershape's default repeat threshold.
const coherentOutput = "A witness reads the effect back from a source the process " +
	"did not author. The task manager records a graded verdict beside the claim it " +
	"makes about itself. Progress is measured from evidence, never asserted by the " +
	"reporting process. A degenerate beat is a silent semantic failure."

// TestShapeWitnessRefusesDegenerateRunaway grades a whitespace-free character runaway
// — the canonical looping/degenerate output — and asserts the witness REFUSES it and
// names the answershape reason in its verdict/detail.
func TestShapeWitnessRefusesDegenerateRunaway(t *testing.T) {
	garbage := strings.Repeat("A", 200)
	rec := ShapeWitness{}.WitnessClaim(Claim{Refs: []EvidenceRef{{Kind: OutputRefKind, Ref: garbage}}})

	if rec.VerifiedState != VerifiedRefused {
		t.Fatalf("degenerate runaway: state = %q, want verified_refused", rec.VerifiedState)
	}
	if rec.Source != "answershape" {
		t.Fatalf("source = %q, want answershape", rec.Source)
	}
	if rec.Verdict == "" {
		t.Fatalf("refusal carries no verdict headline")
	}
	// The detail must NAME the answershape reason (the repetition sub-signal), so the
	// refusal explains WHY without the reader re-running the measurement.
	if !strings.Contains(rec.Detail, "repetitive") {
		t.Fatalf("detail = %q, want it to name the answershape repetition reason", rec.Detail)
	}
	// ...but it must NOT dump the payload: the 200-char run is bounded out of the record.
	if strings.Contains(rec.Detail, strings.Repeat("A", 100)) {
		t.Fatalf("detail dumped the degenerate payload: %q", rec.Detail)
	}
	// The echoed evidence ref is reduced to a byte-count descriptor: the payload Ref is
	// cleared and replaced by a "graded N bytes" note, so no payload byte survives.
	for _, ref := range rec.EvidenceRefs {
		if ref.Kind != OutputRefKind {
			continue
		}
		if ref.Ref != "" {
			t.Fatalf("output ref still carries the payload: %+v", ref)
		}
		if !strings.Contains(ref.Note, "graded") || !strings.Contains(ref.Note, "200") {
			t.Fatalf("output ref note = %q, want a 'graded 200 bytes' descriptor", ref.Note)
		}
	}
}

// TestShapeWitnessRefusesLoopedSentence covers the other degeneration mode: a coherent
// sentence emitted over and over (the n-gram / line-block loop), distinct from the
// single-char runaway above.
func TestShapeWitnessRefusesLoopedSentence(t *testing.T) {
	loop := strings.Repeat("the model is looping on this exact sentence. ", 40)
	rec := ShapeWitness{}.WitnessClaim(Claim{Refs: []EvidenceRef{{Kind: OutputRefKind, Ref: loop}}})
	if rec.VerifiedState != VerifiedRefused {
		t.Fatalf("looped sentence: state = %q, want verified_refused", rec.VerifiedState)
	}
	if rec.Detail == "" || rec.Verdict == "" {
		t.Fatalf("looped-sentence refusal missing verdict/detail: %+v", rec)
	}
}

// TestShapeWitnessConfirmsCoherentOutput grades an in-shape multi-sentence answer and
// asserts the witness CONFIRMS it (verified_done) rather than refusing.
func TestShapeWitnessConfirmsCoherentOutput(t *testing.T) {
	rec := ShapeWitness{}.WitnessClaim(Claim{Refs: []EvidenceRef{{Kind: OutputRefKind, Ref: coherentOutput}}})
	if rec.VerifiedState != VerifiedDone {
		t.Fatalf("coherent output: state = %q (detail %q), want verified_done", rec.VerifiedState, rec.Detail)
	}
}

// TestShapeWitnessUnavailableWithoutOutputRef asserts a claim that carries no "output"
// ref returns verified_unavailable — there is nothing to read back — and never
// silently downgrades to verified_done, mirroring PathWitness's "no path evidence".
func TestShapeWitnessUnavailableWithoutOutputRef(t *testing.T) {
	// No refs at all.
	if rec := (ShapeWitness{}).WitnessClaim(Claim{}); rec.VerifiedState != VerifiedUnavailable {
		t.Fatalf("no refs: state = %q, want verified_unavailable", rec.VerifiedState)
	}
	// Refs of a different kind: still nothing of the right kind to grade.
	rec := ShapeWitness{}.WitnessClaim(Claim{Refs: []EvidenceRef{{Kind: "path", Ref: "/tmp/x"}, {Kind: "commit", Ref: "HEAD"}}})
	if rec.VerifiedState != VerifiedUnavailable {
		t.Fatalf("non-output refs: state = %q, want verified_unavailable", rec.VerifiedState)
	}
	if rec.VerifiedState == VerifiedDone {
		t.Fatalf("unavailable must never downgrade to verified_done")
	}
}

// TestShapeWitnessLimitsConfigurable proves the Limits knobs are honored: the same
// short text reads in-shape under the default repeat threshold but degenerate once the
// caller tightens MaxChars to flag verbosity.
func TestShapeWitnessLimitsConfigurable(t *testing.T) {
	text := coherentOutput
	if rec := (ShapeWitness{}).WitnessClaim(Claim{Refs: []EvidenceRef{{Kind: OutputRefKind, Ref: text}}}); rec.VerifiedState != VerifiedDone {
		t.Fatalf("default limits: state = %q, want verified_done", rec.VerifiedState)
	}
	tight := ShapeWitness{Limits: answershape.Limits{MaxChars: 16}}
	rec := tight.WitnessClaim(Claim{Refs: []EvidenceRef{{Kind: OutputRefKind, Ref: text}}})
	if rec.VerifiedState != VerifiedRefused {
		t.Fatalf("tight max-chars: state = %q, want verified_refused", rec.VerifiedState)
	}
	if !strings.Contains(rec.Detail, "verbose") {
		t.Fatalf("verbosity refusal detail = %q, want it to name the verbose reason", rec.Detail)
	}
}

// TestShapeWitnessEndToEndKeepsClaimBesideRefusal is the core fence test: a task is
// alive and beating (StateRunning) while emitting garbage. The ShapeWitness rung
// records verified_refused BESIDE the claimed running state — it never overwrites it —
// and the witnessed snapshot still validates.
func TestShapeWitnessEndToEndKeepsClaimBesideRefusal(t *testing.T) {
	m := NewManager()
	task, err := m.StartTask(TaskSpec{TaskID: "task_shape", Total: 10})
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	if err := task.SetProgress(3, 10, "tok"); err != nil { // alive: progress + beat
		t.Fatalf("progress: %v", err)
	}
	if err := task.Beat(); err != nil {
		t.Fatalf("beat: %v", err)
	}

	garbage := strings.Repeat("LOOP ", 80) // 400 chars of a repeated unit
	rec, err := m.WitnessTask("task_shape", ShapeWitness{}, []EvidenceRef{{Kind: OutputRefKind, Ref: garbage}})
	if err != nil {
		t.Fatalf("witness: %v", err)
	}
	if rec.VerifiedState != VerifiedRefused {
		t.Fatalf("witness rec = %q, want verified_refused", rec.VerifiedState)
	}

	snap := m.Snapshot()
	if err := ValidateSnapshot(snap); err != nil {
		t.Fatalf("witnessed snapshot failed validation: %v", err)
	}
	got := snap.Tasks[0]
	// The claim survives the refusal: the process still CLAIMS running.
	if got.State != StateRunning {
		t.Fatalf("claimed state = %q, want running (the rung must not overwrite the claim)", got.State)
	}
	if got.Witness == nil || got.Witness.VerifiedState != VerifiedRefused {
		t.Fatalf("witness rung = %+v, want verified_refused beside the claim", got.Witness)
	}
	if got.Witness.CheckedUnixNano == 0 {
		t.Fatalf("witness timestamp not stamped")
	}
	// Running + refused == "alive but emitting garbage": not really progressing.
	if got.VerifiedProgressing() {
		t.Fatalf("running task with a refused witness must report VerifiedProgressing() == false")
	}
	// The snapshot JSON must not carry the raw degenerate payload.
	if b, _ := json.Marshal(got.Witness); strings.Contains(string(b), strings.Repeat("LOOP ", 20)) {
		t.Fatalf("witness JSON dumped the degenerate payload: %s", b)
	}
}

// TestShapeWitnessEndToEndConfirmsCoherent is the in-shape end-to-end: a beat carrying
// coherent output is graded verified_done beside the claim, and the task reports as
// progressing.
func TestShapeWitnessEndToEndConfirmsCoherent(t *testing.T) {
	m := NewManager()
	if _, err := m.StartTask(TaskSpec{TaskID: "task_ok"}); err != nil {
		t.Fatalf("start: %v", err)
	}
	rec, err := m.WitnessTask("task_ok", ShapeWitness{}, []EvidenceRef{{Kind: OutputRefKind, Ref: coherentOutput}})
	if err != nil {
		t.Fatalf("witness: %v", err)
	}
	if rec.VerifiedState != VerifiedDone {
		t.Fatalf("coherent end-to-end = %q, want verified_done", rec.VerifiedState)
	}
	got := m.Snapshot().Tasks[0]
	if !got.VerifiedProgressing() {
		t.Fatalf("confirmed witness must report VerifiedProgressing() == true")
	}
}

func TestBeatTaskWithEvidenceShapeWitnessRefusesDegenerateOutput(t *testing.T) {
	m := NewManager()
	task, err := m.StartTask(TaskSpec{TaskID: "task_beat_shape"})
	if err != nil {
		t.Fatalf("start: %v", err)
	}

	garbage := []byte(strings.Repeat("LOOP ", 80))
	rec, err := task.BeatWithEvidence(garbage)
	if err != nil {
		t.Fatalf("beat with evidence: %v", err)
	}
	if rec.VerifiedState != VerifiedRefused {
		t.Fatalf("beat witness = %q, want verified_refused", rec.VerifiedState)
	}

	got := m.Snapshot().Tasks[0]
	if got.BeatsSeen != 1 || got.LivenessClass != LivenessLive {
		t.Fatalf("beat state = %d/%s, want one live beat", got.BeatsSeen, got.LivenessClass)
	}
	if got.Witness == nil || got.Witness.VerifiedState != VerifiedRefused {
		t.Fatalf("task witness = %+v, want verified_refused", got.Witness)
	}
	if got.VerifiedProgressing() {
		t.Fatalf("beat-time shape refusal must flip VerifiedProgressing() false")
	}
	if b, _ := json.Marshal(got.Witness); strings.Contains(string(b), strings.Repeat("LOOP ", 20)) {
		t.Fatalf("beat witness JSON dumped raw output: %s", b)
	}
}

func TestBeatStepWithEvidenceShapeWitnessRefusesDegenerateOutput(t *testing.T) {
	m := NewManager()
	task, err := m.StartTask(TaskSpec{TaskID: "task_step_beat_shape"})
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	step, err := task.StartStep(StepSpec{StepID: "step_beat_shape"})
	if err != nil {
		t.Fatalf("start step: %v", err)
	}

	rec, err := step.BeatWithEvidence([]byte(strings.Repeat("BAD ", 90)))
	if err != nil {
		t.Fatalf("step beat with evidence: %v", err)
	}
	if rec.VerifiedState != VerifiedRefused {
		t.Fatalf("step beat witness = %q, want verified_refused", rec.VerifiedState)
	}

	got := m.Snapshot().Tasks[0]
	if got.BeatsSeen != 1 || got.Steps[0].BeatsSeen != 1 {
		t.Fatalf("beats task/step = %d/%d, want 1/1", got.BeatsSeen, got.Steps[0].BeatsSeen)
	}
	if got.Steps[0].Witness == nil || got.Steps[0].Witness.VerifiedState != VerifiedRefused {
		t.Fatalf("step witness = %+v, want verified_refused", got.Steps[0].Witness)
	}
	if got.Steps[0].VerifiedProgressing() {
		t.Fatalf("step refusal must flip step VerifiedProgressing() false")
	}
	if got.VerifiedProgressing() {
		t.Fatalf("refused step witness must flip parent task VerifiedProgressing() false")
	}
}

// TestVerifiedProgressingDerivation pins the derived view directly: only a witness
// that ran and REFUSED flips the bit to false; a claimed-only task and every other
// verified state report as progressing. The method is read-only — it reads State and
// the rung, never mutating either.
func TestVerifiedProgressingDerivation(t *testing.T) {
	cases := []struct {
		name string
		ts   TaskSnapshot
		want bool
	}{
		{"claimed-only, no witness", TaskSnapshot{State: StateRunning}, true},
		{"running + refused", TaskSnapshot{State: StateRunning, Witness: &WitnessRecord{VerifiedState: VerifiedRefused}}, false},
		{"running + done", TaskSnapshot{State: StateRunning, Witness: &WitnessRecord{VerifiedState: VerifiedDone}}, true},
		{"running + unavailable", TaskSnapshot{State: StateRunning, Witness: &WitnessRecord{VerifiedState: VerifiedUnavailable}}, true},
		{"running + unknown", TaskSnapshot{State: StateRunning, Witness: &WitnessRecord{VerifiedState: VerifiedUnknown}}, true},
		{"done + refused", TaskSnapshot{State: StateDone, Witness: &WitnessRecord{VerifiedState: VerifiedRefused}}, false},
		{"child step refused", TaskSnapshot{State: StateRunning, Steps: []StepSnapshot{{Witness: &WitnessRecord{VerifiedState: VerifiedRefused}}}}, false},
	}
	for _, tc := range cases {
		if got := tc.ts.VerifiedProgressing(); got != tc.want {
			t.Errorf("%s: VerifiedProgressing() = %v, want %v", tc.name, got, tc.want)
		}
	}
}
