package session

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// TestQualityEnvelopeCanonicalIsDeterministic proves the origin record serializes to
// byte-identical bytes regardless of the order (and duplication) the probes and
// scorecards were declared in — the determinism the portable image's sha256 integrity
// index relies on. Two envelopes with the same controls in different order must produce
// the same canonical form and the same JSON.
func TestQualityEnvelopeCanonicalIsDeterministic(t *testing.T) {
	a := QualityEnvelope{
		Budget:         BudgetEnvelope{Budget: Budget{TurnsLeft: 20, TokensLeft: 200000}},
		WitnessPolicy:  "proof-by-default",
		DogfoodProbes:  []string{"quality-score", "milestone-score", "quality-score", ""},
		ScorecardCards: []string{"milestone_scorecard", "code_quality"},
	}
	b := QualityEnvelope{
		Budget:         BudgetEnvelope{Budget: Budget{TurnsLeft: 20, TokensLeft: 200000}},
		WitnessPolicy:  "proof-by-default",
		DogfoodProbes:  []string{"milestone-score", "quality-score"},
		ScorecardCards: []string{"code_quality", "milestone_scorecard"},
	}

	ca, cb := a.Canonical(), b.Canonical()
	if !reflect.DeepEqual(ca, cb) {
		t.Fatalf("canonical forms differ:\n a=%+v\n b=%+v", ca, cb)
	}
	if want := []string{"milestone-score", "quality-score"}; !reflect.DeepEqual(ca.DogfoodProbes, want) {
		t.Fatalf("dogfood probes = %v, want deduped+sorted %v (empty dropped)", ca.DogfoodProbes, want)
	}

	ja, err := json.Marshal(ca)
	if err != nil {
		t.Fatalf("marshal a: %v", err)
	}
	jb, err := json.Marshal(cb)
	if err != nil {
		t.Fatalf("marshal b: %v", err)
	}
	if string(ja) != string(jb) {
		t.Fatalf("canonical JSON differs:\n a=%s\n b=%s", ja, jb)
	}
}

// TestQualityEnvelopeCanonicalDoesNotMutateReceiver guards the "never mutate the
// caller's slices" contract: Canonical must return a fresh copy so a caller's declared
// order is preserved after it stamps the record.
func TestQualityEnvelopeCanonicalDoesNotMutateReceiver(t *testing.T) {
	e := QualityEnvelope{DogfoodProbes: []string{"z-probe", "a-probe"}}
	_ = e.Canonical()
	if !reflect.DeepEqual(e.DogfoodProbes, []string{"z-probe", "a-probe"}) {
		t.Fatalf("Canonical mutated the receiver's slice: %v", e.DogfoodProbes)
	}
}

// TestQualityEnvelopeSurvivesDriveRoundTrip is the #1964 done-condition witness: the
// envelope rides State (the drive record) through the SAME JSON marshal/unmarshal a
// session snapshot uses to persist and restore the drive (internal/sessionimage's DumpDir
// writes json.MarshalIndent(drive) to session.json; LoadDir reads it back into
// Image.Drive). So a State stamped with an envelope, serialized and deserialized, must
// expose that same envelope unchanged — the "a session snapshot exposes the envelope and
// survives dump/restore unchanged" done condition, proven at the drive-serialization seam.
func TestQualityEnvelopeSurvivesDriveRoundTrip(t *testing.T) {
	env := QualityEnvelope{
		Budget:         BudgetEnvelope{Budget: Budget{TurnsLeft: 20, TokensLeft: 200000}},
		WitnessPolicy:  "proof-by-default",
		DogfoodProbes:  []string{"quality-score", "milestone-score"},
		ScorecardCards: []string{"code_quality", "milestone_scorecard"},
	}
	st := DefaultState("sess-qenv")
	st.QualityEnvelope = env

	b, err := json.Marshal(st)
	if err != nil {
		t.Fatalf("marshal state: %v", err)
	}
	var back State
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("unmarshal state: %v", err)
	}
	if back.QualityEnvelope.IsZero() {
		t.Fatal("restored drive carries no quality envelope; want the stamped one")
	}
	if !reflect.DeepEqual(back.QualityEnvelope, env) {
		t.Fatalf("quality envelope changed across drive round-trip:\n got=%+v\n want=%+v", back.QualityEnvelope, env)
	}
}

// TestQualityEnvelopeAbsentIsWireCompatible anchors the omitzero contract: a drive with
// NO envelope must marshal without a quality_envelope key at all (byte-identical to a
// pre-#1964 State) and restore as Zero — never a phantom populated envelope.
func TestQualityEnvelopeAbsentIsWireCompatible(t *testing.T) {
	b, err := json.Marshal(DefaultState("sess-none"))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(b), "quality_envelope") {
		t.Fatalf("absent envelope leaked a quality_envelope key into the wire form: %s", b)
	}
	var back State
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !back.QualityEnvelope.IsZero() {
		t.Fatalf("absent envelope restored non-zero: %+v", back.QualityEnvelope)
	}
}

// TestQualityEnvelopeIsZero anchors the permissive-default meaning: an all-empty
// envelope is Zero (no QA controls declared), and any populated axis makes it non-zero.
func TestQualityEnvelopeIsZero(t *testing.T) {
	if !(QualityEnvelope{}).IsZero() {
		t.Fatal("empty QualityEnvelope should be Zero")
	}
	if (QualityEnvelope{WitnessPolicy: "proof-by-default"}).IsZero() {
		t.Fatal("a witness policy makes the envelope non-zero")
	}
	if (QualityEnvelope{ScorecardCards: []string{"code_quality"}}).IsZero() {
		t.Fatal("scorecard membership makes the envelope non-zero")
	}
}
