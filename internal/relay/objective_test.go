package relay

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// C5 (issue #1874): the standing objective must cross a relay boundary byte-identical.
// Goal drift is a named failure mode, so the ObjectivePin the Baton carries (pin_id +
// text + digest) must survive Marshal -> Parse unchanged, and the successor must still be
// able to Verify() it. The binding itself is structural — Baton.Objective is a
// ctxplan.ObjectivePin (C1) and the C2 codec round-trips it — so these tests are the
// witness the rung is defined by (run: `go test ./internal/relay -run ObjectivePin`).

// TestObjectivePinSurvivesBatonRoundTrip asserts the digest (and its pin_id/text inputs)
// are byte-identical across a baton round-trip, that the wire form actually embeds the
// digest, and that the re-read pin still self-verifies.
func TestObjectivePinSurvivesBatonRoundTrip(t *testing.T) {
	pin := ctxplan.NewObjectivePin("pin-1874", "Carry the objective across the relay boundary unchanged.", 5)
	if !pin.Verify() {
		t.Fatalf("precondition: a freshly minted pin must verify against its own pin_id+text: %+v", pin)
	}
	b := Baton{Schema: Schema, RelayID: "RLY-1874", Objective: pin}

	data, err := Marshal(b)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	got, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	if got.Objective.Digest != pin.Digest {
		t.Errorf("objective digest drifted across the baton boundary:\n before=%q\n after =%q", pin.Digest, got.Objective.Digest)
	}
	if got.Objective.PinID != pin.PinID || got.Objective.Text != pin.Text {
		t.Errorf("objective pin_id/text not carried verbatim:\n before=%+v\n after =%+v", pin, got.Objective)
	}
	if !got.Objective.Verify() {
		t.Errorf("round-tripped objective must still Verify() (digest matches its own pin_id+text): %+v", got.Objective)
	}
	if want := `"digest":"` + pin.Digest + `"`; !strings.Contains(string(data), want) {
		t.Errorf("baton wire form must embed the objective digest %s; got %s", want, data)
	}
}

// TestObjectivePinTamperDetectedAcrossRoundTrip guards the failure mode itself: if the
// carried objective text is mutated on the wire without recomputing the digest, the
// re-read pin must FAIL Verify(). The boundary cannot silently launder goal drift — a
// stale digest is corrupt baton input to fail closed on, exactly as the schema's reader
// contract requires.
func TestObjectivePinTamperDetectedAcrossRoundTrip(t *testing.T) {
	pin := ctxplan.NewObjectivePin("pin-1874", "Original objective text.", 5)
	b := Baton{Schema: Schema, RelayID: "RLY-1874", Objective: pin}
	data, err := Marshal(b)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	// Swap the text but leave the (now stale) digest in place.
	tampered := strings.Replace(string(data), "Original objective text.", "Swapped objective text!!", 1)
	if tampered == string(data) {
		t.Fatal("test setup: expected to rewrite the objective text on the wire")
	}
	got, err := Parse([]byte(tampered))
	if err != nil {
		t.Fatalf("Parse(tampered): %v", err)
	}
	if got.Objective.Text != "Swapped objective text!!" {
		t.Fatalf("tamper did not take effect: %q", got.Objective.Text)
	}
	if got.Objective.Verify() {
		t.Error("a tampered objective (text changed, digest stale) must not Verify() after the round-trip")
	}
}
