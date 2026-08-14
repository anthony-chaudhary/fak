package humanctl

import (
	"testing"
	"time"
)

func TestSameIntentSupportsDistinctDelivery(t *testing.T) {
	now := time.Date(2026, 8, 14, 12, 0, 0, 0, time.UTC)
	base := Envelope{
		Instruction: Instruction{Verb: Redirect, Target: "the proof path"},
		Addressee:   Addressee{Kind: AddresseeSession, Cardinality: CardinalityOne, IDs: []string{"s-1"}},
		Lifetime:    Lifetime{Duration: DurationTurn},
		Outcome:     Outcome{Receipt: ReceiptAcknowledged, Admission: AdmissionAccepted, Effect: EffectPending},
	}
	for _, delivery := range []Delivery{DeliveryImmediate, DeliveryNextSafePoint, DeliveryNextTurn, DeliveryQueued} {
		e := base
		e.Delivery = delivery
		if err := e.ValidateAt(now); err != nil {
			t.Fatalf("%s changed valid semantic intent: %v", delivery, err)
		}
		if e.Instruction.Verb != Redirect {
			t.Fatalf("delivery %s changed verb to %s", delivery, e.Instruction.Verb)
		}
	}
}

func TestAcceptanceIsNotEffect(t *testing.T) {
	now := time.Date(2026, 8, 14, 12, 0, 0, 0, time.UTC)
	e := validEnvelope(now)
	e.Outcome = Outcome{Receipt: ReceiptAcknowledged, Admission: AdmissionAccepted, Effect: EffectObserved}
	if err := e.ValidateAt(now); err == nil {
		t.Fatal("accepted control reported observed effect without witness")
	}
	e.Outcome.EffectWitness = "session journal epoch 42 read-back"
	if err := e.ValidateAt(now); err != nil {
		t.Fatal(err)
	}
}

func TestEnvelopeRejectsInvalidCombinations(t *testing.T) {
	now := time.Date(2026, 8, 14, 12, 0, 0, 0, time.UTC)
	tests := map[string]func(*Envelope){
		"draft addressed": func(e *Envelope) { e.Delivery = DeliveryDraft },
		"cohort of one": func(e *Envelope) {
			e.Addressee = Addressee{Kind: AddresseeCohort, Cardinality: CardinalityMany, IDs: []string{"one"}}
		},
		"fleet selected IDs": func(e *Envelope) {
			e.Addressee = Addressee{Kind: AddresseeFleet, Cardinality: CardinalityAll, IDs: []string{"f-1"}}
		},
		"past expiry": func(e *Envelope) {
			e.Lifetime = Lifetime{Duration: DurationUntilExpiry, ExpiresAt: now.Add(-time.Second)}
		},
		"rejected with effect": func(e *Envelope) {
			e.Outcome = Outcome{Receipt: ReceiptAcknowledged, Admission: AdmissionRejected, Effect: EffectPending}
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			e := validEnvelope(now)
			mutate(&e)
			if err := e.ValidateAt(now); err == nil {
				t.Fatalf("invalid envelope accepted: %#v", e)
			}
		})
	}
}

func TestDraftHasNoRuntimeOutcome(t *testing.T) {
	now := time.Date(2026, 8, 14, 12, 0, 0, 0, time.UTC)
	e := Envelope{
		Instruction: Instruction{Verb: FlagConcern, Text: "this seems wrong but I do not know why"},
		Delivery:    DeliveryDraft,
		Lifetime:    Lifetime{Duration: DurationOnce},
	}
	if err := e.ValidateAt(now); err != nil {
		t.Fatal(err)
	}
}

func TestSessionControlWireMapping(t *testing.T) {
	want := map[string]Verb{
		"CONTINUE": Continue, "END_TURN": EndTurn,
		"PAUSE_SESSION": Pause, "STOP_SESSION": Stop,
	}
	for token, verb := range want {
		instruction, ok := InstructionFromSessionDecision(token)
		if !ok || instruction.Verb != verb {
			t.Fatalf("%s mapped to %#v, %v; want %s", token, instruction, ok, verb)
		}
		if err := instruction.Validate(); err != nil {
			t.Fatalf("mapped %s is invalid: %v", token, err)
		}
	}
	if _, ok := InstructionFromSessionDecision("MALFORMED"); ok {
		t.Fatal("refusal reason crossed into session control mapping")
	}
}

func validEnvelope(now time.Time) Envelope {
	return Envelope{
		Instruction: Instruction{Verb: Verify, Target: "the shipped effect"},
		Delivery:    DeliveryNextSafePoint,
		Addressee:   Addressee{Kind: AddresseeSubagent, Cardinality: CardinalityOne, IDs: []string{"worker-7"}},
		Lifetime:    Lifetime{Duration: DurationUntilExpiry, ExpiresAt: now.Add(time.Hour)},
		Outcome:     Outcome{Receipt: ReceiptAcknowledged, Admission: AdmissionAccepted, Effect: EffectPending},
	}
}
