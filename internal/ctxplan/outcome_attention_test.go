package ctxplan

import (
	"context"
	"reflect"
	"testing"
)

// outcome_attention_test.go — acceptance for issue #858 (production Outcome producer).
//
// These tests drive OutcomeFromAttention / LearnFromAttention with a known plan + attribution, so
// the witnessed classification (Hit / Wasted / Fault) and the closed learning loop are what is under
// test — not the attribution math (#853) nor the softmax (#852).

// TestOutcomeFromAttentionClassifies: each resident span lands in the right bucket by its witnessed
// mass; a pin is always a Hit; an unwitnessed resident span teaches nothing; a fault names only an
// elided span.
func TestOutcomeFromAttentionClassifies(t *testing.T) {
	p := Plan{
		Selected: []Selection{
			{ID: "A", Cost: 10},               // attended 0.5 -> Hit
			{ID: "B", Cost: 10},               // attended 0.0 -> Wasted
			{ID: "C", Cost: 10, Pinned: true}, // pinned, no mass -> Hit by construction
			{ID: "D", Cost: 10},               // not in attribution -> unaccounted (skipped)
		},
		Elided: []Elision{{ID: "E", Cost: 10}},
	}
	attr := Attribution{"A": 0.5, "B": 0.0}
	faults := []string{"E", "stray"} // stray names no elided span -> dropped

	o := OutcomeFromAttention(p, attr, faults, DefaultHitThreshold)

	if !reflect.DeepEqual(o.Hits, []string{"A", "C"}) {
		t.Errorf("Hits = %v, want [A C] (attended + pinned)", o.Hits)
	}
	if !reflect.DeepEqual(o.Wasted, []string{"B"}) {
		t.Errorf("Wasted = %v, want [B] (witnessed but cold)", o.Wasted)
	}
	if !reflect.DeepEqual(o.Faults, []string{"E"}) {
		t.Errorf("Faults = %v, want [E] (stray filtered to elided ids)", o.Faults)
	}
	// D was never named: it must appear in no bucket (fail-closed, no fabricated label).
	for _, b := range [][]string{o.Hits, o.Wasted, o.Faults} {
		for _, id := range b {
			if id == "D" {
				t.Errorf("unwitnessed span D was labeled %v — must teach nothing", b)
			}
		}
	}
}

// TestWitnessDiffersFromLexical: a span the forecast LEXICALLY predicts (relevance > 0, so the
// lexical-overlap producer would call it a Hit) but that attention witnessed at ~0 is classified
// Wasted — proving the witness sees what lexical overlap misses, the #858 acceptance.
func TestWitnessDiffersFromLexical(t *testing.T) {
	f := Forecast{Intents: []string{"auth token rotation"}}
	// A resident span whose descriptor overlaps the forecast intents (lexical Hit) ...
	lexHit := Span{ID: "lex", Role: "tool", Descriptor: "auth token rotation runbook"}
	if f.relevance(lexHit) <= 0 {
		t.Fatalf("precondition: the lexical producer must predict this span (relevance>0), got %v", f.relevance(lexHit))
	}
	p := Plan{Selected: []Selection{{ID: "lex", Cost: 20}}}
	// ... but the turn never actually attended to it (witnessed mass 0).
	o := OutcomeFromAttention(p, Attribution{"lex": 0.0}, nil, DefaultHitThreshold)

	if contains(o.Hits, "lex") {
		t.Errorf("witnessed Outcome wrongly marked the lexically-plausible span a Hit: %v", o.Hits)
	}
	if !contains(o.Wasted, "lex") {
		t.Errorf("witnessed Outcome should mark the unattended span Wasted (lexical overlap would Hit), got Wasted=%v", o.Wasted)
	}
}

// TestLearnFromAttentionClosesLoop: feeding a witnessed Outcome (a fault) through LearnFromAttention
// revises the forecast — the span the turn faulted is promoted into the next prediction — proving the
// planner learns on the witnessed signal. A no-witness, no-fault call is a deterministic no-op.
func TestLearnFromAttentionClosesLoop(t *testing.T) {
	store := NewMemStore()
	store.Add("Read", DurabilitySession, []byte("the gamma delta incident report"), false) // span:0
	store.Add("Read", DurabilitySession, []byte("auth token rotation policy"), false)      // span:1
	spans, _ := store.Spans(context.Background())

	f := Forecast{Intents: []string{"auth token rotation"}}
	// A span whose content the forecast does NOT yet predict (the gamma report).
	probe := Span{ID: "probe", Role: "tool", Descriptor: "the gamma delta runbook"}
	before := f.relevance(probe)

	// Plan: span:1 resident & attended (a Hit), span:0 elided & demand-paged back (a Fault).
	p := Plan{
		Selected: []Selection{{ID: "span:1", Cost: 10}},
		Elided:   []Elision{{ID: "span:0", Cost: 10}},
	}
	attr := Attribution{"span:1": 0.9}
	lf, lw := LearnFromAttention(f, DefaultWeights(), p, spans, attr, []string{"span:0"}, DefaultHitThreshold, 0)

	if lf.relevance(probe) <= before {
		t.Errorf("the learned forecast must now predict the faulted span's content; before=%v after=%v", before, lf.relevance(probe))
	}
	// The weights moved off the seed too (a Hit and a Fault are positive labels): the loop is closed
	// on the witnessed outcome, not just the forecast intents.
	if lw == DefaultWeights() {
		t.Errorf("weights should be tuned from the witnessed hits/faults, got the unchanged seed %v", lw)
	}

	// No witness, no fault -> nothing to learn: forecast intents and weights are unchanged.
	nf, nw := LearnFromAttention(f, DefaultWeights(), Plan{Selected: []Selection{{ID: "span:1", Cost: 10}}}, spans, nil, nil, DefaultHitThreshold, 0)
	if !reflect.DeepEqual(nf.Intents, f.Intents) {
		t.Errorf("no-witness call changed forecast intents: %v != %v", nf.Intents, f.Intents)
	}
	if nw != DefaultWeights() {
		t.Errorf("no-witness call changed weights: %v", nw)
	}
}

// TestOutcomeFromAttentionShadowPure: the producer is pure — it mutates neither the plan nor the
// attribution, and the same inputs yield the same Outcome (replayable, the shadow-recording bar).
func TestOutcomeFromAttentionShadowPure(t *testing.T) {
	p := Plan{
		Selected: []Selection{{ID: "A", Cost: 10}, {ID: "B", Cost: 10}},
		Elided:   []Elision{{ID: "E", Cost: 10}},
	}
	attr := Attribution{"A": 0.7, "B": 0.0}
	faults := []string{"E"}

	o1 := OutcomeFromAttention(p, attr, faults, DefaultHitThreshold)
	o2 := OutcomeFromAttention(p, attr, faults, DefaultHitThreshold)
	if !reflect.DeepEqual(o1, o2) {
		t.Errorf("non-deterministic Outcome: %+v != %+v", o1, o2)
	}
	// Inputs untouched (shadow: producing an outcome changes nothing the planner reads).
	if len(p.Selected) != 2 || len(attr) != 2 || len(faults) != 1 {
		t.Errorf("producer mutated its inputs: plan=%d attr=%d faults=%d", len(p.Selected), len(attr), len(faults))
	}
}

func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}
