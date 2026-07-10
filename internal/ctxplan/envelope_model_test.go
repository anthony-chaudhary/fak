package ctxplan

import "testing"

func TestEnvelopeForModel(t *testing.T) {
	generic := GenericTurnEnvelope()
	cases := []struct {
		model, provenance string
		cap               int
	}{
		{"fable", ProvenanceModeled, 64000},
		{"claude-3-5-haiku", ProvenanceModeled, 96000},
		{"gpt-5.3-codex", ProvenanceModeled, 272000},
		{"claude-opus-4-8", ProvenanceModeled, generic.HardContextCap},
		{"unknown-future-model", generic.Provenance, generic.HardContextCap},
	}
	for _, c := range cases {
		t.Run(c.model, func(t *testing.T) {
			got := EnvelopeForModel(c.model)
			if got.Provenance != c.provenance || got.HardContextCap != c.cap {
				t.Fatalf("got provenance=%q cap=%d", got.Provenance, got.HardContextCap)
			}
		})
	}
}
