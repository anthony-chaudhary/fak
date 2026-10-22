package agent

import (
	"encoding/json"
	"errors"
	"math"
	"testing"
)

var nativeFreeStringSchemaFormat = json.RawMessage(`{"type":"json_schema","json_schema":{"strict":true,"schema":{"type":"object","properties":{"note":{"type":"string"}},"required":["note"],"additionalProperties":false}}}`)

// fak-test:runtime fast est=40ms lane=default
func TestNativeJSONSchemaMasksSpecialTokensInStrings(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	stops := StopIDs(p.tok, p.m.Cfg)
	special := p.tok.SpecialTokens()
	if len(special) == 0 {
		t.Fatal("probe tokenizer declares no special tokens")
	}
	carriers := map[string]SampleParams{
		"response_format": {ResponseFormat: nativeFreeStringSchemaFormat},
		"guided_json": {GuidedDecode: map[string]json.RawMessage{
			"guided_json": json.RawMessage(`{"type":"object","properties":{"note":{"type":"string"}},"required":["note"],"additionalProperties":false}`),
		}},
	}
	for name, sp := range carriers {
		for _, tc := range []struct {
			history  string
			complete bool
		}{
			{"", false},
			{`{"note":"`, false},
			{`{"note":"x"}`, true},
		} {
			c, err := p.newNativeRequestConstraint(sp, stops)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			ids, err := p.tok.Encode(tc.history)
			if err != nil {
				t.Fatal(err)
			}
			c.history = ids
			effective, err := c.maskLogits(make([]float32, p.m.Cfg.VocabSize), stops)
			if err != nil {
				t.Fatalf("%s after %q: %v", name, tc.history, err)
			}
			for id := range special {
				if finite := !math.IsInf(float64(effective[id]), -1); finite != (tc.complete && id == c.stopID) {
					t.Fatalf("%s after %q: special token %d finite=%v, stop id %d", name, tc.history, id, finite, c.stopID)
				}
			}
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestNativeGuidedPairingReadsResponseFormatByExactKey(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	stops := StopIDs(p.tok, p.m.Cfg)
	guided := map[string]json.RawMessage{"guided_json": json.RawMessage(`{"type":"object"}`)}
	for _, format := range []json.RawMessage{
		json.RawMessage(`{"type":"json_schema","Type":"text","json_schema":{"schema":{"type":"array"}}}`),
		json.RawMessage(`{"Type":"json_schema","type":"text"}`),
		json.RawMessage(`{"type":"json_schema","json_schema":{"schema":{"type":"object"}},"JSON_SCHEMA":{"schema":{"type":"array"}}}`),
	} {
		_, err := p.newNativeRequestConstraint(SampleParams{ResponseFormat: format, GuidedDecode: guided}, stops)
		var carrier *nativeGuidedCarrierError
		if !errors.As(err, &carrier) && !errors.Is(err, ErrNativeStructuredUnsupported) {
			t.Fatalf("format %s with guided_json: err=%v, want a native structured refusal", format, err)
		}
	}
	if _, err := p.newNativeRequestConstraint(SampleParams{ResponseFormat: json.RawMessage(`{"type":"text"}`), GuidedDecode: guided}, stops); err != nil {
		t.Fatalf("plain text response_format with guided_json: %v", err)
	}
}
