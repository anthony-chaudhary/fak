package toolshape

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestResultContractMutationSuite(t *testing.T) {
	contract := ResultContract{
		Schema: resultContractVersion,
		Outputs: []ResultSpec{
			{Name: "summary", Type: "string"},
			{Name: "count", Type: "integer"},
		},
	}
	tests := []struct {
		name string
		raw  string
		want ResultShapeError
	}{
		{name: "exact positive control", raw: `{"summary":"ok","count":2}`},
		{name: "missing", raw: `{"summary":"ok"}`, want: ResultShapeMissing},
		{name: "additional", raw: `{"summary":"ok","count":2,"debug":"secret"}`, want: ResultShapeAdditional},
		{name: "duplicate", raw: `{"summary":"ok","summary":"again","count":2}`, want: ResultShapeDuplicate},
		{name: "fused", raw: `"ok:2"`, want: ResultShapeFused},
		{name: "split", raw: `[{"summary":"ok"},{"count":2}]`, want: ResultShapeSplit},
		{name: "wrong type", raw: `{"summary":"ok","count":"2"}`, want: ResultShapeType},
		{name: "ambiguous pairing", raw: `{"summary":"ok","count":2} {"summary":"again","count":3}`, want: ResultShapeAmbiguous},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receipt, err := contract.ValidateResult(tt.raw)
			if tt.want == ResultShapeOK {
				if err != nil {
					t.Fatalf("positive control refused: %v", err)
				}
				if receipt.Failure != ResultShapeOK {
					t.Fatalf("receipt failure = %q", receipt.Failure)
				}
				return
			}
			var failure *ResultShapeFailure
			if !errors.As(err, &failure) || failure.Kind != tt.want {
				t.Fatalf("failure = %#v, want %q", err, tt.want)
			}
			if receipt.Failure != tt.want {
				t.Fatalf("receipt failure = %q, want %q", receipt.Failure, tt.want)
			}
		})
	}
}

func TestResultContractDetectsDuplicateMembersBeforeJSONMapCollapse(t *testing.T) {
	contract := ResultContract{Schema: resultContractVersion, Outputs: []ResultSpec{{Name: "x", Type: "string"}}}
	_, err := contract.ValidateResult(`{"x":"one","x":"two"}`)
	var failure *ResultShapeFailure
	if !errors.As(err, &failure) || failure.Kind != ResultShapeDuplicate {
		t.Fatalf("failure = %#v, want duplicate", err)
	}
}

func TestResultShapeReceiptContainsNoValues(t *testing.T) {
	secret := "do-not-leak-7f942b"
	contract := ResultContract{Schema: resultContractVersion, Outputs: []ResultSpec{{Name: "answer", Type: "string"}}}
	receipt, err := contract.ValidateResult(`{"answer":"` + secret + `"}`)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), secret) {
		t.Fatalf("shape receipt leaked result content: %s", encoded)
	}
	if got := string(encoded); !strings.Contains(got, `"name":"answer"`) || !strings.Contains(got, `"type":"string"`) {
		t.Fatalf("shape receipt omitted shape: %s", encoded)
	}
}

func TestParseResultContractFromToolParameterSchema(t *testing.T) {
	raw := json.RawMessage(`{
		"type":"object",
		"properties":{"query":{"type":"string"}},
		"x-fak-result-contract":{
			"schema":"fak-result-contract/1",
			"outputs":[{"name":"count","type":"integer"},{"name":"summary","type":"string"}]
		}
	}`)
	contract, present, err := ParseResultContract(raw)
	if err != nil || !present {
		t.Fatalf("ParseResultContract() = %#v, %v, %v", contract, present, err)
	}
	if len(contract.Outputs) != 2 || contract.Outputs[0].Name != "count" || contract.Outputs[1].Name != "summary" {
		t.Fatalf("outputs = %#v", contract.Outputs)
	}
}

func TestParseResultContractRefusesInvalidDeclarations(t *testing.T) {
	tests := []string{
		`{"x-fak-result-contract":{"schema":"future/2","outputs":[{"name":"x","type":"string"}]}}`,
		`{"x-fak-result-contract":{"schema":"fak-result-contract/1","outputs":[]}}`,
		`{"x-fak-result-contract":{"schema":"fak-result-contract/1","outputs":[{"name":"x","type":"string"},{"name":"x","type":"string"}]}}`,
		`{"x-fak-result-contract":{"schema":"fak-result-contract/1","outputs":[{"name":"x","type":"bytes"}]}}`,
		`{"x-fak-result-contract":{"schema":"fak-result-contract/1","outputs":[{"name":"x","type":"string"}],"extension":true}}`,
	}
	for _, raw := range tests {
		if _, present, err := ParseResultContract(json.RawMessage(raw)); !present || err == nil {
			t.Errorf("ParseResultContract(%s) present=%v err=%v", raw, present, err)
		}
	}
}
