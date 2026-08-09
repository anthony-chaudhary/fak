package gateway

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/toolshape"
)

func contractedTool() []agent.ToolDef {
	return []agent.ToolDef{{
		Type: "function",
		Function: agent.ToolDefFunction{
			Name: "lookup",
			Parameters: json.RawMessage(`{
				"type":"object",
				"x-fak-result-contract":{
					"schema":"fak-result-contract/1",
					"outputs":[
						{"name":"count","type":"integer"},
						{"name":"summary","type":"string"}
					]
				}
			}`),
		},
	}}
}

func TestResultContractAdmissionMutationSuite(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want toolshape.ResultShapeError
	}{
		{name: "exact positive control", raw: `{"count":2,"summary":"ok"}`},
		{name: "missing", raw: `{"summary":"ok"}`, want: toolshape.ResultShapeMissing},
		{name: "additional", raw: `{"count":2,"summary":"ok","debug":"secret"}`, want: toolshape.ResultShapeAdditional},
		{name: "duplicate", raw: `{"count":2,"summary":"ok","summary":"again"}`, want: toolshape.ResultShapeDuplicate},
		{name: "fused", raw: `"ok:2"`, want: toolshape.ResultShapeFused},
		{name: "split", raw: `[{"summary":"ok"},{"count":2}]`, want: toolshape.ResultShapeSplit},
		{name: "ambiguous pairing", raw: `{"count":2,"summary":"ok"} {"count":3,"summary":"again"}`, want: toolshape.ResultShapeAmbiguous},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receipt, verdict, forwarded := resultContractAdmission("lookup", contractedTool(), tt.raw)
			if receipt == nil {
				t.Fatal("missing shape receipt")
			}
			if tt.want == toolshape.ResultShapeOK {
				if verdict != nil || forwarded != tt.raw {
					t.Fatalf("positive control verdict=%#v forwarded=%q", verdict, forwarded)
				}
				return
			}
			if verdict == nil || verdict.Kind != "QUARANTINE" || verdict.Reason != reasonResultShapeMismatch {
				t.Fatalf("verdict = %#v", verdict)
			}
			if receipt.Failure != tt.want {
				t.Fatalf("failure = %q, want %q", receipt.Failure, tt.want)
			}
			if forwarded == tt.raw || !strings.Contains(forwarded, `"_quarantined":true`) {
				t.Fatalf("mutant reached consumer: %q", forwarded)
			}
		})
	}
}

func TestResultContractAdmissionReceiptAndStubDoNotLeakValues(t *testing.T) {
	secret := "result-secret-38fc6a"
	receipt, verdict, forwarded := resultContractAdmission(
		"lookup", contractedTool(), `{"count":2,"summary":"ok","debug":"`+secret+`"}`,
	)
	if verdict == nil {
		t.Fatal("additional output was not refused")
	}
	encoded, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), secret) || strings.Contains(forwarded, secret) {
		t.Fatalf("value leaked through shape-only refusal: receipt=%s forwarded=%s", encoded, forwarded)
	}
}

func TestResultContractAdmissionIsOptIn(t *testing.T) {
	tools := []agent.ToolDef{{Function: agent.ToolDefFunction{Name: "lookup", Parameters: json.RawMessage(`{"type":"object"}`)}}}
	raw := "plain text remains compatible"
	receipt, verdict, forwarded := resultContractAdmission("lookup", tools, raw)
	if receipt != nil || verdict != nil || forwarded != raw {
		t.Fatalf("uncontracted tool changed: receipt=%#v verdict=%#v forwarded=%q", receipt, verdict, forwarded)
	}
}

func TestAdmitInboundResultsRefusesShapeMutantBeforeConsumerDelivery(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("result-contract-test", echoEngine{})
	s, err := New(Config{EngineID: "result-contract-test", Model: "m", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	messages := []agent.Message{
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "call-1", Type: "function", Function: agent.Func{Name: "lookup", Arguments: `{}`}}}},
		{Role: agent.RoleTool, ToolCallID: "call-1", Content: `{"count":2,"summary":"ok","debug":"consumer-secret"}`},
	}
	admissions, err := s.admitInboundResults(context.Background(), messages, contractedTool(), "trace-shape")
	if err != nil {
		t.Fatal(err)
	}
	if len(admissions) != 1 || admissions[0].Verdict.Kind != "QUARANTINE" {
		t.Fatalf("admissions = %#v", admissions)
	}
	if admissions[0].ResultShape == nil || admissions[0].ResultShape.Failure != toolshape.ResultShapeAdditional {
		t.Fatalf("shape receipt = %#v", admissions[0].ResultShape)
	}
	if strings.Contains(messages[1].Content, "consumer-secret") || !strings.Contains(messages[1].Content, `"_quarantined":true`) {
		t.Fatalf("consumer received raw mutant: %s", messages[1].Content)
	}
}
