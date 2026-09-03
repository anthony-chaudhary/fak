package gateway

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func callFakAdjudicate(t *testing.T, srv *Server, arguments string) (SyscallResponse, *rpcError) {
	t.Helper()
	result, rpcErr := srv.callTool(context.Background(), json.RawMessage(`{"name":"fak_adjudicate","arguments":`+arguments+`}`))
	if rpcErr != nil {
		return SyscallResponse{}, rpcErr
	}
	blocks, ok := result.(map[string]any)["content"].([]map[string]any)
	if !ok || len(blocks) != 1 {
		t.Fatalf("MCP content = %#v", result)
	}
	var response SyscallResponse
	if err := json.Unmarshal([]byte(blocks[0]["text"].(string)), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	return response, nil
}

func assertAdjudicateReceipt(t *testing.T, receipt *AdjudicateReceipt, outcome string) {
	t.Helper()
	if receipt == nil {
		t.Fatal("receipt is nil")
	}
	if receipt.Schema != "fak-adjudicate-receipt/1" || receipt.Outcome != outcome {
		t.Fatalf("receipt = %+v", receipt)
	}
	if receipt.DurationNS < 0 || receipt.Execution != "not_executed" || receipt.Provenance != "kernel_decide" {
		t.Fatalf("receipt evidence = %+v", receipt)
	}
}

func TestFakAdjudicateReceiptOutcomesAndTrace(t *testing.T) {
	srv := newTestServer(t)
	cases := []struct {
		name, tool, outcome, kind string
		repaired                  bool
	}{
		{"allow", "allow_read", "allowed", "ALLOW", false},
		{"deny", "deny_thing", "denied", "DENY", false},
		{"transform", "transform_x", "transformed", "TRANSFORM", true},
		{"witness", "witness_ship", "witness_required", "REQUIRE_WITNESS", false},
		{"default-deny", "unknown_tool", "denied", "DENY", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			response, rpcErr := callFakAdjudicate(t, srv, `{"tool":"`+tc.tool+`","arguments":{"secret":"must-not-leak"},"trace_id":"kept-trace"}`)
			if rpcErr != nil {
				t.Fatalf("call: %+v", rpcErr)
			}
			if response.Verdict.Kind != tc.kind || response.TraceID != "kept-trace" || response.Result != nil {
				t.Fatalf("legacy response = %+v", response)
			}
			assertAdjudicateReceipt(t, response.Receipt, tc.outcome)
			if tc.repaired != (len(response.RepairedArguments) != 0) {
				t.Fatalf("repaired_arguments = %q, want present=%v", response.RepairedArguments, tc.repaired)
			}
			encoded, _ := json.Marshal(response)
			if strings.Contains(string(encoded), "must-not-leak") {
				t.Fatalf("response leaked raw arguments: %s", encoded)
			}
		})
	}

	minted, rpcErr := callFakAdjudicate(t, srv, `{"tool":"allow_read","arguments":{}}`)
	if rpcErr != nil || !strings.HasPrefix(minted.TraceID, "gw-") {
		t.Fatalf("minted trace response = %+v, error = %+v", minted, rpcErr)
	}
}

func TestFakAdjudicateFailureIsSanitized(t *testing.T) {
	srv := newTestServer(t)
	_, rpcErr := callFakAdjudicate(t, srv, `{"tool":"","arguments":{"path":"C:\\\\private\\\\secret","token":"sk-do-not-leak"}}`)
	if rpcErr == nil {
		t.Fatal("missing tool should fail")
	}
	if rpcErr.Message != "fak_adjudicate failed" {
		t.Fatalf("message = %q", rpcErr.Message)
	}
	encoded, _ := json.Marshal(rpcErr)
	for _, leak := range []string{"private", "secret", "sk-do-not-leak", "missing tool name"} {
		if strings.Contains(string(encoded), leak) {
			t.Fatalf("error leaked %q: %s", leak, encoded)
		}
	}
	data, ok := rpcErr.Data.(AdjudicateReceipt)
	if !ok {
		t.Fatalf("error data type = %T", rpcErr.Data)
	}
	assertAdjudicateReceipt(t, &data, "failed")
	if data.Error == nil || data.Error.Code != "invalid_arguments" || data.Error.Source != "gateway" {
		t.Fatalf("sanitized error = %+v", data.Error)
	}
}

func TestFakAdjudicateDiscoveryDocumentsReceiptAndNeverExecute(t *testing.T) {
	var found map[string]any
	for _, descriptor := range toolDescriptors() {
		if descriptor["name"] == "fak_adjudicate" {
			found = descriptor
			break
		}
	}
	if found == nil {
		t.Fatal("fak_adjudicate descriptor missing")
	}
	description, _ := found["description"].(string)
	for _, want := range []string{"WITHOUT executing", "fak-adjudicate-receipt/1", "not_executed", "kernel_decide", "Repaired arguments appear only for TRANSFORM"} {
		if !strings.Contains(description, want) {
			t.Fatalf("description missing %q: %q", want, description)
		}
	}
}
