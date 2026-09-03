package gateway

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func FuzzFakAdjudicateReceiptInvariants(f *testing.F) {
	for _, seed := range []struct{ tool, args, trace string }{
		{"allow_read", `{}`, "trace"},
		{"deny_thing", `{"path":"C:\\private","secret":"sk-seed"}`, ""},
		{"transform_x", `{"x":1}`, "kept"},
		{"witness_ship", `null`, ""},
		{"", `{"token":"secret"}`, ""},
	} {
		f.Add(seed.tool, seed.args, seed.trace)
	}
	f.Fuzz(func(t *testing.T, tool, args, trace string) {
		if len(tool) > 96 || len(args) > 1024 || len(trace) > 96 || !json.Valid([]byte(args)) {
			t.Skip()
		}
		srv := newTestServer(t)
		request, _ := json.Marshal(map[string]any{"tool": tool, "arguments": json.RawMessage(args), "trace_id": trace})
		result, rpcErr := srv.callTool(context.Background(), json.RawMessage(`{"name":"fak_adjudicate","arguments":`+string(request)+`}`))
		var encoded []byte
		if rpcErr != nil {
			encoded, _ = json.Marshal(rpcErr)
			data, ok := rpcErr.Data.(AdjudicateReceipt)
			if !ok || data.Outcome != "failed" || data.Execution != "not_executed" || data.Error == nil {
				t.Fatalf("failure contract = %#v", rpcErr)
			}
		} else {
			encoded, _ = json.Marshal(result)
			blocks := result.(map[string]any)["content"].([]map[string]any)
			var response SyscallResponse
			if err := json.Unmarshal([]byte(blocks[0]["text"].(string)), &response); err != nil {
				t.Fatalf("decode: %v", err)
			}
			if response.Receipt == nil || response.Receipt.Execution != "not_executed" || response.Receipt.Provenance != "kernel_decide" {
				t.Fatalf("receipt = %+v", response.Receipt)
			}
			valid := map[string]bool{"allowed": true, "denied": true, "transformed": true, "witness_required": true}
			if !valid[response.Receipt.Outcome] {
				t.Fatalf("outcome = %q", response.Receipt.Outcome)
			}
			if (response.Verdict.Kind == "TRANSFORM") != (len(response.RepairedArguments) != 0) {
				t.Fatalf("kind=%q repaired=%q", response.Verdict.Kind, response.RepairedArguments)
			}
		}
		// Arguments are inputs to the kernel, never echoed by this pre-execution surface.
		for _, leak := range []string{"sk-seed", `C:\\private`} {
			if strings.Contains(string(encoded), leak) {
				t.Fatalf("response leaked fixture %q: %s", leak, encoded)
			}
		}
	})
}
