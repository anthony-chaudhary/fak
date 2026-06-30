package codexmcphealth

import (
	"encoding/json"
	"strings"
	"testing"
)

func smokeStdout(t *testing.T, verdict, status, engine, body string) string {
	t.Helper()
	inner := map[string]any{
		"verdict": map[string]any{"kind": verdict, "by": "monitor"},
		"result": map[string]any{
			"status":  status,
			"content": mustJSON(t, map[string]any{"content": body, "file_path": "VERSION"}),
			"meta":    map[string]any{"engine": engine},
		},
		"trace_id": "gw-2",
	}
	initLine := mustJSON(t, map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"result":  map[string]any{"serverInfo": map[string]any{"name": "fak-gateway", "version": "0.34.0"}},
	})
	callLine := mustJSON(t, map[string]any{
		"jsonrpc": "2.0",
		"id":      2,
		"result": map[string]any{
			"content": []any{map[string]any{"type": "text", "text": mustJSON(t, inner)}},
			"isError": false,
		},
	})
	return initLine + "\n" + callLine + "\n"
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestParseSmokeOutputOK(t *testing.T) {
	res := ParseSmokeOutput(smokeStdout(t, "ALLOW", "OK", "fakread", "0.34.0\n"), "0.34.0")
	if !res.OK {
		t.Fatalf("OK = false, reason %q", res.Reason)
	}
	if res.Verdict != "ALLOW" || res.Status != "OK" || res.Engine != "fakread" || !strings.Contains(res.Content, "0.34.0") {
		t.Fatalf("unexpected smoke result: %+v", res)
	}
}

func TestParseSmokeOutputMismatch(t *testing.T) {
	res := ParseSmokeOutput(smokeStdout(t, "DENY", "OK", "fakread", "0.34.0\n"), "0.34.0")
	if res.OK {
		t.Fatal("OK = true on DENY response")
	}
	if !strings.Contains(res.Reason, "DENY") {
		t.Fatalf("reason %q does not name verdict", res.Reason)
	}
}

func TestParseSmokeOutputJSONRPCError(t *testing.T) {
	res := ParseSmokeOutput(`{"jsonrpc":"2.0","id":2,"error":{"code":-32602,"message":"boom"}}`+"\n", "0.34.0")
	if res.OK {
		t.Fatal("OK = true on JSON-RPC error")
	}
	if !strings.Contains(res.Reason, "JSON-RPC error") {
		t.Fatalf("reason = %q", res.Reason)
	}
}

func TestParseSmokeOutputNoCallResponse(t *testing.T) {
	res := ParseSmokeOutput(`{"jsonrpc":"2.0","id":1,"result":{}}`+"\n", "0.34.0")
	if res.OK {
		t.Fatal("OK = true with no id=2 response")
	}
	if !strings.Contains(res.Reason, "id=2") {
		t.Fatalf("reason = %q", res.Reason)
	}
}

func TestParsePowerShellInventory(t *testing.T) {
	payload := mustJSON(t, []map[string]any{
		{"ProcessId": 111, "CommandLine": `C:\work\fak\fak.exe serve --stdio --policy x.json`},
		{"ProcessId": 222, "CommandLine": `C:\work\fak\fak.exe guard --split=auto`},
		{"ProcessId": 333, "CommandLine": `C:\work\fak\fak.exe serve --stdio`},
	})
	rows := ParsePowerShellInventory(payload)
	if len(rows) != 2 || rows[0].PID != 111 || rows[1].PID != 333 {
		t.Fatalf("rows = %+v, want pids 111 and 333", rows)
	}
	if !strings.Contains(rows[0].Command, "serve --stdio") {
		t.Fatalf("command not preserved: %+v", rows[0])
	}
}

func TestParsePowerShellInventorySingleObject(t *testing.T) {
	rows := ParsePowerShellInventory(mustJSON(t, map[string]any{
		"ProcessId":   999,
		"CommandLine": "fak.exe serve --stdio --policy p",
	}))
	if len(rows) != 1 || rows[0].PID != 999 {
		t.Fatalf("rows = %+v, want pid 999", rows)
	}
}

func TestParsePowerShellInventoryEmpty(t *testing.T) {
	if rows := ParsePowerShellInventory(""); len(rows) != 0 {
		t.Fatalf("rows = %+v, want empty", rows)
	}
}

func TestClassifyTransportDeadServerOK(t *testing.T) {
	smoke := ParseSmokeOutput(smokeStdout(t, "ALLOW", "OK", "fakread", "0.34.0\n"), "0.34.0")
	diag := Classify(smoke, false, nil, "")
	if diag.State != TransportDeadServerOK {
		t.Fatalf("state = %s, want %s", diag.State, TransportDeadServerOK)
	}
	if !strings.Contains(strings.ToLower(diag.NextStep), "reconnect") {
		t.Fatalf("next step = %q", diag.NextStep)
	}
}

func TestClassifyServerDeadDominates(t *testing.T) {
	diag := Classify(SmokeResult{OK: false, Reason: "timeout"}, false, []ChildProc{{PID: 42, Command: "fak.exe serve --stdio"}}, "")
	if diag.State != ServerDead {
		t.Fatalf("state = %s, want %s", diag.State, ServerDead)
	}
}

func TestClassifyStaleChildren(t *testing.T) {
	smoke := ParseSmokeOutput(smokeStdout(t, "ALLOW", "OK", "fakread", "0.34.0\n"), "0.34.0")
	diag := Classify(smoke, true, []ChildProc{{PID: 111, Command: "fak.exe serve --stdio"}}, "")
	if diag.State != StaleChildren {
		t.Fatalf("state = %s, want %s", diag.State, StaleChildren)
	}
	if len(diag.StaleChildren) != 1 || diag.StaleChildren[0].PID != 111 {
		t.Fatalf("children = %+v", diag.StaleChildren)
	}
}

func TestClassifyReconnectOK(t *testing.T) {
	smoke := ParseSmokeOutput(smokeStdout(t, "ALLOW", "OK", "fakread", "0.34.0\n"), "0.34.0")
	diag := Classify(smoke, true, nil, "")
	if diag.State != ReconnectOK {
		t.Fatalf("state = %s, want %s", diag.State, ReconnectOK)
	}
}
