package gateway

// tooldefer_no_bypass_test.go — the #3534 security regression (epic #3229):
// cold-tool deferral must never become an adjudication bypass. deferColdTools
// (messages_tooldefer.go) is a PROVIDER-SIDE context-loading hint: it marks a cold
// custom def `defer_loading:true` on the OUTBOUND Anthropic tools[] so the provider
// faults the schema in on demand. It is orthogonal to BOTH adjudication seams —
//   * proposal side: adjudicateProposed keys on the tool NAME in the assistant's
//     tool_use; it never reads the request tools[] or the defer_loading flag.
//   * result side:   admit keys on the RESULT bytes (poison screen + source stamp);
//     it never consults the tool's def either.
// so a tool whose schema was deferred is STILL denied when policy denies it AND its
// poisoned result is STILL quarantined. These tests drive the REAL adjudicator and the
// REAL result-side stack (not doubles) to lock that no future "optimization" pairs the
// defer hint with a trust grant. The mechanism/determinism live in
// messages_tooldefer_test.go; this file is the no-bypass witness the #3537 flip is gated on.

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/ifc"
)

// deferBodyWith builds a minimal Anthropic Messages body carrying one hot tool (Read,
// kept verbatim) plus the named cold custom tools (each deferred when the transform fires).
func deferBodyWith(coldTools ...string) []byte {
	tools := []map[string]any{
		{"name": "Read", "description": "read a file", "input_schema": map[string]any{"type": "object"}},
	}
	for _, name := range coldTools {
		tools = append(tools, map[string]any{
			"name": name, "description": "cold custom tool", "input_schema": map[string]any{"type": "object"},
		})
	}
	body := map[string]any{
		"model":      "claude-opus-4-8",
		"max_tokens": 64,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
		"tools":      tools,
	}
	raw, _ := json.Marshal(body)
	return raw
}

// deferredToolDefs returns the tools[] of an Armed body as decoded objects.
func deferredToolDefs(t *testing.T, armed []byte) []map[string]json.RawMessage {
	t.Helper()
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(armed, &obj); err != nil {
		t.Fatalf("armed body not a JSON object: %v", err)
	}
	var elems []json.RawMessage
	if err := json.Unmarshal(obj["tools"], &elems); err != nil {
		t.Fatalf("armed tools[] undecodable: %v", err)
	}
	defs := make([]map[string]json.RawMessage, len(elems))
	for i, el := range elems {
		if err := json.Unmarshal(el, &defs[i]); err != nil {
			t.Fatalf("armed tool[%d] undecodable: %v", i, err)
		}
	}
	return defs
}

// findDef returns the decoded def with the given name, or nil.
func findDef(defs []map[string]json.RawMessage, name string) map[string]json.RawMessage {
	for _, d := range defs {
		if rawStringField(d, "name") == name {
			return d
		}
	}
	return nil
}

// TestDeferredColdToolStillAdjudicatedOnProposal proves the transform MARKS a cold tool
// deferred while keeping it in the adjudicator's namespace, and that a proposed CALL to
// that tool gets the SAME policy verdict it would without deferral — DENY for a denied
// tool, ALLOW for an allowed one. Deferral moves no verdict in either direction.
func TestDeferredColdToolStillAdjudicatedOnProposal(t *testing.T) {
	arms := DeferColdToolsAB(deferBodyWith("deny_exfil", "allow_read_cold"))
	if !arms.Changed {
		t.Fatalf("transform stood down (%s); want it to fire on the cold tools", arms.Reason)
	}
	if arms.ColdCount != 2 {
		t.Fatalf("ColdCount = %d, want 2 (deny_exfil + allow_read_cold)", arms.ColdCount)
	}

	// Both cold defs are still PRESENT in the deferred tools[] (marked, never dropped) —
	// deferral leaves the adjudicator's tool namespace unchanged.
	defs := deferredToolDefs(t, arms.Armed)
	for _, name := range []string{"deny_exfil", "allow_read_cold"} {
		d := findDef(defs, name)
		if d == nil {
			t.Fatalf("cold tool %q vanished from deferred tools[] — deferral must mark, not drop", name)
		}
		if got := strings.TrimSpace(string(d["defer_loading"])); got != "true" {
			t.Fatalf("cold tool %q defer_loading = %q, want true", name, got)
		}
	}

	// Drive the REAL adjudicator. adjudicateProposed takes only the proposed calls — it
	// never sees the request tools[] — so the deferred def cannot influence its verdict.
	srv := newTestServer(t)
	ctx := context.Background()
	calls := []agent.ToolCall{
		{ID: "c-deny", Type: "function", Function: agent.Func{Name: "deny_exfil", Arguments: `{"url":"http://evil"}`}},
		{ID: "c-allow", Type: "function", Function: agent.Func{Name: "allow_read_cold", Arguments: `{"path":"x"}`}},
	}
	kept, adjs, dropped := srv.adjudicateProposed(ctx, calls, "trace-defer-nobypass")

	byTool := map[string]ToolAdjudication{}
	for _, a := range adjs {
		byTool[a.Tool] = a
	}
	// The deferred, policy-denied tool is STILL denied — no bypass.
	if a := byTool["deny_exfil"]; a.Admitted || a.Verdict.Kind != "DENY" || a.Verdict.Reason != "POLICY_BLOCK" {
		t.Fatalf("deferred deny_exfil: admitted=%v verdict=%+v, want DENY/POLICY_BLOCK not admitted", a.Admitted, a.Verdict)
	}
	// ...and the deferred, policy-allowed tool is STILL allowed — deferral is not a false-deny either.
	if a := byTool["allow_read_cold"]; !a.Admitted || a.Verdict.Kind != "ALLOW" {
		t.Fatalf("deferred allow_read_cold: admitted=%v verdict=%+v, want ALLOW admitted", a.Admitted, a.Verdict)
	}
	if dropped != 1 {
		t.Fatalf("dropped = %d, want exactly 1 (deny_exfil)", dropped)
	}
	if len(kept) != 1 || kept[0].Function.Name != "allow_read_cold" {
		t.Fatalf("kept = %v, want only allow_read_cold survive to the wire", kept)
	}
}

// TestDeferredColdToolResultStillQuarantined proves the result-side seam: a poisoned
// RESULT returned by a tool whose schema was deferred is STILL quarantined and paged out
// of context. admit keys on the result bytes, never on the def, so deferral is no poison
// screen bypass. Drives the REAL result-side stack (ctxmmu poison screen + IFC stamp).
func TestDeferredColdToolResultStillQuarantined(t *testing.T) {
	// First: the tool's schema really was deferred out of the eager context.
	arms := DeferColdToolsAB(deferBodyWith("fetch_cold"))
	if !arms.Changed {
		t.Fatalf("transform stood down (%s); want it to fire", arms.Reason)
	}
	if d := findDef(deferredToolDefs(t, arms.Armed), "fetch_cold"); d == nil || strings.TrimSpace(string(d["defer_loading"])) != "true" {
		t.Fatalf("fetch_cold was not deferred; def=%v", d)
	}

	// The REAL result-side stack, mirroring admit_test.go: ctxmmu quarantine (rank 10) +
	// IFC source-stamp over a readable ledger (rank 20).
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	led := ifc.NewLedger()
	abi.RegisterResultAdmitter(10, ctxmmu.New())
	abi.RegisterResultAdmitter(20, ifc.NewStampGate(led, ifc.Policy{}))
	srv, err := New(Config{EngineID: "test", Model: "m", VDSO: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)

	const trace = "trace-defer-quarantine"
	const secret = "sk-abcdef0123456789abcdef0123"
	poison := `{"page":"config loaded. api_key=` + secret + ` was found in env"}`

	// admit the poisoned result attributed to the DEFERRED tool.
	wv, env, err := srv.admit(context.Background(), "fetch_cold", poison, "", trace)
	if err != nil {
		t.Fatalf("admit: %v", err)
	}
	if wv.Kind != "QUARANTINE" {
		t.Fatalf("deferred-tool poisoned result: verdict = %q, want QUARANTINE — deferral must not bypass the poison screen", wv.Kind)
	}
	if env == nil || env.Meta["admit"] != "quarantined" {
		t.Fatalf("deferred-tool result must be admit-quarantined, got meta %v", envMeta(env))
	}
	if env != nil && strings.Contains(env.Content, secret) {
		t.Fatalf("quarantined deferred-tool result still leaks the secret into context: %q", env.Content)
	}
	if led.Level(trace) == abi.TaintTrusted {
		t.Fatalf("IFC ledger for %q stayed Trusted — the deferred tool's result skipped the source stamp", trace)
	}
}

// TestDeferAddsNoAdmissionBypassKey is the pure structural backstop: the ONLY key the
// transform adds to a cold def is defer_loading, and the ONLY element it appends is a
// normal named tool_search_tool. No trust/allowlist/skip-adjudication key may appear —
// a future edit that pairs the defer hint with any such grant turns this red.
func TestDeferAddsNoAdmissionBypassKey(t *testing.T) {
	input := deferBodyWith("deny_exfil")
	arms := DeferColdToolsAB(input)
	if !arms.Changed {
		t.Fatalf("transform stood down (%s)", arms.Reason)
	}

	// No admission-relevant grant may appear anywhere in the armed body.
	lowered := strings.ToLower(string(arms.Armed))
	for _, banned := range []string{
		"skip_adjudication", "no_gate", "no_adjudicat", "allowlist", "allow_list",
		"bypass", "\"trusted\"", "\"exempt\"", "\"privileged\"", "\"untrusted\":false",
	} {
		if strings.Contains(lowered, banned) {
			t.Fatalf("deferred body carries admission-bypass token %q — deferral must grant no trust:\n%s", banned, arms.Armed)
		}
	}

	// The cold def gained EXACTLY defer_loading (plus its original keys); no extra keys.
	inDef := findDef(deferredToolDefs(t, input), "deny_exfil")
	outDef := findDef(deferredToolDefs(t, arms.Armed), "deny_exfil")
	if inDef == nil || outDef == nil {
		t.Fatalf("deny_exfil missing (in=%v out=%v)", inDef, outDef)
	}
	for k := range outDef {
		if _, had := inDef[k]; !had && k != "defer_loading" {
			t.Fatalf("deferred deny_exfil gained unexpected key %q (only defer_loading is allowed)", k)
		}
	}

	// The single appended element is a real, named, adjudicable tool — not an allowlist.
	defs := deferredToolDefs(t, arms.Armed)
	last := defs[len(defs)-1]
	if rawStringField(last, "name") != toolSearchToolName {
		t.Fatalf("appended tail = %q, want the named tool_search_tool", rawStringField(last, "name"))
	}
}
