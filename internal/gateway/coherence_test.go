package gateway

// coherence_test.go — #4: the cross-agent coherence bus + revocation productionized
// through `fak serve`. The feed subscribes to the process-global vdso.Default, so these
// tests drive that bus directly (the ResetForTest kernel does not re-register the vDSO
// as an emitter) and assert the wire surfaces observe it. Unique witnesses/tools isolate
// each assertion from any peer event already on the shared bus.

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// emit a witnessed read completion into the global vDSO so an entry is admitted under a
// witness this test can then revoke.
func fillWitnessed(tool, witness, payload string) {
	c := &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"id":"` + witness + `"}`)},
		Meta: map[string]string{"readOnlyHint": "true", "idempotentHint": "true", "witness": witness},
	}
	r := &abi.Result{Call: c, Status: abi.StatusOK, Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(payload)}}
	vdso.Default.Emit(abi.Event{Kind: abi.EvComplete, Call: c, Result: r})
}

func findRevocation(events []CoherenceEvent, witness string) *CoherenceEvent {
	for i := range events {
		if events[i].Kind == "revocation" && events[i].Witness == witness {
			return &events[i]
		}
	}
	return nil
}

func findMutation(events []CoherenceEvent, tool string) *CoherenceEvent {
	for i := range events {
		if events[i].Kind == "mutation" && events[i].Tool == tool {
			return &events[i]
		}
	}
	return nil
}

// buildCall threads an external witness into the kernel call's Meta, so a wire read is
// keyed for cross-agent dedup and bound for causal revocation.
func TestBuildCall_WitnessRoundTrips(t *testing.T) {
	srv := newTestServer(t)
	tc, err := srv.buildCall(context.Background(), "get_doc", `{"id":"x"}`, true, "commit-abc", "trace-xyz")
	if err != nil {
		t.Fatalf("buildCall: %v", err)
	}
	if tc.Meta["witness"] != "commit-abc" {
		t.Errorf("witness not threaded into call meta: %q", tc.Meta["witness"])
	}
	if tc.TraceID != "trace-xyz" {
		t.Errorf("TraceID not threaded onto the served call: %q", tc.TraceID)
	}
	// An empty witness must not inject the key (stays v0.1 consistency-only); an
	// empty TraceID must be MINTED to a non-empty id (never the shared "" trace).
	tc2, _ := srv.buildCall(context.Background(), "get_doc", `{"id":"x"}`, true, "", "")
	if _, ok := tc2.Meta["witness"]; ok {
		t.Errorf("empty witness should not set a witness meta key")
	}
	if tc2.TraceID == "" {
		t.Errorf("an omitted TraceID must be minted non-empty on the served path")
	}
}

// A refutation through the gateway evicts the pooled entry, advances the integrity
// epoch, and lands on the cross-agent change feed.
func TestRevoke_EvictsAndPublishesToFeed(t *testing.T) {
	srv := newTestServer(t)
	const wit = "commit-z42-revoke"
	fillWitnessed("get_doc_z42", wit, `{"body":"hi"}`)

	evicted, te := srv.revoke(wit)
	if evicted < 1 {
		t.Errorf("revoke evicted=%d, want >=1 (the witnessed entry)", evicted)
	}
	if te == 0 {
		t.Errorf("trust epoch did not advance on a refutation")
	}

	events, cursor := srv.changes("", 0)
	rv := findRevocation(events, wit)
	if rv == nil {
		t.Fatalf("revocation for %q not on the change feed", wit)
	}
	if rv.Evicted < 1 || rv.TrustEpoch == 0 {
		t.Errorf("feed revocation under-reports: evicted=%d trustEpoch=%d", rv.Evicted, rv.TrustEpoch)
	}
	if cursor < rv.Seq {
		t.Errorf("drain cursor %d < event seq %d", cursor, rv.Seq)
	}

	// Draining PAST the event's cursor returns nothing new for it.
	after, _ := srv.changes("", cursor)
	if findRevocation(after, wit) != nil {
		t.Errorf("event for %q re-delivered after its cursor", wit)
	}
}

// A write-shaped completion lands on the feed as a typed mutation (the "what changed"
// signal), naming the tool and its invalidation scope.
func TestChanges_CapturesWriteMutation(t *testing.T) {
	srv := newTestServer(t)
	const tool = "write_thing_z77" // "write" => destructive => a world bump + bus publish
	wc := &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"k":"v"}`)}}
	vdso.Default.Emit(abi.Event{Kind: abi.EvComplete, Call: wc,
		Result: &abi.Result{Call: wc, Status: abi.StatusOK, Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"ok":true}`)}}})

	events, _ := srv.changes("", 0)
	m := findMutation(events, tool)
	if m == nil {
		t.Fatalf("write mutation for %q not on the feed", tool)
	}
	if len(m.Tags) == 0 {
		t.Errorf("mutation carries no invalidation tags")
	}
}

// The MCP tools/call surface dispatches fak_revoke and fak_changes (the live wire path).
func TestMCP_RevokeAndChanges(t *testing.T) {
	srv := newTestServer(t)
	const wit = "commit-mcp-rev"
	fillWitnessed("get_doc_mcp", wit, `{"body":"x"}`)

	// fak_revoke
	revParams, _ := json.Marshal(map[string]any{"name": "fak_revoke", "arguments": map[string]any{"witness": wit}})
	res, rerr := srv.callTool(context.Background(), revParams)
	if rerr != nil {
		t.Fatalf("fak_revoke rpc error: %v", rerr.Message)
	}
	var rv RevokeResponse
	decodeMCPText(t, res, &rv)
	if rv.Witness != wit || rv.Evicted < 1 {
		t.Errorf("fak_revoke result: %+v (want witness=%s, evicted>=1)", rv, wit)
	}

	// fak_changes since 0 must include the revocation just produced.
	chParams, _ := json.Marshal(map[string]any{"name": "fak_changes", "arguments": map[string]any{"since": 0}})
	res2, rerr2 := srv.callTool(context.Background(), chParams)
	if rerr2 != nil {
		t.Fatalf("fak_changes rpc error: %v", rerr2.Message)
	}
	var ch ChangesResponse
	decodeMCPText(t, res2, &ch)
	if findRevocation(ch.Events, wit) == nil {
		t.Errorf("fak_changes did not surface the revocation for %q", wit)
	}

	// fak_revoke with an empty witness is an invalid-params error, not a silent no-op.
	bad, _ := json.Marshal(map[string]any{"name": "fak_revoke", "arguments": map[string]any{"witness": ""}})
	if _, e := srv.callTool(context.Background(), bad); e == nil {
		t.Errorf("fak_revoke with empty witness should be an rpc error")
	}
}

// decodeMCPText unwraps an MCP tool result's single text content block into v.
func decodeMCPText(t *testing.T, res any, v any) {
	t.Helper()
	m, ok := res.(map[string]any)
	if !ok {
		t.Fatalf("mcp result is not an object: %T", res)
	}
	content, ok := m["content"].([]map[string]any)
	if !ok || len(content) == 0 {
		t.Fatalf("mcp result has no content block: %+v", m)
	}
	text, _ := content[0]["text"].(string)
	if err := json.Unmarshal([]byte(text), v); err != nil {
		t.Fatalf("decode mcp text %q: %v", text, err)
	}
}
