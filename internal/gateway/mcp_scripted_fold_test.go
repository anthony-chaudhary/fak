package gateway

// mcp_scripted_fold_test.go — the #2858 contract: a Hermes-style subagent SCRIPT folds a
// child tool result back into the parent's Python WITHOUT re-entering the model context,
// so that fold never crosses the model-facing admit boundary (subagent_witness_test.go
// exercises THAT site via admitInboundResults). This file proves the SAME subagent-fold
// witness fires on the zero-context-cost RPC path — driven end-to-end through the real
// dispatchRPC wire, the exact bytes an external script writes — so "zero-context-cost"
// does not become "zero-adjudication of the fold":
//
//   - Unwitnessed done-claim (taint): an ALLOW subagent-spawn call whose result CLAIMS
//     done but carries no artifact witness is demoted RESIDUAL/LOOP_DONE_UNWITNESSED on
//     the wire; the content still forwards (parent sees it) and the envelope Meta carries
//     a `subagent_fold` breadcrumb — the witnessed result envelope the issue asks for.
//   - Witnessed done-claim (clean): the SAME claim carrying a commit SHA folds clean — a
//     plain ALLOW, no demotion, no breadcrumb.
//   - Non-spawn call: an ordinary scripted call that happens to claim "completed" is
//     untouched — the fold fires only at the subagent-spawn boundary.
//
// gen/next bookkeeping (#2858, sibling of #2889's scripted-pipeline slice):
//   - Promotion evidence (toward now): this contract green PLUS a real subagent script
//     against a live gateway showing the demotion row reaches an operator readout and the
//     parent script branches on it (not just the wire verdict asserted here).
//   - Demotion/retirement evidence: a native MCP verb that witnesses subagent returns
//     server-side before they reach the script, obsoleting the per-call wire fold; or
//     provider-observed accounting showing scripts never fold unverified child returns in
//     practice (the taint never occurs), making the gate dead weight.
//   - Invalidating assumption: the witness is STRUCTURAL — it gates on an artifact-witness
//     TOKEN (commit SHA / file hash) in the child's prose, not on deep git reachability. A
//     child that prints a plausible but non-ancestor SHA folds clean here; closing that is
//     the costlier dos_verify registry/grep rung this fold references, not runs inline.

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

// scriptedSyscall plays one fak_syscall frame through the real dispatch path — the exact
// bytes a subagent script writes on the stdio wire — and returns the decoded kernel
// response the script would fold into the parent.
func scriptedSyscall(t *testing.T, srv *Server, tool, argsJSON, trace string) SyscallResponse {
	t.Helper()
	frame := fmt.Sprintf(`{"jsonrpc":"2.0","id":1,"method":"tools/call",`+
		`"params":{"name":"fak_syscall","arguments":{"tool":%q,"arguments":%s,"trace_id":%q}}}`,
		tool, argsJSON, trace)
	resp := srv.dispatchRPC(context.Background(), []byte(frame))
	if resp == nil || resp.Error != nil {
		t.Fatalf("fak_syscall round-trip failed: %+v", resp)
	}
	wire, err := json.Marshal(resp.Result)
	if err != nil {
		t.Fatalf("marshal result: %v", err)
	}
	var wrap struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
		IsError bool `json:"isError"`
	}
	if err := json.Unmarshal(wire, &wrap); err != nil || len(wrap.Content) != 1 || wrap.IsError {
		t.Fatalf("tool-result wrap = %s (err=%v)", wire, err)
	}
	var sr SyscallResponse
	if err := json.Unmarshal([]byte(wrap.Content[0].Text), &sr); err != nil {
		t.Fatalf("decode SyscallResponse: %v", err)
	}
	if sr.Result == nil {
		t.Fatalf("no executed result (verdict %+v) — the scripted call must run, not just adjudicate", sr.Verdict)
	}
	return sr
}

// TestScriptedFold_UnwitnessedDoneDemoted is the headline: a subagent-spawn call whose
// result claims done but carries no artifact witness is demoted on the wire — the parent
// script gets the content, but the verdict records it visibly unverified.
func TestScriptedFold_UnwitnessedDoneDemoted(t *testing.T) {
	mcpToolprocTestJournal(t)
	srv := newTestServer(t)

	// allow_* → ALLOW verdict (test adjudicator); the "spawn" shape → subagent-fold
	// boundary. The echoed args are the child's return prose: a bare "completed" claim.
	sr := scriptedSyscall(t, srv, "allow_spawn_child",
		`{"summary":"the child worker is completed already"}`, "t2858-unwit")

	if sr.Verdict.Kind != "RESIDUAL" || sr.Verdict.Reason != ReasonLoopBodyUnwitnessed {
		t.Fatalf("unwitnessed scripted fold: verdict = %+v, want RESIDUAL/%s", sr.Verdict, ReasonLoopBodyUnwitnessed)
	}
	if sr.Verdict.By != subagentFoldBy {
		t.Errorf("demotion By = %q, want %q", sr.Verdict.By, subagentFoldBy)
	}
	if sr.Verdict.Detail["claim"] != "done" {
		t.Errorf("demotion must record the unbacked claim kind, Detail = %v, want claim=done", sr.Verdict.Detail)
	}
	// The child's return string still reaches the parent — only visibly unverified.
	if !strings.Contains(sr.Result.Content, "completed") {
		t.Errorf("tainted content must still reach the parent script, got %q", sr.Result.Content)
	}
	// The witnessed result envelope carries the breadcrumb (#2858 scope).
	if sr.Result.Meta["subagent_fold"] != ReasonLoopBodyUnwitnessed {
		t.Errorf("envelope Meta[subagent_fold] = %q, want %q", sr.Result.Meta["subagent_fold"], ReasonLoopBodyUnwitnessed)
	}
}

// TestScriptedFold_WitnessedDoneClean proves the safe path: the SAME claim carrying a
// commit SHA folds clean — a plain ALLOW, no demotion, no breadcrumb.
func TestScriptedFold_WitnessedDoneClean(t *testing.T) {
	mcpToolprocTestJournal(t)
	srv := newTestServer(t)

	sr := scriptedSyscall(t, srv, "allow_spawn_child",
		`{"summary":"child committed 0123456789abcdef0123456789abcdef01234567"}`, "t2858-wit")

	if sr.Verdict.Kind != "ALLOW" {
		t.Fatalf("witnessed scripted fold must stay clean: verdict = %+v, want ALLOW", sr.Verdict)
	}
	if _, stamped := sr.Result.Meta["subagent_fold"]; stamped {
		t.Errorf("a clean fold must not stamp a demotion breadcrumb, Meta = %v", sr.Result.Meta)
	}
}

// TestScriptedFold_NonSpawnUntouched proves the fold fires only at the subagent-spawn
// boundary: an ordinary scripted call that happens to claim "completed" is admitted as-is.
func TestScriptedFold_NonSpawnUntouched(t *testing.T) {
	mcpToolprocTestJournal(t)
	srv := newTestServer(t)

	sr := scriptedSyscall(t, srv, "allow_plainstep",
		`{"summary":"the config entry is completed"}`, "t2858-plain")

	if sr.Verdict.Kind == "RESIDUAL" {
		t.Fatalf("a non-spawn scripted call must not be subagent-demoted, got %+v", sr.Verdict)
	}
	if _, stamped := sr.Result.Meta["subagent_fold"]; stamped {
		t.Errorf("a non-spawn call must not stamp a fold breadcrumb, Meta = %v", sr.Result.Meta)
	}
}

// TestWitnessScriptedFold_NilEnvSafe is the unit guard: an errored/DENY call that executed
// nothing (nil envelope) has no return string to fold — the base verdict passes through.
func TestWitnessScriptedFold_NilEnvSafe(t *testing.T) {
	base := WireVerdict{Kind: "DENY", Reason: "POLICY_BLOCK", By: "test"}
	if got := witnessScriptedFold("spawn_child", nil, base); !reflect.DeepEqual(got, base) {
		t.Fatalf("nil envelope must return base unchanged, got %+v", got)
	}
}
