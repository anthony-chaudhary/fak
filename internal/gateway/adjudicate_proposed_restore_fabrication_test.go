package gateway

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// adjudicate_proposed_restore_fabrication_test.go — the pi/Qwen3.8-27B
// "fak-sync context restore" misuse (session 01a0c24f, 2026-09-21).
//
// OBSERVED (pi session log, --C--work-fak-private--):
//
//	turn 12: bash  `go run ./cmd/fak-sync context restore ce00e1a5... | head -100`
//	turn 14: bash  `go run ./cmd/fak-sync context restore d43f9a17... | head -200`
//	turn 20: assistant TEXT (verbatim, 5430 bytes, stopReason "stop"):
//	         "\n[fak: restored context id=d43f9a170e2f...]\nunknown verb \"context\"\n..."
//
// The 64-hex handle `d43f9a17...` appears for the FIRST time in the model's own
// turn-14 shell argument and in NO prior tool result. The guard nevertheless
// emitted a `[fak: restored context id=<that exact handle>]` banner into the
// model's assistant prose — a fabricated success for a call that failed
// (`unknown verb "context"`, exit status 2). The model then repeated the failing
// shell call ~40 times across two sessions.
//
// MECHANISM (internal/gateway/adjudicate_proposed.go):
//
//	adjudicateProposedServed -> isRestoreTool(tool) -> restoreContext() success
//	-> served = append(served, "[fak: restored context id=%s]\n%s")
//	-> applyAdjudicatedTurn (http.go) folds servedText into asst.Content
//
// `isRestoreTool` matches ONLY exact restore tool identities
// (fak_context_restore / mcp__fak(_guard)__fak_context_restore / functions.*).
// A `bash` call is never a restore tool, so the banner cannot originate from the
// observed bash call. The banner is emitted only when the model proposes a
// restore-shaped call — which means the model DID emit one, and the guard served
// it inline and told the model "restored".
//
// THE DEFECT the tests below pin is narrower and real: a restore call that
// SUCCEEDS is served inline as prose with a `[fak: restored context` banner, and
// the tool call is dropped from `kept`. When the model's transport/loop cannot
// act on a served inline result (pi's OpenAI-completions wire has no MCP tool to
// execute it), the banner lands in assistant prose as a fabricated success and
// the model's own (failing) shell spelling stays uncorrected. The guard never
// tells the model "that spelling is wrong; the affordance is a tool call, not a
// shell command".

// restoreBannerPresent reports whether served text carries the restore banner.
func restoreBannerPresent(s string) bool {
	return strings.Contains(s, "[fak: restored context")
}

// TestServedRestoreBannerOnRestoreToolCall pins the DESIRED case: a genuine
// fak_context_restore MCP call for a stashed handle is served inline, and the
// banner names the handle the model legitimately asked for.
func TestServedRestoreBannerOnRestoreToolCall(t *testing.T) {
	srv := newTestServer(t)
	const trace = "pi-restore-wellformed"
	id := ctxplan.Digest([]byte("the dropped originating task"))
	srv.stashRestore(trace, id, "originating task", []byte("the dropped originating task"))

	args, err := json.Marshal(ContextRestoreRequest{ID: id})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	calls := []agent.ToolCall{{
		ID: "c-restore", Type: "function",
		Function: agent.Func{Name: "fak_context_restore", Arguments: string(args)},
	}}

	kept, adjs, _, served, hits := srv.adjudicateProposedServed(context.Background(), calls, trace)

	if hits != 1 {
		t.Fatalf("served hits = %d, want 1 (the restoring call is a served hit)", hits)
	}
	if len(kept) != 0 {
		t.Fatalf("kept = %d calls, want 0 (a served hit is not forwarded)", len(kept))
	}
	if !restoreBannerPresent(served) {
		t.Fatalf("served text %q lacks the restore banner the model depends on", served)
	}
	if !strings.Contains(served, id) {
		t.Fatalf("banner %q must name the requested handle %s", served, id)
	}
	if len(adjs) != 1 || adjs[0].Verdict.Reason != "SERVED_INLINE" {
		t.Fatalf("adjudication = %+v, want one SERVED_INLINE", adjs)
	}
}

// TestServedRestoreBannerAbsentForShellShapedCall pins the other half of the
// boundary: a `bash` call that merely embeds the handle in a shell spelling is
// NOT a restore tool call. It must never be served inline and must never
// produce a restore banner — that is the fabricated-success path observed in
// the pi session.
func TestServedRestoreBannerAbsentForShellShapedCall(t *testing.T) {
	srv := newTestServer(t)
	const trace = "pi-restore-shellshape"
	id := ctxplan.Digest([]byte("a stashed span the model never legitimately asked for"))
	srv.stashRestore(trace, id, "stashed span", []byte("a stashed span the model never legitimately asked for"))

	// The exact observed shape: a bash command embedding the handle.
	calls := []agent.ToolCall{{
		ID: "c-bash", Type: "function",
		Function: agent.Func{
			Name:      "bash",
			Arguments: `{"command":"cd C:/work/fak-private && go run ./cmd/fak-sync context restore ` + id + ` 2>&1 | head -120"}`,
		},
	}}

	_, _, _, served, hits := srv.adjudicateProposedServed(context.Background(), calls, trace)

	if hits != 0 {
		t.Fatalf("served hits = %d, want 0: a bash call is not a guard-served restore", hits)
	}
	if restoreBannerPresent(served) {
		t.Fatalf("guard fabricated a restore banner for a shell call: %q", served)
	}
}

// TestServedRestoreBannerNeverNamesAnUnrequestedHandle is the sharpened
// regression for the observed fabrication.
//
// The observed banner named `d43f9a17...`, a handle that appeared ONLY in the
// model's own shell argument. Whatever path emits a restore banner, the banner's
// id must equal an id the model actually proposed in a restore-shaped call — a
// banner naming a handle that came from nowhere is a pure fabrication.
//
// This test asserts the invariant across the non-restore (bash) shape: no banner
// may be produced, so no unrequested handle can ever be named.
func TestServedRestoreBannerNeverNamesAnUnrequestedHandle(t *testing.T) {
	srv := newTestServer(t)
	const trace = "pi-restore-fabrication"
	// The exact observed literal, from pi turn 14.
	const observedHandle = "d43f9a170e2f039d7170365c7b964c0af76033d6a43440f9d3fd6f15f7b15893"
	srv.stashRestore(trace, observedHandle, "the span the model saw advertised", []byte("stashed span body"))

	calls := []agent.ToolCall{{
		ID: "c-fabricate", Type: "function",
		Function: agent.Func{
			Name:      "bash",
			Arguments: `{"command":"go run ./cmd/fak-sync context restore ` + observedHandle + `"}`,
		},
	}}

	_, _, _, served, hits := srv.adjudicateProposedServed(context.Background(), calls, trace)

	if hits != 0 || restoreBannerPresent(served) {
		t.Fatalf("a non-restore call must not yield a restore banner (hits=%d served=%q)", hits, served)
	}
}
