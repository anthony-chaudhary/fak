package gateway

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// adjudicate_proposed_restore_wire_gate_test.go — the wire-affordance gate for
// served-inline context restore (ticket private#2498).
//
// The defect: the gateway's served-inline restore path emitted
// `[fak: restored context id=<handle>]` unconditionally on restoreContext
// success, with no check that the ACTIVE CLIENT WIRE advertises a restore
// affordance. On the pi `openai` wire (explicit tools list, no restore tool
// def, shrink levers inert) a model that mis-spelled the affordance as a shell
// command (`go run ./cmd/fak-sync context restore <64-hex>`) received a
// guard-injected success banner for a handle the model INVENTED, and looped
// ~40 turns.
//
// These tests are written adversarially from the seam contract only. They call
// `adjudicateProposedServed` and assert observable outcomes (servedText, kept,
// adjs, the CompactionRestoredTurns counter) — never implementation internals.
//
// The wire-affordance carrier is `withClientToolSchemas`:
//
//	wired=false (no ctx value)  -> historical fail-open serve stands
//	wired=true, advertised=true -> genuine serve with the banner
//	wired=true, advertised=false-> NO banner; a model-actionable correction
//	                               naming the real affordance (a tool call,
//	                               not a shell command)

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

// wireGateRestoreCall builds a restore-shaped tool call for the given handle.
func wireGateRestoreCall(callID, handle string) agent.ToolCall {
	args, err := json.Marshal(ContextRestoreRequest{ID: handle})
	if err != nil {
		panic(err) // marshal of a plain struct cannot fail
	}
	return agent.ToolCall{
		ID: callID, Type: "function",
		Function: agent.Func{Name: "fak_context_restore", Arguments: string(args)},
	}
}

// wireGateToolDefs builds an explicit client tools list. When withRestore is
// false the list is the pi-like `openai` wire: read/bash/edit, NO restore def.
func wireGateToolDefs(withRestore bool) []agent.ToolDef {
	defs := []agent.ToolDef{
		{Type: "function", Function: agent.ToolDefFunction{Name: "read", Description: "read a file"}},
		{Type: "function", Function: agent.ToolDefFunction{Name: "bash", Description: "run a shell command"}},
		{Type: "function", Function: agent.ToolDefFunction{Name: "edit", Description: "edit a file"}},
	}
	if withRestore {
		defs = append(defs, agent.ToolDef{
			Type: "function",
			Function: agent.ToolDefFunction{
				Name:        "mcp__fak_guard__fak_context_restore",
				Description: "restore a tombstoned context span by id",
			},
		})
	}
	return defs
}

// wireGateTombstone seeds the per-session restore store so a restore for id
// genuinely succeeds, and returns the 64-hex handle.
func wireGateTombstone(t *testing.T, srv *Server, trace, seed string) string {
	t.Helper()
	taskBytes := []byte(seed)
	id := ctxplan.Digest(taskBytes)
	srv.stashRestore(trace, id, seed, taskBytes)
	return id
}

// ---------------------------------------------------------------------------
// case 1 — inert wire declines the banner
// ---------------------------------------------------------------------------

// TestRestoreWireGateInertWireDeclinesBanner: ctx carries an explicit tools
// list (read/bash/edit, NO restore tool) via withClientToolSchemas; a
// restore-shaped `fak_context_restore` call is adjudicated. The wire does not
// advertise the affordance, so the guard must NOT serve a success banner. The
// model must instead receive a correction naming the real affordance (a tool
// call, not a shell command), the compaction-restore counter must not move,
// and the call must not be forwarded to the client as a live tool call.
func TestRestoreWireGateInertWireDeclinesBanner(t *testing.T) {
	srv := newTestServer(t)
	const trace = "wire-gate-inert-decline"
	// The handle the model INVENTED (never tombstoned in this session).
	const invented = "d43f9a170e2f039d7170365c7b964c0af76033d6a43440f9d3fd6f15f7b15893"

	before := srv.AdjudicationSummary().CompactionRestoredTurns

	ctx := withClientToolSchemas(context.Background(), wireGateToolDefs(false))
	calls := []agent.ToolCall{wireGateRestoreCall("c-inert", invented)}

	kept, adjs, _, served, hits := srv.adjudicateProposedServed(ctx, calls, trace)

	if hits != 0 {
		t.Fatalf("served hits = %d, want 0 on a wire with no restore affordance", hits)
	}
	if strings.Contains(served, "[fak: restored context") {
		t.Fatalf("served text fabricated a restore banner on a non-restore wire: %q", served)
	}
	// The correction must be model-actionable: it names the tool-call vs
	// shell-command distinction.
	if !strings.Contains(served, "tool call") || !strings.Contains(served, "not a shell command") {
		t.Fatalf("served text %q lacks a tool-call-not-shell-command correction", served)
	}
	// A declined restore must never read as a success for the invented id.
	if strings.Contains(served, invented) {
		t.Fatalf("declined restore leaked the invented handle %s into served text: %q", invented, served)
	}
	if got := srv.AdjudicationSummary().CompactionRestoredTurns; got != before {
		t.Fatalf("CompactionRestoredTurns moved %d -> %d on a declined restore; want unchanged", before, got)
	}
	if len(kept) != 0 {
		t.Fatalf("kept = %d calls, want 0 (the guard served the correction inline instead of forwarding)", len(kept))
	}
	if len(adjs) != 1 {
		t.Fatalf("adjudications = %d, want 1", len(adjs))
	}
	if adjs[0].Admitted {
		t.Fatalf("verdict %+v: a declined restore must not be admitted", adjs[0])
	}
}

// ---------------------------------------------------------------------------
// case 2 — advertised wire still serves
// ---------------------------------------------------------------------------

// TestRestoreWireGateAdvertisedWireStillServes: ctx tools list INCLUDES a
// restore tool def; a restore call with a genuinely tombstoned id is
// adjudicated. The working path is preserved: the banner names the requested
// handle.
func TestRestoreWireGateAdvertisedWireStillServes(t *testing.T) {
	srv := newTestServer(t)
	const trace = "wire-gate-advertised-serve"
	id := wireGateTombstone(t, srv, trace, "the dropped originating task (advertised wire)")

	before := srv.AdjudicationSummary().CompactionRestoredTurns

	ctx := withClientToolSchemas(context.Background(), wireGateToolDefs(true))
	calls := []agent.ToolCall{wireGateRestoreCall("c-adv", id)}

	kept, adjs, _, served, hits := srv.adjudicateProposedServed(ctx, calls, trace)

	if hits != 1 {
		t.Fatalf("served hits = %d, want 1 on a wire that advertises the restore affordance", hits)
	}
	if !strings.Contains(served, "[fak: restored context id="+id+"]") {
		t.Fatalf("served text %q must carry the banner naming the requested handle %s", served, id)
	}
	if len(kept) != 0 {
		t.Fatalf("kept = %d calls, want 0 (a served hit is not forwarded)", len(kept))
	}
	if got := srv.AdjudicationSummary().CompactionRestoredTurns; got != before+1 {
		t.Fatalf("CompactionRestoredTurns = %d, want %d (one genuine restore)", got, before+1)
	}
	if len(adjs) != 1 || adjs[0].Verdict.Reason != "SERVED_INLINE" {
		t.Fatalf("adjudication = %+v, want one SERVED_INLINE", adjs)
	}
}

// ---------------------------------------------------------------------------
// case 3 — fail-open on absent schemas
// ---------------------------------------------------------------------------

// TestRestoreWireGateFailOpenOnAbsentSchemas: ctx carries NO
// withClientToolSchemas value at all (plain context.Background()-derived ctx,
// same setup as the committed reproducer tests). A restore-shaped call with a
// genuinely tombstoned id still serves the banner — the historical fail-open
// contract stands.
func TestRestoreWireGateFailOpenOnAbsentSchemas(t *testing.T) {
	srv := newTestServer(t)
	const trace = "wire-gate-fail-open"
	id := wireGateTombstone(t, srv, trace, "the dropped originating task (no schemas ctx)")

	before := srv.AdjudicationSummary().CompactionRestoredTurns

	// Deliberately NO withClientToolSchemas: the reproducer-test ctx shape.
	calls := []agent.ToolCall{wireGateRestoreCall("c-failopen", id)}

	kept, _, _, served, hits := srv.adjudicateProposedServed(context.Background(), calls, trace)

	if hits != 1 {
		t.Fatalf("served hits = %d, want 1 (fail-open: no schemas ctx keeps the historical serve)", hits)
	}
	if !strings.Contains(served, "[fak: restored context id="+id+"]") {
		t.Fatalf("served text %q must carry the banner naming the requested handle %s", served, id)
	}
	if len(kept) != 0 {
		t.Fatalf("kept = %d calls, want 0 (a served hit is not forwarded)", len(kept))
	}
	if got := srv.AdjudicationSummary().CompactionRestoredTurns; got != before+1 {
		t.Fatalf("CompactionRestoredTurns = %d, want %d (one genuine restore)", got, before+1)
	}
}

// ---------------------------------------------------------------------------
// case 4 — shell-shaped call never banners (regression, openai wire)
// ---------------------------------------------------------------------------

// TestRestoreWireGateShellShapedCallNeverBanners: on a ctx whose tools list is
// the pi-like openai wire (read/bash/edit, no restore def), a `bash` tool call
// whose ARGUMENTS embed a 64-hex handle is adjudicated. The bash call is a
// real client-side tool: whether the client's tool policy admits or denies it
// is orthogonal policy behavior the seam does not own; the guard must never
// serve it inline as a restore, and no restore banner may appear anywhere in
// served text.
func TestRestoreWireGateShellShapedCallNeverBanners(t *testing.T) {
	srv := newTestServer(t)
	const trace = "wire-gate-shell-shape"
	// Tombstone a span so a restore WOULD succeed if the guard wrongly served it.
	id := wireGateTombstone(t, srv, trace, "a stashed span the model never legitimately asked for")

	before := srv.AdjudicationSummary().CompactionRestoredTurns

	ctx := withClientToolSchemas(context.Background(), wireGateToolDefs(false))
	calls := []agent.ToolCall{{
		ID: "c-bash", Type: "function",
		Function: agent.Func{
			Name:      "bash",
			Arguments: `{"command":"go run ./cmd/fak-sync context restore ` + id + ` 2>&1 | head -120"}`,
		},
	}}

	_, adjs, _, served, hits := srv.adjudicateProposedServed(ctx, calls, trace)

	if hits != 0 {
		t.Fatalf("served hits = %d, want 0: a bash call is not a guard-served restore", hits)
	}
	if strings.Contains(served, "[fak: restored context") {
		t.Fatalf("guard fabricated a restore banner for a shell call on the openai wire: %q", served)
	}
	// Policy-neutral: whatever the client tool policy decides for this bash
	// call (admit, deny, defer), the guard must never serve it inline as a
	// restore; SERVED_INLINE is the restore-serve verdict reason.
	for _, a := range adjs {
		if a.Verdict.Reason == "SERVED_INLINE" {
			t.Fatalf("adjudication %+v cites SERVED_INLINE: a bash call must never be served inline as a restore", a)
		}
	}
	if got := srv.AdjudicationSummary().CompactionRestoredTurns; got != before {
		t.Fatalf("CompactionRestoredTurns moved %d -> %d for a shell-shaped call; want unchanged", before, got)
	}
}

// ---------------------------------------------------------------------------
// case 5 — correction never names the invented handle
// ---------------------------------------------------------------------------

// TestRestoreWireGateCorrectionNeverNamesInventedHandle: in the case-1 shape,
// the servedText must not contain the 64-hex handle the model invented — a
// declined restore must never read as a success for that id. This is the
// sharpened form of the observed fabrication: the banner named a handle that
// appeared ONLY in the model's own argument.
func TestRestoreWireGateCorrectionNeverNamesInventedHandle(t *testing.T) {
	srv := newTestServer(t)
	const trace = "wire-gate-no-invented-handle"
	// The exact observed literal, from pi turn 14.
	const invented = "d43f9a170e2f039d7170365c7b964c0af76033d6a43440f9d3fd6f15f7b15893"

	ctx := withClientToolSchemas(context.Background(), wireGateToolDefs(false))
	calls := []agent.ToolCall{wireGateRestoreCall("c-invent", invented)}

	_, _, _, served, hits := srv.adjudicateProposedServed(ctx, calls, trace)

	if hits != 0 {
		t.Fatalf("served hits = %d, want 0 for an invented handle on a non-restore wire", hits)
	}
	if strings.Contains(served, invented) {
		t.Fatalf("served text %q names the invented handle %s — a declined restore must never read as a success for that id", served, invented)
	}
	if strings.Contains(served, "[fak: restored context") {
		t.Fatalf("served text %q carries a restore banner for an invented handle", served)
	}
}
