package microagent_test

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

// scriptExec builds a floor-adjudicated ToolExec over the trusted in-process
// goroutine backend (no subprocess — a hermetic witness): the injected policy
// affirmatively allows exactly `allow`, and every allowed tool dispatches to an
// in-process func returning `stdout`. A tool NOT in `allow` fails closed at the
// floor (default-deny), so the seam gates it BEFORE dispatch.
func scriptExec(t *testing.T, allow []string, stdout string) *microagent.ToolExec {
	t.Helper()
	allowSet := map[string]bool{}
	for _, tl := range allow {
		allowSet[tl] = true
	}
	k := kernel.New("", kernel.WithAdjudicators([]abi.Adjudicator{
		adjudicator.New(adjudicator.Policy{Allow: allowSet}),
	}))
	be := microagent.NewGoroutineBackend()
	for _, tl := range allow {
		if err := be.Register(tl, func(ctx context.Context, act microagent.ToolAction) (microagent.ToolResult, error) {
			return microagent.ToolResult{Stdout: []byte(stdout)}, nil
		}); err != nil {
			t.Fatalf("GoroutineBackend.Register(%s): %v", tl, err)
		}
	}
	te, err := microagent.NewToolExecBackend(k, be)
	if err != nil {
		t.Fatalf("NewToolExecBackend: %v", err)
	}
	return te
}

// openJournal opens a fresh file-backed decision journal under the test's temp dir.
// It uses a LOCAL journal fed via Emit (the RPCSubagent path), NOT the process-
// global abi.RegisterEmitter — abi.ResetForTest would wipe the global registry out
// from under this shared test binary's other tests (journalsink_quarantine_test.go
// documents the same reasoning).
func openJournal(t *testing.T) (*journal.Journal, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "rpc-subagent.jsonl")
	j, err := journal.Open(path)
	if err != nil {
		t.Fatalf("journal.Open: %v", err)
	}
	t.Cleanup(func() { _ = j.Close() })
	return j, path
}

// TestRPCSubagentCollapsesUnderFloor is the #2931 composite witness: on ONE run of
// a 3-call RPC-subagent script, BOTH properties the issue's acceptance names hold
// together —
//
//	(a) ADJUDICATED + JOURNALED: every intermediate call carries a VerdictAllow and
//	    lands a durable DECIDE row tagged with the subagent id; and
//	(b) NOT BILLED: the orchestrator's context does not grow by the intermediate
//	    tool chatter (a before/after readout), while the subagent's own context
//	    does carry it — and folding the bounded collapsed result costs the
//	    orchestrator far less than the chatter would inline (the collapse is a
//	    MEASURED saving, not a tautology).
func TestRPCSubagentCollapsesUnderFloor(t *testing.T) {
	// A chunky per-call payload so the intermediate chatter is a real, measurable
	// context cost — not a rounding artifact.
	payload := strings.Repeat("intermediate tool output token. ", 64) // ~2 KiB per call
	tools := []string{"fetch", "parse", "summarize"}
	te := scriptExec(t, tools, payload)
	j, path := openJournal(t)

	const subID = "rpc-sub-001"
	sub, err := microagent.NewRPCSubagent(subID, te, 16384, j)
	if err != nil {
		t.Fatalf("NewRPCSubagent: %v", err)
	}

	// The orchestrator's OWN context — a single goal turn, nothing else.
	orch := microagent.NewContext(8192)
	orch.Append("user", "goal: fetch, parse, and summarize the release notes")
	before := orch.Tokens()

	// Run the 3-call pipeline as an RPC subagent.
	script := []microagent.ToolAction{
		{Tool: "fetch", Args: map[string]any{"url": "https://example.test/notes"}},
		{Tool: "parse", Args: map[string]any{"format": "markdown"}},
		{Tool: "summarize", Args: map[string]any{"max_words": 40}},
	}
	res := sub.RunScript(context.Background(), script)
	if err := j.Flush(); err != nil {
		t.Fatalf("journal.Flush: %v", err)
	}

	// ---- Property (b): ZERO ORCHESTRATOR-CONTEXT COST ----
	// Running the whole pipeline appended NOTHING to the orchestrator's context.
	if got := orch.Tokens(); got != before {
		t.Fatalf("orchestrator context grew running the pipeline: before=%d after=%d (intermediate chatter was billed to the orchestrator)", before, got)
	}
	// The work is REAL and CONTAINED, not free: the chatter lives in the subagent's
	// own context.
	intermediate := sub.Context().Tokens()
	if intermediate <= 0 {
		t.Fatal("subagent context is empty — the pipeline did no work")
	}
	// The collapse is a measured saving: folding the bounded collapsed result costs
	// the orchestrator far less than the intermediate chatter.
	orch.Append("tool", res.Collapsed)
	folded := orch.Tokens() - before
	if res.IntermediateTokens != intermediate || res.FoldedTokens <= 0 || res.SavedTokens != res.IntermediateTokens-res.FoldedTokens {
		t.Fatalf("receipt accounting=%+v measured intermediate=%d folded=%d", res, intermediate, folded)
	}
	if folded <= 0 {
		t.Fatal("collapsed result folded to 0 tokens — nothing was returned to the orchestrator")
	}
	if folded >= intermediate {
		t.Fatalf("collapse saved no context: folded=%d intermediate=%d (want folded << intermediate)", folded, intermediate)
	}

	// Contrast: had the orchestrator run the SAME pipeline inline (folding each
	// call+result into its own context), it WOULD have paid the intermediate tokens.
	inline := microagent.NewContext(8192)
	inline.Append("user", "goal: fetch, parse, and summarize the release notes")
	inlineBefore := inline.Tokens()
	for _, act := range script {
		inline.Append("assistant", "call "+act.Tool)
		inline.Append("tool", payload)
	}
	inlineDelta := inline.Tokens() - inlineBefore
	if inlineDelta <= folded {
		t.Fatalf("inline delta=%d not greater than collapsed folded=%d — the collapse banked no saving", inlineDelta, folded)
	}

	// ---- Property (a): ADJUDICATED + JOURNALED (same run) ----
	if res.Allowed != len(script) || res.Denied != 0 || res.Errored != 0 {
		t.Fatalf("script outcome allowed=%d denied=%d errored=%d, want %d/0/0", res.Allowed, res.Denied, res.Errored, len(script))
	}
	for i, st := range res.Steps {
		if !st.Allowed() {
			t.Fatalf("step %d (%s) not allowed: verdict=%v err=%v", i, st.Tool, st.Verdict.Kind, st.Err)
		}
		if !st.Ran {
			t.Fatalf("step %d (%s) did not run despite an allow", i, st.Tool)
		}
	}
	// The durable chain verifies end-to-end and holds exactly one DECIDE row per
	// intermediate call, each an ALLOW tagged with the subagent id and carrying the
	// adjudicated args digest.
	n, err := journal.Verify(path)
	if err != nil {
		t.Fatalf("journal.Verify: %v", err)
	}
	if n != len(script) {
		t.Fatalf("journal has %d rows, want %d (exactly one adjudication per intermediate call)", n, len(script))
	}
	rows, err := journal.ReadRows(path)
	if err != nil {
		t.Fatalf("journal.ReadRows: %v", err)
	}
	seenTool := map[string]bool{}
	for _, r := range rows {
		if r.Kind != "DECIDE" {
			t.Fatalf("row kind = %q, want DECIDE (each intermediate call is an adjudicated decision)", r.Kind)
		}
		if r.Verdict != "ALLOW" {
			t.Fatalf("row for %q verdict = %q, want ALLOW", r.Tool, r.Verdict)
		}
		if r.TraceID != subID {
			t.Fatalf("row TraceID = %q, want the subagent id %q (journal not tagged with the agent)", r.TraceID, subID)
		}
		if r.ArgsDigest == "" {
			t.Fatalf("row for %q has no ArgsDigest — the adjudicated args were not recorded", r.Tool)
		}
		seenTool[r.Tool] = true
	}
	for _, tl := range tools {
		if !seenTool[tl] {
			t.Fatalf("no journal row for intermediate call %q", tl)
		}
	}
}

// TestRPCSubagentDeniedCallIsContainedAndJournaled witnesses the containment half:
// a denied intermediate call NEVER executes (adjudication stays REAL over the RPC
// path, not stubbed) and is journaled as a DENY tagged with the subagent id — so
// the collapse never costs containment (the Hermes gap this leaf closes).
func TestRPCSubagentDeniedCallIsContainedAndJournaled(t *testing.T) {
	// The floor allows "fetch" and "parse" but NOT "exfiltrate".
	te := scriptExec(t, []string{"fetch", "parse"}, "ok-output")
	j, path := openJournal(t)

	const subID = "rpc-sub-deny"
	sub, err := microagent.NewRPCSubagent(subID, te, 4096, j)
	if err != nil {
		t.Fatalf("NewRPCSubagent: %v", err)
	}
	script := []microagent.ToolAction{
		{Tool: "fetch", Args: map[string]any{"url": "https://example.test"}},
		{Tool: "exfiltrate", Args: map[string]any{"to": "attacker.test"}}, // not allowed → floor denies
		{Tool: "parse", Args: map[string]any{"format": "json"}},
	}
	res := sub.RunScript(context.Background(), script)
	if err := j.Flush(); err != nil {
		t.Fatalf("journal.Flush: %v", err)
	}

	if res.Allowed != 2 || res.Denied != 1 || res.Errored != 0 {
		t.Fatalf("allowed=%d denied=%d errored=%d, want 2/1/0", res.Allowed, res.Denied, res.Errored)
	}
	// The denied step is contained: it never ran.
	deny := res.Steps[1]
	if deny.Tool != "exfiltrate" {
		t.Fatalf("step 1 tool = %q, want exfiltrate", deny.Tool)
	}
	if deny.Ran {
		t.Fatal("denied call Ran=true — containment dropped over the RPC-subagent path")
	}
	if deny.Verdict.Kind != abi.VerdictDeny {
		t.Fatalf("denied verdict = %v, want VerdictDeny", deny.Verdict.Kind)
	}

	// The chain holds exactly one DENY (the refused call) tagged with the subagent,
	// plus the two allowed DECIDE rows, and verifies end-to-end.
	rows, err := journal.ReadRows(path)
	if err != nil {
		t.Fatalf("journal.ReadRows: %v", err)
	}
	denyRows := 0
	for _, r := range rows {
		if r.Kind == "DENY" {
			denyRows++
			if r.Tool != "exfiltrate" {
				t.Errorf("DENY row tool = %q, want exfiltrate", r.Tool)
			}
			if r.TraceID != subID {
				t.Errorf("DENY row TraceID = %q, want %q", r.TraceID, subID)
			}
		}
	}
	if denyRows != 1 {
		t.Fatalf("DENY rows = %d, want exactly 1 (the refused intermediate call)", denyRows)
	}
	if n, err := journal.Verify(path); err != nil || n != 3 {
		t.Fatalf("journal.Verify = (%d, %v), want (3, nil) — 2 DECIDE + 1 DENY", n, err)
	}
}
