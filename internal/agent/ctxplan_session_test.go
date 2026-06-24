package agent

import (
	"context"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// TestSessionPlannerIncrementalEqualsBatch is THE half-b maintenance witness (issue #558): a
// SessionPlanner fed the session turn-by-turn (growing prefixes, the live-loop shape) maintains
// an index STRUCTURALLY IDENTICAL to one fed all the messages at once, and renders the SAME
// history at the final turn. Incremental maintenance is not merely behavior-equivalent — it
// reconstructs the exact index a rebuild would, so the Θ(c·N) compute flatten is real on the
// loop and the per-turn output never depends on how the history was chunked.
func TestSessionPlannerIncrementalEqualsBatch(t *testing.T) {
	ctx := context.Background()
	session := recordedSession()
	const budget = 64

	// Incremental: ingest a growing prefix each turn, exactly as the live loop appends.
	incr := NewSessionPlanner(budget)
	for k := 1; k <= len(session); k++ {
		incr.RenderTurn(ctx, session[:k])
	}
	// Batch: ingest the whole session in one shot.
	batch := NewSessionPlanner(budget)
	batch.RenderTurn(ctx, session)

	// The maintained indexes are structurally identical (span table, posting lists, durable
	// set, id index) — the strongest equivalence, the same reflect.DeepEqual maintain_test uses.
	if !reflect.DeepEqual(incr.Index(), batch.Index()) {
		t.Fatalf("incremental index != batch index structurally")
	}
	// And the final rendered history is byte-identical.
	if !reflect.DeepEqual(incr.RenderTurn(ctx, session), batch.RenderTurn(ctx, session)) {
		t.Fatalf("incremental render != batch render at the final turn")
	}
}

// TestSessionPlannerBoundedMatchesStatelessFullScan ties the new bounded path to the proven
// one: on a real session whose candidate set fits the recency window, the persistent
// SessionPlanner (bounded probe, ctxplan.Index.PlanCells) renders the EXACT SAME history as the
// stateless CtxViewPlanner (full scan, ctxplan.Materialize). The index changes the planner's
// cost, not its output, whenever the bounded candidate set suffices — the common case.
func TestSessionPlannerBoundedMatchesStatelessFullScan(t *testing.T) {
	ctx := context.Background()
	session := recordedSession()
	const budget = 28 // tight enough to elide, small session so the probe covers every span

	stateless := &CtxViewPlanner{Enabled: true, Budget: budget}
	statelessOut, err := stateless.RenderTurn(ctx, session)
	if err != nil {
		t.Fatalf("stateless RenderTurn: %v", err)
	}

	sp := NewSessionPlanner(budget)
	spOut := sp.RenderTurn(ctx, session)

	if !reflect.DeepEqual(spOut, statelessOut) {
		t.Fatalf("bounded SessionPlanner render != stateless full-scan render:\n bounded=%+v\n full=%+v", spOut, statelessOut)
	}
}

// TestSessionPlannerMaintenanceIsBounded is the cost witness: after T appended turns the index
// holds exactly T spans (each message Add-ed once — O(total), never O(turns²)), re-feeding the
// same history is an idempotent no-op (the append cursor skips already-indexed messages), and
// the probe the planner scores each turn stays bounded by MaxCandidates even as the session
// grows far past it — the per-turn work is O(c), not O(N).
func TestSessionPlannerMaintenanceIsBounded(t *testing.T) {
	ctx := context.Background()
	session := recordedSession()

	sp := NewSessionPlanner(64)
	for k := 1; k <= len(session); k++ {
		sp.RenderTurn(ctx, session[:k])
	}
	if sp.Len() != len(session) {
		t.Fatalf("after %d turns the index must hold %d spans, got %d", len(session), len(session), sp.Len())
	}
	// Idempotent: re-feeding the same (already-ingested) history adds nothing.
	sp.RenderTurn(ctx, session)
	if sp.Len() != len(session) {
		t.Fatalf("re-feeding an already-ingested history must not grow the index: got %d", sp.Len())
	}

	// A long, noisy session: the probe stays bounded and far below N.
	big := noisySession(600)
	bp := NewSessionPlanner(256)
	bp.RenderTurn(ctx, big)
	if bp.Len() != len(big) {
		t.Fatalf("index Len=%d != N=%d", bp.Len(), len(big))
	}
	probe := bp.Index().Probe(bp.forecast(), bp.Opts)
	if len(probe) > ctxplan.DefaultMaxCandidates {
		t.Fatalf("probe size %d exceeded MaxCandidates %d — per-turn scan is not bounded", len(probe), ctxplan.DefaultMaxCandidates)
	}
	if len(probe) >= bp.Len() {
		t.Fatalf("probe (%d) did not shrink below the full scan (%d)", len(probe), bp.Len())
	}
}

// TestSessionPlannerExactRecallPreserved is the honesty fence at the seam: a span the bounded
// plan ELIDES (over budget, or pruned from the probe) is never lost — it stays in the lossless
// store and Materialize pages it back in VERBATIM. A forecast/budget miss costs one demand-page,
// never a lost fact, even though the planner only ever probed a bounded subset.
func TestSessionPlannerExactRecallPreserved(t *testing.T) {
	ctx := context.Background()
	session := recordedSession()

	sp := NewSessionPlanner(28) // tight: pins + runbook fit, something elides
	plan := sp.PlanTurn(session)

	selected := map[string]bool{}
	for _, s := range plan.Selected {
		selected[s.ID] = true
	}
	var elidedID string
	for _, e := range plan.Elided {
		if !selected[e.ID] {
			elidedID = e.ID
			break
		}
	}
	if elidedID == "" {
		t.Skip("no elided span under this budget; adjust the test budget")
	}
	body, err := sp.Materialize(ctx, elidedID)
	if err != nil {
		t.Fatalf("an elided span must still page in from the lossless store: %v", err)
	}
	if want := originalContent(session, elidedID); string(body) != want {
		t.Errorf("exact recall broken for %s: recovered %q want verbatim %q", elidedID, string(body), want)
	}
}

// TestSessionPlannerIndexPersistsAndReattaches connects half a to half b: the persistent
// per-session index the seam maintains round-trips through the ctxplan image (the same form
// recall.PersistIndex writes alongside the core image) and re-attaches to a structurally
// identical index that plans the turn identically — so a resumed session re-attaches the seam's
// index instead of rebuilding it.
func TestSessionPlannerIndexPersistsAndReattaches(t *testing.T) {
	session := recordedSession()
	sp := NewSessionPlanner(64)
	sp.PlanTurn(session)

	restored, err := ctxplan.RestoreIndex(sp.Index().Image())
	if err != nil {
		t.Fatalf("RestoreIndex: %v", err)
	}
	if !reflect.DeepEqual(restored, sp.Index()) {
		t.Fatalf("re-attached seam index != live seam index structurally")
	}
	f := sp.forecast()
	live := selIDs(sp.Index().PlanCells(f, ctxplan.Budget{Tokens: 64}, nil, sp.Opts))
	reattached := selIDs(restored.PlanCells(f, ctxplan.Budget{Tokens: 64}, nil, sp.Opts))
	if !reflect.DeepEqual(live, reattached) {
		t.Errorf("re-attached plan != live plan: got %v want %v", reattached, live)
	}
}

// TestSessionPlannerOffByDefaultMirrorsConfig keeps the factory honest: NewSession inherits the
// CtxViewPlanner's budget so a per-session planner plans under the same window the shared config
// declares.
func TestSessionPlannerInheritsBudgetFromCtxViewPlanner(t *testing.T) {
	cfg := &CtxViewPlanner{Enabled: true, Budget: 777}
	sp := cfg.NewSession()
	if sp.Budget != 777 {
		t.Errorf("NewSession must inherit the CtxViewPlanner budget, got %d", sp.Budget)
	}
	// A zero-budget config falls back to the default seed.
	if got := (&CtxViewPlanner{}).NewSession().Budget; got != DefaultCtxViewBudget {
		t.Errorf("a zero-budget config must seed DefaultCtxViewBudget, got %d", got)
	}
}

// noisySession builds a long session: a system prompt, a user goal, a relevant tool result, a
// block of irrelevant turn-scoped noise, then a final user follow-up — the buried-relevant-span
// shape the bounded index must keep tractable.
func noisySession(noise int) []Message {
	msgs := []Message{
		{Role: RoleSystem, Content: "You are a support agent."},
		{Role: RoleUser, Content: "rotate the auth token across all services"},
		{Role: RoleTool, Name: "WebSearch", Content: "auth token rotation runbook: mint roll revoke"},
	}
	for i := 0; i < noise; i++ {
		msgs = append(msgs, Message{Role: RoleTool, Name: "Bash", Content: "build log line " + itoa(i) + " compiled some files"})
	}
	msgs = append(msgs, Message{Role: RoleUser, Content: "now revoke the expiring auth token"})
	return msgs
}

// selIDs is the id set of a plan's resident selection — a local helper for the seam tests.
func selIDs(p ctxplan.Plan) map[string]bool {
	out := make(map[string]bool, len(p.Selected))
	for _, s := range p.Selected {
		out[s.ID] = true
	}
	return out
}
