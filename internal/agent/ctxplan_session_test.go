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

	// Incremental: ingest a growing prefix each turn, exactly as the live loop appends — and
	// assert at EVERY turn (not just the last) that the maintained index is structurally
	// identical to a fresh batch build of that exact prefix. This witnesses incremental
	// maintenance SHAPE: a rebuild-every-turn implementation would also pass, but a maintenance
	// bug that only converged at the end (e.g. a posting list appended in the wrong order on an
	// intermediate turn) would be caught here, where final-state-only equality would miss it.
	incr := NewSessionPlanner(budget)
	for k := 1; k <= len(session); k++ {
		incr.RenderTurn(ctx, session[:k])
		fresh := NewSessionPlanner(budget)
		fresh.PlanTurn(session[:k])
		if !reflect.DeepEqual(incr.Index(), fresh.Index()) {
			t.Fatalf("after turn %d the incremental index != a fresh batch build of session[:%d]", k, k)
		}
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

// TestSessionPlannerBoundedMatchesStatelessFullScan ties the new bounded path to the proven one
// in the FITS-IN-WINDOW regime: on a small session whose entire candidate set fits the recency
// window (every span is probed, so the bounded probe and the full scan score the same set), the
// persistent SessionPlanner renders the EXACT SAME history as the stateless CtxViewPlanner. This
// is the equality case — when nothing is pruned the bounded plan is byte-identical to the full
// scan. The under-pruning regime (where the probe is a strict subset) is the separate
// behavior-PRESERVATION witness TestSessionPlannerBoundedPreservesHighValueUnderPruning, since
// exact equality cannot hold once the full scan fills leftover budget with noise the index prunes.
func TestSessionPlannerBoundedMatchesStatelessFullScan(t *testing.T) {
	ctx := context.Background()
	session := recordedSession()
	const budget = 28 // tight enough to elide, small session so the probe covers every span

	// Guard the premise: the probe genuinely covers every span (nothing pruned), so this test is
	// exercising the equality regime it claims, not silently a pruned one.
	guard := NewSessionPlanner(budget)
	guard.PlanTurn(session)
	if got := len(guard.Index().Probe(guard.forecast(), guard.Opts)); got != guard.Len() {
		t.Fatalf("premise broken: probe (%d) != N (%d); this test only covers the fits-in-window regime", got, guard.Len())
	}

	stateless := &CtxViewPlanner{Enabled: true, Budget: budget}
	statelessOut, err := stateless.RenderTurn(ctx, session)
	if err != nil {
		t.Fatalf("stateless RenderTurn: %v", err)
	}
	spOut := NewSessionPlanner(budget).RenderTurn(ctx, session)
	if !reflect.DeepEqual(spOut, statelessOut) {
		t.Fatalf("bounded SessionPlanner render != stateless full-scan render:\n bounded=%+v\n full=%+v", spOut, statelessOut)
	}
}

// TestSessionPlannerBoundedPreservesHighValueUnderPruning is the load-bearing seam witness: over
// a 600-span session under a tight window where the bounded probe ACTUALLY prunes (a strict
// subset of N), the bounded plan never drops an available high-value span — every span the full
// scan selected that the bounded probe ALSO reached is still selected, and the pins are always
// kept. This is the ctxplan index-vs-full-scan behavior-preservation property (index_test.go),
// proven at the agent seam: pruning changes cost, and only ever drops marginal spans the index
// did not probe, never an available load-bearing one.
func TestSessionPlannerBoundedPreservesHighValueUnderPruning(t *testing.T) {
	ctx := context.Background()
	session := noisySession(600)
	const budget = 48

	sp := NewSessionPlanner(budget)
	sp.Opts = ctxplan.ProbeOptions{RecencyWindow: 8} // tight window so the probe is a strict subset of N
	boundedPlan := sp.PlanTurn(session)

	probe := sp.Index().Probe(sp.forecast(), sp.Opts)
	if len(probe) >= sp.Len() {
		t.Fatalf("probe (%d) did not prune below N (%d) — the test would be vacuous", len(probe), sp.Len())
	}
	probed := map[string]bool{}
	for _, s := range probe {
		probed[s.ID] = true
	}

	stateless := &CtxViewPlanner{Enabled: true, Budget: budget}
	view, err := stateless.PlanTurn(ctx, session)
	if err != nil {
		t.Fatalf("stateless PlanTurn: %v", err)
	}
	fullSel := selIDs(view.Plan)
	boundedSel := selIDs(boundedPlan)

	// No available high-value span is dropped: a full-scan-selected span the bounded probe also
	// reached must still be selected (pruning only frees budget for the survivors).
	for id := range fullSel {
		if probed[id] && !boundedSel[id] {
			t.Errorf("full scan selected %s and the bounded probe reached it, yet the bounded plan dropped it", id)
		}
	}
	// The pinned essentials (system prompt span:0, the goal span:1) are always kept.
	for _, must := range []string{"span:0", "span:1"} {
		if !boundedSel[must] {
			t.Errorf("bounded plan must keep the pinned high-value span %s", must)
		}
	}
	if boundedPlan.CostUsed > budget {
		t.Errorf("bounded plan used %d tokens over budget %d", boundedPlan.CostUsed, budget)
	}
}

// TestSessionPlannerMatchesStatelessOnEmptyRole locks the role-basis edge: a message with an
// EMPTY role is normalized to user for the store + pins, but heuristicForecast derives intents
// only from RAW user roles. SessionPlanner keeps the two bases distinct, so its render stays
// byte-identical to the stateless full-scan path even when a turn carries an unroled message.
func TestSessionPlannerMatchesStatelessOnEmptyRole(t *testing.T) {
	ctx := context.Background()
	session := []Message{
		{Role: RoleSystem, Content: "You are a support agent."},
		{Role: RoleUser, Content: "rotate the auth token"},
		{Role: "", Content: "an unroled trailing note about billing"}, // normalized->user for store/pins, NOT for intents
	}
	const budget = 64

	stateless := &CtxViewPlanner{Enabled: true, Budget: budget}
	statelessOut, err := stateless.RenderTurn(ctx, session)
	if err != nil {
		t.Fatalf("stateless RenderTurn: %v", err)
	}
	spOut := NewSessionPlanner(budget).RenderTurn(ctx, session)
	if !reflect.DeepEqual(spOut, statelessOut) {
		t.Fatalf("SessionPlanner diverged from the stateless path on an empty-role message:\n bounded=%+v\n full=%+v", spOut, statelessOut)
	}
}

// TestSessionPlannerProbeCapEnforced proves the MaxCandidates cap actually BITES: in a session
// where every span matches the forecast intents (so the relevance + recency union far exceeds a
// small cap), the probe is truncated to exactly MaxCandidates, and the pins survive the
// truncation (they are tier 0, never dropped). Without this the "bounded by MaxCandidates" claim
// would be witnessed only on inputs whose natural union coincidentally stays under the cap — the
// cap could be removed and those tests would still pass.
func TestSessionPlannerProbeCapEnforced(t *testing.T) {
	msgs := []Message{
		{Role: RoleSystem, Content: "You are a support agent."},
		{Role: RoleUser, Content: "auth token rotation"},
	}
	for i := 0; i < 100; i++ {
		msgs = append(msgs, Message{Role: RoleTool, Name: "Read", Content: "auth token rotation detail " + itoa(i)})
	}
	msgs = append(msgs, Message{Role: RoleUser, Content: "auth token rotation"})

	sp := NewSessionPlanner(4096)
	sp.Opts = ctxplan.ProbeOptions{MaxCandidates: 16}
	sp.PlanTurn(msgs)

	probe := sp.Index().Probe(sp.forecast(), sp.Opts)
	if len(probe) != 16 {
		t.Fatalf("the MaxCandidates cap must bite: want exactly 16 probed of %d spans, got %d", sp.Len(), len(probe))
	}
	probed := map[string]bool{}
	for _, s := range probe {
		probed[s.ID] = true
	}
	for _, pin := range sp.pins() {
		if !probed[pin] {
			t.Errorf("a pin (%s) must survive the cap truncation (pins are tier 0)", pin)
		}
	}
}

// TestSessionPlannerMaintenanceIsBounded is the cost witness: after T appended turns the index
// holds exactly T spans (each message Add-ed once — O(total), never O(turns²)), re-feeding the
// same history is an idempotent no-op (the append cursor skips already-indexed messages), and
// the probe the planner scores each turn is a STRICT SUBSET of N (the index is selective, not a
// full scan) and never exceeds the cap. The cap-actually-truncates case is the separate
// TestSessionPlannerProbeCapEnforced; here the point is selectivity + O(total) maintenance.
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

// TestSessionPlannerIndexPersistsAndReattaches connects half a to half b at the IN-MEMORY image
// boundary: the persistent per-session index the seam maintains round-trips through the ctxplan
// image (ctxplan.Image -> RestoreIndex — the exact serialization recall.PersistIndex writes to
// disk) and re-attaches to a structurally identical index that plans the turn identically. The
// disk-I/O path (write index.json, read it back) is the separately-witnessed recall layer
// (recall.TestPersistIndexReattachEqualsRebuild); this test scopes itself to the image form the
// seam exposes via Index(), which is what a caller hands to recall.PersistIndex.
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
	layout := ctxplan.DefaultLayout()
	layout.Current = ctxplan.AreaPolicy{MaxSpans: 1, Precision: ctxplan.PrecisionExact}
	layout.Recent = ctxplan.AreaPolicy{MaxSpans: -1}
	layout.Deep = ctxplan.AreaPolicy{MaxSpans: -1}
	cfg := &CtxViewPlanner{Enabled: true, Budget: 777, Layout: &layout}
	sp := cfg.NewSession()
	if sp.Budget != 777 {
		t.Errorf("NewSession must inherit the CtxViewPlanner budget, got %d", sp.Budget)
	}
	if sp.Layout == nil || sp.Layout.Current.Precision != ctxplan.PrecisionExact {
		t.Fatalf("NewSession must inherit the CtxViewPlanner layout, got %+v", sp.Layout)
	}
	layout.Current.Precision = ctxplan.PrecisionPointer
	if sp.Layout.Current.Precision != ctxplan.PrecisionExact {
		t.Fatal("NewSession must copy the layout value so later config mutation does not rewrite a live session")
	}
	// A zero-budget config falls back to the default seed.
	if got := (&CtxViewPlanner{}).NewSession().Budget; got != DefaultCtxViewBudget {
		t.Errorf("a zero-budget config must seed DefaultCtxViewBudget, got %d", got)
	}
}

func TestSessionPlannerFlexibleLayoutKeepsCurrentExact(t *testing.T) {
	session := noisySession(200)
	layout := ctxplan.Layout{
		Base:          ctxplan.AreaPolicy{MaxSpans: 2, Precision: ctxplan.PrecisionExact},
		Current:       ctxplan.AreaPolicy{MaxSpans: 1, Precision: ctxplan.PrecisionExact},
		Recent:        ctxplan.AreaPolicy{MaxSpans: -1},
		Deep:          ctxplan.AreaPolicy{MaxSpans: 2, Precision: ctxplan.PrecisionPointer},
		MaxCandidates: -1,
	}
	sp := NewSessionPlanner(96)
	sp.Layout = &layout
	plan := sp.PlanTurn(session)
	selected := selIDs(plan)

	lastID := ctxplanSpanID(len(session) - 1)
	if !selected[lastID] {
		t.Fatalf("current exact area must force newest span %s resident, selected=%v", lastID, selected)
	}
	var currentMeta bool
	for _, s := range plan.Selected {
		if s.ID == lastID {
			currentMeta = s.Area == ctxplan.AreaCurrent && s.Precision == ctxplan.PrecisionExact && s.Pinned
		}
	}
	if !currentMeta {
		t.Fatalf("newest span must carry current/exact metadata, selected=%+v", plan.Selected)
	}
	for _, e := range plan.Elided {
		if e.Area == ctxplan.AreaDeep && e.Precision == ctxplan.PrecisionPointer && e.Reason != ctxplan.ElidePointer {
			t.Fatalf("deep pointer area must elide as recoverable pointers, got %+v", e)
		}
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
