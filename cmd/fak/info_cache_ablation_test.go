package main

import (
	"strings"
	"testing"
)

// TestRenderInfoCacheAblationRowsMechanisms proves the live Cache-tab ablation section projects
// the /debug/vars CacheAttribution counters into one savings-bar row per mechanism, in the
// done-order (provider prompt-cache, fak compaction shed, fak KV-prefix reuse) plus the vDSO
// avoided-CALLS line, with the provider read/write split spelled out. The bar of the largest
// mechanism is full; a smaller mechanism carries empty cells — so the bars are honestly scaled.
func TestRenderInfoCacheAblationRowsMechanisms(t *testing.T) {
	v := guardInfoVars{}
	v.CacheAttribution = &guardInfoCacheAttribution{
		ProviderTokenEquiv:                        65_100,
		FakTokenEquiv:                             77_300,
		TotalTokenEquiv:                           142_400,
		FakVDSOAvoidedCalls:                       3,
		ProviderPromptCacheReadTokenEquiv:         67_800,
		ProviderPromptCacheWritePremiumTokenEquiv: -2_700,
		FakCompactionShedTokens:                   52_300,
		FakKVPrefixReusedTokens:                   25_000,
	}
	rows := renderInfoCacheAblationRows(guardInfoPanelCtx{v: v, width: 120})
	joined := strings.Join(rows, "\n")
	for _, want := range []string{
		"── cache ablation",
		"provider prompt-cache",
		"fak compaction shed",
		"fak KV-prefix reuse",
		"fak vDSO memo",
		"reads 67.8k · writes -2.7k",
		"+65.1k tok",
		"+52.3k tok",
		"+25.0k tok",
		"3 engine calls avoided",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("ablation rows missing %q\n%s", want, joined)
		}
	}
	// The rows are BYTE-CLEAN: color is layered later by colorizeGuardInfoBlock, never here.
	if strings.Contains(joined, "\x1b") {
		t.Errorf("ablation rows must be byte-clean (no SGR), got %q", joined)
	}
	// Honest bar scaling: provider (65.1k) is the largest, so its bar is full (no empty ░ cells);
	// fak KV-prefix reuse (25.0k of 65.1k) must carry empty cells.
	var providerRow, kvRow string
	for _, r := range rows {
		switch {
		case strings.Contains(r, "provider prompt-cache"):
			providerRow = r
		case strings.Contains(r, "fak KV-prefix reuse"):
			kvRow = r
		}
	}
	if strings.Contains(providerRow, "░") {
		t.Errorf("largest mechanism (provider) should have a full bar, got %q", providerRow)
	}
	if !strings.Contains(kvRow, "░") {
		t.Errorf("smaller mechanism (kv reuse) should have empty bar cells, got %q", kvRow)
	}
}

// TestCacheAblationMechsReconcileWithRollup proves the per-mechanism ablation bars decompose the
// SAME owner split the rollup line shows: on a WARM session (where the raw shed count is far larger
// than its honest warm/cold-blended value), the compaction-shed bar is priced at the blended value,
// NOT the raw shed — so provider + shed + prefix sum to TotalTokenEquiv exactly. This is the
// cross-check that the bars and the "split default cache X% + fak Y%" line can never tell two
// different stories. Booking the raw shed here (the prior bug) made the sum overshoot the total by
// the warm discount, so a warm session's shed bar dwarfed the provider bar while the rollup's fak
// slice was a fraction of it.
func TestCacheAblationMechsReconcileWithRollup(t *testing.T) {
	// A warm session: 40k shed, 36k of it witnessed as provider cache_reads → the honest shed value
	// is 36k*0.1 + 4k = 7.6k, NOT the raw 40k. FakTokenEquiv = 7.6k + 10k prefix = 17.6k.
	ca := &guardInfoCacheAttribution{
		ProviderTokenEquiv:           60_000,
		FakTokenEquiv:                17_600,
		TotalTokenEquiv:              77_600,
		FakCompactionShedTokens:      40_000,
		FakCompactionCacheReadTokens: 36_000,
		FakKVPrefixReusedTokens:      10_000,
	}

	// The bars sum to the rollup total by construction.
	var sum float64
	for _, m := range cacheAblationMechs(ca) {
		sum += m.tokEq
	}
	if diff := sum - ca.TotalTokenEquiv; diff < -0.5 || diff > 0.5 {
		t.Errorf("mechanism bars sum to %.1f, want TotalTokenEquiv %.1f (bars must decompose the rollup total)", sum, ca.TotalTokenEquiv)
	}

	// The shed bar shows the PRICED 7.6k, never the raw 40.0k, and names the warm discount.
	v := guardInfoVars{}
	v.CacheAttribution = ca
	joined := strings.Join(renderInfoCacheAblationRows(guardInfoPanelCtx{v: v, width: 120}), "\n")
	if !strings.Contains(joined, "+7.6k tok") {
		t.Errorf("shed bar must show the priced 7.6k token-equiv, got:\n%s", joined)
	}
	if strings.Contains(joined, "+40.0k tok") {
		t.Errorf("shed bar must NOT show the raw 40.0k shed count (that overstates the warm session), got:\n%s", joined)
	}
	// The collapsed row names the DELTA (40.0k raw → 7.6k value ≈ 5.3× less) and its cause on the row
	// itself, so the gap the operator sees explains itself at a glance instead of only after a click.
	if !strings.Contains(joined, "worth 5.3× less (already cached)") {
		t.Errorf("shed row should name the warm-discount delta (worth 5.3× less, already cached), got:\n%s", joined)
	}
}

// TestApplyInfoCacheMechClickTogglesDetail proves a click on an ablation bar row expands that
// mechanism's provenance sub-panel, a second click on the same row collapses it, and a click that
// lands off any bar (or off the Cache tab) is inert. The hit-test resolves against the rows the
// interactive block actually renders, so it stays correct through the scroll window and width caps.
func TestApplyInfoCacheMechClickTogglesDetail(t *testing.T) {
	v := guardInfoVars{}
	v.CacheAttribution = &guardInfoCacheAttribution{
		ProviderTokenEquiv:                        60_000,
		FakTokenEquiv:                             17_600,
		TotalTokenEquiv:                           77_600,
		ProviderPromptCacheReadTokenEquiv:         67_800,
		ProviderPromptCacheWritePremiumTokenEquiv: -7_800,
		FakCompactionShedTokens:                   40_000,
		FakCompactionCacheReadTokens:              36_000,
		FakKVPrefixReusedTokens:                   10_000,
	}
	const width, height = 120, 0 // height 0 = roomy: no scroll window, every full row shown
	tr := newGuardInfoTrend(guardInfoTrendCap)
	tr.push(v)
	st := infoViewState{active: viewCache}

	// Locate the shed bar's block row in the rendered block, then click it.
	rows := strings.Split(renderGuardInfoInteractiveBlock(st, v, tr, width, height), "\n")
	shedY := -1
	for i, r := range rows {
		if strings.Contains(r, "fak compaction shed") {
			shedY = i + 1 // block rows are 1-based (row 1 = tab bar)
		}
	}
	if shedY < 0 {
		t.Fatalf("shed bar row not found in rendered block:\n%s", strings.Join(rows, "\n"))
	}

	// Click expands mechanism 2 (shed) and injects its warm/cold provenance.
	st = applyInfoCacheMechClick(st, v, tr, width, height, shedY)
	if st.cacheMech != 2 {
		t.Fatalf("click on shed bar → cacheMech = %d, want 2", st.cacheMech)
	}
	expanded := strings.Join(strings.Split(renderGuardInfoInteractiveBlock(st, v, tr, width, height), "\n"), "\n")
	if !strings.Contains(expanded, "36.0k warm @0.1×") {
		t.Errorf("expanded shed detail should show the warm/cold blend, got:\n%s", expanded)
	}
	// The drill-down installs the intuition, not just the arithmetic: the plain "cheap to keep → cheap
	// to drop" couplet that answers WHY the value is ~5× below the raw shed the operator saw.
	if !strings.Contains(expanded, "cheap to keep (0.1×), so cheap to drop") {
		t.Errorf("expanded shed detail should explain WHY the value is discounted, got:\n%s", expanded)
	}
	if !strings.Contains(expanded, "5.3× gap was already banked by the cache") {
		t.Errorf("expanded shed detail should name the delta the cache already banked, got:\n%s", expanded)
	}
	if !strings.Contains(expanded, " » fak compaction shed") {
		t.Errorf("expanded mechanism row should carry the » marker, got:\n%s", expanded)
	}

	// Re-clicking the same bar collapses it. The bar's row shifts by the two detail lines now above
	// it only if it were below them; shed is the row itself, so re-read its position before clicking.
	rows = strings.Split(renderGuardInfoInteractiveBlock(st, v, tr, width, height), "\n")
	for i, r := range rows {
		if strings.Contains(r, "fak compaction shed") {
			shedY = i + 1
		}
	}
	st = applyInfoCacheMechClick(st, v, tr, width, height, shedY)
	if st.cacheMech != 0 {
		t.Fatalf("re-click on open shed bar → cacheMech = %d, want 0 (collapsed)", st.cacheMech)
	}

	// A click on the section rule row (block row 2, the "── cache ablation ──" line) is inert.
	inert := applyInfoCacheMechClick(infoViewState{active: viewCache}, v, tr, width, height, 2)
	if inert.cacheMech != 0 {
		t.Errorf("click on a non-bar row toggled a mechanism (cacheMech=%d), want inert", inert.cacheMech)
	}

	// A click while a different tab is active never touches cacheMech.
	off := applyInfoCacheMechClick(infoViewState{active: viewAgents}, v, tr, width, height, shedY)
	if off.cacheMech != 0 {
		t.Errorf("click off the Cache tab toggled a mechanism (cacheMech=%d), want inert", off.cacheMech)
	}
}

// TestRenderInfoCacheAblationRowsColdToolDeferDistinct is the #3647 witness: the cold-tool-defer
// shed renders as its OWN block, visually separate from the compaction shed, and names WHICH tools
// went cold. The separation is load-bearing, not cosmetic — defer buys no token-equiv (every def
// still ships; the reduction is provider-side), so it must never acquire a savings bar that would
// read as tokens fak saved. The three shed mechanisms stay distinguishable: compaction shed and
// KV-prefix reuse keep their priced bars, defer gets a ` · ` counted-event row like the vDSO line.
func TestRenderInfoCacheAblationRowsColdToolDeferDistinct(t *testing.T) {
	v := guardInfoVars{}
	v.CacheAttribution = &guardInfoCacheAttribution{
		ProviderTokenEquiv:      60_000,
		FakTokenEquiv:           17_600,
		TotalTokenEquiv:         77_600,
		FakCompactionShedTokens: 40_000,
		FakKVPrefixReusedTokens: 10_000,
		FakDeferColdTurns:       3,
		FakDeferColdCount:       12,
		FakDeferColdToolNames:   []string{"mcp__dos__dos_arbitrate", "mcp__fak__fak_index_docs"},
	}
	rows := renderInfoCacheAblationRows(guardInfoPanelCtx{v: v, width: 120})
	joined := strings.Join(rows, "\n")

	// The defer block reports its own count/turns and names the deferred tools.
	for _, want := range []string{
		"fak cold-tool defer",
		"12 cold tool defs deferred over 3 turns",
		"provider-side, not tokens",
		"deferred: mcp__dos__dos_arbitrate, mcp__fak__fak_index_docs",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("defer block missing %q\n%s", want, joined)
		}
	}

	// The defer row is NOT a savings bar: it carries no gauge cells and no "tok" total, so it can
	// never be misread as a token-equiv the way the compaction-shed bar above it legitimately is.
	var deferRow, shedRow string
	for _, r := range rows {
		switch {
		case strings.Contains(r, "fak cold-tool defer"):
			deferRow = r
		case strings.Contains(r, "fak compaction shed"):
			shedRow = r
		}
	}
	if deferRow == "" || shedRow == "" {
		t.Fatalf("want both a defer row and a shed row, got:\n%s", joined)
	}
	// (The row says "not tokens" in prose, so the discriminator is the gauge glyphs, not "tok".)
	if strings.ContainsAny(deferRow, "█░") {
		t.Errorf("defer row must not render a token savings bar, got %q", deferRow)
	}
	// ...while the compaction shed keeps its priced bar, so the two mechanisms stay distinct.
	if !strings.ContainsAny(shedRow, "█░") {
		t.Errorf("compaction shed must keep its priced bar, got %q", shedRow)
	}
	if deferRow == shedRow {
		t.Errorf("defer and compaction shed collapsed into one row: %q", deferRow)
	}
	if strings.Contains(joined, "\x1b") {
		t.Errorf("ablation rows must be byte-clean (no SGR), got %q", joined)
	}

	// A click hit-test must stay inert on the defer rows: they are informational, not priced bars,
	// so the defer label must not collide with any mechanism label the click resolver matches on.
	for _, m := range cacheAblationMechs(v.CacheAttribution) {
		if strings.Contains(deferRow, m.label) {
			t.Errorf("defer row %q contains mechanism label %q — a click on it would expand that bar", deferRow, m.label)
		}
	}
}

// TestRenderInfoCacheAblationRowsDeferOnlySessionStillRenders is the regression lock on the seam the
// producer already honors and the pane did not: gateway.cacheAttributionVars EMITS the block for a
// defer-on session with zero token activity (defer prices no tokens, so it can never reach
// HasAnyTokenActivity). Before #3647 the pane decoded that populated block and then discarded it
// behind "nothing to ablate on a cold/passthrough session" — calling the session cold while holding
// the witness that fak had deferred tools on it. This is exactly the "defer-on session" the done
// condition names, so it must show the defer shed instead of the cold-session line.
func TestRenderInfoCacheAblationRowsDeferOnlySessionStillRenders(t *testing.T) {
	v := guardInfoVars{}
	v.CacheAttribution = &guardInfoCacheAttribution{
		FakDeferColdTurns:     1,
		FakDeferColdCount:     4,
		FakDeferColdToolNames: []string{"mcp__fak__fak_capabilities"},
	}
	joined := strings.Join(renderInfoCacheAblationRows(guardInfoPanelCtx{v: v, width: 120}), "\n")
	if strings.Contains(joined, "nothing to ablate") {
		t.Errorf("a defer-on session must not read as a cold/passthrough session, got:\n%s", joined)
	}
	if !strings.Contains(joined, "4 cold tool defs deferred over 1 turn ") {
		t.Errorf("defer-only session should report its count/turns (singular turn), got:\n%s", joined)
	}
	if !strings.Contains(joined, "deferred: mcp__fak__fak_capabilities") {
		t.Errorf("defer-only session should name the deferred tool, got:\n%s", joined)
	}
}

// TestDeferColdAblationRowsCapsNamedTools proves a long cold tail collapses instead of flooding the
// pane: an MCP-heavy session defers dozens of tools, and naming every one would dwarf the section.
// The producer sorts the names, so the prefix shown is stable rather than a reshuffling sample.
func TestDeferColdAblationRowsCapsNamedTools(t *testing.T) {
	names := []string{"t01", "t02", "t03", "t04", "t05", "t06", "t07", "t08", "t09"}
	rows := deferColdAblationRows(&guardInfoCacheAttribution{
		FakDeferColdTurns: 2, FakDeferColdCount: uint64(len(names)), FakDeferColdToolNames: names,
	})
	joined := strings.Join(rows, "\n")
	if !strings.Contains(joined, "t06") || strings.Contains(joined, "t07") {
		t.Errorf("want the first %d names only, got:\n%s", deferColdMaxNamed, joined)
	}
	if !strings.Contains(joined, "(+3 more)") {
		t.Errorf("want the collapsed tail count (+3 more), got:\n%s", joined)
	}
	// A defer-off session adds nothing at all, so its pane stays byte-identical to before #3647.
	if got := deferColdAblationRows(&guardInfoCacheAttribution{FakCompactionShedTokens: 40_000}); got != nil {
		t.Errorf("defer-off session must add no rows, got %v", got)
	}
}

// TestRenderInfoCacheAblationRowsEmptyStates proves the section still explains itself when there
// is nothing to show: a nil attribution block (an older gateway) and an all-zero attribution (a
// cold / plain-passthrough session) each render one honest line under the rule, never a blank.
func TestRenderInfoCacheAblationRowsEmptyStates(t *testing.T) {
	nilRows := renderInfoCacheAblationRows(guardInfoPanelCtx{v: guardInfoVars{}, width: 100})
	if !strings.Contains(strings.Join(nilRows, "\n"), "no attribution yet") {
		t.Errorf("nil attribution should render the no-attribution line, got %v", nilRows)
	}

	v := guardInfoVars{}
	v.CacheAttribution = &guardInfoCacheAttribution{}
	zeroRows := renderInfoCacheAblationRows(guardInfoPanelCtx{v: v, width: 100})
	if !strings.Contains(strings.Join(zeroRows, "\n"), "nothing to ablate") {
		t.Errorf("all-zero attribution should render the nothing-to-ablate line, got %v", zeroRows)
	}
}
