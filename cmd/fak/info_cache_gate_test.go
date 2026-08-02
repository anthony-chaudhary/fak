package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// TestCompactionGateRowSurfacesDominantBailOnASilentSession is the #5430 witness: a session whose
// compaction lever ran EVERY turn and bailed draws a ZERO shed bar, and before this the pane called
// that a "cold/passthrough session" — the exact misreading that let a mis-sized budget run for hours
// unnoticed. The gate row must now name the EFFECTIVE budget the gateway is really running at and
// the dominant bail reason, and the pane must stop claiming the session was cold.
func TestCompactionGateRowSurfacesDominantBailOnASilentSession(t *testing.T) {
	v := guardInfoVars{}
	v.CacheAttribution = &guardInfoCacheAttribution{} // nothing priced: the calm-zero pane
	v.Adjudication = &gateway.AdjudicationSummary{
		CompactionBudget:      96_000,
		CompactionBailed:      51,
		CompactionBailReasons: map[string]uint64{"under_budget": 51},
	}
	joined := strings.Join(renderInfoCacheAblationRows(guardInfoPanelCtx{v: v, width: 120}), "\n")

	for _, want := range []string{
		compactionGateLabel,
		"budget 96k tok",
		"0 fired / 51 bailed",
		"dominant bail: under_budget x51",
		// The unit misreading the issue names: the budget is applied to messages[] alone.
		"system+tools block is NOT counted",
		"LOWER --compact-history-budget",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("compaction gate row missing %q\n%s", want, joined)
		}
	}
	// The old line was a false statement about a session that compacted 51 times.
	if strings.Contains(joined, "cold/passthrough session") {
		t.Errorf("a session with 51 compaction bails must not be reported as cold/passthrough:\n%s", joined)
	}
	// Byte-clean like every other row in this section: color is layered later.
	if strings.Contains(joined, "\x1b") {
		t.Errorf("gate rows must be byte-clean (no SGR), got %q", joined)
	}
}

// TestCompactionGateRowRidesUnderTheShedBar proves the diagnostic sits with the mechanism it
// explains on a session that DID price savings — the gate row must follow the "fak compaction shed"
// bar, not float off at the end of the section next to an unrelated mechanism.
func TestCompactionGateRowRidesUnderTheShedBar(t *testing.T) {
	v := guardInfoVars{}
	v.CacheAttribution = &guardInfoCacheAttribution{
		ProviderTokenEquiv:      65_100,
		FakTokenEquiv:           30_000,
		FakCompactionShedTokens: 52_300,
		FakKVPrefixReusedTokens: 25_000,
	}
	v.Adjudication = &gateway.AdjudicationSummary{
		CompactionBudget:      48_000,
		CompactionFired:       2,
		CompactionBailed:      9,
		CompactionBailReasons: map[string]uint64{"burst_unprofitable": 9},
	}
	rows := renderInfoCacheAblationRows(guardInfoPanelCtx{v: v, width: 120})
	shedIdx, gateIdx, kvIdx := -1, -1, -1
	for i, r := range rows {
		switch {
		case strings.Contains(r, "fak compaction shed"):
			shedIdx = i
		case strings.Contains(r, compactionGateLabel):
			gateIdx = i
		case strings.Contains(r, "fak KV-prefix reuse"):
			kvIdx = i
		}
	}
	if shedIdx < 0 || gateIdx < 0 || kvIdx < 0 {
		t.Fatalf("want a shed bar, a gate row and a KV bar, got shed=%d gate=%d kv=%d:\n%s",
			shedIdx, gateIdx, kvIdx, strings.Join(rows, "\n"))
	}
	if !(shedIdx < gateIdx && gateIdx < kvIdx) {
		t.Errorf("gate row must sit between the shed bar and the next mechanism, got shed=%d gate=%d kv=%d:\n%s",
			shedIdx, gateIdx, kvIdx, strings.Join(rows, "\n"))
	}
	joined := strings.Join(rows, "\n")
	if !strings.Contains(joined, "2 fired / 9 bailed") || !strings.Contains(joined, "dominant bail: burst_unprofitable x9") {
		t.Errorf("gate row must carry the fired/bailed split and the dominant reason:\n%s", joined)
	}
	if !strings.Contains(joined, "no repaying turn horizon") {
		t.Errorf("burst_unprofitable needs its own actionable gloss, not the under_budget one:\n%s", joined)
	}
}

// TestCompactionGateRowAnchorStarvedOverridesTheUnderBudgetAdvice pins the one gloss that must not
// be generic. anchor-starved is a SUBSET of under_budget and its operational opposite: the protected
// prefix already exceeds the budget, so the plain "lower the budget" advice would send the operator
// the wrong way (no budget value can make the cut fire — #1407).
func TestCompactionGateRowAnchorStarvedOverridesTheUnderBudgetAdvice(t *testing.T) {
	adj := &gateway.AdjudicationSummary{
		CompactionBudget:        96_000,
		CompactionBailed:        12,
		CompactionBailReasons:   map[string]uint64{"under_budget": 12},
		CompactionAnchorStarved: 12,
	}
	joined := strings.Join(compactionGateRows(adj), "\n")
	if !strings.Contains(joined, "ANCHOR-STARVED x12") {
		t.Errorf("want the anchor-starved call-out, got:\n%s", joined)
	}
	if strings.Contains(joined, "LOWER --compact-history-budget") {
		t.Errorf("anchor-starved must NOT advise a lower budget — no budget value makes it fire:\n%s", joined)
	}
}

// TestCompactionGateRowsSilentWithoutACompactionWitness proves the row never fabricates a posture:
// a gateway that reported no adjudication block, or one that attempted no compaction at all, adds
// nothing — so a pane with no compaction witness stays byte-identical to before, and the honest
// "nothing to ablate on a cold/passthrough session" line survives for a genuinely cold session.
func TestCompactionGateRowsSilentWithoutACompactionWitness(t *testing.T) {
	if got := compactionGateRows(nil); got != nil {
		t.Errorf("nil adjudication must add no rows, got %v", got)
	}
	if got := compactionGateRows(&gateway.AdjudicationSummary{CompactionBudget: 48_000}); got != nil {
		t.Errorf("zero attempts must add no rows, got %v", got)
	}
	v := guardInfoVars{}
	v.CacheAttribution = &guardInfoCacheAttribution{}
	v.Adjudication = &gateway.AdjudicationSummary{CompactionBudget: 48_000} // present, but nothing attempted
	joined := strings.Join(renderInfoCacheAblationRows(guardInfoPanelCtx{v: v, width: 100}), "\n")
	if !strings.Contains(joined, "nothing to ablate on a cold/passthrough session") {
		t.Errorf("a truly cold session must keep its honest line, got:\n%s", joined)
	}
	if strings.Contains(joined, compactionGateLabel) {
		t.Errorf("no compaction attempt must render no gate row, got:\n%s", joined)
	}
}

// TestCompactionGateLabelCannotBeMistakenForAPricedBar guards the Cache-tab click hit-test:
// applyInfoCacheMechClick matches a clicked row to a mechanism with strings.Contains(line, label),
// so a diagnostic row carrying a mechanism label would expand a mechanism nobody clicked. The gate
// rows must contain NO mechanism label — the same discipline deferColdMechLabel keeps.
func TestCompactionGateLabelCannotBeMistakenForAPricedBar(t *testing.T) {
	adj := &gateway.AdjudicationSummary{
		CompactionBudget:      96_000,
		CompactionBailed:      3,
		CompactionBailReasons: map[string]uint64{"prefix_mismatch": 3},
	}
	rows := compactionGateRows(adj)
	if len(rows) == 0 {
		t.Fatal("want gate rows to test against")
	}
	for _, m := range cacheAblationMechs(&guardInfoCacheAttribution{}) {
		for _, r := range rows {
			if strings.Contains(r, m.label) {
				t.Errorf("gate row %q contains mechanism label %q — the click hit-test would expand it", r, m.label)
			}
		}
	}
	// The three fak-fault reasons must read as a fault, never as working-as-designed.
	if !strings.Contains(strings.Join(rows, "\n"), "fak-fault") {
		t.Errorf("prefix_mismatch must be called out as a fak-fault:\n%s", strings.Join(rows, "\n"))
	}
}

// TestDominantCompactionBailIsStableUnderATie proves the row does not flicker: with two reasons tied
// at the same count, Go's randomized map iteration would pick a different "dominant" reason each
// render, so an operator would watch the diagnostic oscillate between two equally-true answers.
// The pick is name-ordered, so it is the same one every time.
func TestDominantCompactionBailIsStableUnderATie(t *testing.T) {
	reasons := map[string]uint64{"under_budget": 7, "burst_unprofitable": 7, "no_breakpoint": 2}
	first, n := dominantCompactionBail(reasons)
	if n != 7 {
		t.Fatalf("dominant count = %d, want 7", n)
	}
	for i := 0; i < 200; i++ {
		if got, gotN := dominantCompactionBail(reasons); got != first || gotN != n {
			t.Fatalf("tie-break flickered on iteration %d: %q x%d then %q x%d", i, first, n, got, gotN)
		}
	}
	if got, gotN := dominantCompactionBail(nil); got != "" || gotN != 0 {
		t.Errorf("no bails must yield no dominant reason, got %q x%d", got, gotN)
	}
}

// TestCompactionGateRowNamesTheOffPosture proves the disabled lever is not reported as a budget:
// --compact-history-budget 0 forwards the body byte-for-byte, and printing "budget 0 tok" next to
// "0 fired" would read as a lever that is on and idle — the exact ambiguity CompactionBudget exists
// to resolve.
func TestCompactionGateRowNamesTheOffPosture(t *testing.T) {
	joined := strings.Join(compactionGateRows(&gateway.AdjudicationSummary{CompactionOff: 14}), "\n")
	if !strings.Contains(joined, "OFF") || !strings.Contains(joined, "byte-for-byte") {
		t.Errorf("budget 0 must render as OFF, got:\n%s", joined)
	}
	if strings.Contains(joined, "budget 0 tok") {
		t.Errorf("a disabled lever must not print a budget, got:\n%s", joined)
	}
}

// TestCompactionBailGlossNeverGuesses proves an unrecognised CompactReason* prints its raw name and
// count with NO clause rather than an invented explanation — the pane may be silent about a reason
// it does not know, never wrong about it.
func TestCompactionBailGlossNeverGuesses(t *testing.T) {
	adj := &gateway.AdjudicationSummary{
		CompactionBudget:      48_000,
		CompactionBailed:      4,
		CompactionBailReasons: map[string]uint64{"some_future_reason": 4},
	}
	if gloss := compactionBailGloss("some_future_reason", adj); gloss != "" {
		t.Errorf("an unknown reason must get no gloss, got %q", gloss)
	}
	joined := strings.Join(compactionGateRows(adj), "\n")
	if !strings.Contains(joined, "dominant bail: some_future_reason x4") {
		t.Errorf("an unknown reason must still be named and counted:\n%s", joined)
	}
	if strings.Contains(joined, " — ") {
		t.Errorf("an unknown reason must carry no trailing clause separator:\n%s", joined)
	}
}

// TestDominantCompactionBailExcludesPreEligibleReasons mirrors the committed ledger scenario
// (internal/gatewayusageledger/compaction_noncandidate_test.go): the compactor is attempted on
// EVERY Anthropic passthrough, so a session's auxiliary pings pile into too_few_msgs and that
// benign, pre-eligible bucket outnumbers the actionable bail 190-to-3. Ranking the raw map hands
// the "dominant bail" row to the one group with nothing to do and buries the under_budget bail that
// IS the #5430 failure — the exact misdirection the row exists to prevent. The ranking is over
// compaction CANDIDATES only (agent.CompactBailPreEligible), as HEAD's offline twin already is.
func TestDominantCompactionBailExcludesPreEligibleReasons(t *testing.T) {
	reasons := map[string]uint64{
		"too_few_msgs":  190, // pre-eligible: decided before any compactible span existed
		"non_json":      1,
		"decode_failed": 2,
		"under_budget":  3, // the only actionable one, and it must win despite losing 190-to-3
	}
	got, n := dominantCompactionBail(reasons)
	if got != "under_budget" || n != 3 {
		t.Fatalf("dominant bail = %q x%d, want under_budget x3 (a pre-eligible reason must not win on volume)", got, n)
	}
	// An unregistered reason still ranks as a candidate, so a future CompactReason* can never be
	// silently dropped from the row.
	if got, n := dominantCompactionBail(map[string]uint64{"too_few_msgs": 40, "some_future_reason": 2}); got != "some_future_reason" || n != 2 {
		t.Errorf("an unregistered reason must rank as a candidate, got %q x%d", got, n)
	}
	// The split is exhaustive: every bail lands on exactly one side, so the two totals reconstruct
	// the posture row's raw count and the disclosed denominator can be trusted.
	if cand, pre := candidateBailTotal(reasons), preEligibleBailTotal(reasons); cand != 3 || pre != 193 {
		t.Errorf("bail split = %d candidate / %d pre-eligible, want 3 / 193 (196 total)", cand, pre)
	}
	// The rendered row must disclose its own denominator: the posture row counts 196 bailed while
	// the reason is counted over 3 candidates, and without the held-out clause that reads as an
	// arithmetic error.
	joined := strings.Join(compactionGateRows(&gateway.AdjudicationSummary{
		CompactionBudget:      96_000,
		CompactionBailed:      196,
		CompactionBailReasons: reasons,
	}), "\n")
	if !strings.Contains(joined, "dominant bail: under_budget x3") {
		t.Errorf("gate row must name the candidate winner:\n%s", joined)
	}
	if !strings.Contains(joined, "193 pre-eligible held out") {
		t.Errorf("gate row must disclose the held-out pre-eligible lump:\n%s", joined)
	}
	if !strings.Contains(joined, "LOWER --compact-history-budget") {
		t.Errorf("the winning candidate must carry its own actionable gloss:\n%s", joined)
	}
}

// TestDominantCompactionBailDegradesWhenEveryBailIsPreEligible pins the honest degrade. With the
// filter in place a session can bail N times and have NO candidate to rank, and the row must say
// that — falling silent would leave "N bailed" on screen with no line under it, and naming the
// biggest pre-eligible reason would point the operator at the bucket with nothing to do.
func TestDominantCompactionBailDegradesWhenEveryBailIsPreEligible(t *testing.T) {
	reasons := map[string]uint64{"too_few_msgs": 190, "non_json": 4, "no_messages_key": 1, "decode_failed": 2}
	if got, n := dominantCompactionBail(reasons); got != "" || n != 0 {
		t.Fatalf("an all-pre-eligible map must yield no candidate winner, got %q x%d", got, n)
	}
	joined := strings.Join(compactionGateRows(&gateway.AdjudicationSummary{
		CompactionBudget:      96_000,
		CompactionBailed:      197,
		CompactionBailReasons: reasons,
	}), "\n")
	if !strings.Contains(joined, "no candidate bails: all 197 were pre-eligible") {
		t.Errorf("an all-pre-eligible session must say so, not fall silent:\n%s", joined)
	}
	if strings.Contains(joined, "dominant bail:") {
		t.Errorf("there is no dominant bail to report here:\n%s", joined)
	}
	// Nothing bailed at all is a different fact and keeps its silence.
	if joined := strings.Join(compactionGateRows(&gateway.AdjudicationSummary{
		CompactionBudget: 96_000,
		CompactionFired:  4,
	}), "\n"); strings.Contains(joined, "pre-eligible") {
		t.Errorf("a session with no bails must add no bail line:\n%s", joined)
	}
}

// TestCompactionBailGlossSeparatesMalformedBodyFromACacheBurst holds the row to what the compactor
// actually did. prefix_mismatch / splice_failed / redecode_failed are the three the guard exit
// summary calls a would-be cache BURST (guard_format.go); malformed_body is a different fault —
// the spliced body re-decodes for fak but is Anthropic-invalid, so the request would 400 with the
// protected prefix untouched (internal/agent/anthropic_compact.go). Both are fak's own bug and both
// must stay 0; only one of them is about the cache.
func TestCompactionBailGlossSeparatesMalformedBodyFromACacheBurst(t *testing.T) {
	adj := &gateway.AdjudicationSummary{CompactionBudget: 96_000, CompactionBailed: 2}
	const burst = "would have burst the cache"
	for _, r := range []string{"prefix_mismatch", "splice_failed", "redecode_failed"} {
		gloss := compactionBailGloss(r, adj)
		if !strings.Contains(gloss, burst) || !strings.Contains(gloss, "fak-fault") {
			t.Errorf("%s must keep the guard summary's cache-burst wording, got %q", r, gloss)
		}
	}
	gloss := compactionBailGloss("malformed_body", adj)
	if strings.Contains(gloss, burst) {
		t.Errorf("malformed_body is a 400, not a cache burst — the prefix is intact: %q", gloss)
	}
	if !strings.Contains(gloss, "fak-fault") || !strings.Contains(gloss, "must stay 0") {
		t.Errorf("malformed_body is still fak's own bug and must read as one, got %q", gloss)
	}
	if !strings.Contains(gloss, "400") {
		t.Errorf("malformed_body's gloss must name what actually happens (a 400), got %q", gloss)
	}
}

// TestCompactionGateRowBindsToTheShedBarByLabel earns the "cannot drift" claim the gate row's
// placement rests on. The row is placed by matching compactionShedMechLabel against the mechanism
// being rendered — the same identity applyInfoCacheMechClick hit-tests with — so reordering
// cacheAblationMechs moves the bar and its diagnostic together. cacheMechDetailLines still expands
// the shed provenance under a slot NUMBER (case 1), which no label match can protect, so the slot
// is pinned here too.
func TestCompactionGateRowBindsToTheShedBarByLabel(t *testing.T) {
	mechs := cacheAblationMechs(&guardInfoCacheAttribution{})
	hits := 0
	for _, m := range mechs {
		if m.label == compactionShedMechLabel {
			hits++
		}
	}
	if hits != 1 {
		t.Fatalf("compactionShedMechLabel %q must match exactly one mechanism, matched %d of %d", compactionShedMechLabel, hits, len(mechs))
	}
	if len(mechs) < 2 || mechs[1].label != compactionShedMechLabel {
		t.Errorf("cacheMechDetailLines expands the compaction shed under case 1; slot 1 is %q, want %q — move the case with the mechanism",
			mechs[1].label, compactionShedMechLabel)
	}
	v := guardInfoVars{}
	v.CacheAttribution = &guardInfoCacheAttribution{
		ProviderTokenEquiv:      65_100,
		FakTokenEquiv:           30_000,
		FakCompactionShedTokens: 52_300,
		FakKVPrefixReusedTokens: 25_000,
	}
	v.Adjudication = &gateway.AdjudicationSummary{
		CompactionBudget:      48_000,
		CompactionFired:       2,
		CompactionBailed:      9,
		CompactionBailReasons: map[string]uint64{"burst_unprofitable": 9},
	}
	rows := renderInfoCacheAblationRows(guardInfoPanelCtx{v: v, width: 120})
	labelIdx, gateIdx := -1, -1
	for i, r := range rows {
		switch {
		case strings.Contains(r, compactionShedMechLabel):
			labelIdx = i
		case strings.Contains(r, compactionGateLabel):
			gateIdx = i
		}
	}
	if labelIdx < 0 || gateIdx != labelIdx+1 {
		t.Errorf("the gate row must be the row immediately after the bar carrying %q, got bar=%d gate=%d:\n%s",
			compactionShedMechLabel, labelIdx, gateIdx, strings.Join(rows, "\n"))
	}
}
