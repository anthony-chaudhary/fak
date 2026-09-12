package cachevalue

import (
	"math"
	"strings"
	"testing"
)

// TestSessionPnLOwnerSplitProvesBloatPerSession is the #10948/#1496
// load-bearing case: two sessions with IDENTICAL aggregate token counts —
// indistinguishable in today's fleet-average rollup — but opposite per-owner
// economics. A warm-hit session nets a saving; a churny session whose prefix
// keeps mutating pays full-price writes and nets NEGATIVE. The owner split
// must surface each in its own bucket, so default-on vCache enablement can
// see exactly which owner's workload the cache pays for.
func TestSessionPnLOwnerSplitProvesBloatPerSession(t *testing.T) {
	const prompt = 100000
	warm := Row{
		GeneratedAt:         "2026-09-11T10:00:00Z",
		Owner:               "tenant-a",
		InputTokens:         1000,
		CacheReadTokens:     prompt - 1000 - 1100,
		CacheCreationTokens: 1100,
	}
	churny := Row{
		GeneratedAt:         "2026-09-11T11:00:00Z",
		Owner:               "tenant-b",
		InputTokens:         1000,
		CacheReadTokens:     1000,
		CacheCreationTokens: prompt - 1000 - 1000,
	}
	warmP := ScoreSessionPnL(warm)
	churnyP := ScoreSessionPnL(churny)

	if warmP.Owner != "tenant-a" || churnyP.Owner != "tenant-b" {
		t.Fatalf("owner attribution ran: %q / %q", warmP.Owner, churnyP.Owner)
	}
	// Same prompt extent (100k), so the baselines match — only the cache
	// split differs. Baseline = 100k @ 1x.
	if !within(warmP.BaselineTokEq, prompt, tol) || !within(churnyP.BaselineTokEq, prompt, tol) {
		t.Fatalf("baselines = %.1f / %.1f, want %.1f each (same prompt extent)", warmP.BaselineTokEq, churnyP.BaselineTokEq, float64(prompt))
	}
	// Warm: 97.9k reads @0.1x + 1.1k writes @1.25x + 1k @1x ≈ 12.165k → big net save.
	if warmP.Reason != ReasonSessionNetSaved {
		t.Fatalf("warm session reason = %q, want %q", warmP.Reason, ReasonSessionNetSaved)
	}
	// Churny: only 1k reads @0.1x, 98k writes @1.25x + 1k @1x ≈ 123.6k → net NEGATIVE.
	if churnyP.Reason != ReasonSessionNetNegative {
		t.Fatalf("churny session reason = %q, want %q (the cache-bloat direction)", churnyP.Reason, ReasonSessionNetNegative)
	}
	if !within(warmP.NetSavedTokEq, 100000-(97900*0.1+1100*1.25+1000), tol) {
		t.Errorf("warm net = %.1f, want %.1f", warmP.NetSavedTokEq, 100000-(97900*0.1+1100*1.25+1000))
	}
	if !within(churnyP.NetSavedTokEq, 100000-(1000*0.1+98000*1.25+1000), tol) {
		t.Errorf("churny net = %.1f, want %.1f", churnyP.NetSavedTokEq, 100000-(1000*0.1+98000*1.25+1000))
	}

	// The owner split: identical aggregates would blend these; the split
	// must keep them apart and count the negative session in its bucket.
	splits := SplitOwnerPnL([]Metrics{metrics(warm), metrics(churny)})
	if len(splits) != 2 {
		t.Fatalf("owner split produced %d buckets, want 2", len(splits))
	}
	if splits[0].Owner != "tenant-a" || splits[1].Owner != "tenant-b" {
		t.Fatalf("split order = %q, %q; want deterministic ascending", splits[0].Owner, splits[1].Owner)
	}
	if splits[0].NegativeSessions != 0 || splits[1].NegativeSessions != 1 {
		t.Errorf("negative sessions = %d / %d, want 0 / 1 (the bloat signal)", splits[0].NegativeSessions, splits[1].NegativeSessions)
	}
	if splits[1].NetSavedTokEq >= 0 {
		t.Errorf("tenant-b net = %.1f, want negative", splits[1].NetSavedTokEq)
	}
}

// TestSessionPnLUnstampedOwnerGroupsUnderUnknown: a pre-owner ledger row must
// not vanish — it groups under OwnerUnknown so unattributed economics stay
// visible.
func TestSessionPnLUnstampedOwnerGroupsUnderUnknown(t *testing.T) {
	p := ScoreSessionPnL(Row{GeneratedAt: "t1", CacheReadTokens: 10})
	if p.Owner != OwnerUnknown {
		t.Fatalf("owner = %q, want %q", p.Owner, OwnerUnknown)
	}
}

// TestSessionPnLAllZeroRowIsNeutral: an all-zero row is honestly neutral,
// not skipped — the fail-open discipline the package already uses everywhere.
func TestSessionPnLAllZeroRowIsNeutral(t *testing.T) {
	p := ScoreSessionPnL(Row{GeneratedAt: "t-empty"})
	if p.Reason != ReasonSessionNeutral {
		t.Fatalf("reason = %q, want %q", p.Reason, ReasonSessionNeutral)
	}
	if p.NetSavedTokEq != 0 || p.BaselineTokEq != 0 {
		t.Errorf("zero row should price zero economics, got baseline=%.1f net=%.1f", p.BaselineTokEq, p.NetSavedTokEq)
	}
}

// TestSessionPnLFoldEndToEnd: the ledger DECODES the schema-additive owner
// field — a line stamped with a real owner keeps it after a JSONL round trip
// through the same Fold the report verb uses.
func TestSessionPnLFoldEndToEnd(t *testing.T) {
	r := strings.NewReader(`{"schema":"fak-cache-savings-ledger/1","generated_at":"t1","owner":"tenant-a","input_tokens":100,"cache_read_tokens":900,"cache_creation_tokens":0}
{"schema":"fak-cache-savings-ledger/1","generated_at":"t2","input_tokens":100,"cache_read_tokens":0,"cache_creation_tokens":900}`)
	ms, err := Fold(r)
	if err != nil {
		t.Fatalf("fold: %v", err)
	}
	if len(ms) != 2 || ms[0].Row.Owner != "tenant-a" || ms[1].Row.Owner != "" {
		t.Fatalf("owner decode wrong: %q / %q", ms[0].Row.Owner, ms[1].Row.Owner)
	}
	splits := SplitOwnerPnL(ms)
	if len(splits) != 2 || splits[0].Owner != "tenant-a" || splits[1].Owner != OwnerUnknown {
		t.Fatalf("split order = %+v; want ascending with unknown last", splits)
	}
}

// TestTranslationAcrossEngineAdapters is the #10948/#1498 witness: the SAME
// session's prefix lands on three engines — the fak-native in-kernel KV pages
// (16-token pages), vLLM paged blocks (16-token), SGLang radix units
// (32-token) — each adapter supplying its native block counts. Translation
// must yield per-engine canonical VBlocks whose TOKEN extents (not block
// counts) are cross-engine comparable, partition exactly into cached/written,
// and aggregate per engine deterministically.
func TestTranslationAcrossEngineAdapters(t *testing.T) {
	const session = "2026-09-11T12:00:00Z"
	rows := []EngineBlockRow{
		{Engine: EngineNative, SessionKey: session, Blocks: 4, BlockTokens: 16, Cached: 3, BlockIDs: []string{"n0", "n1", "n2", "n3"}},
		{Engine: EngineVLLM, SessionKey: session, Blocks: 2, BlockTokens: 16, Cached: 1, BlockIDs: []string{"v0", "v1"}},
		{Engine: EngineSGLang, SessionKey: session, Blocks: 2, BlockTokens: 64, Cached: 0, BlockIDs: []string{"sg0", "sg1"}},
		{Engine: "", SessionKey: session, Blocks: 1, BlockTokens: 8, Cached: 1, BlockIDs: []string{"u0"}}, // unstamped → unknown
	}
	sets, err := TranslateRows(rows)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	if len(sets) != 4 {
		t.Fatalf("translated %d sets, want 4", len(sets))
	}

	native := sets[0]
	if native.Engine != EngineNative || len(native.Blocks) != 4 || native.TotalTokens != 64 {
		t.Fatalf("native set = %+v", native)
	}
	if native.CachedTokens != 48 || native.WrittenTokens != 16 {
		t.Errorf("native split = %d/%d, want 48/16 (3 cached of 4 @16 tokens)", native.CachedTokens, native.WrittenTokens)
	}
	// Canonical key is engine+session+index unique; the native page id is
	// still traceable via EngineBlockID.
	if got := native.Blocks[2].Key; got != blockKey(EngineNative, session, "n2") {
		t.Errorf("key = %q, want %q", got, blockKey(EngineNative, session, "n2"))
	}
	if native.Blocks[2].EngineBlockID != "n2" || native.Blocks[2].Tokens != 16 {
		t.Errorf("native block row = %+v", native.Blocks[2])
	}
	// Cross-engine comparability: vLLM's 1 cached block of 2 @16 = 16 cached
	// tokens, HALF its extent — the same fraction as native's 3/4 there.
	vllm := sets[1]
	if vllm.TotalTokens != 32 || vllm.CachedTokens != 16 {
		t.Errorf("vllm tokens = %d/%d, want 32 total / 16 cached", vllm.TotalTokens, vllm.CachedTokens)
	}
	sg := sets[2]
	if sg.Engine != EngineSGLang || sg.TotalTokens != 128 || sg.WrittenTokens != 128 {
		t.Errorf("sglang wrote all %d tokens, cached %d", sg.TotalTokens, sg.CachedTokens)
	}
	unk := sets[3]
	if unk.Engine != EngineUnknown {
		t.Errorf("unstamped engine = %q, want %q", unk.Engine, EngineUnknown)
	}

	// Fail-closed: a row whose cached count exceeds blocks is REJECTED, not
	// silently clamped — a mis-mapped adapter must not yield an economy.
	if _, err := TranslateRow(EngineBlockRow{Engine: EngineVLLM, SessionKey: "s", Blocks: 2, BlockTokens: 16, Cached: 3}); err == nil {
		t.Fatal("cached > blocks accepted; want an invariant error")
	}
	if _, err := TranslateRow(EngineBlockRow{Engine: EngineVLLM, SessionKey: "s", Blocks: -1, BlockTokens: 16}); err == nil {
		t.Fatal("negative blocks accepted; want an invariant error")
	}
	// Partial rejection: the whole translation fails, not a partial set.
	if _, err := TranslateRows([]EngineBlockRow{rows[0], {Engine: EngineVLLM, SessionKey: "s", Blocks: 1, BlockTokens: -1}}); err == nil {
		t.Fatal("TranslateRows accepted a set containing an invalid row; want fail-closed")
	}

	aggs, err := AggregateByEngine(sets)
	if err != nil {
		t.Fatalf("aggregate: %v", err)
	}
	if len(aggs) != 4 || aggs[0].Engine != EngineNative || aggs[3].Engine != EngineVLLM {
		t.Fatalf("aggregates = %+v", aggs)
	}
	if aggs[1].Engine != EngineSGLang || aggs[2].Engine != EngineUnknown {
		t.Errorf("ascending order wrong: %+v", aggs)
	}
	nat := aggs[0]
	if nat.Sessions != 1 || nat.TotalBlocks != 4 || nat.TotalTokens != 64 || nat.CachedTokens != 48 || nat.WrittenTokens != 16 {
		t.Errorf("native aggregate = %+v", nat)
	}
	total := int64(0)
	for _, a := range aggs {
		total += a.TotalTokens
	}
	if total != 64+32+128+8 {
		t.Errorf("summed engine tokens = %d, want 232 (exact across all adapters)", total)
	}
}

// TestE2E_SessionPnLJoinedToVBlocks proves the two seams the issue names
// compose: a ledger row prices the session's P&L, and the SAME session's
// translated blocks explain WHERE the tokens went — the written-token extent
// the P&L bills at 1.25x is the written VBlock extent the translation split,
// per engine, on the shared session key.
func TestE2E_SessionPnLJoinedToVBlocks(t *testing.T) {
	const session = "2026-09-11T13:00:00Z"
	row := Row{
		Schema:              "fak-cache-savings-ledger/1",
		GeneratedAt:         session,
		Owner:               "tenant-a",
		InputTokens:         0,
		CacheReadTokens:     64, // 3 native pages + 1 vLLM block warm
		CacheCreationTokens: 96, // 1 native page + 1 vLLM + 2 sglang @32 written
	}
	p := ScoreSessionPnL(row)
	if p.Reason != ReasonSessionNetSaved {
		t.Fatalf("reason = %q, want %q", p.Reason, ReasonSessionNetSaved)
	}
	sets, err := TranslateRows([]EngineBlockRow{
		{Engine: EngineNative, SessionKey: session, Blocks: 4, BlockTokens: 16, Cached: 3, BlockIDs: []string{"n0", "n1", "n2", "n3"}},
		{Engine: EngineVLLM, SessionKey: session, Blocks: 2, BlockTokens: 16, Cached: 1, BlockIDs: []string{"v0", "v1"}},
		{Engine: EngineSGLang, SessionKey: session, Blocks: 2, BlockTokens: 32, Cached: 0, BlockIDs: []string{"sg0", "sg1"}},
	})
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	cached, written := int64(0), int64(0)
	for _, s := range sets {
		cached += s.CachedTokens
		written += s.WrittenTokens
	}
	if cached != row.CacheReadTokens || written != row.CacheCreationTokens {
		t.Fatalf("block split %d/%d does not reconcile with ledger %d/%d on session %q",
			cached, written, row.CacheReadTokens, row.CacheCreationTokens, session)
	}
	if !within(p.ActualTokEq, 64*0.1+96*1.25, tol) {
		t.Errorf("actual = %.1f, want %.1f", p.ActualTokEq, 64*0.1+96*1.25)
	}
	if p.BaselineTokEq != float64(cached+written) {
		t.Errorf("baseline = %.1f, want %.1f (the no-cache counterfactual over the same extent)", p.BaselineTokEq, float64(cached+written))
	}
}

func TestTranslateRowsRejectsDuplicateCanonicalKeys(t *testing.T) {
	_, err := TranslateRows([]EngineBlockRow{
		{Engine: EngineVLLM, SessionKey: "s", Blocks: 1, BlockTokens: 16, BlockIDs: []string{"native-7"}},
		{Engine: EngineVLLM, SessionKey: "s", Blocks: 1, BlockTokens: 16, BlockIDs: []string{"native-7"}},
	})
	if err == nil {
		t.Fatal("duplicate canonical key accepted")
	}
}

func TestTranslateRowRejectsTokenExtentOverflow(t *testing.T) {
	_, err := TranslateRow(EngineBlockRow{Engine: EngineVLLM, SessionKey: "s", Blocks: 2, BlockTokens: math.MaxInt64, BlockIDs: []string{"a", "b"}})
	if err == nil {
		t.Fatal("overflowing token extent accepted")
	}
	_, err = AggregateByEngine([]VBlockSet{
		{Engine: EngineVLLM, TotalTokens: math.MaxInt64},
		{Engine: EngineVLLM, TotalTokens: 1},
	})
	if err == nil {
		t.Fatal("overflowing aggregate accepted")
	}
}

func TestTranslateRowNativeIdentityContract(t *testing.T) {
	for _, tc := range []struct {
		name string
		row  EngineBlockRow
	}{
		{"short", EngineBlockRow{Blocks: 2, BlockTokens: 1, BlockIDs: []string{"a"}}},
		{"extra", EngineBlockRow{Blocks: 1, BlockTokens: 1, BlockIDs: []string{"a", "b"}}},
		{"empty", EngineBlockRow{Blocks: 1, BlockTokens: 1, BlockIDs: []string{""}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := TranslateRow(tc.row); err == nil {
				t.Fatal("invalid native identity list accepted")
			}
		})
	}
	if blockKey("a", "b:c", "d") == blockKey("a:b", "c", "d") {
		t.Fatal("tuple boundaries collide")
	}
	a, err := TranslateRow(EngineBlockRow{Engine: EngineVLLM, SessionKey: "s", Blocks: 2, BlockTokens: 1, BlockIDs: []string{"stable", "other"}})
	if err != nil {
		t.Fatal(err)
	}
	b, err := TranslateRow(EngineBlockRow{Engine: EngineVLLM, SessionKey: "s", Blocks: 2, BlockTokens: 1, BlockIDs: []string{"other", "stable"}})
	if err != nil {
		t.Fatal(err)
	}
	if a.Blocks[0].Key != b.Blocks[1].Key {
		t.Fatal("native identity changed when adapter row order changed")
	}
}
