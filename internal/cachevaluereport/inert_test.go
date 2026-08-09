package cachevaluereport

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

// findingFor returns the finding for (session, lever) in a report, or nil if absent.
func findingFor(r InertReport, session, lever string) *InertLever {
	for i := range r.Findings {
		if r.Findings[i].Session == session && r.Findings[i].Lever == lever {
			return &r.Findings[i]
		}
	}
	return nil
}

func TestFoldConfiguredButInert_EmptyIsInsufficientButOK(t *testing.T) {
	r := FoldConfiguredButInert(nil, fixedNow)
	if !r.OK {
		t.Fatalf("empty diff should be OK (a loop report, not a gate); got OK=false")
	}
	if r.Verdict != VerdictInsufficient {
		t.Fatalf("empty diff verdict = %q, want %q", r.Verdict, VerdictInsufficient)
	}
	if len(r.Findings) != 0 || r.Sessions != 0 || r.LeversOn != 0 {
		t.Fatalf("empty diff should have no findings/sessions/levers; got %d/%d/%d", len(r.Findings), r.Sessions, r.LeversOn)
	}
	if r.Schema != InertSchema {
		t.Fatalf("schema = %q, want %q", r.Schema, InertSchema)
	}
}

// The first half of the done condition: "a session with defer enabled but 0 cold-defers
// ... emits CONFIGURED_BUT_INERT naming the lever". Every other lever here is effective,
// so ONLY defer_cold_tools must be named.
func TestFoldConfiguredButInert_DeferEnabledZeroColdDefers(t *testing.T) {
	sessions := []SessionLevers{{
		Session:            "sess-defer",
		ManagedCacheActive: true, DeferColdTools: true, UpgradeEnabled: true,
		UpgradesFired: 4, ColdDefers: 0, ReuseRatio: 0.6, // defer is the only dead lever
	}}
	r := FoldConfiguredButInert(sessions, fixedNow)
	if r.Verdict != VerdictConfiguredButInert {
		t.Fatalf("verdict = %q, want %q", r.Verdict, VerdictConfiguredButInert)
	}
	if r.InertLevers != 1 {
		t.Fatalf("inert levers = %d, want 1 (only defer); findings=%+v", r.InertLevers, r.Findings)
	}
	if f := findingFor(r, "sess-defer", LeverDeferColdTools); f == nil {
		t.Fatalf("want a %s finding for sess-defer; got %+v", LeverDeferColdTools, r.Findings)
	}
	if f := findingFor(r, "sess-defer", LeverManagedCache); f != nil {
		t.Fatalf("managed_cache reused (0.6) — must NOT be flagged inert")
	}
	if f := findingFor(r, "sess-defer", LeverCacheTTLUpgrade); f != nil {
		t.Fatalf("upgrades fired (4) — must NOT be flagged inert")
	}
}

// The second half of the done condition: "(or ACTIVE posture but 0 upgrades)". The
// upgrade lever is enabled under an active posture and fires nothing, so cache_ttl_upgrade
// is named; managed-cache reuse is present so managed_cache is NOT inert.
func TestFoldConfiguredButInert_ActivePostureZeroUpgrades(t *testing.T) {
	sessions := []SessionLevers{{
		Session:            "sess-upgrade",
		ManagedCacheActive: true, UpgradeEnabled: true,
		UpgradesFired: 0, ReuseRatio: 0.42, // reuse works, upgrade does not
	}}
	r := FoldConfiguredButInert(sessions, fixedNow)
	if r.Verdict != VerdictConfiguredButInert {
		t.Fatalf("verdict = %q, want %q", r.Verdict, VerdictConfiguredButInert)
	}
	if f := findingFor(r, "sess-upgrade", LeverCacheTTLUpgrade); f == nil {
		t.Fatalf("want a %s finding for 0 upgrades under active posture; got %+v", LeverCacheTTLUpgrade, r.Findings)
	}
	if f := findingFor(r, "sess-upgrade", LeverManagedCache); f != nil {
		t.Fatalf("managed_cache reused (0.42) — must NOT be flagged inert")
	}
}

// The byte-identical-body case: active posture, zero realized reuse → managed_cache inert.
func TestFoldConfiguredButInert_ActivePostureByteIdenticalBody(t *testing.T) {
	sessions := []SessionLevers{{
		Session:            "sess-cold",
		ManagedCacheActive: true, UpgradeEnabled: true,
		UpgradesFired: 2, ReuseRatio: 0, // upgrade fired, but nothing was reused
	}}
	r := FoldConfiguredButInert(sessions, fixedNow)
	if f := findingFor(r, "sess-cold", LeverManagedCache); f == nil {
		t.Fatalf("want a %s finding for 0 reuse under active posture; got %+v", LeverManagedCache, r.Findings)
	}
	if !strings.Contains(r.Findings[0].Effect, "byte-identical body") {
		t.Fatalf("managed_cache effect should name the byte-identical body; got %q", r.Findings[0].Effect)
	}
	if f := findingFor(r, "sess-cold", LeverCacheTTLUpgrade); f != nil {
		t.Fatalf("upgrades fired (2) — must NOT be flagged inert")
	}
}

// A fully-effective session emits none.
func TestFoldConfiguredButInert_FullyEffectiveEmitsNone(t *testing.T) {
	sessions := []SessionLevers{{
		Session:            "sess-good",
		ManagedCacheActive: true, DeferColdTools: true, UpgradeEnabled: true,
		UpgradesFired: 3, ColdDefers: 11, ReuseRatio: 0.71,
	}}
	r := FoldConfiguredButInert(sessions, fixedNow)
	if r.Verdict != VerdictClean {
		t.Fatalf("fully-effective verdict = %q, want %q", r.Verdict, VerdictClean)
	}
	if len(r.Findings) != 0 {
		t.Fatalf("fully-effective session must emit no finding; got %+v", r.Findings)
	}
	if r.LeversOn != 3 {
		t.Fatalf("levers on = %d, want 3", r.LeversOn)
	}
}

// A lever that is OFF can never be inert — a session with nothing armed is INSUFFICIENT,
// not a false CONFIGURED_BUT_INERT.
func TestFoldConfiguredButInert_LeversOffEmitNothing(t *testing.T) {
	sessions := []SessionLevers{{Session: "sess-bare"}}
	r := FoldConfiguredButInert(sessions, fixedNow)
	if r.Verdict != VerdictInsufficient {
		t.Fatalf("no-lever session verdict = %q, want %q", r.Verdict, VerdictInsufficient)
	}
	if len(r.Findings) != 0 || r.LeversOn != 0 {
		t.Fatalf("no armed lever → no findings/levers; got %d findings, %d levers", len(r.Findings), r.LeversOn)
	}
}

// Findings across sessions are ordered by session then lever rank, deterministically.
func TestFoldConfiguredButInert_DeterministicOrder(t *testing.T) {
	sessions := []SessionLevers{
		{Session: "b-sess", ManagedCacheActive: true, UpgradeEnabled: true, ReuseRatio: 0, UpgradesFired: 0},
		{Session: "a-sess", DeferColdTools: true, UpgradeEnabled: true, ColdDefers: 0, UpgradesFired: 0},
	}
	r := FoldConfiguredButInert(sessions, fixedNow)
	if len(r.Findings) != 4 {
		t.Fatalf("want 4 findings (2 per session); got %d: %+v", len(r.Findings), r.Findings)
	}
	wantOrder := []struct{ session, lever string }{
		{"a-sess", LeverDeferColdTools},
		{"a-sess", LeverCacheTTLUpgrade},
		{"b-sess", LeverManagedCache},
		{"b-sess", LeverCacheTTLUpgrade},
	}
	for i, w := range wantOrder {
		if r.Findings[i].Session != w.session || r.Findings[i].Lever != w.lever {
			t.Fatalf("finding[%d] = (%s,%s), want (%s,%s)", i, r.Findings[i].Session, r.Findings[i].Lever, w.session, w.lever)
		}
	}
}

// The captured-session witness: a real gateway-usage EXIT row whose 1h-TTL upgrade lever
// was armed (refusal reasons present) but fired 0 upgrades — the "armed but every head
// refused" session — folds to a CONFIGURED_BUT_INERT finding naming cache_ttl_upgrade.
func TestFoldUsageRowsConfiguredButInert_CapturedAllRefusedSession(t *testing.T) {
	rows := []gatewayusageledger.Row{
		{
			Kind: "exit", SessionType: "guard", SessionID: "guard-refused", PID: 4242, UnixMillis: 1_700_000_000_000,
			Counters: gatewayusageledger.Counters{
				CacheTTLUpgradesUpgraded: 0,
				CacheTTLUpgradeReasons:   map[string]uint64{"head_too_young": 2},
			},
		},
	}
	r := FoldUsageRowsConfiguredButInert(rows, fixedNow)
	if r.Verdict != VerdictConfiguredButInert {
		t.Fatalf("captured all-refused session verdict = %q, want %q; report=%+v", r.Verdict, VerdictConfiguredButInert, r)
	}
	if f := findingFor(r, "guard-refused", LeverCacheTTLUpgrade); f == nil {
		t.Fatalf("want a %s finding for the all-refused session; got %+v", LeverCacheTTLUpgrade, r.Findings)
	}
}

// The effective counterpart: an exit row that actually fired upgrades emits nothing, and
// a periodic row is skipped (it double-counts a still-running session).
func TestFoldUsageRowsConfiguredButInert_EffectiveAndSkipsPeriodic(t *testing.T) {
	rows := []gatewayusageledger.Row{
		{Kind: "exit", SessionID: "good", Counters: gatewayusageledger.Counters{CacheTTLUpgradesUpgraded: 5}},
		{Kind: "periodic", SessionID: "live", Counters: gatewayusageledger.Counters{CacheTTLUpgradeReasons: map[string]uint64{"head_too_young": 9}}},
	}
	r := FoldUsageRowsConfiguredButInert(rows, fixedNow)
	if r.Sessions != 1 {
		t.Fatalf("periodic row must be skipped; diffed sessions = %d, want 1", r.Sessions)
	}
	if r.Verdict != VerdictClean {
		t.Fatalf("an effective upgrade session verdict = %q, want %q", r.Verdict, VerdictClean)
	}
	if len(r.Findings) != 0 {
		t.Fatalf("effective session must emit no finding; got %+v", r.Findings)
	}
}

// The #4349 done condition, half one: a captured gateway-usage EXIT row from a session
// with --defer-cold-tools ARMED but 0 cold-defers folds to CONFIGURED_BUT_INERT naming
// defer_cold_tools. Before the durable intent flag existed this row was indistinguishable
// from a session that never armed the lever, so the finding could not be raised at all.
func TestFoldUsageRowsConfiguredButInert_CapturedArmedDeferZeroColdDefers(t *testing.T) {
	rows := []gatewayusageledger.Row{{
		Kind: "exit", SessionType: "guard", SessionID: "guard-defer-inert", PID: 71, UnixMillis: 1_700_000_000_000,
		Counters: gatewayusageledger.Counters{
			DeferColdToolsArmed: true,
			DeferColdCount:      0, // armed, deferred nothing — the inert lever
			// Reuse is healthy and the upgrade lever is untouched, so defer is the ONLY
			// dead lever: this pins that the new field alone drives the finding.
			KVPrefixPromptTokens: 1000, KVPrefixReusedTokens: 600,
			ManagedCacheActive: true,
		},
	}}
	r := FoldUsageRowsConfiguredButInert(rows, fixedNow)
	if r.Verdict != VerdictConfiguredButInert {
		t.Fatalf("verdict = %q, want %q; report=%+v", r.Verdict, VerdictConfiguredButInert, r)
	}
	if f := findingFor(r, "guard-defer-inert", LeverDeferColdTools); f == nil {
		t.Fatalf("want a %s finding on the armed-but-inert defer row; got %+v", LeverDeferColdTools, r.Findings)
	}
	if f := findingFor(r, "guard-defer-inert", LeverManagedCache); f != nil {
		t.Fatalf("managed_cache reused 600/1000 tokens — it must NOT be named inert: %+v", f)
	}
}

// Half two: an ACTIVE managed-cache posture that realized zero reuse folds to
// CONFIGURED_BUT_INERT naming managed_cache. The explicit posture flag is what makes this
// answerable — inferring intent from KVPrefix reuse alone cannot tell "posture off" from
// "posture on, reused nothing".
func TestFoldUsageRowsConfiguredButInert_CapturedActivePostureZeroReuse(t *testing.T) {
	rows := []gatewayusageledger.Row{{
		Kind: "exit", SessionType: "serve", SessionID: "serve-posture-inert", PID: 72, UnixMillis: 1_700_000_001_000,
		Counters: gatewayusageledger.Counters{
			ManagedCacheActive:   true,
			KVPrefixPromptTokens: 5000, KVPrefixReusedTokens: 0,
			CachedPromptTokens: 0, // nothing reused by EITHER mechanism
			InputTokens:        5000,
		},
	}}
	r := FoldUsageRowsConfiguredButInert(rows, fixedNow)
	if r.Verdict != VerdictConfiguredButInert {
		t.Fatalf("verdict = %q, want %q; report=%+v", r.Verdict, VerdictConfiguredButInert, r)
	}
	if f := findingFor(r, "serve-posture-inert", LeverManagedCache); f == nil {
		t.Fatalf("want a %s finding on the active-posture zero-reuse row; got %+v", LeverManagedCache, r.Findings)
	}
}

// The false "not working" this bridge must never emit: a provider-prompt-cache-only
// session legitimately reuses ZERO fak-authored KV-prefix tokens while its whole payoff
// lands in CachedPromptTokens. Counting only KV reuse would call that active posture
// inert. It is not — and this is the case that kept the lever unarmed before #4349.
func TestFoldUsageRowsConfiguredButInert_ProviderCacheOnlyIsNotInert(t *testing.T) {
	rows := []gatewayusageledger.Row{{
		Kind: "exit", SessionType: "guard", SessionID: "guard-provider-cache", PID: 73, UnixMillis: 1_700_000_002_000,
		Counters: gatewayusageledger.Counters{
			ManagedCacheActive:   true,
			KVPrefixPromptTokens: 0, KVPrefixReusedTokens: 0, // no fak-authored KV reuse at all
			CachedPromptTokens: 4000, // the provider read its own cache — the lever DID pay off
			InputTokens:        1000,
		},
	}}
	r := FoldUsageRowsConfiguredButInert(rows, fixedNow)
	if f := findingFor(r, "guard-provider-cache", LeverManagedCache); f != nil {
		t.Fatalf("provider-side reuse is real reuse — managed_cache must not be named inert: %+v", f)
	}
	if r.Verdict != VerdictClean {
		t.Fatalf("verdict = %q, want %q (the only armed lever paid off)", r.Verdict, VerdictClean)
	}
}

// An unarmed row stays silent: absent intent flags read NOT INSTRUMENTED, never "the
// lever was off and therefore inert". This is the under-report-rather-than-invent
// direction the omitempty fence buys.
func TestFoldUsageRowsConfiguredButInert_UnarmedRowNamesNothing(t *testing.T) {
	rows := []gatewayusageledger.Row{{
		Kind: "exit", SessionType: "serve", SessionID: "serve-bare",
		Counters: gatewayusageledger.Counters{InputTokens: 900},
	}}
	r := FoldUsageRowsConfiguredButInert(rows, fixedNow)
	if len(r.Findings) != 0 {
		t.Fatalf("a row with no lever armed must name nothing; got %+v", r.Findings)
	}
	if r.Verdict != VerdictInsufficient {
		t.Fatalf("verdict = %q, want %q", r.Verdict, VerdictInsufficient)
	}
}

// The render is deterministic and names each inert lever.
func TestRenderConfiguredButInert_NamesLevers(t *testing.T) {
	sessions := []SessionLevers{{Session: "s1", DeferColdTools: true, ColdDefers: 0}}
	out := RenderConfiguredButInert(FoldConfiguredButInert(sessions, fixedNow))
	if !strings.Contains(out, VerdictConfiguredButInert) || !strings.Contains(out, LeverDeferColdTools) {
		t.Fatalf("render should name the verdict and the inert lever; got:\n%s", out)
	}
}
