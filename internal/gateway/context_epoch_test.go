package gateway

import (
	"context"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/engine"
)

// context_epoch_test.go — witness tests for the Context Epoch port
// (internal/gateway/context_epoch.go; upstream anomalyco/opencode @ 4eb29a64f005,
// MIT). One behavior per test; closed contracts only (typed outcomes, exact tag
// strings, snapshot contents), never error prose.

func epochConstSource(key, value string) ContextEpochSource {
	return ContextEpochSource{
		Key:      key,
		Load:     func(context.Context) (string, bool) { return value, false },
		Baseline: func(v string) string { return v },
	}
}

func TestContextEpochBaselineBlocksOnUnavailableSource(t *testing.T) {
	source := ContextEpochSource{
		Key:      "gateway/system",
		Load:     func(context.Context) (string, bool) { return "", true },
		Baseline: func(v string) string { return v },
	}
	obs := RenderContextEpochBaseline(context.Background(), []ContextEpochSource{source})
	if obs.Outcome != ContextEpochInitializationBlocked {
		t.Fatalf("outcome = %d, want ContextEpochInitializationBlocked", obs.Outcome)
	}
	if len(obs.BlockedKeys) != 1 || obs.BlockedKeys[0] != "gateway/system" {
		t.Fatalf("blocked keys = %v, want [gateway/system]", obs.BlockedKeys)
	}
	if obs.Generation.Baseline != "" {
		t.Fatalf("baseline = %q, want empty (nothing persisted)", obs.Generation.Baseline)
	}
}

func TestContextEpochStorePrepareMintsOnceAndIsIdempotent(t *testing.T) {
	store := NewContextEpochStore()
	source := epochConstSource("gateway/system", "sys")
	first := store.Prepare(context.Background(), "sess-1", []ContextEpochSource{source}, 0, 5)
	if first.Outcome != ContextEpochReplacementReady || !first.Turnover {
		t.Fatalf("first prepare outcome=%d turnover=%v, want ReplacementReady/true", first.Outcome, first.Turnover)
	}
	if first.Epoch.BaselineSeq != 5 {
		t.Fatalf("first BaselineSeq = %d, want latestSeq 5", first.Epoch.BaselineSeq)
	}
	second := store.Prepare(context.Background(), "sess-1", []ContextEpochSource{source}, 0, 9)
	if second.Outcome != ContextEpochUnchanged || second.Turnover {
		t.Fatalf("second prepare outcome=%d turnover=%v, want Unchanged/false", second.Outcome, second.Turnover)
	}
	if second.Epoch.Baseline != first.Epoch.Baseline {
		t.Fatalf("baseline moved on unchanged turn: %q -> %q", first.Epoch.Baseline, second.Epoch.Baseline)
	}
	if second.Epoch.BaselineSeq != 5 {
		t.Fatalf("second BaselineSeq = %d, want unchanged 5 (higher latestSeq must not move it)", second.Epoch.BaselineSeq)
	}
}

func TestContextEpochUnchangedTurnKeepsBaseline(t *testing.T) {
	store := NewContextEpochStore()
	source := epochConstSource("gateway/system", "sys")
	first := store.Prepare(context.Background(), "sess-1", []ContextEpochSource{source}, 0, 1)
	second := store.Prepare(context.Background(), "sess-1", []ContextEpochSource{source}, 0, 2)
	if second.Outcome != ContextEpochUnchanged || second.Turnover || second.Update != "" {
		t.Fatalf("outcome=%d turnover=%v update=%q, want Unchanged/false/empty", second.Outcome, second.Turnover, second.Update)
	}
	if second.Epoch.Baseline != first.Epoch.Baseline || second.Epoch.BaselineSeq != first.Epoch.BaselineSeq {
		t.Fatalf("baseline/seq moved: %q@%d -> %q@%d",
			first.Epoch.Baseline, first.Epoch.BaselineSeq, second.Epoch.Baseline, second.Epoch.BaselineSeq)
	}
}

func TestContextEpochCompactionTurnsOverAndRestampsBaselineSeq(t *testing.T) {
	store := NewContextEpochStore()
	value := "sys-a"
	source := ContextEpochSource{
		Key:      "gateway/system",
		Load:     func(context.Context) (string, bool) { return value, false },
		Baseline: func(v string) string { return v },
	}
	first := store.Prepare(context.Background(), "sess-1", []ContextEpochSource{source}, 0, 3)
	if first.Epoch.BaselineSeq != 3 {
		t.Fatalf("minted BaselineSeq = %d, want 3", first.Epoch.BaselineSeq)
	}
	value = "sys-b"
	second := store.Prepare(context.Background(), "sess-1", []ContextEpochSource{source}, 7, 9)
	if second.Outcome != ContextEpochReplacementReady || !second.Turnover {
		t.Fatalf("outcome=%d turnover=%v, want ReplacementReady/true", second.Outcome, second.Turnover)
	}
	if second.Epoch.BaselineSeq != 7 {
		t.Fatalf("replacement BaselineSeq = %d, want replacementSeq 7 (not latestSeq 9)", second.Epoch.BaselineSeq)
	}
	if second.Epoch.Baseline == first.Epoch.Baseline {
		t.Fatalf("baseline did not change on replacement: %q", second.Epoch.Baseline)
	}
}

func TestContextEpochStaleReplacementSeqIsIgnored(t *testing.T) {
	store := NewContextEpochStore()
	source := epochConstSource("gateway/system", "sys")
	first := store.Prepare(context.Background(), "sess-1", []ContextEpochSource{source}, 0, 5)
	// replacementSeq <= the stored BaselineSeq must not turn the epoch over
	// (port of context-epoch.ts:59).
	second := store.Prepare(context.Background(), "sess-1", []ContextEpochSource{source}, 5, 9)
	if second.Outcome != ContextEpochUnchanged || second.Turnover {
		t.Fatalf("outcome=%d turnover=%v, want Unchanged/false", second.Outcome, second.Turnover)
	}
	if second.Epoch.BaselineSeq != first.Epoch.BaselineSeq {
		t.Fatalf("stale replacement moved BaselineSeq: %d -> %d", first.Epoch.BaselineSeq, second.Epoch.BaselineSeq)
	}
}

func TestContextEpochChangedSourceWithoutUpdateTurnsOver(t *testing.T) {
	store := NewContextEpochStore()
	value := "sys-a"
	source := ContextEpochSource{
		Key:      "gateway/system",
		Load:     func(context.Context) (string, bool) { return value, false },
		Baseline: func(v string) string { return v },
		// Update nil: the change has no in-place representation.
	}
	first := store.Prepare(context.Background(), "sess-1", []ContextEpochSource{source}, 0, 1)
	value = "sys-b"
	second := store.Prepare(context.Background(), "sess-1", []ContextEpochSource{source}, 0, 2)
	if !second.Turnover {
		t.Fatalf("turnover = false, want true (changed source with no Update renderer)")
	}
	if second.Epoch.Baseline == first.Epoch.Baseline {
		t.Fatalf("baseline unchanged on turnover: %q", second.Epoch.Baseline)
	}
	if second.Epoch.BaselineSeq != 2 {
		t.Fatalf("BaselineSeq = %d, want latestSeq 2", second.Epoch.BaselineSeq)
	}
}

func TestContextEpochChangedSourceWithUpdateStaysInEpoch(t *testing.T) {
	store := NewContextEpochStore()
	value := "sys-a"
	source := ContextEpochSource{
		Key:      "gateway/system",
		Load:     func(context.Context) (string, bool) { return value, false },
		Baseline: func(v string) string { return v },
		Update:   func(previous, current string) string { return "changed " + previous + "->" + current },
	}
	first := store.Prepare(context.Background(), "sess-1", []ContextEpochSource{source}, 0, 1)
	value = "sys-b"
	second := store.Prepare(context.Background(), "sess-1", []ContextEpochSource{source}, 0, 2)
	if second.Outcome != ContextEpochUpdated || second.Turnover {
		t.Fatalf("outcome=%d turnover=%v, want Updated/false", second.Outcome, second.Turnover)
	}
	if second.Update != "changed sys-a->sys-b" {
		t.Fatalf("update = %q, want %q", second.Update, "changed sys-a->sys-b")
	}
	if second.Epoch.Baseline != first.Epoch.Baseline || second.Epoch.BaselineSeq != first.Epoch.BaselineSeq {
		t.Fatalf("in-epoch update moved baseline/seq: %q@%d -> %q@%d",
			first.Epoch.Baseline, first.Epoch.BaselineSeq, second.Epoch.Baseline, second.Epoch.BaselineSeq)
	}
}

func TestContextEpochUndecodableStoredValueTurnsOver(t *testing.T) {
	store := NewContextEpochStore()
	source := ContextEpochSource{
		Key:      "gateway/system",
		Load:     func(context.Context) (string, bool) { return "sys", false },
		Baseline: func(v string) string { return v },
		// Decode always fails: the stored value cannot be compared (upstream
		// codec Incompatible path).
		Decode: func(string) (string, bool) { return "", false },
	}
	store.Prepare(context.Background(), "sess-1", []ContextEpochSource{source}, 0, 1)
	second := store.Prepare(context.Background(), "sess-1", []ContextEpochSource{source}, 0, 2)
	if !second.Turnover {
		t.Fatalf("turnover = false, want true on undecodable stored value")
	}
}

func TestContextEpochRemovedSourceWithoutRemovalRendererTurnsOver(t *testing.T) {
	store := NewContextEpochStore()
	keep := ContextEpochSource{
		Key:      "a",
		Load:     func(context.Context) (string, bool) { return "A", false },
		Baseline: func(v string) string { return v },
		Removed:  func(string) string { return "a removed" },
	}
	gone := ContextEpochSource{
		Key:      "b",
		Load:     func(context.Context) (string, bool) { return "B", false },
		Baseline: func(v string) string { return v },
		// Removed nil: the disappearance cannot be rendered in place.
	}
	store.Prepare(context.Background(), "sess-1", []ContextEpochSource{keep, gone}, 0, 1)
	second := store.Prepare(context.Background(), "sess-1", []ContextEpochSource{keep}, 0, 2)
	if !second.Turnover {
		t.Fatalf("turnover = false, want true (removed source with no removal renderer)")
	}
	if second.Epoch.Baseline != "A" {
		t.Fatalf("replacement baseline = %q, want %q", second.Epoch.Baseline, "A")
	}
}

func TestContextEpochRemovedSourceWithRemovalRendererUpdatesInPlace(t *testing.T) {
	store := NewContextEpochStore()
	keep := ContextEpochSource{
		Key:      "a",
		Load:     func(context.Context) (string, bool) { return "A", false },
		Baseline: func(v string) string { return v },
		Removed:  func(string) string { return "a removed" },
	}
	gone := ContextEpochSource{
		Key:      "b",
		Load:     func(context.Context) (string, bool) { return "B", false },
		Baseline: func(v string) string { return v },
		Removed:  func(string) string { return "b removed" },
	}
	first := store.Prepare(context.Background(), "sess-1", []ContextEpochSource{keep, gone}, 0, 1)
	second := store.Prepare(context.Background(), "sess-1", []ContextEpochSource{keep}, 0, 2)
	if second.Outcome != ContextEpochUpdated || second.Turnover {
		t.Fatalf("outcome=%d turnover=%v, want Updated/false", second.Outcome, second.Turnover)
	}
	if second.Update != "b removed" {
		t.Fatalf("update = %q, want %q", second.Update, "b removed")
	}
	if second.Epoch.Baseline != first.Epoch.Baseline || second.Epoch.BaselineSeq != first.Epoch.BaselineSeq {
		t.Fatalf("in-place removal moved baseline/seq: %q@%d -> %q@%d",
			first.Epoch.Baseline, first.Epoch.BaselineSeq, second.Epoch.Baseline, second.Epoch.BaselineSeq)
	}
}

func TestContextEpochUnavailableTurnPreservesAdmittedSnapshot(t *testing.T) {
	store := NewContextEpochStore()
	available := true
	source := ContextEpochSource{
		Key: "gateway/system",
		Load: func(context.Context) (string, bool) {
			return "sys", !available
		},
		Baseline: func(v string) string { return v },
	}
	first := store.Prepare(context.Background(), "sess-1", []ContextEpochSource{source}, 0, 1)
	available = false
	second := store.Prepare(context.Background(), "sess-1", []ContextEpochSource{source}, 0, 2)
	if second.Turnover {
		t.Fatalf("unavailable turn turned the epoch over; unavailable != removed")
	}
	if second.Epoch.Baseline != first.Epoch.Baseline || second.Epoch.BaselineSeq != first.Epoch.BaselineSeq {
		t.Fatalf("unavailable turn moved baseline/seq: %q@%d -> %q@%d",
			first.Epoch.Baseline, first.Epoch.BaselineSeq, second.Epoch.Baseline, second.Epoch.BaselineSeq)
	}
	if _, ok := second.Epoch.Snapshot["gateway/system"]; !ok {
		t.Fatalf("admitted snapshot was dropped on an unavailable turn")
	}
}

func TestContextEpochResetForcesReinitialization(t *testing.T) {
	store := NewContextEpochStore()
	source := epochConstSource("gateway/system", "sys")
	first := store.Prepare(context.Background(), "sess-1", []ContextEpochSource{source}, 0, 1)
	store.Reset("sess-1")
	if _, ok := store.Lookup("sess-1"); ok {
		t.Fatalf("Reset left the session epoch in place")
	}
	second := store.Prepare(context.Background(), "sess-1", []ContextEpochSource{source}, 0, 2)
	if second.Outcome != ContextEpochReplacementReady || !second.Turnover {
		t.Fatalf("post-reset prepare outcome=%d turnover=%v, want a fresh ReplacementReady/true", second.Outcome, second.Turnover)
	}
	if second.Epoch.BaselineSeq != 2 {
		t.Fatalf("re-initialized BaselineSeq = %d, want latestSeq 2", second.Epoch.BaselineSeq)
	}
	if second.Epoch.Baseline != first.Epoch.Baseline {
		t.Fatalf("re-initialized baseline = %q, want the same rendered source %q", second.Epoch.Baseline, first.Epoch.Baseline)
	}
}

func TestContextEpochGateIsDefaultOff(t *testing.T) {
	gate := newContextEpochGate(false)
	ctx := withPrefixReuseSession(context.Background(), "sess-1")
	messages := []agent.Message{{Role: agent.RoleSystem, Content: "sys"}}
	bound := gate.bind(ctx, messages)
	if tag := contextEpochTagFromContext(bound); tag != "" {
		t.Fatalf("unarmed gate bound tag %q, want empty", tag)
	}
	if _, ok := gate.store.Lookup("sess-1"); ok {
		t.Fatalf("unarmed gate stored an epoch row")
	}
	// A nil receiver must be inert, never panic.
	var nilGate *ContextEpochGate
	if tag := contextEpochTagFromContext(nilGate.bind(ctx, messages)); tag != "" {
		t.Fatalf("nil gate bound tag %q, want empty", tag)
	}
}

func TestContextEpochGateBindsTagAndTurnoverRestamps(t *testing.T) {
	gate := newContextEpochGate(true)
	base := withPrefixReuseSession(context.Background(), "sess-1")
	turn1 := []agent.Message{{Role: agent.RoleSystem, Content: "alpha"}}
	bound1 := gate.bind(base, turn1)
	stored, ok := gate.store.Lookup("sess-1")
	if !ok {
		t.Fatalf("armed gate did not mint an epoch row")
	}
	want1 := "sess-1@" + strconv.FormatInt(stored.BaselineSeq, 10)
	if got := contextEpochTagFromContext(bound1); got != want1 {
		t.Fatalf("turn 1 tag = %q, want %q", got, want1)
	}
	// Same messages: identical tag, no churn.
	bound2 := gate.bind(base, turn1)
	if got := contextEpochTagFromContext(bound2); got != want1 {
		t.Fatalf("turn 2 tag = %q, want identical %q", got, want1)
	}
	// A different leading system block turns the epoch over and restamps the tag.
	turn3 := []agent.Message{{Role: agent.RoleSystem, Content: "beta"}}
	bound3 := gate.bind(base, turn3)
	tag3 := contextEpochTagFromContext(bound3)
	if tag3 == "" || tag3 == want1 {
		t.Fatalf("turn 3 tag = %q, want a fresh non-empty tag distinct from %q", tag3, want1)
	}
	// A ctx with no session identity binds no tag.
	if got := contextEpochTagFromContext(gate.bind(context.Background(), turn1)); got != "" {
		t.Fatalf("session-less ctx bound tag %q, want empty", got)
	}
	// The bound prefix-cache identity carries the same tag.
	identified := agent.WithPrefixCacheIdentity(base, "tenant-a", "")
	boundIdentified := gate.bind(identified, turn1)
	owner, ok := agent.PrefixCacheIdentityFromContext(boundIdentified)
	if !ok {
		t.Fatalf("prefix-cache identity was dropped by bind")
	}
	if owner.Epoch != contextEpochTagFromContext(boundIdentified) {
		t.Fatalf("identity Epoch = %q, want the bound tag %q", owner.Epoch, contextEpochTagFromContext(boundIdentified))
	}
}

// TestContextEpochGateSessionlessTurnMintsNoEpoch pins a KNOWN, WITNESSED
// LIMITATION of the Context Epoch port: bind returns early when the turn carries no
// harness session identity, even if a prefix-cache owner is bound, so an
// unauthenticated/sessionless turn is deliberately NOT epoch-isolated. Any future
// fix that closes this gap must rename this test deliberately.
func TestContextEpochGateSessionlessTurnMintsNoEpoch(t *testing.T) {
	gate := newContextEpochGate(true)
	// A bound prefix-cache identity without a harness prefix-reuse session: the
	// session == "" branch in bind must return ctx untouched.
	ctx := agent.WithPrefixCacheIdentity(context.Background(), "tenant-a", "")
	messages := []agent.Message{{Role: agent.RoleSystem, Content: "alpha"}}
	bound := gate.bind(ctx, messages)
	if tag := contextEpochTagFromContext(bound); tag != "" {
		t.Fatalf("sessionless ctx bound epoch tag %q, want empty (documented limitation)", tag)
	}
	owner, ok := agent.PrefixCacheIdentityFromContext(bound)
	if !ok {
		t.Fatalf("prefix-cache identity was dropped by bind")
	}
	if owner.Epoch != "" {
		t.Fatalf("sessionless ctx epoch = %q, want empty (deliberately NOT epoch-isolated)", owner.Epoch)
	}
	if _, ok := gate.store.Lookup("tenant-a"); ok {
		t.Fatalf("sessionless turn minted an epoch row")
	}
}

func TestContextEpochGateReusesGateOnRealServerPath(t *testing.T) {
	// A sibling test in this package resets the ABI; re-register the mock engine so
	// this witness is order-independent under the focused -run selection.
	abi.RegisterEngine("mock", engine.MockEngine)
	t.Setenv("FAK_ABLATE_CONTEXT_EPOCH", "1")
	armed := newAblateLeverServer(t, Config{})
	if !armed.contextEpoch.isArmed() {
		t.Fatalf("FAK_ABLATE_CONTEXT_EPOCH=1 did not arm the server's ContextEpochGate")
	}
	ctx := withPrefixReuseSession(context.Background(), "sess-1")
	messages := []agent.Message{{Role: agent.RoleSystem, Content: "alpha"}}
	bound := plannerTurnContext(ctx, nil, messages, armed.contextEpoch)
	if tag := contextEpochTagFromContext(bound); tag == "" {
		t.Fatalf("armed server's plannerTurnContext did not bind a Context Epoch tag")
	}
	t.Setenv("FAK_ABLATE_CONTEXT_EPOCH", "")
	unarmed := newAblateLeverServer(t, Config{})
	if unarmed.contextEpoch.isArmed() {
		t.Fatalf("ContextEpochGate armed with no ablation env")
	}
	plain := plannerTurnContext(ctx, nil, messages, unarmed.contextEpoch)
	if tag := contextEpochTagFromContext(plain); tag != "" {
		t.Fatalf("unarmed server's plannerTurnContext bound tag %q, want empty", tag)
	}
}
