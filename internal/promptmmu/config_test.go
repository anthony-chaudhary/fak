package promptmmu

import (
	"bytes"
	"strings"
	"testing"
)

// TestParseMode_DefaultOn proves the epic posture: an unset flag resolves to the
// default-on curate, and every recognized value round-trips.
func TestParseMode_DefaultOn(t *testing.T) {
	cases := []struct {
		in   string
		want Mode
		ok   bool
	}{
		{"", ModeOn, true}, // unset ⇒ default-on
		{"on", ModeOn, true},
		{"shadow", ModeShadow, true},
		{"off", ModeOff, true},
	}
	for _, c := range cases {
		m, ok := ParseMode(c.in)
		if m != c.want || ok != c.ok {
			t.Errorf("ParseMode(%q) = (%q,%v), want (%q,%v)", c.in, m, ok, c.want, c.ok)
		}
	}
	if DefaultMode != ModeOn {
		t.Fatalf("DefaultMode = %q, want on (epic curate-out-of-the-box posture)", DefaultMode)
	}
}

// TestParseMode_UnknownFailsTowardOn: an unrecognized flag does NOT silently disable
// the curate — it resolves to the default-on posture with ok=false so the host warns.
func TestParseMode_UnknownFailsTowardOn(t *testing.T) {
	m, ok := ParseMode("bogus")
	if ok {
		t.Fatalf("unknown mode should report ok=false")
	}
	if m != DefaultMode || m == ModeOff {
		t.Fatalf("unknown mode must fail toward the default-on posture, got %q", m)
	}
}

// TestModeAppliesAndComputes pins the two control bits the host branches on.
func TestModeAppliesAndComputes(t *testing.T) {
	if !ModeOn.Applies() || ModeShadow.Applies() || ModeOff.Applies() {
		t.Errorf("only ModeOn applies (mutates the body)")
	}
	if !ModeOn.Computes() || !ModeShadow.Computes() || ModeOff.Computes() {
		t.Errorf("ModeOn and ModeShadow compute the plan; ModeOff short-circuits")
	}
}

// TestObserve_OnAppliesAndNamesPruned: in ModeOn a real prune is Applied and the
// pruned names are reported (invariant 3 — the drop is named).
func TestObserve_OnAppliesAndNamesPruned(t *testing.T) {
	res := PruneResult{Pruned: []string{"denied"}, Changed: true}
	obs := Observe(ModeOn, res)
	if !obs.Applied {
		t.Fatalf("ModeOn with a change must be Applied")
	}
	if len(obs.WouldPrune) != 1 || obs.WouldPrune[0] != "denied" {
		t.Fatalf("WouldPrune = %v, want [denied]", obs.WouldPrune)
	}
	if obs.SkipReason != "" {
		t.Fatalf("a real prune has no SkipReason, got %q", obs.SkipReason)
	}
}

// TestObserve_ShadowComputesButDoesNotApply: the safe dogfood rung names what WOULD
// prune but never reports Applied — the body is not mutated.
func TestObserve_ShadowComputesButDoesNotApply(t *testing.T) {
	// The spine ran and WOULD have dropped "denied", but in shadow the host keeps res.Body unused.
	res := PruneResult{Pruned: []string{"denied"}, Changed: true}
	obs := Observe(ModeShadow, res)
	if obs.Applied {
		t.Fatalf("ModeShadow must NEVER report Applied")
	}
	if len(obs.WouldPrune) != 1 || obs.WouldPrune[0] != "denied" {
		t.Fatalf("shadow WouldPrune = %v, want [denied] (audit what it would do)", obs.WouldPrune)
	}
}

// TestObserve_ShadowIdentityCarriesSpineReason: when the spine returns identity in
// shadow mode, the named spine reason is surfaced (no silent nothing).
func TestObserve_ShadowIdentityCarriesSpineReason(t *testing.T) {
	res := PruneResult{Changed: false, SkipReason: SkipNothingAfter}
	obs := Observe(ModeShadow, res)
	if obs.Applied || len(obs.WouldPrune) != 0 {
		t.Fatalf("shadow identity must not apply or name a prune")
	}
	if obs.SkipReason != SkipNothingAfter {
		t.Fatalf("SkipReason = %q, want the spine's named reason %q", obs.SkipReason, SkipNothingAfter)
	}
}

// TestObserve_OffShortCircuits: ModeOff computes nothing and names itself as the
// reason — the body is forwarded byte-identical.
func TestObserve_OffShortCircuits(t *testing.T) {
	// Even if a stray result is passed, off ignores it.
	obs := Observe(ModeOff, PruneResult{Pruned: []string{"x"}, Changed: true})
	if obs.Applied {
		t.Fatalf("ModeOff must never apply")
	}
	if len(obs.WouldPrune) != 0 {
		t.Fatalf("ModeOff computes nothing, WouldPrune should be empty, got %v", obs.WouldPrune)
	}
	if obs.SkipReason != string(ModeOff) {
		t.Fatalf("SkipReason = %q, want %q", obs.SkipReason, ModeOff)
	}
}

// TestObserve_OnIdentityCarriesReason: ModeOn on a spine identity is not Applied and
// carries the spine's named reason (e.g. a mid-session pre-breakpoint refusal).
func TestObserve_OnIdentityCarriesReason(t *testing.T) {
	res := PruneResult{Changed: false, SkipReason: SkipNoBreakpoint}
	obs := Observe(ModeOn, res)
	if obs.Applied {
		t.Fatalf("identity must not be Applied")
	}
	if obs.SkipReason != SkipNoBreakpoint {
		t.Fatalf("SkipReason = %q, want %q", obs.SkipReason, SkipNoBreakpoint)
	}
}

// curatableBody builds a real, well-formed /v1/messages body whose tools[] has a
// cache_control breakpoint on tool[0] and a plan-droppable tool ("denied") strictly
// after it — so the spine genuinely prunes in ModeOn. Returned with the matching plan.
func curatableBody(tb testing.TB) ([]byte, ToolPlan) {
	tb.Helper()
	raw := body(tb, []map[string]any{tool("read_file", true), tool("keep", false), tool("denied", false)}, false)
	return raw, drop("denied")
}

// TestCurate_ShadowNeverMutatesButComputesSamePlanAsOn is the load-bearing rung-4
// invariant: over ONE real body, ModeOn curates (req.Raw mutated, the drop named in
// the Observation), ModeShadow computes the SAME plan but the forwarded body is
// BYTE-FOR-BYTE the original (only logged, never mutated), and ModeOff does nothing.
// The shadow-never-mutates assertion (invariant 3, the safe dogfood rung) is the one
// this test exists to defend — a regression that let shadow swap in the spliced body
// would fail HERE on real bytes, not on a hand-built PruneResult.
func TestCurate_ShadowNeverMutatesButComputesSamePlanAsOn(t *testing.T) {
	raw, plan := curatableBody(t)
	// A defensive copy of the ORIGINAL bytes: any in-place mutation of raw by a buggy
	// shadow path would diverge from this snapshot even if the returned slice aliased raw.
	orig := append([]byte(nil), raw...)

	// --- ModeOn: the body IS mutated; the prune is real and named. ---
	onBody, onObs := Curate(ModeOn, raw, plan, okDecode)
	if !onObs.Applied {
		t.Fatalf("ModeOn over a prunable body must report Applied, obs=%+v", onObs)
	}
	if len(onObs.WouldPrune) != 1 || onObs.WouldPrune[0] != "denied" {
		t.Fatalf("ModeOn WouldPrune = %v, want [denied]", onObs.WouldPrune)
	}
	if bytes.Equal(onBody, orig) {
		t.Fatalf("ModeOn must MUTATE the forwarded body (drop the denied tool); it was byte-identical to the input")
	}
	if len(onBody) >= len(orig) {
		t.Fatalf("ModeOn prune must SHRINK the body: in=%d out=%d", len(orig), len(onBody))
	}
	if contains(toolNamesIn(t, onBody), "denied") {
		t.Fatalf("ModeOn forwarded body still advertises the denied tool: %v", toolNamesIn(t, onBody))
	}

	// --- ModeShadow: SAME plan computed, but the body is NEVER mutated. ---
	shadowBody, shadowObs := Curate(ModeShadow, raw, plan, okDecode)
	if shadowObs.Applied {
		t.Fatalf("ModeShadow must NEVER report Applied, obs=%+v", shadowObs)
	}
	// It computed the SAME plan ModeOn applied — the audit value of the shadow rung.
	if len(shadowObs.WouldPrune) != 1 || shadowObs.WouldPrune[0] != "denied" {
		t.Fatalf("ModeShadow must compute the same WouldPrune as ModeOn, got %v want [denied]", shadowObs.WouldPrune)
	}
	if !eqStrs(shadowObs.WouldPrune, onObs.WouldPrune) {
		t.Fatalf("shadow plan %v differs from on plan %v — shadow must compute the IDENTICAL drop", shadowObs.WouldPrune, onObs.WouldPrune)
	}
	// THE load-bearing assertion: the forwarded body is byte-for-byte the original.
	if !bytes.Equal(shadowBody, orig) {
		t.Fatalf("SHADOW MUTATED THE BODY: forwarded body is not byte-identical to the input (shadow must only LOG, never mutate)")
	}
	// And raw itself was not mutated in place by any path above.
	if !bytes.Equal(raw, orig) {
		t.Fatalf("the original request bytes were mutated in place (a curate path is writing through raw)")
	}

	// --- ModeOff: no plan, no prune, the original body forwarded verbatim. ---
	offBody, offObs := Curate(ModeOff, raw, plan, okDecode)
	if offObs.Applied {
		t.Fatalf("ModeOff must never apply, obs=%+v", offObs)
	}
	if len(offObs.WouldPrune) != 0 {
		t.Fatalf("ModeOff computes nothing; WouldPrune must be empty, got %v", offObs.WouldPrune)
	}
	if offObs.SkipReason != string(ModeOff) {
		t.Fatalf("ModeOff SkipReason = %q, want %q", offObs.SkipReason, ModeOff)
	}
	if !bytes.Equal(offBody, orig) {
		t.Fatalf("ModeOff must forward the body byte-identical to the input")
	}
}

// TestCurate_OffDoesNotRunSpine proves ModeOff short-circuits BEFORE the spine: even a
// decode callback that always errors (which would force the spine to an identity with a
// different SkipReason) is never invoked, and the body is forwarded verbatim.
func TestCurate_OffDoesNotRunSpine(t *testing.T) {
	raw, plan := curatableBody(t)
	called := false
	spyDecode := func(b []byte) error { called = true; return errAlways }

	body, obs := Curate(ModeOff, raw, plan, spyDecode)
	if called {
		t.Fatalf("ModeOff must NOT run the spine (decode was invoked)")
	}
	if obs.SkipReason != string(ModeOff) {
		t.Fatalf("ModeOff SkipReason = %q, want %q", obs.SkipReason, ModeOff)
	}
	if !bytes.Equal(body, raw) {
		t.Fatalf("ModeOff body must be the input verbatim")
	}
}

// TestCurate_ShadowSurfacesSpineIdentityReason: when the spine returns identity in
// shadow (e.g. a pre-breakpoint drop refused mid-session), shadow forwards the body
// unchanged AND surfaces the spine's NAMED reason — no silent nothing.
func TestCurate_ShadowSurfacesSpineIdentityReason(t *testing.T) {
	// breakpoint on the LAST tool: nothing is strictly after it, so the spine refuses.
	raw := body(t, []map[string]any{tool("alpha", false), tool("beta", false), tool("gamma", true)}, false)
	orig := append([]byte(nil), raw...)
	plan := drop("alpha", "beta") // pre-breakpoint ⇒ SkipNothingAfter

	got, obs := Curate(ModeShadow, raw, plan, okDecode)
	if obs.Applied || len(obs.WouldPrune) != 0 {
		t.Fatalf("shadow over a spine identity must not apply or name a prune, obs=%+v", obs)
	}
	if obs.SkipReason != SkipNothingAfter {
		t.Fatalf("shadow must surface the spine's named identity reason, got %q want %q", obs.SkipReason, SkipNothingAfter)
	}
	if !bytes.Equal(got, orig) {
		t.Fatalf("shadow on identity must forward the body byte-identical to the input")
	}
}

func TestExplainToolSchemaStrategyRemoveVsMaskTradeoff(t *testing.T) {
	remove := ExplainToolSchemaStrategy(PruneResult{Changed: true, Pruned: []string{"delete_repo", "web_fetch"}})
	if remove.Strategy != ToolSchemaRemove {
		t.Fatalf("remove strategy = %q, want %q", remove.Strategy, ToolSchemaRemove)
	}
	if len(remove.Removed) != 2 || remove.Removed[0] != "delete_repo" || remove.Removed[1] != "web_fetch" {
		t.Fatalf("remove Removed = %v, want [delete_repo web_fetch]", remove.Removed)
	}
	if !strings.Contains(remove.CacheTradeoff, "byte-identical") || !strings.Contains(remove.TokenTradeoff, "removed 2") {
		t.Fatalf("remove tradeoff not explanatory enough: cache=%q token=%q", remove.CacheTradeoff, remove.TokenTradeoff)
	}

	mask := ExplainToolSchemaStrategy(PruneResult{Changed: false, SkipReason: SkipNothingAfter})
	if mask.Strategy != ToolSchemaMask {
		t.Fatalf("mask strategy = %q, want %q", mask.Strategy, ToolSchemaMask)
	}
	if mask.SkipReason != SkipNothingAfter {
		t.Fatalf("mask SkipReason = %q, want %q", mask.SkipReason, SkipNothingAfter)
	}
	if !strings.Contains(mask.CacheTradeoff, "preserve") || !strings.Contains(mask.TokenTradeoff, "call-time floor") {
		t.Fatalf("mask tradeoff not explanatory enough: cache=%q token=%q", mask.CacheTradeoff, mask.TokenTradeoff)
	}
}
