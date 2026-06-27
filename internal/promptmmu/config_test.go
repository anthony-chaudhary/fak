package promptmmu

import "testing"

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
