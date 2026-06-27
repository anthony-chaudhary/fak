package l3region

import (
	"bytes"
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// TestL3PromotionFailsClosed pins the gate's per-class decision under ENFORCE: the
// default floor admits bounded+durable, denies turn+session, and fails CLOSED on a
// missing/unknown class (acceptance criteria 1 + 2). The typed reason distinguishes a
// recognized-but-too-short class from an unknown one.
func TestL3PromotionFailsClosed(t *testing.T) {
	cases := map[string]struct {
		wantAdmit  bool
		wantReason L3Reason
	}{
		"":                    {false, L3ReasonDeniedUnknown},    // missing key — the forward-compat case
		"garbage":             {false, L3ReasonDeniedUnknown},    // unrecognized
		ctxmmu.DurabilityTurn: {false, L3ReasonDeniedBelowFloor}, // recognized, below floor
		"session":             {false, L3ReasonDeniedBelowFloor}, // recognized, below floor
		DurabilityBounded:     {true, L3ReasonAdmitted},          // at the floor (rung-2 reserved, pre-admitted)
		"durable":             {true, L3ReasonAdmitted},          // the canonical promotable class
	}
	for class, want := range cases {
		gate := NewL3PromotionGate().WithMode(L3PromotionEnforce)
		d := gate.Admit(class)
		if d.Admit != want.wantAdmit {
			t.Fatalf("Admit(%q).Admit = %v, want %v", class, d.Admit, want.wantAdmit)
		}
		if d.Reason != want.wantReason {
			t.Fatalf("Admit(%q).Reason = %v, want %v", class, d.Reason, want.wantReason)
		}
		// Promotable is posture-independent: it equals wantAdmit here because the gate is
		// in ENFORCE (Admit == Promotable when enforcing).
		if d.Promotable != want.wantAdmit {
			t.Fatalf("Admit(%q).Promotable = %v, want %v", class, d.Promotable, want.wantAdmit)
		}
	}
}

// TestL3PromotionWarnIsNonBehaviorChanging proves the warn-then-enforce rollout
// (acceptance criterion 4): under WARN a below-floor page is STILL admitted (so a
// misclassification cannot silently strand it), but the would-deny is counted so an
// operator can size the enforce impact before flipping it.
func TestL3PromotionWarnIsNonBehaviorChanging(t *testing.T) {
	gate := NewL3PromotionGate() // default = WARN
	turn := gate.Admit(ctxmmu.DurabilityTurn)
	if !turn.Admit {
		t.Fatalf("WARN: a turn page must still be admitted (non-behavior-changing), got Admit=false")
	}
	if turn.Promotable {
		t.Fatalf("WARN: a turn page must report Promotable=false (it would be denied under ENFORCE)")
	}
	if turn.Reason != L3ReasonDeniedBelowFloor {
		t.Fatalf("WARN: turn reason = %v, want %v", turn.Reason, L3ReasonDeniedBelowFloor)
	}
	durable := gate.Admit(ctxmmu.DurabilityDurable)
	if !durable.Admit || !durable.Promotable || durable.Reason != L3ReasonAdmitted {
		t.Fatalf("WARN: durable decision = %+v, want admitted+promotable", durable)
	}
	if got := gate.DeniedPromotions(); got != 1 {
		t.Fatalf("WARN: would-deny count = %d, want 1 (only the turn page)", got)
	}
}

// TestL3PromotionGateBite is the headline witness (acceptance criterion 3) against the
// REAL L3 store: a turn-class page denied L3 promotion even at high access frequency,
// while a durable page is admitted on its FIRST write — the inversion of CAMA's
// frequency-based admission into a truth-duration one.
func TestL3PromotionGateBite(t *testing.T) {
	ctx := context.Background()
	store := NewL3Store()
	be := New(store)
	gate := NewL3PromotionGate().WithMode(L3PromotionEnforce)

	// --- A hot turn-class page: Put-gated MANY times (simulating high access frequency,
	// the exact signal a W-TinyLFU/SIEVE cache would admit on). The durability gate
	// ignores frequency entirely: every attempt is denied and NOT a single page lands. ---
	turnPage := payload(PageBytes*2 + 7)
	const hits = 50
	for i := 0; i < hits; i++ {
		ref, d, err := be.PutGated(ctx, turnPage, ctxmmu.DurabilityTurn, gate)
		if err != nil {
			t.Fatalf("PutGated(turn) #%d: %v", i, err)
		}
		if d.Admit {
			t.Fatalf("PutGated(turn) #%d: admitted a turn page (Admit=true) — frequency must not promote", i)
		}
		if ref.Digest != "" {
			t.Fatalf("PutGated(turn) #%d: returned a non-zero Ref %q for a denied page", i, ref.Digest)
		}
	}
	if sets, _, _ := store.Stats(); sets != 0 {
		t.Fatalf("hot turn page: L3 saw %d msets, want 0 (a denied page never reaches the shared tier)", sets)
	}
	if got := gate.DeniedPromotions(); got != hits {
		t.Fatalf("denied count = %d, want %d (one per hot attempt)", got, hits)
	}

	// --- A durable page, admitted on its FIRST write (frequency 1). It is mset into L3
	// and resolves bit-exact through the tier. ---
	durablePage := payload(PageBytes + 33)
	ref, d, err := be.PutGated(ctx, durablePage, ctxmmu.DurabilityDurable, gate)
	if err != nil {
		t.Fatalf("PutGated(durable): %v", err)
	}
	if !d.Admit || d.Reason != L3ReasonAdmitted {
		t.Fatalf("PutGated(durable): decision = %+v, want admitted on first write", d)
	}
	if ref.Digest != digest(durablePage) {
		t.Fatalf("PutGated(durable): Ref.Digest = %q, want %q", ref.Digest, digest(durablePage))
	}
	got, err := be.Resolve(ctx, ref)
	if err != nil {
		t.Fatalf("Resolve(durable): %v", err)
	}
	if !bytes.Equal(got, durablePage) {
		t.Fatal("durable page did not round-trip bit-exact through the L3 tier")
	}
}

// TestL3PromotionConfigurableFloor proves the floor is configurable (acceptance
// criterion 1): tightening it to durable-only denies a bounded page that the default
// floor admits; widening it to include session admits a session page.
func TestL3PromotionConfigurableFloor(t *testing.T) {
	// Tighter: only durable earns L3.
	tight := NewL3PromotionGate().WithMode(L3PromotionEnforce).WithFloor(DurabilityDurable)
	if d := tight.Admit(DurabilityBounded); d.Admit {
		t.Fatalf("tight floor: bounded admitted, want denied (only durable earns L3)")
	}
	if d := tight.Admit(DurabilityDurable); !d.Admit {
		t.Fatalf("tight floor: durable denied, want admitted")
	}
	// Wider: session is also durable enough for this deployment.
	wide := NewL3PromotionGate().WithMode(L3PromotionEnforce).WithFloor(DurabilitySession, DurabilityBounded, DurabilityDurable)
	if d := wide.Admit(DurabilitySession); !d.Admit {
		t.Fatalf("wide floor: session denied, want admitted (floor widened)")
	}
	if d := wide.Admit(DurabilityTurn); d.Admit {
		t.Fatalf("wide floor: turn admitted, want denied (turn is still below the widened floor)")
	}
	// A typo in WithFloor cannot silently widen the floor: an unrecognized name is dropped.
	typo := NewL3PromotionGate().WithMode(L3PromotionEnforce).WithFloor("durabel")
	if d := typo.Admit(DurabilityDurable); d.Admit {
		t.Fatalf("typo floor: durable admitted, want denied (the typo'd floor admits nothing)")
	}
}

// TestL3DurabilityVocabularyMatchesCtxmmu is the drift guard: l3region (tier 1) mirrors
// ctxmmu's (tier 2) write-time durability strings as local constants because it may not
// import ctxmmu in production. This test CAN import ctxmmu (architest's layering scan
// excludes _test.go), so it pins the mirror to the source — a divergence reds the build,
// which would otherwise silently make a class the write gate stamps unreadable here.
func TestL3DurabilityVocabularyMatchesCtxmmu(t *testing.T) {
	if DurabilityKey != ctxmmu.DurabilityKey {
		t.Fatalf("DurabilityKey = %q, want ctxmmu's %q", DurabilityKey, ctxmmu.DurabilityKey)
	}
	if DurabilityTurn != ctxmmu.DurabilityTurn {
		t.Fatalf("DurabilityTurn = %q, want ctxmmu's %q", DurabilityTurn, ctxmmu.DurabilityTurn)
	}
	if DurabilitySession != ctxmmu.DurabilitySession {
		t.Fatalf("DurabilitySession = %q, want ctxmmu's %q", DurabilitySession, ctxmmu.DurabilitySession)
	}
	if DurabilityDurable != ctxmmu.DurabilityDurable {
		t.Fatalf("DurabilityDurable = %q, want ctxmmu's %q", DurabilityDurable, ctxmmu.DurabilityDurable)
	}
	// ctxmmu does not export a `bounded` constant (the lexical prior does not emit it
	// yet — it is the rung-2 reserved value documented in mmu.go). Pin the literal so the
	// L3 floor admits the same string the classifier will eventually stamp.
	if DurabilityBounded != "bounded" {
		t.Fatalf("DurabilityBounded = %q, want the reserved literal %q", DurabilityBounded, "bounded")
	}
}

// TestL3PromotionWarnPutGatedStillSets proves the warn-then-enforce rollout (acceptance
// criterion 4) END-TO-END through PutGated against the REAL L3 store — the counterpart of
// TestL3PromotionGateBite's ENFORCE arm. Under WARN a below-floor (turn) page is STILL
// mset into the tier and resolves bit-exact: the rollout is non-behavior-changing, so a
// misclassification cannot strand a page before enforce bites. Yet the would-deny is
// counted, so an operator can size the eventual enforce impact before flipping the posture.
func TestL3PromotionWarnPutGatedStillSets(t *testing.T) {
	ctx := context.Background()
	store := NewL3Store()
	be := New(store)
	gate := NewL3PromotionGate() // default = WARN

	turnPage := payload(PageBytes - 9) // a single L3 page, so one admitted mset is exact
	ref, d, err := be.PutGated(ctx, turnPage, ctxmmu.DurabilityTurn, gate)
	if err != nil {
		t.Fatalf("PutGated(turn, WARN): %v", err)
	}
	// Non-behavior-changing: the turn page IS admitted and lands a real handle...
	if !d.Admit {
		t.Fatalf("WARN: a turn page must still be admitted (non-behavior-changing), got Admit=false")
	}
	if d.Promotable {
		t.Fatalf("WARN: a turn page must report Promotable=false (it would be denied under ENFORCE)")
	}
	if d.Reason != L3ReasonDeniedBelowFloor {
		t.Fatalf("WARN: turn reason = %v, want %v", d.Reason, L3ReasonDeniedBelowFloor)
	}
	if ref.Digest != digest(turnPage) {
		t.Fatalf("WARN: Ref.Digest = %q, want %q (a warn-admitted page lands a real handle)", ref.Digest, digest(turnPage))
	}
	// ...the bytes actually reached the shared tier (a real mset happened, unlike ENFORCE)...
	if sets, _, _ := store.Stats(); sets != 1 {
		t.Fatalf("WARN: mset count = %d, want 1 (a warn-admitted turn page still reaches the tier)", sets)
	}
	got, err := be.Resolve(ctx, ref)
	if err != nil {
		t.Fatalf("WARN: Resolve(turn): %v", err)
	}
	if !bytes.Equal(got, turnPage) {
		t.Fatal("WARN: turn page did not round-trip bit-exact through the tier")
	}
	// ...but the would-deny is still counted so the enforce impact is sized before the flip.
	if got := gate.DeniedPromotions(); got != 1 {
		t.Fatalf("WARN: would-deny count = %d, want 1 (the turn page sized for enforce)", got)
	}
}

// TestL3ReasonString pins the typed reason's audit/metric rendering — the strings an
// operator reads on the rollout dashboards when the gate admits or denies a page —
// including the defensive default that an out-of-range value renders as a deny label
// (fail-closed: an unrecognized reason is never reported as an admit).
func TestL3ReasonString(t *testing.T) {
	cases := map[L3Reason]string{
		L3ReasonAdmitted:         "admitted",
		L3ReasonDeniedBelowFloor: "denied_below_floor",
		L3ReasonDeniedUnknown:    "denied_unknown",
		L3Reason(255):            "denied_unknown", // out-of-range falls closed to a deny label
	}
	for r, want := range cases {
		if got := r.String(); got != want {
			t.Fatalf("L3Reason(%d).String() = %q, want %q", uint8(r), got, want)
		}
	}
}
