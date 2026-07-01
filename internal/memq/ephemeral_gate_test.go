package memq_test

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/memq"
)

// TestEphemeralFactRefusedForDurableMemory is the memq-side half of the #1592 done
// condition: a timestamp observation, a current-step observation, and a mood
// observation each attempting to write DIRECTLY into MemStore's durable class are
// refused by AddPromotedIfDurable with no explicit reclassification.
func TestEphemeralFactRefusedForDurableMemory(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{"timestamp", "it's 3pm"},
		{"current_step", "I'm on step 4"},
		{"mood", "I am tired today"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := memq.NewMemStore()
			cell, err := m.AddIfDurable("user", "user", memq.DurabilityDurable, []byte(tc.body), false, memq.ReclassifyNone)
			if err == nil {
				t.Fatalf("AddIfDurable(%q) succeeded, want ErrEphemeralRefused", tc.body)
			}
			if !errors.Is(err, memq.ErrEphemeralRefused) {
				t.Fatalf("AddIfDurable(%q) error = %v, want wrapping ErrEphemeralRefused", tc.body, err)
			}
			if cell.ID != "" {
				t.Fatalf("a refused AddIfDurable must return a zero Cell, got %+v", cell)
			}
			// Confirm nothing was actually stored (the gate ran BEFORE the write, not
			// merely flagged after).
			cells, cerr := m.Cells(nil)
			if cerr != nil {
				t.Fatalf("Cells() error: %v", cerr)
			}
			if len(cells) != 0 {
				t.Fatalf("refused promotion must not append a cell, got %d cells", len(cells))
			}
		})
	}
}

// TestEphemeralFactAllowedWithExplicitReclassification proves the memq-side gate is
// overridable: the same fixtures succeed once an explicit reclassification is supplied,
// and the resulting cell is a real durable cell in the store.
func TestEphemeralFactAllowedWithExplicitReclassification(t *testing.T) {
	fixtures := []string{"it's 3pm", "I'm on step 4", "I am tired today"}
	overrides := []string{
		memq.ReclassifyExplicitConsent,
		memq.ReclassifyUserConfirmed,
		memq.ReclassifyEstablishedPattern,
	}

	for _, body := range fixtures {
		for _, r := range overrides {
			m := memq.NewMemStore()
			cell, err := m.AddIfDurable("user", "user", memq.DurabilityDurable, []byte(body), false, r)
			if err != nil {
				t.Fatalf("AddIfDurable(%q, %q) refused despite explicit reclassification: %v", body, r, err)
			}
			if cell.ID == "" {
				t.Fatalf("AddIfDurable(%q, %q) returned a zero cell on success", body, r)
			}
			if cell.Durability != memq.DurabilityDurable {
				t.Fatalf("promoted cell Durability = %q, want %q", cell.Durability, memq.DurabilityDurable)
			}
			cells, cerr := m.Cells(nil)
			if cerr != nil {
				t.Fatalf("Cells() error: %v", cerr)
			}
			if len(cells) != 1 {
				t.Fatalf("allowed promotion must append exactly one cell, got %d", len(cells))
			}
		}
	}
}

// TestEphemeralGateOnlyAppliesToDurableTargetedWrites confirms the gate does not
// misfire on ordinary turn/session-class writes — those were never eligible for
// promotion in the first place (CONTEXT-IS-NOT-MEMORY.md's decision tree), so gating
// them would incorrectly refuse plain context writes that never targeted durable
// memory at all.
func TestEphemeralGateOnlyAppliesToDurableTargetedWrites(t *testing.T) {
	m := memq.NewMemStore()
	cell, err := m.AddIfDurable("clock", "system", memq.DurabilityTurn, []byte("it's 3pm"), false, memq.ReclassifyNone)
	if err != nil {
		t.Fatalf("a turn-targeted write must never be gated, got error: %v", err)
	}
	if cell.Durability != memq.DurabilityTurn {
		t.Fatalf("cell Durability = %q, want %q", cell.Durability, memq.DurabilityTurn)
	}

	m2 := memq.NewMemStore()
	cell2, err2 := m2.AddIfDurable("task", "system", memq.DurabilitySession, []byte("I'm on step 4"), false, memq.ReclassifyNone)
	if err2 != nil {
		t.Fatalf("a session-targeted write must never be gated, got error: %v", err2)
	}
	if cell2.Durability != memq.DurabilitySession {
		t.Fatalf("cell Durability = %q, want %q", cell2.Durability, memq.DurabilitySession)
	}
}

// TestEphemeralGateAllowsGenuineDurableWrites proves the memq-side wiring does not
// over-refuse: a genuinely durable-shaped fact (a stated preference) targeting durable
// memory succeeds with no reclassification needed.
func TestEphemeralGateAllowsGenuineDurableWrites(t *testing.T) {
	m := memq.NewMemStore()
	cell, err := m.AddIfDurable("user", "user", memq.DurabilityDurable, []byte("I prefer concise answers"), false, memq.ReclassifyNone)
	if err != nil {
		t.Fatalf("a genuinely durable fact must not be refused: %v", err)
	}
	if cell.Durability != memq.DurabilityDurable {
		t.Fatalf("cell Durability = %q, want %q", cell.Durability, memq.DurabilityDurable)
	}
}

// TestEphemeralGateSealedBypassesGate confirms a sealed (quarantined) write is not
// run through the ephemeral gate at all — a sealed cell's bytes never promote (the
// trust gate already refuses page-in), so the ephemeral gate has nothing to add and
// must not spuriously block a legitimate quarantine record.
func TestEphemeralGateSealedBypassesGate(t *testing.T) {
	m := memq.NewMemStore()
	cell, err := m.AddIfDurable("tool", "tool_result", memq.DurabilityDurable, []byte("it's 3pm"), true, memq.ReclassifyNone)
	if err != nil {
		t.Fatalf("a sealed write must bypass the ephemeral gate, got error: %v", err)
	}
	if !cell.Sealed {
		t.Fatalf("cell.Sealed = false, want true")
	}
}

// TestGateEphemeralPromotionWrapsCtxmmu proves memq's thin wrapper delegates to
// ctxmmu.GateEphemeral faithfully (outcome shape, not reimplemented logic), and that
// memq's plain-string reclassification vocabulary normalizes fail-closed exactly like
// NormConsent/NormDurability.
func TestGateEphemeralPromotionWrapsCtxmmu(t *testing.T) {
	direct := ctxmmu.GateEphemeral("I am tired today", ctxmmu.ReclassifyNone)
	viaMemq := memq.GateEphemeralPromotion("I am tired today", memq.ReclassifyNone)
	if direct.Allowed != viaMemq.Allowed || direct.Situational != viaMemq.Situational {
		t.Fatalf("memq.GateEphemeralPromotion diverged from ctxmmu.GateEphemeral: direct=%+v viaMemq=%+v", direct, viaMemq)
	}

	// An unrecognized reclassification string fails closed to "no override" — same
	// posture as NormConsent/NormDurability.
	if got := memq.NormReclassification("yolo"); got != memq.ReclassifyNone {
		t.Errorf("NormReclassification(bogus) = %q, want %q", got, memq.ReclassifyNone)
	}
	bogus := memq.GateEphemeralPromotion("I am tired today", "yolo")
	if bogus.Allowed {
		t.Fatalf("an unrecognized reclassification string must not override the gate, got Allowed=true")
	}
}
