package recall

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// TestPromotionClassFailsClosed pins the reader-side forward-compat default of the
// rung-1 promotion gate (#499 item 4 / #500 item 5): a missing key (a verdict from a
// gate that does not classify) and any unknown/reserved value (e.g. `bounded`, which
// has no validity home until rung 2) both normalize to turn — the shortest, refused
// class. Mirrors abi.FallbackDeny: remembering-when-wrong is the expensive direction.
func TestPromotionClassFailsClosed(t *testing.T) {
	cases := map[string]string{
		"":        ctxmmu.DurabilityTurn,    // missing key — the literal forward-compat case
		"bounded": ctxmmu.DurabilityTurn,    // reserved, no validity home until rung 2
		"garbage": ctxmmu.DurabilityTurn,    // unrecognized
		"turn":    ctxmmu.DurabilityTurn,    // recognized, passes through
		"session": ctxmmu.DurabilitySession, // recognized, passes through (still non-durable)
		"durable": ctxmmu.DurabilityDurable, // the only promotable class
	}
	for in, want := range cases {
		if got := promotionClass(in); got != want {
			t.Fatalf("promotionClass(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestDurabilityPromotionGateBite is the single end-to-end witness of the rung-1
// default-expire inversion (#499/#500) on real shipped code — the write gate
// (ctxmmu.MMU.Admit stamping Verdict.Meta["durability"]) and the durable boundary
// (recall's promotion gate) — proving the HEADLINE thesis: expire by default,
// promotion is the earned exception.
func TestDurabilityPromotionGateBite(t *testing.T) {
	ctx := context.Background()

	const turnFact = "it's 3pm" // a transient observation: true only this turn
	const durableFact = "the user prefers afternoons"

	// --- 1. WARN mode (the default) is NON-BEHAVIOR-CHANGING: the turn page STILL
	// persists and round-trips, but Page.Durability is stamped and the would-refuse is
	// counted. This proves the two-commit honesty split before enforce bites. ---
	t.Run("warn_stamps_but_persists", func(t *testing.T) {
		r := NewRecorder("warn-mode") // default = PromotionWarn
		r.Record(ctx, "clock", []byte(turnFact))
		if got := r.RefusedPromotions(); got != 1 {
			t.Fatalf("WARN: would-refuse count = %d, want 1", got)
		}
		dir := t.TempDir()
		if err := r.Persist(dir); err != nil {
			t.Fatalf("persist: %v", err)
		}
		s, err := Load(dir)
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		if n := len(s.Pages()); n != 1 {
			t.Fatalf("WARN: page count = %d, want 1 (non-behavior-changing: still persists)", n)
		}
		if d := s.Pages()[0].Durability; d != ctxmmu.DurabilityTurn {
			t.Fatalf("WARN: Page.Durability = %q, want %q (stamped)", d, ctxmmu.DurabilityTurn)
		}
		if _, err := s.Resolve(ctx, 0); err != nil {
			t.Fatalf("WARN: turn page must still round-trip, got err: %v", err)
		}
	})

	// --- 2. ENFORCE mode: the turn page is NOT promoted. Its bytes never reach
	// cas.json and its page never reaches manifest.json, so it cannot be recalled in a
	// later process. The earned exception (durable) IS promoted and resolvable. ---
	t.Run("enforce_turn_refused_durable_promoted", func(t *testing.T) {
		// 2a. turn -> refused.
		{
			r := NewRecorder("enforce-turn").WithPromotion(PromotionEnforce)
			r.Record(ctx, "clock", []byte(turnFact))
			if got := r.RefusedPromotions(); got != 1 {
				t.Fatalf("ENFORCE turn: refused count = %d, want 1", got)
			}
			dir := t.TempDir()
			if err := r.Persist(dir); err != nil {
				t.Fatalf("persist: %v", err)
			}
			s, err := Load(dir)
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			if n := len(s.Pages()); n != 0 {
				t.Fatalf("ENFORCE turn: page count = %d, want 0 (not promoted)", n)
			}
			if _, err := s.Resolve(ctx, 0); err == nil {
				t.Fatalf("ENFORCE turn: Resolve(0) must error (page absent), got nil")
			}
		}
		// 2b. durable -> promoted, resolvable, byte-identical.
		{
			r := NewRecorder("enforce-durable").WithPromotion(PromotionEnforce)
			r.Record(ctx, "read_memory", []byte(durableFact))
			if got := r.RefusedPromotions(); got != 0 {
				t.Fatalf("ENFORCE durable: refused count = %d, want 0 (the earned exception)", got)
			}
			dir := t.TempDir()
			if err := r.Persist(dir); err != nil {
				t.Fatalf("persist: %v", err)
			}
			s, err := Load(dir)
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			if n := len(s.Pages()); n != 1 {
				t.Fatalf("ENFORCE durable: page count = %d, want 1 (promoted)", n)
			}
			if d := s.Pages()[0].Durability; d != ctxmmu.DurabilityDurable {
				t.Fatalf("ENFORCE durable: Page.Durability = %q, want %q", d, ctxmmu.DurabilityDurable)
			}
			got, err := s.Resolve(ctx, 0)
			if err != nil {
				t.Fatalf("ENFORCE durable: Resolve(0) must succeed, got err: %v", err)
			}
			if string(got) != durableFact {
				t.Fatalf("ENFORCE durable: resolved %q, want %q", got, durableFact)
			}
		}
	})

	// --- 3. The composite bite: in ONE session [turn, durable], only the durable fact
	// crosses the boundary, landing at step 0 with contiguous numbering over kept
	// pages. The transient 3pm fact is gone from the persisted core image. ---
	t.Run("composite_only_durable_survives", func(t *testing.T) {
		r := NewRecorder("enforce-composite").WithPromotion(PromotionEnforce)
		r.Record(ctx, "clock", []byte(turnFact))          // dropped
		r.Record(ctx, "read_memory", []byte(durableFact)) // kept
		if got := r.RefusedPromotions(); got != 1 {
			t.Fatalf("composite: refused count = %d, want 1", got)
		}
		dir := t.TempDir()
		if err := r.Persist(dir); err != nil {
			t.Fatalf("persist: %v", err)
		}
		s, err := Load(dir)
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		if n := len(s.Pages()); n != 1 {
			t.Fatalf("composite: page count = %d, want 1 (only durable survives)", n)
		}
		got, err := s.Resolve(ctx, 0)
		if err != nil {
			t.Fatalf("composite: Resolve(0) must return the durable fact, got err: %v", err)
		}
		if string(got) != durableFact {
			t.Fatalf("composite: step 0 = %q, want the durable fact %q", got, durableFact)
		}
	})

	// --- 4. A recognized-but-non-durable class (session) is ALSO refused under
	// enforce — proving the gate admits ONLY durable, not merely "not turn". This
	// subsumes the forward-compat default (unknown -> turn -> refused). ---
	t.Run("enforce_session_also_refused", func(t *testing.T) {
		r := NewRecorder("enforce-session").WithPromotion(PromotionEnforce)
		// "today's task ..." classifies session (a session-scoped frame), not durable.
		r.Record(ctx, "task", []byte("today's task is the durability gate"))
		dir := t.TempDir()
		if err := r.Persist(dir); err != nil {
			t.Fatalf("persist: %v", err)
		}
		s, err := Load(dir)
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		if n := len(s.Pages()); n != 0 {
			t.Fatalf("ENFORCE session: page count = %d, want 0 (session is non-durable, refused)", n)
		}
	})

	// --- 5. The quarantine path is orthogonal: a sealed page is ALWAYS recorded (the
	// seal is the audit record) and is NEVER dropped by the durability gate, even under
	// enforce — and it still refuses page-in without a witness clear. ---
	t.Run("enforce_does_not_drop_quarantined", func(t *testing.T) {
		r := NewRecorder("enforce-quarantine").WithPromotion(PromotionEnforce)
		r.Record(ctx, "read_file", []byte("api_key=sk-abcdef0123456789abcdef0123 leaked"))
		dir := t.TempDir()
		if err := r.Persist(dir); err != nil {
			t.Fatalf("persist: %v", err)
		}
		s, err := Load(dir)
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		pages := s.Pages()
		if len(pages) != 1 || !pages[0].Quarantined {
			t.Fatalf("ENFORCE quarantine: want 1 sealed page recorded, got %+v", pages)
		}
		if _, err := s.Resolve(ctx, 0); !errors.Is(err, ErrSealed) {
			t.Fatalf("ENFORCE quarantine: sealed page must still refuse page-in (ErrSealed), got %v", err)
		}
	})
}
