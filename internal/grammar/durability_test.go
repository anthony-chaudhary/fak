package grammar

import (
	"reflect"
	"testing"
)

// TestDurabilityVocabularyIsClosed pins the four recognized classes as the closed set
// (acceptance #1, dos_check_reason style): each member is KnownDurability=true, and a
// 5th value (and the empty/unclassified value) is rejected, not tolerated.
func TestDurabilityVocabularyIsClosed(t *testing.T) {
	for _, class := range []DurabilityClass{
		DurabilityTurn, DurabilitySession, DurabilityDurable, DurabilityBounded,
	} {
		if !KnownDurability(class) {
			t.Fatalf("KnownDurability(%q) = false, want true for a vocabulary member", class)
		}
	}

	// A 5th class is NOT in the closed set: rejected, not silently tolerated.
	if KnownDurability(DurabilityClass("eternal")) {
		t.Fatalf("KnownDurability(eternal) = true; a 5th class must be rejected (closed set)")
	}
	// The empty / unclassified value is also unknown.
	if KnownDurability(DurabilityClass("")) {
		t.Fatalf("KnownDurability(\"\") = true; an unclassified write must be unknown")
	}

	// DurabilityVocabulary enumerates exactly the four, and the returned slice is a
	// copy a caller cannot use to mutate the closed set.
	vocab := DurabilityVocabulary()
	if len(vocab) != 4 {
		t.Fatalf("DurabilityVocabulary len = %d, want 4", len(vocab))
	}
	seen := map[DurabilityClass]bool{}
	for _, c := range vocab {
		seen[c] = true
	}
	for _, want := range []DurabilityClass{DurabilityTurn, DurabilitySession, DurabilityDurable, DurabilityBounded} {
		if !seen[want] {
			t.Fatalf("DurabilityVocabulary missing %q", want)
		}
	}
	vocab[0] = DurabilityClass("mutated")
	if DurabilityVocabulary()[0] == DurabilityClass("mutated") {
		t.Fatalf("DurabilityVocabulary returned an aliased slice; closed set is mutable")
	}
}

// TestMayPromoteFailClosed is the core promotion-predicate contract (acceptance #2):
// only a stamped `durable` class promotes; an unclassified or non-`durable` write does
// NOT promote, with the right closed-set Reason token.
func TestMayPromoteFailClosed(t *testing.T) {
	tests := []struct {
		name        string
		class       DurabilityClass
		wantPromote bool
		wantKnown   bool
		wantReason  string
	}{
		{"durable promotes", DurabilityDurable, true, true, ""},
		{"turn does not promote", DurabilityTurn, false, true, PromoteReasonNonDurable},
		{"session does not promote", DurabilitySession, false, true, PromoteReasonNonDurable},
		{"bounded does not promote by default", DurabilityBounded, false, true, PromoteReasonNonDurable},
		{"unclassified (empty) fails closed", DurabilityClass(""), false, false, PromoteReasonUnknownClass},
		{"unknown 5th class fails closed", DurabilityClass("eternal"), false, false, PromoteReasonUnknownClass},
	}
	// The verdict is mode-independent in WHICH classes promote: enforce posture here.
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			v := MayPromote(tc.class, PromotionEnforce)
			if v.Promote != tc.wantPromote {
				t.Fatalf("MayPromote(%q).Promote = %v, want %v", tc.class, v.Promote, tc.wantPromote)
			}
			if v.Known != tc.wantKnown {
				t.Fatalf("MayPromote(%q).Known = %v, want %v", tc.class, v.Known, tc.wantKnown)
			}
			if v.Reason != tc.wantReason {
				t.Fatalf("MayPromote(%q).Reason = %q, want %q", tc.class, v.Reason, tc.wantReason)
			}
			if v.Class != tc.class {
				t.Fatalf("MayPromote(%q).Class = %q, want it echoed verbatim", tc.class, v.Class)
			}
		})
	}
}

// TestMayPromoteIsModeIndependentForPromotion confirms the SAME write promotes (or
// not) regardless of mode — the mode changes only whether a non-promotion BITES, not
// which classes are promotable. Evidence-bound: the class is the only input that
// moves Promote.
func TestMayPromoteModeIndependentDecision(t *testing.T) {
	for _, mode := range []PromotionMode{PromotionWarn, PromotionEnforce} {
		if !MayPromote(DurabilityDurable, mode).Promote {
			t.Fatalf("durable did not promote in mode %v", mode)
		}
		if MayPromote(DurabilityTurn, mode).Promote {
			t.Fatalf("turn promoted in mode %v (should never promote)", mode)
		}
		if MayPromote(DurabilityClass(""), mode).Promote {
			t.Fatalf("unclassified promoted in mode %v (should never promote)", mode)
		}
	}
}

// TestShouldPersistWarnVsEnforce proves the audit-only -> enforce shape: in warn mode
// EVERY write persists (non-behavior-changing) while the would-refusal is still
// recorded; in enforce mode only a promoting (`durable`) write persists.
func TestShouldPersistWarnVsEnforce(t *testing.T) {
	classes := []DurabilityClass{
		DurabilityDurable, DurabilitySession, DurabilityTurn, DurabilityBounded,
		DurabilityClass(""), DurabilityClass("eternal"),
	}

	// WARN: every write persists, regardless of class — but the would-refusal bit is
	// still set on the non-durable ones.
	for _, c := range classes {
		v := MayPromote(c, PromotionWarn)
		if !v.ShouldPersist() {
			t.Fatalf("warn mode: ShouldPersist=false for class %q (warn is audit-only)", c)
		}
		wantWouldRefuse := c != DurabilityDurable
		if v.WouldRefuse() != wantWouldRefuse {
			t.Fatalf("warn mode: WouldRefuse(%q) = %v, want %v", c, v.WouldRefuse(), wantWouldRefuse)
		}
	}

	// ENFORCE: only the `durable` write persists; everything else is blocked.
	for _, c := range classes {
		v := MayPromote(c, PromotionEnforce)
		wantPersist := c == DurabilityDurable
		if v.ShouldPersist() != wantPersist {
			t.Fatalf("enforce mode: ShouldPersist(%q) = %v, want %v", c, v.ShouldPersist(), wantPersist)
		}
	}
}

// TestWouldRefuseCountsNonPromotions confirms WouldRefuse is the auditable would-refusal
// signal: it is true on exactly the non-promoting verdicts, the count a warn-mode caller
// accrues before flipping to enforce.
func TestWouldRefuseCountsNonPromotions(t *testing.T) {
	refused := 0
	for _, c := range []DurabilityClass{
		DurabilityDurable, DurabilitySession, DurabilityTurn, DurabilityBounded, DurabilityClass(""),
	} {
		if MayPromote(c, PromotionWarn).WouldRefuse() {
			refused++
		}
	}
	// durable is the only one that does NOT count as a would-refusal.
	if refused != 4 {
		t.Fatalf("would-refusals = %d, want 4 (every class but durable)", refused)
	}
}

// TestPromotionModeString pins the diagnostic rendering of the two postures.
func TestPromotionModeString(t *testing.T) {
	if got := PromotionWarn.String(); got != "warn" {
		t.Fatalf("PromotionWarn.String() = %q, want warn", got)
	}
	if got := PromotionEnforce.String(); got != "enforce" {
		t.Fatalf("PromotionEnforce.String() = %q, want enforce", got)
	}
}

// TestVerdictIsEvidenceBound is the acceptance #3 guard: MayPromote echoes the stamped
// class verbatim and never upgrades an unknown stamp by inspecting anything else — the
// only input that can yield Promote=true is the literal `durable` stamp.
func TestVerdictIsEvidenceBound(t *testing.T) {
	// A value that merely CONTAINS "durable" as a substring is still a distinct,
	// unknown class — the predicate matches the closed vocabulary exactly, it does not
	// infer durability from the text of the class.
	v := MayPromote(DurabilityClass("durable-ish"), PromotionEnforce)
	if v.Known || v.Promote {
		t.Fatalf("a near-miss class %q was treated as durable; predicate must match exactly", v.Class)
	}
	if v.Reason != PromoteReasonUnknownClass {
		t.Fatalf("near-miss Reason = %q, want unknown_class", v.Reason)
	}

	// The echoed Class is exactly what was stamped (self-describing for audit).
	if v.Class != DurabilityClass("durable-ish") {
		t.Fatalf("verdict Class = %q, want the stamped value echoed", v.Class)
	}

	// Sanity: the full verdict for an unclassified write is the closed fail-closed shape.
	want := PromotionVerdict{
		Class: "", Known: false, Promote: false, Mode: PromotionEnforce, Reason: PromoteReasonUnknownClass,
	}
	if got := MayPromote(DurabilityClass(""), PromotionEnforce); !reflect.DeepEqual(got, want) {
		t.Fatalf("unclassified verdict = %+v, want %+v", got, want)
	}
}
