package rsiloop

import (
	"path/filepath"
	"testing"
)

// TestDetectSelfFulfillingSkill is the #2842 acceptance witness: a skill with a
// high use_count but zero/negative value net of its own invocations is flagged as
// self-fulfilling, while a skill with real counterfactual value (and one never
// invoked) is not. The counterfactual value is computed by subtracting the
// skill's own invocations, so a skill whose "value" is only being invoked
// collapses to <= 0 and fails the detector.
func TestDetectSelfFulfillingSkill(t *testing.T) {
	// The reflexive skill: invoked 40 times, but its entire GrossValue is the
	// per-invocation credit (40 * 1.0), so net of its own invocations it delivered
	// nothing. use_count looks great; counterfactual value collapses to 0.
	reflexive := SkillValue{Skill: "reflexive", UseCount: 40, GrossValue: 40, ValuePerInvocation: 1}
	got := DetectSelfFulfillingSkill(reflexive)
	if !got.Flagged || got.Verdict != SkillSelfFulfilling {
		t.Fatalf("reflexive skill: verdict=%q flagged=%v, want %q/flagged", got.Verdict, got.Flagged, SkillSelfFulfilling)
	}
	if got.CounterfactualValue != 0 {
		t.Fatalf("reflexive counterfactual value = %v, want 0 (value net of its own invocations)", got.CounterfactualValue)
	}

	// A skill that actively hurts once you net out its invocations (negative
	// counterfactual value) is likewise flagged — being invoked is not value.
	harmful := SkillValue{Skill: "harmful", UseCount: 10, GrossValue: 4, ValuePerInvocation: 1}
	if h := DetectSelfFulfillingSkill(harmful); !h.Flagged || h.CounterfactualValue >= 0 {
		t.Fatalf("harmful skill: flagged=%v counterfactual=%v, want flagged with negative counterfactual", h.Flagged, h.CounterfactualValue)
	}

	// The witnessed skill: invoked 10 times, but its value (50) far exceeds the
	// per-invocation credit (10), so it holds real value net of its own
	// invocations. It must NOT be flagged — the detector refuses to reward raising
	// use_count alone, not delivering value.
	witnessed := SkillValue{Skill: "witnessed", UseCount: 10, GrossValue: 50, ValuePerInvocation: 1}
	if w := DetectSelfFulfillingSkill(witnessed); w.Flagged || w.Verdict != SkillWitnessed {
		t.Fatalf("witnessed skill: verdict=%q flagged=%v, want %q/not-flagged", w.Verdict, w.Flagged, SkillWitnessed)
	}

	// A skill that was never invoked raised no metric, so there is nothing
	// self-fulfilling to check — it is not flagged.
	if u := DetectSelfFulfillingSkill(SkillValue{Skill: "unused"}); u.Flagged || u.Verdict != SkillUnused {
		t.Fatalf("unused skill: verdict=%q flagged=%v, want %q/not-flagged", u.Verdict, u.Flagged, SkillUnused)
	}
}

// TestSelfFulfillingSkillRoutesToRevert is the wire into the keep gate (#2842 ->
// #2841): a flagged skill's verdict yields a structured curator reason that the
// per-decision revert ledger accepts and answers, while an unflagged verdict
// yields no reason so a kept skill never routes to revert. The self_fulfilling
// reason is distinct from slop_scored (the #2841 confusion-risk rule) and
// survives a disk round-trip.
func TestSelfFulfillingSkillRoutesToRevert(t *testing.T) {
	path := filepath.Join(t.TempDir(), "curator.jsonl")
	l, err := OpenCuratorLedger(path)
	if err != nil {
		t.Fatalf("open ledger: %v", err)
	}

	// A kept skill yields no reason — it must not be routable to revert.
	kept := DetectSelfFulfillingSkill(SkillValue{Skill: "kept", UseCount: 5, GrossValue: 30, ValuePerInvocation: 1})
	if _, ok := kept.SelfFulfillingReason(); ok {
		t.Fatalf("a witnessed skill produced a revert reason; it must not route to revert")
	}

	// A flagged skill yields a structured reason that the keep gate archives.
	flagged := DetectSelfFulfillingSkill(SkillValue{Skill: "vanity", UseCount: 40, GrossValue: 40, ValuePerInvocation: 1})
	reason, ok := flagged.SelfFulfillingReason()
	if !ok {
		t.Fatalf("a flagged self-fulfilling skill produced no revert reason")
	}
	if !reason.Valid() {
		t.Fatalf("self-fulfilling reason is not Valid(): %+v", reason)
	}
	seq, err := l.Archive("vanity", reason)
	if err != nil {
		t.Fatalf("archive flagged skill: %v", err)
	}

	// The read-path answers "why is this skill gone?" with the structured token,
	// carrying the gameable use_count and the collapsed counterfactual value.
	r, gone := l.Why("vanity")
	if !gone || r.Kind != ReasonSelfFulfilling || r.UseCount != 40 {
		t.Fatalf("vanity reason = %+v gone=%v, want self_fulfilling/use_count=40", r, gone)
	}

	// The self_fulfilling token must not collapse into slop_scored — they are
	// genuinely distinct reasons an operator can act on differently.
	if _, err := l.Archive("sloppy", CuratorReason{Kind: ReasonSlopScored, SlopScore: 0.8}); err != nil {
		t.Fatalf("archive sloppy: %v", err)
	}
	if sf, _ := l.Why("vanity"); sf.String() == mustWhy(t, l, "sloppy") {
		t.Fatalf("self_fulfilling and slop_scored rendered identically (%q) — reasons collapsed", sf.String())
	}

	// Per-decision revert restores exactly the flagged skill (the #2841 guarantee):
	// undo the self-fulfilling archive and "vanity" returns, sibling "sloppy" stays.
	if err := l.Revert(seq); err != nil {
		t.Fatalf("revert vanity: %v", err)
	}
	if _, gone := l.Why("vanity"); gone {
		t.Fatalf("vanity should be restored after per-decision revert")
	}
	if _, gone := l.Why("sloppy"); !gone {
		t.Fatalf("sibling sloppy was rolled back by vanity's revert")
	}

	// "From the journal alone": reopen from disk and the self_fulfilling reason,
	// with its use_count and counterfactual value, reconstructs intact.
	reopened, err := OpenCuratorLedger(path)
	if err != nil {
		t.Fatalf("reopen ledger: %v", err)
	}
	if rr, gone := reopened.Why("sloppy"); !gone || rr.Kind != ReasonSlopScored {
		t.Fatalf("reopened sloppy reason = %+v gone=%v, want slop_scored", rr, gone)
	}
}
