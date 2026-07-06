package memvaluescore

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// selffulfilling_test.go — #2914 regression: the learning-surface
// self-fulfilling-metric detector must FLAG a deliberately gaming skill (a note
// kept only to raise "skills kept", with no witnessed value) and CLEAR an
// honest witnessed learning update. Both windows drive the REAL card
// (BuildWith) so the detector reads the actual store_notes / frontier axes, not
// a synthetic pair.

// gamingSkill is the reward-hack a witnessed learning loop invites: a note added
// ONLY to raise the store's "skills kept" count — the Hermes "be ACTIVE, produce
// a skill" nudge — that delivers no witnessed recall value.
func gamingSkill(i int) string {
	return fmt.Sprintf(`---
name: gaming-skill-%d
description: A skill kept only to raise the skills-kept count; it delivers no witnessed value.
metadata:
  type: reference
---

Prose only. Nothing checkable, nothing witnessed — pure activity to game the loop.
`, i)
}

func TestDetector_FlagsGamingSkill_ClearsWitnessedLearning(t *testing.T) {
	ctx := context.Background()

	// Before: a small clean store, no ledger yet (frontier fails low = 0).
	beforeStore := writeStore(t, cleanIndex, cleanFiles())
	before := MetricFromPayload(BuildWith(ctx, beforeStore,
		filepath.Join(beforeStore, "no-ledger.jsonl"), staleByValue("@@none@@")))

	// Gaming window: add five "skill" notes (raise skills_kept) but append NO
	// witnessed value to the ledger (the frontier stays flat).
	gamingFiles := cleanFiles()
	gamingIndex := cleanIndex
	for i := 0; i < 5; i++ {
		name := fmt.Sprintf("gaming-skill-%d.md", i)
		gamingFiles[name] = gamingSkill(i)
		gamingIndex += fmt.Sprintf("- [Gaming %d](%s) — hook\n", i, name)
	}
	gamingStore := writeStore(t, gamingIndex, gamingFiles)
	gaming := MetricFromPayload(BuildWith(ctx, gamingStore,
		filepath.Join(gamingStore, "no-ledger.jsonl"), staleByValue("@@none@@")))

	// The gaming window raised the gameable metric...
	if gaming.SkillsKept <= before.SkillsKept {
		t.Fatalf("gaming skills did not raise skills_kept: before=%d after=%d", before.SkillsKept, gaming.SkillsKept)
	}
	// ...without moving the net-value witness...
	if gaming.NetValue != before.NetValue {
		t.Fatalf("gaming must not move the net-value frontier: before=%d after=%d", before.NetValue, gaming.NetValue)
	}
	// ...so the detector MUST flag the self-fulfilling raise.
	got := DetectSelfFulfilling(before, gaming)
	if !got.Flagged || got.Verdict != VerdictSelfFulfilling {
		t.Fatalf("gaming skill not flagged as self-fulfilling: %+v", got)
	}

	// Honest window: add ONE skill AND append witnessed recall value to the
	// ledger (a fresh_rendered plus a stale_withheld the recall gate refused).
	honestFiles := cleanFiles()
	honestFiles["real-skill.md"] = `---
name: real-skill
description: A skill kept because it delivered witnessed recall value.
metadata:
  type: project
---

Points at internal/memvaluescore/score.go — a checkable, witnessed artifact.
`
	honestIndex := cleanIndex + "- [Real](real-skill.md) — hook\n"
	honestStore := writeStore(t, honestIndex, honestFiles)
	honestLedger := filepath.Join(honestStore, "memory-value.jsonl")
	if err := os.WriteFile(honestLedger,
		[]byte(`{"schema":"fak-memory-value-ledger/1","fresh":3,"withheld_stale":1}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	honest := MetricFromPayload(BuildWith(ctx, honestStore, honestLedger, staleByValue("@@none@@")))

	if honest.SkillsKept <= before.SkillsKept {
		t.Fatalf("honest update did not raise skills_kept: before=%d after=%d", before.SkillsKept, honest.SkillsKept)
	}
	if honest.NetValue <= before.NetValue {
		t.Fatalf("honest update must raise the net-value frontier: before=%d after=%d", before.NetValue, honest.NetValue)
	}
	if hv := DetectSelfFulfilling(before, honest); hv.Flagged || hv.Verdict != VerdictWitnessed {
		t.Fatalf("honest witnessed learning must NOT be flagged: %+v", hv)
	}
}

// TestDetector_Verdicts pins the three-way verdict directly on the metric pair:
// a raise without value is flagged, a raise with value is witnessed, and no
// raise (even when value happens to fall) is never flagged — the detector
// refuses to reward raising the metric, not delivering value.
func TestDetector_Verdicts(t *testing.T) {
	cases := []struct {
		name          string
		before, after LearningMetric
		wantFlagged   bool
		wantVerdict   string
	}{
		{"gaming: kept up, value flat", LearningMetric{2, 20}, LearningMetric{7, 20}, true, VerdictSelfFulfilling},
		{"gaming: kept up, value down", LearningMetric{2, 20}, LearningMetric{3, 12}, true, VerdictSelfFulfilling},
		{"witnessed: kept up, value up", LearningMetric{2, 20}, LearningMetric{3, 28}, false, VerdictWitnessed},
		{"no raise: kept flat", LearningMetric{2, 20}, LearningMetric{2, 20}, false, VerdictNoRaise},
		{"no raise: value up, kept flat", LearningMetric{2, 20}, LearningMetric{2, 40}, false, VerdictNoRaise},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := DetectSelfFulfilling(c.before, c.after)
			if got.Flagged != c.wantFlagged || got.Verdict != c.wantVerdict {
				t.Fatalf("Detect(%+v,%+v) = flagged %v/%q, want %v/%q",
					c.before, c.after, got.Flagged, got.Verdict, c.wantFlagged, c.wantVerdict)
			}
		})
	}
}
