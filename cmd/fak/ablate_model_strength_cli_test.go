package main

// ablate_model_strength_cli_test.go — the cmd-level acceptance for `fak ablate --models`
// (#4412).
//
// internal/ablate already proves the CLASSIFIER (ParseModelTiers, ClassifyStrength,
// AnnotateModelStrength). What those tests cannot see is whether an OPERATOR can reach
// it: that the flag parses before anything expensive runs, that omitting it leaves the
// legacy table untouched, that the production stub renders as honestly on the terminal
// as it does in the struct, and that a measuring scorer's trajectory reaches both the
// table and --json. So every test here drives the real runAblate argv.

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ablate"
)

// fakeTierScorer is the measuring TierScorer the seam exists for: fixed per-(tier, arm)
// scores, so the whole axis is deterministic and $0. Anything it was not given reports
// UNMEASURED rather than a zero score — the same honesty bit production's stub carries.
type fakeTierScorer struct {
	scores map[string]map[string]float64 // tier -> armID -> score
}

func (f fakeTierScorer) ScoreArm(_ context.Context, tier, armID string, _ map[string]string) (ablate.TierOutcome, error) {
	score, ok := f.scores[tier][armID]
	if !ok {
		return ablate.TierOutcome{Detail: "fake scorer has no score for this rung"}, nil
	}
	return ablate.TierOutcome{Score: score, Measured: true}, nil
}

// withFakeTierScorer swaps the production stub for a measuring fake for one test — the
// same injection idiom as withFakeArmRunner.
func withFakeTierScorer(t *testing.T, scorer ablate.TierScorer) {
	t.Helper()
	orig := ablateTierScorer
	t.Cleanup(func() { ablateTierScorer = orig })
	ablateTierScorer = scorer
}

// A bad --models token is refused UP FRONT — before the trace is even opened. The
// nonexistent --trace is the discriminator: loading it would exit 1, so an exit 2 with
// no output proves the tier spec was validated before any load, replay, or sweep. A
// typo in the axis spec must cost a usage error, not a whole sweep.
func TestAblateModelsRefusesUnknownTierBeforeAnyWork(t *testing.T) {
	var out, errb bytes.Buffer
	code := runAblate(&out, &errb, []string{
		"--trace", "does-not-exist-on-purpose.json", "--sweep", "vdso", "--models", "galaxy",
	})
	if code != 2 {
		t.Fatalf("--models galaxy exit=%d, want 2 (usage refusal, not 1 = the trace-load failure that would follow)\nstdout=%s\nstderr=%s",
			code, out.String(), errb.String())
	}
	if got := errb.String(); !strings.Contains(got, "galaxy") {
		t.Fatalf("refusal does not name the offending token: %s", got)
	}
	// The message must name the ladder, so the operator can fix the flag from the error
	// alone rather than going to read the source.
	for _, tier := range ablate.KnownModelTiers() {
		if !strings.Contains(errb.String(), tier) {
			t.Fatalf("refusal omits known tier %q, so it does not say what IS accepted: %s", tier, errb.String())
		}
	}
	if out.Len() != 0 {
		t.Fatalf("a refused --models spec still produced output, so work ran before validation:\n%s", out.String())
	}
}

// Omitting --models keeps the legacy single-tier output: no axis block on the table, and
// no strength keys in the JSON. This is the compatibility pin — the axis is opt-in, and
// every existing `fak ablate` caller must be unaffected by its arrival.
func TestAblateWithoutModelsKeepsLegacyOutput(t *testing.T) {
	code, out, errb := runAB("--sweep", "vdso")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	for _, unwanted := range []string{"model-strength axis", "UNMEASURED", "unmeasured"} {
		if strings.Contains(out, unwanted) {
			t.Fatalf("legacy table (no --models) leaked %q:\n%s", unwanted, out)
		}
	}

	code, jsonOut, errb := runAB("--sweep", "vdso", "--json")
	if code != 0 {
		t.Fatalf("--json exit=%d stderr=%s", code, errb)
	}
	var rep struct {
		Runs []struct {
			ArmID         string           `json:"arm_id"`
			ModelStrength *json.RawMessage `json:"model_strength"`
		} `json:"runs"`
		Verdicts []json.RawMessage `json:"verdicts"`
	}
	if err := json.Unmarshal([]byte(jsonOut), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, jsonOut)
	}
	if len(rep.Verdicts) != 0 {
		t.Fatalf("no --models but the report carries %d verdict rows", len(rep.Verdicts))
	}
	for _, r := range rep.Runs {
		if r.ModelStrength != nil {
			t.Fatalf("no --models but arm %q carries a strength card: %s", r.ArmID, *r.ModelStrength)
		}
	}
}

// The production posture, rendered. `--models` under the default StubTierScorer must
// report UNMEASURED and print each rung as "unmeasured" — NEVER as a +0.0000 delta. A
// rung nobody scored reading as a rung that scored zero is the one failure this axis
// cannot afford, because it is indistinguishable on the page from a real measurement of
// no effect, and #4414 files deletion issues off these grades.
func TestAblateModelsUnderProductionStubRendersUnmeasured(t *testing.T) {
	code, out, errb := runAB("--sweep", "vdso", "--models", "weak,strong")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	for _, want := range []string{"model-strength axis", ablate.VerdictUnmeasured, "weak=unmeasured strong=unmeasured"} {
		if !strings.Contains(out, want) {
			t.Fatalf("stub-scored axis missing %q:\n%s", want, out)
		}
	}
	for _, banned := range []string{"weak=+", "weak=-", "strong=+", "strong=-"} {
		if strings.Contains(out, banned) {
			t.Fatalf("an unmeasured rung rendered as a numeric delta (%q), which reads as a real measurement:\n%s", banned, out)
		}
	}
	// The baseline arm is deliberately ungraded: its delta against itself is 0 by
	// construction, so a card on it would be a fabricated grade for the reference row.
	if strings.Count(out, ablate.VerdictUnmeasured) != 1 {
		t.Fatalf("want exactly one graded arm (the vdso arm; all-off is the baseline), got %d:\n%s",
			strings.Count(out, ablate.VerdictUnmeasured), out)
	}
}

// A measuring scorer end to end: a feature that HELPS the weak model and HURTS the
// strong one — the exact case the single-tier table averages into "fine". The table
// must show the trajectory turning over, the verdict must be HOBBLING (classified on
// the strongest rung, never the mean), and --json must carry both the per-arm card and
// the flat verdict row #4414's router reads.
func TestAblateModelsGradesTrajectoryFromMeasuringScorer(t *testing.T) {
	withFakeTierScorer(t, fakeTierScorer{scores: map[string]map[string]float64{
		ablate.TierWeak:   {"all-off": 0.50, "vdso": 0.80}, // +0.30: the scaffold carries a weak model
		ablate.TierStrong: {"all-off": 0.50, "vdso": 0.45}, // -0.05: the strong model is better without it
	}})

	code, out, errb := runAB("--sweep", "vdso", "--models", "strong,weak")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	// Typed strongest-first, rendered weakest-first: the ladder order is the
	// classifier's invariant, not the operator's typing order.
	if want := "weak=+0.3000 strong=-0.0500"; !strings.Contains(out, want) {
		t.Fatalf("trajectory missing %q:\n%s", want, out)
	}
	if !strings.Contains(out, ablate.VerdictHobbling) {
		t.Fatalf("a feature that costs the strongest model capability must grade %s:\n%s", ablate.VerdictHobbling, out)
	}

	code, jsonOut, errb := runAB("--sweep", "vdso", "--models", "strong,weak", "--json")
	if code != 0 {
		t.Fatalf("--json exit=%d stderr=%s", code, errb)
	}
	var rep struct {
		Runs []struct {
			ArmID         string                `json:"arm_id"`
			ModelStrength *ablate.ModelStrength `json:"model_strength"`
		} `json:"runs"`
		Verdicts []ablate.StrengthVerdict `json:"verdicts"`
	}
	if err := json.Unmarshal([]byte(jsonOut), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, jsonOut)
	}
	var card *ablate.ModelStrength
	for _, r := range rep.Runs {
		if r.ArmID == "all-off" && r.ModelStrength != nil {
			t.Fatalf("the baseline arm carries a strength card: %+v", r.ModelStrength)
		}
		if r.ArmID == "vdso" {
			card = r.ModelStrength
		}
	}
	if card == nil {
		t.Fatalf("the vdso arm carries no strength card in --json:\n%s", jsonOut)
	}
	if card.Verdict != ablate.VerdictHobbling || card.StrongestTier != ablate.TierStrong {
		t.Fatalf("card = %s at %q, want %s classified on %q", card.Verdict, card.StrongestTier, ablate.VerdictHobbling, ablate.TierStrong)
	}
	if len(card.Tiers) != 2 || !card.Tiers[0].Measured || card.Tiers[0].Tier != ablate.TierWeak {
		t.Fatalf("card trajectory = %+v, want two measured rungs weakest-first", card.Tiers)
	}
	if len(rep.Verdicts) != 1 || rep.Verdicts[0].ID != "vdso" || rep.Verdicts[0].Grade != ablate.VerdictHobbling {
		t.Fatalf("flat verdict rows = %+v, want one vdso/%s row for the #4414 router", rep.Verdicts, ablate.VerdictHobbling)
	}
}
