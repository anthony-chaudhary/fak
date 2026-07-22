package main

// Tests for the #3568 negframe telemetry line and the FAK_ABLATE reframe lever.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ablate"
	"github.com/anthony-chaudhary/fak/internal/negframe"
)

// negframeNegativeProse pairs a MECHANICAL negative idiom ("do not forget to X", which the
// lexicon flips to "remember to X") with a JUDGEMENT-tier one ("never", detected but never
// auto-rewritten). The mechanical half makes the two arms distinguishable by bytes alone; the
// judgement half guarantees a non-zero residual on both arms, so the summary counts are real.
const negframeNegativeProse = "Do not forget to run the gate. Never bypass the guard."

// TestGuardNegframeSummaryLine pins the exit-summary line's contract: it reports the
// injected-directive negframe counts with the active arm labelled, and returns "" on a
// missing or broken journal — the best-effort behaviour guardToolprocSummaryLine keeps.
func TestGuardNegframeSummaryLine(t *testing.T) {
	t.Run("reports arm, residual and fallback counts", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "journal.jsonl")
		guardNegframeRecord(path, guardNegframeRow{
			Arm: guardNegframeArmOn, Applied: 3, Residual: 2, VerbatimFallback: 1,
		})
		got := guardNegframeSummaryLine(path)
		if got == "" {
			t.Fatal("summary line is empty for a well-formed journal")
		}
		for _, want := range []string{"negframe", "reframe on", "3", "2", "1"} {
			if !strings.Contains(got, want) {
				t.Fatalf("summary line missing %q:\n%s", want, got)
			}
		}
	})

	t.Run("labels the ablated control arm", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "journal.jsonl")
		guardNegframeRecord(path, guardNegframeRow{Arm: guardNegframeArmOff, Residual: 4})
		got := guardNegframeSummaryLine(path)
		if !strings.Contains(got, "reframe OFF") || !strings.Contains(got, ablate.FeatureNegframeReframe) {
			t.Fatalf("control arm is not labelled as ablated:\n%s", got)
		}
	})

	t.Run("sums rows across the turn", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "journal.jsonl")
		guardNegframeRecord(path, guardNegframeRow{Arm: guardNegframeArmOn, Applied: 2, Residual: 1})
		guardNegframeRecord(path, guardNegframeRow{Arm: guardNegframeArmOn, Applied: 5, Residual: 3})
		got := guardNegframeSummaryLine(path)
		if !strings.Contains(got, "7") || !strings.Contains(got, "4") {
			t.Fatalf("rows were not summed (want applied=7 residual=4):\n%s", got)
		}
	})

	// The degradation this line exists to surface: a gate that refuses every mechanical
	// candidate ships prose unreframed while the arm still reads "on". That must be a
	// VISIBLE spike, not a silent zero.
	t.Run("forced-all-refuse gate spikes visibly", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "journal.jsonl")
		guardNegframeRecord(path, guardNegframeRow{
			Arm: guardNegframeArmOn, Applied: 0, Residual: 9, VerbatimFallback: 9,
		})
		got := guardNegframeSummaryLine(path)
		if !strings.Contains(got, "⚠") {
			t.Fatalf("an all-refuse gate did not raise the visible warning:\n%s", got)
		}
	})

	// The line is PER-TURN, not lifetime-cumulative: SessionStart resets the stream, so a
	// previous session's rows never inflate this turn's counts.
	t.Run("session start resets the stream", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "journal.jsonl")
		guardNegframeRecord(path, guardNegframeRow{Arm: guardNegframeArmOn, Applied: 99, Residual: 99})
		guardNegframeBegin(path, guardNegframeRow{Arm: guardNegframeArmOn, Applied: 1, Residual: 0})
		got := guardNegframeSummaryLine(path)
		if strings.Contains(got, "99") {
			t.Fatalf("a prior session's counts leaked into this turn's line:\n%s", got)
		}
		if !strings.Contains(got, guardRow("reframes applied", "1")) {
			t.Fatalf("this session's own count is missing:\n%s", got)
		}
	})

	// GOLDEN: the exact end-of-turn bytes. The counts and the arm label are what #3546 reads
	// off a run, so a silent reformat of this line is a measurement change, not cosmetics.
	t.Run("golden render", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "journal.jsonl")
		guardNegframeRecord(path, guardNegframeRow{
			Arm: guardNegframeArmOn, Applied: 4, Residual: 2, VerbatimFallback: 1,
		})
		want := guardSection("injected-directive negframe") +
			guardRow("arm", "reframe on (treatment)") +
			guardRow("reframes applied", "4") +
			guardRow("residual negatives", "2") +
			guardRow("verbatim fallbacks", "1") +
			guardNote("the A/B lever for #3546: `FAK_ABLATE=negframe_reframe` runs the unreframed control arm; unset reframes")
		if got := guardNegframeSummaryLine(path); got != want {
			t.Fatalf("golden mismatch:\n got %q\nwant %q", got, want)
		}
	})

	t.Run("best-effort: missing journal returns empty", func(t *testing.T) {
		if got := guardNegframeSummaryLine(filepath.Join(t.TempDir(), "absent.jsonl")); got != "" {
			t.Fatalf("missing journal returned %q, want %q", got, "")
		}
	})

	t.Run("best-effort: broken journal returns empty", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "broken.jsonl")
		if err := os.WriteFile(path, []byte("{not json\nalso not json\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		if got := guardNegframeSummaryLine(path); got != "" {
			t.Fatalf("broken journal returned %q, want %q", got, "")
		}
	})

	t.Run("best-effort: empty journal returns empty", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "empty.jsonl")
		if err := os.WriteFile(path, nil, 0o644); err != nil {
			t.Fatal(err)
		}
		if got := guardNegframeSummaryLine(path); got != "" {
			t.Fatalf("empty journal returned %q, want %q", got, "")
		}
	})
}

// TestNegframeAblationFlag pins the lever's DIRECTION, which #3546's arms depend on:
// FAK_ABLATE=negframe_reframe routes injected prose UNREFRAMED (control), and an unset
// environment routes it REFRAMED (treatment, the default).
func TestNegframeAblationFlag(t *testing.T) {
	// Reframing must actually change these bytes, else the arms are indistinguishable and
	// every assertion below would pass vacuously.
	reframed := negframe.Reframe(negframeNegativeProse)
	if reframed == negframeNegativeProse {
		t.Fatalf("fixture prose is already a reframe fixed point; the arms cannot be told apart: %q", negframeNegativeProse)
	}

	t.Run("unset env reframes (treatment, default on)", func(t *testing.T) {
		t.Setenv("FAK_ABLATE", "")
		t.Setenv(guardNegframeEnvVar, "")
		if !negframeReframeEnabled() {
			t.Fatal("reframe is disabled with an unset env; the lever must default ON")
		}
		got, row := guardNegframeReframe(negframe.Fak(negframeNegativeProse))
		if got != reframed {
			t.Fatalf("treatment arm did not reframe:\n got %q\nwant %q", got, reframed)
		}
		if row.Arm != guardNegframeArmOn {
			t.Fatalf("row arm = %q, want %q", row.Arm, guardNegframeArmOn)
		}
		if row.Applied == 0 {
			t.Fatal("treatment arm reported 0 reframes applied over prose it demonstrably rewrote")
		}
	})

	t.Run("FAK_ABLATE token routes prose unreframed (control)", func(t *testing.T) {
		t.Setenv("FAK_ABLATE", ablate.FeatureNegframeReframe)
		t.Setenv(guardNegframeEnvVar, "")
		if negframeReframeEnabled() {
			t.Fatal("FAK_ABLATE=negframe_reframe did not disable the reframe pass")
		}
		got, row := guardNegframeReframe(negframe.Fak(negframeNegativeProse))
		if got != negframeNegativeProse {
			t.Fatalf("control arm rewrote the prose:\n got %q\nwant %q", got, negframeNegativeProse)
		}
		if row.Arm != guardNegframeArmOff {
			t.Fatalf("row arm = %q, want %q", row.Arm, guardNegframeArmOff)
		}
		if row.Applied != 0 {
			t.Fatalf("control arm applied %d reframes, want 0", row.Applied)
		}
	})

	// An `fak ablate --sweep` child carries the canonical per-feature env, not the coarse
	// list, so both spellings must drive the same lever.
	t.Run("sweep child env drives both arms", func(t *testing.T) {
		t.Setenv("FAK_ABLATE", "")
		t.Setenv(guardNegframeEnvVar, "0")
		if negframeReframeEnabled() {
			t.Fatalf("%s=0 did not select the control arm", guardNegframeEnvVar)
		}
		t.Setenv(guardNegframeEnvVar, "1")
		if !negframeReframeEnabled() {
			t.Fatalf("%s=1 did not select the treatment arm", guardNegframeEnvVar)
		}
	})

	// The lever switches fak's OWN voice only: an operator's or user's text is opaque and
	// must survive byte-for-byte on both arms.
	t.Run("opaque fragments survive on both arms", func(t *testing.T) {
		// Deliberately carries a MECHANICAL idiom: if provenance were ignored, the treatment
		// arm would rewrite this to "remember to touch" and the assertion would catch it.
		const opaque = "Do not forget to touch this operator text."
		for _, env := range []string{"", ablate.FeatureNegframeReframe} {
			t.Setenv("FAK_ABLATE", env)
			t.Setenv(guardNegframeEnvVar, "")
			got, _ := guardNegframeReframe(negframe.Opaque(opaque))
			if got != opaque {
				t.Fatalf("FAK_ABLATE=%q mangled opaque text:\n got %q\nwant %q", env, got, opaque)
			}
		}
	})
}

// TestNegframeAblationFeatureRegistered proves the lever is a first-class sweepable feature:
// present in the CLOSED KnownFeatures set, carded in the catalog, and routed through the
// rung-2 subprocess arm so `fak ablate --sweep negframe_reframe` re-execs a child that
// genuinely reads the env.
func TestNegframeAblationFeatureRegistered(t *testing.T) {
	var known bool
	for _, f := range ablate.KnownFeatures() {
		if f == ablate.FeatureNegframeReframe {
			known = true
			break
		}
	}
	if !known {
		t.Fatalf("%q missing from KnownFeatures: %v", ablate.FeatureNegframeReframe, ablate.KnownFeatures())
	}
	if !ablate.EnvGated(ablate.FeatureNegframeReframe) {
		t.Fatalf("EnvGated(%q) = false, want true (the sweep must take the subprocess rung)", ablate.FeatureNegframeReframe)
	}
	var carded bool
	for _, c := range ablate.FeatureCatalog() {
		if c.Token == ablate.FeatureNegframeReframe {
			carded = true
			if c.EnvVar != guardNegframeEnvVar {
				t.Fatalf("card EnvVar = %q, want %q (the runtime gate reads the card's env)", c.EnvVar, guardNegframeEnvVar)
			}
		}
	}
	if !carded {
		t.Fatalf("%q has no FeatureCard; `fak ablate --list` would omit the lever", ablate.FeatureNegframeReframe)
	}

	// `fak ablate --sweep negframe_reframe` must plan BOTH arms, else the A/B has nothing to
	// compare: the sweep is what carries the lever into an AblationReport.
	configs, err := ablate.BuildSweep([]string{ablate.FeatureNegframeReframe})
	if err != nil {
		t.Fatalf("BuildSweep(%q): %v", ablate.FeatureNegframeReframe, err)
	}
	arms := map[string]bool{}
	for _, c := range configs {
		arms[c.EnvFeatures[ablate.FeatureNegframeReframe]] = true
	}
	if !arms["on"] || !arms["off"] {
		t.Fatalf("sweep planned arms %v, want both an on and an off arm (configs=%+v)", arms, configs)
	}
}
