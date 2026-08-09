package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestMeasuredArmsComeFromEnvToggle is the load-bearing #5851 test: it proves the A/B's two arms
// are PRODUCED BY #3568's env lever over a real run of the measured path, not read out of the
// hand-authored fixture corpus.
//
// Why a fixture-only corpus reds this test: every assertion below is anchored to a file on disk
// that only a real run can create. Each arm gets a FRESH t.TempDir() workspace, so the journal at
// <workspace>/.fak/negframe/journal.jsonl cannot pre-exist; the arm label the report carries is
// then required to round-trip out of that file's raw bytes. Swap runMeasuredAB back to reading
// fx.ArmA/fx.ArmB and there is no journal to open, so this fails at the read -- which is exactly
// the regression #5851 exists to prevent.
func TestMeasuredArmsComeFromEnvToggle(t *testing.T) {
	dir := t.TempDir()
	bin, source, err := resolveFakBinary(filepath.Join(dir, "build"))
	if err != nil {
		if mkErr := os.MkdirAll(filepath.Join(dir, "build"), 0o755); mkErr != nil {
			t.Fatalf("make build dir: %v", mkErr)
		}
		t.Fatalf("resolve fak binary for the measured path: %v", err)
	}
	t.Logf("measuring through %s (%s)", bin, source)

	work := filepath.Join(dir, "ws")
	measured, err := runMeasuredAB(bin, source, work)
	if err != nil {
		t.Fatalf("run the measured A/B: %v", err)
	}

	// 1. The env toggle, and only the env toggle, separates the arms.
	if measured.Control.Env != measuredAblateEnv+"="+measuredAblateToken {
		t.Errorf("control arm selected by %q, want %s=%s", measured.Control.Env, measuredAblateEnv, measuredAblateToken)
	}
	if !strings.Contains(measured.Treatment.Env, "unset") {
		t.Errorf("treatment arm selected by %q, want the lever unset", measured.Treatment.Env)
	}

	// 2. The arm labels are the ones #3568 writes, and they DIFFER -- a constant on both sides
	//    would mean the lever never took effect.
	if measured.Control.Arm != measuredArmOff {
		t.Errorf("control arm label = %q, want %q", measured.Control.Arm, measuredArmOff)
	}
	if measured.Treatment.Arm != measuredArmOn {
		t.Errorf("treatment arm label = %q, want %q", measured.Treatment.Arm, measuredArmOn)
	}
	if measured.Control.Arm == measured.Treatment.Arm {
		t.Fatalf("both arms report %q: the env toggle did not select distinct arms", measured.Control.Arm)
	}

	// 3. Each label round-trips out of the per-turn journal file the run wrote -- the DoD's
	//    "the arm label in the report comes from the run rather than from fixture construction".
	for _, arm := range []MeasuredArm{measured.Control, measured.Treatment} {
		want := filepath.Join(work, arm.Label, filepath.FromSlash(measuredJournalRel))
		if arm.JournalPath != want {
			t.Errorf("%s arm journal path = %q, want %q", arm.Label, arm.JournalPath, want)
		}
		raw, readErr := os.ReadFile(arm.JournalPath)
		if readErr != nil {
			t.Fatalf("%s arm: read the journal the run should have written: %v", arm.Label, readErr)
		}
		var row negframeJournalRow
		line := strings.TrimSpace(string(bytes.SplitN(bytes.TrimSpace(raw), []byte("\n"), 2)[0]))
		if unErr := json.Unmarshal([]byte(line), &row); unErr != nil {
			t.Fatalf("%s arm: journal row %q is not a negframe row: %v", arm.Label, line, unErr)
		}
		if row.Arm != arm.Arm {
			t.Errorf("%s arm: report says %q but the journal on disk says %q", arm.Label, arm.Arm, row.Arm)
		}
		if row.Applied != arm.Applied || row.Residual != arm.Residual || row.VerbatimFallback != arm.VerbatimFallback {
			t.Errorf("%s arm: report counts (%d/%d/%d) do not match the journal row (%d/%d/%d)",
				arm.Label, arm.Applied, arm.Residual, arm.VerbatimFallback,
				row.Applied, row.Residual, row.VerbatimFallback)
		}
	}

	// 4. No fixture text leaked in: the measured half must be sourced entirely from the run.
	for _, fx := range fixtures {
		if measured.Control.Arm == fx.ArmA || measured.Treatment.Arm == fx.ArmB {
			t.Fatalf("measured arm label matches fixture %q -- the measured half is fixture-sourced", fx.ID)
		}
	}
}

// TestMeasuredLoadSplit pins how a journal row is mapped onto the shared cost model: the control
// arm's residual is split into (mechanical, judgement) using the treatment arm's applied count,
// and the treatment arm scores everything left at the judgement tier. The clamp case matters --
// a degraded run must not manufacture a negative judgement count.
func TestMeasuredLoadSplit(t *testing.T) {
	cases := []struct {
		name             string
		row              negframeJournalRow
		treatmentApplied int
		wantMech         int
		wantJudge        int
	}{
		{
			name:             "control splits residual by the treatment's applied count",
			row:              negframeJournalRow{Arm: measuredArmOff, Residual: 5},
			treatmentApplied: 3,
			wantMech:         3,
			wantJudge:        2,
		},
		{
			name:             "treatment scores its survivors at the judgement tier",
			row:              negframeJournalRow{Arm: measuredArmOn, Applied: 3, Residual: 2},
			treatmentApplied: 3,
			wantMech:         0,
			wantJudge:        2,
		},
		{
			name:             "control clamps a mechanical count that exceeds its own residual",
			row:              negframeJournalRow{Arm: measuredArmOff, Residual: 1},
			treatmentApplied: 4,
			wantMech:         1,
			wantJudge:        0,
		},
		{
			name:             "a stream with no negation scores the ceiling on both arms",
			row:              negframeJournalRow{Arm: measuredArmOff},
			treatmentApplied: 0,
			wantMech:         0,
			wantJudge:        0,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			mech, judge := measuredLoad(c.row, c.treatmentApplied)
			if mech != c.wantMech || judge != c.wantJudge {
				t.Fatalf("measuredLoad(%+v, %d) = (%d,%d), want (%d,%d)",
					c.row, c.treatmentApplied, mech, judge, c.wantMech, c.wantJudge)
			}
			if judge < 0 {
				t.Fatalf("negative judgement load %d", judge)
			}
		})
	}
}

// TestPrintHumanSeparatesMeasuredFromModeled proves the rendered report cannot be read without
// seeing which numbers came from a run and which from the cost model -- the DoD's fourth box.
func TestPrintHumanSeparatesMeasuredFromModeled(t *testing.T) {
	r := runExperiment()
	r.Measured = &MeasuredAB{
		Schema:     "fak-negframe-steerability-ab-measured/1",
		Provenance: measuredProvenance,
		Binary:     "fak",
		Argv:       measuredArgv,
		Control:    MeasuredArm{Label: "control", Env: measuredAblateEnv + "=" + measuredAblateToken, Arm: measuredArmOff, Residual: 2, Compliance: 0.89},
		Treatment:  MeasuredArm{Label: "treatment", Env: "(" + measuredAblateEnv + " unset)", Arm: measuredArmOn, Applied: 1, Residual: 1, Compliance: 0.93},
	}
	var buf bytes.Buffer
	printHuman(&buf, r)
	out := buf.String()
	for _, want := range []string{
		"MEASURED half (#5851)",
		"MODELED half",
		"arm (MEASURED)",
		"compliance (MODELED)",
		measuredArmOff,
		measuredArmOn,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("human report missing %q:\n%s", want, out)
		}
	}
}

// TestPrintHumanStatesMeasuredUnavailable proves the honest degradation path: with no measured
// half the report says so in words rather than quietly presenting the modeled fixture delta as if
// it were the measured A/B.
func TestPrintHumanStatesMeasuredUnavailable(t *testing.T) {
	r := runExperiment()
	r.MeasuredUnavailable = "resolve fak binary: none found"
	var buf bytes.Buffer
	printHuman(&buf, r)
	out := buf.String()
	if !strings.Contains(out, "MEASURED half (#5851): UNAVAILABLE") {
		t.Errorf("report does not state the measured half is unavailable:\n%s", out)
	}
	if !strings.Contains(out, "resolve fak binary: none found") {
		t.Errorf("report does not carry the unavailability reason:\n%s", out)
	}
}

// TestSelfCheckMeasuredArms pins the selfcheck's measured-arm verdict: a correct pair PASSES, a
// pair whose labels did not come from the env toggle FAILS, and an absent measured half SKIPS
// without failing the offline spine.
func TestSelfCheckMeasuredArms(t *testing.T) {
	good := runExperiment()
	good.Measured = &MeasuredAB{
		Control:   MeasuredArm{Label: "control", Arm: measuredArmOff},
		Treatment: MeasuredArm{Label: "treatment", Arm: measuredArmOn},
	}
	var buf bytes.Buffer
	if ok := selfCheckReport(&buf, good); !ok {
		t.Fatalf("selfcheck failed on a well-formed measured pair:\n%s", buf.String())
	}
	if !strings.Contains(buf.String(), "PASS measured arms") {
		t.Errorf("selfcheck missing the measured PASS line:\n%s", buf.String())
	}

	bad := runExperiment()
	bad.Measured = &MeasuredAB{
		Control:   MeasuredArm{Label: "control", Arm: measuredArmOn},
		Treatment: MeasuredArm{Label: "treatment", Arm: measuredArmOn},
	}
	buf.Reset()
	if ok := selfCheckReport(&buf, bad); ok {
		t.Fatalf("selfcheck passed with both arms labelled %q:\n%s", measuredArmOn, buf.String())
	}

	buf.Reset()
	if ok := selfCheckReport(&buf, runExperiment()); !ok {
		t.Fatalf("selfcheck failed with no measured half; it should SKIP:\n%s", buf.String())
	}
	if !strings.Contains(buf.String(), "SKIP measured arms") {
		t.Errorf("selfcheck missing the measured SKIP line:\n%s", buf.String())
	}
}
