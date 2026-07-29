package main

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	conceptbench "github.com/anthony-chaudhary/fak/internal/conceptbench"
)

func spineLiveFixturePath() string { return filepath.Join("testdata", "spine", "spine-live.json") }

// bindStubGateway swaps the --spine run's Transport binder for one that binds a
// stub live gateway Transport, then restores it after the test — the seam #5311
// wires so the resolve->drive->grade pipeline runs with no key/GPU.
func bindStubGateway(t *testing.T, stub conceptbench.Transport) {
	t.Helper()
	prev := spineTransportBinder
	spineTransportBinder = func(reg *conceptbench.Registry) {
		reg.Bind(conceptbench.ArmGateway, stub, false)
	}
	t.Cleanup(func() { spineTransportBinder = prev })
}

// TestSpineLiveArmEmitsHeadlineRow is #5311's witness: a live-source arm bound to a
// stub Transport resolves through the #2731 registry, is graded by a real dos
// commit-audit, and the emitted fak.conceptbench.v1's COMPUTED honesty gate reports
// headline_rows >= 1 with result_claim_allowed:true — the exact row report.go's
// #868 gate was built to pass but had never been handed one.
func TestSpineLiveArmEmitsHeadlineRow(t *testing.T) {
	if _, err := exec.LookPath("dos"); err != nil {
		t.Skip("dos CLI not on PATH — the live-arm grade is a real dos commit-audit call")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH — the spine grades arms in a scratch git repo")
	}
	// The stub returns the commit subject each arm "produced": opus a stamped,
	// path-scoped subject (deterministic PASS over the seeded diff); haiku a
	// subject-only claim (deterministic FAIL over an empty diff). Both are live,
	// referee-witnessed rows.
	bindStubGateway(t, func(model, _ string) (string, string, error) {
		if model == "claude-opus-4-8" {
			return "fix(gateway): treat same-tick ready as positive (fak gateway)", "", nil
		}
		return "fix(gateway): resolve the ready race, all tests pass", "", nil
	})

	code, out := captureRun(t, []string{"--spine", spineLiveFixturePath()})
	if code != 0 {
		t.Fatalf("exit=%d, want 0\n%s", code, out)
	}

	var rep struct {
		Schema             string `json:"schema"`
		Mode               string `json:"mode"`
		ResultClaimAllowed bool   `json:"result_claim_allowed"`
		HonestyGate        struct {
			HeadlineRows    int `json:"headline_rows"`
			ReplayRows      int `json:"replay_rows"`
			UnwitnessedRows int `json:"unwitnessed_rows"`
		} `json:"honesty_gate"`
		Rows []struct {
			Model         string `json:"model"`
			Source        string `json:"source"`
			WitnessSource string `json:"witness_source"`
		} `json:"rows"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("parse report: %v\n%s", err, out)
	}
	if rep.Schema != "fak.conceptbench.v1" {
		t.Errorf("schema=%q, want fak.conceptbench.v1", rep.Schema)
	}
	if rep.Mode != "live" {
		t.Errorf("mode=%q, want live (a live arm ran)", rep.Mode)
	}
	if rep.HonestyGate.HeadlineRows < 1 {
		t.Fatalf("headline_rows=%d, want >=1 (a live referee-witnessed row):\n%s", rep.HonestyGate.HeadlineRows, out)
	}
	if !rep.ResultClaimAllowed {
		t.Errorf("result_claim_allowed=false; a live headline row with no unwitnessed rows must allow the claim:\n%s", out)
	}
	for _, r := range rep.Rows {
		if r.Source != "live" {
			t.Errorf("row %s source=%q, want live", r.Model, r.Source)
		}
		if r.WitnessSource != "dos_commit_audit" {
			t.Errorf("row %s witness_source=%q, want dos_commit_audit", r.Model, r.WitnessSource)
		}
	}
}

// TestSpineLiveArmGatedWithoutTransport is #5311's typed-refusal witness: a
// live-source arm with NO bound Transport must refuse arm_gated (exit 2, typed, not
// a crash) — live calls need a key/GPU. It does not require the dos CLI, because the
// registry refusal fires before the grader is consulted.
func TestSpineLiveArmGatedWithoutTransport(t *testing.T) {
	prev := spineTransportBinder
	spineTransportBinder = func(*conceptbench.Registry) {} // bind nothing → gateway arm is gated
	t.Cleanup(func() { spineTransportBinder = prev })

	code, _ := captureRun(t, []string{"--spine", spineLiveFixturePath()})
	if code != 2 {
		t.Fatalf("exit=%d, want 2 (arm_gated: a live arm with no bound Transport must refuse)", code)
	}
}

// TestSpineFixtureStampLeaf pins the leaf the #5380 hint template is built from,
// without needing the dos CLI: an explicit fixture "leaf" wins, else the seed paths
// name it, else the empty answer that makes the injection fail closed.
func TestSpineFixtureStampLeaf(t *testing.T) {
	shipped, err := loadSpineFixture(spineLiveFixturePath())
	if err != nil {
		t.Fatalf("load shipped fixture: %v", err)
	}
	if got := shipped.stampLeaf(); got != "gateway" {
		t.Errorf("shipped fixture stampLeaf()=%q, want %q (its seed is gateway/tick.go and its task text names the `(fak gateway)` trailer)", got, "gateway")
	}
	explicit := shipped
	explicit.Leaf = "conceptbench"
	if got := explicit.stampLeaf(); got != "conceptbench" {
		t.Errorf("explicit leaf ignored: stampLeaf()=%q, want conceptbench", got)
	}
	rootOnly := spineFixture{Seed: map[string]string{"README.md": "x"}}
	if got := rootOnly.stampLeaf(); got != "" {
		t.Errorf("stampLeaf()=%q for root-level seed, want \"\" so the hint fails closed", got)
	}
}

// TestSpineAffordanceHintReachesOnlyTheSmallArmsFrame is #5380's WIRING witness: it
// reads the prompt each arm's Transport actually received. The library contract test
// (internal/conceptbench/affordance_test.go) proves what the hint says and when the
// gate admits it; this one proves the spine asks the gate at all, asks it PER ARM,
// and leaves the frontier arm's frame byte-identical to the un-flagged run.
//
// The assertions deliberately do NOT key on "(fak gateway)": the fixture's own task
// text already names that trailer, so it cannot tell a hinted frame from a plain
// one. They key on the spelled-out template line and the offline check step, which
// exist only in the hint.
func TestSpineAffordanceHintReachesOnlyTheSmallArmsFrame(t *testing.T) {
	if _, err := exec.LookPath("dos"); err != nil {
		t.Skip("dos CLI not on PATH — the spine grade is a real dos commit-audit call")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH — the spine grades arms in a scratch git repo")
	}
	raw, err := os.ReadFile(spineLiveFixturePath())
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx struct {
		Task string `json:"task"`
	}
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}

	const (
		frontierArm = "claude-opus-4-8"  // registry tier: frontier — the control
		smallArm    = "claude-3-5-haiku" // registry tier: small — the treated arm
		// Two strings that live ONLY in the hint.
		templateLine = "type(scope): <verb> <what> (fak gateway)"
		checkStep    = "tools/check_commit_msg.py"
	)

	runArms := func(t *testing.T, argv []string) (map[string]string, string) {
		t.Helper()
		seen := map[string]string{}
		bindStubGateway(t, func(model, prompt string) (string, string, error) {
			seen[model] = prompt
			if model == frontierArm {
				return "fix(gateway): treat same-tick ready as positive (fak gateway)", "", nil
			}
			return "fix(gateway): resolve the ready race, all tests pass", "", nil
		})
		code, out := captureRun(t, argv)
		if code != 0 {
			t.Fatalf("exit=%d, want 0\n%s", code, out)
		}
		if len(seen) != 2 {
			t.Fatalf("drove %d arm(s), want 2: %v", len(seen), seen)
		}
		return seen, out
	}

	t.Run("opted_in", func(t *testing.T) {
		seen, out := runArms(t, []string{"--affordance-hint", "--spine", spineLiveFixturePath()})

		small := seen[smallArm]
		if !strings.Contains(small, templateLine) {
			t.Errorf("the small arm's frame is missing the trailer template %q:\n%s", templateLine, small)
		}
		if !strings.Contains(small, checkStep) {
			t.Errorf("the small arm's frame is missing the checkable step %q:\n%s", checkStep, small)
		}
		if !strings.Contains(small, "not yet") {
			t.Errorf("the small arm's frame is missing the report-`not yet` rule:\n%s", small)
		}
		if !strings.HasPrefix(small, strings.TrimRight(fx.Task, "\n")) {
			t.Errorf("the hint replaced the task text instead of extending it:\n%s", small)
		}

		// The control: byte-identical to the fixture's own task text.
		if got := seen[frontierArm]; got != fx.Task {
			t.Errorf("the frontier arm's frame was rewritten by --affordance-hint; the promotion contrast needs an untouched control\n got %q\nwant %q", got, fx.Task)
		}

		// The artifact must LABEL which row was treated, or a re-run row cannot be
		// read as a comparison.
		hinted := spineHintedModels(t, out)
		if !hinted[smallArm] {
			t.Errorf("row %s has affordance_hint=false; the treated arm must be labeled\n%s", smallArm, out)
		}
		if hinted[frontierArm] {
			t.Errorf("row %s has affordance_hint=true; the control must not be labeled treated\n%s", frontierArm, out)
		}
	})

	t.Run("opted_out", func(t *testing.T) {
		seen, out := runArms(t, []string{"--spine", spineLiveFixturePath()})
		for _, m := range []string{smallArm, frontierArm} {
			if got := seen[m]; got != fx.Task {
				t.Errorf("without --affordance-hint, arm %s's frame is not the fixture's task text\n got %q\nwant %q", m, got, fx.Task)
			}
		}
		if hinted := spineHintedModels(t, out); len(hinted) != 0 {
			t.Errorf("an un-opted-in run labeled %v as treated\n%s", hinted, out)
		}
	})
}

// spineHintedModels reads the emitted fak.conceptbench.v1 and returns the set of
// models whose row records the affordance hint as having been in its frame.
func spineHintedModels(t *testing.T, out string) map[string]bool {
	t.Helper()
	var rep struct {
		Rows []struct {
			Model          string `json:"model"`
			AffordanceHint bool   `json:"affordance_hint"`
		} `json:"rows"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("parse report: %v\n%s", err, out)
	}
	hinted := map[string]bool{}
	for _, r := range rep.Rows {
		if r.AffordanceHint {
			hinted[r.Model] = true
		}
	}
	return hinted
}
