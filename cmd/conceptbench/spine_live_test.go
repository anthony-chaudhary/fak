package main

import (
	"encoding/json"
	"os/exec"
	"path/filepath"
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
