package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestSelfcheckReproducesSafetyInvariants pins the browser demo's own data path:
// runSelfcheck replays every shipped scenario through the kernel (the same
// turnbench.RunWithCalls the browser drives down both columns) and must reproduce
// the documented safety-floor invariants (exit 0) — WITHOUT fak breaches the
// expected count, WITH fak breaches zero, on every scenario. This is the browserless
// guard the -selfcheck flag exposes to operators, now also gating CI and
// cross-platform (mac/arm64) runs so a regression fails here, not in a demo nobody reran.
//
// turnTaxDir() resolves the fixtures via its "../../testdata/turntax" candidate when
// the test runs with CWD = cmd/guarddemo, so no working-dir juggling is needed.
func TestSelfcheckReproducesSafetyInvariants(t *testing.T) {
	if code := runSelfcheck(); code != 0 {
		t.Fatalf("runSelfcheck() = %d, want 0 (a shipped guarddemo scenario drifted from its documented safety-floor invariants)", code)
	}
}

// TestHandleScenarios exercises the /api/scenarios endpoint the page calls on load:
// it must list guard-redteam as present with its 7 calls. This guards the actual HTTP
// path (not just the turnbench call runSelfcheck uses), so a handler/JSON-shape drift
// the browser would hit fails here.
func TestHandleScenarios(t *testing.T) {
	rr := httptest.NewRecorder()
	handleScenarios(rr, httptest.NewRequest(http.MethodGet, "/api/scenarios", nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("/api/scenarios status = %d, want 200", rr.Code)
	}
	var resp struct {
		Scenarios []struct {
			ID      string `json:"id"`
			Present bool   `json:"present"`
			Calls   int    `json:"calls"`
		} `json:"scenarios"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode /api/scenarios: %v", err)
	}
	var found bool
	for _, s := range resp.Scenarios {
		if s.ID == "guard-redteam" {
			found = true
			if !s.Present {
				t.Errorf("guard-redteam present = false, want true (fixture missing under %s?)", turnTaxDir())
			}
			if s.Calls != 7 {
				t.Errorf("guard-redteam calls = %d, want 7", s.Calls)
			}
		}
	}
	if !found {
		t.Fatalf("/api/scenarios did not list guard-redteam (got %d scenarios)", len(resp.Scenarios))
	}
}

// TestHandleRun exercises the /api/run endpoint the page drives both columns from. It
// must return, for the red-team scenario, the exact safety-floor headline the page
// renders (4 breaches WITHOUT fak = 1 injection + 3 destructive, 0 WITH fak) and the
// clean control (0 breaches) — through the real HTTP + JSON path the browser uses.
func TestHandleRun(t *testing.T) {
	type runResp struct {
		Scenario string `json:"scenario"`
		Breaches int    `json:"breaches"`
		Calls    []struct {
			Tool string `json:"tool"`
			Axis string `json:"axis"`
		} `json:"calls"`
		Report struct {
			Safety struct {
				InjBaseline   int `json:"injections_admitted_baseline"`
				InjFak        int `json:"injections_admitted_fak"`
				DestrBaseline int `json:"destructive_executed_baseline"`
				DestrFak      int `json:"destructive_executed_fak"`
			} `json:"safety_floor"`
			Consistency string `json:"consistency_check"`
		} `json:"report"`
	}
	run := func(scenario string) runResp {
		t.Helper()
		rr := httptest.NewRecorder()
		handleRun(rr, httptest.NewRequest(http.MethodGet, "/api/run?scenario="+scenario, nil))
		if rr.Code != http.StatusOK {
			t.Fatalf("/api/run?scenario=%s status = %d, want 200 (body: %s)", scenario, rr.Code, rr.Body.String())
		}
		var resp runResp
		if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
			t.Fatalf("decode /api/run?scenario=%s: %v", scenario, err)
		}
		return resp
	}

	red := run("guard-redteam")
	if red.Breaches != 4 {
		t.Errorf("guard-redteam breaches = %d, want 4", red.Breaches)
	}
	if len(red.Calls) != 7 {
		t.Errorf("guard-redteam calls = %d, want 7", len(red.Calls))
	}
	if red.Report.Safety.InjBaseline != 1 || red.Report.Safety.DestrBaseline != 3 {
		t.Errorf("guard-redteam baseline = inj %d / destr %d, want 1 / 3", red.Report.Safety.InjBaseline, red.Report.Safety.DestrBaseline)
	}
	if red.Report.Safety.InjFak != 0 || red.Report.Safety.DestrFak != 0 {
		t.Errorf("guard-redteam fak = inj %d / destr %d, want 0 / 0 (fak must NEVER breach)", red.Report.Safety.InjFak, red.Report.Safety.DestrFak)
	}
	if red.Report.Consistency != "ok" {
		t.Errorf("guard-redteam consistency = %q, want \"ok\"", red.Report.Consistency)
	}

	if happy := run("turntax-happy"); happy.Breaches != 0 {
		t.Errorf("turntax-happy breaches = %d, want 0 (the anti-fear-mongering control)", happy.Breaches)
	}
}
