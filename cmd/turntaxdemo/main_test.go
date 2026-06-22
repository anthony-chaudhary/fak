package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestSelfcheckReproducesDocumentedInvariants pins the browser demo's own data
// path: runSelfcheck replays every shipped suite through the kernel (the same
// turnbench.RunWithCalls the browser drives) and must reproduce the documented
// turn-tax + safety-floor invariants (exit 0). This is the browserless guard the
// -selfcheck flag exposes to operators, now also gating CI and cross-platform
// (mac/arm64) runs so a regression fails here instead of in a demo nobody reran.
//
// turnTaxDir() resolves the fixtures via its "../../testdata/turntax" candidate
// when the test runs with CWD = cmd/turntaxdemo, so no working-dir juggling is
// needed.
func TestSelfcheckReproducesDocumentedInvariants(t *testing.T) {
	if code := runSelfcheck(); code != 0 {
		t.Fatalf("runSelfcheck() = %d, want 0 (a shipped turntax suite drifted from its documented invariants)", code)
	}
}

// TestHandleRun exercises the /api/run endpoint the side-by-side page drives all
// three columns from — the real HTTP + JSON path, not just the turnbench call
// runSelfcheck uses. The airline suite must return the documented turn-tax headline
// (9 saved = 5 forced + 4 elision) plus the safety subset (1 injection + 1
// destructive on the baseline, 0 with fak); the happy control inflates nothing. A
// handler or JSON-shape drift the rewritten page would hit now fails here.
func TestHandleRun(t *testing.T) {
	type runResp struct {
		Suite string `json:"suite"`
		Calls []struct {
			Tool string `json:"tool"`
		} `json:"calls"`
		Report struct {
			Net struct {
				TurnsSaved int `json:"turns_saved"`
			} `json:"net"`
			TurnKinds struct {
				Forced  int `json:"forced"`
				Elision int `json:"elision"`
			} `json:"turn_kinds"`
			Safety struct {
				InjBaseline   int `json:"injections_admitted_baseline"`
				DestrBaseline int `json:"destructive_executed_baseline"`
				InjFak        int `json:"injections_admitted_fak"`
				DestrFak      int `json:"destructive_executed_fak"`
			} `json:"safety_floor"`
			Consistency string `json:"consistency_check"`
		} `json:"report"`
	}
	run := func(suite string) runResp {
		t.Helper()
		rr := httptest.NewRecorder()
		handleRun(rr, httptest.NewRequest(http.MethodGet, "/api/run?suite="+suite, nil))
		if rr.Code != http.StatusOK {
			t.Fatalf("/api/run?suite=%s status = %d, want 200 (body: %s)", suite, rr.Code, rr.Body.String())
		}
		var resp runResp
		if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
			t.Fatalf("decode /api/run?suite=%s: %v", suite, err)
		}
		return resp
	}

	air := run("turntax-airline")
	if air.Report.Net.TurnsSaved != 9 {
		t.Errorf("airline turns_saved = %d, want 9", air.Report.Net.TurnsSaved)
	}
	if air.Report.TurnKinds.Forced != 5 || air.Report.TurnKinds.Elision != 4 {
		t.Errorf("airline turn_kinds = forced %d / elision %d, want 5 / 4", air.Report.TurnKinds.Forced, air.Report.TurnKinds.Elision)
	}
	if air.Report.Safety.InjBaseline != 1 || air.Report.Safety.DestrBaseline != 1 {
		t.Errorf("airline baseline safety = inj %d / destr %d, want 1 / 1", air.Report.Safety.InjBaseline, air.Report.Safety.DestrBaseline)
	}
	if air.Report.Safety.InjFak != 0 || air.Report.Safety.DestrFak != 0 {
		t.Errorf("airline fak safety = inj %d / destr %d, want 0 / 0", air.Report.Safety.InjFak, air.Report.Safety.DestrFak)
	}
	if air.Report.Consistency != "ok" {
		t.Errorf("airline consistency = %q, want \"ok\"", air.Report.Consistency)
	}
	if len(air.Calls) == 0 {
		t.Error("airline returned 0 calls (the page renders one row per call)")
	}

	if happy := run("turntax-happy"); happy.Report.Net.TurnsSaved != 0 {
		t.Errorf("happy turns_saved = %d, want 0 (the anti-inflation control)", happy.Report.Net.TurnsSaved)
	}
}
