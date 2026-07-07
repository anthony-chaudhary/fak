package focusscore

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// buildFromRows folds a hand-built row set through the same path Build uses, bypassing
// disk. WIPCap defaults to DefaultWIPCap unless the test pins one.
func buildFromRows(rows []trajctl.Row, wipCap int) ScorecardPayload {
	return Build(Options{Root: ".", WIPCap: wipCap, rows: rows, useInputs: true})
}

// obj is a terse Objective row builder.
func obj(id, parent string, status trajctl.ObjectiveStatus, budgetTurns int) trajctl.Row {
	o := trajctl.Objective{ID: id, ParentID: parent, Statement: id + " statement", Status: status}
	if budgetTurns > 0 {
		o.Budget = trajctl.Budget{Turns: budgetTurns}
	}
	return trajctl.ObjectiveRecord(o)
}

// commit is a witnessed-commit-progress score row (the W3 progress curve DRIFT/STALL read).
func commit(objID string, value float64, unixMillis int64) trajctl.Row {
	return trajctl.ScoreRecord(trajctl.ScoreRow{
		ObjectiveID: objID,
		Value:       value,
		Method:      trajctl.CommitScorerMethod,
		Version:     "test",
		Witness:     trajctl.W3,
		UnixMillis:  unixMillis,
	})
}

// divergence is an activity-progress-divergence score row (the W2 signal STALL needs).
func divergence(objID string, value float64, unixMillis int64) trajctl.Row {
	return trajctl.ScoreRecord(trajctl.ScoreRow{
		ObjectiveID: objID,
		Value:       value,
		Method:      trajctl.ActivityDivergenceScorerMethod,
		Version:     "test",
		Witness:     trajctl.W2,
		UnixMillis:  unixMillis,
	})
}

func TestFold(t *testing.T) {
	cases := []struct {
		name     string
		rows     []trajctl.Row
		wipCap   int
		wantDebt int
		wantOK   bool
		// per-evidence assertions (checked only when non-negative)
		active, drift, stall, overrun, excessWIP, healthy int
	}{
		{
			// A fresh tree with no objectives: vacuously converged, no slander. The
			// ledger_present HARD KPI still passes vacuously (LedgerPresent false yet no
			// debt, because debt is magnitude-counted breadth/non-convergence, and there
			// is none). ledger_present failing does NOT add to focus_debt by design —
			// focus_debt counts fan-out, not the absence of a ledger.
			name: "empty ledger is clean", rows: nil,
			wantDebt: 0, wantOK: true,
			active: 0, drift: 0, stall: 0, overrun: 0, excessWIP: 0, healthy: 0,
		},
		{
			// Two active objectives, both with a RISING witnessed curve: converging, under
			// the cap. Zero debt.
			name: "converged fleet is clean",
			rows: []trajctl.Row{
				obj("a", "", trajctl.StatusActive, 0),
				commit("a", 0.2, 1), commit("a", 0.6, 2),
				obj("b", "", trajctl.StatusActive, 0),
				commit("b", 0.3, 1), commit("b", 0.9, 2),
				obj("done", "", trajctl.StatusMet, 0),
			},
			wantDebt: 0, wantOK: true,
			active: 2, drift: 0, stall: 0, overrun: 0, excessWIP: 0, healthy: 2,
		},
		{
			// Five active objectives, all healthy, but the WIP cap is 3: two over. That is
			// two focus debt purely from breadth even though every curve is rising — fan-out.
			name: "broad fleet over WIP cap",
			rows: []trajctl.Row{
				obj("a", "", trajctl.StatusActive, 0), commit("a", 0.1, 1), commit("a", 0.5, 2),
				obj("b", "", trajctl.StatusActive, 0), commit("b", 0.1, 1), commit("b", 0.5, 2),
				obj("c", "", trajctl.StatusActive, 0), commit("c", 0.1, 1), commit("c", 0.5, 2),
				obj("d", "", trajctl.StatusActive, 0), commit("d", 0.1, 1), commit("d", 0.5, 2),
				obj("e", "", trajctl.StatusActive, 0), commit("e", 0.1, 1), commit("e", 0.5, 2),
			},
			wantDebt: 2, wantOK: false,
			active: 5, drift: 0, stall: 0, overrun: 0, excessWIP: 2, healthy: 5,
		},
		{
			// One active objective whose witnessed curve DECLINES (0.7 -> 0.4): DRIFT. One
			// debt, under the WIP cap. The non-convergence axis catches it even though
			// breadth is fine.
			name: "single drifting objective",
			rows: []trajctl.Row{
				obj("a", "", trajctl.StatusActive, 0),
				commit("a", 0.7, 1), commit("a", 0.4, 2),
			},
			wantDebt: 1, wantOK: false,
			active: 1, drift: 1, stall: 0, overrun: 0, excessWIP: 0, healthy: 0,
		},
		{
			// One active objective, flat witnessed curve WITH an activity-divergence signal:
			// STALL (busy but not moving). One debt.
			name: "single stalled objective",
			rows: []trajctl.Row{
				obj("a", "", trajctl.StatusActive, 0),
				commit("a", 0.5, 1), commit("a", 0.5, 2),
				divergence("a", 0.8, 2),
			},
			wantDebt: 1, wantOK: false,
			active: 1, drift: 0, stall: 1, overrun: 0, excessWIP: 0, healthy: 0,
		},
		{
			// A paused parent and a child detour with a 1-turn budget that has scored 3
			// witnessed-progress turns: DETOUR_OVERRUN. The child is active (open), the
			// parent paused. One debt. The paused parent is itself an open objective with
			// no declining curve, so it classifies HEALTHY (healthy=1); it is NOT counted
			// as active, so it does not push excess WIP.
			name: "detour overrun past budget",
			rows: []trajctl.Row{
				obj("parent", "", trajctl.StatusPaused, 0),
				obj("child", "parent", trajctl.StatusActive, 1),
				commit("child", 0.2, 1), commit("child", 0.3, 2), commit("child", 0.4, 3),
			},
			wantDebt: 1, wantOK: false,
			active: 1, drift: 0, stall: 0, overrun: 1, excessWIP: 0, healthy: 1,
		},
		{
			// Debt is magnitude-summed: 4 active over a cap of 2 (excess 2) + one drifting
			// objective among them = 3 total debt (2 breadth + 1 non-convergence). Proves
			// focus_debt is NOT the HARD-fail count (which would be 2 KPIs).
			name:   "magnitude-summed debt across axes",
			wipCap: 2,
			rows: []trajctl.Row{
				obj("a", "", trajctl.StatusActive, 0), commit("a", 0.1, 1), commit("a", 0.5, 2),
				obj("b", "", trajctl.StatusActive, 0), commit("b", 0.1, 1), commit("b", 0.5, 2),
				obj("c", "", trajctl.StatusActive, 0), commit("c", 0.1, 1), commit("c", 0.5, 2),
				obj("d", "", trajctl.StatusActive, 0), commit("d", 0.8, 1), commit("d", 0.3, 2), // drifts
			},
			wantDebt: 3, wantOK: false,
			active: 4, drift: 1, stall: 0, overrun: 0, excessWIP: 2, healthy: 3,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := buildFromRows(tc.rows, tc.wipCap)
			if got := anyInt(p.Corpus[DebtKey]); got != tc.wantDebt {
				t.Errorf("focus_debt = %d, want %d (reason: %s)", got, tc.wantDebt, p.Reason)
			}
			if p.OK != tc.wantOK {
				t.Errorf("OK = %v, want %v", p.OK, tc.wantOK)
			}
			ev := p.Evidence
			if ev.Active != tc.active {
				t.Errorf("active = %d, want %d", ev.Active, tc.active)
			}
			if ev.Drift != tc.drift {
				t.Errorf("drift = %d, want %d", ev.Drift, tc.drift)
			}
			if ev.Stall != tc.stall {
				t.Errorf("stall = %d, want %d", ev.Stall, tc.stall)
			}
			if ev.DetourOverrun != tc.overrun {
				t.Errorf("detour_overrun = %d, want %d", ev.DetourOverrun, tc.overrun)
			}
			if ev.ExcessWIP != tc.excessWIP {
				t.Errorf("excess_wip = %d, want %d", ev.ExcessWIP, tc.excessWIP)
			}
			if ev.Healthy != tc.healthy {
				t.Errorf("healthy = %d, want %d", ev.Healthy, tc.healthy)
			}
		})
	}
}

// TestDebtIsMagnitudeNotKPICount pins the core anti-saturation property: a fleet many
// objectives over the cap reports debt at magnitude, not capped at one-per-KPI.
func TestDebtIsMagnitudeNotKPICount(t *testing.T) {
	rows := []trajctl.Row{}
	for _, id := range []string{"a", "b", "c", "d", "e", "f", "g"} {
		rows = append(rows, obj(id, "", trajctl.StatusActive, 0), commit(id, 0.1, 1), commit(id, 0.5, 2))
	}
	p := buildFromRows(rows, 3) // 7 active, cap 3 => 4 over
	if got := anyInt(p.Corpus[DebtKey]); got != 4 {
		t.Fatalf("focus_debt = %d, want 4 (one per excess active objective, not per-KPI)", got)
	}
}

// TestCleanIsGradeA proves a fully-converged fleet grades A with a clean verdict.
func TestCleanIsGradeA(t *testing.T) {
	rows := []trajctl.Row{
		obj("a", "", trajctl.StatusActive, 0), commit("a", 0.2, 1), commit("a", 0.9, 2),
		obj("done", "", trajctl.StatusMet, 0),
	}
	p := buildFromRows(rows, 3)
	if !p.OK {
		t.Fatalf("clean fleet not OK: %s", p.Reason)
	}
	if grade := anyStr(p.Corpus["grade"]); grade != "A" {
		t.Errorf("grade = %q, want A", grade)
	}
	if p.Verdict != "OK" {
		t.Errorf("verdict = %q, want OK", p.Verdict)
	}
}

// TestJSONShape proves the control-pane payload marshals with the headline debt key and
// the value/grade the fold family contract requires.
func TestJSONShape(t *testing.T) {
	p := buildFromRows([]trajctl.Row{
		obj("a", "", trajctl.StatusActive, 0), commit("a", 0.7, 1), commit("a", 0.3, 2),
	}, 3)
	b, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var round map[string]any
	if err := json.Unmarshal(b, &round); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	corpus, ok := round["corpus"].(map[string]any)
	if !ok {
		t.Fatal("corpus missing or wrong type")
	}
	for _, key := range []string{DebtKey, "value", "grade", "score", "convergence_value", "breadth_value"} {
		if _, ok := corpus[key]; !ok {
			t.Errorf("corpus missing key %q", key)
		}
	}
	if p.Schema != Schema {
		t.Errorf("schema = %q, want %q", p.Schema, Schema)
	}
}

// TestRenderAndMarkdownMentionDebt is a cheap smoke that the two human renderers surface
// the headline and don't panic on a debt-carrying payload.
func TestRenderAndMarkdownMentionDebt(t *testing.T) {
	p := buildFromRows([]trajctl.Row{
		obj("a", "", trajctl.StatusActive, 0), commit("a", 0.7, 1), commit("a", 0.3, 2),
	}, 3)
	if r := Render(p); !strings.Contains(r, "focus_debt") || !strings.Contains(r, "drifting") {
		t.Errorf("Render missing headline/finding:\n%s", r)
	}
	if m := Markdown(p); !strings.Contains(m, "focus_debt") || !strings.Contains(m, "focus scorecard") {
		t.Errorf("Markdown missing headline/title")
	}
}

// TestCompareVerdicts pins the compare verdict transitions the loop reads.
func TestCompareVerdicts(t *testing.T) {
	drifting := buildFromRows([]trajctl.Row{
		obj("a", "", trajctl.StatusActive, 0), commit("a", 0.7, 1), commit("a", 0.3, 2),
	}, 3)
	clean := buildFromRows([]trajctl.Row{
		obj("a", "", trajctl.StatusActive, 0), commit("a", 0.3, 1), commit("a", 0.8, 2),
	}, 3)

	// baseline drifting (debt 1) -> current clean (debt 0): converged.
	baseline := map[string]any{"corpus": drifting.Corpus}
	if out := Compare(clean, baseline); !strings.Contains(out, "converged") {
		t.Errorf("expected converged verdict, got:\n%s", out)
	}
	// baseline clean (debt 0) -> current drifting (debt 1): fanning out.
	baseClean := map[string]any{"corpus": clean.Corpus}
	if out := Compare(drifting, baseClean); !strings.Contains(out, "fanning out") {
		t.Errorf("expected fanning-out verdict, got:\n%s", out)
	}
}
