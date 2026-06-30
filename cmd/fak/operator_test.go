package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cadencereport"
	"github.com/anthony-chaudhary/fak/internal/heavinessscore"
	"github.com/anthony-chaudhary/fak/internal/milestonereport"
	"github.com/anthony-chaudhary/fak/internal/operatorbrief"
	"github.com/anthony-chaudhary/fak/internal/programreport"
	"github.com/anthony-chaudhary/fak/internal/worktype"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func TestOperatorBriefJSONFromCadenceInput(t *testing.T) {
	c := cadencereport.FoldWithMaturity(
		cadencereport.Scores{Err: "score pane timed out", TrendDirection: "unknown"},
		cadencereport.Maturity{OK: true},
		cadencereport.Work{WindowDays: 7},
		cadencereport.Releases{Version: "v1.0.0", ActionKind: "wait", OK: true},
		cadencereport.FoldOpts{Workspace: t.TempDir(), Commit: "abc1234", Date: "2026-06-30"},
	)
	path := writeOperatorBriefJSON(t, "cadence.json", c)

	var out, errb bytes.Buffer
	code := runOperatorBrief(&out, &errb, []string{"--workspace", t.TempDir(), "--cadence", path, "--json"})
	if code != 1 {
		t.Fatalf("brief with missing sources + unmeasured cadence should exit 1, got %d stderr=%s", code, errb.String())
	}
	var brief operatorbrief.Report
	if err := json.Unmarshal(out.Bytes(), &brief); err != nil {
		t.Fatalf("operator brief JSON did not parse: %v\n%s", err, out.String())
	}
	if brief.Schema != operatorbrief.Schema || brief.Finding != "operator_input_needed" {
		t.Fatalf("brief envelope = %+v", brief)
	}
	if len(brief.Human) == 0 {
		t.Fatalf("brief should carry human items for incomplete input: %+v", brief)
	}
	if brief.State.Mode != "intervene" || len(brief.Choices) == 0 || len(brief.Challenges) == 0 {
		t.Fatalf("brief should carry state/choices/challenges for operators: %+v", brief)
	}
	if brief.Attention.Level != "interrupt" || brief.Attention.BudgetMinutes == 0 {
		t.Fatalf("brief should carry a concrete attention plan: %+v", brief.Attention)
	}
	if brief.Agenda.Focus != "witness before judgment" || !strings.Contains(brief.Agenda.Skip, "transcript") {
		t.Fatalf("brief should carry a bounded learning agenda: %+v", brief.Agenda)
	}
	if brief.Coherence.Status != "partial" || !strings.Contains(brief.Coherence.Summary, "missing") {
		t.Fatalf("brief should surface partial source coherence: %+v", brief.Coherence)
	}
	if !strings.Contains(brief.HumanUse.UseHumanFor, "restore missing evidence") || !strings.Contains(brief.HumanUse.Avoid, "partial pane") {
		t.Fatalf("brief should carry a human-use contract: %+v", brief.HumanUse)
	}
	if len(brief.Learning) == 0 {
		t.Fatalf("brief should carry learning notes for operators: %+v", brief)
	}
}

func TestOperatorBriefPreviousAddsDelta(t *testing.T) {
	c := cadencereport.FoldWithMaturity(
		cadencereport.Scores{Debt: 1, GradeDebt: 1, Measured: 3, TrendDirection: "flat", OK: true},
		cadencereport.Maturity{OK: true},
		cadencereport.Work{WindowDays: 7},
		cadencereport.Releases{Version: "v1.0.0", ActionKind: "wait", OK: true},
		cadencereport.FoldOpts{Workspace: t.TempDir(), Commit: "abc1234", Date: "2026-06-30"},
	)
	cadencePath := writeOperatorBriefJSON(t, "cadence.json", c)
	previousPath := writeOperatorBriefJSON(t, "previous.json", operatorbrief.Report{
		Schema: operatorbrief.Schema,
		Pace:   "monitor",
	})

	var out, errb bytes.Buffer
	code := runOperatorBrief(&out, &errb, []string{"--workspace", t.TempDir(), "--cadence", cadencePath, "--previous", previousPath, "--json"})
	if code != 1 {
		t.Fatalf("brief with missing sources should exit 1, got %d stderr=%s", code, errb.String())
	}
	var brief operatorbrief.Report
	if err := json.Unmarshal(out.Bytes(), &brief); err != nil {
		t.Fatalf("operator brief JSON did not parse: %v\n%s", err, out.String())
	}
	if brief.Delta == nil || brief.Delta.Status != "changed" || brief.Delta.PaceFrom != "monitor" || brief.Delta.PaceTo != "intervene" {
		t.Fatalf("previous brief should produce a changed delta, got %+v", brief.Delta)
	}
	if brief.Delta.NewCount == 0 || len(brief.Delta.New) == 0 {
		t.Fatalf("delta should expose new human items, got %+v", brief.Delta)
	}
	if brief.Attention.ReadOrder[0] != "since_previous" {
		t.Fatalf("changed previous delta should lead read order, got %+v", brief.Attention)
	}
}

func TestOperatorBriefRejectsWrongSchema(t *testing.T) {
	path := writeOperatorBriefJSON(t, "wrong.json", map[string]any{"schema": "not-cadence"})

	var out, errb bytes.Buffer
	code := runOperatorBrief(&out, &errb, []string{"--cadence", path})
	if code != 2 {
		t.Fatalf("wrong schema exit = %d, want 2; stdout=%s stderr=%s", code, out.String(), errb.String())
	}
	if !strings.Contains(errb.String(), cadencereport.Schema) {
		t.Fatalf("stderr should name wanted schema, got %q", errb.String())
	}
}

func TestOperatorBriefCollectsMissingInputs(t *testing.T) {
	oldCadence, oldProgram, oldMilestone, oldHeaviness := operatorCollectCadence, operatorCollectProgram, operatorCollectMilestone, operatorCollectHeaviness
	defer func() {
		operatorCollectCadence, operatorCollectProgram, operatorCollectMilestone = oldCadence, oldProgram, oldMilestone
		operatorCollectHeaviness = oldHeaviness
	}()

	var cadenceArgs struct {
		timeout    int
		scoresFrom string
	}
	var programLedger string
	var milestoneArgs struct {
		repo      string
		epicsFrom string
	}
	var heavinessRoot string
	operatorCollectCadence = func(root, date string, timeout int, scoresFrom string) (cadencereport.Report, error) {
		cadenceArgs.timeout = timeout
		cadenceArgs.scoresFrom = scoresFrom
		r := cadencereport.FoldWithMaturity(
			cadencereport.Scores{Debt: 1, GradeDebt: 1, Measured: 3, TrendDirection: "flat", OK: true},
			cadencereport.Maturity{Score: 80, Grade: "B", Capabilities: 4, Debt: 1, Backlog: 2, RouteLane: "maturity", RouteItem: "dogfood operator brief", OK: true},
			cadencereport.Work{WindowDays: 7, Commits: 3, Ships: 2},
			cadencereport.Releases{Version: "v1.0.0", ActionKind: "wait", OK: true},
			cadencereport.FoldOpts{Workspace: root, Commit: "abc1234", Date: date},
		)
		return r, nil
	}
	operatorCollectProgram = func(root, date, cacheLedger string) (programreport.Report, error) {
		programLedger = cacheLedger
		p := programreport.InterpretPrograms([]programreport.Signal{
			{Class: worktype.KernelOptimization, Label: "kernel-optimization", Frontier: "perf work landing", Direction: "advancing", Activity: 1, OK: true},
		})
		return programreport.Fold(p, programreport.FoldOpts{Workspace: root, Commit: "abc1234", Date: date}), nil
	}
	operatorCollectMilestone = func(root, date, repo, epicsFrom string) (milestonereport.Report, error) {
		milestoneArgs.repo = repo
		milestoneArgs.epicsFrom = epicsFrom
		m := milestonereport.Maturity{Cells: 1, Matured: 1, Highest: "M4", ProgressPct: 57.1, Dist: map[string]int{"M4": 1}, OK: true}
		e := milestonereport.InterpretEpics(
			[]milestonereport.EpicSpec{{Number: 7, Title: "operator brief"}},
			[]milestonereport.EpicCounts{{Number: 7, Closed: 1, Total: 1, Source: "label"}},
			"",
		)
		return milestonereport.Fold(m, e, milestonereport.FoldOpts{Workspace: root, Commit: "abc1234", Date: date}), nil
	}
	operatorCollectHeaviness = func(root string) (scorecard.Payload, error) {
		heavinessRoot = root
		return scorecard.Payload{
			Schema:    heavinessscore.Schema,
			OK:        true,
			Verdict:   "OK",
			Finding:   "operator surface light",
			Workspace: root,
			Corpus: map[string]any{
				heavinessscore.DebtKey: 0,
				"heaviness_pressure":   0,
				"verbs":                3,
				"front_door_flags":     2,
				"refusal_reasons":      2,
			},
		}, nil
	}

	var out, errb bytes.Buffer
	code := runOperatorBrief(&out, &errb, []string{
		"--workspace", t.TempDir(),
		"--date", "2026-06-30",
		"--collect",
		"--collect-timeout", "7",
		"--scores-from", "scores.json",
		"--cache-ledger", "cache.jsonl",
		"--repo", "owner/repo",
		"--epics-from", "epics.json",
		"--json",
	})
	if code != 0 {
		t.Fatalf("collect brief exit = %d, stderr=%s", code, errb.String())
	}
	if cadenceArgs.timeout != 7 || cadenceArgs.scoresFrom != "scores.json" {
		t.Fatalf("cadence collect args = %+v", cadenceArgs)
	}
	if programLedger != "cache.jsonl" {
		t.Fatalf("program cache ledger = %q", programLedger)
	}
	if milestoneArgs.repo != "owner/repo" || milestoneArgs.epicsFrom != "epics.json" {
		t.Fatalf("milestone collect args = %+v", milestoneArgs)
	}
	if heavinessRoot == "" {
		t.Fatalf("heaviness collect was not called")
	}
	var brief operatorbrief.Report
	if err := json.Unmarshal(out.Bytes(), &brief); err != nil {
		t.Fatalf("parse collect brief: %v\n%s", err, out.String())
	}
	if brief.Finding != "agent_work_ready" || len(brief.Human) != 0 || len(brief.Agent) == 0 {
		t.Fatalf("collect should fold measured sources into delegable work, got %+v", brief)
	}
	if brief.Attention.Level != "delegate" || brief.Attention.BudgetMinutes != 5 {
		t.Fatalf("collect brief attention = %+v", brief.Attention)
	}
	if brief.Coherence.Status != "coherent" {
		t.Fatalf("collect brief coherence = %+v", brief.Coherence)
	}
	if brief.Agenda.Focus != "delegation boundary" {
		t.Fatalf("collect brief learning agenda = %+v", brief.Agenda)
	}
	if !sourcePresent(brief.Sources, "heaviness") {
		t.Fatalf("collect brief should include heaviness source: %+v", brief.Sources)
	}
	if !strings.Contains(brief.HumanUse.LetAgentsDo, "agent bucket") {
		t.Fatalf("collect brief human-use contract = %+v", brief.HumanUse)
	}
	if len(brief.Strengths) == 0 || len(brief.Learning) == 0 {
		t.Fatalf("collect brief should retain strengths and learning: %+v", brief)
	}
}

func sourcePresent(srcs []operatorbrief.SourceState, name string) bool {
	for _, s := range srcs {
		if s.Name == name {
			return true
		}
	}
	return false
}

func TestOperatorHeavinessJSONReadsWorkspace(t *testing.T) {
	root := writeOperatorHeavinessWorkspace(t)

	var out, errb bytes.Buffer
	code := runOperatorHeaviness(&out, &errb, []string{"--workspace", root, "--json"})
	if code != 0 {
		t.Fatalf("operator heaviness exit = %d, stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var payload struct {
		Schema string         `json:"schema"`
		OK     bool           `json:"ok"`
		Corpus map[string]any `json:"corpus"`
		KPIs   []struct {
			Key     string   `json:"kpi"`
			Defects []string `json:"defects"`
		} `json:"kpis"`
	}
	if err := json.Unmarshal(out.Bytes(), &payload); err != nil {
		t.Fatalf("operator heaviness JSON did not parse: %v\n%s", err, out.String())
	}
	if payload.Schema != heavinessscore.Schema || !payload.OK {
		t.Fatalf("operator heaviness payload = %+v", payload)
	}
	if debt := payload.Corpus[heavinessscore.DebtKey]; debt != float64(0) {
		t.Fatalf("operator heaviness debt = %v, want 0; payload=%+v", debt, payload)
	}
	if pressure := payload.Corpus["heaviness_pressure"]; pressure != float64(0) {
		t.Fatalf("operator heaviness pressure = %v, want 0; payload=%+v", pressure, payload)
	}
	if len(payload.KPIs) == 0 {
		t.Fatalf("operator heaviness should emit KPI rows: %+v", payload)
	}
}

func writeOperatorBriefJSON(t *testing.T, name string, v any) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, name)
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeOperatorHeavinessWorkspace(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	writeOperatorFile(t, root, "cmd/fak/main.go", `package main

func dispatch(name string) {
	switch name {
	case "complain":
	case "guard":
	case "operator":
	}
}
`)
	writeOperatorFile(t, root, "cmd/fak/guard.go", `package main

func guardFlags(fs interface{
	Bool(string, bool, string) *bool
	String(string, string, string) *string
}) {
	fs.Bool("check", false, "")
	fs.String("policy", "", "")
}
`)
	writeOperatorFile(t, root, "dos.toml", `[reasons.OFF_TRUNK]
summary = "stay on trunk"

[reasons.PUBLIC_LEAK]
summary = "scrub public copy"
`)
	writeOperatorFile(t, root, "llms.txt", `- [Steerability scorecard](docs/STEERABILITY-SCORECARD.md): run fak steering.
- [Operator-heaviness scorecard](docs/OPERATOR-HEAVINESS.md): heaviness_pressure via fak operator heaviness.
`)
	return root
}

func writeOperatorFile(t *testing.T, root, rel, body string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
}
