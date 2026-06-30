package operatorbrief

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cadencereport"
	"github.com/anthony-chaudhary/fak/internal/milestonereport"
	"github.com/anthony-chaudhary/fak/internal/programreport"
	"github.com/anthony-chaudhary/fak/internal/worktype"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func cleanCadence() cadencereport.Report {
	return cadencereport.FoldWithMaturity(
		cadencereport.Scores{Debt: 4, GradeDebt: 4, Measured: 12, TrendDirection: "flat", OK: true},
		cadencereport.Maturity{Score: 90, Grade: "A", Capabilities: 10, OK: true},
		cadencereport.Work{WindowDays: 7, Commits: 8, Ships: 6},
		cadencereport.Releases{Version: "v1.2.3", ActionKind: "wait", ActionDetail: "nothing release-worthy pending", Verdict: "OK", OK: true},
		cadencereport.FoldOpts{Workspace: "/repo", Commit: "abc1234", Date: "2026-06-30"},
	)
}

func cleanProgram() programreport.Report {
	p := programreport.InterpretPrograms([]programreport.Signal{
		{Class: worktype.KernelOptimization, Label: "kernel-optimization", Frontier: "perf work landing", Metric: 2, Direction: "advancing", Activity: 2, Window: "7d", OK: true},
		{Class: worktype.CacheOptimization, Label: "cache-optimization", Frontier: "realized reuse 0.620", Metric: 0.62, Direction: "holding", OK: true},
	})
	return programreport.Fold(p, programreport.FoldOpts{Workspace: "/repo", Commit: "abc1234", Date: "2026-06-30"})
}

func cleanMilestone() milestonereport.Report {
	m := milestonereport.Maturity{
		Cells:       3,
		Dist:        map[string]int{"M0": 0, "M1": 0, "M2": 0, "M3": 0, "M4": 2, "M5": 1, "M6": 0, "M7": 0},
		Matured:     3,
		Highest:     "M5",
		HighestRank: 5,
		ProgressPct: 71.4,
		OK:          true,
	}
	e := milestonereport.InterpretEpics(
		[]milestonereport.EpicSpec{{Number: 1, Title: "native harness"}},
		[]milestonereport.EpicCounts{{Number: 1, Closed: 4, Total: 4, Source: "label"}},
		"",
	)
	return milestonereport.Fold(m, e, milestonereport.FoldOpts{Workspace: "/repo", Commit: "abc1234", Date: "2026-06-30"})
}

func TestFoldPagesOnUnmeasuredSource(t *testing.T) {
	c := cleanCadence()
	c.OK, c.Verdict, c.Finding = false, "ACTION", "cadence_unmeasured"
	c.Reason = "cadence report incomplete - could not measure scores"
	c.NextAction = "repair scores, then rerun `fak cadence`"
	p := cleanProgram()
	m := cleanMilestone()

	got := Fold(Inputs{Cadence: &c, Program: &p, Milestone: &m})
	if got.OK || got.Finding != "operator_input_needed" || got.Pace != "intervene" {
		t.Fatalf("unmeasured source should page the operator, got %+v", got)
	}
	if len(got.Human) != 1 || !strings.Contains(got.Human[0].Title, "cadence") {
		t.Fatalf("human items = %+v, want one cadence page", got.Human)
	}
	if got.State.Mode != "intervene" || !strings.Contains(got.State.OperatorUse, "missing witness") {
		t.Fatalf("state should tell the operator how to intervene, got %+v", got.State)
	}
	if got.Coherence.Status != "coherent" {
		t.Fatalf("present source stamps should be coherent, got %+v", got.Coherence)
	}
	if got.Attention.Level != "interrupt" || got.Attention.BudgetMinutes != 15 || got.Attention.ReadOrder[0] != "human" {
		t.Fatalf("intervention attention plan = %+v", got.Attention)
	}
	if got.Agenda.Focus != "witness before judgment" || !strings.Contains(got.Agenda.Skip, "unrelated transcript") {
		t.Fatalf("intervention learning agenda = %+v", got.Agenda)
	}
	if !strings.Contains(got.HumanUse.UseHumanFor, "restore missing evidence") || !strings.Contains(got.HumanUse.Avoid, "partial pane") {
		t.Fatalf("intervention human-use contract = %+v", got.HumanUse)
	}
	if len(got.Choices) != 1 || got.Choices[0].Default != "intervene" {
		t.Fatalf("human page should become an intervene choice, got %+v", got.Choices)
	}
	if len(got.Challenges) != 1 || got.Challenges[0].Kind != "missing_or_unmeasured_signal" {
		t.Fatalf("human page should become a missing-signal challenge, got %+v", got.Challenges)
	}
	if len(got.Learning) == 0 || got.Learning[0].Topic != "witness before judgment" {
		t.Fatalf("human page should teach witness-before-judgment, got %+v", got.Learning)
	}
	if code, msg := CheckGate(got); code != 1 || !strings.Contains(msg, "OPERATOR ACTION") {
		t.Fatalf("operator action must gate 1, got %d %q", code, msg)
	}
}

func TestFoldDelegatesMaturityDebtToAgents(t *testing.T) {
	c := cleanCadence()
	c.Maturity.Debt = 2
	c.Maturity.Backlog = 5
	c.Maturity.RouteLane = "maturity"
	c.Maturity.RouteItem = "dogfood the operator pane"
	c.Finding = "cadence_advisory"
	c.NextAction = "run `fak maturity route --fetch-existing --limit 3`"
	p := cleanProgram()
	m := cleanMilestone()

	got := Fold(Inputs{Cadence: &c, Program: &p, Milestone: &m})
	if !got.OK || got.Finding != "agent_work_ready" || got.Pace != "delegate" {
		t.Fatalf("agent-ready work should be delegated without paging, got %+v", got)
	}
	if len(got.Human) != 0 || len(got.Agent) == 0 {
		t.Fatalf("human=%+v agent=%+v, want only agent work", got.Human, got.Agent)
	}
	if len(got.Choices) != 1 || got.Choices[0].Default != "delegate" {
		t.Fatalf("delegable work should expose a delegate choice, got %+v", got.Choices)
	}
	if got.Attention.Level != "delegate" || got.Attention.BudgetMinutes != 5 || got.Attention.Cadence != "at dispatch boundary" {
		t.Fatalf("delegation attention plan = %+v", got.Attention)
	}
	if got.Agenda.Focus != "delegation boundary" || !strings.Contains(got.Agenda.Practice, "maturity route") {
		t.Fatalf("delegation learning agenda = %+v", got.Agenda)
	}
	if !strings.Contains(got.HumanUse.UseHumanFor, "confirm the default delegation") || !strings.Contains(got.HumanUse.LetAgentsDo, "agent bucket") {
		t.Fatalf("delegation human-use contract = %+v", got.HumanUse)
	}
	if len(got.Challenges) != 0 {
		t.Fatalf("delegable work alone should not be a challenge, got %+v", got.Challenges)
	}
	if len(got.Strengths) == 0 || got.Strengths[0].Kind != "delegable" {
		t.Fatalf("delegable work should become a strength, got %+v", got.Strengths)
	}
	if len(got.Learning) == 0 || got.Learning[0].Topic != "delegation boundary" {
		t.Fatalf("delegable work should teach the delegation boundary, got %+v", got.Learning)
	}
	if code, _ := CheckGate(got); code != 0 {
		t.Fatalf("delegable agent work must gate 0, got %d", code)
	}
}

func TestFoldCarriesGenerationReadoutFromMilestone(t *testing.T) {
	c := cleanCadence()
	p := cleanProgram()
	m := cleanMilestone()
	m.Epics = milestonereport.InterpretEpics(
		[]milestonereport.EpicSpec{
			{Number: 1315, Title: "native harness", Generation: "now"},
			{Number: 1010, Title: "GLM kernel", Generation: "next"},
			{Number: 42, Title: "future option", Generation: "future"},
		},
		[]milestonereport.EpicCounts{
			{Number: 1315, Closed: 1, Total: 3, Source: "label"},
			{Number: 1010, Closed: 7, Total: 10, Source: "label"},
			{Number: 42, Closed: 1, Total: 1, Source: "checklist"},
		},
		"",
	)

	got := Fold(Inputs{Cadence: &c, Program: &p, Milestone: &m})
	if got.Generation == nil {
		t.Fatal("generation readout missing")
	}
	if !strings.Contains(got.Generation.Summary, "ship-now lane has 2 open discrete") || !strings.Contains(got.Generation.Summary, "2 later-horizon") {
		t.Fatalf("generation summary = %q", got.Generation.Summary)
	}
	if !strings.Contains(got.Generation.Attention, "now lane first") {
		t.Fatalf("generation attention = %q", got.Generation.Attention)
	}
	byGen := map[string]GenerationLane{}
	for _, lane := range got.Generation.Lanes {
		byGen[lane.Generation] = lane
	}
	if lane := byGen["now"]; lane.OpenDiscrete != 2 || lane.Discrete != 1 || lane.OverallPct != 33.3 {
		t.Fatalf("now lane = %+v, want 2 open discrete at 33.3%%", lane)
	}
	if lane := byGen["next"]; lane.Programs != 1 || lane.OpenDiscrete != 0 {
		t.Fatalf("next lane = %+v, want one ongoing program and no discrete open count", lane)
	}
	rendered := Render(got)
	for _, want := range []string{"generation ship-now lane", "delegate from the now lane first", "now: 1 tracked", "next: 1 tracked", "future: 1 tracked"} {
		if !strings.Contains(rendered, want) {
			t.Fatalf("render missing %q:\n%s", want, rendered)
		}
	}
}

func TestFoldReviewModeForRegressedProgramFrontier(t *testing.T) {
	c := cleanCadence()
	m := cleanMilestone()
	p := programreport.Fold(programreport.InterpretPrograms([]programreport.Signal{
		{Class: worktype.CacheOptimization, Label: "cache-optimization", Frontier: "realized reuse fell", Metric: 0.4, Direction: "regressed", OK: true},
	}), programreport.FoldOpts{Workspace: "/repo", Commit: "abc1234", Date: "2026-06-30"})

	got := Fold(Inputs{Cadence: &c, Program: &p, Milestone: &m})
	if !got.OK || got.Finding != "operator_watchlist" || got.Pace != "review" {
		t.Fatalf("regressed program should be watchlist-only, got %+v", got)
	}
	if len(got.Choices) != 1 || got.Choices[0].Default != "review" {
		t.Fatalf("watchlist should expose a review choice, got %+v", got.Choices)
	}
	if got.Attention.Level != "review" || got.Attention.BudgetMinutes != 10 || got.Attention.ReadOrder[0] != "challenges" {
		t.Fatalf("watchlist attention plan = %+v", got.Attention)
	}
	if got.Agenda.Focus != "watchlist vs page" || !strings.Contains(got.Agenda.Skip, "interruption") {
		t.Fatalf("watchlist learning agenda = %+v", got.Agenda)
	}
	if !strings.Contains(got.HumanUse.UseHumanFor, "slow or redirect dispatch") || !strings.Contains(got.HumanUse.Avoid, "watch item") {
		t.Fatalf("review human-use contract = %+v", got.HumanUse)
	}
	if len(got.Challenges) != 1 || got.Challenges[0].Kind != "watch" {
		t.Fatalf("watchlist should expose a watch challenge, got %+v", got.Challenges)
	}
	if len(got.Learning) == 0 || got.Learning[0].Topic != "pace control" {
		t.Fatalf("watchlist should teach pace control, got %+v", got.Learning)
	}
	rendered := Render(got)
	for _, want := range []string{"operator brief", "pace", "attention", "human use", "strengths:", "choices:", "challenges:", "learning agenda:", "learning:", "watch:", "cache-optimization frontier regressed"} {
		if !strings.Contains(rendered, want) {
			t.Fatalf("render missing %q:\n%s", want, rendered)
		}
	}
}

func TestFoldHeavinessPressureIsWatchOnly(t *testing.T) {
	c := cleanCadence()
	p := cleanProgram()
	m := cleanMilestone()
	h := scorecard.Payload{
		Schema:     "fak-operator-heaviness-scorecard/1",
		OK:         true,
		Verdict:    "OK",
		Finding:    "operator surface clean of hard friction; heaviness pressure 86",
		NextAction: "hold; consolidate only if pressure rises",
		Workspace:  "/repo",
		Corpus: map[string]any{
			"heaviness_debt":     0,
			"heaviness_pressure": 86,
			"verbs":              136,
			"front_door_flags":   49,
			"refusal_reasons":    20,
		},
	}

	got := Fold(Inputs{Cadence: &c, Program: &p, Milestone: &m, Heaviness: &h})
	if !got.OK || got.Finding != "operator_watchlist" || got.Pace != "review" {
		t.Fatalf("heaviness pressure should be watchlist-only, got %+v", got)
	}
	if len(got.Human) != 0 || len(got.Agent) != 0 {
		t.Fatalf("heaviness pressure should not page or create agent work, human=%+v agent=%+v", got.Human, got.Agent)
	}
	if len(got.Watch) != 1 || got.Watch[0].Source != "heaviness" || !strings.Contains(got.Watch[0].Detail, "heaviness_pressure 86") {
		t.Fatalf("heaviness pressure watch item = %+v", got.Watch)
	}
	if got.Attention.Level != "review" || got.Choices[0].Default != "review" {
		t.Fatalf("heaviness pressure should tune review attention, attention=%+v choices=%+v", got.Attention, got.Choices)
	}
	if got.Agenda.Focus != "watchlist vs page" || !strings.Contains(got.Agenda.Practice, "pressure rises") {
		t.Fatalf("heaviness pressure should set a bounded learning agenda, got %+v", got.Agenda)
	}
	if got.Coherence.Status != "coherent" || !sourcePresent(got.Sources, "heaviness") {
		t.Fatalf("heaviness source should fold without breaking source coherence: coherence=%+v sources=%+v", got.Coherence, got.Sources)
	}
}

func TestFoldMonitorModeHasZeroAttentionBudget(t *testing.T) {
	c := cleanCadence()
	p := cleanProgram()
	m := cleanMilestone()

	got := Fold(Inputs{Cadence: &c, Program: &p, Milestone: &m})
	if !got.OK || got.Finding != "brief_clear" || got.Pace != "monitor" {
		t.Fatalf("clean inputs should monitor, got %+v", got)
	}
	if got.Attention.Level != "none" || got.Attention.BudgetMinutes != 0 {
		t.Fatalf("clear brief should not consume human attention, got %+v", got.Attention)
	}
	if got.Attention.ReadOrder[0] != "state" {
		t.Fatalf("clear read order = %+v", got.Attention.ReadOrder)
	}
	if got.Agenda.Focus != "negative signal discipline" || !strings.Contains(got.Agenda.Skip, "transcripts") {
		t.Fatalf("clear learning agenda = %+v", got.Agenda)
	}
	if got.Coherence.Status != "coherent" || len(got.Coherence.Stamps) != 3 {
		t.Fatalf("clear coherence = %+v", got.Coherence)
	}
	if !strings.Contains(got.HumanUse.Avoid, "many agents") || !strings.Contains(got.HumanUse.EscalateWhen, "source report changes") {
		t.Fatalf("clear human-use contract = %+v", got.HumanUse)
	}
}

func TestFoldReviewModeForMixedSourceSnapshot(t *testing.T) {
	c := cleanCadence()
	p := cleanProgram()
	p.Commit = "def5678"
	p.Date = "2026-06-29"
	m := cleanMilestone()

	got := Fold(Inputs{Cadence: &c, Program: &p, Milestone: &m})
	if !got.OK || got.Finding != "operator_watchlist" || got.Pace != "review" {
		t.Fatalf("mixed source snapshots should be review-only, got %+v", got)
	}
	if got.Coherence.Status != "mixed" || !strings.Contains(got.Coherence.Summary, "date stamp") {
		t.Fatalf("mixed coherence = %+v", got.Coherence)
	}
	if len(got.Watch) != 1 || got.Watch[0].Source != "sources" {
		t.Fatalf("mixed source snapshot should create source watch item, got %+v", got.Watch)
	}
	rendered := Render(got)
	for _, want := range []string{"coherence", "source snapshots differ", "regenerate source reports together"} {
		if !strings.Contains(rendered, want) {
			t.Fatalf("render missing %q:\n%s", want, rendered)
		}
	}
}

func TestFoldSincePreviousCompressesChange(t *testing.T) {
	prevCadence := cleanCadence()
	prevCadence.Maturity.Debt = 2
	prevCadence.Maturity.Backlog = 5
	prevCadence.Maturity.RouteLane = "maturity"
	prevCadence.Maturity.RouteItem = "dogfood operator brief"
	prevCadence.Finding = "cadence_advisory"
	prev := Fold(Inputs{Cadence: &prevCadence, Program: ptrProgram(cleanProgram()), Milestone: ptrMilestone(cleanMilestone())})

	c := cleanCadence()
	m := cleanMilestone()
	p := programreport.Fold(programreport.InterpretPrograms([]programreport.Signal{
		{Class: worktype.CacheOptimization, Label: "cache-optimization", Frontier: "realized reuse fell", Metric: 0.4, Direction: "regressed", OK: true},
	}), programreport.FoldOpts{Workspace: "/repo", Commit: "abc1234", Date: "2026-06-30"})

	got := Fold(Inputs{Cadence: &c, Program: &p, Milestone: &m, Previous: &prev})
	if got.Delta == nil {
		t.Fatalf("delta missing")
	}
	if got.Delta.Status != "changed" || !got.Delta.PaceChanged || got.Delta.PaceFrom != "delegate" || got.Delta.PaceTo != "review" {
		t.Fatalf("delta pace/status = %+v", got.Delta)
	}
	if got.Delta.NewCount != 1 || got.Delta.ResolvedCount != 1 || got.Delta.PersistentCount != 0 {
		t.Fatalf("delta counts = %+v", got.Delta)
	}
	if len(got.Delta.New) != 1 || got.Delta.New[0].Bucket != "watch" || !strings.Contains(got.Delta.New[0].Title, "frontier regressed") {
		t.Fatalf("new delta items = %+v", got.Delta.New)
	}
	if len(got.Delta.Resolved) != 1 || got.Delta.Resolved[0].Bucket != "agent" || !strings.Contains(got.Delta.Resolved[0].Title, "maturity") {
		t.Fatalf("resolved delta items = %+v", got.Delta.Resolved)
	}
	if got.Attention.ReadOrder[0] != "since_previous" {
		t.Fatalf("changed brief should put delta first in read order: %+v", got.Attention)
	}
	rendered := Render(got)
	for _, want := range []string{"since previous:", "changed:", "new: [watch]", "resolved: [agent]"} {
		if !strings.Contains(rendered, want) {
			t.Fatalf("render missing %q:\n%s", want, rendered)
		}
	}
}

func ptrProgram(r programreport.Report) *programreport.Report {
	return &r
}

func ptrMilestone(r milestonereport.Report) *milestonereport.Report {
	return &r
}

func sourcePresent(srcs []SourceState, name string) bool {
	for _, s := range srcs {
		if s.Name == name {
			return true
		}
	}
	return false
}
