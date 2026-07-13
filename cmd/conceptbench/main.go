// Command conceptbench is the runnable wrapper for the FAK-concept fidelity
// benchmark (epic #2721, issue #2740): it runs a set of models against a set of
// fak-concept tasks (stamp/lane/refusal/verdict/handoff/witness) and emits a
// fak.conceptbench.report.v1 carrying a per-(model x concept) pass@1 leaderboard
// behind the #868 result-claim honesty gate.
//
// Today it drives the REPLAY path: --replay <dir> reads a committed task corpus
// (tasks.json) plus recorded per-(model,task) attempts (attempts.json) and grades
// each attempt with a deterministic, self-contained replay grader (the witness a
// model PRODUCED must equal the task's EXPECTED witness). Replay results are
// fixtures, not a live dos-refereed run, so result_claim_allowed is ALWAYS false —
// a leaderboard CLAIM waits for the live model-driver registry (#2731) and the
// dos-refereed grader (#2732) this command is built to plug in behind the same
// --models/--concepts/--out contract.
//
// --spine <fixture.json> drives the #2729 spine (epic #2721): ONE concept
// (commit_stamp + trunk fidelity) x TWO model arms x a REAL `dos commit-audit`
// grade x one fak.conceptbench.v1 row per model. Unlike the replay grader, each
// arm's recorded transcript is replayed into a fresh scratch git repo and its
// PRODUCED commit is graded by the live dos kernel referee — see spine.go.
//
// --contract emits a fak.conceptbench.contract.v1 (models, concepts, task ids, core
// budget) BEFORE a live run, carrying result_claim_allowed:false and NO scores,
// matching the #868 official-run contract discipline.
//
// Core budget: like cmd/modelbench, the resolved worker count follows
// FAK_WORKERS=<n> (absolute) > FAK_BUDGET=<fraction> > -budget=<fraction> > all
// logical cores (internal/model/budget.go). conceptbench does not run a forward
// pass, so the budget only sizes any concurrent grading and is RECORDED in the
// artifact so a scheduled fleet run states the regime it was taken at.
//
// The report/contract are emitted through benchcli.MarshalReport / WriteReport so
// the four lineage axes (version/utc/git/machine) are stamped onto every artifact —
// the contract the internal/benchlineagegate gate enforces for every cmd/*bench*.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/model"

	"github.com/anthony-chaudhary/fak/internal/maputil"
)

const (
	reportSchema   = "fak.conceptbench.report.v1"
	contractSchema = "fak.conceptbench.contract.v1"
	tasksSchema    = "fak.conceptbench.tasks.v1"
	attemptsSchema = "fak.conceptbench.attempts.v1"

	// grader is the self-contained replay grader id. It is deliberately NOT the full
	// dos-refereed grader (#2732): it compares the recorded produced witness to the
	// task's expected witness, so a replay run is reproducible offline while the live
	// kernel-refereed grader is wired.
	graderID = "replay-exact-witness/1"

	// replayClaimReason is the honesty-gate reason a replay report can never claim a
	// leaderboard result: fixtures are recorded, not a live dos-refereed run.
	replayClaimReason = "replay fixtures are recorded attempts, not a live dos-refereed run; a leaderboard claim is gated until the model-driver registry (#2731) and the dos-refereed grader (#2732) run live (mirrors #868 result_claim discipline)"
)

func main() { os.Exit(run(os.Args[1:])) }

// expectation is a checkable concept witness: a kind (refusal_token, stamp_leaf, ...)
// plus its value. The replay grader passes a task iff the produced expectation equals
// the expected one under witnessEqual.
type expectation struct {
	Kind  string `json:"kind"`
	Value string `json:"value"`
}

type task struct {
	TaskID  string      `json:"task_id"`
	Concept string      `json:"concept"`
	Prompt  string      `json:"prompt,omitempty"`
	Expect  expectation `json:"expect"`
}

type tasksFile struct {
	Schema string `json:"schema"`
	Tasks  []task `json:"tasks"`
}

type attempt struct {
	Model    string      `json:"model"`
	TaskID   string      `json:"task_id"`
	Produced expectation `json:"produced"`
}

type attemptsFile struct {
	Schema   string    `json:"schema"`
	Attempts []attempt `json:"attempts"`
}

// budgetInfo records how the core-budget knobs resolved for THIS run so a scheduled
// fleet artifact states the regime a number was taken at (mirrors cmd/modelbench).
type budgetInfo struct {
	Workers    int     `json:"workers"`
	Source     string  `json:"source"`
	BudgetFlag float64 `json:"budget_flag,omitempty"`
	FakWorkers string  `json:"fak_workers_env,omitempty"`
	FakBudget  string  `json:"fak_budget_env,omitempty"`
}

type gradedCell struct {
	Model       string      `json:"model"`
	Concept     string      `json:"concept"`
	TaskID      string      `json:"task_id"`
	Pass        bool        `json:"pass"`
	Expected    expectation `json:"expected"`
	Produced    expectation `json:"produced"`
	GradeReason string      `json:"grade_reason"`
}

type conceptScore struct {
	Concept string `json:"concept"`
	Passed  int    `json:"passed"`
	Total   int    `json:"total"`
}

type modelScore struct {
	Model     string         `json:"model"`
	PassAt1   float64        `json:"pass_at_1"`
	Passed    int            `json:"passed"`
	Total     int            `json:"total"`
	ByConcept []conceptScore `json:"by_concept"`
}

type reportTotals struct {
	Models   int `json:"models"`
	Concepts int `json:"concepts"`
	Tasks    int `json:"tasks"`
	Attempts int `json:"attempts"`
	Passed   int `json:"passed"`
	Failed   int `json:"failed"`
}

type report struct {
	Schema             string       `json:"schema"`
	AppVersion         string       `json:"app_version"`
	Mode               string       `json:"mode"`
	ReplayDir          string       `json:"replay_dir,omitempty"`
	Grader             string       `json:"grader"`
	Models             []string     `json:"models"`
	Concepts           []string     `json:"concepts"`
	Tasks              []string     `json:"tasks"`
	Budget             budgetInfo   `json:"budget"`
	ResultClaimAllowed bool         `json:"result_claim_allowed"`
	ResultClaimReason  string       `json:"result_claim_reason"`
	Totals             reportTotals `json:"totals"`
	Leaderboard        []modelScore `json:"leaderboard"`
	Cells              []gradedCell `json:"cells"`
}

type contract struct {
	Schema             string     `json:"schema"`
	AppVersion         string     `json:"app_version"`
	Kind               string     `json:"kind"`
	Issue              string     `json:"issue"`
	Models             []string   `json:"models"`
	Concepts           []string   `json:"concepts"`
	TaskIDs            []string   `json:"task_ids"`
	Budget             budgetInfo `json:"budget"`
	ResultClaimAllowed bool       `json:"result_claim_allowed"`
	Note               string     `json:"note"`
}

type flags struct {
	replay   string
	spine    string
	tasks    string
	models   string
	concepts string
	out      string
	contract bool
	budget   float64
	issue    string
}

func run(argv []string) int {
	fs := flag.NewFlagSet("conceptbench", flag.ContinueOnError)
	var f flags
	fs.StringVar(&f.replay, "replay", "", "replay directory holding tasks.json + attempts.json; grades recorded attempts offline")
	fs.StringVar(&f.spine, "spine", "", "spine fixture JSON (#2729): grade two model arms' produced commits with a real dos commit-audit call, one fak.conceptbench.v1 row per model")
	fs.StringVar(&f.tasks, "tasks", "", "explicit task corpus JSON (overrides <replay>/tasks.json); required for --contract without --replay")
	fs.StringVar(&f.models, "models", "", "comma-separated models to include (default: every model in the recorded attempts); REQUIRED for --contract")
	fs.StringVar(&f.concepts, "concepts", "", "comma-separated concepts to include (default: every concept in the corpus)")
	fs.StringVar(&f.out, "out", "", "write the JSON artifact here (default: stdout)")
	fs.BoolVar(&f.contract, "contract", false, "emit a pre-run official-run contract (models, concepts, task ids, budget) with result_claim_allowed:false and NO scores")
	fs.Float64Var(&f.budget, "budget", 0, "fractional core budget for this run (0.75 = up to 75% of the machine's logical cores); 0 = unset; FAK_WORKERS/FAK_BUDGET still apply")
	fs.StringVar(&f.issue, "issue", "#2740", "issue reference recorded in the contract artifact")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "conceptbench: unexpected positional arguments")
		return 2
	}

	if f.budget > 0 {
		if os.Getenv("FAK_WORKERS") != "" {
			fmt.Fprintf(os.Stderr, "[fak] FAK_WORKERS is set; ignoring -budget %g (absolute override wins)\n", f.budget)
		} else if err := model.SetWorkerBudget(f.budget); err != nil {
			fmt.Fprintln(os.Stderr, "budget:", err)
			return 2
		}
	}
	budget := budgetInfo{
		Workers:    model.NumWorkers(),
		Source:     model.WorkerBudget(),
		BudgetFlag: f.budget,
		FakWorkers: os.Getenv("FAK_WORKERS"),
		FakBudget:  os.Getenv("FAK_BUDGET"),
	}

	if f.spine != "" {
		return runSpine(f, budget)
	}
	if f.contract {
		return runContract(f, budget)
	}
	if f.replay == "" {
		fmt.Fprintln(os.Stderr, "conceptbench: a live (non-replay) run needs the model-driver registry (#2731) + dos-refereed grader (#2732), which are not yet wired.")
		fmt.Fprintln(os.Stderr, "  pass --replay <dir> to grade recorded attempts offline, or --contract to emit a pre-run contract.")
		return 2
	}
	return runReplay(f, budget)
}

// runReplay grades every recorded attempt against its task's expected witness and
// emits the fak.conceptbench.report.v1. result_claim_allowed is pinned false.
func runReplay(f flags, budget budgetInfo) int {
	tasks, err := loadTasks(tasksPath(f))
	if err != nil {
		fmt.Fprintln(os.Stderr, "conceptbench:", err)
		return 1
	}
	atts, err := loadAttempts(filepath.Join(f.replay, "attempts.json"))
	if err != nil {
		fmt.Fprintln(os.Stderr, "conceptbench:", err)
		return 1
	}

	conceptFilter := csvSet(f.concepts)
	modelFilter := csvSet(f.models)

	// Select the tasks in scope and index them by id.
	taskByID := map[string]task{}
	conceptSeen := map[string]bool{}
	var taskIDs []string
	for _, t := range tasks {
		if len(conceptFilter) > 0 && !conceptFilter[t.Concept] {
			continue
		}
		taskByID[t.TaskID] = t
		conceptSeen[t.Concept] = true
		taskIDs = append(taskIDs, t.TaskID)
	}
	if len(taskByID) == 0 {
		fmt.Fprintln(os.Stderr, "conceptbench: no tasks in scope (check --concepts against the corpus)")
		return 1
	}
	sort.Strings(taskIDs)

	// Grade each recorded attempt whose model+task are in scope.
	var cells []gradedCell
	modelSeen := map[string]bool{}
	for _, a := range atts {
		if len(modelFilter) > 0 && !modelFilter[a.Model] {
			continue
		}
		t, ok := taskByID[a.TaskID]
		if !ok {
			continue // attempt for an out-of-scope / unknown task
		}
		modelSeen[a.Model] = true
		pass, reason := grade(t.Expect, a.Produced)
		cells = append(cells, gradedCell{
			Model: a.Model, Concept: t.Concept, TaskID: t.TaskID,
			Pass: pass, Expected: t.Expect, Produced: a.Produced, GradeReason: reason,
		})
	}
	if len(cells) == 0 {
		fmt.Fprintln(os.Stderr, "conceptbench: no recorded attempts matched the selected models/tasks")
		return 1
	}
	sort.Slice(cells, func(i, j int) bool {
		if cells[i].Model != cells[j].Model {
			return cells[i].Model < cells[j].Model
		}
		if cells[i].Concept != cells[j].Concept {
			return cells[i].Concept < cells[j].Concept
		}
		return cells[i].TaskID < cells[j].TaskID
	})

	rep := report{
		Schema:             reportSchema,
		AppVersion:         appversion.Current(),
		Mode:               "replay",
		ReplayDir:          filepath.ToSlash(f.replay),
		Grader:             graderID,
		Models:             maputil.SortedKeys(modelSeen),
		Concepts:           maputil.SortedKeys(conceptSeen),
		Tasks:              taskIDs,
		Budget:             budget,
		ResultClaimAllowed: false,
		ResultClaimReason:  replayClaimReason,
		Leaderboard:        leaderboard(cells),
		Cells:              cells,
	}
	rep.Totals = totals(rep)
	return writeArtifact(f.out, rep)
}

// runContract emits the pre-run official-run contract: which models will run which
// concepts over which task ids, at what budget, with NO scores and the claim gate
// held shut. It requires --models (the operator declares the run's model set).
func runContract(f flags, budget budgetInfo) int {
	models := csvList(f.models)
	if len(models) == 0 {
		fmt.Fprintln(os.Stderr, "conceptbench --contract: --models is required (declare the models the official run will cover)")
		return 2
	}
	tp := tasksPath(f)
	if tp == "" {
		fmt.Fprintln(os.Stderr, "conceptbench --contract: pass --tasks <corpus.json> or --replay <dir> so the contract can list task ids")
		return 2
	}
	tasks, err := loadTasks(tp)
	if err != nil {
		fmt.Fprintln(os.Stderr, "conceptbench:", err)
		return 1
	}
	conceptFilter := csvSet(f.concepts)
	conceptSeen := map[string]bool{}
	var taskIDs []string
	for _, t := range tasks {
		if len(conceptFilter) > 0 && !conceptFilter[t.Concept] {
			continue
		}
		conceptSeen[t.Concept] = true
		taskIDs = append(taskIDs, t.TaskID)
	}
	if len(taskIDs) == 0 {
		fmt.Fprintln(os.Stderr, "conceptbench --contract: no tasks in scope (check --concepts against the corpus)")
		return 1
	}
	sort.Strings(taskIDs)
	sort.Strings(models)

	c := contract{
		Schema:             contractSchema,
		AppVersion:         appversion.Current(),
		Kind:               "official-run-contract",
		Issue:              f.issue,
		Models:             models,
		Concepts:           maputil.SortedKeys(conceptSeen),
		TaskIDs:            taskIDs,
		Budget:             budget,
		ResultClaimAllowed: false,
		Note:               "pre-run contract (mirrors #868): no scores are present and no leaderboard claim is allowed until each task is graded by a live dos-refereed run",
	}
	return writeArtifact(f.out, c)
}

// grade is the replay grader: a produced witness passes iff its kind matches and its
// value equals the expected value under witnessEqual (trim + case-fold), so trivial
// whitespace/case noise does not fail a correct token while genuine prose still does.
func grade(expect, produced expectation) (bool, string) {
	if !strings.EqualFold(strings.TrimSpace(produced.Kind), strings.TrimSpace(expect.Kind)) {
		return false, fmt.Sprintf("witness kind %q != expected %q", produced.Kind, expect.Kind)
	}
	if witnessEqual(produced.Value, expect.Value) {
		return true, fmt.Sprintf("%s witness matched %q", expect.Kind, expect.Value)
	}
	return false, fmt.Sprintf("%s witness %q != expected %q", expect.Kind, produced.Value, expect.Value)
}

func witnessEqual(a, b string) bool {
	return strings.EqualFold(strings.TrimSpace(a), strings.TrimSpace(b))
}

// leaderboard folds the graded cells into a per-model pass@1 with a per-concept
// breakdown, sorted by model for a byte-stable artifact.
func leaderboard(cells []gradedCell) []modelScore {
	type acc struct {
		passed, total int
		byConcept     map[string]*conceptScore
	}
	accs := map[string]*acc{}
	for _, c := range cells {
		a := accs[c.Model]
		if a == nil {
			a = &acc{byConcept: map[string]*conceptScore{}}
			accs[c.Model] = a
		}
		a.total++
		cs := a.byConcept[c.Concept]
		if cs == nil {
			cs = &conceptScore{Concept: c.Concept}
			a.byConcept[c.Concept] = cs
		}
		cs.Total++
		if c.Pass {
			a.passed++
			cs.Passed++
		}
	}
	var out []modelScore
	for m, a := range accs {
		var byC []conceptScore
		for _, cs := range a.byConcept {
			byC = append(byC, *cs)
		}
		sort.Slice(byC, func(i, j int) bool { return byC[i].Concept < byC[j].Concept })
		out = append(out, modelScore{
			Model: m, PassAt1: ratio(a.passed, a.total), Passed: a.passed, Total: a.total, ByConcept: byC,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Model < out[j].Model })
	return out
}

func totals(r report) reportTotals {
	t := reportTotals{
		Models: len(r.Models), Concepts: len(r.Concepts), Tasks: len(r.Tasks), Attempts: len(r.Cells),
	}
	for _, c := range r.Cells {
		if c.Pass {
			t.Passed++
		} else {
			t.Failed++
		}
	}
	return t
}

func ratio(n, d int) float64 {
	if d == 0 {
		return 0
	}
	return float64(n) / float64(d)
}

// tasksPath resolves the task corpus path: an explicit --tasks wins, else
// <replay>/tasks.json, else "" (no corpus available).
func tasksPath(f flags) string {
	if f.tasks != "" {
		return f.tasks
	}
	if f.replay != "" {
		return filepath.Join(f.replay, "tasks.json")
	}
	return ""
}

func loadTasks(path string) ([]task, error) {
	if path == "" {
		return nil, fmt.Errorf("no task corpus (pass --tasks or --replay)")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tasks %s: %w", path, err)
	}
	var tf tasksFile
	if err := json.Unmarshal(b, &tf); err != nil {
		return nil, fmt.Errorf("parse tasks %s: %w", path, err)
	}
	if len(tf.Tasks) == 0 {
		return nil, fmt.Errorf("tasks %s is empty", path)
	}
	return tf.Tasks, nil
}

func loadAttempts(path string) ([]attempt, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read attempts %s: %w", path, err)
	}
	var af attemptsFile
	if err := json.Unmarshal(b, &af); err != nil {
		return nil, fmt.Errorf("parse attempts %s: %w", path, err)
	}
	if len(af.Attempts) == 0 {
		return nil, fmt.Errorf("attempts %s is empty", path)
	}
	return af.Attempts, nil
}

// writeArtifact routes every artifact through benchcli so the lineage axes are
// stamped (internal/benchlineagegate). --out writes a file; default prints to stdout.
func writeArtifact(out string, v any) int {
	if out != "" {
		if err := benchcli.WriteReport(out, v); err != nil {
			fmt.Fprintln(os.Stderr, "conceptbench:", err)
			return 1
		}
		fmt.Fprintln(os.Stderr, "wrote", out)
		return 0
	}
	b, err := benchcli.MarshalReport(v)
	if err != nil {
		fmt.Fprintln(os.Stderr, "conceptbench:", err)
		return 1
	}
	fmt.Println(string(b))
	return 0
}

func csvList(s string) []string {
	var out []string
	seen := map[string]bool{}
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		if p == "" || seen[p] {
			continue
		}
		seen[p] = true
		out = append(out, p)
	}
	return out
}

func csvSet(s string) map[string]bool {
	set := map[string]bool{}
	for _, p := range csvList(s) {
		set[p] = true
	}
	return set
}
