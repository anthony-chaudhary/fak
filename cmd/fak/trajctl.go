package main

// fak trajctl -- issue #2535, spine step 2 of the trajectory-control epic
// (#2533): the objective lifecycle CLI (declare/close/list) over
// internal/trajctl's append-only ledger. Score anything needs a way to name
// the thing being scored before any scorer can run; this is that naming
// surface.
//
// UNPARKED (#2765, the CLI half): the main.go dispatch case
// `case "trajctl": cmdTrajctl(os.Args[2:])` is now wired -- the peer-WIP fence
// that held it out (the verbFlagUsage sweep on main.go) has cleared, so `fak
// trajctl` is a live operator front door and the superloop improve-trajectory
// enter-hints (`fak trajctl curve ...`) are runnable-as-printed. The
// steering-ladder / regime-gated nudge half of #2765 stays open.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/depthadmit"
	"github.com/anthony-chaudhary/fak/internal/loopdrive"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

func cmdTrajctl(argv []string) { os.Exit(runTrajctl(os.Stdout, os.Stderr, argv)) }

const trajctlUsage = `fak trajctl - trajectory-control objective lifecycle (over internal/trajctl, #2533)

  fak trajctl declare --id ID --statement TEXT [--parent ID] [--plan TEXT ...]
                       [--budget-turns N] [--budget-tokens N] [--ledger FILE] [--json]
      Append a new objective row (status=active). Re-declaring an id appends a
      fresh row -- the ledger fold keeps only the latest, so this also reopens
      a closed objective.

  fak trajctl declare --from-goal GOAL.md [--id ID] [--parent ID] [--ledger FILE] [--json]
      Import loopdrive.Spec's Objective/Plan/Budget from GOAL.md as the
      canonical session objective. --id defaults to the GOAL.md 'loop:' id.
      Mutually exclusive with --statement/--plan/--budget-turns/--budget-tokens.

      Rubric gate (#2544, opt-in — costs ONE model call): add
      --rubric-base-url URL [--rubric-model M] [--rubric-max-call-tokens N]
      [--api-key-env NAME] to either declare form to generate an
      issue-specific judging rubric at declare time and cache it on the
      objective row. Later "score --method judge" runs read the cached rubric
      (rubric-based prompt, per-criterion attribution) — never regenerate it.
      A failed or over-budget rubric call fails the declare (nothing appended).

  fak trajctl close --id ID [--status met|abandoned] [--force] [--ledger FILE] [--json]
      Append a status-flip row for ID (default --status met). The prior
      statement/plan/budget/parent carry over unchanged. Fails if ID has never
      been declared.

      Depth gate (internal/depthadmit): --status met is REFUSED with
      DEPTH_NOT_CARRIED unless every declared plan phase has a witness in the
      latest W3 witnessed-commit-progress row -- so "done" costs a carried plan
      instead of a self-report. An objective with NO plan is refused too: a met
      with nothing declared claims a depth nobody can check. --status abandoned
      is never refused (it claims nothing) but still records the depth reached.
      --force closes shallow anyway, loudly.

  fak trajctl depth [--id ID] [--ledger FILE] [--json]
      Read how far down its declared plan an objective actually got: the
      carried/declared phase count, the FRONTIER (the first phase with no
      witness -- the concrete next step), any off-plan witnesses, and the
      one-line handoff a successor session resumes from instead of re-planning.
      Default is every open (active+paused) objective.

  fak trajctl list [--status active|paused|met|abandoned|open|all] [--ledger FILE] [--json]
      List objectives in id order. Default --status is "open" (active+paused).

  fak trajctl curve [--objective ID] [--ledger FILE] [--json]
      Fold the accumulated score rows into a time-ordered curve per
      (objective, method) and derive one closed-vocabulary signal:
      HEALTHY | STALL | DRIFT | DETOUR_OVERRUN. With --objective, fold that
      one objective; without it, list every open objective worst-first.
      --json emits the pinned ` + trajctl.CurveSchema + ` report.

  fak trajctl score --objective ID --method judge --base-url URL [--model M]
                     [--state TEXT] [--max-call-tokens N] [--api-key-env NAME]
                     [--ledger FILE] [--json]
      Run one registered scorer over one objective and append its rows.
      --method judge (#2543) asks the gateway for a pinned-schema structured
      verdict against the objective statement and appends a W1 row carrying the
      verdict blob as evidence; the per-call token cap is enforced on both the
      request and the returned usage (over-budget or failed calls append
      nothing). Costs real tokens -- never runs on the stop-hook path.

  fak trajctl scorers [--ledger FILE] [--json]
      Calibration leaderboard: per scorer method+version, how well its scores
      correlate with the W3 witnessed outcome (` + trajctl.GroundTruthMethod + `),
      ranked best-first so the worst-calibrated repair target sits at the foot.
      Low-calibration scorers are annotated, never dropped. --json emits the
      pinned ` + trajctl.CalibrationSchema + ` report.

  fak trajctl quote --corpus FILE --capability-version V --policy-version V
                     --index-version V --index-coverage N --quality-min N
                     --quality-witness METHOD [--ledger FILE] [--at RFC3339]
      Emit an immutable repo_question cost quote from witnessed history.
      Unsupported cold starts are refused (exit 3), never guessed.

  fak trajctl quote-revise --quote QUOTE.json --reason route_failure
                            [--revision N] [--ledger FILE] [--at RFC3339]
      Append a widening revision bound to the initial quote; never overwrite it.

  fak trajctl quote-complete --quote QUOTE.json --ledger FILE
                              --quality-score N --quality-witness REF
                              --raw-cost N --cost-unit UNIT [--censored]
      Bind the declared quality contract to an external witness and raw cost.

  fak trajctl quote-backtest --corpus FILE
      Chronological held-out p50/p80/p95 coverage, retaining censored outcomes.

  fak trajctl backtest --scorer X --corpus Y [--incumbent M] [--json]
      THE QUALIFICATION GATE — a scorer ships with its backtest (#2573).
      Replay a candidate scorer over a RECORDED corpus (a trajctl ledger
      exported earlier) and calibrate its fresh readings against the witnessed
      W3 outcomes already in that history, optionally alongside the incumbent
      method it would replace. Offline and free: nothing is appended, no model
      is called, and the live ledger is never the default corpus.

      A new or changed scoring method must earn QUALIFIED here before it is
      registered for live use: its readings must track the outcome at the
      well-calibrated bar AND not materially regress the incumbent.

      The verdict vocabulary is closed, so "could not run" is never mistaken
      for "passed": QUALIFIED | NOT_QUALIFIED | INCONCLUSIVE (the corpus could
      not decide -- NOT a pass) | BACKTEST_ERROR. --json emits the pinned
      ` + trajctl.BacktestSchema + ` report.
      Backtest exit: 0 qualified, 3 ran-and-refused, 1 could-not-run, 2 usage.

Ledger default: <root>/` + trajctl.DefaultLedgerRel + `
Exit: 0 ok, 2 usage error, 1 ledger/parse failure.`

func runTrajctl(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, trajctlUsage)
		return 2
	}
	sub, rest := argv[0], argv[1:]
	switch sub {
	case "declare":
		return runTrajctlDeclare(stdout, stderr, rest)
	case "close":
		return runTrajctlClose(stdout, stderr, rest)
	case "depth":
		return runTrajctlDepth(stdout, stderr, rest)
	case "list":
		return runTrajctlList(stdout, stderr, rest)
	case "curve":
		return runTrajctlCurve(stdout, stderr, rest)
	case "score":
		return runTrajctlScore(stdout, stderr, rest)
	case "scorers":
		return runTrajctlScorers(stdout, stderr, rest)
	case "backtest":
		return runTrajctlBacktest(stdout, stderr, rest)
	case "quote":
		return runTrajctlQuote(stdout, stderr, rest)
	case "quote-revise":
		return runTrajctlQuoteRevise(stdout, stderr, rest)
	case "quote-complete":
		return runTrajctlQuoteComplete(stdout, stderr, rest)
	case "quote-backtest":
		return runTrajctlQuoteBacktest(stdout, stderr, rest)
	case "-h", "--help", "help":
		fmt.Fprintln(stdout, trajctlUsage)
		return 0
	default:
		fmt.Fprintf(stderr, "fak trajctl: unknown subcommand %q\n%s\n", sub, trajctlUsage)
		return 2
	}
}

func trajctlLedgerPath(flagVal string) string {
	if flagVal != "" {
		return flagVal
	}
	return repoRoot() + string(os.PathSeparator) + trajctl.DefaultLedgerRel
}

// trajctlPlanFlag accumulates each `--plan TEXT` into a slice, mirroring
// leaseref.go's repeatedString.
type trajctlPlanFlag []string

func (p *trajctlPlanFlag) String() string { return strings.Join(*p, ",") }
func (p *trajctlPlanFlag) Set(v string) error {
	*p = append(*p, v)
	return nil
}

func trajctlPlanFromText(items []string) []trajctl.PlanPhase {
	out := make([]trajctl.PlanPhase, 0, len(items))
	for i, text := range items {
		out = append(out, trajctl.PlanPhase{ID: fmt.Sprintf("phase-%d", i+1), Title: text})
	}
	return out
}

func trajctlPlanFromGoal(items []loopdrive.PlanItem) []trajctl.PlanPhase {
	out := make([]trajctl.PlanPhase, 0, len(items))
	for i, item := range items {
		out = append(out, trajctl.PlanPhase{ID: fmt.Sprintf("phase-%d", i+1), Title: item.Text})
	}
	return out
}

func runTrajctlDeclare(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak trajctl declare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	id := fs.String("id", "", "objective id (default when --from-goal: the GOAL.md 'loop:' id)")
	parent := fs.String("parent", "", "parent objective id, for hierarchy")
	statement := fs.String("statement", "", "the objective statement")
	var plan trajctlPlanFlag
	fs.Var(&plan, "plan", "a plan-phase title (repeatable, in order)")
	budgetTurns := fs.Int("budget-turns", 0, "detour-style turn budget (0 = unspecified)")
	budgetTokens := fs.Int("budget-tokens", 0, "detour-style token budget (0 = unspecified)")
	fromGoal := fs.String("from-goal", "", "import Objective/Plan/Budget from a GOAL.md path")
	rubricBaseURL := fs.String("rubric-base-url", "", "gate: generate an issue-specific judging rubric at declare time via this OpenAI-compatible API root (#2544; one model call, cached on the objective)")
	rubricModel := fs.String("rubric-model", "", "model id for the rubric call (empty: the gateway's default)")
	rubricMaxCallTokens := fs.Int("rubric-max-call-tokens", trajctl.DefaultRubricMaxCallTokens, "rubric per-call token budget cap, enforced on request and return")
	apiKeyEnv := fs.String("api-key-env", "FAK_GATEWAY_KEY", "env var NAMING the gateway bearer (never the secret itself)")
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+trajctl.DefaultLedgerRel+")")
	asJSON := fs.Bool("json", false, "emit the written objective as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	var obj trajctl.Objective
	if *fromGoal != "" {
		if *statement != "" || len(plan) > 0 || *budgetTurns != 0 || *budgetTokens != 0 {
			fmt.Fprintln(stderr, "fak trajctl declare: --from-goal is mutually exclusive with --statement/--plan/--budget-turns/--budget-tokens")
			return 2
		}
		data, err := os.ReadFile(*fromGoal)
		if err != nil {
			fmt.Fprintf(stderr, "fak trajctl declare: read %s: %v\n", *fromGoal, err)
			return 1
		}
		spec, err := loopdrive.Parse(data)
		if err != nil {
			fmt.Fprintf(stderr, "fak trajctl declare: parse %s: %v\n", *fromGoal, err)
			return 1
		}
		objID := *id
		if objID == "" {
			objID = spec.Loop
		}
		if objID == "" {
			fmt.Fprintln(stderr, "fak trajctl declare: --id is required (or set 'loop:' in the GOAL.md frontmatter)")
			return 2
		}
		obj = trajctl.Objective{
			ID:        objID,
			ParentID:  *parent,
			Statement: spec.Objective,
			Plan:      trajctlPlanFromGoal(spec.Plan),
			Budget:    trajctl.Budget{Turns: spec.Budget.MaxIters, Tokens: int(spec.Budget.MaxTokens)},
			Status:    trajctl.StatusActive,
		}
	} else {
		if *id == "" {
			fmt.Fprintln(stderr, "fak trajctl declare: --id is required")
			return 2
		}
		if *statement == "" {
			fmt.Fprintln(stderr, "fak trajctl declare: --statement is required (or use --from-goal)")
			return 2
		}
		obj = trajctl.Objective{
			ID:        *id,
			ParentID:  *parent,
			Statement: *statement,
			Plan:      trajctlPlanFromText(plan),
			Budget:    trajctl.Budget{Turns: *budgetTurns, Tokens: *budgetTokens},
			Status:    trajctl.StatusActive,
		}
	}

	// The rubric gate (#2544): one model call at declare time, cached on the
	// objective row. Fail-closed — a failed or over-budget rubric call fails
	// the declare before anything is appended, so a declared objective either
	// carries the rubric the operator asked for or does not exist.
	if strings.TrimSpace(*rubricBaseURL) != "" {
		rubric, err := trajctl.GenerateObjectiveRubric(&trajctl.GatewayRubricClient{
			BaseURL: *rubricBaseURL,
			APIKey:  os.Getenv(*apiKeyEnv),
			Model:   *rubricModel,
		}, obj, *rubricMaxCallTokens)
		if err != nil {
			fmt.Fprintf(stderr, "fak trajctl declare: %v\n", err)
			return 1
		}
		obj.Rubric = rubric
	}

	path := trajctlLedgerPath(*ledger)
	if err := trajctl.Append(path, trajctl.ObjectiveRecord(obj)); err != nil {
		fmt.Fprintf(stderr, "fak trajctl declare: %v\n", err)
		return 1
	}
	return trajctlEmitObjective(stdout, stderr, obj, "declared", *asJSON)
}

func runTrajctlClose(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak trajctl close", flag.ContinueOnError)
	fs.SetOutput(stderr)
	id := fs.String("id", "", "objective id to close")
	status := fs.String("status", string(trajctl.StatusMet), "terminal status: met|abandoned")
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+trajctl.DefaultLedgerRel+")")
	asJSON := fs.Bool("json", false, "emit the closed objective as JSON")
	force := fs.Bool("force", false, "close --status met even when the declared plan is not carried ("+depthadmit.RefusalReason+")")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if *id == "" {
		fmt.Fprintln(stderr, "fak trajctl close: --id is required")
		return 2
	}
	newStatus := trajctl.ObjectiveStatus(*status)
	if newStatus != trajctl.StatusMet && newStatus != trajctl.StatusAbandoned {
		fmt.Fprintf(stderr, "fak trajctl close: --status must be %q or %q, got %q\n", trajctl.StatusMet, trajctl.StatusAbandoned, *status)
		return 2
	}

	path := trajctlLedgerPath(*ledger)
	st := trajctl.Fold(trajctl.ReadLedgerFile(path))
	obj, ok := st.Objectives[*id]
	if !ok {
		fmt.Fprintf(stderr, "fak trajctl close: unknown objective %q\n", *id)
		return 1
	}
	// Depth gate (internal/depthadmit): `met` claims the objective's goal was
	// reached, so it costs a witnessed phase per declared phase. Without this, a
	// six-phase plan closed `met` on one witnessed phase read as a clean close to
	// every downstream fold — the curve and focus scores measure motion and
	// breadth, neither of which can see a line that stopped short. `abandoned` is
	// never gated: it claims nothing, and refusing it would only trap dead
	// objectives open.
	decision := depthadmit.Admit(trajctlDepthInput(obj, st), depthadmit.Closure(newStatus))
	if !decision.Admitted {
		if !*force {
			fmt.Fprintf(stderr, "fak trajctl close: %s: %s\n", decision.Reason, decision.Detail)
			fmt.Fprintf(stderr, "  %s\n", depthadmit.HandoffLine(obj.ID, decision.Report))
			fmt.Fprintln(stderr, "  carry the remaining phases (bind each landing commit with a `Trajctl-Phase: <id>` trailer,")
			fmt.Fprintln(stderr, "  then `fak trajctl score --objective "+obj.ID+"` to witness them), close with --status abandoned")
			fmt.Fprintln(stderr, "  to drop the line honestly, or re-run with --force to close it shallow anyway.")
			return 1
		}
		fmt.Fprintf(stderr, "fak trajctl close: --force overrode %s: %s\n", decision.Reason, decision.Detail)
	}

	obj.Status = newStatus
	if err := trajctl.Append(path, trajctl.ObjectiveRecord(obj)); err != nil {
		fmt.Fprintf(stderr, "fak trajctl close: %v\n", err)
		return 1
	}
	return trajctlEmitObjective(stdout, stderr, obj, "closed", *asJSON)
}

// trajctlStatusOpen is the synthetic list filter meaning active+paused --
// distinct from trajctl.ObjectiveStatus, which never carries this value.
const trajctlStatusOpen = "open"
const trajctlStatusAll = "all"

func runTrajctlList(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak trajctl list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	status := fs.String("status", trajctlStatusOpen, "filter: active|paused|met|abandoned|open|all")
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+trajctl.DefaultLedgerRel+")")
	asJSON := fs.Bool("json", false, "emit the list as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	switch *status {
	case trajctlStatusOpen, trajctlStatusAll, string(trajctl.StatusActive), string(trajctl.StatusPaused), string(trajctl.StatusMet), string(trajctl.StatusAbandoned):
	default:
		fmt.Fprintf(stderr, "fak trajctl list: --status %q is not one of active|paused|met|abandoned|open|all\n", *status)
		return 2
	}

	path := trajctlLedgerPath(*ledger)
	st := trajctl.Fold(trajctl.ReadLedgerFile(path))
	out := make([]trajctl.Objective, 0)
	for _, id := range st.ObjectiveIDs() {
		obj := st.Objectives[id]
		if trajctlListMatches(*status, obj.Status) {
			out = append(out, obj)
		}
	}
	if *asJSON {
		return trajctlEmitJSON(stdout, stderr, out)
	}
	if len(out) == 0 {
		fmt.Fprintf(stdout, "no objectives (status=%s) in %s\n", *status, path)
		return 0
	}
	for _, obj := range out {
		parent := ""
		if obj.ParentID != "" {
			parent = " parent=" + obj.ParentID
		}
		fmt.Fprintf(stdout, "%-24s status=%-9s%s %s\n", obj.ID, obj.Status, parent, obj.Statement)
	}
	return 0
}

func trajctlListMatches(filter string, status trajctl.ObjectiveStatus) bool {
	switch filter {
	case trajctlStatusAll:
		return true
	case trajctlStatusOpen:
		return status == trajctl.StatusActive || status == trajctl.StatusPaused
	default:
		return string(status) == filter
	}
}

func runTrajctlCurve(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak trajctl curve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	objective := fs.String("objective", "", "objective id to fold (default: worst-first across all open objectives)")
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+trajctl.DefaultLedgerRel+")")
	asJSON := fs.Bool("json", false, "emit the pinned "+trajctl.CurveSchema+" report as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	path := trajctlLedgerPath(*ledger)
	st := trajctl.Fold(trajctl.ReadLedgerFile(path))

	var rep trajctl.CurveReport
	if *objective != "" {
		r, ok := st.CurveReportFor(*objective)
		if !ok {
			fmt.Fprintf(stderr, "fak trajctl curve: unknown objective %q\n", *objective)
			return 1
		}
		rep = r
	} else {
		rep = st.OpenCurves()
	}

	if *asJSON {
		return trajctlEmitJSON(stdout, stderr, rep)
	}
	return trajctlRenderCurve(stdout, rep, path, *objective)
}

// trajctlRenderCurve prints one worst-first line per objective: id, signal,
// latest progress + delta, optional parent, and the human-readable detail.
func trajctlRenderCurve(stdout io.Writer, rep trajctl.CurveReport, path, objective string) int {
	if len(rep.Objectives) == 0 {
		if objective == "" {
			fmt.Fprintf(stdout, "no open objectives in %s\n", path)
		}
		return 0
	}
	for _, oc := range rep.Objectives {
		parent := ""
		if oc.ParentID != "" {
			parent = " parent=" + oc.ParentID
		}
		fmt.Fprintf(stdout, "%-24s %-14s latest=%.2f delta=%+.2f%s  %s\n",
			oc.ObjectiveID, oc.Signal, oc.Latest, oc.Delta, parent, oc.Detail)
	}
	return 0
}

// runTrajctlScorers renders the scorer calibration leaderboard (#2566): per scorer
// method+version, how well its numbers correlate with the W3 witnessed outcome, ranked
// best-first so the worst-calibrated (repair-target) scorer sits at the bottom.
func runTrajctlScorers(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak trajctl scorers", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+trajctl.DefaultLedgerRel+")")
	asJSON := fs.Bool("json", false, "emit the pinned "+trajctl.CalibrationSchema+" report as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	path := trajctlLedgerPath(*ledger)
	rep := trajctl.Fold(trajctl.ReadLedgerFile(path)).Calibrate()
	if *asJSON {
		return trajctlEmitJSON(stdout, stderr, rep)
	}
	return trajctlRenderScorers(stdout, rep, path)
}

// trajctlRenderScorers prints one line per scorer, best-first, and names the worst-first
// repair target at the foot when one is measured.
func trajctlRenderScorers(stdout io.Writer, rep trajctl.CalibrationReport, path string) int {
	if len(rep.Scorers) == 0 {
		fmt.Fprintf(stdout, "no scorers to calibrate in %s (need score rows plus a %s outcome)\n", path, rep.GroundTruth)
		return 0
	}
	fmt.Fprintf(stdout, "scorer calibration vs %s outcome (best-first):\n", rep.GroundTruth)
	for i, sc := range rep.Scorers {
		coeff := "  n/a"
		if sc.Measured {
			coeff = fmt.Sprintf("%+.2f", sc.Coefficient)
		}
		fmt.Fprintf(stdout, "  %2d. %-28s v%-4s r=%s  %-16s  %s\n",
			i+1, sc.Method, sc.Version, coeff, sc.Verdict, sc.Detail)
	}
	if worst, ok := rep.WorstCalibrated(); ok {
		fmt.Fprintf(stdout, "worst-first: repair %s (v%s, r=%+.2f) first\n", worst.Method, worst.Version, worst.Coefficient)
	}
	return 0
}

func trajctlEmitObjective(stdout, stderr io.Writer, obj trajctl.Objective, verb string, asJSON bool) int {
	if asJSON {
		return trajctlEmitJSON(stdout, stderr, obj)
	}
	fmt.Fprintf(stdout, "%s %s (status=%s)\n", verb, obj.ID, obj.Status)
	return 0
}

func trajctlEmitJSON(stdout, stderr io.Writer, v any) int {
	return encodeJSONOrFail(stdout, stderr, v, "fak trajctl")
}
