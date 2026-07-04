package main

// fak trajctl -- issue #2535, spine step 2 of the trajectory-control epic
// (#2533): the objective lifecycle CLI (declare/close/list) over
// internal/trajctl's append-only ledger. Score anything needs a way to name
// the thing being scored before any scorer can run; this is that naming
// surface.
//
// The main.go dispatch case is intentionally NOT wired in this change --
// main.go is carrying unrelated in-flight work (the verbFlagUsage sweep) and a
// shared-tree pathspec commit would sweep it (the peer-sweep-commit fence; see
// cmd/fak/memvaluescore.go for the precedent). The one-line
// `case "trajctl": cmdTrajctl(os.Args[2:])` lands when that lane clears; until
// then the verb runs through its package tests (runTrajctl below).

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/loopdrive"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

func cmdTrajctl(argv []string) { os.Exit(runTrajctl(os.Stdout, os.Stderr, argv)) }

var _ = cmdTrajctl // parked: referenced by the main.go case once the cmd lane clears

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

  fak trajctl close --id ID [--status met|abandoned] [--ledger FILE] [--json]
      Append a status-flip row for ID (default --status met). The prior
      statement/plan/budget/parent carry over unchanged. Fails if ID has never
      been declared.

  fak trajctl list [--status active|paused|met|abandoned|open|all] [--ledger FILE] [--json]
      List objectives in id order. Default --status is "open" (active+paused).

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
	case "list":
		return runTrajctlList(stdout, stderr, rest)
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

func trajctlEmitObjective(stdout, stderr io.Writer, obj trajctl.Objective, verb string, asJSON bool) int {
	if asJSON {
		return trajctlEmitJSON(stdout, stderr, obj)
	}
	fmt.Fprintf(stdout, "%s %s (status=%s)\n", verb, obj.ID, obj.Status)
	return 0
}

func trajctlEmitJSON(stdout, stderr io.Writer, v any) int {
	if err := writeIndentedJSON(stdout, v); err != nil {
		fmt.Fprintf(stderr, "fak trajctl: encode json: %v\n", err)
		return 1
	}
	return 0
}
