package main

// fak naive-control — the naive dispatch arm as a measured control group.
//
// The verb is the I/O shell over internal/naivecontrol: it runs the 45-character
// naive arm, reads the orchestrated arm's witness sidecars, hands both to the same
// git-ancestry verifier, appends one row per run to .fak/naive-control.jsonl, and
// prints both arms side by side. Every decision lives in the package; this file only
// gathers inputs.
//
// It self-routes from init() (the cmd/fak/server.go precedent) so the verb lands as
// one new file without editing the dispatch switch in main.go.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/issueorchestrator"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/naivecontrol"
)

const naiveControlUsage = `usage: fak naive-control <run|harvest|orchestrated|compare> [flags]

Record the naive dispatch arm (` + naivecontrol.NaiveCommand + `) and the
orchestrated dispatch loop on one ruler, then compare them. Ship counts come only
from git ancestry; a field that could not be read is UNKNOWN, never 0.

  fak naive-control run          [--plan FILE] [--run-id ID] [--json]
      run the naive arm once, record its picks and HEAD as the scan base (PLANNED)
  fak naive-control harvest      --run-id ID [--claims FILE] [--wall-ms N] [--cost-usd X] [--json]
      git-verify a run: scan base..HEAD for its picks, check every claimed sha (HARVESTED)
  fak naive-control orchestrated [--since 24h] [--runs-dir DIR] [--json]
      fold the orchestrated arm's exited-worker witness sidecars through the same verifier
  fak naive-control compare      [--since 720h] [--json]
      print both arms side by side

Common flags: --workspace DIR (default: repo root), --ledger PATH
(default: <workspace>/` + naivecontrol.DefaultLedgerRel + `).

Schedule it with the loop runner, e.g. from cron or launchd:
  fak loop run --loop naive-control --source cron -- fak naive-control run`

func init() {
	if len(os.Args) > 1 && os.Args[1] == "naive-control" {
		os.Exit(runNaiveControl(os.Stdout, os.Stderr, os.Args[2:]))
	}
}

// Test seams: git, the clock, and the arm's process are the only impure inputs.
var (
	naiveControlGit    naivecontrol.GitRunner = naivecontrol.ExecGit
	naiveControlNow                           = time.Now
	naiveControlRunArm                        = execNaiveArm
)

// naiveArmArgv is the naive arm plus --json, which changes only the output format.
var naiveArmArgv = []string{"issue-orchestrator", "--top", "10", "--max-waves", "1", "--json"}

func runNaiveControl(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, naiveControlUsage)
		return 2
	}
	switch argv[0] {
	case "run":
		return runNaiveControlRun(stdout, stderr, argv[1:])
	case "harvest":
		return runNaiveControlHarvest(stdout, stderr, argv[1:])
	case "orchestrated":
		return runNaiveControlOrchestrated(stdout, stderr, argv[1:])
	case "compare":
		return runNaiveControlCompare(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		fmt.Fprintln(stdout, naiveControlUsage)
		return 0
	}
	fmt.Fprintf(stderr, "fak naive-control: unknown subcommand %q\n%s\n", argv[0], naiveControlUsage)
	return 2
}

type naiveControlCommon struct {
	workspace *string
	ledger    *string
	asJSON    *bool
}

func naiveControlFlags(name string, stderr io.Writer) (*flag.FlagSet, naiveControlCommon) {
	fs := flag.NewFlagSet("fak naive-control "+name, flag.ContinueOnError)
	fs.SetOutput(stderr)
	return fs, naiveControlCommon{
		workspace: fs.String("workspace", "", "workspace (default: repo root)"),
		ledger:    fs.String("ledger", "", "ledger path (default: <workspace>/"+naivecontrol.DefaultLedgerRel+")"),
		asJSON:    fs.Bool("json", false, "emit JSON"),
	}
}

func (c naiveControlCommon) paths() (string, string) {
	ws := strings.TrimSpace(*c.workspace)
	if ws == "" {
		ws = repoRoot()
	}
	ledger := strings.TrimSpace(*c.ledger)
	if ledger == "" {
		ledger = filepath.Join(ws, naivecontrol.DefaultLedgerRel)
	}
	return ws, ledger
}

func runNaiveControlRun(stdout, stderr io.Writer, argv []string) int {
	fs, common := naiveControlFlags("run", stderr)
	planPath := fs.String("plan", "", "read a saved issue-orchestrator --json plan instead of running the arm")
	runID := fs.String("run-id", "", "run id (default: naive-<UTC stamp>)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	ws, ledger := common.paths()
	ctx := context.Background()
	start := naiveControlNow()
	id := strings.TrimSpace(*runID)
	if id == "" {
		id = "naive-" + start.UTC().Format("20060102-150405")
	}

	// HEAD before the arm runs is the base every pick's landed work is scanned from.
	head, headErr := naivecontrol.ResolveHead(ctx, naiveControlGit, ws)

	in := naivecontrol.RunInput{
		Arm:       naivecontrol.ArmNaive,
		RunID:     id,
		StartedAt: start,
		WallMS:    naivecontrol.Missing[int64]("the fanout had not finished when the arm was recorded; harvest --wall-ms supplies it"),
		CostUSD:   naivecontrol.Missing[float64]("no cost source at plan time; harvest --cost-usd supplies it"),
	}
	var planJSON []byte
	armFailed := false
	if p := strings.TrimSpace(*planPath); p != "" {
		in.Command = naivecontrol.NaiveCommand + " (saved plan " + filepath.Base(p) + ")"
		in.ControllerMS = naivecontrol.Missing[int64]("plan read from a file; the arm did not run here")
		b, err := os.ReadFile(p)
		if err != nil {
			in.PicksWhy = "read plan: " + err.Error()
		}
		planJSON = b
	} else {
		in.Command = "fak " + strings.Join(naiveArmArgv, " ")
		out, err := naiveControlRunArm(ctx, ws)
		in.ControllerMS = naivecontrol.Measured(naiveControlNow().Sub(start).Milliseconds())
		if err != nil {
			armFailed = true
			in.PicksWhy = "arm failed: " + err.Error()
		}
		planJSON = out
	}
	if in.PicksWhy == "" {
		issues, err := naiveControlPlanPicks(planJSON)
		if err != nil {
			in.PicksWhy = err.Error()
		} else {
			in.PicksKnown = true
			base := head
			for _, n := range issues {
				in.Picks = append(in.Picks, naivecontrol.Pick{Issue: n, Base: base})
			}
		}
	}
	in.RecordedAt = naiveControlNow()
	row := naivecontrol.Build(in)
	if err := naivecontrol.Append(ledger, row); err != nil {
		fmt.Fprintf(stderr, "fak naive-control run: %v\n", err)
		return 1
	}
	if rc := emitNaiveControlRow(stdout, stderr, row, *common.asJSON); rc != 0 {
		return rc
	}
	if head == "" {
		fmt.Fprintf(stderr, "fak naive-control run: HEAD unresolved (%s); harvest cannot scan this run, so its picks stay UNKNOWN\n", headErr)
	}
	if armFailed {
		fmt.Fprintf(stderr, "fak naive-control run: the arm failed; the failed run is recorded with picks UNKNOWN\n")
		return 1
	}
	return 0
}

// naiveControlPlanPicks reads wave 1 of an issue-orchestrator plan. A plan with no
// wave offered nothing, which is a measured 0, not an unreadable plan.
func naiveControlPlanPicks(b []byte) ([]int, error) {
	var plan issueorchestrator.Plan
	if err := json.Unmarshal(b, &plan); err != nil {
		return nil, fmt.Errorf("plan output is not issue-orchestrator JSON: %v", err)
	}
	if plan.Schema != issueorchestrator.WavePlanSchema {
		return nil, fmt.Errorf("plan schema %q, want %q", plan.Schema, issueorchestrator.WavePlanSchema)
	}
	if len(plan.Waves) == 0 {
		return []int{}, nil
	}
	return plan.Waves[0].IssueNumbers, nil
}

func execNaiveArm(ctx context.Context, workspace string) ([]byte, error) {
	self, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("locate fak binary: %w", err)
	}
	cmd := exec.CommandContext(ctx, self, naiveArmArgv...)
	cmd.Dir = workspace
	var stderr strings.Builder
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		msg := strings.TrimSpace(stderr.String())
		if len(msg) > 300 {
			msg = msg[:300] + "..."
		}
		return out, fmt.Errorf("%v: %s", err, msg)
	}
	return out, nil
}

func runNaiveControlHarvest(stdout, stderr io.Writer, argv []string) int {
	fs, common := naiveControlFlags("harvest", stderr)
	runID := fs.String("run-id", "", "run id to harvest (required)")
	arm := fs.String("arm", string(naivecontrol.ArmNaive), "arm of the run: naive|orchestrated")
	claimsPath := fs.String("claims", "", "self-reported claims: a JSON array of {issue, sha} or an issue-orchestrator harvest receipt")
	wallMS := fs.String("wall-ms", "", "end-to-end wall time of the run in ms (omit when unobserved)")
	costUSD := fs.String("cost-usd", "", "cost of the run in USD (omit when unobserved)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if strings.TrimSpace(*runID) == "" {
		fmt.Fprintln(stderr, "fak naive-control harvest: --run-id is required")
		return 2
	}
	armV := naivecontrol.Arm(strings.TrimSpace(*arm))
	if !armV.Valid() {
		fmt.Fprintf(stderr, "fak naive-control harvest: --arm %q is not naive|orchestrated\n", *arm)
		return 2
	}
	ws, ledger := common.paths()
	rows, _, err := naivecontrol.ReadLedger(ledger)
	if err != nil {
		fmt.Fprintf(stderr, "fak naive-control harvest: %v\n", err)
		return 1
	}
	prior, ok := naivecontrol.FindRun(rows, armV, *runID)
	if !ok {
		fmt.Fprintf(stderr, "fak naive-control harvest: no %s run %q in %s\n", armV, *runID, ledger)
		return 1
	}

	in := naivecontrol.RunInput{
		Arm:          armV,
		RunID:        prior.RunID,
		Command:      prior.Command,
		Picks:        naivecontrol.PicksOf(prior),
		PicksKnown:   prior.PicksOffered.IsMeasured(),
		PicksWhy:     prior.PicksOffered.WhyMissing(),
		ControllerMS: prior.ControllerMS,
		WallMS:       prior.WallMS,
		CostUSD:      prior.CostUSD,
	}
	if prior.StartedUnixNano > 0 {
		in.StartedAt = time.Unix(0, prior.StartedUnixNano)
	}
	if strings.TrimSpace(*claimsPath) != "" {
		claims, err := readNaiveControlClaims(*claimsPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak naive-control harvest: %v\n", err)
			return 1
		}
		in.SelfReported, in.SelfReportRead = claims, true
	}
	if s := strings.TrimSpace(*wallMS); s != "" {
		v, err := strconv.ParseInt(s, 10, 64)
		if err != nil || v < 0 {
			fmt.Fprintf(stderr, "fak naive-control harvest: --wall-ms %q is not a non-negative integer\n", s)
			return 2
		}
		in.WallMS = naivecontrol.Measured(v)
	}
	if s := strings.TrimSpace(*costUSD); s != "" {
		v, err := strconv.ParseFloat(s, 64)
		if err != nil || v < 0 {
			fmt.Fprintf(stderr, "fak naive-control harvest: --cost-usd %q is not a non-negative number\n", s)
			return 2
		}
		in.CostUSD = naivecontrol.Measured(v)
	}
	v := naivecontrol.Verify(context.Background(), naiveControlGit, ws, in.Picks, in.SelfReported)
	in.Verification = &v
	in.RecordedAt = naiveControlNow()
	row := naivecontrol.Build(in)
	if err := naivecontrol.Append(ledger, row); err != nil {
		fmt.Fprintf(stderr, "fak naive-control harvest: %v\n", err)
		return 1
	}
	return emitNaiveControlRow(stdout, stderr, row, *common.asJSON)
}

// readNaiveControlClaims accepts the two self-report shapes a naive fanout leaves
// behind: a plain [{issue, sha}] list, or an issue-orchestrator harvest receipt.
func readNaiveControlClaims(path string) ([]naivecontrol.Claim, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read claims: %w", err)
	}
	trimmed := strings.TrimSpace(string(b))
	if strings.HasPrefix(trimmed, "[") {
		var claims []naivecontrol.Claim
		if err := json.Unmarshal(b, &claims); err != nil {
			return nil, fmt.Errorf("claims %s: %w", path, err)
		}
		for i := range claims {
			claims[i].Source = naivecontrol.SourceSelfReport
		}
		return claims, nil
	}
	var receipt issueorchestrator.HarvestReceipt
	if err := json.Unmarshal(b, &receipt); err != nil {
		return nil, fmt.Errorf("claims %s is neither a claim list nor a harvest receipt: %w", path, err)
	}
	var claims []naivecontrol.Claim
	for _, leaf := range receipt.Leaves {
		if sha := strings.TrimSpace(leaf.CommitSHA); sha != "" {
			claims = append(claims, naivecontrol.Claim{Issue: leaf.IssueNumber, SHA: sha, Source: naivecontrol.SourceSelfReport})
		}
	}
	return claims, nil
}

func runNaiveControlOrchestrated(stdout, stderr io.Writer, argv []string) int {
	fs, common := naiveControlFlags("orchestrated", stderr)
	runsDir := fs.String("runs-dir", "", "dispatch runs dir (default: <workspace>/.dispatch-runs)")
	loopLedger := fs.String("loop-ledger", "", "loop ledger for tick_total_ms (default: $FAK_LOOP_LEDGER or <workspace>/.fak/loops.jsonl)")
	since := fs.Duration("since", 24*time.Hour, "window of worker spawns to fold")
	runID := fs.String("run-id", "", "run id (default: orchestrated-<UTC stamp>)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if *since <= 0 {
		fmt.Fprintln(stderr, "fak naive-control orchestrated: --since must be positive")
		return 2
	}
	ws, ledger := common.paths()
	dir := strings.TrimSpace(*runsDir)
	if dir == "" {
		dir = filepath.Join(ws, ".dispatch-runs")
	}
	loops := strings.TrimSpace(*loopLedger)
	if loops == "" {
		loops = defaultLoopLedger()
		if !filepath.IsAbs(loops) {
			loops = filepath.Join(ws, loops)
		}
	}
	now := naiveControlNow()
	from := now.Add(-*since)
	id := strings.TrimSpace(*runID)
	if id == "" {
		id = "orchestrated-" + now.UTC().Format("20060102-150405")
	}

	in := naivecontrol.RunInput{
		Arm:        naivecontrol.ArmOrchestrated,
		RunID:      id,
		StartedAt:  from,
		RecordedAt: now,
		Command:    fmt.Sprintf("fak dispatch tick (exited-worker witness sidecars in %s, window %s)", filepath.Base(dir), since.String()),
		WallMS:     naivecontrol.Missing[int64]("orchestrated workers are detached and no worker end event is recorded; tick_total_ms is controller time only"),
		CostUSD:    naivecontrol.Missing[float64]("the orchestrated path records no per-run USD: guard exit summaries are DOLLAR-BLIND text and dispatch-sessions cost is token-equivalents"),
	}
	workers, err := scanOrchestratedWorkers(dir, from, now)
	if err != nil {
		in.PicksWhy = err.Error()
	} else {
		in.Picks, in.SelfReported = naivecontrol.OrchestratedPicks(workers)
		in.PicksKnown, in.SelfReportRead = true, true
	}
	in.ControllerMS = orchestratedControllerMS(loops, from, now)

	v := naivecontrol.Verify(context.Background(), naiveControlGit, ws, in.Picks, in.SelfReported)
	in.Verification = &v
	row := naivecontrol.Build(in)
	if err := naivecontrol.Append(ledger, row); err != nil {
		fmt.Fprintf(stderr, "fak naive-control orchestrated: %v\n", err)
		return 1
	}
	return emitNaiveControlRow(stdout, stderr, row, *common.asJSON)
}

// scanOrchestratedWorkers reads the exited workers spawned in [from, to]. A worker
// still running has no sidecar yet and is not counted; a sidecar that does not parse
// makes the offered set unreadable rather than silently smaller.
func scanOrchestratedWorkers(dir string, from, to time.Time) ([]naivecontrol.OrchestratedWorker, error) {
	if _, err := os.Stat(dir); err != nil {
		return nil, fmt.Errorf("runs dir unreadable: %v", err)
	}
	paths, err := filepath.Glob(filepath.Join(dir, "resolve-*.witness"))
	if err != nil {
		return nil, fmt.Errorf("glob witness sidecars: %v", err)
	}
	sort.Strings(paths)
	var out []naivecontrol.OrchestratedWorker
	var bad []string
	for _, p := range paths {
		name := filepath.Base(p)
		at, ok := naivecontrol.SpawnStamp(name)
		if !ok || at.Before(from) || at.After(to) {
			continue
		}
		b, err := os.ReadFile(p)
		if err != nil {
			bad = append(bad, name+": "+err.Error())
			continue
		}
		sc, err := naivecontrol.ParseWitnessSidecar(b)
		if err != nil {
			bad = append(bad, name+": "+err.Error())
			continue
		}
		w := naivecontrol.OrchestratedWorker{Sidecar: sc}
		if base, err := os.ReadFile(strings.TrimSuffix(p, ".witness") + dispatchtick.BaseSHASidecarSuffix); err == nil {
			w.Base = strings.TrimSpace(string(base))
		}
		out = append(out, w)
	}
	if len(bad) > 0 {
		return nil, fmt.Errorf("%d witness sidecar(s) unreadable, first: %s", len(bad), bad[0])
	}
	return out, nil
}

// orchestratedControllerMS sums the dispatch ticks' tick_total_ms in the window —
// the counterpart of the naive arm's controller_ms (the time the arm spent choosing
// and admitting work). No tick in the window is UNKNOWN, not 0: ticks can run with
// the loop ledger disabled.
func orchestratedControllerMS(path string, from, to time.Time) naivecontrol.Metric[int64] {
	events, err := loopmgr.LoadAll(path)
	if err != nil {
		return naivecontrol.Missing[int64]("loop ledger unreadable: " + err.Error())
	}
	var sum int64
	ticks := 0
	for _, e := range events {
		at := time.Unix(0, e.TSUnixNano)
		if !strings.HasPrefix(e.LoopID, "issue-resolve-dispatch") || at.Before(from) || at.After(to) {
			continue
		}
		if ms, ok := e.Metrics["tick_total_ms"]; ok {
			sum += ms
			ticks++
		}
	}
	if ticks == 0 {
		return naivecontrol.Missing[int64](fmt.Sprintf("no issue-resolve-dispatch tick with tick_total_ms in the window (%s)", filepath.Base(path)))
	}
	return naivecontrol.Measured(sum)
}

func runNaiveControlCompare(stdout, stderr io.Writer, argv []string) int {
	fs, common := naiveControlFlags("compare", stderr)
	since := fs.Duration("since", 0, "compare only runs recorded within this window (default: the whole ledger)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if *since < 0 {
		fmt.Fprintln(stderr, "fak naive-control compare: --since must not be negative")
		return 2
	}
	_, ledger := common.paths()
	rows, health, err := naivecontrol.ReadLedger(ledger)
	if err != nil {
		fmt.Fprintf(stderr, "fak naive-control compare: %v\n", err)
		return 1
	}
	var from time.Time
	if *since > 0 {
		from = naiveControlNow().Add(-*since)
		rows = naivecontrol.Since(rows, from)
	}
	c := naivecontrol.Compare(rows, health)
	if !from.IsZero() {
		c.WindowStartUnixNano = from.UnixNano()
	}
	if *common.asJSON {
		return encodeJSONOrFail(stdout, stderr, c, "fak naive-control compare")
	}
	if err := naivecontrol.RenderComparison(stdout, c); err != nil {
		fmt.Fprintf(stderr, "fak naive-control compare: %v\n", err)
		return 1
	}
	return 0
}

func emitNaiveControlRow(stdout, stderr io.Writer, r naivecontrol.Row, asJSON bool) int {
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, r, "fak naive-control")
	}
	fmt.Fprintf(stdout, "%s %s run %s (%s)\n", r.Stage, r.Arm, r.RunID, r.Command)
	for _, f := range []struct {
		name string
		m    interface {
			String() string
			WhyMissing() string
		}
	}{
		{"picks offered", r.PicksOffered},
		{"picks shipped (git)", r.PicksShipped},
		{"commits claimed", r.CommitsClaimed},
		{"commits landed (git)", r.CommitsLanded},
		{"wall_ms", r.WallMS},
		{"controller_ms", r.ControllerMS},
		{"cost_usd", r.CostUSD},
	} {
		line := fmt.Sprintf("  %-21s %s", f.name, f.m.String())
		if why := f.m.WhyMissing(); why != "" {
			line += "  (" + why + ")"
		}
		fmt.Fprintln(stdout, line)
	}
	if r.HeadSHA != "" {
		fmt.Fprintf(stdout, "  verified against HEAD %s\n", r.HeadSHA)
	}
	return 0
}
