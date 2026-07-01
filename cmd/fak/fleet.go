package main

// `fak fleet` — the headless-worker fleet control surface (#1856–#1859). One door
// for the four things a fleet run needs beyond a worker's own "busy":
//
//	fak fleet monitor  [--plan P] [--json] [--state S]   classify every worker from evidence
//	fak fleet janitor  [--plan P] [--json] [--apply]     find (and, with --apply, reap) stale child trees
//	fak fleet fold     [--plan P] [--json] [--ledger L] [--write]  fold final reports into a witnessed ledger
//	fak fleet replace  --session S [--index N] [--force] [--json]  render a safe replacement for a stuck worker
//
// monitor/janitor/fold are reads by default; janitor mutates only with --apply,
// fold appends to the ledger only with --write, replace launches only with
// --apply. Every decision is made by the pure internal/fleetmon package over an
// injected snapshot — this shell only gathers the snapshot (registry rows via
// internal/fleetaccounts, process relations + tree-kill via internal/procguard,
// transcript bytes via internal/fleetmon.ReadTranscript) and renders the result.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
	"github.com/anthony-chaudhary/fak/internal/fleetmon"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

func cmdFleet(argv []string) {
	dispatchSubcommands("fleet", "monitor | janitor | fold | replace", argv,
		subcommand{"monitor", runFleetMonitor},
		subcommand{"janitor", runFleetJanitor},
		subcommand{"fold", runFleetFold},
		subcommand{"replace", runFleetReplace},
	)
}

// --- injectable seams (overridden in tests so no live fleet is touched) --- //

// fleetCollectRelations returns the process relations snapshot (PPID/cmdline/age/
// start). Production uses procguard.CollectRelations; tests inject a fixture.
var fleetCollectRelations = procguard.CollectRelations

// fleetKillPID is the destructive tree reaper (taskkill /T on Windows). Tests
// inject a recorder so nothing is spawned or killed.
var fleetKillPID = procguard.KillPID

// fleetNow is the clock seam.
var fleetNow = time.Now

// --- shared plan + snapshot loading --------------------------------------- //

func loadFleetPlan(path string) (fleetmon.RunPlan, error) {
	if strings.TrimSpace(path) == "" {
		return fleetmon.RunPlan{}, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return fleetmon.RunPlan{}, err
	}
	return fleetmon.ParseRunPlan(data)
}

// livePIDs folds a relations snapshot into a liveness set.
func livePIDs(procs []procguard.Proc) map[int]bool {
	live := map[int]bool{}
	for _, p := range procs {
		if p.PID > 0 {
			live[p.PID] = true
		}
	}
	return live
}

// resolveTranscriptPath finds a worker's transcript file: the plan's explicit
// path wins; otherwise a single *.jsonl under the plan namespace; otherwise "".
func resolveTranscriptPath(w fleetmon.PlanWorker, home string) string {
	if w.TranscriptPath != "" {
		return w.TranscriptPath
	}
	if w.Namespace == "" || home == "" {
		return ""
	}
	dir := filepath.Join(home, ".claude", "projects", w.Namespace)
	matches, _ := filepath.Glob(filepath.Join(dir, "*.jsonl"))
	// Prefer a file whose name contains the session id; else the sole file.
	for _, m := range matches {
		if w.Session != "" && strings.Contains(filepath.Base(m), w.Session) {
			return m
		}
	}
	if len(matches) == 1 {
		return matches[0]
	}
	return ""
}

func fleetUserHome() string {
	if h := os.Getenv("FLEET_USER_HOME"); h != "" {
		return h
	}
	if h, err := os.UserHomeDir(); err == nil {
		return h
	}
	return ""
}

// registryMatch finds the registry session row that best matches a plan worker:
// explicit session/issue evidence in project/last wins, with account match as a
// fallback. A registry can carry several rows for one account, so returning the
// first account row can pin a worker to an unrelated auth block.
func registryMatch(reg fleetaccounts.Registry, w fleetmon.PlanWorker) (disp, action string) {
	var best fleetaccounts.Session
	bestScore := 0
	for _, s := range reg.Sessions {
		score := registryMatchScore(s, w)
		if score > bestScore {
			best, bestScore = s, score
		}
	}
	if bestScore > 0 {
		return best.Disp, best.Action
	}
	return "", ""
}

func registryMatchScore(s fleetaccounts.Session, w fleetmon.PlanWorker) int {
	score := 0
	if w.Account != "" && s.Account == w.Account {
		score += 10
	}
	if registryTextMatchesWorker(s, w) {
		score += 100
	}
	return score
}

func registryTextMatchesWorker(s fleetaccounts.Session, w fleetmon.PlanWorker) bool {
	hay := strings.ToLower(s.Project + " " + s.Last)
	if w.Session != "" && strings.Contains(hay, strings.ToLower(w.Session)) {
		return true
	}
	return w.Issue > 0 && containsIssueNumber(hay, w.Issue)
}

func containsIssueNumber(hay string, issue int) bool {
	needle := fmt.Sprintf("%d", issue)
	for start := strings.Index(hay, needle); start >= 0; {
		end := start + len(needle)
		beforeOK := start == 0 || !isASCIIDigit(hay[start-1])
		afterOK := end == len(hay) || !isASCIIDigit(hay[end])
		if beforeOK && afterOK {
			return true
		}
		next := strings.Index(hay[end:], needle)
		if next < 0 {
			break
		}
		start = end + next
	}
	return false
}

func isASCIIDigit(b byte) bool { return b >= '0' && b <= '9' }

// --- monitor (#1856) ------------------------------------------------------ //

type fleetState struct {
	Generated string                   `json:"generated"`
	Workers   map[string]fleetStateRow `json:"workers"`
}

type fleetStateRow struct {
	Lines   int     `json:"lines"`
	CPUSecs float64 `json:"cpu_secs"`
}

func runFleetMonitor(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fleet monitor", flag.ContinueOnError)
	fs.SetOutput(stderr)
	planPath := fs.String("plan", "", "run plan JSON (issue -> worker mapping)")
	asJSON := fs.Bool("json", false, "emit JSON instead of a table")
	home := fs.String("home", fleetUserHome(), "Claude config home (for transcript discovery)")
	statePath := fs.String("state", "", "previous-sample state file (for line/CPU deltas); also written back")
	staleTx := fs.Duration("stale-transcript", 20*time.Minute, "transcript-idle staleness floor")
	staleSimple := fs.Duration("stale-child-simple", 5*time.Minute, "simple child-command staleness floor")
	staleTest := fs.Duration("stale-child-test", 10*time.Minute, "test child-command staleness floor")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	plan, err := loadFleetPlan(*planPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak fleet monitor: %v\n", err)
		return 1
	}
	th := fleetmon.Thresholds{StaleTranscript: *staleTx, StaleChildSimple: *staleSimple, StaleChildTest: *staleTest}

	cwd, _ := os.Getwd()
	paths := fleetaccounts.ResolvePaths(filepath.Join(findRepoRoot(cwd), "tools"))
	reg := fleetaccounts.LoadRegistry(paths.RegistryPath)
	procs, collectErr := fleetCollectRelations()
	live := livePIDs(procs)
	prev := loadFleetState(*statePath)
	now := fleetNow()

	workers := plan.Workers
	if len(workers) == 0 {
		workers = discoverWorkers(*home)
	}

	// One janitor scan attributes stale children to worker roots. The monitor's
	// child-staleness flags must feed this scan; otherwise the classifier sees
	// stale children under default ceilings no matter what the operator passed.
	janitorPolicy := fleetmon.DefaultJanitorPolicy()
	janitorPolicy.SimpleShell = *staleSimple
	janitorPolicy.Test = *staleTest
	janitor := scanJanitor(procs, workers, janitorPolicy, now)
	staleByWorker := map[int][]fleetmon.ChildCommand{}
	for _, c := range janitor.Stale {
		staleByWorker[c.WorkerPID] = append(staleByWorker[c.WorkerPID], c)
	}

	var samples []fleetmon.WorkerSample
	newState := fleetState{Generated: now.UTC().Format(time.RFC3339), Workers: map[string]fleetStateRow{}}
	for _, w := range workers {
		tPath := resolveTranscriptPath(w, *home)
		sig := fleetmon.TranscriptSignal{}
		if tPath != "" {
			sig = fleetmon.ReadTranscript(tPath)
		}
		disp, action := registryMatch(reg, w)
		ev := fleetmon.WorkerEvidence{
			Issue:          w.Issue,
			Session:        w.Session,
			Account:        w.Account,
			RegistryDisp:   disp,
			RegistryAction: action,
			HasPID:         w.PID > 0,
			PID:            w.PID,
			PIDAlive:       w.PID > 0 && live[w.PID],
			Transcript:     sig,
			StaleChildren:  staleByWorker[w.PID],
		}
		if pr, ok := prev.Workers[w.Session]; ok {
			pl := pr.Lines
			ev.PrevLines = &pl
		}
		samples = append(samples, fleetmon.Classify(ev, now, th))
		newState.Workers[w.Session] = fleetStateRow{Lines: sig.Lines}
	}

	if *statePath != "" {
		saveFleetState(*statePath, newState)
	}

	payload := fleetmon.NewMonitorPayload(plan.RunID, samples, now)
	if collectErr != "" {
		fmt.Fprintf(stderr, "fak fleet monitor: process scan warning: %s\n", collectErr)
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, payload, "fak fleet monitor")
	}
	renderMonitorTable(stdout, payload)
	return 0
}

func renderMonitorTable(w io.Writer, p fleetmon.MonitorPayload) {
	fmt.Fprintf(w, "fleet monitor — %d worker(s) @ %s\n", p.Total, p.GeneratedAt)
	classes := make([]fleetmon.Classification, 0, len(p.ByClass))
	for c := range p.ByClass {
		classes = append(classes, c)
	}
	sort.Slice(classes, func(i, j int) bool { return classes[i] < classes[j] })
	var parts []string
	for _, c := range classes {
		parts = append(parts, fmt.Sprintf("%s=%d", c, p.ByClass[c]))
	}
	fmt.Fprintf(w, "  %s\n\n", strings.Join(parts, "  "))
	fmt.Fprintf(w, "%-6s %-20s %-20s %-6s %-10s %s\n", "ISSUE", "SESSION", "CLASS", "PID", "TX-AGE", "WHY")
	for _, s := range p.Workers {
		issue := ""
		if s.Issue != 0 {
			issue = "#" + itoaFleet(s.Issue)
		}
		age := "-"
		if s.TranscriptAgeSec != nil {
			age = (time.Duration(*s.TranscriptAgeSec) * time.Second).Round(time.Second).String()
		}
		pid := "-"
		if s.PID != 0 {
			pid = itoaFleet(s.PID)
			if !s.PIDAlive {
				pid += "✗"
			}
		}
		why := ""
		if len(s.Reasons) > 0 {
			why = s.Reasons[0]
		}
		fmt.Fprintf(w, "%-6s %-20s %-20s %-6s %-10s %s\n", issue, truncateFleet(s.Session, 20), s.Class, pid, age, why)
	}
}

// --- janitor (#1857) ------------------------------------------------------ //

func runFleetJanitor(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fleet janitor", flag.ContinueOnError)
	fs.SetOutput(stderr)
	planPath := fs.String("plan", "", "run plan JSON (worker roots to protect + attribute children to)")
	asJSON := fs.Bool("json", false, "emit JSON instead of a table")
	apply := fs.Bool("apply", false, "TERMINATE the stale child trees (default is a dry-run listing)")
	simple := fs.Duration("stale-simple", 5*time.Minute, "simple-shell staleness ceiling")
	test := fs.Duration("stale-test", 10*time.Minute, "test staleness ceiling")
	scan := fs.Duration("stale-scan", 5*time.Minute, "broad-scan staleness ceiling")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	plan, err := loadFleetPlan(*planPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak fleet janitor: %v\n", err)
		return 1
	}
	pol := fleetmon.DefaultJanitorPolicy()
	pol.SimpleShell, pol.Test, pol.BroadScan = *simple, *test, *scan

	procs, collectErr := fleetCollectRelations()
	if collectErr != "" {
		fmt.Fprintf(stderr, "fak fleet janitor: process scan warning: %s\n", collectErr)
	}
	now := fleetNow()
	result := scanJanitor(procs, plan.Workers, pol, now)

	type applied struct {
		RootPID int    `json:"root_pid"`
		Name    string `json:"name"`
		OK      bool   `json:"ok"`
		Detail  string `json:"detail"`
	}
	var reaped []applied
	if *apply {
		for _, c := range result.Stale {
			ok, detail := fleetKillPID(c.RootPID)
			reaped = append(reaped, applied{c.RootPID, c.Name, ok, detail})
		}
	}

	out := struct {
		fleetmon.JanitorResult
		Applied bool      `json:"applied"`
		Reaped  []applied `json:"reaped,omitempty"`
	}{JanitorResult: result, Applied: *apply, Reaped: reaped}

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, out, "fak fleet janitor")
	}
	fmt.Fprintf(stdout, "fleet janitor — scanned %d process(es); %d stale child tree(s), %d protected\n",
		result.Scanned, len(result.Stale), len(result.Protected))
	mode := "DRY-RUN (re-run with --apply to terminate)"
	if *apply {
		mode = "APPLIED"
	}
	fmt.Fprintf(stdout, "  mode: %s\n", mode)
	for _, c := range result.Stale {
		status := "would terminate"
		for _, r := range reaped {
			if r.RootPID == c.RootPID {
				if r.OK {
					status = "terminated"
				} else {
					status = "kill FAILED: " + r.Detail
				}
			}
		}
		fmt.Fprintf(stdout, "  - %s pid %d [%s] age %s (worker pid %d) → %s\n     %s\n",
			c.Name, c.RootPID, c.Class, (time.Duration(c.AgeSec) * time.Second).Round(time.Second), c.WorkerPID, status, truncateFleet(c.Command, 100))
	}
	if len(result.Stale) == 0 {
		fmt.Fprintf(stdout, "  %s\n", result.NextAction)
	}
	return 0
}

// scanJanitor builds the WorkerRoot list from a plan (+ process start times) and
// runs the pure evaluator.
func scanJanitor(procs []procguard.Proc, planWorkers []fleetmon.PlanWorker, pol fleetmon.JanitorPolicy, now time.Time) fleetmon.JanitorResult {
	startByPID := map[int]string{}
	for _, p := range procs {
		startByPID[p.PID] = p.Start
	}
	var roots []fleetmon.WorkerRoot
	for _, w := range planWorkers {
		if w.PID > 0 {
			roots = append(roots, fleetmon.WorkerRoot{PID: w.PID, Session: w.Session, Issue: w.Issue, Start: startByPID[w.PID]})
		}
	}
	return fleetmon.EvaluateJanitor(fleetmon.JanitorInput{Procs: procs, Workers: roots, Policy: pol, Now: now})
}

// --- fold (#1858) --------------------------------------------------------- //

func runFleetFold(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fleet fold", flag.ContinueOnError)
	fs.SetOutput(stderr)
	planPath := fs.String("plan", "", "run plan JSON (required to name the worker set)")
	asJSON := fs.Bool("json", false, "emit the JSON ledger instead of markdown")
	home := fs.String("home", fleetUserHome(), "Claude config home (for transcript discovery)")
	ledgerPath := fs.String("ledger", "", "append the folded rows to this JSONL ledger (with --write)")
	write := fs.Bool("write", false, "append the folded rows to --ledger (default is print-only)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	plan, err := loadFleetPlan(*planPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak fleet fold: %v\n", err)
		return 1
	}
	if len(plan.Workers) == 0 {
		fmt.Fprintln(stderr, "fak fleet fold: no workers (pass --plan with a worker set)")
		return 2
	}
	procs, collectErr := fleetCollectRelations()
	live := livePIDs(procs)
	now := fleetNow()
	if collectErr != "" {
		fmt.Fprintf(stderr, "fak fleet fold: process scan warning: %s\n", collectErr)
	}

	// Which issues have a replacement (a later session for the same issue)?
	replacedBy := map[string]string{} // original session -> replacement session
	for _, w := range plan.Workers {
		if w.ReplacementOf != "" {
			replacedBy[w.ReplacementOf] = w.Session
		}
	}

	var rows []fleetmon.LedgerRow
	for _, w := range plan.Workers {
		tPath := resolveTranscriptPath(w, *home)
		sig := fleetmon.TranscriptSignal{}
		if tPath != "" {
			sig = fleetmon.ReadTranscript(tPath)
		}
		in := fleetmon.FoldInput{RunID: plan.RunID, Worker: w, Transcript: sig, ProcessScanError: collectErr, Now: now}
		if w.PID > 0 && collectErr == "" {
			alive := live[w.PID]
			in.PIDAlive = &alive
		}
		if repl, ok := replacedBy[w.Session]; ok {
			in.Superseded, in.SupersededBy = true, repl
		}
		rows = append(rows, fleetmon.FoldWorker(in))
	}

	summary := fleetmon.Summarize(plan.RunID, rows)
	if *write && *ledgerPath != "" {
		if err := appendLedgerRows(*ledgerPath, rows); err != nil {
			fmt.Fprintf(stderr, "fak fleet fold: append ledger: %v\n", err)
			return 1
		}
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, summary, "fak fleet fold")
	}
	fmt.Fprint(stdout, renderFoldMarkdown(summary))
	return 0
}

func appendLedgerRows(path string, rows []fleetmon.LedgerRow) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	for _, r := range rows {
		line, err := fleetmon.AppendLedgerLine(r)
		if err != nil {
			return err
		}
		if _, err := f.WriteString(line + "\n"); err != nil {
			return err
		}
	}
	return nil
}

func renderFoldMarkdown(s fleetmon.RunLedgerSummary) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Fleet run ledger")
	if s.RunID != "" {
		fmt.Fprintf(&b, " — %s", s.RunID)
	}
	fmt.Fprintf(&b, "\n\n%d worker(s) folded.\n\n", s.Total)
	outcomes := make([]fleetmon.Outcome, 0, len(s.ByOutcome))
	for o := range s.ByOutcome {
		outcomes = append(outcomes, o)
	}
	sort.Slice(outcomes, func(i, j int) bool { return outcomes[i] < outcomes[j] })
	for _, o := range outcomes {
		fmt.Fprintf(&b, "- **%s**: %d\n", o, s.ByOutcome[o])
	}
	fmt.Fprintln(&b, "\n| Issue | Session | Outcome | Files | Witness | Follow-up |")
	fmt.Fprintln(&b, "|---|---|---|---:|---|---|")
	for _, r := range s.Rows {
		fmt.Fprintf(&b, "| #%d | %s | %s | %d | %s | %s |\n",
			r.Issue, truncateFleet(r.Session, 24), r.Outcome, len(r.ChangedFiles), truncateFleet(r.Witness, 40), truncateFleet(firstNonEmptyFleet(r.FollowUp, r.Blocker), 48))
	}
	if len(s.Defects) > 0 {
		fmt.Fprintf(&b, "\n## Ledger defects (%d)\n\n", len(s.Defects))
		for _, d := range s.Defects {
			fmt.Fprintf(&b, "- line %d (%s): %s\n", d.Line, d.Session, d.Reason)
		}
	}
	return b.String()
}

// --- replace (#1859) ------------------------------------------------------ //

func runFleetReplace(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fleet replace", flag.ContinueOnError)
	fs.SetOutput(stderr)
	planPath := fs.String("plan", "", "run plan JSON (to look up the failed worker)")
	session := fs.String("session", "", "the failed session id to replace (required)")
	class := fs.String("class", "", "monitor classification of the original (dead|auth-or-rate-blocked|stale-transcript|...)")
	index := fs.Int("index", 1, "replacement index (issue-<n>-replacement-<index>)")
	account := fs.String("account", "", "account/config bucket override")
	templatePath := fs.String("template", "", "corrected prompt template file (optional)")
	force := fs.Bool("force", false, "treat the original as explicitly unrecoverable (override the class gate)")
	asJSON := fs.Bool("json", false, "emit JSON instead of a human preview")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if strings.TrimSpace(*session) == "" {
		fmt.Fprintln(stderr, "fak fleet replace: --session is required")
		return 2
	}
	plan, err := loadFleetPlan(*planPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak fleet replace: %v\n", err)
		return 1
	}
	worker, ok := plan.WorkerBySession(*session)
	if !ok {
		// Allow replacing a session not in the plan by synthesizing a minimal worker.
		worker = fleetmon.PlanWorker{Session: *session, Account: *account}
	}
	tpl := ""
	if *templatePath != "" {
		data, err := os.ReadFile(*templatePath)
		if err != nil {
			fmt.Fprintf(stderr, "fak fleet replace: read template: %v\n", err)
			return 1
		}
		tpl = string(data)
	}

	decision := fleetmon.EvaluateReplace(fleetmon.ReplaceRequest{
		Worker:   worker,
		Class:    fleetmon.Classification(*class),
		Index:    *index,
		Account:  *account,
		Template: tpl,
		Force:    *force,
		RunID:    plan.RunID,
		Now:      fleetNow(),
	})

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, decision, "fak fleet replace")
	}
	if !decision.Eligible {
		fmt.Fprintf(stdout, "fleet replace — REFUSED\n  %s\n", decision.Reason)
		return 1
	}
	fmt.Fprintf(stdout, "fleet replace — eligible (%s)\n", decision.Reason)
	fmt.Fprintf(stdout, "  original session : %s\n", worker.Session)
	fmt.Fprintf(stdout, "  new session      : %s\n", decision.NewSession)
	if decision.Account != "" {
		fmt.Fprintf(stdout, "  account bucket   : %s\n", decision.Account)
	}
	fmt.Fprintf(stdout, "  ledger row       : %s superseded_by %s\n", worker.Session, decision.NewSession)
	fmt.Fprintln(stdout, "\n--- replacement prompt (preview) ---")
	fmt.Fprintln(stdout, decision.Prompt)
	return 0
}

// --- state persistence + small helpers ------------------------------------ //

func loadFleetState(path string) fleetState {
	st := fleetState{Workers: map[string]fleetStateRow{}}
	if path == "" {
		return st
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return st
	}
	_ = json.Unmarshal(data, &st)
	if st.Workers == nil {
		st.Workers = map[string]fleetStateRow{}
	}
	return st
}

func saveFleetState(path string, st fleetState) {
	if data, err := json.Marshal(st); err == nil {
		_ = os.WriteFile(path, data, 0o644)
	}
}

// discoverWorkers builds a best-effort worker list from recent transcripts under
// the workspace namespace when no run plan is provided. Issue/PID are unknown, so
// these classify from transcript evidence alone (typically completed-final,
// stale-transcript, or attention).
func discoverWorkers(home string) []fleetmon.PlanWorker {
	if home == "" {
		return nil
	}
	root := filepath.Join(home, ".claude", "projects")
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	var out []fleetmon.PlanWorker
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		files, _ := filepath.Glob(filepath.Join(root, e.Name(), "*.jsonl"))
		for _, f := range files {
			base := strings.TrimSuffix(filepath.Base(f), ".jsonl")
			out = append(out, fleetmon.PlanWorker{Session: base, Namespace: e.Name(), TranscriptPath: f})
		}
	}
	return out
}

func itoaFleet(n int) string { return fmt.Sprintf("%d", n) }

func truncateFleet(s string, n int) string {
	if len(s) <= n {
		return s
	}
	if n <= 1 {
		return s[:n]
	}
	return s[:n-1] + "…"
}

func firstNonEmptyFleet(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
