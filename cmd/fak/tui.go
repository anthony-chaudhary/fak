package main

// fak console is the native terminal control pane spine. The first surface is an
// issue queue view because issue triage is already one of fak's dogfood loops:
// fetch or load the GitHub issue shape, fold it into a ranked model, then render
// a compact terminal dashboard without adding a TUI dependency.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	acct "github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

const (
	tuiIssuesSchema   = "fak.tui.issues.v1"
	tuiLoopsSchema    = "fak.tui.loops.v1"
	tuiSessionsSchema = "fak.tui.sessions.v1"
	tuiGardenSchema   = "fak.tui.garden.v1"
	tuiGuardSchema    = "fak.tui.guard.v1"
	tuiAgentSchema    = "fak.tui.agent.v1"
	tuiOverviewSchema = "fak.tui.overview.v1"
)

var (
	tuiPriorityWeights = map[string]int{"priority/P0": 1000, "priority/P1": 400, "priority/P2": 150}
	tuiKindLabels      = map[string]bool{
		"bug": true, "enhancement": true, "documentation": true, "question": true,
		"performance": true, "build": true, "research": true,
	}
	tuiAreaLabels = map[string]bool{
		"agentic-serving": true, "trust-floor": true, "model-arch": true, "compute": true,
		"gpu": true, "model": true, "substrate": true, "loader": true, "security": true,
		"dispatch": true, "rsi": true, "licensing": true,
	}
	tuiWordRE  = regexp.MustCompile(`[A-Za-z0-9_-]{3,}`)
	tuiScopeRE = regexp.MustCompile(`\b(\w+)\(([^)]+)\)`)
)

func cmdTUI(argv []string) { os.Exit(runTUI(os.Stdout, os.Stderr, argv)) }

func runTUI(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		tuiUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "issues":
		return runTUIIssues(stdout, stderr, argv[1:])
	case "loops":
		return runTUILoops(stdout, stderr, argv[1:])
	case "sessions":
		return runTUISessions(stdout, stderr, argv[1:])
	case "garden":
		return runTUIGarden(stdout, stderr, argv[1:])
	case "guard":
		return runTUIGuard(stdout, stderr, argv[1:])
	case "agent":
		return runTUIAgent(stdout, stderr, argv[1:])
	case "overview":
		return runTUIOverview(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		tuiUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak console: unknown subcommand %q\n", argv[0])
		tuiUsage(stderr)
		return 2
	}
}

func runTUIIssues(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("tui issues", flag.ContinueOnError)
	fs.SetOutput(stderr)
	issuesJSON := fs.String("issues-json", "", "read gh issue JSON from a file instead of shelling out to gh")
	repo := fs.String("repo", "", "owner/repo for gh; default is current repo")
	state := fs.String("state", "open", "issue state for gh: open|closed|all")
	limit := fs.Int("limit", 500, "maximum issues to fetch from gh")
	asOfText := fs.String("as-of", "", "date used for age/idle math (YYYY-MM-DD, default: today UTC)")
	epic := fs.Int("epic", 0, "highlight one epic issue number and issues whose title/body references #N")
	top := fs.Int("top", 25, "number of ranked rows to render in human mode")
	width := fs.Int("width", 120, "target terminal width for human rendering")
	asJSON := fs.Bool("json", false, "emit the issue TUI model as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak console issues: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *limit <= 0 {
		fmt.Fprintln(stderr, "fak console issues: --limit must be positive")
		return 2
	}
	if *top <= 0 {
		fmt.Fprintln(stderr, "fak console issues: --top must be positive")
		return 2
	}
	if *width < 72 {
		*width = 72
	}
	asOf, err := parseTUIDay(*asOfText)
	if err != nil {
		fmt.Fprintf(stderr, "fak console issues: %v\n", err)
		return 2
	}

	issues, source, err := loadTUIIssues(*issuesJSON, *repo, *state, *limit)
	if err != nil {
		fmt.Fprintf(stderr, "fak console issues: %v\n", err)
		return 2
	}
	report := buildTUIIssueReport(issues, source, asOf, *epic)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak console issues")
	}
	fmt.Fprint(stdout, renderTUIIssues(report, *top, *width))
	return 0
}

func runTUILoops(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("tui loops", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", defaultLoopLedger(), "loop JSONL ledger path")
	atText := fs.String("at", "", "snapshot time (RFC3339 or YYYY-MM-DD, default: now)")
	top := fs.Int("top", 25, "number of loop rows to render in human mode")
	width := fs.Int("width", 120, "target terminal width for human rendering")
	asJSON := fs.Bool("json", false, "emit the loop TUI model as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak console loops: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *top <= 0 {
		fmt.Fprintln(stderr, "fak console loops: --top must be positive")
		return 2
	}
	if *width < 72 {
		*width = 72
	}
	at, err := parseTUITime(*atText)
	if err != nil {
		fmt.Fprintf(stderr, "fak console loops: %v\n", err)
		return 2
	}
	// Tolerant read: a forked/corrupt ledger (e.g. two processes that raced an
	// append) must not blank the pane — render the recovered prefix and surface the
	// break as a banner. A true I/O fault still fails; only a chain break degrades.
	st, integ, err := loopmgr.SnapshotFilePartial(*ledger, at)
	if err != nil {
		fmt.Fprintf(stderr, "fak console loops: %v\n", err)
		return 1
	}
	report := buildTUILoopReport(st, at)
	if integ.Broken {
		report.Integrity = &tuiLedgerIntegrity{
			Broken:    true,
			AtLine:    integ.AtLine,
			AtSeq:     integ.AtSeq,
			Reason:    integ.Reason,
			Recovered: integ.Recovered,
		}
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak console loops")
	}
	fmt.Fprint(stdout, renderTUILoops(report, *top, *width))
	return 0
}

func runTUISessions(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("tui sessions", flag.ContinueOnError)
	fs.SetOutput(stderr)
	sessionsJSON := fs.String("sessions-json", "", "read SessionListResponse JSON from a file instead of a live gateway")
	addr := fs.String("addr", defaultSessionAddr(), "gateway base URL")
	key := fs.String("key", defaultGatewayBearerToken(), "bearer credential (only if the gateway sets --require-key)")
	atText := fs.String("at", "", "snapshot time (RFC3339 or YYYY-MM-DD, default: now)")
	top := fs.Int("top", 25, "number of session rows to render in human mode")
	width := fs.Int("width", 120, "target terminal width for human rendering")
	asJSON := fs.Bool("json", false, "emit the session TUI model as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak console sessions: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *top <= 0 {
		fmt.Fprintln(stderr, "fak console sessions: --top must be positive")
		return 2
	}
	if *width < 72 {
		*width = 72
	}
	at, err := parseTUITime(*atText)
	if err != nil {
		fmt.Fprintf(stderr, "fak console sessions: %v\n", err)
		return 2
	}
	list, source, err := loadTUISessions(*sessionsJSON, *addr, *key)
	if err != nil {
		fmt.Fprintf(stderr, "fak console sessions: %v\n", err)
		return 1
	}
	report := buildTUISessionReport(list, source, at)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak console sessions")
	}
	fmt.Fprint(stdout, renderTUISessions(report, *top, *width))
	return 0
}

func runTUIGarden(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("tui garden", flag.ContinueOnError)
	fs.SetOutput(stderr)
	gardenJSON := fs.String("garden-json", "", "read fak garden JSON from a file instead of running the bundle")
	workspace := fs.String("workspace", "", "workspace root for a live bundle run (default: repo root)")
	deep := fs.Bool("deep", false, "include the slower loop-audit member on a live bundle run")
	timeout := fs.Int("timeout", 240, "per-member timeout seconds for a live bundle run")
	check := fs.Bool("check", false, "include the garden gate decision in the TUI model")
	atText := fs.String("at", "", "snapshot time (RFC3339 or YYYY-MM-DD, default: now)")
	width := fs.Int("width", 120, "target terminal width for human rendering")
	asJSON := fs.Bool("json", false, "emit the garden TUI model as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak console garden: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *timeout <= 0 {
		fmt.Fprintln(stderr, "fak console garden: --timeout must be positive")
		return 2
	}
	if *width < 72 {
		*width = 72
	}
	at, err := parseTUITime(*atText)
	if err != nil {
		fmt.Fprintf(stderr, "fak console garden: %v\n", err)
		return 2
	}
	payload, source, err := loadTUIGarden(*gardenJSON, *workspace, *deep, time.Duration(*timeout)*time.Second)
	if err != nil {
		fmt.Fprintf(stderr, "fak console garden: %v\n", err)
		return 1
	}
	report := buildTUIGardenReport(payload, source, at, *check)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak console garden")
	}
	fmt.Fprint(stdout, renderTUIGarden(report, *width))
	return 0
}

func runTUIGuard(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("tui guard", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var guardJSON stringList
	fs.Var(&guardJSON, "guard-json", "read a guard artifact JSON file (repeatable)")
	journalPath := fs.String("journal", "", "tail the durable, hash-chained guard DECISION JOURNAL at this path instead of static --guard-json artifacts (#843): each adjudication row is folded through the same guard model, redaction-safe (the journal carries no payloads, only digests)")
	tail := fs.Bool("tail", false, "tail the CANONICAL guard journal (FAK_AUDIT_JOURNAL, else newest .dispatch-runs/guard-audit/*.jsonl) — equivalent to --journal <canonical-path>")
	follow := fs.Bool("follow", false, "with --journal/--tail: keep following the journal and print each NEW adjudication row as it lands (Ctrl-C to stop)")
	maxRows := fs.Int("rows", 50, "cap the number of (highest-attention) journal rows rendered in the pane")
	atText := fs.String("at", "", "snapshot time (RFC3339 or YYYY-MM-DD, default: now)")
	width := fs.Int("width", 120, "target terminal width for human rendering")
	asJSON := fs.Bool("json", false, "emit the guard TUI model as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak console guard: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *width < 72 {
		*width = 72
	}
	at, err := parseTUITime(*atText)
	if err != nil {
		fmt.Fprintf(stderr, "fak console guard: %v\n", err)
		return 2
	}

	// Live guard-journal mode (#843): tail the canonical hash-chained guard decision
	// journal and render its denial surface through the SAME guard model, or follow it
	// live. Selected by --journal/--tail; otherwise the static --guard-json pane runs.
	useJournal := *journalPath != "" || *tail
	if useJournal && len(guardJSON) > 0 {
		fmt.Fprintln(stderr, "fak console guard: pass EITHER --guard-json artifacts OR --journal/--tail, not both")
		return 2
	}
	if useJournal {
		path := *journalPath
		if path == "" {
			path = canonicalGuardJournalPath()
		}
		if path == "" {
			fmt.Fprintln(stderr, "fak console guard: --tail could not resolve a canonical guard journal path (set FAK_AUDIT_JOURNAL or pass --journal PATH)")
			return 2
		}
		return runTUIGuardJournal(stdout, stderr, path, at, *width, *maxRows, *asJSON, *follow)
	}
	if len(guardJSON) == 0 {
		fmt.Fprintln(stderr, "fak console guard: at least one --guard-json artifact (or --journal/--tail) is required")
		return 2
	}
	artifacts, err := loadTUIGuard([]string(guardJSON))
	if err != nil {
		fmt.Fprintf(stderr, "fak console guard: %v\n", err)
		return 1
	}
	report := buildTUIGuardReport(artifacts, at)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak console guard")
	}
	fmt.Fprint(stdout, renderTUIGuard(report, *width))
	return 0
}

// runTUIGuardJournal renders the live guard-journal pane (#843): it reads the durable
// hash-chained guard decision journal at path, folds its adjudication rows through the
// SAME guard model (scoreTUIGuardRow / countTUIGuard / tuiGuardActions) the static
// artifact pane uses, and renders the report (or JSON). A missing/empty journal yields
// a well-formed empty pane, not an error — a not-yet-written journal is a valid "no
// adjudications yet" state. With follow, it then tails the journal, printing each new
// row as it lands until interrupted. Redaction is preserved by construction: the
// journal carries only decision fields + content digests, never a prompt/arg/result
// payload, so nothing sensitive can reach the model.
func runTUIGuardJournal(stdout, stderr io.Writer, path string, at time.Time, width, maxRows int, asJSON, follow bool) int {
	rows, err := journal.ReadRows(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak console guard: %v\n", err)
		return 1
	}
	report := buildTUIGuardJournalReport(rows, path, at, maxRows)
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak console guard")
	}
	fmt.Fprint(stdout, renderTUIGuard(report, width))
	if follow {
		return followGuardJournal(stdout, path, width, lastSeqOf(rows))
	}
	return 0
}

// buildTUIGuardJournalReport folds journal rows into the guard report model. Each row
// becomes one tuiGuardRow scored by the committed scorer, so DENY / POLICY_BLOCK /
// DEFAULT_DENY / QUARANTINE rows rise to the top of the attention sort and the counts
// line surfaces the denial surface. Counts are computed over ALL rows (an honest
// total); only the rendered table is capped to maxRows (the highest-attention ones).
func buildTUIGuardJournalReport(rows []journal.Row, path string, at time.Time, maxRows int) tuiGuardReport {
	name := tuiGuardArtifactName(path)
	if name == "" {
		name = "guard-audit"
	}
	guardRows := make([]tuiGuardRow, 0, len(rows))
	for _, r := range rows {
		guardRows = append(guardRows, tuiGuardRow{
			Artifact: name,
			Kind:     "audit-" + strings.ToLower(r.Kind),
			Tool:     r.Tool,
			Verdict:  strings.ToUpper(r.Verdict),
			Reason:   strings.ToUpper(r.Reason),
			By:       r.By,
			Detail:   tuiGuardJournalDetail(r),
			Count:    1,
		})
	}
	for i := range guardRows {
		guardRows[i].Tags, guardRows[i].Attention = scoreTUIGuardRow(guardRows[i])
	}
	sort.SliceStable(guardRows, func(i, j int) bool {
		if guardRows[i].Attention != guardRows[j].Attention {
			return guardRows[i].Attention > guardRows[j].Attention
		}
		if guardRows[i].Kind != guardRows[j].Kind {
			return guardRows[i].Kind < guardRows[j].Kind
		}
		return guardRows[i].Tool < guardRows[j].Tool
	})
	sources := []tuiGuardSource{{Path: path, Schema: "fak-guard-audit-journal/1"}}
	counts := countTUIGuard(guardRows, sources)
	if maxRows > 0 && len(guardRows) > maxRows {
		guardRows = guardRows[:maxRows]
	}
	status := tuiGuardStatus(counts)
	return tuiGuardReport{
		Schema:  tuiGuardSchema,
		At:      at.UTC().Format(time.RFC3339),
		Source:  name,
		Status:  status,
		Counts:  counts,
		Actions: tuiGuardActions(counts),
		Rows:    guardRows,
		Sources: sources,
	}
}

// tuiGuardJournalDetail builds the per-row detail from the journal's bounded-disclosure
// fields ONLY (the witness claim that names which glob/arg tripped the deny, plus the
// trace id) — never a payload. It is the redaction-safe "why" for an audited decision.
func tuiGuardJournalDetail(r journal.Row) string {
	return strings.TrimSpace(strings.Join(nonEmptyTUI([]string{r.Witness, r.TraceID}), "  "))
}

// canonicalGuardJournalPath resolves the canonical guard decision journal: the
// documented FAK_AUDIT_JOURNAL override, else the newest repo-local guard journal.
func canonicalGuardJournalPath() string {
	return guardReadableAuditPath()
}

// followGuardJournal tails the journal after the initial snapshot, printing each NEW
// adjudication row (seq beyond lastSeq) as a compact one-line entry as it lands. It
// polls (no fsnotify dependency, matching the rest of the kernel) and stops on Ctrl-C.
func followGuardJournal(stdout io.Writer, path string, width int, lastSeq uint64) int {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return 0
		case <-ticker.C:
			rows, err := journal.ReadRows(path)
			if err != nil {
				continue // transient (mid-rotate / I/O blip): keep tailing
			}
			for _, r := range rows {
				if r.Seq <= lastSeq {
					continue
				}
				lastSeq = r.Seq
				fmt.Fprintln(stdout, formatGuardJournalLine(r, width))
			}
		}
	}
}

// formatGuardJournalLine renders one journal row as a compact tail line — decision
// fields + the witness claim only, never a payload (the #840 redaction contract).
func formatGuardJournalLine(r journal.Row, width int) string {
	parts := []string{fmt.Sprintf("seq=%d", r.Seq), r.Kind}
	for _, s := range []string{r.Tool, r.Verdict, r.Reason} {
		if s != "" {
			parts = append(parts, s)
		}
	}
	if r.Witness != "" {
		parts = append(parts, "("+r.Witness+")")
	}
	return trimTUI(strings.Join(parts, "  "), maxTUI(40, width))
}

// lastSeqOf returns the highest seq in a row slice (0 for none) — the follow watermark.
func lastSeqOf(rows []journal.Row) uint64 {
	var m uint64
	for _, r := range rows {
		if r.Seq > m {
			m = r.Seq
		}
	}
	return m
}

// resolveAutoTarget applies the #939 `--auto` policy: when auto is set it rejects
// a conflicting explicit target/--gateway-url, ranks the registered compute
// targets (healthy first, then cheapest/most-local) and returns the winner's name
// so the caller's normal target-resolution path takes over. It returns the
// (possibly unchanged) selected target, whether auto picked it, and a (code, done)
// pair: done=true means the caller should return code immediately — either because
// a usage/load error fired or because `--auto --json` emitted the ranked decision
// instead of launching. When auto is off it is a pass-through (done=false).
func resolveAutoTarget(auto bool, selectedTarget string, setFlags map[string]bool, regErr error, reg *targetRegistry, asJSON bool, stdout, stderr io.Writer) (target string, autoSelected bool, code int, done bool) {
	if !auto {
		return selectedTarget, false, 0, false
	}
	if selectedTarget != "" {
		fmt.Fprintf(stderr, "fak console agent: --auto selects a target automatically; do not also pass a target (%q)\n", selectedTarget)
		return selectedTarget, false, 2, true
	}
	if setFlags["gateway-url"] {
		fmt.Fprintln(stderr, "fak console agent: --auto ranks the registered targets; it cannot combine with an explicit --gateway-url")
		return selectedTarget, false, 2, true
	}
	if regErr != nil {
		fmt.Fprintf(stderr, "fak console agent: --auto: load compute targets: %v\n", regErr)
		return selectedTarget, false, 1, true
	}
	hc := &http.Client{Timeout: 3 * time.Second}
	decision, winner, autoErr := autoSelectComputeTarget(context.Background(), reg, hc, 3*time.Second)
	if asJSON {
		// --auto --json emits the ranked decision (not a launch plan) and does not launch.
		if encErr := writeIndentedJSON(stdout, decision); encErr != nil {
			fmt.Fprintf(stderr, "fak console agent: encode json: %v\n", encErr)
			return selectedTarget, false, 1, true
		}
		if autoErr != nil {
			return selectedTarget, false, 1, true
		}
		return selectedTarget, false, 0, true
	}
	// Log the ranked decision so the operator sees WHY the winner won (or why nothing did).
	renderAutoDecision(stderr, decision)
	if autoErr != nil {
		fmt.Fprintf(stderr, "fak console agent: --auto: %v\n", autoErr)
		return selectedTarget, false, 1, true
	}
	return winner.Name, true, 0, false
}

func runTUIAgent(stdout, stderr io.Writer, argv []string) int {
	// #938: a leading non-flag token may NAME a compute target (`fak c mac`). Resolve
	// it against the registry BEFORE flag parsing, because Go's flag package stops at
	// the first positional — a leading token would otherwise swallow every trailing
	// flag. A KNOWN target is stripped here and applied below; an UNKNOWN leading token
	// is left in place so it still forwards to `claude` verbatim (back-compat: the
	// `fak c mac`→`claude mac` footgun only changes once `mac` is a real target).
	reg, regErr := loadComputeTargets(defaultComputeTargetsFile())
	var leadingTarget string
	if regErr == nil {
		leadingTarget, argv = resolveLeadingTarget(argv, reg, stderr)
	}

	fs := flag.NewFlagSet("tui agent", flag.ContinueOnError)
	fs.SetOutput(stderr)
	defHome, _ := os.UserHomeDir()
	regDefault := os.Getenv("FAK_ACCOUNTS_REGISTRY")
	if regDefault == "" && defHome != "" {
		regDefault = filepath.Join(defHome, ".claude-accounts", "registry.json")
	}
	backend := fs.String("backend", "claude", "backend agent to launch (currently: claude)")
	command := fs.String("command", "claude", "Claude Code command or path to execute")
	account := fs.String("account", "", "Claude config-home account name from `fak accounts`")
	claudeConfigDir := fs.String("claude-config-dir", "", "explicit Claude config directory to pass as CLAUDE_CONFIG_DIR")
	registry := fs.String("registry", regDefault, "path to the fak accounts registry.json")
	home := fs.String("home", defHome, "home dir used when discovering Claude account homes")
	prompt := fs.String("prompt", "", "append `claude -p PROMPT` for a non-interactive backend run")
	permissionMode := fs.String("permission-mode", "bypassPermissions", "Claude --permission-mode for every spawned account session (default bypassPermissions so the guarded backend, not Claude's own prompt, mediates tools); pass \"\" to omit, or override it in the trailing `claude args`")
	policyPath := fs.String("policy", "", "capability-floor manifest for the guard child (default: built-in guard floor)")
	model := fs.String("model", "", "upstream Claude model override for the guard child")
	sessionID := fs.String("session-id", "tui-agent", "trace/session id for the guard session")
	contextBudget := fs.Int("context-budget-tokens", 0, "seed a context-token budget in the guard session")
	restartOnBudget := fs.Bool("restart-on-budget", false, "ask guard to relaunch Claude on context-budget exhaustion")
	restartLimit := fs.Int("restart-limit", 0, "maximum guard relaunches for --restart-on-budget; 0 means unlimited")
	passthrough := fs.Bool("passthrough", false, "do not force subscription OAuth; let Claude Code forward its own credential")
	gatewayURL := fs.String("gateway-url", "", "existing fak serve gateway to use instead of starting a local guard, e.g. http://node:8080")
	gatewayKeyEnv := fs.String("gateway-key-env", "FAK_GATEWAY_KEY", "env var holding the existing gateway bearer for --gateway-url")
	apiTimeoutMS := fs.Int("api-timeout-ms", 1800000, "API_TIMEOUT_MS for --gateway-url launches; 0 leaves it inherited")
	debugStats := fs.Bool("debug-stats", true, "print one compact per-turn line to stderr that leads with a verdict (ok/warming/degraded/cold) + the NET write-premium-aware token saving, then cache health and compaction; wired to fak guard --debug-stats")
	atText := fs.String("at", "", "snapshot time (RFC3339 or YYYY-MM-DD, default: now)")
	width := fs.Int("width", 120, "target terminal width for dry-run human rendering")
	dryRun := fs.Bool("dry-run", false, "render the launch plan without starting the backend agent")
	asJSON := fs.Bool("json", false, "emit the launch model as JSON and do not start the backend agent")
	listTargets := fs.Bool("list-targets", false, "list the named compute targets (mac/gcp/local/anthropic + ~/.fak/targets.json) with a live /healthz column, then exit")
	targetFlag := fs.String("target", "", "named compute target to chat against (mac/gcp/local/anthropic + ~/.fak/targets.json); the explicit form of the leading `fak c <target>` token")
	auto := fs.Bool("auto", false, "health/cost/quota-aware automatic compute-target selection with failover (#939): probe every registered target, rank by the documented policy (healthy first, then cheapest/most-local), and launch the best one; --json emits the ranked decision instead of launching")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *listTargets {
		return runListComputeTargets(stdout, stderr, *asJSON)
	}
	// Which flags did the user set explicitly? A resolved target fills in only the
	// fields the user did NOT pass, so `fak c mac --model foo` keeps foo.
	setFlags := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { setFlags[f.Name] = true })
	// The explicit --target flag and the leading positional token must agree; pick the
	// one that is set (flag value wins if they are equal, errors if they conflict).
	selectedTarget := strings.TrimSpace(*targetFlag)
	if selectedTarget != "" && leadingTarget != "" && !strings.EqualFold(selectedTarget, leadingTarget) {
		fmt.Fprintf(stderr, "fak console agent: conflicting target: positional %q vs --target %q (pass one)\n", leadingTarget, selectedTarget)
		return 2
	}
	if selectedTarget == "" {
		selectedTarget = leadingTarget
	}
	// #939: --auto ranks the registered targets (healthy first, then cheapest/most-local)
	// and selects the winner, which then flows through the same resolution path as an
	// explicit target below. It is mutually exclusive with a named target or --gateway-url.
	selectedTarget, autoSelected, code, done := resolveAutoTarget(
		*auto, selectedTarget, setFlags, regErr, reg, *asJSON, stdout, stderr)
	if done {
		return code
	}
	if *width < 72 {
		*width = 72
	}
	if *contextBudget < 0 {
		fmt.Fprintln(stderr, "fak console agent: --context-budget-tokens must be non-negative")
		return 2
	}
	if *restartLimit < 0 {
		fmt.Fprintln(stderr, "fak console agent: --restart-limit must be non-negative")
		return 2
	}
	if *restartOnBudget && *contextBudget <= 0 {
		fmt.Fprintln(stderr, "fak console agent: --restart-on-budget requires --context-budget-tokens N")
		return 2
	}
	if *apiTimeoutMS < 0 {
		fmt.Fprintln(stderr, "fak console agent: --api-timeout-ms must be non-negative")
		return 2
	}
	at, err := parseTUITime(*atText)
	if err != nil {
		fmt.Fprintf(stderr, "fak console agent: %v\n", err)
		return 2
	}
	opts := tuiAgentOptions{
		Backend:             *backend,
		Command:             *command,
		CommandArgs:         fs.Args(),
		Prompt:              *prompt,
		PermissionMode:      *permissionMode,
		Account:             *account,
		ClaudeConfigDir:     *claudeConfigDir,
		Registry:            *registry,
		Home:                *home,
		Policy:              *policyPath,
		Model:               *model,
		SessionID:           *sessionID,
		ContextBudgetTokens: *contextBudget,
		RestartOnBudget:     *restartOnBudget,
		RestartLimit:        *restartLimit,
		Passthrough:         *passthrough,
		GatewayURL:          *gatewayURL,
		GatewayKeyEnv:       *gatewayKeyEnv,
		APITimeoutMS:        *apiTimeoutMS,
		DebugStats:          *debugStats,
	}
	// #938: fold a resolved compute target into the launch options. A positional that
	// reached here is always a known target; an unknown --target value is an error
	// (unlike an unknown positional, which already passed through to claude above).
	var resolvedTarget *computeTarget
	if selectedTarget != "" {
		if regErr != nil {
			fmt.Fprintf(stderr, "fak console agent: load compute targets: %v\n", regErr)
			return 1
		}
		tgt, ok := reg.resolve(selectedTarget)
		if !ok {
			fmt.Fprintf(stderr, "fak console agent: unknown --target %q (see `fak c --list-targets`)\n", selectedTarget)
			if hint := reg.nearest(selectedTarget); hint != "" {
				fmt.Fprintf(stderr, "  did you mean %q?\n", hint)
			}
			return 2
		}
		applyComputeTarget(&opts, tgt, setFlags)
		resolvedTarget = &tgt
		// A target can be /healthz-up yet un-launchable because its bearer env var is
		// unset (the mac gateway's healthz is unauthenticated). Fail here with a
		// target-named, actionable message instead of the generic downstream
		// "--gateway-url requires FAK_GATEWAY_KEY to be set" — this covers both explicit
		// `fak c mac` and an `--auto` winner that resolved to a credential-gated target.
		if msg, missing := computeTargetCredMissing(tgt, opts.GatewayKeyEnv, os.Getenv); missing {
			fmt.Fprintln(stderr, msg)
			return 2
		}
	}
	report, err := buildTUIAgentReport(opts, at, tuiExecutable(), os.Getenv)
	if err != nil {
		fmt.Fprintf(stderr, "fak console agent: %v\n", err)
		return 2
	}
	report.Target = selectedTarget
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak console agent")
	}
	if *dryRun {
		fmt.Fprint(stdout, renderTUIAgent(report, *width))
		return 0
	}
	// #938: gate an interactive launch against a resolved remote target on a reachable
	// gateway — mirror the claude-mac-fak preflight so `fak c mac/gcp/local` never hands
	// the terminal to Claude against a dead/mock backend. A target with no /healthz (the
	// real Anthropic API) is n/a and never blocks. --auto already probed every target and
	// picked a healthy winner, so it skips this redundant second probe.
	if resolvedTarget != nil && !autoSelected {
		if code, gated := preflightComputeTarget(stdout, stderr, *resolvedTarget); gated {
			return code
		}
	}
	return launchTUIAgent(stdout, stderr, report)
}

// resolveLeadingTarget peeks at the first arg. When it is a non-flag token that names a
// registered compute target, it returns that name and the remaining args (the token
// stripped) so the rest can be flag-parsed normally. An unknown token is left in place —
// back-compat: it forwards to `claude` exactly as today — with a one-line "did you mean"
// hint to stderr only when the token is lexically close to a real target name (#938).
func resolveLeadingTarget(argv []string, reg *targetRegistry, stderr io.Writer) (string, []string) {
	if len(argv) == 0 || reg == nil {
		return "", argv
	}
	tok := argv[0]
	if tok == "" || strings.HasPrefix(tok, "-") {
		return "", argv // a flag (or empty) — never a leading target token
	}
	if _, ok := reg.resolve(tok); ok {
		return tok, argv[1:]
	}
	if hint := reg.nearest(tok); hint != "" {
		fmt.Fprintf(stderr, "fak console agent: %q is not a known compute target (did you mean %q? — `fak c --list-targets`); forwarding it to claude\n", tok, hint)
	}
	return "", argv
}

// computeTargetCredMissing reports whether a resolved gateway-routed target declares a
// credential env var that is empty in the environment, and returns an actionable,
// target-named message. It exists because a target selected purely on a live /healthz
// (which is unauthenticated) can still be un-launchable when its bearer env var is unset:
// today that surfaces only as the generic downstream "--gateway-url requires FAK_GATEWAY_KEY
// to be set", which names neither the target nor how to supply the key. It fires ONLY for a
// REMOTE gateway-url/local-spawn target with a non-empty effective key env whose value is
// unset — a loopback local serve and the anthropic provider-proxy (OAuth via guard, empty
// GatewayURL) are exempt, matching buildTUIAgentGatewayReport's own bearer tolerance.
func computeTargetCredMissing(tgt computeTarget, keyEnv string, getenv func(string) string) (string, bool) {
	if tgt.Kind != targetGatewayURL && tgt.Kind != targetLocalSpawn {
		return "", false
	}
	if gatewayIsLocal(tgt.GatewayURL) {
		return "", false
	}
	env := strings.TrimSpace(keyEnv)
	if env == "" {
		env = strings.TrimSpace(tgt.CredEnv)
	}
	if env == "" || strings.TrimSpace(getenv(env)) != "" {
		return "", false // target needs no bearer, or the bearer is present
	}
	var b strings.Builder
	fmt.Fprintf(&b, "fak console agent: target %q is reachable but its gateway credential $%s is empty.\n", tgt.Name, env)
	fmt.Fprintf(&b, "  export it (e.g. %s=$(ssh <gateway-host> 'cat ~/.fak-gateway-key')), or pick another target (`fak c --list-targets`)", env)
	if strings.EqualFold(tgt.Name, "mac") {
		fmt.Fprint(&b, ";\n  or run `fak claude-mac-fak`, which fetches the Mac bearer over SSH for you")
	}
	return b.String(), true
}

// applyComputeTarget folds a resolved target into the launch options WITHOUT clobbering
// any flag the user set explicitly (setFlags wins), so `fak c mac --model foo` keeps foo.
// A gateway-url / local-spawn target routes the launch through the existing --gateway-url
// path (gateway + model + the cred env-var NAME); the anthropic provider-proxy target IS
// the default guard path, so it leaves GatewayURL empty and carries only its model.
func applyComputeTarget(opt *tuiAgentOptions, tgt computeTarget, setFlags map[string]bool) {
	switch tgt.Kind {
	case targetGatewayURL, targetLocalSpawn:
		if !setFlags["gateway-url"] {
			opt.GatewayURL = tgt.GatewayURL
		}
		if !setFlags["model"] && tgt.Model != "" {
			opt.Model = tgt.Model
		}
		if !setFlags["gateway-key-env"] && tgt.CredEnv != "" {
			opt.GatewayKeyEnv = tgt.CredEnv
		}
	case targetProviderProxy:
		// The default guard path (provider anthropic, subscription OAuth): leave
		// GatewayURL empty so buildTUIAgentReport takes the guard branch, and carry
		// only the named model.
		if !setFlags["model"] && tgt.Model != "" {
			opt.Model = tgt.Model
		}
	}
}

// preflightComputeTarget probes a resolved target's /healthz before an interactive launch
// and gates a launch against a dead gateway (#938), reusing the registry probe. A target
// with no /healthz endpoint (the real Anthropic API) is n/a and never blocks. It returns
// gated=true with an exit code ONLY when the launch must be aborted.
func preflightComputeTarget(stdout, stderr io.Writer, tgt computeTarget) (int, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	health := tgt.probe(ctx, &http.Client{Timeout: 3 * time.Second})
	switch health.State {
	case "down":
		fmt.Fprintf(stderr, "fak console agent: target %q gateway is unreachable: %s\n", tgt.Name, health.Detail)
		fmt.Fprintf(stderr, "  not launching claude against a dead backend — check the gateway, or pick another target (`fak c --list-targets`)\n")
		return 1, true
	case "up":
		fmt.Fprintf(stdout, "fak console agent: target %q gateway is up — launching claude ...\n", tgt.Name)
	}
	// "up" and "n/a" both proceed; "n/a" (no /healthz, e.g. anthropic) prints nothing.
	return 0, false
}

func runTUIOverview(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("tui overview", flag.ContinueOnError)
	fs.SetOutput(stderr)
	issuesJSON := fs.String("issues-json", "", "read gh issue JSON and include the issue pane card")
	epic := fs.Int("epic", 0, "highlight one epic issue number for the issue pane card")
	ledger := fs.String("ledger", "", "read loop JSONL ledger and include the loop pane card")
	sessionsJSON := fs.String("sessions-json", "", "read SessionListResponse JSON and include the session pane card")
	gardenJSON := fs.String("garden-json", "", "read fak garden JSON and include the garden pane card")
	var guardJSON stringList
	fs.Var(&guardJSON, "guard-json", "read a guard artifact JSON file and include the guard pane card (repeatable)")
	check := fs.Bool("check", false, "include the garden gate decision when --garden-json is set")
	asOfText := fs.String("as-of", "", "date used for issue age/idle math (YYYY-MM-DD, default: today UTC)")
	atText := fs.String("at", "", "snapshot time for non-issue panes (RFC3339 or YYYY-MM-DD, default: now)")
	width := fs.Int("width", 120, "target terminal width for human rendering")
	asJSON := fs.Bool("json", false, "emit the overview TUI model as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak console overview: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *width < 72 {
		*width = 72
	}
	asOf, err := parseTUIDay(*asOfText)
	if err != nil {
		fmt.Fprintf(stderr, "fak console overview: %v\n", err)
		return 2
	}
	at, err := parseTUITime(*atText)
	if err != nil {
		fmt.Fprintf(stderr, "fak console overview: %v\n", err)
		return 2
	}
	report, err := loadTUIOverview(tuiOverviewOptions{
		IssuesJSON:   *issuesJSON,
		Epic:         *epic,
		Ledger:       *ledger,
		SessionsJSON: *sessionsJSON,
		GardenJSON:   *gardenJSON,
		GuardJSON:    []string(guardJSON),
		CheckGarden:  *check,
		AsOf:         asOf,
		At:           at,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak console overview: %v\n", err)
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak console overview")
	}
	fmt.Fprint(stdout, renderTUIOverview(report, *width))
	return 0
}

type tuiOverviewOptions struct {
	IssuesJSON   string
	Epic         int
	Ledger       string
	SessionsJSON string
	GardenJSON   string
	GuardJSON    []string
	CheckGarden  bool
	AsOf         time.Time
	At           time.Time
}

func parseTUIDay(s string) (time.Time, error) {
	if strings.TrimSpace(s) == "" {
		now := time.Now().UTC()
		return time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.UTC), nil
	}
	t, err := time.Parse("2006-01-02", strings.TrimSpace(s))
	if err != nil {
		return time.Time{}, fmt.Errorf("--as-of must be YYYY-MM-DD")
	}
	return t, nil
}

func parseTUITime(s string) (time.Time, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Now().UTC(), nil
	}
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t.UTC(), nil
	}
	if t, err := time.Parse("2006-01-02", s); err == nil {
		return t.UTC(), nil
	}
	return time.Time{}, fmt.Errorf("--at must be RFC3339 or YYYY-MM-DD")
}

// claudeArgsHavePermissionMode reports whether the operator already set a
// --permission-mode in the trailing `claude args`, in which case the default
// bypassPermissions must not be injected (Claude rejects a duplicated flag).
// It matches both the split form (`--permission-mode X`) and the joined form
// (`--permission-mode=X`).
func claudeArgsHavePermissionMode(args []string) bool {
	for _, a := range args {
		if a == "--permission-mode" || strings.HasPrefix(a, "--permission-mode=") {
			return true
		}
	}
	return false
}

func buildTUIAgentReport(opt tuiAgentOptions, at time.Time, fakPath string, getenv func(string) string) (tuiAgentReport, error) {
	backend := strings.ToLower(strings.TrimSpace(opt.Backend))
	if backend == "" {
		backend = "claude"
	}
	if backend != "claude" {
		return tuiAgentReport{}, fmt.Errorf("unknown --backend %q (want claude)", opt.Backend)
	}
	commandName := strings.TrimSpace(opt.Command)
	if commandName == "" {
		return tuiAgentReport{}, fmt.Errorf("--command must not be empty")
	}
	sessionID := strings.TrimSpace(opt.SessionID)
	if sessionID == "" {
		sessionID = "tui-agent"
	}
	if strings.TrimSpace(opt.Account) != "" && strings.TrimSpace(opt.ClaudeConfigDir) != "" {
		return tuiAgentReport{}, fmt.Errorf("--account and --claude-config-dir are mutually exclusive")
	}
	if fakPath == "" {
		fakPath = "fak"
	}

	// Every spawned account session defaults to Claude's --permission-mode
	// bypassPermissions: the launch is already wrapped by `fak guard` (or pinned
	// at a fak serve gateway), so the reference monitor — not Claude's own
	// interactive permission prompt — mediates tool calls. Forcing it here means
	// ALL accounts spawn non-interactively-gated by default. An operator override
	// in the trailing `claude args` wins (we don't duplicate the flag), and
	// --permission-mode "" opts out entirely.
	permissionMode := strings.TrimSpace(opt.PermissionMode)
	if permissionMode != "" && claudeArgsHavePermissionMode(opt.CommandArgs) {
		permissionMode = "" // operator already set it in the passthrough args
	}
	command := []string{commandName}
	if permissionMode != "" {
		command = append(command, "--permission-mode", permissionMode)
	}
	command = append(command, opt.CommandArgs...)
	if strings.TrimSpace(opt.Prompt) != "" {
		command = append(command, "-p", opt.Prompt)
	}

	env, cfgDir, cfgSource, resolvedAccount, identity, notes, err := resolveTUIAgentClaudeConfig(opt, getenv)
	if err != nil {
		return tuiAgentReport{}, err
	}
	if permissionMode != "" {
		notes = append(notes, fmt.Sprintf("permission-mode=%s: every spawned account session is launched with this Claude --permission-mode by default, so the guarded backend mediates tools instead of Claude's interactive prompt (override in the trailing claude args, or pass --permission-mode \"\" to omit)", permissionMode))
	}
	if strings.TrimSpace(opt.GatewayURL) != "" {
		return buildTUIAgentGatewayReport(opt, at, backend, command, permissionMode, env, cfgDir, cfgSource, resolvedAccount, identity, notes, getenv)
	}
	guardArgs := []string{"guard", "--provider", "anthropic", "--session-id", sessionID}
	auth := "claude-subscription-oauth"
	if opt.Passthrough {
		auth = "passthrough"
		notes = append(notes, "Claude Code forwards its own credential through the gateway")
	} else {
		guardArgs = append(guardArgs, "--anthropic-oauth")
		notes = append(notes, "guard forces the Claude Pro/Max subscription OAuth path and fails loud if no token is available")
	}
	if strings.TrimSpace(opt.Policy) != "" {
		guardArgs = append(guardArgs, "--policy", strings.TrimSpace(opt.Policy))
	}
	if strings.TrimSpace(opt.Model) != "" {
		guardArgs = append(guardArgs, "--model", strings.TrimSpace(opt.Model))
	}
	if opt.ContextBudgetTokens > 0 {
		guardArgs = append(guardArgs, "--context-budget-tokens", strconv.Itoa(opt.ContextBudgetTokens))
	}
	if opt.RestartOnBudget {
		guardArgs = append(guardArgs, "--restart-on-budget")
	}
	if opt.RestartLimit > 0 {
		guardArgs = append(guardArgs, "--restart-limit", strconv.Itoa(opt.RestartLimit))
	}
	// Token-saving defaults: compact-history-budget and elide-result-bytes are already
	// default-on in guard, but we pass them explicitly so they appear in dry-run output
	// and the operator can see the active savings without reading guard's defaults.
	guardArgs = append(guardArgs,
		"--compact-history-budget", strconv.Itoa(gateway.DefaultCompactHistoryBudget),
		"--elide-result-bytes", strconv.Itoa(gateway.DefaultElideResultBytes),
	)
	notes = append(notes,
		fmt.Sprintf("compact-history-budget=%d: guard sheds un-cached middle turns once resident tokens exceed this threshold, preserving the upstream cache prefix", gateway.DefaultCompactHistoryBudget),
		fmt.Sprintf("elide-result-bytes=%d: guard shrinks oversized tool results outside the active working set to a bounded head+tail form", gateway.DefaultElideResultBytes),
	)
	if opt.DebugStats {
		guardArgs = append(guardArgs, "--debug-stats")
		notes = append(notes, "debug-stats=on: one compact per-turn line to stderr leading with a verdict (ok/warming/degraded/cold) + the NET write-premium-aware token saving, then cache health + compaction")
	}
	guardArgs = append(guardArgs, "--")
	launch := append([]string{fakPath}, guardArgs...)
	launch = append(launch, command...)

	return tuiAgentReport{
		Schema:              tuiAgentSchema,
		At:                  at.UTC().Format(time.RFC3339),
		Backend:             backend,
		Mode:                "launch",
		Provider:            "anthropic",
		Auth:                auth,
		Account:             strings.TrimSpace(opt.Account),
		ResolvedAccount:     resolvedAccount,
		AccountIdentity:     identity,
		ClaudeConfigDir:     cfgDir,
		ConfigSource:        cfgSource,
		SessionID:           sessionID,
		PermissionMode:      permissionMode,
		Policy:              strings.TrimSpace(opt.Policy),
		Model:               strings.TrimSpace(opt.Model),
		ContextBudget:       opt.ContextBudgetTokens,
		RestartOnBudget:     opt.RestartOnBudget,
		RestartLimit:        opt.RestartLimit,
		DebugStats:          opt.DebugStats,
		CompactHistoryLimit: gateway.DefaultCompactHistoryBudget,
		ElideResultBytes:    gateway.DefaultElideResultBytes,
		Command:             command,
		Launch:              launch,
		Env:                 env,
		Notes:               notes,
	}, nil
}

func buildTUIAgentGatewayReport(opt tuiAgentOptions, at time.Time, backend string, command []string, permissionMode string, env []tuiAgentEnv, cfgDir, cfgSource, resolvedAccount, identity string, notes []string, getenv func(string) string) (tuiAgentReport, error) {
	if strings.TrimSpace(opt.Policy) != "" || opt.ContextBudgetTokens > 0 || opt.RestartOnBudget || opt.RestartLimit > 0 || opt.Passthrough {
		return tuiAgentReport{}, fmt.Errorf("--gateway-url launches an existing gateway; guard-only options (--policy, --context-budget-tokens, --restart-on-budget, --restart-limit, --passthrough) do not apply")
	}
	gatewayURL, err := normalizeTUIAgentGatewayURL(opt.GatewayURL)
	if err != nil {
		return tuiAgentReport{}, err
	}
	keyEnv := strings.TrimSpace(opt.GatewayKeyEnv)
	if keyEnv == "" {
		keyEnv = "FAK_GATEWAY_KEY"
	}
	bearer := strings.TrimSpace(getenv(keyEnv))
	// A loopback gateway is a local `fak serve` that, unless started with
	// --require-key-env, accepts unauthenticated requests — so tolerate an empty bearer
	// for it (mirrors claude_mac_fak's gatewayIsLocal tolerance), which is what makes
	// `fak c local` launch without demanding a bogus key. A REMOTE gateway still requires
	// the bearer to be set.
	localGateway := gatewayIsLocal(gatewayURL)
	if bearer == "" && !localGateway {
		return tuiAgentReport{}, fmt.Errorf("--gateway-url requires %s to be set (or pass --gateway-key-env VAR)", keyEnv)
	}
	notes = filterTUIAgentGatewayNotes(notes)
	env = append(env, tuiAgentEnv{Name: "ANTHROPIC_BASE_URL", Value: gatewayURL, Source: "gateway-url"})
	auth := "gateway-bearer"
	if bearer != "" {
		env = append(env, tuiAgentEnv{Name: "ANTHROPIC_API_KEY", Source: "env:" + keyEnv, FromEnv: keyEnv, Sensitive: true})
	} else {
		// loopback, no bearer: do not inject an empty ANTHROPIC_API_KEY (Claude Code
		// reads an empty value as "no key"); record the unauthenticated posture instead.
		auth = "none"
		notes = append(notes, fmt.Sprintf("loopback gateway %s with no %s set — launching unauthenticated (a local fak serve without --require-key-env)", gatewayURL, keyEnv))
	}
	if strings.TrimSpace(getenv("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC")) == "" {
		env = append(env, tuiAgentEnv{Name: "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", Value: "1", Source: "gateway-default"})
	}
	model := strings.TrimSpace(opt.Model)
	if model != "" {
		for _, name := range []string{
			"ANTHROPIC_MODEL",
			"ANTHROPIC_DEFAULT_OPUS_MODEL",
			"ANTHROPIC_DEFAULT_SONNET_MODEL",
			"ANTHROPIC_DEFAULT_HAIKU_MODEL",
			"ANTHROPIC_SMALL_FAST_MODEL",
		} {
			env = append(env, tuiAgentEnv{Name: name, Value: model, Source: "model"})
		}
	}
	if opt.APITimeoutMS > 0 && strings.TrimSpace(getenv("API_TIMEOUT_MS")) == "" {
		env = append(env, tuiAgentEnv{Name: "API_TIMEOUT_MS", Value: strconv.Itoa(opt.APITimeoutMS), Source: "gateway-default"})
	}
	notes = append(notes,
		"launches the agent directly against an existing fak serve gateway; no local fak guard is started",
		fmt.Sprintf("gateway bearer is read from %s at launch and is not printed in dry-run output", keyEnv),
	)
	sessionID := strings.TrimSpace(opt.SessionID)
	if sessionID == "" {
		sessionID = "tui-agent"
	}
	return tuiAgentReport{
		Schema:          tuiAgentSchema,
		At:              at.UTC().Format(time.RFC3339),
		Backend:         backend,
		Mode:            "launch",
		Provider:        "existing-fak-gateway",
		Auth:            auth,
		GatewayURL:      gatewayURL,
		Account:         strings.TrimSpace(opt.Account),
		ResolvedAccount: resolvedAccount,
		AccountIdentity: identity,
		ClaudeConfigDir: cfgDir,
		ConfigSource:    cfgSource,
		SessionID:       sessionID,
		PermissionMode:  permissionMode,
		Model:           model,
		Command:         command,
		Launch:          command,
		Env:             env,
		Notes:           notes,
	}, nil
}

func filterTUIAgentGatewayNotes(notes []string) []string {
	if len(notes) == 0 {
		return notes
	}
	out := notes[:0]
	for _, note := range notes {
		if strings.Contains(note, "Claude may prompt for login") {
			continue
		}
		out = append(out, note)
	}
	return out
}

func normalizeTUIAgentGatewayURL(raw string) (string, error) {
	u := strings.TrimSpace(raw)
	if u == "" {
		return "", fmt.Errorf("--gateway-url must not be empty")
	}
	u = strings.TrimRight(u, "/")
	if strings.HasSuffix(u, "/v1") {
		u = strings.TrimSuffix(u, "/v1")
	}
	if !strings.HasPrefix(u, "http://") && !strings.HasPrefix(u, "https://") {
		return "", fmt.Errorf("--gateway-url must start with http:// or https://")
	}
	return u, nil
}

func resolveTUIAgentClaudeConfig(opt tuiAgentOptions, getenv func(string) string) ([]tuiAgentEnv, string, string, string, string, []string, error) {
	if getenv == nil {
		getenv = os.Getenv
	}
	var env []tuiAgentEnv
	var notes []string
	account := strings.TrimSpace(opt.Account)
	if account != "" {
		reg, err := loadOrDiscover(opt.Registry, opt.Home)
		if err != nil {
			return nil, "", "", "", "", nil, err
		}
		reg = reg.Refresh()
		home, chain, err := reg.Serve(account)
		if err != nil {
			return nil, "", "", "", "", nil, err
		}
		for i, hop := range chain {
			to := home.Name
			if i+1 < len(chain) {
				to = chain[i+1]
			}
			notes = append(notes, fmt.Sprintf("%q can't serve; rehomed to %q", hop, to))
		}
		if note := tuiLoginNote(home); note != "" {
			notes = append(notes, note)
		}
		env = append(env, tuiAgentEnv{Name: "CLAUDE_CONFIG_DIR", Value: home.Dir, Source: "account:" + home.Name})
		return env, home.Dir, "account:" + home.Name, home.Name, home.Identity.Email, notes, nil
	}
	if dir := strings.TrimSpace(opt.ClaudeConfigDir); dir != "" {
		id := acct.DeriveIdentity(dir)
		if note := tuiLoginNote(acct.Home{Dir: dir, Identity: id}); note != "" {
			notes = append(notes, note)
		}
		env = append(env, tuiAgentEnv{Name: "CLAUDE_CONFIG_DIR", Value: dir, Source: "flag"})
		return env, dir, "flag", "", id.Email, notes, nil
	}
	if dir := strings.TrimSpace(getenv("CLAUDE_CONFIG_DIR")); dir != "" {
		id := acct.DeriveIdentity(dir)
		if note := tuiLoginNote(acct.Home{Dir: dir, Identity: id}); note != "" {
			notes = append(notes, note)
		}
		return nil, dir, "inherited-env", "", id.Email, notes, nil
	}
	dir := guardClaudeConfigDir()
	id := acct.DeriveIdentity(dir)
	if note := tuiLoginNote(acct.Home{Dir: dir, Identity: id}); note != "" {
		notes = append(notes, note)
	}
	return nil, dir, "default", "", id.Email, notes, nil
}

func tuiLoginNote(home acct.Home) string {
	status := home.LoginStatus()
	if status == acct.LoginReady {
		return ""
	}
	reason, action := acct.LoginReasonAction(status, home)
	subject := home.Dir
	if home.Name != "" {
		subject = fmt.Sprintf("%q (%s)", home.Name, home.Dir)
	}
	if action != "" {
		return fmt.Sprintf("%s login=%s - %s; %s; Claude may prompt for login", subject, status, reason, action)
	}
	return fmt.Sprintf("%s login=%s - %s; Claude may prompt for login", subject, status, reason)
}

func tuiExecutable() string {
	if exe, err := os.Executable(); err == nil && strings.TrimSpace(exe) != "" {
		return exe
	}
	if len(os.Args) > 0 && strings.TrimSpace(os.Args[0]) != "" {
		return os.Args[0]
	}
	return "fak"
}

func launchTUIAgent(stdout, stderr io.Writer, report tuiAgentReport) int {
	if len(report.Launch) == 0 {
		fmt.Fprintln(stderr, "fak console agent: empty launch command")
		return 1
	}
	child := exec.Command(report.Launch[0], report.Launch[1:]...)
	child.Stdin = os.Stdin
	child.Stdout = stdout
	child.Stderr = stderr
	child.Env = mergeTUIAgentEnv(os.Environ(), report.Env)
	if err := child.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode()
		}
		fmt.Fprintf(stderr, "fak console agent: launch %q: %v\n", report.Launch[0], err)
		return 1
	}
	return 0
}

func mergeTUIAgentEnv(base []string, pairs []tuiAgentEnv) []string {
	out := append([]string{}, base...)
	for _, pair := range pairs {
		name := strings.TrimSpace(pair.Name)
		if name == "" {
			continue
		}
		value := pair.Value
		if strings.TrimSpace(pair.FromEnv) != "" {
			value = os.Getenv(strings.TrimSpace(pair.FromEnv))
		}
		line := name + "=" + value
		replaced := false
		for i, cur := range out {
			k, _, ok := strings.Cut(cur, "=")
			if ok && strings.EqualFold(k, name) {
				out[i] = line
				replaced = true
				break
			}
		}
		if !replaced {
			out = append(out, line)
		}
	}
	return out
}

func loadTUISessions(path, addr, key string) (gateway.SessionListResponse, string, error) {
	if path != "" {
		b, err := os.ReadFile(path)
		if err != nil {
			return gateway.SessionListResponse{}, "", err
		}
		list, err := decodeTUISessions(b)
		return list, path, err
	}
	c := &sessionClient{
		base: strings.TrimRight(addr, "/"),
		key:  key,
		hc:   &http.Client{Timeout: 15 * time.Second},
	}
	list, err := c.list()
	if err != nil {
		return gateway.SessionListResponse{}, "", err
	}
	return list, c.base + "/v1/fak/sessions", nil
}

func decodeTUISessions(b []byte) (gateway.SessionListResponse, error) {
	var list gateway.SessionListResponse
	if err := json.Unmarshal(b, &list); err == nil && (list.Sessions != nil || list.Count != 0) {
		if list.Count == 0 {
			list.Count = len(list.Sessions)
		}
		return list, nil
	}
	var sessions []gateway.SessionState
	if err := json.Unmarshal(b, &sessions); err != nil {
		return gateway.SessionListResponse{}, fmt.Errorf("session JSON must be a SessionListResponse or SessionState array: %w", err)
	}
	return gateway.SessionListResponse{Sessions: sessions, Count: len(sessions)}, nil
}
