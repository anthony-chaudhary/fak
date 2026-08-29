package main

// fak console is the native terminal control pane spine. The first surface is an
// issue queue view because issue triage is already one of fak's dogfood loops:
// fetch or load the GitHub issue shape, fold it into a ranked model, then render
// a compact terminal dashboard without adding a TUI dependency.

import (
	"context"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/tuiplugin"
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
	// Sourced from the canonical dispatchtick priority constants so the picker, the
	// triage scorer, and this TUI never disagree (see internal/dispatchtick/priority.go's
	// lockstep note). The unlabeled default lives at the lookup site in
	// tui_issues_garden.go (dispatchtick.PriorityWeightDefault).
	tuiPriorityWeights = map[string]int{
		"priority/P0": dispatchtick.PriorityWeightP0,
		"priority/P1": dispatchtick.PriorityWeightP1,
		"priority/P2": dispatchtick.PriorityWeightP2,
	}
	tuiKindLabels = map[string]bool{
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
	case "-h", "--help", "help":
		tuiUsage(stdout)
		return 0
	}
	paneID := argv[0]
	if paneID == "config" {
		paneID = "settings"
	}
	pane, ok := tuiplugin.Lookup(paneID)
	if !ok {
		fmt.Fprintf(stderr, "fak console: unknown subcommand %q\n", argv[0])
		tuiUsage(stderr)
		return 2
	}
	paneArgv, err := prepareTUIPaneArgs(pane, argv[1:])
	if err != nil {
		fmt.Fprintf(stderr, "fak console %s: %v\n", pane.ID, err)
		return 2
	}
	return pane.Run(stdout, stderr, paneArgv)
}

func runTUIIssues(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("tui issues", flag.ContinueOnError)
	fs.SetOutput(stderr)
	issuesJSON := fs.String("issues-json", "", "read gh issue JSON from a file instead of shelling out to gh")
	repo := fs.String("repo", "", "owner/repo for gh; default is current repo")
	state := fs.String("state", "open", "issue state for gh: open|closed|all")
	limit := fs.Int("limit", 100000, "maximum issues to fetch from gh; ranking refuses if the issue-only total exceeds it")
	asOfText := fs.String("as-of", "", "date used for age/idle math (YYYY-MM-DD, default: today UTC)")
	epic := fs.Int("epic", 0, "highlight one epic issue number and issues whose title/body references #N")
	top := fs.Int("top", 25, "number of ranked rows to render in human mode")
	repairBatchSize := fs.Int("repair-batch-size", 50, "maximum review-only issue numbers per taxonomy repair batch")
	width := fs.Int("width", 120, "target terminal width for human rendering")
	asJSON := fs.Bool("json", false, "emit the issue TUI model as JSON")
	if !parseFlags(fs, argv) {
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
	if *repairBatchSize <= 0 {
		fmt.Fprintln(stderr, "fak console issues: --repair-batch-size must be positive")
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

	snapshot, source, err := loadTUIIssueSnapshot(*issuesJSON, *repo, *state, *limit)
	if err != nil {
		fmt.Fprintf(stderr, "fak console issues: %v\n", err)
		return 2
	}
	report, err := buildTUIIssueReportWithCensus(snapshot.Issues, source, asOf, *epic, snapshot.Census, *repairBatchSize)
	if err != nil {
		fmt.Fprintf(stderr, "fak console issues: %v\n", err)
		return 3
	}
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
	if !parseFlags(fs, argv) {
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
	controlKey := fs.String("press", "", "OOB control (#2763): drive one lifecycle op on the selected session — p pause, r resume, t throttle, d drain — through the same session-control route, instead of rendering the pane")
	controlSession := fs.String("session", "", "with --press: the session trace id to control (default: the highlighted/top-of-attention row)")
	confirm := fs.Bool("confirm", false, "with --press: confirm a destructive control op (drain), which is otherwise withheld")
	if !parseFlags(fs, argv) {
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
	// OOB control keybinding (#2763): when the operator presses a control key, drive one
	// lifecycle op on the selected session through the shared session-control route rather
	// than rendering the read-only pane. A sessions-JSON fixture is a read-only snapshot
	// (no gateway to POST to), so a control key there is refused with an explanatory error.
	if strings.TrimSpace(*controlKey) != "" {
		if strings.TrimSpace(*sessionsJSON) != "" {
			fmt.Fprintln(stderr, "fak console sessions: --press drives a live gateway; it cannot combine with --sessions-json (a read-only snapshot)")
			return 2
		}
		return runTUISessionControlKey(stdout, stderr, *addr, *key, report, *controlKey, *controlSession, *confirm, *asJSON)
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak console sessions")
	}
	fmt.Fprint(stdout, renderTUISessions(report, *top, *width))
	return 0
}

// runTUISessionControlKey drives one OOB control keybinding (#2763): it resolves the
// selected session, plans the control request the pressed key emits (the pure, gated
// core), and — when the plan is emit-ready — dispatches it through the shared
// session-control route. A withheld plan (unbound key, destructive-needs-confirm, or a
// deferred op like redirect) prints why and returns non-zero without dialing the route,
// so a destructive op is never fired without confirmation.
func runTUISessionControlKey(stdout, stderr io.Writer, addr, key string, report tuiSessionReport, controlKey, controlSession string, confirm, asJSON bool) int {
	runes := []rune(controlKey)
	if len(runes) != 1 {
		fmt.Fprintf(stderr, "fak console sessions: --press must be a single control key (one of %s)\n", strings.Join(tuiSessionControlKeys(), " "))
		return 2
	}
	row, ok := selectTUISessionRow(report, controlSession)
	if !ok {
		if strings.TrimSpace(controlSession) != "" {
			fmt.Fprintf(stderr, "fak console sessions: --session %q not found among %d live session(s)\n", controlSession, len(report.Rows))
		} else {
			fmt.Fprintln(stderr, "fak console sessions: no live sessions to control (pass --session <id> once one is running)")
		}
		return 1
	}
	plan, bound := planTUISessionControlKey(row, runes[0], confirm)
	if !bound {
		fmt.Fprintf(stderr, "fak console sessions: %q is not a control key (one of %s)\n", controlKey, strings.Join(tuiSessionControlKeys(), " "))
		return 2
	}
	if !plan.Emit {
		switch {
		case plan.Deferred != "":
			fmt.Fprintf(stderr, "fak console sessions: %s deferred — %s\n", plan.Label, plan.Deferred)
		case plan.NeedsConfirm:
			fmt.Fprintf(stderr, "fak console sessions: %s is destructive — re-run with --confirm to drain %s\n", plan.Label, plan.TraceID)
		default:
			fmt.Fprintf(stderr, "fak console sessions: %s emitted no control op\n", plan.Label)
		}
		return 1
	}
	c := &sessionClient{base: strings.TrimRight(addr, "/"), key: key, hc: &http.Client{Timeout: 15 * time.Second}}
	st, err := dispatchTUISessionControl(c, plan)
	if err != nil {
		fmt.Fprintf(stderr, "fak console sessions: %s %s: %v\n", plan.Label, plan.TraceID, err)
		return 1
	}
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, st, "fak console sessions")
	}
	fmt.Fprintf(stdout, "%s -> %s\n%s\n", plan.Label, plan.TraceID, formatSessionState(st))
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
	if !parseFlags(fs, argv) {
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
	colorMode := fs.String("color", "auto", "colorize human output: auto, always, or never (NO_COLOR disables color)")
	atText := fs.String("at", "", "snapshot time (RFC3339 or YYYY-MM-DD, default: now)")
	width := fs.Int("width", 120, "target terminal width for human rendering")
	asJSON := fs.Bool("json", false, "emit the guard TUI model as JSON")
	if !parseFlags(fs, argv) {
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
	color, err := tuiColorEnabled(stdout, *colorMode)
	if err != nil {
		fmt.Fprintf(stderr, "fak console guard: %v\n", err)
		return 2
	}
	style := tuiGuardRenderStyle{Color: color}

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
		return runTUIGuardJournal(stdout, stderr, path, at, *width, *maxRows, *asJSON, *follow, style)
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
	fmt.Fprint(stdout, renderTUIGuardStyled(report, *width, style))
	return 0
}

func tuiColorEnabled(stdout io.Writer, mode string) (bool, error) {
	mode = strings.ToLower(strings.TrimSpace(mode))
	if mode == "" {
		mode = "auto"
	}
	switch mode {
	case "always", "on", "yes", "true", "1":
		return os.Getenv("NO_COLOR") == "", nil
	case "never", "off", "no", "false", "0":
		return false, nil
	case "auto":
		if os.Getenv("NO_COLOR") != "" {
			return false, nil
		}
		f, ok := stdout.(*os.File)
		return ok && guardFdIsTerminal(int(f.Fd())), nil
	default:
		return false, fmt.Errorf("--color must be auto, always, or never")
	}
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
func runTUIGuardJournal(stdout, stderr io.Writer, path string, at time.Time, width, maxRows int, asJSON, follow bool, style tuiGuardRenderStyle) int {
	// This pane is deliberately TAIL-ONLY (#6488): it is a live attention view of what
	// the guard is deciding now, and --follow continues from the live file, so reading
	// the sealed segments would only push the recent rows out of the frame. That makes
	// it the one reader that keeps the live-file read — but it must not pass a tail off
	// as the whole journal, so ReadTail's omission is stated out loud. Every consumer
	// that reports a TOTAL uses journal.ReadAllSegments instead.
	rows, omission, err := journal.ReadTail(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak console guard: %v\n", err)
		return 1
	}
	if omission.Omitted() {
		fmt.Fprintf(stderr, "fak console guard: showing the live segment of %s only — %s\n", path, omission)
	}
	// The rotation anchor is bookkeeping, not an adjudication: it must not render as a
	// guard row (it would be the first row of every post-cut segment).
	rows = journal.WithoutCutAnchors(rows)
	report := buildTUIGuardJournalReport(rows, path, at, maxRows)
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak console guard")
	}
	fmt.Fprint(stdout, renderTUIGuardStyled(report, width, style))
	if follow {
		return followGuardJournal(stdout, path, width, lastSeqOf(rows), style)
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
func followGuardJournal(stdout io.Writer, path string, width int, lastSeq uint64, style tuiGuardRenderStyle) int {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	offset := int64(0)
	if info, err := os.Stat(path); err == nil {
		offset = info.Size()
	}
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return 0
		case <-ticker.C:
			rows, next, err := journal.ReadRowsFrom(path, offset)
			if err != nil {
				continue
			}
			offset = next
			for _, r := range rows {
				if r.Seq <= lastSeq {
					continue
				}
				lastSeq = r.Seq
				fmt.Fprintln(stdout, formatGuardJournalLine(r, width, style))
			}
		}
	}
}

// formatGuardJournalLine renders one journal row as a compact tail line — decision
// fields + the witness claim only, never a payload (the #840 redaction contract).
func formatGuardJournalLine(r journal.Row, width int, style tuiGuardRenderStyle) string {
	parts := []string{fmt.Sprintf("seq=%d", r.Seq), r.Kind}
	for _, s := range []string{r.Tool, r.Verdict, r.Reason} {
		if s != "" {
			parts = append(parts, s)
		}
	}
	if r.Witness != "" {
		parts = append(parts, "("+r.Witness+")")
	}
	row := tuiGuardRow{
		Kind:    "audit-" + strings.ToLower(r.Kind),
		Tool:    r.Tool,
		Verdict: strings.ToUpper(r.Verdict),
		Reason:  strings.ToUpper(r.Reason),
		Count:   1,
	}
	row.Tags, row.Attention = scoreTUIGuardRow(row)
	visual := tuiGuardRowVisual(row)
	prefix := style.paint(visual.SGR, padRightTUI(visual.Symbol, 3))
	body := trimTUI(strings.Join(parts, "  "), maxTUI(40, width-4))
	return prefix + " " + style.paint(visual.SGR, body)
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
