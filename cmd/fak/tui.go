package main

// fak console is the native terminal control pane spine. The first surface is an
// issue queue view because issue triage is already one of fak's dogfood loops:
// fetch or load the GitHub issue shape, fold it into a ranked model, then render
// a compact terminal dashboard without adding a TUI dependency.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	acct "github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/gardenbundle"
	"github.com/anthony-chaudhary/fak/internal/gateway"
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

type tuiIssue struct {
	Number    int             `json:"number"`
	Title     string          `json:"title"`
	URL       string          `json:"url"`
	State     string          `json:"state"`
	Body      string          `json:"body"`
	Labels    []tuiLabel      `json:"labels"`
	CreatedAt string          `json:"createdAt"`
	UpdatedAt string          `json:"updatedAt"`
	Author    *tuiUser        `json:"author"`
	Assignees []tuiUser       `json:"assignees"`
	Milestone *tuiMilestone   `json:"milestone"`
	Comments  tuiCommentCount `json:"comments"`
}

type tuiLabel struct {
	Name string `json:"name"`
}

type tuiUser struct {
	Login string `json:"login"`
}

type tuiMilestone struct {
	Title string `json:"title"`
}

type tuiCommentCount int

func (c *tuiCommentCount) UnmarshalJSON(b []byte) error {
	var n int
	if err := json.Unmarshal(b, &n); err == nil {
		*c = tuiCommentCount(n)
		return nil
	}
	var list []json.RawMessage
	if err := json.Unmarshal(b, &list); err == nil {
		*c = tuiCommentCount(len(list))
		return nil
	}
	var obj struct {
		TotalCount int `json:"totalCount"`
	}
	if err := json.Unmarshal(b, &obj); err == nil {
		*c = tuiCommentCount(obj.TotalCount)
		return nil
	}
	return fmt.Errorf("comments must be a count, list, or totalCount object")
}

type tuiIssueRow struct {
	Number     int      `json:"number"`
	Title      string   `json:"title"`
	URL        string   `json:"url,omitempty"`
	State      string   `json:"state,omitempty"`
	Labels     []string `json:"labels"`
	Author     string   `json:"author,omitempty"`
	Assignees  []string `json:"assignees,omitempty"`
	Milestone  string   `json:"milestone,omitempty"`
	Comments   int      `json:"comments,omitempty"`
	AgeDays    int      `json:"age_days"`
	IdleDays   int      `json:"idle_days"`
	Priority   string   `json:"priority,omitempty"`
	InProgress bool     `json:"in_progress,omitempty"`
	Tags       []string `json:"tags,omitempty"`
	Score      int      `json:"score"`
	Related    bool     `json:"related,omitempty"`
}

type tuiIssueCounts struct {
	Open            int `json:"open"`
	P0              int `json:"p0"`
	P1              int `json:"p1"`
	P2              int `json:"p2"`
	NeedsPriority   int `json:"needs_priority"`
	NeedsKind       int `json:"needs_kind"`
	NeedsArea       int `json:"needs_area"`
	Orphan          int `json:"orphan"`
	Stale           int `json:"stale"`
	DormantQuestion int `json:"dormant_question"`
	LikelyDup       int `json:"likely_dup"`
	Bare            int `json:"bare"`
	Related         int `json:"related,omitempty"`
}

type tuiLane struct {
	Name         string `json:"name"`
	Count        int    `json:"count"`
	Orphan       int    `json:"orphan,omitempty"`
	NeedsArea    int    `json:"needs_area,omitempty"`
	NeedsKind    int    `json:"needs_kind,omitempty"`
	MaxIdleDays  int    `json:"max_idle_days,omitempty"`
	TopIssue     int    `json:"top_issue,omitempty"`
	TopIssueText string `json:"top_issue_text,omitempty"`
}

type tuiIssueAction struct {
	Number  int    `json:"number"`
	Kind    string `json:"kind"`
	Reason  string `json:"reason"`
	Command string `json:"cmd,omitempty"`
}

type tuiIssueReport struct {
	Schema  string           `json:"schema"`
	AsOf    string           `json:"as_of"`
	Source  string           `json:"source"`
	Epic    *tuiIssueRow     `json:"epic,omitempty"`
	Counts  tuiIssueCounts   `json:"counts"`
	Lanes   []tuiLane        `json:"lanes"`
	Rows    []tuiIssueRow    `json:"rows"`
	Actions []tuiIssueAction `json:"actions,omitempty"`
}

type tuiLoopReport struct {
	Schema string        `json:"schema"`
	At     string        `json:"at"`
	Ledger string        `json:"ledger"`
	Counts tuiLoopCounts `json:"counts"`
	Lanes  []tuiLoopLane `json:"lanes"`
	Rows   []tuiLoopRow  `json:"rows"`
}

type tuiLoopCounts struct {
	Loops         int `json:"loops"`
	Running       int `json:"running"`
	Refused       int `json:"refused"`
	Failed        int `json:"failed"`
	Witnessed     int `json:"witnessed"`
	WitnessGaps   int `json:"witness_gaps"`
	Notifications int `json:"notifications"`
}

type tuiLoopLane struct {
	Name        string `json:"name"`
	Count       int    `json:"count"`
	TopLoop     string `json:"top_loop,omitempty"`
	TopLoopText string `json:"top_loop_text,omitempty"`
}

type tuiLoopRow struct {
	LoopID              string   `json:"loop_id"`
	State               string   `json:"state"`
	LastKind            string   `json:"last_kind,omitempty"`
	LastSeq             uint64   `json:"last_seq,omitempty"`
	AgeSeconds          int64    `json:"age_s,omitempty"`
	CurrentRunID        string   `json:"current_run_id,omitempty"`
	LastRunStatus       string   `json:"last_run_status,omitempty"`
	LastRunReason       string   `json:"last_run_reason,omitempty"`
	LastRunSummary      string   `json:"last_run_summary,omitempty"`
	Fires               uint64   `json:"fires,omitempty"`
	Admitted            uint64   `json:"admitted,omitempty"`
	Refused             uint64   `json:"refused,omitempty"`
	ConsecutiveRefusals uint64   `json:"consecutive_refusals,omitempty"`
	Started             uint64   `json:"started,omitempty"`
	Ended               uint64   `json:"ended,omitempty"`
	Witnessed           uint64   `json:"witnessed,omitempty"`
	WitnessRefused      uint64   `json:"witness_refused,omitempty"`
	WitnessUnavailable  uint64   `json:"witness_unavailable,omitempty"`
	Notifications       uint64   `json:"notifications,omitempty"`
	WitnessRate         *float64 `json:"witness_rate,omitempty"`
	Attention           int      `json:"attention"`
	Tags                []string `json:"tags,omitempty"`
}

type tuiSessionReport struct {
	Schema string           `json:"schema"`
	At     string           `json:"at"`
	Source string           `json:"source"`
	Counts tuiSessionCounts `json:"counts"`
	Lanes  []tuiSessionLane `json:"lanes"`
	Rows   []tuiSessionRow  `json:"rows"`
}

type tuiSessionCounts struct {
	Sessions      int `json:"sessions"`
	Running       int `json:"running"`
	Throttled     int `json:"throttled"`
	Paused        int `json:"paused"`
	Draining      int `json:"draining"`
	Stopped       int `json:"stopped"`
	Budgeted      int `json:"budgeted"`
	ContextBudget int `json:"context_budget"`
	Lineage       int `json:"lineage"`
	WithReason    int `json:"with_reason"`
}

type tuiSessionLane struct {
	Name       string `json:"name"`
	Count      int    `json:"count"`
	TopSession string `json:"top_session,omitempty"`
	TopSummary string `json:"top_summary,omitempty"`
}

type tuiSessionRow struct {
	TraceID           string   `json:"trace_id"`
	Run               string   `json:"run"`
	Priority          int      `json:"priority"`
	Rev               uint64   `json:"rev"`
	Reason            string   `json:"reason,omitempty"`
	TurnsLeft         int      `json:"turns_left"`
	TokensLeft        int      `json:"tokens_left"`
	ContextTokensLeft int      `json:"context_tokens_left,omitempty"`
	MaxTokensPerTurn  int      `json:"max_tokens_per_turn,omitempty"`
	MinTurnGapMs      int      `json:"min_turn_gap_ms,omitempty"`
	ContinuationID    string   `json:"continuation_id,omitempty"`
	ParentTrace       string   `json:"parent_trace,omitempty"`
	Generation        int      `json:"generation,omitempty"`
	Attention         int      `json:"attention"`
	Tags              []string `json:"tags,omitempty"`
}

type tuiGardenReport struct {
	Schema      string          `json:"schema"`
	At          string          `json:"at"`
	Source      string          `json:"source"`
	Workspace   string          `json:"workspace,omitempty"`
	Commit      string          `json:"commit,omitempty"`
	OK          bool            `json:"ok"`
	Verdict     string          `json:"verdict"`
	Finding     string          `json:"finding"`
	Reason      string          `json:"reason"`
	NextAction  string          `json:"next_action"`
	GateExit    int             `json:"gate_exit"`
	GateMessage string          `json:"gate_message,omitempty"`
	Counts      tuiGardenCounts `json:"counts"`
	Rows        []tuiGardenRow  `json:"rows"`
}

type tuiGardenCounts struct {
	Members int `json:"members"`
	OK      int `json:"ok"`
	Action  int `json:"action"`
	Red     int `json:"red"`
	Errored int `json:"errored"`
	Gating  int `json:"gating"`
	Skipped int `json:"skipped,omitempty"`
}

type tuiGardenRow struct {
	Key       string         `json:"key"`
	Label     string         `json:"label"`
	State     string         `json:"state"`
	OK        bool           `json:"ok"`
	Gates     bool           `json:"gates,omitempty"`
	ExitCode  int            `json:"exit_code"`
	Verdict   string         `json:"verdict,omitempty"`
	Detail    string         `json:"detail,omitempty"`
	Counts    map[string]int `json:"counts,omitempty"`
	Attention int            `json:"attention"`
	Tags      []string       `json:"tags,omitempty"`
}

type tuiGuardArtifact struct {
	Path string
	Raw  map[string]any
}

type tuiGuardReport struct {
	Schema  string           `json:"schema"`
	At      string           `json:"at"`
	Source  string           `json:"source"`
	Status  string           `json:"status"`
	Counts  tuiGuardCounts   `json:"counts"`
	Actions []string         `json:"actions,omitempty"`
	Rows    []tuiGuardRow    `json:"rows"`
	Sources []tuiGuardSource `json:"sources"`
}

type tuiGuardSource struct {
	Path   string `json:"path"`
	Schema string `json:"schema,omitempty"`
	Status string `json:"status,omitempty"`
}

type tuiGuardCounts struct {
	Artifacts   int `json:"artifacts"`
	Rows        int `json:"rows"`
	Pass        int `json:"pass"`
	Warn        int `json:"warn"`
	Fail        int `json:"fail"`
	Allow       int `json:"allow"`
	Deny        int `json:"deny"`
	Transform   int `json:"transform"`
	Quarantine  int `json:"quarantine"`
	PolicyBlock int `json:"policy_block"`
	DefaultDeny int `json:"default_deny"`
	Expected    int `json:"expected"`
	Unexpected  int `json:"unexpected"`
}

type tuiGuardRow struct {
	Artifact  string   `json:"artifact"`
	Kind      string   `json:"kind"`
	Tool      string   `json:"tool,omitempty"`
	Verdict   string   `json:"verdict,omitempty"`
	Reason    string   `json:"reason,omitempty"`
	By        string   `json:"by,omitempty"`
	Status    string   `json:"status,omitempty"`
	Detail    string   `json:"detail,omitempty"`
	Count     int      `json:"count,omitempty"`
	Attention int      `json:"attention"`
	Tags      []string `json:"tags,omitempty"`
}

type tuiOverviewReport struct {
	Schema  string              `json:"schema"`
	At      string              `json:"at"`
	Source  string              `json:"source"`
	Counts  tuiOverviewCounts   `json:"counts"`
	Cards   []tuiOverviewCard   `json:"cards"`
	Actions []tuiOverviewAction `json:"actions,omitempty"`
}

type tuiOverviewCounts struct {
	Cards   int `json:"cards"`
	OK      int `json:"ok"`
	Action  int `json:"action"`
	Warn    int `json:"warn"`
	Missing int `json:"missing"`
}

type tuiOverviewCard struct {
	Pane      string         `json:"pane"`
	Status    string         `json:"status"`
	Source    string         `json:"source,omitempty"`
	Summary   string         `json:"summary"`
	Command   string         `json:"command"`
	Attention int            `json:"attention"`
	Counts    map[string]int `json:"counts,omitempty"`
	Tags      []string       `json:"tags,omitempty"`
}

type tuiOverviewAction struct {
	Pane    string `json:"pane"`
	Command string `json:"command"`
	Reason  string `json:"reason"`
}

type tuiAgentReport struct {
	Schema          string        `json:"schema"`
	At              string        `json:"at"`
	Backend         string        `json:"backend"`
	Mode            string        `json:"mode"`
	Provider        string        `json:"provider"`
	Auth            string        `json:"auth"`
	Account         string        `json:"account,omitempty"`
	ResolvedAccount string        `json:"resolved_account,omitempty"`
	AccountIdentity string        `json:"account_identity,omitempty"`
	ClaudeConfigDir string        `json:"claude_config_dir,omitempty"`
	ConfigSource    string        `json:"config_source,omitempty"`
	SessionID       string        `json:"session_id,omitempty"`
	Policy          string        `json:"policy,omitempty"`
	Model           string        `json:"model,omitempty"`
	ContextBudget   int           `json:"context_budget_tokens,omitempty"`
	RestartOnBudget bool          `json:"restart_on_budget,omitempty"`
	RestartLimit    int           `json:"restart_limit,omitempty"`
	Command         []string      `json:"command"`
	Launch          []string      `json:"launch"`
	Env             []tuiAgentEnv `json:"env,omitempty"`
	Notes           []string      `json:"notes,omitempty"`
}

type tuiAgentEnv struct {
	Name   string `json:"name"`
	Value  string `json:"value"`
	Source string `json:"source"`
}

type tuiAgentOptions struct {
	Backend             string
	Command             string
	CommandArgs         []string
	Prompt              string
	Account             string
	ClaudeConfigDir     string
	Registry            string
	Home                string
	Policy              string
	Model               string
	SessionID           string
	ContextBudgetTokens int
	RestartOnBudget     bool
	RestartLimit        int
	Passthrough         bool
}

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
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak console issues: encode json: %v\n", err)
			return 1
		}
		return 0
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
	st, err := loopmgr.SnapshotFile(*ledger, at)
	if err != nil {
		fmt.Fprintf(stderr, "fak console loops: %v\n", err)
		return 1
	}
	report := buildTUILoopReport(st, at)
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak console loops: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprint(stdout, renderTUILoops(report, *top, *width))
	return 0
}

func runTUISessions(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("tui sessions", flag.ContinueOnError)
	fs.SetOutput(stderr)
	sessionsJSON := fs.String("sessions-json", "", "read SessionListResponse JSON from a file instead of a live gateway")
	addr := fs.String("addr", defaultSessionAddr(), "gateway base URL")
	key := fs.String("key", os.Getenv("FAK_KEY"), "bearer credential (only if the gateway sets --require-key)")
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
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak console sessions: encode json: %v\n", err)
			return 1
		}
		return 0
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
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak console garden: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprint(stdout, renderTUIGarden(report, *width))
	return 0
}

func runTUIGuard(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("tui guard", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var guardJSON stringList
	fs.Var(&guardJSON, "guard-json", "read a guard artifact JSON file (repeatable)")
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
	if len(guardJSON) == 0 {
		fmt.Fprintln(stderr, "fak console guard: at least one --guard-json artifact is required")
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
	artifacts, err := loadTUIGuard([]string(guardJSON))
	if err != nil {
		fmt.Fprintf(stderr, "fak console guard: %v\n", err)
		return 1
	}
	report := buildTUIGuardReport(artifacts, at)
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak console guard: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprint(stdout, renderTUIGuard(report, *width))
	return 0
}

func runTUIAgent(stdout, stderr io.Writer, argv []string) int {
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
	policyPath := fs.String("policy", "", "capability-floor manifest for the guard child (default: built-in guard floor)")
	model := fs.String("model", "", "upstream Claude model override for the guard child")
	sessionID := fs.String("session-id", "tui-agent", "trace/session id for the guard session")
	contextBudget := fs.Int("context-budget-tokens", 0, "seed a context-token budget in the guard session")
	restartOnBudget := fs.Bool("restart-on-budget", false, "ask guard to relaunch Claude on context-budget exhaustion")
	restartLimit := fs.Int("restart-limit", 0, "maximum guard relaunches for --restart-on-budget; 0 means unlimited")
	passthrough := fs.Bool("passthrough", false, "do not force subscription OAuth; let Claude Code forward its own credential")
	atText := fs.String("at", "", "snapshot time (RFC3339 or YYYY-MM-DD, default: now)")
	width := fs.Int("width", 120, "target terminal width for dry-run human rendering")
	dryRun := fs.Bool("dry-run", false, "render the launch plan without starting the backend agent")
	asJSON := fs.Bool("json", false, "emit the launch model as JSON and do not start the backend agent")
	if err := fs.Parse(argv); err != nil {
		return 2
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
	at, err := parseTUITime(*atText)
	if err != nil {
		fmt.Fprintf(stderr, "fak console agent: %v\n", err)
		return 2
	}
	report, err := buildTUIAgentReport(tuiAgentOptions{
		Backend:             *backend,
		Command:             *command,
		CommandArgs:         fs.Args(),
		Prompt:              *prompt,
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
	}, at, tuiExecutable(), os.Getenv)
	if err != nil {
		fmt.Fprintf(stderr, "fak console agent: %v\n", err)
		return 2
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak console agent: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	if *dryRun {
		fmt.Fprint(stdout, renderTUIAgent(report, *width))
		return 0
	}
	return launchTUIAgent(stdout, stderr, report)
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
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak console overview: encode json: %v\n", err)
			return 1
		}
		return 0
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

	command := append([]string{commandName}, opt.CommandArgs...)
	if strings.TrimSpace(opt.Prompt) != "" {
		command = append(command, "-p", opt.Prompt)
	}

	env, cfgDir, cfgSource, resolvedAccount, identity, notes, err := resolveTUIAgentClaudeConfig(opt, getenv)
	if err != nil {
		return tuiAgentReport{}, err
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
	guardArgs = append(guardArgs, "--")
	launch := append([]string{fakPath}, guardArgs...)
	launch = append(launch, command...)

	return tuiAgentReport{
		Schema:          tuiAgentSchema,
		At:              at.UTC().Format(time.RFC3339),
		Backend:         backend,
		Mode:            "launch",
		Provider:        "anthropic",
		Auth:            auth,
		Account:         strings.TrimSpace(opt.Account),
		ResolvedAccount: resolvedAccount,
		AccountIdentity: identity,
		ClaudeConfigDir: cfgDir,
		ConfigSource:    cfgSource,
		SessionID:       sessionID,
		Policy:          strings.TrimSpace(opt.Policy),
		Model:           strings.TrimSpace(opt.Model),
		ContextBudget:   opt.ContextBudgetTokens,
		RestartOnBudget: opt.RestartOnBudget,
		RestartLimit:    opt.RestartLimit,
		Command:         command,
		Launch:          launch,
		Env:             env,
		Notes:           notes,
	}, nil
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
		id := acct.DeriveIdentity(home.Dir)
		if !id.HasCreds {
			notes = append(notes, fmt.Sprintf("%q (%s) has no live credentials; Claude may prompt for login", home.Name, home.Dir))
		}
		env = append(env, tuiAgentEnv{Name: "CLAUDE_CONFIG_DIR", Value: home.Dir, Source: "account:" + home.Name})
		return env, home.Dir, "account:" + home.Name, home.Name, id.Email, notes, nil
	}
	if dir := strings.TrimSpace(opt.ClaudeConfigDir); dir != "" {
		id := acct.DeriveIdentity(dir)
		if !id.HasCreds {
			notes = append(notes, fmt.Sprintf("%s has no live credentials; Claude may prompt for login", dir))
		}
		env = append(env, tuiAgentEnv{Name: "CLAUDE_CONFIG_DIR", Value: dir, Source: "flag"})
		return env, dir, "flag", "", id.Email, notes, nil
	}
	if dir := strings.TrimSpace(getenv("CLAUDE_CONFIG_DIR")); dir != "" {
		id := acct.DeriveIdentity(dir)
		return nil, dir, "inherited-env", "", id.Email, notes, nil
	}
	return nil, guardClaudeConfigDir(), "default", "", "", notes, nil
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
		line := name + "=" + pair.Value
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

func loadTUIGarden(path, workspace string, deep bool, timeout time.Duration) (gardenbundle.Payload, string, error) {
	if path != "" {
		b, err := os.ReadFile(path)
		if err != nil {
			return gardenbundle.Payload{}, "", err
		}
		payload, err := decodeTUIGarden(b)
		return payload, path, err
	}
	root := workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	commit := gardenbundle.HeadCommit(root)
	if gardenbundle.GardenOff() {
		return gardenbundle.SkippedPayload(root, commit), "live:garden-skipped", nil
	}
	results := gardenbundle.Collect(root, "", timeout, deep)
	return gardenbundle.Fold(results, root, commit), "live:garden-bundle", nil
}

func decodeTUIGarden(b []byte) (gardenbundle.Payload, error) {
	var raw struct {
		Schema     string `json:"schema"`
		OK         bool   `json:"ok"`
		Verdict    string `json:"verdict"`
		Finding    string `json:"finding"`
		Reason     string `json:"reason"`
		NextAction string `json:"next_action"`
		Workspace  string `json:"workspace"`
		Commit     string `json:"commit"`
		Members    []struct {
			Key      string         `json:"key"`
			Label    string         `json:"label"`
			Gates    bool           `json:"gates"`
			ExitCode int            `json:"exit_code"`
			State    string         `json:"state"`
			OK       bool           `json:"ok"`
			Verdict  string         `json:"verdict"`
			Detail   string         `json:"detail"`
			Counts   map[string]int `json:"counts"`
		} `json:"members"`
		MemberCount int      `json:"member_count"`
		Gating      []string `json:"gating"`
		Skipped     bool     `json:"skipped"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return gardenbundle.Payload{}, fmt.Errorf("garden JSON must be a fak garden envelope: %w", err)
	}
	if raw.Schema != "" && raw.Schema != gardenbundle.Schema {
		return gardenbundle.Payload{}, fmt.Errorf("garden JSON schema = %q, want %q", raw.Schema, gardenbundle.Schema)
	}
	members := make([]gardenbundle.MemberResult, 0, len(raw.Members))
	for _, m := range raw.Members {
		members = append(members, gardenbundle.MemberResult{
			Key:      m.Key,
			Label:    m.Label,
			Gates:    m.Gates,
			ExitCode: m.ExitCode,
			State:    m.State,
			OK:       m.OK,
			Verdict:  m.Verdict,
			Detail:   m.Detail,
			Counts:   m.Counts,
		})
	}
	if raw.MemberCount == 0 {
		raw.MemberCount = len(members)
	}
	return gardenbundle.Payload{
		OK:          raw.OK,
		Verdict:     raw.Verdict,
		Finding:     raw.Finding,
		Reason:      raw.Reason,
		NextAction:  raw.NextAction,
		Workspace:   raw.Workspace,
		Commit:      raw.Commit,
		Members:     members,
		MemberCount: raw.MemberCount,
		Gating:      raw.Gating,
		Skipped:     raw.Skipped,
	}, nil
}

func loadTUIGuard(paths []string) ([]tuiGuardArtifact, error) {
	artifacts := make([]tuiGuardArtifact, 0, len(paths))
	for _, path := range paths {
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		var raw map[string]any
		if err := json.Unmarshal(b, &raw); err != nil {
			return nil, fmt.Errorf("%s: guard JSON must be an object: %w", path, err)
		}
		artifacts = append(artifacts, tuiGuardArtifact{Path: path, Raw: raw})
	}
	return artifacts, nil
}

func loadTUIIssues(path, repo, state string, limit int) ([]tuiIssue, string, error) {
	if path != "" {
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, "", err
		}
		issues, err := decodeTUIIssues(b)
		return issues, path, err
	}
	args := []string{
		"issue", "list",
		"--state", state,
		"--limit", strconv.Itoa(limit),
		"--json", "number,title,url,state,body,labels,createdAt,updatedAt,author,assignees,milestone,comments",
	}
	if repo != "" {
		args = append(args, "--repo", repo)
	}
	cmd := exec.Command("gh", args...)
	var stderr strings.Builder
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return nil, "", fmt.Errorf("gh %s: %s", strings.Join(args, " "), msg)
	}
	issues, err := decodeTUIIssues(out)
	if err != nil {
		return nil, "", err
	}
	source := "gh issue list"
	if repo != "" {
		source += " --repo " + repo
	}
	return issues, source, nil
}

func decodeTUIIssues(b []byte) ([]tuiIssue, error) {
	var issues []tuiIssue
	if err := json.Unmarshal(b, &issues); err != nil {
		return nil, fmt.Errorf("issue JSON must be a gh issue list array: %w", err)
	}
	for i := range issues {
		if issues[i].State == "" {
			issues[i].State = "OPEN"
		}
	}
	return issues, nil
}

func buildTUIIssueReport(issues []tuiIssue, source string, asOf time.Time, epic int) tuiIssueReport {
	dups := tuiDuplicateGroups(issues)
	rows := make([]tuiIssueRow, 0, len(issues))
	var epicRow *tuiIssueRow
	for _, issue := range issues {
		row := classifyTUIIssue(issue, asOf, dups)
		if epic > 0 {
			row.Related = issue.Number == epic || tuiIssueReferences(issue, epic)
		}
		if issue.Number == epic {
			cp := row
			epicRow = &cp
		}
		rows = append(rows, row)
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].Score != rows[j].Score {
			return rows[i].Score > rows[j].Score
		}
		return rows[i].Number > rows[j].Number
	})
	return tuiIssueReport{
		Schema:  tuiIssuesSchema,
		AsOf:    asOf.Format("2006-01-02"),
		Source:  source,
		Epic:    epicRow,
		Counts:  countTUIIssues(rows),
		Lanes:   buildTUILanes(rows),
		Rows:    rows,
		Actions: buildTUIActions(rows),
	}
}

func classifyTUIIssue(issue tuiIssue, asOf time.Time, dups map[int]int) tuiIssueRow {
	labels := tuiLabelNames(issue)
	labelSet := map[string]bool{}
	for _, label := range labels {
		labelSet[label] = true
	}
	prio := ""
	for _, p := range []string{"priority/P0", "priority/P1", "priority/P2"} {
		if labelSet[p] {
			prio = p
			break
		}
	}
	assigned := len(issue.Assignees) > 0
	inProgress := labelSet["in-progress"]
	ageDays := tuiDaysSince(issue.CreatedAt, asOf)
	idleDays := tuiDaysSince(issue.UpdatedAt, asOf)
	tags := []string{}
	if prio == "" {
		tags = append(tags, "needs-priority")
	}
	if !tuiHasAny(labelSet, tuiKindLabels) {
		tags = append(tags, "needs-kind")
	}
	if !tuiHasAny(labelSet, tuiAreaLabels) {
		tags = append(tags, "needs-area")
	}
	if len(labels) == 0 {
		tags = append(tags, "bare")
	}
	if (prio == "priority/P0" || prio == "priority/P1") && !inProgress && !assigned {
		tags = append(tags, "orphan")
	}
	if idleDays >= 60 && !inProgress {
		tags = append(tags, "stale")
	}
	if labelSet["question"] && idleDays >= 30 {
		tags = append(tags, "dormant-question")
	}
	if _, ok := dups[issue.Number]; ok {
		tags = append(tags, "likely-dup")
	}

	score := tuiPriorityWeights[prio]
	if score == 0 {
		score = 60
	}
	if (prio == "priority/P0" || prio == "priority/P1") && !inProgress && !assigned {
		score += 300
	}
	if labelSet["bug"] {
		score += 40
	}
	if labelSet["documentation"] {
		score -= 20
	}
	if idleDays > 90 {
		score += 90
	} else {
		score += idleDays
	}
	if labelSet["question"] && idleDays < 30 {
		score -= 200
	}

	return tuiIssueRow{
		Number:     issue.Number,
		Title:      issue.Title,
		URL:        issue.URL,
		State:      issue.State,
		Labels:     labels,
		Author:     tuiLogin(issue.Author),
		Assignees:  tuiAssigneeLogins(issue.Assignees),
		Milestone:  tuiMilestoneTitle(issue.Milestone),
		Comments:   int(issue.Comments),
		AgeDays:    ageDays,
		IdleDays:   idleDays,
		Priority:   prio,
		InProgress: inProgress,
		Tags:       tags,
		Score:      score,
	}
}

func tuiLabelNames(issue tuiIssue) []string {
	labels := make([]string, 0, len(issue.Labels))
	seen := map[string]bool{}
	for _, label := range issue.Labels {
		name := strings.TrimSpace(label.Name)
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		labels = append(labels, name)
	}
	sort.Strings(labels)
	return labels
}

func tuiHasAny(labels map[string]bool, allowed map[string]bool) bool {
	for label := range labels {
		if allowed[label] {
			return true
		}
	}
	return false
}

func tuiDaysSince(iso string, asOf time.Time) int {
	if strings.TrimSpace(iso) == "" {
		return 0
	}
	t, err := time.Parse(time.RFC3339, iso)
	if err != nil {
		return 0
	}
	days := int(asOf.Sub(t.UTC()).Hours() / 24)
	if days < 0 {
		return 0
	}
	return days
}

func tuiLogin(u *tuiUser) string {
	if u == nil {
		return ""
	}
	return u.Login
}

func tuiAssigneeLogins(users []tuiUser) []string {
	out := make([]string, 0, len(users))
	for _, u := range users {
		if u.Login != "" {
			out = append(out, u.Login)
		}
	}
	sort.Strings(out)
	return out
}

func tuiMilestoneTitle(m *tuiMilestone) string {
	if m == nil {
		return ""
	}
	return m.Title
}

func countTUIIssues(rows []tuiIssueRow) tuiIssueCounts {
	var c tuiIssueCounts
	for _, row := range rows {
		if strings.EqualFold(row.State, "closed") {
			continue
		}
		c.Open++
		switch row.Priority {
		case "priority/P0":
			c.P0++
		case "priority/P1":
			c.P1++
		case "priority/P2":
			c.P2++
		}
		if row.Related {
			c.Related++
		}
		for _, tag := range row.Tags {
			switch tag {
			case "needs-priority":
				c.NeedsPriority++
			case "needs-kind":
				c.NeedsKind++
			case "needs-area":
				c.NeedsArea++
			case "orphan":
				c.Orphan++
			case "stale":
				c.Stale++
			case "dormant-question":
				c.DormantQuestion++
			case "likely-dup":
				c.LikelyDup++
			case "bare":
				c.Bare++
			}
		}
	}
	return c
}

func buildTUILanes(rows []tuiIssueRow) []tuiLane {
	names := []string{"priority/P0", "priority/P1", "priority/P2", "unprioritized"}
	lanes := make([]tuiLane, 0, len(names))
	for _, name := range names {
		lane := tuiLane{Name: name}
		for _, row := range rows {
			if row.Priority != name && !(name == "unprioritized" && row.Priority == "") {
				continue
			}
			lane.Count++
			if tuiHasTag(row, "orphan") {
				lane.Orphan++
			}
			if tuiHasTag(row, "needs-area") {
				lane.NeedsArea++
			}
			if tuiHasTag(row, "needs-kind") {
				lane.NeedsKind++
			}
			if row.IdleDays > lane.MaxIdleDays {
				lane.MaxIdleDays = row.IdleDays
			}
			if lane.TopIssue == 0 {
				lane.TopIssue = row.Number
				lane.TopIssueText = row.Title
			}
		}
		lanes = append(lanes, lane)
	}
	return lanes
}

func buildTUIActions(rows []tuiIssueRow) []tuiIssueAction {
	actions := []tuiIssueAction{}
	for _, row := range rows {
		switch {
		case tuiHasTag(row, "dormant-question"):
			actions = append(actions, tuiIssueAction{
				Number: row.Number,
				Kind:   "close-dormant-question",
				Reason: fmt.Sprintf("question idle %dd", row.IdleDays),
				Command: fmt.Sprintf("gh issue close %d --reason \"not planned\" --comment \"Closing as dormant: question idle %dd. Reopen with new info if it is still live.\"",
					row.Number, row.IdleDays),
			})
		case tuiHasTag(row, "stale") && row.Priority != "priority/P0" && row.Priority != "priority/P1":
			actions = append(actions, tuiIssueAction{
				Number:  row.Number,
				Kind:    "mark-stale",
				Reason:  fmt.Sprintf("idle %dd, not in-progress, not P0/P1", row.IdleDays),
				Command: fmt.Sprintf("gh issue edit %d --add-label \"stale\"", row.Number),
			})
		case len(row.Tags) > 0:
			actions = append(actions, tuiIssueAction{
				Number: row.Number,
				Kind:   "review",
				Reason: strings.Join(row.Tags, ", "),
			})
		}
	}
	return actions
}

func tuiHasTag(row tuiIssueRow, tag string) bool {
	for _, got := range row.Tags {
		if got == tag {
			return true
		}
	}
	return false
}

func tuiIssueReferences(issue tuiIssue, epic int) bool {
	ref := "#" + strconv.Itoa(epic)
	return strings.Contains(issue.Title, ref) || strings.Contains(issue.Body, ref)
}

func tuiDuplicateGroups(issues []tuiIssue) map[int]int {
	type pair struct {
		num int
		tok map[string]bool
	}
	pairs := make([]pair, 0, len(issues))
	for _, issue := range issues {
		pairs = append(pairs, pair{num: issue.Number, tok: tuiTitleTokens(issue.Title)})
	}
	parent := map[int]int{}
	for _, p := range pairs {
		parent[p.num] = p.num
	}
	var find func(int) int
	find = func(x int) int {
		for parent[x] != x {
			parent[x] = parent[parent[x]]
			x = parent[x]
		}
		return x
	}
	union := func(a, b int) {
		ra, rb := find(a), find(b)
		if ra != rb {
			parent[ra] = rb
		}
	}
	for i := 0; i < len(pairs); i++ {
		for j := i + 1; j < len(pairs); j++ {
			if tuiJaccard(pairs[i].tok, pairs[j].tok) >= 0.60 {
				union(pairs[i].num, pairs[j].num)
			}
		}
	}
	members := map[int][]int{}
	for _, p := range pairs {
		root := find(p.num)
		members[root] = append(members[root], p.num)
	}
	out := map[int]int{}
	gid := 0
	for _, nums := range members {
		if len(nums) < 2 {
			continue
		}
		for _, n := range nums {
			out[n] = gid
		}
		gid++
	}
	return out
}

func tuiTitleTokens(title string) map[string]bool {
	stop := map[string]bool{
		"the": true, "and": true, "for": true, "with": true, "issue": true,
		"feat": true, "fix": true, "add": true, "new": true, "needs": true,
		"work": true, "support": true, "implement": true,
	}
	out := map[string]bool{}
	for _, m := range tuiScopeRE.FindAllStringSubmatch(title, -1) {
		if len(m) == 3 {
			out[strings.ToLower(m[0])] = true
			out[strings.ToLower(m[2])] = true
		}
	}
	for _, word := range tuiWordRE.FindAllString(title, -1) {
		w := strings.ToLower(word)
		if !stop[w] {
			out[w] = true
		}
	}
	return out
}

func tuiJaccard(a, b map[string]bool) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	inter := 0
	for k := range a {
		if b[k] {
			inter++
		}
	}
	union := len(a)
	for k := range b {
		if !a[k] {
			union++
		}
	}
	if union == 0 {
		return 0
	}
	return float64(inter) / float64(union)
}

func buildTUIGardenReport(payload gardenbundle.Payload, source string, at time.Time, includeGate bool) tuiGardenReport {
	rows := make([]tuiGardenRow, 0, len(payload.Members))
	for _, member := range payload.Members {
		rows = append(rows, classifyTUIGardenMember(member))
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].Attention != rows[j].Attention {
			return rows[i].Attention > rows[j].Attention
		}
		return rows[i].Key < rows[j].Key
	})
	counts := countTUIGarden(rows)
	if payload.Skipped {
		counts.Skipped = 1
	}
	gateExit := 0
	gateMessage := ""
	if includeGate {
		gateExit, gateMessage = gardenbundle.CheckGate(payload)
	}
	return tuiGardenReport{
		Schema:      tuiGardenSchema,
		At:          at.UTC().Format(time.RFC3339),
		Source:      source,
		Workspace:   payload.Workspace,
		Commit:      payload.Commit,
		OK:          payload.OK,
		Verdict:     payload.Verdict,
		Finding:     payload.Finding,
		Reason:      payload.Reason,
		NextAction:  payload.NextAction,
		GateExit:    gateExit,
		GateMessage: gateMessage,
		Counts:      counts,
		Rows:        rows,
	}
}

func classifyTUIGardenMember(member gardenbundle.MemberResult) tuiGardenRow {
	row := tuiGardenRow{
		Key:      member.Key,
		Label:    member.Label,
		State:    member.State,
		OK:       member.OK,
		Gates:    member.Gates,
		ExitCode: member.ExitCode,
		Verdict:  member.Verdict,
		Detail:   member.Detail,
		Counts:   member.Counts,
	}
	row.Tags, row.Attention = scoreTUIGardenRow(row)
	return row
}

func scoreTUIGardenRow(row tuiGardenRow) ([]string, int) {
	tags := []string{}
	score := 0
	switch row.State {
	case "errored":
		tags = append(tags, "errored")
		score += 100
	case "red":
		tags = append(tags, "red")
		score += 90
	case "action":
		tags = append(tags, "action")
		score += 55
	case "ok":
		tags = append(tags, "ok")
	default:
		tags = append(tags, "unknown")
		score += 20
	}
	if row.Gates {
		tags = append(tags, "gates")
		score += 20
	}
	if row.ExitCode != 0 {
		tags = append(tags, "nonzero-exit")
		score += 10
	}
	if row.Counts != nil {
		if row.Counts["broken"] > 0 {
			tags = append(tags, "broken-loops")
			score += row.Counts["broken"] * 20
		}
		if row.Counts["action"] > 0 {
			tags = append(tags, "loop-action")
			score += row.Counts["action"] * 10
		}
	}
	return tags, score
}

func countTUIGarden(rows []tuiGardenRow) tuiGardenCounts {
	var c tuiGardenCounts
	for _, row := range rows {
		c.Members++
		if row.Gates {
			c.Gating++
		}
		switch row.State {
		case "ok":
			c.OK++
		case "action":
			c.Action++
		case "red":
			c.Red++
		case "errored":
			c.Errored++
		}
	}
	return c
}

func renderTUIGarden(report tuiGardenReport, width int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak console garden  at=%s  source=%s\n", report.At, report.Source)
	fmt.Fprintf(&b, "verdict=%s  finding=%s  ok=%v  members=%d  action=%d  red=%d  errored=%d  gating=%d\n",
		report.Verdict, report.Finding, report.OK, report.Counts.Members, report.Counts.Action,
		report.Counts.Red, report.Counts.Errored, report.Counts.Gating)
	if report.GateMessage != "" {
		fmt.Fprintf(&b, "gate=%d  %s\n", report.GateExit, trimTUI(report.GateMessage, width-8))
	}
	if report.Workspace != "" || report.Commit != "" {
		fmt.Fprintf(&b, "workspace=%s  commit=%s\n", report.Workspace, report.Commit)
	}
	if report.Reason != "" {
		fmt.Fprintf(&b, "reason: %s\n", trimTUI(report.Reason, maxTUI(20, width-8)))
	}
	if report.NextAction != "" {
		fmt.Fprintf(&b, "next:   %s\n", trimTUI(report.NextAction, maxTUI(20, width-8)))
	}
	if len(report.Rows) == 0 {
		if report.Counts.Skipped > 0 {
			fmt.Fprintln(&b, "\n(skipped)")
		} else {
			fmt.Fprintln(&b, "\nno garden members")
		}
		return b.String()
	}
	fmt.Fprintln(&b, "\nMembers")
	fmt.Fprintln(&b, "attention member                    state    gate exit verdict tags")
	for _, row := range report.Rows {
		gate := "-"
		if row.Gates {
			gate = "yes"
		}
		tags := displayTUITags(row.Tags, 4)
		detail := row.Detail
		if detail != "" {
			tags += "  " + detail
		}
		fmt.Fprintf(&b, "%9d %-25s %-8s %-4s %-4d %-7s %s\n",
			row.Attention, trimTUI(row.Label, 25), trimTUI(row.State, 8), gate, row.ExitCode,
			trimTUI(row.Verdict, 7), trimTUI(tags, maxTUI(14, width-66)))
	}
	return b.String()
}

func buildTUIGuardReport(artifacts []tuiGuardArtifact, at time.Time) tuiGuardReport {
	rows := []tuiGuardRow{}
	sources := make([]tuiGuardSource, 0, len(artifacts))
	for _, artifact := range artifacts {
		schema := tuiGuardString(artifact.Raw, "schema")
		status := tuiGuardString(artifact.Raw, "status")
		sources = append(sources, tuiGuardSource{
			Path:   artifact.Path,
			Schema: schema,
			Status: status,
		})
		rows = append(rows, tuiGuardRowsForArtifact(artifact)...)
	}
	for i := range rows {
		rows[i].Tags, rows[i].Attention = scoreTUIGuardRow(rows[i])
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].Attention != rows[j].Attention {
			return rows[i].Attention > rows[j].Attention
		}
		if rows[i].Artifact != rows[j].Artifact {
			return rows[i].Artifact < rows[j].Artifact
		}
		return rows[i].Kind < rows[j].Kind
	})
	counts := countTUIGuard(rows, sources)
	status := "PASS"
	switch {
	case counts.Fail > 0 || counts.Unexpected > 0:
		status = "FAIL"
	case counts.Warn > 0:
		status = "WARN"
	}
	return tuiGuardReport{
		Schema:  tuiGuardSchema,
		At:      at.UTC().Format(time.RFC3339),
		Source:  tuiGuardSourceLabel(artifacts),
		Status:  status,
		Counts:  counts,
		Actions: tuiGuardActions(counts),
		Rows:    rows,
		Sources: sources,
	}
}

func tuiGuardRowsForArtifact(artifact tuiGuardArtifact) []tuiGuardRow {
	raw := artifact.Raw
	schema := tuiGuardString(raw, "schema")
	if preflight := tuiGuardMap(raw["preflight"]); preflight != nil {
		return []tuiGuardRow{tuiGuardPreflightProbeRow(artifact, preflight)}
	}
	if schema == "fak-claude-historical-guard-audit/1" || raw["verdict_counts"] != nil || raw["non_allow_samples"] != nil {
		return tuiGuardHistoricalRows(artifact)
	}
	if schema == "fak-vendor-live-pilot/1" || raw["dangerous_attempt"] != nil {
		return tuiGuardVendorRows(artifact)
	}
	if schema == "fak-combined-dogfood/1" || raw["floor"] != nil {
		return tuiGuardCombinedRows(artifact)
	}
	if checks, ok := raw["checks"].([]any); ok {
		return tuiGuardCheckRows(artifact, checks)
	}
	row := tuiGuardGenericRow(artifact)
	if row.Kind == "" {
		return nil
	}
	return []tuiGuardRow{row}
}

func tuiGuardPreflightProbeRow(artifact tuiGuardArtifact, preflight map[string]any) tuiGuardRow {
	raw := artifact.Raw
	status := tuiGuardString(raw, "status")
	verdict := tuiGuardString(preflight, "verdict")
	reason := tuiGuardString(preflight, "reason")
	expected := ""
	if tuiGuardBool(raw, "expect_deny") {
		expected = "expected deny"
		if want := tuiGuardString(raw, "expect_reason"); want != "" {
			expected += " " + want
		}
	}
	detail := strings.TrimSpace(strings.Join(nonEmptyTUI([]string{
		tuiGuardString(raw, "command_label"),
		tuiGuardString(raw, "policy"),
		expected,
	}), "  "))
	return tuiGuardRow{
		Artifact: tuiGuardArtifactName(artifact.Path),
		Kind:     "preflight-probe",
		Tool:     tuiGuardString(raw, "tool"),
		Verdict:  verdict,
		Reason:   reason,
		By:       tuiGuardString(preflight, "by"),
		Status:   status,
		Detail:   detail,
		Count:    1,
	}
}

func tuiGuardHistoricalRows(artifact tuiGuardArtifact) []tuiGuardRow {
	raw := artifact.Raw
	artifactName := tuiGuardArtifactName(artifact.Path)
	rows := []tuiGuardRow{}
	for verdict, count := range tuiGuardIntMap(raw["verdict_counts"]) {
		rows = append(rows, tuiGuardRow{
			Artifact: artifactName,
			Kind:     "verdict-count",
			Verdict:  strings.ToUpper(verdict),
			Status:   tuiGuardString(raw, "status"),
			Detail:   tuiGuardHistoricalDetail(raw),
			Count:    count,
		})
	}
	for reason, count := range tuiGuardIntMap(raw["reason_counts"]) {
		rows = append(rows, tuiGuardRow{
			Artifact: artifactName,
			Kind:     "reason-count",
			Reason:   strings.ToUpper(reason),
			Status:   tuiGuardString(raw, "status"),
			Detail:   tuiGuardString(raw, "policy"),
			Count:    count,
		})
	}
	if samples, ok := raw["non_allow_samples"].([]any); ok {
		for _, sample := range samples {
			m := tuiGuardMap(sample)
			if m == nil {
				continue
			}
			rows = append(rows, tuiGuardRow{
				Artifact: artifactName,
				Kind:     "sample",
				Tool:     tuiGuardString(m, "tool"),
				Verdict:  tuiGuardString(m, "verdict"),
				Reason:   tuiGuardString(m, "reason"),
				By:       tuiGuardString(m, "by"),
				Status:   tuiGuardString(raw, "status"),
				Detail:   firstNonEmptyTUI(tuiGuardString(m, "claim"), tuiGuardString(m, "call_digest")),
			})
		}
	}
	return rows
}

func tuiGuardHistoricalDetail(raw map[string]any) string {
	bits := []string{}
	if policy := tuiGuardString(raw, "policy"); policy != "" {
		bits = append(bits, "policy="+policy)
	}
	if calls := tuiGuardInt(raw, "tool_calls_seen"); calls > 0 {
		bits = append(bits, fmt.Sprintf("tool_calls=%d", calls))
	}
	if sessions := tuiGuardInt(raw, "sessions_audited"); sessions > 0 {
		bits = append(bits, fmt.Sprintf("sessions=%d", sessions))
	}
	return strings.Join(bits, "  ")
}

func tuiGuardVendorRows(artifact tuiGuardArtifact) []tuiGuardRow {
	raw := artifact.Raw
	artifactName := tuiGuardArtifactName(artifact.Path)
	rows := []tuiGuardRow{}
	if m := tuiGuardMap(raw["dangerous_attempt"]); m != nil {
		rows = append(rows, tuiGuardDecisionRow(artifactName, "dangerous-attempt", tuiGuardString(raw, "status"), m))
	}
	if m := tuiGuardMap(raw["useful_continuation"]); m != nil {
		rows = append(rows, tuiGuardDecisionRow(artifactName, "useful-continuation", tuiGuardString(raw, "status"), m))
	}
	return rows
}

func tuiGuardCombinedRows(artifact tuiGuardArtifact) []tuiGuardRow {
	raw := artifact.Raw
	floor := tuiGuardMap(raw["floor"])
	if floor == nil {
		return nil
	}
	artifactName := tuiGuardArtifactName(artifact.Path)
	status := tuiGuardString(raw, "verdict")
	denyReason := ""
	if strings.Contains(tuiGuardString(floor, "deny_response_excerpt"), "POLICY_BLOCK") {
		denyReason = "POLICY_BLOCK"
	}
	return []tuiGuardRow{
		{
			Artifact: artifactName,
			Kind:     "floor-deny",
			Tool:     tuiGuardString(floor, "deny_call"),
			Verdict:  "DENY",
			Reason:   denyReason,
			Status:   status,
			Detail:   fmt.Sprintf("pass=%v", tuiGuardBool(floor, "pass")),
			Count:    1,
		},
		{
			Artifact: artifactName,
			Kind:     "floor-allow",
			Tool:     tuiGuardString(floor, "allow_call"),
			Verdict:  "ALLOW",
			Status:   status,
			Detail:   fmt.Sprintf("pass=%v", tuiGuardBool(floor, "pass")),
			Count:    1,
		},
	}
}

func tuiGuardCheckRows(artifact tuiGuardArtifact, checks []any) []tuiGuardRow {
	rows := []tuiGuardRow{}
	for _, check := range checks {
		m := tuiGuardMap(check)
		if m == nil {
			continue
		}
		rows = append(rows, tuiGuardRow{
			Artifact: tuiGuardArtifactName(artifact.Path),
			Kind:     "check",
			Status:   tuiGuardString(m, "status"),
			Detail:   strings.TrimSpace(tuiGuardString(m, "name") + "  " + tuiGuardString(m, "detail")),
		})
	}
	return rows
}

func tuiGuardGenericRow(artifact tuiGuardArtifact) tuiGuardRow {
	raw := artifact.Raw
	verdict := firstNonEmptyTUI(tuiGuardString(raw, "verdict"), tuiGuardString(raw, "kind"))
	status := tuiGuardString(raw, "status")
	if verdict == "" && status == "" {
		return tuiGuardRow{}
	}
	return tuiGuardRow{
		Artifact: tuiGuardArtifactName(artifact.Path),
		Kind:     "artifact",
		Tool:     tuiGuardString(raw, "tool"),
		Verdict:  verdict,
		Reason:   tuiGuardString(raw, "reason"),
		By:       tuiGuardString(raw, "by"),
		Status:   status,
		Detail:   tuiGuardString(raw, "finding"),
		Count:    1,
	}
}

func tuiGuardDecisionRow(artifact, kind, status string, m map[string]any) tuiGuardRow {
	tool := tuiGuardString(m, "tool")
	if args := tuiGuardMap(m["arguments"]); args != nil {
		tool = firstNonEmptyTUI(tuiGuardString(args, "tool"), tool)
	}
	verdict, reason, by := "", "", ""
	if audit := tuiGuardMap(m["fak_audit"]); audit != nil {
		verdict = tuiGuardString(audit, "verdict")
		reason = tuiGuardString(audit, "reason")
		by = tuiGuardString(audit, "by")
	}
	if verdict == "" {
		if v := tuiGuardMap(m["fak_verdict"]); v != nil {
			verdict = firstNonEmptyTUI(tuiGuardString(v, "kind"), tuiGuardString(v, "verdict"))
			reason = tuiGuardString(v, "reason")
			by = tuiGuardString(v, "by")
		}
	}
	detail := strings.Join(nonEmptyTUI([]string{
		tuiGuardString(m, "mcp_tool"),
		tuiGuardString(m, "trace_id"),
		tuiGuardString(m, "assistant_visible_refusal"),
	}), "  ")
	return tuiGuardRow{
		Artifact: artifact,
		Kind:     kind,
		Tool:     tool,
		Verdict:  verdict,
		Reason:   reason,
		By:       by,
		Status:   status,
		Detail:   detail,
		Count:    1,
	}
}

func scoreTUIGuardRow(row tuiGuardRow) ([]string, int) {
	tags := []string{}
	score := 0
	status := strings.ToUpper(row.Status)
	verdict := strings.ToUpper(row.Verdict)
	reason := strings.ToUpper(row.Reason)
	switch verdict {
	case "DENY":
		tags = append(tags, "deny")
		score += 45
	case "ALLOW":
		tags = append(tags, "allow")
		score += 5
	case "TRANSFORM":
		tags = append(tags, "transform")
		score += 25
	case "QUARANTINE":
		tags = append(tags, "quarantine")
		score += 70
	}
	switch reason {
	case "POLICY_BLOCK":
		tags = append(tags, "policy-block")
		score += 25
	case "DEFAULT_DENY":
		tags = append(tags, "default-deny")
		score += 15
	case "TRUST_VIOLATION":
		tags = append(tags, "trust-violation")
		score += 30
	}
	if row.Kind == "sample" {
		tags = append(tags, "sample")
		score += 15
	}
	if strings.Contains(status, "DENIED_EXPECTED") {
		tags = append(tags, "expected-deny")
		score += 5
	}
	if status == "FAIL" || strings.Contains(status, "UNEXPECTED") || strings.Contains(status, "ERROR") || strings.Contains(status, "BLOCKED") {
		tags = append(tags, "unexpected")
		score += 100
	}
	if row.Count > 1 {
		score += minTUI(row.Count, 50)
	}
	if len(tags) == 0 {
		tags = append(tags, "status")
	}
	return tags, score
}

func countTUIGuard(rows []tuiGuardRow, sources []tuiGuardSource) tuiGuardCounts {
	c := tuiGuardCounts{Artifacts: len(sources), Rows: len(rows)}
	for _, source := range sources {
		switch tuiGuardStatusClass(source.Status) {
		case "pass":
			c.Pass++
		case "warn":
			c.Warn++
		case "fail":
			c.Fail++
		}
	}
	for _, row := range rows {
		n := row.Count
		if n <= 0 {
			if hasStringTUI(row.Tags, "unexpected") {
				c.Unexpected++
			}
			continue
		}
		switch strings.ToUpper(row.Verdict) {
		case "ALLOW":
			c.Allow += n
		case "DENY":
			c.Deny += n
		case "TRANSFORM":
			c.Transform += n
		case "QUARANTINE":
			c.Quarantine += n
		}
		switch strings.ToUpper(row.Reason) {
		case "POLICY_BLOCK":
			c.PolicyBlock += n
		case "DEFAULT_DENY":
			c.DefaultDeny += n
		}
		if hasStringTUI(row.Tags, "expected-deny") {
			c.Expected += n
		}
		if hasStringTUI(row.Tags, "unexpected") {
			c.Unexpected += n
		}
	}
	return c
}

func tuiGuardStatusClass(status string) string {
	status = strings.ToUpper(strings.TrimSpace(status))
	switch {
	case status == "":
		return ""
	case status == "PASS" || status == "OK" || status == "DENIED_EXPECTED":
		return "pass"
	case strings.Contains(status, "WARN"):
		return "warn"
	case strings.Contains(status, "FAIL") || strings.Contains(status, "ERROR") || strings.Contains(status, "BLOCKED") || strings.Contains(status, "UNEXPECTED"):
		return "fail"
	default:
		return "pass"
	}
}

func tuiGuardActions(counts tuiGuardCounts) []string {
	switch {
	case counts.Unexpected > 0 || counts.Fail > 0:
		return []string{"inspect failing or unexpected guard artifacts before treating the proof packet as current"}
	case counts.Deny == 0 && counts.Quarantine == 0:
		return []string{"add a recent guard proof with at least one denial or quarantine"}
	case counts.PolicyBlock == 0:
		return []string{"capture a POLICY_BLOCK proof for destructive tool refusals"}
	default:
		return []string{"keep feeding recent guard artifacts into this pane; the denial surface is visible"}
	}
}

func renderTUIGuard(report tuiGuardReport, width int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak console guard  at=%s  source=%s\n", report.At, report.Source)
	fmt.Fprintf(&b, "status=%s  artifacts=%d  rows=%d  allow=%d  deny=%d  quarantine=%d  policy_block=%d  default_deny=%d  expected=%d  unexpected=%d\n",
		report.Status, report.Counts.Artifacts, report.Counts.Rows, report.Counts.Allow,
		report.Counts.Deny, report.Counts.Quarantine, report.Counts.PolicyBlock,
		report.Counts.DefaultDeny, report.Counts.Expected, report.Counts.Unexpected)
	if len(report.Actions) > 0 {
		fmt.Fprintf(&b, "next: %s\n", trimTUI(report.Actions[0], maxTUI(20, width-6)))
	}
	if len(report.Rows) == 0 {
		fmt.Fprintln(&b, "\nno guard rows")
		return b.String()
	}
	fmt.Fprintln(&b, "\nRows")
	fmt.Fprintln(&b, "attention artifact                 kind                 tool             verdict reason         count tags")
	for _, row := range report.Rows {
		count := "-"
		if row.Count > 0 {
			count = strconv.Itoa(row.Count)
		}
		tags := displayTUITags(row.Tags, 4)
		if row.Detail != "" {
			tags += "  " + row.Detail
		}
		fmt.Fprintf(&b, "%9d %-24s %-20s %-16s %-7s %-14s %-5s %s\n",
			row.Attention, trimTUI(row.Artifact, 24), trimTUI(row.Kind, 20), trimTUI(row.Tool, 16),
			trimTUI(row.Verdict, 7), trimTUI(row.Reason, 14), count, trimTUI(tags, maxTUI(12, width-91)))
	}
	return b.String()
}

func tuiGuardSourceLabel(artifacts []tuiGuardArtifact) string {
	if len(artifacts) == 1 {
		return artifacts[0].Path
	}
	return fmt.Sprintf("%d artifacts", len(artifacts))
}

func tuiGuardArtifactName(path string) string {
	name := filepath.Base(path)
	if name == "." || name == string(filepath.Separator) {
		return path
	}
	return name
}

func tuiGuardMap(v any) map[string]any {
	m, ok := v.(map[string]any)
	if !ok {
		return nil
	}
	return m
}

func tuiGuardString(m map[string]any, key string) string {
	if m == nil {
		return ""
	}
	v := m[key]
	if v == nil {
		return ""
	}
	switch s := v.(type) {
	case string:
		return s
	case bool:
		return strconv.FormatBool(s)
	case float64:
		if s == float64(int64(s)) {
			return strconv.FormatInt(int64(s), 10)
		}
		return strconv.FormatFloat(s, 'f', -1, 64)
	default:
		return fmt.Sprintf("%v", s)
	}
}

func tuiGuardBool(m map[string]any, key string) bool {
	if m == nil {
		return false
	}
	b, ok := m[key].(bool)
	return ok && b
}

func tuiGuardInt(m map[string]any, key string) int {
	if m == nil {
		return 0
	}
	switch n := m[key].(type) {
	case float64:
		return int(n)
	case int:
		return n
	case string:
		i, _ := strconv.Atoi(strings.TrimSpace(n))
		return i
	default:
		return 0
	}
}

func tuiGuardIntMap(v any) map[string]int {
	out := map[string]int{}
	m, ok := v.(map[string]any)
	if !ok {
		return out
	}
	for k, raw := range m {
		out[k] = tuiGuardInt(map[string]any{"v": raw}, "v")
	}
	return out
}

func loadTUIOverview(opt tuiOverviewOptions) (tuiOverviewReport, error) {
	cards := []tuiOverviewCard{}
	if opt.IssuesJSON != "" {
		issues, source, err := loadTUIIssues(opt.IssuesJSON, "", "open", 1)
		if err != nil {
			return tuiOverviewReport{}, err
		}
		cards = append(cards, overviewIssueCard(buildTUIIssueReport(issues, source, opt.AsOf, opt.Epic), opt.Epic))
	} else {
		cards = append(cards, missingOverviewCard("issues", "fak console overview --issues-json issues.json --epic 837"))
	}
	if opt.Ledger != "" {
		st, err := loopmgr.SnapshotFile(opt.Ledger, opt.At)
		if err != nil {
			return tuiOverviewReport{}, err
		}
		cards = append(cards, overviewLoopCard(buildTUILoopReport(st, opt.At)))
	} else {
		cards = append(cards, missingOverviewCard("loops", "fak console overview --ledger .fak/loop-ledger.jsonl"))
	}
	if opt.SessionsJSON != "" {
		list, source, err := loadTUISessions(opt.SessionsJSON, "", "")
		if err != nil {
			return tuiOverviewReport{}, err
		}
		cards = append(cards, overviewSessionCard(buildTUISessionReport(list, source, opt.At)))
	} else {
		cards = append(cards, missingOverviewCard("sessions", "fak console overview --sessions-json sessions.json"))
	}
	if opt.GardenJSON != "" {
		payload, source, err := loadTUIGarden(opt.GardenJSON, "", false, 0)
		if err != nil {
			return tuiOverviewReport{}, err
		}
		cards = append(cards, overviewGardenCard(buildTUIGardenReport(payload, source, opt.At, opt.CheckGarden)))
	} else {
		cards = append(cards, missingOverviewCard("garden", "fak console overview --garden-json garden.json --check"))
	}
	if len(opt.GuardJSON) > 0 {
		artifacts, err := loadTUIGuard(opt.GuardJSON)
		if err != nil {
			return tuiOverviewReport{}, err
		}
		cards = append(cards, overviewGuardCard(buildTUIGuardReport(artifacts, opt.At)))
	} else {
		cards = append(cards, missingOverviewCard("guard", "fak console overview --guard-json guard-proof.json"))
	}
	sort.SliceStable(cards, func(i, j int) bool {
		if cards[i].Attention != cards[j].Attention {
			return cards[i].Attention > cards[j].Attention
		}
		return cards[i].Pane < cards[j].Pane
	})
	counts := countTUIOverview(cards)
	return tuiOverviewReport{
		Schema:  tuiOverviewSchema,
		At:      opt.At.UTC().Format(time.RFC3339),
		Source:  "selected panes",
		Counts:  counts,
		Cards:   cards,
		Actions: overviewActions(cards),
	}, nil
}

func overviewIssueCard(report tuiIssueReport, epic int) tuiOverviewCard {
	counts := map[string]int{
		"open":            report.Counts.Open,
		"p0":              report.Counts.P0,
		"p1":              report.Counts.P1,
		"orphan":          report.Counts.Orphan,
		"stale":           report.Counts.Stale,
		"needs_priority":  report.Counts.NeedsPriority,
		"needs_kind":      report.Counts.NeedsKind,
		"needs_area":      report.Counts.NeedsArea,
		"related_to_epic": report.Counts.Related,
	}
	attention := report.Counts.P0*50 + report.Counts.Orphan*15 + report.Counts.Stale*8 +
		report.Counts.NeedsPriority*20 + report.Counts.NeedsKind*10 + report.Counts.NeedsArea*10
	status := "ok"
	tags := []string{"issue-queue"}
	if attention > 0 {
		status = "action"
		tags = append(tags, "triage")
	}
	if epic > 0 {
		tags = append(tags, "epic")
	}
	summary := fmt.Sprintf("open=%d P0=%d orphan=%d stale=%d", report.Counts.Open, report.Counts.P0, report.Counts.Orphan, report.Counts.Stale)
	if epic > 0 {
		summary += fmt.Sprintf(" related=%d", report.Counts.Related)
	}
	return tuiOverviewCard{
		Pane:      "issues",
		Status:    status,
		Source:    report.Source,
		Summary:   summary,
		Command:   "fak console issues --issues-json " + report.Source,
		Attention: attention,
		Counts:    counts,
		Tags:      tags,
	}
}

func overviewLoopCard(report tuiLoopReport) tuiOverviewCard {
	counts := map[string]int{
		"loops":        report.Counts.Loops,
		"running":      report.Counts.Running,
		"refused":      report.Counts.Refused,
		"failed":       report.Counts.Failed,
		"witness_gaps": report.Counts.WitnessGaps,
		"witnessed":    report.Counts.Witnessed,
	}
	attention := report.Counts.Failed*90 + report.Counts.Refused*65 + report.Counts.WitnessGaps*45 + report.Counts.Running*8
	status := "ok"
	tags := []string{"loop-ledger"}
	if report.Counts.Failed > 0 || report.Counts.Refused > 0 || report.Counts.WitnessGaps > 0 {
		status = "action"
		tags = append(tags, "loop-attention")
	} else if report.Counts.Running > 0 {
		status = "warn"
		tags = append(tags, "running")
	}
	return tuiOverviewCard{
		Pane:      "loops",
		Status:    status,
		Source:    report.Ledger,
		Summary:   fmt.Sprintf("loops=%d running=%d refused=%d witness_gaps=%d", report.Counts.Loops, report.Counts.Running, report.Counts.Refused, report.Counts.WitnessGaps),
		Command:   "fak console loops --ledger " + report.Ledger,
		Attention: attention,
		Counts:    counts,
		Tags:      tags,
	}
}

func overviewSessionCard(report tuiSessionReport) tuiOverviewCard {
	counts := map[string]int{
		"sessions":       report.Counts.Sessions,
		"running":        report.Counts.Running,
		"throttled":      report.Counts.Throttled,
		"paused":         report.Counts.Paused,
		"stopped":        report.Counts.Stopped,
		"budgeted":       report.Counts.Budgeted,
		"context_budget": report.Counts.ContextBudget,
		"lineage":        report.Counts.Lineage,
	}
	lowBudget := 0
	for _, row := range report.Rows {
		if hasStringTUI(row.Tags, "low-turns") || hasStringTUI(row.Tags, "low-tokens") || hasStringTUI(row.Tags, "low-context") {
			lowBudget++
		}
	}
	counts["low_budget"] = lowBudget
	attention := report.Counts.Stopped*80 + report.Counts.Paused*45 + report.Counts.Throttled*25 + lowBudget*35
	status := "ok"
	tags := []string{"sessions"}
	if report.Counts.Stopped > 0 || report.Counts.Paused > 0 || lowBudget > 0 {
		status = "action"
		tags = append(tags, "operator-attention")
	} else if report.Counts.Throttled > 0 {
		status = "warn"
		tags = append(tags, "throttled")
	}
	return tuiOverviewCard{
		Pane:      "sessions",
		Status:    status,
		Source:    report.Source,
		Summary:   fmt.Sprintf("sessions=%d running=%d paused=%d stopped=%d low_budget=%d", report.Counts.Sessions, report.Counts.Running, report.Counts.Paused, report.Counts.Stopped, lowBudget),
		Command:   "fak console sessions --sessions-json " + report.Source,
		Attention: attention,
		Counts:    counts,
		Tags:      tags,
	}
}

func overviewGardenCard(report tuiGardenReport) tuiOverviewCard {
	counts := map[string]int{
		"members": report.Counts.Members,
		"ok":      report.Counts.OK,
		"action":  report.Counts.Action,
		"red":     report.Counts.Red,
		"errored": report.Counts.Errored,
		"gating":  report.Counts.Gating,
		"skipped": report.Counts.Skipped,
	}
	attention := report.Counts.Errored*100 + report.Counts.Red*90 + report.Counts.Action*45 + report.GateExit*100
	status := "ok"
	tags := []string{"garden"}
	if report.GateExit != 0 || report.Counts.Red > 0 || report.Counts.Errored > 0 {
		status = "action"
		tags = append(tags, "garden-red")
	} else if report.Counts.Action > 0 || report.Counts.Skipped > 0 {
		status = "warn"
		tags = append(tags, "advisory")
	}
	return tuiOverviewCard{
		Pane:      "garden",
		Status:    status,
		Source:    report.Source,
		Summary:   fmt.Sprintf("finding=%s members=%d red=%d errored=%d gate=%d", report.Finding, report.Counts.Members, report.Counts.Red, report.Counts.Errored, report.GateExit),
		Command:   "fak console garden --garden-json " + report.Source + " --check",
		Attention: attention,
		Counts:    counts,
		Tags:      tags,
	}
}

func overviewGuardCard(report tuiGuardReport) tuiOverviewCard {
	counts := map[string]int{
		"artifacts":    report.Counts.Artifacts,
		"rows":         report.Counts.Rows,
		"allow":        report.Counts.Allow,
		"deny":         report.Counts.Deny,
		"quarantine":   report.Counts.Quarantine,
		"policy_block": report.Counts.PolicyBlock,
		"default_deny": report.Counts.DefaultDeny,
		"expected":     report.Counts.Expected,
		"unexpected":   report.Counts.Unexpected,
	}
	attention := report.Counts.Unexpected*100 + report.Counts.Quarantine*80 + report.Counts.PolicyBlock*45 + report.Counts.DefaultDeny*25
	status := "ok"
	tags := []string{"guard"}
	if report.Counts.Unexpected > 0 || report.Status == "FAIL" {
		status = "action"
		tags = append(tags, "proof-gap")
	} else if report.Counts.Deny == 0 && report.Counts.Quarantine == 0 {
		status = "warn"
		tags = append(tags, "no-deny-proof")
	}
	return tuiOverviewCard{
		Pane:      "guard",
		Status:    status,
		Source:    report.Source,
		Summary:   fmt.Sprintf("artifacts=%d deny=%d policy_block=%d unexpected=%d", report.Counts.Artifacts, report.Counts.Deny, report.Counts.PolicyBlock, report.Counts.Unexpected),
		Command:   "fak console guard --guard-json <artifact>",
		Attention: attention,
		Counts:    counts,
		Tags:      tags,
	}
}

func missingOverviewCard(pane, command string) tuiOverviewCard {
	return tuiOverviewCard{
		Pane:    pane,
		Status:  "missing",
		Summary: "no source selected",
		Command: command,
		Tags:    []string{"missing-source"},
	}
}

func countTUIOverview(cards []tuiOverviewCard) tuiOverviewCounts {
	c := tuiOverviewCounts{Cards: len(cards)}
	for _, card := range cards {
		switch card.Status {
		case "ok":
			c.OK++
		case "action":
			c.Action++
		case "warn":
			c.Warn++
		case "missing":
			c.Missing++
		}
	}
	return c
}

func overviewActions(cards []tuiOverviewCard) []tuiOverviewAction {
	actions := []tuiOverviewAction{}
	for _, card := range cards {
		switch card.Status {
		case "action":
			actions = append(actions, tuiOverviewAction{Pane: card.Pane, Command: card.Command, Reason: card.Summary})
		case "missing":
			actions = append(actions, tuiOverviewAction{Pane: card.Pane, Command: card.Command, Reason: "add this pane's source to the overview"})
		}
	}
	return actions
}

func renderTUIAgent(report tuiAgentReport, width int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak console agent  at=%s  backend=%s  provider=%s  mode=dry-run\n", report.At, report.Backend, report.Provider)
	fmt.Fprintf(&b, "auth=%s  session=%s", report.Auth, report.SessionID)
	if report.Account != "" {
		fmt.Fprintf(&b, "  account=%s", report.Account)
	}
	if report.ResolvedAccount != "" && report.ResolvedAccount != report.Account {
		fmt.Fprintf(&b, "->%s", report.ResolvedAccount)
	}
	fmt.Fprintln(&b)
	if report.ClaudeConfigDir != "" {
		fmt.Fprintf(&b, "claude_config=%s  source=%s\n", trimTUI(report.ClaudeConfigDir, maxTUI(20, width-31)), report.ConfigSource)
	}
	if report.AccountIdentity != "" {
		fmt.Fprintf(&b, "identity=%s\n", report.AccountIdentity)
	}
	if report.Policy != "" || report.Model != "" || report.ContextBudget > 0 || report.RestartOnBudget {
		fmt.Fprintf(&b, "guard_options policy=%s model=%s context=%d restart=%v limit=%d\n",
			blankTUI(report.Policy), blankTUI(report.Model), report.ContextBudget, report.RestartOnBudget, report.RestartLimit)
	}
	if len(report.Env) > 0 {
		fmt.Fprintln(&b, "\nEnv")
		for _, kv := range report.Env {
			fmt.Fprintf(&b, "%-18s %-12s %s\n", kv.Name, kv.Source, trimTUI(kv.Value, maxTUI(20, width-33)))
		}
	}
	fmt.Fprintln(&b, "\nBackend Command")
	fmt.Fprintf(&b, "  %s\n", trimTUI(shellLineTUI(report.Command), maxTUI(20, width-2)))
	fmt.Fprintln(&b, "\nLaunch")
	fmt.Fprintf(&b, "  %s\n", trimTUI(shellLineTUI(report.Launch), maxTUI(20, width-2)))
	if len(report.Notes) > 0 {
		fmt.Fprintln(&b, "\nNotes")
		for _, note := range report.Notes {
			fmt.Fprintf(&b, "- %s\n", trimTUI(note, maxTUI(20, width-2)))
		}
	}
	return b.String()
}

func renderTUIOverview(report tuiOverviewReport, width int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak console overview  at=%s  source=%s\n", report.At, report.Source)
	fmt.Fprintf(&b, "cards=%d  ok=%d  action=%d  warn=%d  missing=%d\n",
		report.Counts.Cards, report.Counts.OK, report.Counts.Action, report.Counts.Warn, report.Counts.Missing)
	fmt.Fprintln(&b, "\nPanes")
	fmt.Fprintln(&b, "attention pane       status   tags                 summary")
	for _, card := range report.Cards {
		fmt.Fprintf(&b, "%9d %-10s %-8s %-20s %s\n",
			card.Attention, card.Pane, card.Status, trimTUI(displayTUITags(card.Tags, 3), 20),
			trimTUI(card.Summary, maxTUI(20, width-53)))
	}
	if len(report.Actions) > 0 {
		fmt.Fprintln(&b, "\nNext")
		limit := minTUI(len(report.Actions), 8)
		for _, action := range report.Actions[:limit] {
			fmt.Fprintf(&b, "%-10s %s\n", action.Pane, trimTUI(action.Command, maxTUI(20, width-11)))
		}
	}
	return b.String()
}

func buildTUISessionReport(list gateway.SessionListResponse, source string, at time.Time) tuiSessionReport {
	rows := make([]tuiSessionRow, 0, len(list.Sessions))
	for _, st := range list.Sessions {
		rows = append(rows, classifyTUISession(st))
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].Attention != rows[j].Attention {
			return rows[i].Attention > rows[j].Attention
		}
		if rows[i].Priority != rows[j].Priority {
			return rows[i].Priority < rows[j].Priority
		}
		return rows[i].TraceID < rows[j].TraceID
	})
	return tuiSessionReport{
		Schema: tuiSessionsSchema,
		At:     at.UTC().Format(time.RFC3339),
		Source: source,
		Counts: countTUISessions(rows),
		Lanes:  buildTUISessionLanes(rows),
		Rows:   rows,
	}
}

func classifyTUISession(st gateway.SessionState) tuiSessionRow {
	run := strings.TrimSpace(st.Run)
	if run == "" {
		run = "running"
	}
	row := tuiSessionRow{
		TraceID:           st.TraceID,
		Run:               run,
		Priority:          st.Priority,
		Rev:               st.Rev,
		Reason:            st.Reason,
		TurnsLeft:         st.Budget.TurnsLeft,
		TokensLeft:        st.Budget.TokensLeft,
		ContextTokensLeft: st.Budget.ContextTokensLeft,
		MaxTokensPerTurn:  st.Pace.MaxTokensPerTurn,
		MinTurnGapMs:      st.Pace.MinTurnGapMs,
		ContinuationID:    st.ContinuationID,
		ParentTrace:       st.ParentTrace,
		Generation:        st.Generation,
	}
	row.Tags, row.Attention = scoreTUISession(row)
	return row
}

func scoreTUISession(row tuiSessionRow) ([]string, int) {
	tags := []string{}
	score := 0
	switch strings.ToLower(row.Run) {
	case "running":
		tags = append(tags, "running")
	case "throttled":
		tags = append(tags, "throttled")
		score += 35
	case "paused":
		tags = append(tags, "paused")
		score += 55
	case "draining":
		tags = append(tags, "draining")
		score += 65
	case "stopped":
		tags = append(tags, "stopped")
		score += 80
	default:
		tags = append(tags, "unknown-run")
		score += 30
	}
	if row.Reason != "" {
		tags = append(tags, "reason")
		score += 10
	}
	if row.TurnsLeft >= 0 {
		if row.TurnsLeft <= 1 {
			tags = append(tags, "low-turns")
			score += 45
		}
		tags = append(tags, "turn-budget")
	}
	if row.TokensLeft >= 0 {
		if row.TokensLeft <= 1000 {
			tags = append(tags, "low-tokens")
			score += 35
		}
		tags = append(tags, "token-budget")
	}
	if row.ContextTokensLeft > 0 {
		if row.ContextTokensLeft <= 2000 {
			tags = append(tags, "low-context")
			score += 25
		}
		tags = append(tags, "context-budget")
	}
	if row.MaxTokensPerTurn > 0 || row.MinTurnGapMs > 0 {
		tags = append(tags, "paced")
	}
	if row.ParentTrace != "" || row.ContinuationID != "" || row.Generation > 0 {
		tags = append(tags, "lineage")
	}
	return tags, score
}

func countTUISessions(rows []tuiSessionRow) tuiSessionCounts {
	var c tuiSessionCounts
	for _, row := range rows {
		c.Sessions++
		switch strings.ToLower(row.Run) {
		case "running":
			c.Running++
		case "throttled":
			c.Throttled++
		case "paused":
			c.Paused++
		case "draining":
			c.Draining++
		case "stopped":
			c.Stopped++
		}
		if row.TurnsLeft >= 0 || row.TokensLeft >= 0 || row.ContextTokensLeft > 0 {
			c.Budgeted++
		}
		if row.ContextTokensLeft > 0 {
			c.ContextBudget++
		}
		if row.ParentTrace != "" || row.ContinuationID != "" || row.Generation > 0 {
			c.Lineage++
		}
		if row.Reason != "" {
			c.WithReason++
		}
	}
	return c
}

func buildTUISessionLanes(rows []tuiSessionRow) []tuiSessionLane {
	names := []string{"running", "throttled", "paused", "draining", "stopped", "other"}
	lanes := make([]tuiSessionLane, 0, len(names))
	for _, name := range names {
		lane := tuiSessionLane{Name: name}
		for _, row := range rows {
			if !rowInTUISessionLane(row, name) {
				continue
			}
			lane.Count++
			if lane.TopSession == "" {
				lane.TopSession = row.TraceID
				lane.TopSummary = sessionSummary(row)
			}
		}
		lanes = append(lanes, lane)
	}
	return lanes
}

func rowInTUISessionLane(row tuiSessionRow, lane string) bool {
	run := strings.ToLower(row.Run)
	if lane == "other" {
		switch run {
		case "running", "throttled", "paused", "draining", "stopped":
			return false
		default:
			return true
		}
	}
	return run == lane
}

func renderTUISessions(report tuiSessionReport, top, width int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak console sessions  at=%s  source=%s\n", report.At, report.Source)
	fmt.Fprintf(&b, "sessions=%d  running=%d  throttled=%d  paused=%d  draining=%d  stopped=%d  budgeted=%d  context=%d  lineage=%d\n",
		report.Counts.Sessions, report.Counts.Running, report.Counts.Throttled, report.Counts.Paused,
		report.Counts.Draining, report.Counts.Stopped, report.Counts.Budgeted, report.Counts.ContextBudget, report.Counts.Lineage)
	if len(report.Rows) == 0 {
		fmt.Fprintln(&b, "\nno sessions found")
		return b.String()
	}
	fmt.Fprintln(&b, "\nLanes")
	fmt.Fprintln(&b, "lane             count top")
	for _, lane := range report.Lanes {
		topText := "-"
		if lane.TopSession != "" {
			topText = lane.TopSession
			if lane.TopSummary != "" {
				topText += " " + lane.TopSummary
			}
		}
		fmt.Fprintf(&b, "%-16s %5d %s\n", lane.Name, lane.Count, trimTUI(topText, maxTUI(20, width-24)))
	}
	fmt.Fprintln(&b, "\nSession Queue")
	renderTUISessionRows(&b, report.Rows, minTUI(top, len(report.Rows)), width)
	return b.String()
}

func renderTUISessionRows(b *strings.Builder, rows []tuiSessionRow, limit, width int) {
	fmt.Fprintln(b, "attention session                    run        prio rev budget                         pace          tags")
	for _, row := range rows[:limit] {
		budget := fmt.Sprintf("t=%s tok=%s ctx=%s",
			budgetAxis(row.TurnsLeft), budgetAxis(row.TokensLeft), contextBudgetAxis(row.ContextTokensLeft))
		pace := fmt.Sprintf("max=%d gap=%d", row.MaxTokensPerTurn, row.MinTurnGapMs)
		summary := displayTUITags(row.Tags, 3)
		fmt.Fprintf(b, "%9d %-26s %-10s %4d %-3d %-30s %-13s %s\n",
			row.Attention, trimTUI(row.TraceID, 26), trimTUI(row.Run, 10), row.Priority, row.Rev,
			trimTUI(budget, 30), trimTUI(pace, 13), trimTUI(summary, maxTUI(14, width-95)))
	}
}

func displayTUITags(tags []string, limit int) string {
	if len(tags) == 0 {
		return "-"
	}
	if limit <= 0 || len(tags) <= limit {
		return strings.Join(tags, ",")
	}
	return strings.Join(tags[:limit], ",")
}

func sessionSummary(row tuiSessionRow) string {
	parts := []string{}
	if row.Reason != "" {
		parts = append(parts, row.Reason)
	}
	if row.TurnsLeft >= 0 {
		parts = append(parts, "turns="+budgetAxis(row.TurnsLeft))
	}
	if row.TokensLeft >= 0 {
		parts = append(parts, "tokens="+budgetAxis(row.TokensLeft))
	}
	if row.ContextTokensLeft > 0 {
		parts = append(parts, "context="+contextBudgetAxis(row.ContextTokensLeft))
	}
	if len(parts) == 0 {
		parts = append(parts, strings.Join(row.Tags, ","))
	}
	return strings.Join(parts, " ")
}

func buildTUILoopReport(st loopmgr.Status, at time.Time) tuiLoopReport {
	rows := make([]tuiLoopRow, 0, len(st.Loops))
	for _, loop := range st.Loops {
		rows = append(rows, classifyTUILoop(loop, at))
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].Attention != rows[j].Attention {
			return rows[i].Attention > rows[j].Attention
		}
		return rows[i].LoopID < rows[j].LoopID
	})
	return tuiLoopReport{
		Schema: tuiLoopsSchema,
		At:     at.UTC().Format(time.RFC3339),
		Ledger: st.LedgerPath,
		Counts: countTUILoops(rows),
		Lanes:  buildTUILoopLanes(rows),
		Rows:   rows,
	}
}

func classifyTUILoop(loop loopmgr.LoopSnapshot, at time.Time) tuiLoopRow {
	state := loop.State
	if strings.TrimSpace(state) == "" {
		state = "-"
	}
	age := int64(0)
	if loop.LastEventUnixNano > 0 {
		d := at.UTC().Sub(time.Unix(0, loop.LastEventUnixNano).UTC())
		if d > 0 {
			age = int64(d.Seconds())
		}
	}
	row := tuiLoopRow{
		LoopID:              loop.LoopID,
		State:               state,
		LastKind:            string(loop.LastKind),
		LastSeq:             loop.LastSeq,
		AgeSeconds:          age,
		CurrentRunID:        loop.CurrentRunID,
		Fires:               loop.Fires,
		Admitted:            loop.Admitted,
		Refused:             loop.Refused,
		ConsecutiveRefusals: loop.ConsecutiveRefusals,
		Started:             loop.Started,
		Ended:               loop.Ended,
		Witnessed:           loop.Witnessed,
		WitnessRefused:      loop.WitnessRefused,
		WitnessUnavailable:  loop.WitnessUnavailable,
		Notifications:       loop.Notifications,
	}
	if loop.LastRun != nil {
		row.LastRunStatus = string(loop.LastRun.Status)
		row.LastRunReason = loop.LastRun.Reason
		row.LastRunSummary = loop.LastRun.Summary
	}
	if loop.Ended > 0 {
		rate := float64(loop.Witnessed) / float64(loop.Ended)
		row.WitnessRate = &rate
	}
	row.Tags, row.Attention = scoreTUILoop(row)
	return row
}

func scoreTUILoop(row tuiLoopRow) ([]string, int) {
	tags := []string{}
	score := 0
	state := strings.ToLower(row.State)
	status := strings.ToLower(row.LastRunStatus)
	if state == string(loopmgr.StateRunning) || status == string(loopmgr.StatusRunning) {
		tags = append(tags, "running")
		score += 70
	}
	if state == string(loopmgr.StatusRefused) || status == string(loopmgr.StatusRefused) || row.ConsecutiveRefusals > 0 {
		tags = append(tags, "refused")
		score += 80 + int(row.ConsecutiveRefusals)*20
	}
	if status == string(loopmgr.StatusFailed) || state == string(loopmgr.StatusFailed) {
		tags = append(tags, "failed")
		score += 100
	}
	if row.Ended > row.Witnessed {
		tags = append(tags, "needs-witness")
		score += int(row.Ended-row.Witnessed) * 15
	}
	if row.WitnessRefused > 0 {
		tags = append(tags, "witness-refused")
		score += int(row.WitnessRefused) * 20
	}
	if row.WitnessUnavailable > 0 {
		tags = append(tags, "witness-unavailable")
		score += int(row.WitnessUnavailable) * 10
	}
	if row.AgeSeconds > int64(6*time.Hour/time.Second) && (state == "running" || status == "running") {
		tags = append(tags, "old-running")
		score += 40
	}
	if score == 0 && (status == string(loopmgr.StatusWitnessedDone) || row.Witnessed > 0) {
		tags = append(tags, "witnessed")
	}
	return tags, score
}

func countTUILoops(rows []tuiLoopRow) tuiLoopCounts {
	var c tuiLoopCounts
	for _, row := range rows {
		c.Loops++
		if tuiLoopHasTag(row, "running") {
			c.Running++
		}
		if tuiLoopHasTag(row, "refused") {
			c.Refused++
		}
		if tuiLoopHasTag(row, "failed") {
			c.Failed++
		}
		if row.Witnessed > 0 {
			c.Witnessed++
		}
		if tuiLoopHasTag(row, "needs-witness") || tuiLoopHasTag(row, "witness-refused") || tuiLoopHasTag(row, "witness-unavailable") {
			c.WitnessGaps++
		}
		if row.Notifications > 0 {
			c.Notifications++
		}
	}
	return c
}

func buildTUILoopLanes(rows []tuiLoopRow) []tuiLoopLane {
	names := []string{"running", "refused", "needs-witness", "witnessed", "other"}
	lanes := make([]tuiLoopLane, 0, len(names))
	for _, name := range names {
		lane := tuiLoopLane{Name: name}
		for _, row := range rows {
			if !rowInTUILoopLane(row, name) {
				continue
			}
			lane.Count++
			if lane.TopLoop == "" {
				lane.TopLoop = row.LoopID
				lane.TopLoopText = row.LastRunSummary
				if lane.TopLoopText == "" {
					lane.TopLoopText = strings.Join(row.Tags, ",")
				}
			}
		}
		lanes = append(lanes, lane)
	}
	return lanes
}

func rowInTUILoopLane(row tuiLoopRow, lane string) bool {
	switch lane {
	case "running":
		return tuiLoopHasTag(row, "running")
	case "refused":
		return tuiLoopHasTag(row, "refused") || tuiLoopHasTag(row, "failed")
	case "needs-witness":
		return tuiLoopHasTag(row, "needs-witness") || tuiLoopHasTag(row, "witness-refused") || tuiLoopHasTag(row, "witness-unavailable")
	case "witnessed":
		return tuiLoopHasTag(row, "witnessed")
	case "other":
		return len(row.Tags) == 0
	default:
		return false
	}
}

func tuiLoopHasTag(row tuiLoopRow, tag string) bool {
	for _, got := range row.Tags {
		if got == tag {
			return true
		}
	}
	return false
}

func renderTUILoops(report tuiLoopReport, top, width int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak console loops  at=%s  ledger=%s\n", report.At, report.Ledger)
	fmt.Fprintf(&b, "loops=%d  running=%d  refused=%d  failed=%d  witnessed=%d  witness-gaps=%d  notified=%d\n",
		report.Counts.Loops, report.Counts.Running, report.Counts.Refused, report.Counts.Failed,
		report.Counts.Witnessed, report.Counts.WitnessGaps, report.Counts.Notifications)
	if len(report.Rows) == 0 {
		fmt.Fprintln(&b, "\nno loops found")
		return b.String()
	}
	fmt.Fprintln(&b, "\nLanes")
	fmt.Fprintln(&b, "lane             count top")
	for _, lane := range report.Lanes {
		topText := "-"
		if lane.TopLoop != "" {
			topText = lane.TopLoop
			if lane.TopLoopText != "" {
				topText += " " + lane.TopLoopText
			}
		}
		fmt.Fprintf(&b, "%-16s %5d %s\n", lane.Name, lane.Count, trimTUI(topText, maxTUI(20, width-24)))
	}
	fmt.Fprintln(&b, "\nLoop Queue")
	renderTUILoopRows(&b, report.Rows, minTUI(top, len(report.Rows)), width)
	return b.String()
}

func renderTUILoopRows(b *strings.Builder, rows []tuiLoopRow, limit, width int) {
	fmt.Fprintln(b, "attention loop                         state          age    runs             witness tags")
	for _, row := range rows[:limit] {
		runs := fmt.Sprintf("f%d/a%d/r%d/e%d", row.Fires, row.Admitted, row.Refused, row.Ended)
		witness := "-"
		if row.WitnessRate != nil {
			witness = trimFloat(*row.WitnessRate * 100)
			witness += "%"
		}
		tags := strings.Join(row.Tags, ",")
		if tags == "" {
			tags = "-"
		}
		summary := row.LastRunSummary
		if summary == "" {
			summary = row.LastRunReason
		}
		lineTail := tags
		if summary != "" {
			lineTail += "  " + summary
		}
		fmt.Fprintf(b, "%9d %-28s %-14s %-6s %-16s %-7s %s\n",
			row.Attention, trimTUI(row.LoopID, 28), trimTUI(row.State, 14),
			durationTUIText(row.AgeSeconds), runs, witness, trimTUI(lineTail, maxTUI(16, width-88)))
	}
}

func durationTUIText(seconds int64) string {
	if seconds <= 0 {
		return "0s"
	}
	if seconds < 60 {
		return strconv.FormatInt(seconds, 10) + "s"
	}
	minutes := seconds / 60
	if minutes < 60 {
		return strconv.FormatInt(minutes, 10) + "m"
	}
	hours := minutes / 60
	if hours < 48 {
		return strconv.FormatInt(hours, 10) + "h"
	}
	return strconv.FormatInt(hours/24, 10) + "d"
}

func renderTUIIssues(report tuiIssueReport, top, width int) string {
	var b strings.Builder
	title := "fak console issues"
	fmt.Fprintf(&b, "%s  as_of=%s  source=%s\n", title, report.AsOf, report.Source)
	fmt.Fprintf(&b, "open=%d  P0=%d  P1=%d  P2=%d  orphan=%d  stale=%d  needs: prio=%d kind=%d area=%d\n",
		report.Counts.Open, report.Counts.P0, report.Counts.P1, report.Counts.P2,
		report.Counts.Orphan, report.Counts.Stale, report.Counts.NeedsPriority,
		report.Counts.NeedsKind, report.Counts.NeedsArea)
	if report.Epic != nil {
		fmt.Fprintf(&b, "\nEpic #%d  score=%d  idle=%dd\n", report.Epic.Number, report.Epic.Score, report.Epic.IdleDays)
		fmt.Fprintf(&b, "  %s\n", trimTUI(report.Epic.Title, width-2))
		fmt.Fprintf(&b, "  related loaded issues: %d\n", report.Counts.Related)
	}
	fmt.Fprintln(&b, "\nLanes")
	fmt.Fprintln(&b, "lane             count orphan needs-kind needs-area max-idle top")
	for _, lane := range report.Lanes {
		topText := "-"
		if lane.TopIssue != 0 {
			topText = fmt.Sprintf("#%d %s", lane.TopIssue, lane.TopIssueText)
		}
		fmt.Fprintf(&b, "%-16s %5d %6d %10d %10d %8dd %s\n",
			lane.Name, lane.Count, lane.Orphan, lane.NeedsKind, lane.NeedsArea,
			lane.MaxIdleDays, trimTUI(topText, maxTUI(20, width-62)))
	}
	rows := report.Rows
	if report.Epic != nil {
		related := []tuiIssueRow{}
		for _, row := range report.Rows {
			if row.Related && row.Number != report.Epic.Number {
				related = append(related, row)
			}
		}
		if len(related) > 0 {
			fmt.Fprintln(&b, "\nRelated")
			renderTUIIssueRows(&b, related, minTUI(top, len(related)), width)
		}
	}
	fmt.Fprintln(&b, "\nRanked Queue")
	renderTUIIssueRows(&b, rows, minTUI(top, len(rows)), width)
	if len(report.Actions) > 0 {
		fmt.Fprintln(&b, "\nReview Actions")
		limit := minTUI(8, len(report.Actions))
		for _, action := range report.Actions[:limit] {
			fmt.Fprintf(&b, "#%-5d %-23s %s\n", action.Number, action.Kind, trimTUI(action.Reason, width-32))
		}
		if len(report.Actions) > limit {
			fmt.Fprintf(&b, "... %d more actions in --json\n", len(report.Actions)-limit)
		}
	}
	return b.String()
}

func renderTUIIssueRows(b *strings.Builder, rows []tuiIssueRow, limit, width int) {
	fmt.Fprintln(b, "#      score prio idle tags                         title")
	for _, row := range rows[:limit] {
		prio := row.Priority
		if prio == "" {
			prio = "-"
		} else {
			prio = strings.TrimPrefix(prio, "priority/")
		}
		tags := strings.Join(row.Tags, ",")
		if tags == "" {
			tags = "-"
		}
		titleWidth := maxTUI(20, width-49)
		fmt.Fprintf(b, "#%-5d %5d %-4s %4dd %-28s %s\n",
			row.Number, row.Score, prio, row.IdleDays, trimTUI(tags, 28), trimTUI(row.Title, titleWidth))
	}
}

func trimTUI(s string, width int) string {
	s = strings.Join(strings.Fields(s), " ")
	if width <= 0 || len(s) <= width {
		return s
	}
	if width <= 3 {
		return s[:width]
	}
	return s[:width-3] + "..."
}

func minTUI(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxTUI(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func nonEmptyTUI(values []string) []string {
	out := []string{}
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			out = append(out, strings.TrimSpace(value))
		}
	}
	return out
}

func firstNonEmptyTUI(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func hasStringTUI(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func blankTUI(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return strings.TrimSpace(s)
}

func shellLineTUI(args []string) string {
	parts := make([]string, 0, len(args))
	for _, arg := range args {
		parts = append(parts, shellArgTUI(arg))
	}
	return strings.Join(parts, " ")
}

func shellArgTUI(arg string) string {
	if arg == "" {
		return `""`
	}
	if strings.IndexFunc(arg, func(r rune) bool {
		return r == '"' || r == '\'' || r == '\\' || r == '$' || r == '`' || r <= ' '
	}) >= 0 {
		return strconv.Quote(arg)
	}
	return arg
}

func tuiUsage(w io.Writer) {
	fmt.Fprint(w, `fak console - native terminal control panes
Alias: fak tui

  fak console issues [--issues-json FILE] [--json] [--epic N]
                 [--repo owner/repo] [--state open|closed|all]
                 [--limit N] [--top N] [--width N] [--as-of YYYY-MM-DD]
  fak console loops  [--ledger FILE] [--json] [--top N] [--width N]
                 [--at RFC3339|YYYY-MM-DD]
  fak console sessions [--sessions-json FILE] [--json] [--addr URL] [--key K]
                   [--top N] [--width N] [--at RFC3339|YYYY-MM-DD]
  fak console garden [--garden-json FILE] [--json] [--check]
                 [--workspace DIR] [--deep] [--timeout N] [--width N]
  fak console guard  --guard-json FILE [--guard-json FILE ...] [--json]
                 [--width N] [--at RFC3339|YYYY-MM-DD]
  fak console agent [--account NAME | --claude-config-dir DIR] [--dry-run]
                [--prompt STR] [--session-id ID] [--passthrough] [--json]
                [--] [claude args...]
  fak console overview [--issues-json FILE] [--ledger FILE] [--sessions-json FILE]
                   [--garden-json FILE] [--guard-json FILE ...] [--json]

The issues pane folds GitHub issues into a ranked terminal model: priority lanes,
orphan/stale/label gaps, optional epic-related rows, and review actions. With no
--issues-json it shells out to gh issue list; fixtures keep the model testable.

The loops pane folds fak's hash-chained loop ledger into the same terminal model:
running/refused/witness-gap lanes, attention ranking, and machine-readable JSON.

The sessions pane reads GET /v1/fak/sessions or a fixture JSON and renders live
DRIVE state: run-state lanes, budgets, pace, priority, lineage, and reasons.

The garden pane reads `+"`fak garden --json`"+` envelopes or runs the read-only garden
bundle and renders member health, gating regressions, and advisory actions.

The guard pane reads existing guard/adjudication JSON artifacts and renders
denials, reasons, audit status, and proof-packet gaps without replaying calls.

The agent pane launches a real Claude Code backend through `+"`fak guard`"+`. It can pin
CLAUDE_CONFIG_DIR from `+"`fak accounts`"+` and defaults to the Claude subscription OAuth
path, so the native TUI can start an actual account-backed agent without an API key.

The overview pane composes selected pane models into one ranked spine so
operators can see issue, loop, session, garden, and guard pressure together.
`)
}
