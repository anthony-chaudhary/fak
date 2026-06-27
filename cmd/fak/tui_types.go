// tui_types.go holds the `fak console` report/row data shapes, extracted from
// tui.go so the console surface stays under the steerability god-file line (#984).
// Pure code motion: the type declarations are unchanged.
package main

import (
	"encoding/json"
	"fmt"
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
	Schema    string              `json:"schema"`
	At        string              `json:"at"`
	Ledger    string              `json:"ledger"`
	Counts    tuiLoopCounts       `json:"counts"`
	Lanes     []tuiLoopLane       `json:"lanes"`
	Rows      []tuiLoopRow        `json:"rows"`
	Integrity *tuiLedgerIntegrity `json:"ledger_integrity,omitempty"`
}

// tuiLedgerIntegrity carries a loop-ledger chain break into the report model. It is
// populated only when the ledger is forked/corrupt (the loops console then renders the
// recovered prefix plus a banner instead of going dark); a clean ledger leaves it nil,
// so JSON output for healthy ledgers is byte-identical.
type tuiLedgerIntegrity struct {
	Broken    bool   `json:"broken"`
	AtLine    int    `json:"at_line,omitempty"`
	AtSeq     uint64 `json:"at_seq,omitempty"`
	Reason    string `json:"reason,omitempty"`
	Recovered int    `json:"recovered_events"`
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
	Rank      int      `json:"rank,omitempty"`
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
	Schema              string        `json:"schema"`
	At                  string        `json:"at"`
	Backend             string        `json:"backend"`
	Target              string        `json:"target,omitempty"` // #938: the named compute target this launch resolved (mac/gcp/local/anthropic/...), empty for an unnamed default launch
	Mode                string        `json:"mode"`
	Provider            string        `json:"provider"`
	Auth                string        `json:"auth"`
	GatewayURL          string        `json:"gateway_url,omitempty"`
	Account             string        `json:"account,omitempty"`
	ResolvedAccount     string        `json:"resolved_account,omitempty"`
	AccountIdentity     string        `json:"account_identity,omitempty"`
	ClaudeConfigDir     string        `json:"claude_config_dir,omitempty"`
	ConfigSource        string        `json:"config_source,omitempty"`
	SessionID           string        `json:"session_id,omitempty"`
	PermissionMode      string        `json:"permission_mode,omitempty"`
	Policy              string        `json:"policy,omitempty"`
	Model               string        `json:"model,omitempty"`
	ContextBudget       int           `json:"context_budget_tokens,omitempty"`
	RestartOnBudget     bool          `json:"restart_on_budget,omitempty"`
	RestartLimit        int           `json:"restart_limit,omitempty"`
	DebugStats          bool          `json:"debug_stats,omitempty"`
	CompactHistoryLimit int           `json:"compact_history_limit_tokens,omitempty"`
	ElideResultBytes    int           `json:"elide_result_bytes,omitempty"`
	Command             []string      `json:"command"`
	Launch              []string      `json:"launch"`
	Env                 []tuiAgentEnv `json:"env,omitempty"`
	Notes               []string      `json:"notes,omitempty"`
}

type tuiAgentEnv struct {
	Name      string `json:"name"`
	Value     string `json:"value,omitempty"`
	Source    string `json:"source"`
	FromEnv   string `json:"from_env,omitempty"`
	Sensitive bool   `json:"sensitive,omitempty"`
}

type tuiAgentOptions struct {
	Backend             string
	Command             string
	CommandArgs         []string
	Prompt              string
	PermissionMode      string
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
	GatewayURL          string
	GatewayKeyEnv       string
	APITimeoutMS        int
	DebugStats          bool
}
