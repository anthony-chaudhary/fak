// watchdog.go — the pure per-row decision core of the cross-account resume watchdog
// (`fak resume watchdog`), the Go port of tools/fleet_resume_watchdog.py's tick loop.
// The watchdog is the LAUNCH half of the resume process: sweep/stopped/scan discover and
// bucket the dead sessions, fleet_sessions writes the AUTO_RESUME plan, and each watchdog
// tick walks that plan deciding, per session, whether to fire `claude --resume` under the
// owning account's CLAUDE_CONFIG_DIR.
//
// # What this leaf owns
//
// The ordered guard chain the Python main loop applied inline, as one total function:
//
//  1. self-guard    — never resume the session the watchdog itself runs inside (a live
//     operator session can briefly look like a stopped autonomous worker, and a
//     self-resume races two `claude` processes on one transcript);
//  2. policy guard  — never resume onto an account the fleet policy no longer offers as
//     a worker (a stale plan file can predate a tombstone);
//  3. retry gate    — the outcome-aware once-gate (RetryGate): blocked unless the last
//     attempt failed recoverably and the attempt budget is not spent.
//
// The verdict carries the closed action token plus the attempt number a launch would
// record, so the shell's ledger row and the gate's arithmetic can never disagree.
//
// # What stays in the shell
//
// Everything with a clock or a side effect: the per-tick launch cap, the host
// source-admission gate (AdmitSource — consulted only when a launch would actually fire),
// launch spacing, the re-home copy, the spawn, and the ledger append. This leaf reads no
// clock and does no I/O — same row + history + outcome in, same verdict out.
package resume

import (
	"fmt"
	"strings"
)

// WatchdogPlanRow is one AUTO_RESUME entry of the on-disk resume plan
// (tools/_registry/resume_plan.json, written by fleet_sessions.py) — only the fields the
// tick reads, json-tagged with the plan's own key names so the shell can decode rows
// directly. Unknown fields are dropped, not trusted.
type WatchdogPlanRow struct {
	// Session is the session id `claude --resume` takes.
	Session string `json:"session"`
	// Account is the owning account's dir basename (e.g. ".claude-gem7").
	Account string `json:"account"`
	// ResumeAccount is the account the plan re-homed the session onto ("" = the owner).
	ResumeAccount string `json:"resume_account,omitempty"`
	// Project is the project slug the transcript lives under.
	Project string `json:"project,omitempty"`
	// CWD is the working directory the resumed session should run from.
	CWD string `json:"cwd,omitempty"`
	// ConfigDir is the owner's CLAUDE_CONFIG_DIR.
	ConfigDir string `json:"config_dir,omitempty"`
	// ResumeConfigDir is the CLAUDE_CONFIG_DIR to resume under (falls back to ConfigDir).
	ResumeConfigDir string `json:"resume_config_dir,omitempty"`
	// SourceConfigDir is the re-home copy source ("" = ConfigDir).
	SourceConfigDir string `json:"source_config_dir,omitempty"`
	// Disp is the classifier disposition that put the session on the plan (the ledger
	// row's cause, e.g. STOPPED_MIDTOOL).
	Disp string `json:"disp,omitempty"`
	// Rehomed marks a plan row whose transcript must be copied onto ResumeConfigDir
	// before the resume can find it.
	Rehomed bool `json:"rehomed,omitempty"`
}

// ResumeTarget is the config dir a launch must pin CLAUDE_CONFIG_DIR to: the re-home
// target when the plan named one, else the owner's dir.
func (r WatchdogPlanRow) ResumeTarget() string {
	if strings.TrimSpace(r.ResumeConfigDir) != "" {
		return r.ResumeConfigDir
	}
	return r.ConfigDir
}

// RehomeSource is the config dir a re-home copy reads from: the plan's explicit source
// when named, else the owner's dir.
func (r WatchdogPlanRow) RehomeSource() string {
	if strings.TrimSpace(r.SourceConfigDir) != "" {
		return r.SourceConfigDir
	}
	return r.ConfigDir
}

// WatchdogAction is the closed per-row verdict vocabulary of one tick step.
type WatchdogAction string

const (
	// WatchdogLaunch: fire `claude --resume` for this row now (the shell still fronts the
	// spawn with the host source-admission gate and the per-tick cap).
	WatchdogLaunch WatchdogAction = "launch"
	// WatchdogSkipSelf: the row IS the session this watchdog runs inside — never
	// self-resume (two `claude` processes would race one transcript).
	WatchdogSkipSelf WatchdogAction = "skip_self"
	// WatchdogSkipNonWorker: the row's account is not an offered worker under the current
	// fleet policy (tombstoned/excluded) — a stale plan must not resurrect it.
	WatchdogSkipNonWorker WatchdogAction = "skip_non_worker"
	// WatchdogSkipBlocked: the outcome-aware retry gate blocks a new resume (already
	// took, auth wall, attempt cap, or operator-settled). The reason carries the why.
	WatchdogSkipBlocked WatchdogAction = "skip_blocked"
)

// WatchdogGuards is the tick-constant context every row is judged against.
type WatchdogGuards struct {
	// SelfSID is the id of the session the watchdog runs inside (CLAUDE_CODE_SESSION_ID;
	// "" outside a Claude session, which leaves the self-guard inert).
	SelfSID string `json:"self_sid,omitempty"`
	// WorkerAccounts is the set of account dir-basenames the fleet policy still offers as
	// workers. Empty/nil leaves the policy guard INERT (fail-open, matching the Python:
	// a failed roster read must not strand the whole watchdog).
	WorkerAccounts map[string]bool `json:"worker_accounts,omitempty"`
	// MaxAttempts is the retry gate's give-up cap; <= 0 takes DefaultMaxResumeAttempts.
	MaxAttempts int `json:"max_attempts,omitempty"`
}

// WatchdogRowDecision is the leaf's verdict for one plan row.
type WatchdogRowDecision struct {
	Action WatchdogAction `json:"action"`
	// Reason is the closed human one-liner (the note/ledger line the shell logs).
	Reason string `json:"reason"`
	// Attempt is the 1-based attempt number a launch fired now would record — the fired
	// launches already on the ledger plus one. Zero for non-launch verdicts.
	Attempt int `json:"attempt,omitempty"`
}

// DecideWatchdogRow applies the ordered guard chain to one plan row. history is the
// session's prior ledger rows (oldest first); outcome is the terminal-turn classification
// of the last attempt (ClassifyOutcome over the newest transcript — shell-extracted).
// Total over any input: an empty row fails no guard and folds to a launch with attempt 1,
// but a real caller only feeds rows the plan actually carries.
func DecideWatchdogRow(row WatchdogPlanRow, g WatchdogGuards, history []Attempt, outcome Outcome) WatchdogRowDecision {
	if g.SelfSID != "" && row.Session == g.SelfSID {
		return WatchdogRowDecision{Action: WatchdogSkipSelf,
			Reason: "this is the live session running the watchdog (self-resume guard)"}
	}
	if len(g.WorkerAccounts) > 0 && !g.WorkerAccounts[row.Account] {
		return WatchdogRowDecision{Action: WatchdogSkipNonWorker,
			Reason: fmt.Sprintf("account %s is not an offered worker (policy/tombstoned)", row.Account)}
	}
	if d := RetryGate(history, outcome, g.MaxAttempts); d.Blocked {
		return WatchdogRowDecision{Action: WatchdogSkipBlocked, Reason: d.Reason}
	}
	return WatchdogRowDecision{Action: WatchdogLaunch,
		Reason: "retry gate allows a resume", Attempt: CountAttempts(history) + 1}
}

// WatchdogChildEnvDrop is the closed set of environment keys a resumed child must NOT
// inherit from the watchdog's own process:
//
//   - The model-API wiring of a guarded/Claude parent session (ANTHROPIC_API_KEY +
//     ANTHROPIC_BASE_URL point at the parent's loopback fak-guard gateway, and env auth
//     takes precedence over the seat's OAuth login). A child inheriting them routes every
//     request through the parent's proxy — wrong seat, account routing nullified — and
//     dies with the parent: the whole-wave-crashes-at-one-instant signature (2026-07-01).
//   - The parent's harness session identity (CLAUDE_CODE_SESSION_ID /
//     CLAUDE_CODE_CHILD_SESSION), which would make the child look like the parent to
//     every self-guard downstream.
//   - JOB_SUPERVISED_WORKER, so the resumed session is not mistaken for a supervised
//     job worker by the supervisor loop.
//
// The child authenticates with its own CLAUDE_CONFIG_DIR seat instead. Exported so the
// shell and its tests share one list — the strip IS the fix for the mass-crash wave, and
// a silent drift here re-opens it.
var WatchdogChildEnvDrop = []string{
	"ANTHROPIC_API_KEY",
	"ANTHROPIC_AUTH_TOKEN",
	"ANTHROPIC_BASE_URL",
	"CLAUDE_CODE_SESSION_ID",
	"CLAUDE_CODE_CHILD_SESSION",
	"JOB_SUPERVISED_WORKER",
}

// WatchdogChildEnv builds the environment for a resumed child from the watchdog's own
// environ: the guard/harness keys are stripped (WatchdogChildEnvDrop) and
// CLAUDE_CONFIG_DIR is pinned to the resume target. Pure over its inputs (environ is the
// caller's os.Environ() snapshot).
func WatchdogChildEnv(environ []string, configDir string) []string {
	drop := make(map[string]bool, len(WatchdogChildEnvDrop)+1)
	for _, k := range WatchdogChildEnvDrop {
		drop[k] = true
	}
	drop["CLAUDE_CONFIG_DIR"] = true // re-pinned below, never duplicated
	out := make([]string, 0, len(environ)+1)
	for _, kv := range environ {
		key := kv
		if i := strings.IndexByte(kv, '='); i >= 0 {
			key = kv[:i]
		}
		if drop[key] {
			continue
		}
		out = append(out, kv)
	}
	return append(out, "CLAUDE_CONFIG_DIR="+configDir)
}

// ResolveWatchdogProbeMode resolves the FAK_PROBE setting for a tick: "auto" probes
// blocked accounts only on a LIVE tick (so the default dry-run stays side-effect-free —
// no probe spend), and an explicit setting is honored as-is. Mirrors
// fleet_resume_watchdog.resolve_probe_mode, which the .ps1's -Probe auto behavior set.
func ResolveWatchdogProbeMode(setting string, live bool) string {
	if strings.TrimSpace(strings.ToLower(setting)) == "auto" || strings.TrimSpace(setting) == "" {
		if live {
			return "blocked"
		}
		return "none"
	}
	return strings.TrimSpace(strings.ToLower(setting))
}
