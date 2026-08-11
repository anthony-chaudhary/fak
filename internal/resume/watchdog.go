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
//  1. self-guard      — never resume the session the watchdog itself runs inside (a live
//     operator session can briefly look like a stopped autonomous worker, and a
//     self-resume races two `claude` processes on one transcript);
//  2. operator-hold   — never resume a session an operator DELIBERATELY paused / drained /
//     stopped via the durable, UUID-keyed drive-state store (resume_drivestate.jsonl,
//     written by `fak resume hold`). Operator intent OUTRANKS transcript-forensic resume,
//     and this sits above the retry gate so it beats even a proven re-death revive — the
//     watchdog must not resurrect what the operator terminated. It is keyed by the Claude
//     transcript UUID the plan row already carries (the descriptor registry is a disjoint
//     keyspace — guardTraceID — so it can never provide this join);
//  3. policy guard    — never resume onto an account the fleet policy no longer offers as
//     a worker (a stale plan file can predate a tombstone);
//  4. retry gate      — the outcome-aware once-gate (RetryGate): blocked unless the last
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
	// PartialBlocks is the interrupted trailing turn's emitted blocks (#5927), attached by
	// the SHELL from the newest transcript tail at decision time — never part of the
	// fleet_sessions.py plan JSON. The shell feeds blocks only for a turn the transcript
	// proves INCOMPLETE (the model still owes a reply, or a tool call has no matching
	// result); a cleanly-completed trailing answer is not a partial turn. Nil/empty
	// (unreadable transcript, nothing pending) leaves the replay-safety gate inert.
	PartialBlocks []EmittedBlock `json:"partial_blocks,omitempty"`

	// Harness is explicit because a resume actuator must never infer the provider from
	// prompt prose or an executable basename. Empty remains the legacy Claude default.
	Harness string `json:"harness,omitempty"`
	// Rollout and GoalFile are Codex-only continuation coordinates. GoalFile contains
	// operator-authored continuation data and keeps prompt text out of argv/launch ledgers.
	Rollout    string `json:"rollout,omitempty"`
	GoalFile   string `json:"goal_file,omitempty"`
	ResultFile string `json:"result_file,omitempty"`
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
	// WatchdogSkipLive: a live `claude --resume <sid>` process is ALREADY driving this
	// session (some OTHER account dir holds a newer, actively-advancing copy) — launching
	// would race a second driver onto a live transcript. The generalization of the
	// self-guard to any live driver, closing the "queued a live session as crashed" harm
	// (#3459): a stale/older copy can classify STOPPED_APIERR while the session is alive.
	WatchdogSkipLive WatchdogAction = "skip_live"
	// WatchdogSkipNonWorker: the row's account is not an offered worker under the current
	// fleet policy (tombstoned/excluded) — a stale plan must not resurrect it.
	WatchdogSkipNonWorker WatchdogAction = "skip_non_worker"
	// WatchdogSkipBlocked: the outcome-aware retry gate blocks a new resume (already
	// took, auth wall, attempt cap, or operator-settled). The reason carries the why.
	WatchdogSkipBlocked WatchdogAction = "skip_blocked"
	// WatchdogSkipOperatorHold: an operator deliberately drove this session to a hold
	// (paused / draining) or a terminal stop via the durable drive-state store — operator
	// drive-state OUTRANKS transcript-forensic resume, so never resurrect it, even past a
	// re-death latch. The reason names which drive-state held it.
	WatchdogSkipOperatorHold WatchdogAction = "skip_operator_hold"
)

// The operator drive-state vocabulary (WatchdogDriveState, HeldByOperator, HoldReason) and the
// durable-store fold (DriveStateRow, FoldDriveStates) live in drivestate.go — the pure leaf the
// shell folds resume_drivestate.jsonl through before handing this guard its DriveStates map.

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
	// LiveSIDs is the set of session ids a live `claude --resume <sid>` process is currently
	// driving — the SHELL folds it from the same audited process census `fak resume admit`
	// counts with (liveResumeSIDs / procguard.CollectRelations). A nil/empty map, or a
	// session with no entry, leaves the liveness guard INERT (fail-open, per-key — an
	// unreadable process table must never strand a genuinely-crashed session). The plan is
	// written by fleet_sessions.py off the on-disk transcripts, which can classify a
	// stale/older copy as crashed while a newer copy under another account dir is alive
	// (#3459); this gate is the driver-side defense-in-depth that refuses to launch a second
	// driver regardless of what disposition the stale copy produced.
	LiveSIDs map[string]bool `json:"live_sids,omitempty"`
	// DriveStates maps a session id (the Claude transcript UUID the plan row carries) to the
	// operator drive-state the SHELL folded from the durable drive-state store. A nil/empty
	// map, a session with no entry, or a running/throttled/unknown token all leave the
	// operator-hold guard INERT (fail-open, per-key — NOT the roster-style "absent ⇒ skip"
	// rule the worker-account guard uses; the guard fires only on an explicit held token).
	DriveStates map[string]WatchdogDriveState `json:"drive_states,omitempty"`
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
	// Liveness guard (#3459): a live `claude --resume` process is already driving this
	// session under some account dir — the plan classified a stale/older transcript copy as
	// crashed while a newer copy is actively advancing. Sits right below the self-guard (the
	// same "never race two drivers on one transcript" rule, generalized past the watchdog's
	// own session) and above every disposition-dependent gate, so a live driver is honored
	// no matter what STOPPED_* disposition the stale copy carried. Fail-open per key: an
	// absent/empty census (unreadable process table) leaves it inert.
	if g.LiveSIDs[row.Session] {
		return WatchdogRowDecision{Action: WatchdogSkipLive,
			Reason: "a live claude --resume process is already driving this session (liveness guard)"}
	}
	// Operator-hold: an operator deliberately paused/drained/stopped this session via the
	// durable drive-state store. It sits ABOVE the policy and retry gates (and the shell has
	// already applied ReviveOutcome before this call), so an operator Stop beats a proven
	// re-death revive, a spent cap, and a non-worker account alike. Fail-open per key: only
	// an explicit held token fires — a missing/running/unknown state falls through.
	if st := g.DriveStates[row.Session]; st.HeldByOperator() {
		return WatchdogRowDecision{Action: WatchdogSkipOperatorHold, Reason: st.HoldReason()}
	}
	if len(g.WorkerAccounts) > 0 && !g.WorkerAccounts[row.Account] {
		return WatchdogRowDecision{Action: WatchdogSkipNonWorker,
			Reason: fmt.Sprintf("account %s is not an offered worker (policy/tombstoned)", row.Account)}
	}
	if d := RetryGate(history, outcome, g.MaxAttempts); d.Blocked {
		return WatchdogRowDecision{Action: WatchdogSkipBlocked, Reason: d.Reason}
	}
	// Replay-safety conjunct (#5927): the retry gate above judged the ERROR; this judges
	// what the interrupted turn already EMITTED. It is an additional conjunct only — it can
	// narrow an eligible retry, never overturn a Blocked verdict or reclassify the error.
	// A partial whose tool calls all carry matching results is preserve-and-continued:
	// `claude --resume` continues the preserved transcript, which is exactly that
	// actuation, so it launches with the distinct reason. A partial that emitted
	// replay-unsafe output — flagship: a tool call with no matching result, whose side
	// effect cannot be proven absent — suppresses the auto-retry with the reason on the
	// row, never silently.
	switch pd := DecidePartialRetry(RetryableError, row.PartialBlocks); pd.Action {
	case PartialRetrySuppressed:
		return WatchdogRowDecision{Action: WatchdogSkipBlocked,
			Reason: fmt.Sprintf("interrupted turn emitted replay-unsafe output; auto-retry suppressed (%s)", pd.Reason)}
	case PartialPreserveContinue:
		return WatchdogRowDecision{Action: WatchdogLaunch,
			Reason:  fmt.Sprintf("resume continues the preserved partial turn (%s)", pd.Reason),
			Attempt: CountAttempts(history) + 1}
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

// CodexWatchdogChildEnv prevents a fresh Codex process from inheriting Claude gateway,
// Claude seat, or parent harness identity while preserving CODEX_HOME and provider auth.
func CodexWatchdogChildEnv(environ []string) []string {
	drop := map[string]bool{
		"ANTHROPIC_API_KEY": true, "ANTHROPIC_AUTH_TOKEN": true, "ANTHROPIC_BASE_URL": true,
		"CLAUDE_CONFIG_DIR": true, "CLAUDE_CODE_SESSION_ID": true, "CLAUDE_CODE_CHILD_SESSION": true,
		"JOB_SUPERVISED_WORKER": true,
	}
	out := make([]string, 0, len(environ))
	for _, kv := range environ {
		key := kv
		if at := strings.IndexByte(kv, '='); at >= 0 {
			key = kv[:at]
		}
		if !drop[key] {
			out = append(out, kv)
		}
	}
	return out
}

// --- the stale resume-took latch (#2368) --------------------------------------------

// ReDeathEvidence is the typed, shell-extracted proof that a session whose last resume
// TOOK (or whose outcome is unreadable) has since died AGAIN. The burn-once retry gate
// reads a progressed/unknown outcome as "resumed once — done"; that latch goes stale
// the moment the session stops a second time: the sweep re-plans the session every
// tick and the watchdog skips it every tick with "already resumed once (resume took)",
// forever. Every field defaults to "unproven", so the zero value revives nothing and a
// caller that cannot gather evidence keeps today's conservative burn-once behavior.
type ReDeathEvidence struct {
	// ProcessScanOK: the process table was actually readable. False = liveness unknown
	// → never revive (racing two `claude` processes on one transcript is worse than
	// waiting a tick for a readable table).
	ProcessScanOK bool `json:"process_scan_ok"`
	// ProcessLive: some live process's command line names this session id (the
	// `claude --resume <sid>` child still running).
	ProcessLive bool `json:"process_live"`
	// TranscriptIdleSeconds: seconds since the session's newest transcript copy last
	// changed; -1 when no transcript could be found (unknown → never revive).
	TranscriptIdleSeconds int64 `json:"transcript_idle_seconds"`
}

// DeadTranscriptIdleFloorSeconds is how long a session's transcript must have been
// silent before "no live process" reads as a new death rather than startup latency: a
// just-fired resume can take a while to write its first record, and a live session
// deep in one long tool call writes nothing either — but neither state survives ten
// minutes of transcript silence WITH no process holding the session id.
const DeadTranscriptIdleFloorSeconds int64 = 10 * 60

// DiedAgain reports whether the evidence proves a new death: the process table was
// readable, no live process holds the session id, and the transcript has been silent
// past the dead floor. Any unknown fact fails the proof — a wrong revive races two
// processes on one transcript NOW, while a wrong keep just waits for stronger evidence
// on a later tick.
func (ev ReDeathEvidence) DiedAgain() bool {
	return ev.ProcessScanOK && !ev.ProcessLive &&
		ev.TranscriptIdleSeconds >= DeadTranscriptIdleFloorSeconds
}

// ReviveOutcome folds re-death evidence into the outcome the retry gate reads. With
// proof of a new death, a progressed/unknown outcome reads RECOVERABLE, which
// re-admits the session through RetryGate under the same attempt cap while every
// higher-precedence block (operator-settled, auth wall, spent cap) keeps binding
// exactly as before. The bool reports whether the latch was released, so the shell
// narrates the override instead of silently re-launching.
func ReviveOutcome(outcome Outcome, ev ReDeathEvidence) (Outcome, bool) {
	if (outcome == OutcomeProgressed || outcome == OutcomeUnknown) && ev.DiedAgain() {
		return OutcomeRecoverable, true
	}
	return outcome, false
}

// ResolveWatchdogProbeMode resolves the FAK_PROBE setting for a tick: "auto" probes
// STALE accounts only on a LIVE tick (so the default dry-run stays side-effect-free —
// no probe spend), and an explicit setting is honored as-is. Mirrors
// fleet_resume_watchdog.resolve_probe_mode, which the .ps1's -Probe auto behavior set.
//
// "stale" (blocked OR idle with no live-session evidence) rather than "blocked": a
// passive available verdict only proves the seat was serving at its LAST activity. An
// idle seat that burned its session budget after that activity still reads available,
// so probing only blocked accounts let the planner re-home a crashed session onto a
// limit-walled seat (observed 2026-07-06: july6 idle 90m past its wall, admitted as a
// rehome target, the resume died on arrival). One paced pong per idle seat keeps the
// positive evidence fresh enough to take load.
func ResolveWatchdogProbeMode(setting string, live bool) string {
	if strings.TrimSpace(strings.ToLower(setting)) == "auto" || strings.TrimSpace(setting) == "" {
		if live {
			return "stale"
		}
		return "none"
	}
	return strings.TrimSpace(strings.ToLower(setting))
}
