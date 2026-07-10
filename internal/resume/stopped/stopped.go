// Package stopped is the pure decision core of the stopped-session triage: given the
// parsed tail of a top-level Claude Code transcript, classify how the session stopped
// (its DISPOSITION) and decide which stopped sessions are safe to resume headlessly,
// which must wait (account throttled / auth-walled), and which to leave alone. It is the
// Go port of tools/stopped_sessions.py's classify/decide core.
//
// The authoritative signals, learned from the on-disk transcript format:
//
//   - throttle  — a `<synthetic>` assistant message "... limit · resets <time>" means the
//     OWNING ACCOUNT is rate-limited until <time>; it is only CURRENT when the terminal
//     meaningful turn is that banner (a later clean turn supersedes it).
//   - mid-tool  — the last meaningful record is an assistant tool_use with no following
//     tool_result: the process died mid-work. The safest resume candidate.
//   - interrupt — the last meaningful text is a login/user interruption.
//   - waiting   — the last assistant text says it is parked on a background task; do not
//     resume (the harness will wake it).
//   - done      — the last assistant text reads as a wrap-up.
//   - residual  — none of the above matched: split by the terminal record's role (#3783).
//     An assistant-final tail is STOPPED_DONE (a finished turn went idle — leave alone); a
//     user-final tail is STOPPED_MIDTURN (a tool_result/user input was last — the model was
//     about to act, so work is stranded — resume-eligible); an empty tail stays STOPPED_QUIET.
//
// Liveness is mtime-based (a live agent appends within LiveMinutes); the shell supplies
// the age. Pure by construction: the I/O shell (cmd/fak resume stopped) walks the account
// dirs, tails each transcript, and extracts per-record facts; this leaf only classifies
// and decides. No clock, no filesystem.
package stopped

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/sessionsignals"
)

// LiveMinutes is the mtime freshness within which a session counts as LIVE — a live
// agent appends to its transcript at least this often.
const LiveMinutes = 4.0

// Record is the closed set of facts about ONE transcript line the classifier needs. The
// shell extracts these from the raw JSONL; the leaf never sees the JSON.
type Record struct {
	// Type is the record's top-level type; only "user" and "assistant" carry signals
	// (mode/permission-mode/summary/system and the other bookkeeping types are skipped).
	Type string
	// Role is the message role ("user"/"assistant"), falling back to Type when absent.
	Role string
	// Text is the record's human text (text blocks plus tool_result payload text).
	Text string
	// Timestamp is the record's ISO timestamp (informational; ordering is positional).
	Timestamp string
	// Synthetic marks message.model == "<synthetic>" — the injected banner channel the
	// throttle detection keys on.
	Synthetic bool
	// ToolUseName is the name of the LAST tool_use block in an assistant record ("" when
	// none) — an unmatched one at the tail means the process died mid-work.
	ToolUseName string
	// HasToolResult marks a user record carrying a tool_result block, which clears the
	// pending tool_use.
	HasToolResult bool
	// Session/context identity, updated from whichever records carry them.
	CWD, GitBranch, Version, SessionID string
}

// The closed disposition vocabulary.
const (
	// DispLive: the transcript was appended to within LiveMinutes — a live agent owns it.
	DispLive = "LIVE"
	// DispStoppedLimit: the terminal turn is a synthetic usage-limit banner — the owning
	// account is rate-limited until the named reset.
	DispStoppedLimit = "STOPPED_LIMIT"
	// DispStoppedAuth: the terminal text is an auth/credit/access wall.
	DispStoppedAuth = "STOPPED_AUTH"
	// DispStoppedInterrupt: the terminal text is a login/user interruption.
	DispStoppedInterrupt = "STOPPED_INTERRUPT"
	// DispStoppedMidtool: an assistant tool_use never got its tool_result — died mid-work.
	DispStoppedMidtool = "STOPPED_MIDTOOL"
	// DispParkedWait: the last assistant text says it is awaiting a background task — the
	// session is parked, not dead.
	DispParkedWait = "PARKED_WAIT"
	// DispDone: the last assistant text reads as a wrap-up.
	DispDone = "DONE"
	// DispStoppedDone: a residual stop whose terminal record is an ASSISTANT turn with
	// nothing after it — the model finished a turn and the session went idle without a
	// recognizable wrap-up phrase. Leave alone: a finished turn is not stranded work. Split
	// out of the STOPPED_QUIET umbrella by last_role (#3783).
	DispStoppedDone = "STOPPED_DONE"
	// DispStoppedMidturn: a residual stop whose terminal record is a USER turn (a tool_result
	// or user/file input) — the model was ABOUT TO ACT when the process died, so real work is
	// stranded mid-turn. Resume-eligible, unfinished work. Split out of the STOPPED_QUIET
	// umbrella by last_role (#3783). (An unmatched trailing tool_use is the more specific
	// DispStoppedMidtool, which still wins over this residual split.)
	DispStoppedMidturn = "STOPPED_MIDTURN"
	// DispStoppedQuiet: the residual umbrella — stopped with no recognizable terminal signal
	// AND no resolvable terminal role (an empty tail). Since #3783 the assistant-final and
	// user-final residuals are the typed DispStoppedDone / DispStoppedMidturn; this remains
	// only for the genuinely-unknown tail and as the back-compat alias.
	DispStoppedQuiet = "STOPPED_QUIET"
	// DispDupLive: a stopped session whose work a LIVE session in the same project already
	// owns — a crashed duplicate. Resuming it re-runs work in flight and collides on the
	// shared trunk, so it is a SKIP (never relaunched). Assigned by Decide, not Classify:
	// it is a cross-session verdict, not a property of one transcript.
	DispDupLive = "DUP_LIVE"
)

// Row is the classified verdict for one session transcript — the same fields the Python
// emitted, so the machine record keeps its shape.
type Row struct {
	Disp    string  `json:"disp"`
	AgeMin  float64 `json:"age_min"`
	SizeKB  int64   `json:"size_kb"`
	SeenUTC string  `json:"seen_utc"`
	Session string  `json:"session"`
	CWD     string  `json:"cwd,omitempty"`
	Git     string  `json:"git,omitempty"`
	Version string  `json:"version,omitempty"`
	// ThrottleReset is the banner's reset window ONLY when the throttle is current (the
	// terminal turn is the banner); ThrottleSeen is the last banner seen anywhere in the
	// tail, kept for observability even when a later clean turn superseded it.
	ThrottleReset   string `json:"throttle_reset,omitempty"`
	ThrottleSeen    string `json:"throttle_seen,omitempty"`
	ThrottleCurrent bool   `json:"throttle_current"`
	PendingTool     string `json:"pending_tool,omitempty"`
	LastRole        string `json:"last_role,omitempty"`
	Last            string `json:"last"`
	Path            string `json:"path"`
	Account         string `json:"account,omitempty"`
	Project         string `json:"project,omitempty"`
	// WorkKey is the caller-supplied identity of the work this session is doing — the
	// authoritative /goal, /loop lane, or issue number, sourced from the transcript's FIRST
	// user turn (and/or dispatch metadata) by the shell, NOT derived here (the tail the
	// classifier reads is too noisy to cluster on). Empty when the caller could not resolve
	// one; an empty WorkKey never dedups (a session with unknown work is never treated as a
	// duplicate — fail-open, resume it). See DupLiveScan in the shell.
	WorkKey string `json:"work_key,omitempty"`
	// BlockedBy is filled by Decide on deferred rows: why this session cannot resume now.
	BlockedBy string `json:"blocked_by,omitempty"`
	// MaxContextTokens is the target model's context window for the session this row would
	// resume ONTO — the ceiling the replay-safety fit check measures the estimated replayed
	// transcript against. The shell supplies it per session (from the target model's
	// advertised window); zero means "unknown," and Decide falls back to
	// DefaultResumeContextWindowTokens rather than skipping the fit check.
	MaxContextTokens int64 `json:"max_context_tokens,omitempty"`
}

var (
	// interruptRE is deliberately case-sensitive, matching the exact harness strings.
	interruptRE = regexp.MustCompile(`Login interrupted|\[Request interrupted by user`)
	parkedRE    = regexp.MustCompile(`(?i)still running|awaiting|wait for|will notify me|harness will|notify me when it completes|background`)
	doneRE      = regexp.MustCompile(`(?i)^\s*(Done|Shipped|Complete|Summary|All set|✅)\b|delivered\b|committed and pushed|pushed .* to origin`)
)

// Classify buckets one session from its parsed transcript tail. ageMin/sizeKB/seenUTC
// come from the file's stat (the shell's I/O); fallbackSession names the session when no
// record carried a sessionId (the filename stem); path is echoed for the operator.
func Classify(recs []Record, ageMin float64, sizeKB int64, seenUTC, fallbackSession, path string) Row {
	var cwd, git, ver, sid string
	throttleSeen := ""
	pendingTool := ""
	var last *Record
	for i := range recs {
		o := &recs[i]
		if o.CWD != "" {
			cwd = o.CWD
		}
		if o.GitBranch != "" {
			git = o.GitBranch
		}
		if o.Version != "" {
			ver = o.Version
		}
		if o.SessionID != "" {
			sid = o.SessionID
		}
		if o.Type != "user" && o.Type != "assistant" {
			continue
		}
		if o.Synthetic {
			if reset := sessionsignals.LimitReset(o.Text); reset != "" {
				throttleSeen = reset
			}
		}
		last = o
		if o.Type == "assistant" {
			if o.ToolUseName != "" {
				pendingTool = o.ToolUseName
			}
		} else if o.HasToolResult {
			// A user turn carrying a tool_result clears the pending tool_use.
			pendingTool = ""
		}
	}

	lt := ""
	lastRole := ""
	lastSynthetic := false
	if last != nil {
		lt = last.Text
		lastRole = last.Role
		if lastRole == "" {
			lastRole = last.Type
		}
		lastSynthetic = last.Synthetic
	}
	throttleCurrent := throttleSeen != "" && lastSynthetic

	disp := DispStoppedQuiet
	switch {
	case throttleCurrent:
		disp = DispStoppedLimit
	case sessionsignals.IsAuthError(lt):
		disp = DispStoppedAuth
	case ageMin <= LiveMinutes:
		disp = DispLive
	case interruptRE.MatchString(lt):
		disp = DispStoppedInterrupt
	case pendingTool != "":
		disp = DispStoppedMidtool
	case parkedRE.MatchString(lt):
		disp = DispParkedWait
	case doneRE.MatchString(lt):
		disp = DispDone
	case lastRole == "assistant":
		// Residual stop, assistant-final: the model finished a turn and nothing followed —
		// idle/done, safe to leave (#3783). (No explicit wrap-up phrase, or doneRE would have
		// matched above.)
		disp = DispStoppedDone
	case lastRole == "user":
		// Residual stop, user-final: a tool_result or user/file input was the last record —
		// the model was mid-turn when it died, so real work is stranded. Resume-eligible (#3783).
		disp = DispStoppedMidturn
	}
	// disp stays DispStoppedQuiet only when no terminal role resolved (an empty tail).

	session := sid
	if session == "" {
		session = fallbackSession
	}
	throttleReset := ""
	if throttleCurrent {
		throttleReset = throttleSeen
	}
	return Row{
		Disp: disp, AgeMin: math.Round(ageMin*10) / 10, SizeKB: sizeKB, SeenUTC: seenUTC,
		Session: session, CWD: cwd, Git: git, Version: ver,
		ThrottleReset: throttleReset, ThrottleSeen: throttleSeen, ThrottleCurrent: throttleCurrent,
		PendingTool: pendingTool, LastRole: lastRole, Last: clipLast(lt, 300), Path: path,
	}
}

// Throttle is one account's most-recent active limit banner: the reset window and how
// old the banner is (the freshest banner per account wins).
type Throttle struct {
	Reset  string  `json:"reset"`
	AgeMin float64 `json:"age_min"`
}

// Decisions is the triage verdict over all classified rows.
type Decisions struct {
	// AccountThrottle maps an account to its most-recent ACTIVE limit banner — the
	// account-level block a resume of ANY of its sessions must wait behind.
	AccountThrottle map[string]Throttle `json:"account_throttle"`
	Counts          map[string]int      `json:"counts"`
	// Resume: safe to resume headlessly now. Defer: blocked (each row's BlockedBy says
	// why). Skip: LIVE / PARKED_WAIT / DONE — not resume candidates at all.
	Resume []Row `json:"resume"`
	Defer  []Row `json:"defer"`
	Skip   []Row `json:"skip"`
	// Rows is every classified row, youngest first — the full observability record.
	Rows []Row `json:"rows"`
}

// Replay-safety fit-check constants (#3355). A resume replays the stopped session's
// transcript back into the target model; if that replay would exceed the model's context
// window, a blind relaunch silently truncates or corrupts the session. These pin the
// conservative, fail-closed estimate the fit check refuses on.
const (
	// EstimatedTokensPerKB converts a transcript's on-disk size to an estimated replayed
	// token count. It is deliberately an OVER-estimate of the replayed tokens (it treats
	// every on-disk KB as ~4-byte tokens of replayed text, though a JSONL transcript's
	// structural/tool-metadata bytes do not all replay into the prompt) — so the fit check
	// errs toward refusing a borderline session rather than resuming one that would
	// overflow. The asymmetry is deliberate: a falsely-refused session is DEFERRED to a
	// human (recoverable), a falsely-resumed over-window session is truncated (data loss).
	EstimatedTokensPerKB int64 = 256
	// DefaultResumeContextWindowTokens is the target-window fallback used when a Row carries
	// no MaxContextTokens — a modern large-context window, so the fit check trips only on a
	// genuinely oversized transcript (over ~780 KB) rather than false-refusing ordinary
	// sessions. A caller that knows the exact target model supplies Row.MaxContextTokens.
	DefaultResumeContextWindowTokens int64 = 200000
)

// estimatedReplayTokens is the conservative replayed-token estimate for a transcript of
// sizeKB kilobytes. Non-positive size yields zero (nothing to replay).
func estimatedReplayTokens(sizeKB int64) int64 {
	if sizeKB <= 0 {
		return 0
	}
	return sizeKB * EstimatedTokensPerKB
}

// replaySafetyBlock returns a non-empty, witnessed BlockedBy reason when a row must NOT be
// resumed on a replay-safety precondition — today the single rule: the estimated replayed
// transcript would overflow the target model's context window. It is fail-closed by
// construction: an over-window resume silently truncates/corrupts, so the row is deferred
// (a human or a clean-continuation reset owns it) instead of blindly relaunched. An empty
// return means the row cleared the fit check. The verdict is derived mechanically from the
// row's own SizeKB and target window, never from any self-report.
func replaySafetyBlock(r Row) string {
	window := r.MaxContextTokens
	if window <= 0 {
		window = DefaultResumeContextWindowTokens
	}
	if est := estimatedReplayTokens(r.SizeKB); est > window {
		return fmt.Sprintf("replayed transcript ~%d tokens exceeds target context window %d — resume would overflow", est, window)
	}
	return ""
}

// Decide sorts rows youngest-first, folds the per-account active throttles, and buckets
// every row into resume/defer/skip. throttleActive reports whether a reset window is
// still blocking (unparseable resets are conservatively active); it is injected so the
// decision stays clock-free.
func Decide(rows []Row, throttleActive func(reset string) bool) Decisions {
	sorted := make([]Row, len(rows))
	copy(sorted, rows)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].AgeMin < sorted[j].AgeMin })

	acctThrottle := map[string]Throttle{}
	for _, r := range sorted {
		if r.ThrottleReset == "" || r.Disp != DispStoppedLimit || !throttleActive(r.ThrottleReset) {
			continue
		}
		cur, ok := acctThrottle[r.Account]
		if !ok || r.AgeMin < cur.AgeMin {
			acctThrottle[r.Account] = Throttle{Reset: r.ThrottleReset, AgeMin: r.AgeMin}
		}
	}

	// Cross-session dedup: a LIVE session's work-key is OWNED. A stopped session that shares
	// that (project, work-key) is a crashed duplicate — resuming it re-runs work in flight and
	// collides on the shared trunk. Build the owned set from the LIVE rows, keyed by project so
	// the same /loop lane in two different repos never cross-dedups. An empty work-key never
	// participates (fail-open: unknown work is resumed, never silently dropped).
	liveOwned := map[string]bool{}
	for _, r := range sorted {
		if r.Disp == DispLive && r.WorkKey != "" {
			liveOwned[r.Project+"\x00"+r.WorkKey] = true
		}
	}

	d := Decisions{AccountThrottle: acctThrottle, Counts: map[string]int{}, Rows: sorted}
	for _, r := range sorted {
		// A crashed duplicate of a live session skips before any resume decision. The
		// leave-alone states (LIVE / DONE / STOPPED_DONE / PARKED) are left to their own
		// buckets — only a would-be resume candidate is redirected. Excluding STOPPED_DONE
		// keeps a finished session's cause intact rather than masking it with DUP_LIVE (#3783).
		if r.WorkKey != "" && r.Disp != DispLive && r.Disp != DispDone && r.Disp != DispStoppedDone &&
			r.Disp != DispParkedWait && liveOwned[r.Project+"\x00"+r.WorkKey] {
			r.Disp = DispDupLive
			r.BlockedBy = "duplicate of a live session owning the same work (" + r.WorkKey + ")"
			d.Counts[DispDupLive]++
			d.Skip = append(d.Skip, r)
			continue
		}
		d.Counts[r.Disp]++
		switch r.Disp {
		case DispStoppedMidtool, DispStoppedInterrupt, DispStoppedMidturn, DispStoppedQuiet:
			// Replay-safety precondition first: a resume whose replayed transcript would
			// overflow the target context window is a STRUCTURAL block that no reset clears,
			// so it is reported ahead of a transient account throttle. Fail-closed — an
			// over-window row is deferred, never blindly relaunched into a truncated context.
			if reason := replaySafetyBlock(r); reason != "" {
				r.BlockedBy = reason
				d.Defer = append(d.Defer, r)
			} else if thr, ok := acctThrottle[r.Account]; ok {
				r.BlockedBy = "account throttled, resets " + thr.Reset
				d.Defer = append(d.Defer, r)
			} else {
				d.Resume = append(d.Resume, r)
			}
		case DispStoppedLimit:
			r.BlockedBy = "session limit, resets " + r.ThrottleReset
			d.Defer = append(d.Defer, r)
		case DispStoppedAuth:
			r.BlockedBy = "account auth/subscription disabled"
			d.Defer = append(d.Defer, r)
		default: // LIVE / PARKED_WAIT / DONE / STOPPED_DONE — leave alone, never a resume candidate
			d.Skip = append(d.Skip, r)
		}
	}
	return d
}

// clipLast bounds the terminal-text echo to width runes on one line, the Python
// lt[:300].replace("\n", " ") contract.
func clipLast(s string, width int) string {
	rs := []rune(s)
	if len(rs) > width {
		rs = rs[:width]
	}
	return strings.ReplaceAll(string(rs), "\n", " ")
}
