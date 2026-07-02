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
//
// Liveness is mtime-based (a live agent appends within LiveMinutes); the shell supplies
// the age. Pure by construction: the I/O shell (cmd/fak resume stopped) walks the account
// dirs, tails each transcript, and extracts per-record facts; this leaf only classifies
// and decides. No clock, no filesystem.
package stopped

import (
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
	// DispStoppedQuiet: stopped with no recognizable terminal signal.
	DispStoppedQuiet = "STOPPED_QUIET"
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
	// BlockedBy is filled by Decide on deferred rows: why this session cannot resume now.
	BlockedBy string `json:"blocked_by,omitempty"`
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
	}

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

	d := Decisions{AccountThrottle: acctThrottle, Counts: map[string]int{}, Rows: sorted}
	for _, r := range sorted {
		d.Counts[r.Disp]++
		switch r.Disp {
		case DispStoppedMidtool, DispStoppedInterrupt, DispStoppedQuiet:
			if thr, ok := acctThrottle[r.Account]; ok {
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
		default: // LIVE / PARKED_WAIT / DONE
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
