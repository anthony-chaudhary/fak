package sessionreg

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionsignals"
)

// LiveMinutes is the transcript-append freshness inside which a session reads LIVE.
const LiveMinutes = 4.0

// neverStartedMaxBytes is the read_tail window: a transcript this small is read whole, so
// "no assistant turn" is authoritative rather than an artifact of the tail cut.
const neverStartedMaxBytes = 512 * 1024

const headRecords = 30

var (
	uuidRE      = regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)
	doneRE      = mustCompilePy(`(?i)\b(complete|completed|shipped|pushed|committed|delivered|lease released|all checks|all set|terminated cleanly|goal is met|witness (?:holds|is met)|this completes)\b`)
	parkRE      = regexp.MustCompile(`(?i)still running|awaiting|wait for the|will notify me|harness will|notify me when|waiting for the (?:exit|completion|result)|holding`)
	autonRE     = mustCompilePy(`(?i)<command-name>/(goal|loop|dispatch|next-up|fanout)\b|Stop hook is now active|/loop\b|ScheduleWakeup|autonomous-loop|keep working until`)
	supervRE    = mustCompilePy(`(?i)JOB_SUPERVISED_WORKER|supervisor-spawned|/dispatch-loop\b`)
	wrapperRE   = mustCompilePy(`(?i)^\s*(?:Caveat:|<system-reminder|<command-name>|<command-message>|<command-args>|<local-command-|<user-memory|Codebase and user instructions)`)
	taskCmdRE   = mustCompilePy(`(?i)<command-name>\s*/(goal|loop|dispatch|fanout|next-up)\b`)
	cmdArgsRE   = regexp.MustCompile(`(?is)<command-args>(.*?)</command-args>`)
	userCloseRE = mustCompilePy(`<command-name>/(quit|exit|clear|logout|login)\b`)
	// The classifier's transient-transport vocabulary. sessionsignals.IsAPIError also folds
	// codex rate-limit wording and bare request timeouts, which this classifier has never
	// read as STOPPED_APIERR.
	apiErrRE = mustCompilePy(`(?i)isApiErrorMessage|API Error|overloaded_error|\boverloaded\b|\b429\b|\b529\b|\b503\b|fetch failed|ECONNRESET|ETIMEDOUT|socket hang up|Internal Server Error|service unavailable|connection error|network error`)
	// The sessionsignals forms of these two use Go's ASCII \b, which finds "529" in "é529".
	httpStatusRE   = mustCompilePy(`\b(401|403|429|500|502|503|529)\b`)
	operatorStopRE = mustCompilePy(`(?i)BUDGET_CONTEXT_EXHAUSTED|\b409\b.*operator\s+(?:control|stop)|restart_fresh_session`)
)

func httpStatus(text string) string {
	if m := httpStatusRE.FindStringSubmatch(text); m != nil {
		return m[1]
	}
	return ""
}

// resumePromptPrefix opens every re-homed transcript identically, so it is never a task
// instruction.
const resumePromptPrefix = "Resume where you left off"

type category struct{ category, cause string }

var categories = map[string]category{
	"LIVE":                      {"LIVE", "live"},
	"DONE":                      {"AGENT", "completed"},
	"DEAD_MIDTOOL":              {"AGENT", "crash_mid_tool"},
	"DEAD_KILLED":               {"AGENT", "killed_mid_turn"},
	"USER_CLOSED":               {"USER", "user_stopped"},
	"STOPPED_LIMIT":             {"INFRA", "rate_limit"},
	"STOPPED_APIERR":            {"INFRA", "api_error"},
	"STOPPED_CONTEXT_EXHAUSTED": {"INFRA", "context_exhausted"},
	"INFRA_AUTH":                {"INFRA", "auth"},
	"INFRA_ORG_DISABLED":        {"INFRA", "org_disabled"},
	"NEVER_STARTED":             {"AGENT", "never_started"},
	"PARKED_WAIT":               {"HANGING", "parked_on_task"},
	"STOPPED_QUIET":             {"HANGING", "ambiguous_quiet"},
}

// Row is one classified transcript copy plus the decision stamped onto it. Fields that the
// Python writer emits as null when absent carry a has-flag or an `any`.
type Row struct {
	Disp, Category, Cause, Reason string
	LastKind                      string // "" -> null
	AgeMin                        float64
	SeenUTC                       string
	LastTS                        string
	ThrottleReset                 string // "" -> null
	ThrottleWeekly                string // "" -> null
	ThrottleSeen                  string // "" -> null
	ThrottleCurrent               bool
	PendingTool                   any
	HTTPStatus                    string // "" -> null
	InitiatedBy                   string
	Session                       string
	Cwd                           any
	Git                           any
	Last                          string
	Path                          string
	Autonomous, Supervised        bool
	TaskSig                       string
	Records                       int64

	Account, Project string

	Action          string
	actionEarly     bool // stamped by a pre-pass, so its key precedes the decide() keys
	decided         bool
	SourceConfigDir string
	ResumeAccount   string
	ResumeConfigDir string
	Rehomed         bool
	ResumeCmd       string
	hasResumeCmd    bool
}

func (r *Row) cwdString() string {
	s, _ := r.Cwd.(string)
	return s
}

func (r *Row) stampEarly(action string) {
	if r.Action == "" {
		r.actionEarly = true
	}
	r.Action = action
}

// readTail returns the last 512 KiB of path decoded and split the way Python's
// read_tail does (errors="replace", str.splitlines).
func readTail(path string) []string {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	size, err := f.Seek(0, io.SeekEnd)
	if err != nil {
		return nil
	}
	start := size - neverStartedMaxBytes
	if start < 0 {
		start = 0
	}
	if _, err := f.Seek(start, io.SeekStart); err != nil {
		return nil
	}
	b, err := io.ReadAll(f)
	if err != nil {
		return nil
	}
	return pySplitlines(validUTF8(b))
}

// readHead returns the first n lines in text mode (universal newlines: \n, \r\n, \r).
func readHead(path string, n int) []string {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	br := bufio.NewReader(f)
	var out []string
	var cur []byte
	flush := func() {
		out = append(out, validUTF8(cur))
		cur = cur[:0]
	}
	for len(out) < n {
		c, err := br.ReadByte()
		if err != nil {
			if len(cur) > 0 {
				flush()
			}
			break
		}
		switch c {
		case '\n':
			flush()
		case '\r':
			if nx, perr := br.Peek(1); perr == nil && nx[0] == '\n' {
				_, _ = br.ReadByte()
			}
			flush()
		default:
			cur = append(cur, c)
		}
	}
	return out
}

func parseRecord(line string) map[string]any {
	line = pyStrip(line)
	if line == "" {
		return nil
	}
	var o map[string]any
	if json.Unmarshal([]byte(line), &o) == nil {
		return o
	}
	// json.loads accepts the NaN/Infinity constants, so such a record still counts there.
	if relaxed, ok := pyJSONConstantsToNull(line); ok && json.Unmarshal([]byte(relaxed), &o) == nil {
		return o
	}
	// A line that is valid JSON but not an object stops the Python scan with an
	// AttributeError; skipping it keeps one stray line from failing the whole registry.
	return nil
}

// pyJSONConstantsToNull rewrites the NaN, Infinity and -Infinity tokens outside strings as
// null, reporting whether any was found.
func pyJSONConstantsToNull(s string) (string, bool) {
	var b strings.Builder
	found, inStr := false, false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if inStr {
			b.WriteByte(c)
			switch c {
			case '\\':
				if i+1 < len(s) {
					i++
					b.WriteByte(s[i])
				}
			case '"':
				inStr = false
			}
			continue
		}
		if c == '"' {
			inStr = true
			b.WriteByte(c)
			continue
		}
		matched := false
		for _, tok := range []string{"-Infinity", "Infinity", "NaN"} {
			if strings.HasPrefix(s[i:], tok) {
				b.WriteString("null")
				i += len(tok) - 1
				found, matched = true, true
				break
			}
		}
		if !matched {
			b.WriteByte(c)
		}
	}
	return b.String(), found
}

func asMap(v any) map[string]any {
	m, _ := v.(map[string]any)
	return m
}

// textOf flattens a message content value: strings verbatim, text blocks and tool_result
// payloads joined by single spaces.
func textOf(content any) string {
	switch t := content.(type) {
	case string:
		return t
	case []any:
		var out []string
		for _, e := range t {
			b := asMap(e)
			if b == nil {
				continue
			}
			switch b["type"] {
			case "text":
				if s, ok := b["text"].(string); ok && s != "" {
					out = append(out, s)
				}
			case "tool_result":
				var s string
				if cs, ok := b["content"].(string); ok {
					s = cs
				} else {
					s = textOf(b["content"])
				}
				if s != "" {
					out = append(out, s)
				}
			}
		}
		return strings.Join(out, " ")
	}
	return ""
}

func lastToolUse(content any) any {
	list, ok := content.([]any)
	if !ok {
		return nil
	}
	for i := len(list) - 1; i >= 0; i-- {
		if b := asMap(list[i]); b != nil && b["type"] == "tool_use" {
			return b["name"]
		}
	}
	return nil
}

func headText(ho map[string]any) string {
	if t := ho["type"]; t != "user" && t != "system" {
		return ""
	}
	var mc any = ""
	msg := asMap(ho["message"])
	if v, ok := msg["content"]; ok && msg != nil {
		mc = v
	} else if v, ok := ho["content"]; ok {
		mc = v
	}
	if s, ok := mc.(string); ok {
		return s
	}
	return textOf(mc)
}

// firstInstruction is the session's real first task instruction (the dedup identity): the
// first head user/system text that is not a harness wrapper or the re-home resume prompt,
// prefixed by a task-defining command's argument payload when one was seen.
func firstInstruction(head []map[string]any) string {
	cmdArgs := ""
	for _, ho := range head {
		if t := ho["type"]; t != "user" && t != "system" {
			continue
		}
		txt := headText(ho)
		if pyStrip(txt) == "" {
			continue
		}
		if taskCmdRE.MatchString(txt) {
			if m := cmdArgsRE.FindStringSubmatch(txt); m != nil && pyStrip(m[1]) != "" {
				cmdArgs = pySplitJoin(m[1])
			}
		}
		stripped := pyStrip(txt)
		if wrapperRE.MatchString(stripped) || strings.HasPrefix(stripped, resumePromptPrefix) {
			continue
		}
		if cmdArgs != "" {
			return pySplitJoin(cmdArgs + " " + stripped)
		}
		return pySplitJoin(stripped)
	}
	return cmdArgs
}

func taskSig(project, cwd, instruction string) string {
	if instruction == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(project + "\x00" + cwd + "\x00" + runeHead(instruction, 400)))
	return hex.EncodeToString(sum[:])[:16]
}

// classify reads one transcript copy and assigns its disposition from the terminal turn.
func classify(path string, st os.FileInfo, now time.Time) *Row {
	mtime := mtimeUTC(st.ModTime())
	age := minutesSince(now, mtime)
	var cwd, git, sid any
	throttle, throttleWeekly := "", ""
	var pending any
	type lastRec struct {
		role any
		txt  string
		syn  bool
	}
	var last *lastRec
	lastTS, lastKind := "", ""
	sawAssistant := false
	for _, ln := range readTail(path) {
		o := parseRecord(ln)
		if o == nil {
			continue
		}
		if v, ok := o["cwd"]; ok {
			cwd = v
		}
		if v, ok := o["gitBranch"]; ok {
			git = v
		}
		if v, ok := o["sessionId"]; ok {
			sid = v
		}
		typ := o["type"]
		if typ != "user" && typ != "assistant" {
			continue
		}
		m := asMap(o["message"])
		c := m["content"]
		txt := textOf(c)
		syn := m["model"] == "<synthetic>"
		if syn {
			w := sessionsignals.LimitResets(txt)
			primary := w.Daily
			if primary == "" {
				primary = w.Weekly
			}
			if primary != "" {
				throttle, throttleWeekly = primary, w.Weekly
			}
		}
		role, ok := m["role"]
		if !ok {
			role = typ
		}
		last = &lastRec{role: role, txt: txt, syn: syn}
		if ts := o["timestamp"]; truthy(ts) {
			if s := pyStr(ts); s != "" {
				lastTS = s
			}
		}
		if typ == "assistant" {
			if !syn {
				sawAssistant = true
			}
			if n := lastToolUse(c); truthy(n) {
				pending = n
				lastKind = "assistant_tooluse"
			} else if m["stop_reason"] == "end_turn" {
				lastKind = "assistant_end"
			} else {
				lastKind = "assistant_text"
			}
			continue
		}
		isTR := false
		if list, ok := c.([]any); ok {
			for _, e := range list {
				if b := asMap(e); b != nil && b["type"] == "tool_result" {
					isTR = true
					break
				}
			}
		}
		if isTR {
			pending = nil
		}
		switch {
		case strings.Contains(txt, "[Request interrupted by user"):
			lastKind = "user_interrupt"
		case strings.Contains(txt, "Login interrupted") || userCloseRE.MatchString(txt):
			lastKind = "user_close"
		case isTR:
			lastKind = "user_toolresult"
		default:
			lastKind = "user_text"
		}
	}
	lt := ""
	if last != nil {
		lt = last.txt
	}
	lastIsAssistant := last != nil && last.role == "assistant"
	throttleCurrent := throttle != "" && last != nil && last.syn
	var disp, reason string
	switch {
	case throttleCurrent:
		disp, reason = "STOPPED_LIMIT", "hit account session limit; resets "+throttle
	case sessionsignals.IsAuthError(lt):
		switch sessionsignals.AuthBlockKind(lt) {
		case "access":
			disp, reason = "INFRA_ORG_DISABLED", "stopped on a Claude subscription/access wall (re-home to a usable account; /login won't fix the owner)"
		case "credit":
			disp, reason = "INFRA_AUTH", "stopped on account credit/billing state"
		default:
			disp, reason = "INFRA_AUTH", "stopped on an auth/login requirement (needs re-login)"
		}
	case operatorStopRE.MatchString(lt) && lastKind != "assistant_end":
		disp, reason = "STOPPED_CONTEXT_EXHAUSTED", "stopped on an operator-stop/context-exhausted wall (needs a fresh continuation, not a resume)"
	case apiErrRE.MatchString(lt) && !sessionsignals.IsAuthError(lt) && !operatorStopRE.MatchString(lt) && lastKind != "assistant_end":
		disp, reason = "STOPPED_APIERR", "stopped on an API/transport error (transient infra)"
	case age <= LiveMinutes:
		disp, reason = "LIVE", "appended within 4 min"
	case lastKind == "user_interrupt" || lastKind == "user_close":
		disp, reason = "USER_CLOSED", "ended on "+lastKind+" (user intentionally stopped it)"
	case truthy(pending):
		disp, reason = "DEAD_MIDTOOL", "died mid tool_use ("+pyStr(pending)+") with no tool_result"
	case !sawAssistant && st.Size() <= neverStartedMaxBytes:
		disp, reason = "NEVER_STARTED", "launched but produced no assistant turn (never started)"
	case parkRE.MatchString(lt):
		disp, reason = "PARKED_WAIT", "parked awaiting a background task"
	case lastKind == "assistant_end" || (lastIsAssistant && doneRE.MatchString(lt)):
		disp, reason = "DONE", "last assistant turn ended cleanly (stop_reason=end_turn / wrap-up)"
	case lastKind == "user_toolresult":
		disp, reason = "DEAD_KILLED", "killed after a tool_result before the next assistant turn"
	case lastIsAssistant:
		disp, reason = "DONE", "ended on an assistant message"
	default:
		disp, reason = "STOPPED_QUIET", "quiet; no completion/crash/close signal"
	}
	cat, ok := categories[disp]
	if !ok {
		cat = category{"HANGING", "unknown"}
	}

	autonomous, supervised := false, false
	var head []map[string]any
	for _, hl := range readHead(path, headRecords) {
		ho := parseRecord(hl)
		if ho == nil {
			continue
		}
		head = append(head, ho)
		htxt := headText(ho)
		if autonRE.MatchString(htxt) {
			autonomous = true
		}
		if supervRE.MatchString(htxt) {
			supervised, autonomous = true, true
		}
	}
	project := filepath.Base(filepath.Dir(path))
	sig := ""
	if autonomous || disp == "LIVE" || disp == "DONE" {
		cwdText := ""
		if truthy(cwd) {
			cwdText = pyStr(cwd)
		}
		sig = taskSig(project, cwdText, firstInstruction(head))
	}
	r := &Row{
		Disp: disp, Category: cat.category, Cause: cat.cause, Reason: reason,
		LastKind: lastKind, AgeMin: pyRound(age, 1),
		SeenUTC: isoUTC(mtime), LastTS: lastTS,
		ThrottleSeen: throttle, ThrottleCurrent: throttleCurrent,
		PendingTool: pending,
		HTTPStatus:  httpStatus(lt),
		InitiatedBy: "user",
		Cwd:         cwd, Git: git,
		Last:       strings.ReplaceAll(runeHead(lt, 200), "\n", " "),
		Path:       path,
		Autonomous: autonomous, Supervised: supervised,
		TaskSig: sig, Records: st.Size(),
	}
	if throttleCurrent {
		r.ThrottleReset, r.ThrottleWeekly = throttle, throttleWeekly
	}
	if autonomous {
		r.InitiatedBy = "agent"
	}
	if truthy(sid) {
		r.Session = pyStr(sid)
	} else {
		r.Session = strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	}
	return r
}
