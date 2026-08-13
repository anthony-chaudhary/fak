package watchdoghealth

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// auditgate.go — the GATE the n2/n3 meta-watchdog audit never had.
//
// `tools/watchdog_watchdog_audit.ps1` already produces a real verdict (GREEN/AMBER/RED plus
// a reason list) and already exits 0/2/3 on it. What was missing is everything downstream:
// the scheduled wrapper captured the audit's stdout, appended it to a retained log, and then
// `exit 0`. Task Scheduler therefore recorded LastTaskResult=0 every 15 minutes while the
// retained log accumulated 47 [RED] and 51 [AMBER] findings — three DOWN tasks, unproven
// resumes, a backlog that would not drain, and the auditor's own warning that its Interactive
// logon will be refused under RDP. That is telemetry without a gate: nothing about a red
// finding altered health, and the same warning was appended forever without ever aging into a
// decision.
//
// This leaf is the pure half of the fix, and it holds four invariants the shell cannot be
// trusted to hold on its own:
//
//   - A verdict maps to a TYPED exit code and only GREEN maps to 0 (AuditExitCode). RED is 3,
//     AMBER is a distinct 2, and an audit whose output could not be read at all is 4 — never
//     0, because "I could not read the audit" is not a pass.
//   - The verdict is ESCALATED by the findings, never merely echoed (AuditGate). A report that
//     claims GREEN while carrying a [RED] reason exits 3. The gate does not believe the header.
//   - Repeated findings are DEDUPLICATED into durable records with a first-seen, a recurrence
//     count, and an explicit resolution (FoldAuditLedger), so a standing fault reads as one
//     aging decision rather than N identical log lines.
//   - Every finding is routed to the owner who can actually clear it (AuditOwner), and a
//     finding no rule recognises is UNROUTED — surfaced to a person, never silently dropped.
//
// Everything with a clock or a side effect stays in the shell: running the audit, reading and
// writing the ledger file, and setting the process exit status. This core reads no clock and
// does no I/O — the caller passes nowUnix in — so the "a RED finding cannot yield scheduler
// result 0" claim is a table test, not a story about a PowerShell wrapper.

// AuditSchema is the gate result's self-describing version tag.
const AuditSchema = "fak.watchdog-audit-gate.v1"

// AuditVerdict is the closed verdict vocabulary of the meta-watchdog audit. UNREADABLE is not
// a verdict the audit itself emits — it is the gate's own token for "the audit produced output
// I could not parse", which must be distinguishable from a genuine pass.
type AuditVerdict string

const (
	// AuditGreen: the tower is ticking, nothing down, resumes proven. The ONLY verdict that
	// maps to exit 0.
	AuditGreen AuditVerdict = "GREEN"
	// AuditAmber: no hard-down task, but latent or unproven items are open.
	AuditAmber AuditVerdict = "AMBER"
	// AuditRed: at least one hard fault — a stall, a DOWN tower task, or a terminally unproven
	// resume. The supervision tower is not covering the fleet.
	AuditRed AuditVerdict = "RED"
	// AuditUnreadable: the gate could not extract a verdict or a single finding from the
	// audit's output. Treated as its own failure class: a wrapper that swallowed the audit's
	// stdout looks exactly like this, and it must not read as GREEN.
	AuditUnreadable AuditVerdict = "UNREADABLE"
)

// auditRank orders verdicts for the worst-of escalation. RED dominates everything; UNREADABLE
// sits above AMBER (an unparseable audit is worse than a known-latent one) but below RED (a
// concrete red finding is a stronger fact than a parse failure).
func auditRank(v AuditVerdict) int {
	switch v {
	case AuditGreen:
		return 0
	case AuditAmber:
		return 1
	case AuditUnreadable:
		return 2
	case AuditRed:
		return 3
	default:
		return 2 // an unrecognized token is unreadable, never green
	}
}

// Audit gate exit codes. These are the process statuses the scheduled wrapper must propagate
// so Task Scheduler's LastTaskResult carries the verdict instead of a hard-coded 0. They match
// the audit script's own 0/2/3 contract and add 4 for the gate's UNREADABLE class.
const (
	AuditExitGreen      = 0
	AuditExitAmber      = 2
	AuditExitRed        = 3
	AuditExitUnreadable = 4
)

// AuditExitCode maps a verdict to the process status the scheduler must see. The default arm
// is deliberately UNREADABLE's nonzero code, not 0: an unknown verdict token is a gate that
// does not understand its input, and the whole defect this leaf closes is a non-green audit
// reporting success. Only an exact GREEN yields 0.
func AuditExitCode(v AuditVerdict) int {
	switch v {
	case AuditGreen:
		return AuditExitGreen
	case AuditAmber:
		return AuditExitAmber
	case AuditRed:
		return AuditExitRed
	default:
		return AuditExitUnreadable
	}
}

// ParseAuditVerdict maps a raw verdict token (the audit's `verdict` field, or the word after
// "VERDICT:" in its human output) into the closed vocabulary. Anything it cannot recognise —
// including the empty string — is UNREADABLE, so a missing or garbled verdict can never be
// mistaken for a pass.
func ParseAuditVerdict(token string) AuditVerdict {
	switch strings.ToUpper(strings.TrimSpace(token)) {
	case "GREEN", "OK", "PASS":
		return AuditGreen
	case "AMBER", "WARN", "YELLOW":
		return AuditAmber
	case "RED", "FAIL", "CRITICAL":
		return AuditRed
	default:
		return AuditUnreadable
	}
}

// WorstAuditVerdict returns the more severe of two verdicts. Used to escalate a reported
// header verdict with the severities actually present in the reason list.
func WorstAuditVerdict(a, b AuditVerdict) AuditVerdict {
	if auditRank(b) > auditRank(a) {
		return b
	}
	return a
}

// AuditOwner is the closed vocabulary of who can actually clear a finding — the "actionable
// owner" half of the report. It is deliberately not a free-text name: the only question that
// matters at the gate is whether some autonomous actor already clears this on its next tick,
// or whether the finding will sit open until a person acts.
type AuditOwner string

const (
	// AuditOwnerFleet: a running actor clears it (the autoheal restarts the monitor, the next
	// watchdog tick rewrites the ledger, the drain works the backlog). No page.
	AuditOwnerFleet AuditOwner = "fleet"
	// AuditOwnerOperator: it needs elevation or a priority call no automation holds — the S4U
	// migration, a re-registration, scheduling the auditor orthogonally.
	AuditOwnerOperator AuditOwner = "operator"
	// AuditOwnerUnrouted: no routing rule recognised this finding. It waits on a person by
	// construction — the conservative fail-toward-paging rung the sibling folds also take, so a
	// finding the table has not learned yet is surfaced rather than silently absorbed.
	AuditOwnerUnrouted AuditOwner = "unrouted"
)

// NeedsHuman reports whether an open finding with this owner waits on a person. Only the
// fleet's own findings do not.
func (o AuditOwner) NeedsHuman() bool { return o != AuditOwnerFleet }

// auditRoute is one ownership rule: any of Match (matched case-insensitively against the
// finding text) routes the finding to Owner with the named next move.
type auditRoute struct {
	Match  []string
	Owner  AuditOwner
	Action string
}

// auditRoutes routes a finding to the actor that can clear it. Order is precedence: the first
// rule whose Match hits wins, so the S4U/LogonType rule is first (it is the diagnosis behind
// most DOWN findings on this box, and it is the one that genuinely needs elevation) and the
// generic DOWN rule sits after it. Every string here is a substring of a reason the audit
// script actually emits; a finding matching none of them is UNROUTED on purpose.
var auditRoutes = []auditRoute{
	{
		Match: []string{"0x800710e0", "logontype=interactive", "migrate to s4u", "latent 0x800710e0"},
		Owner: AuditOwnerOperator,
		Action: "elevated: tools\\migrate_fleet_tasks_to_s4u.ps1 -Apply -VerifyRun " +
			"(setting an S4U principal is denied unelevated)",
	},
	{
		Match: []string{"n3 gap", "auditor", "watchdogaudit"},
		Owner: AuditOwnerOperator,
		Action: "schedule this audit S4U, or host it on an orthogonal /loop — an auditor that " +
			"shares the failure mode it detects cannot report that failure",
	},
	{
		Match:  []string{" down "},
		Owner:  AuditOwnerOperator,
		Action: "re-register the task from tools/register_*.ps1 and force-run it until LastTaskResult is 0x0",
	},
	{
		Match:  []string{"stall:", "not ticking", "no resume ledger"},
		Owner:  AuditOwnerFleet,
		Action: "`fak watchdog heal` restarts the resume watchdog; the next tick rewrites the ledger",
	},
	{
		Match:  []string{"launched_unproven", "unproven"},
		Owner:  AuditOwnerFleet,
		Action: "one live tick re-revives the unproven sessions: tools/fleet_resume_watchdog.ps1 -Live",
	},
	{
		Match: []string{"backlog"},
		Owner: AuditOwnerFleet,
		Action: "raise recovery capacity (MaxPerTick) — a backlog that outlives the throttle " +
			"reset means capacity, not account pressure, is the limiter",
	},
	{
		Match:  []string{"fak not on path", "could not read fak resume watchdog"},
		Owner:  AuditOwnerOperator,
		Action: "put fak on the task's PATH so the launched-vs-proven witness can be read",
	},
	{
		// The catch-all, last by construction: strings.Contains(x, "") is always true, so a
		// finding none of the rules above recognised lands here as UNROUTED and waits on a
		// person. Spelling it out as a rule (rather than as an inline fallback) keeps this table
		// the single exhaustive statement of every owner a finding can be routed to.
		Match: []string{""},
		Owner: AuditOwnerUnrouted,
	},
}

// routeAuditFinding returns the owner and next move for a finding's text. The table's last
// rule is a catch-all, so the trailing return is only a guard against an emptied table.
func routeAuditFinding(text string) (AuditOwner, string) {
	low := " " + strings.ToLower(text) + " "
	for _, r := range auditRoutes {
		for _, m := range r.Match {
			if strings.Contains(low, m) {
				return r.Owner, r.Action
			}
		}
	}
	return AuditOwnerUnrouted, ""
}

// AuditFinding is one reason line from the audit — a severity, the human text, the recurrence
// key that makes "the same finding again" decidable, and the owner routing.
type AuditFinding struct {
	Severity AuditVerdict `json:"severity"`
	Text     string       `json:"text"`
	// Key is the recurrence fingerprint: the text with every purely numeric token collapsed, so
	// "newest ledger write 37 min ago" and "... 52 min ago" are ONE standing finding rather than
	// two. See AuditFindingKey.
	Key    string     `json:"key"`
	Owner  AuditOwner `json:"owner"`
	Action string     `json:"action,omitempty"`
}

// NewAuditFinding builds a finding from a severity and a raw reason line, computing the
// recurrence key and the ownership routing. A leading "[RED] " / "[AMBER] " marker is stripped
// (and, when present, overrides the passed severity — the marker is the audit's own word).
func NewAuditFinding(sev AuditVerdict, text string) AuditFinding {
	text = strings.TrimSpace(text)
	if marked, rest, ok := splitAuditMarker(text); ok {
		sev, text = marked, rest
	}
	owner, action := routeAuditFinding(text)
	return AuditFinding{
		Severity: ParseAuditVerdict(string(sev)),
		Text:     text,
		Key:      AuditFindingKey(text),
		Owner:    owner,
		Action:   action,
	}
}

// splitAuditMarker peels a leading "[RED] " / "[AMBER] " / "[GREEN] " marker off a reason line,
// returning the severity it names and the remaining text.
func splitAuditMarker(line string) (AuditVerdict, string, bool) {
	line = strings.TrimSpace(line)
	if !strings.HasPrefix(line, "[") {
		return "", line, false
	}
	end := strings.Index(line, "]")
	if end < 0 {
		return "", line, false
	}
	v := ParseAuditVerdict(line[1:end])
	if v == AuditUnreadable {
		return "", line, false // "[2026-08-11T...]" and other bracketed prefixes are not markers
	}
	return v, strings.TrimSpace(line[end+1:]), true
}

// AuditFindingKey is the recurrence fingerprint for a reason line: lowercased, whitespace
// normalized, with every token that is PURELY a number collapsed to "#". Collapsing only fully
// numeric tokens is what keeps the key both stable and specific — the minute counts and depths
// that change on every tick fall out, while the tokens that carry identity (a task name like
// UserSeatDrain-1010, an HRESULT like 0x800710E0) survive because they contain non-digits.
func AuditFindingKey(text string) string {
	if _, rest, ok := splitAuditMarker(text); ok {
		text = rest
	}
	fields := strings.Fields(strings.ToLower(text))
	for i, f := range fields {
		fields[i] = normalizeAuditToken(f)
	}
	return strings.Join(fields, " ")
}

// normalizeAuditToken collapses a token made only of digits and numeric punctuation to "#".
// Any letter or other symbol in the token means it identifies something, so it is kept verbatim.
func normalizeAuditToken(tok string) string {
	digit := false
	for _, r := range tok {
		switch {
		case r >= '0' && r <= '9':
			digit = true
		case r == '.' || r == ',' || r == '%':
			// numeric punctuation rides along
		default:
			return tok
		}
	}
	if !digit {
		return tok
	}
	return "#"
}

// ParseAuditFindings turns reason lines into findings, keeping only the lines that carry a
// severity marker. Lines without one (the audit's banner, table rows, the ACTION footer) are
// not findings and are dropped.
func ParseAuditFindings(lines []string) []AuditFinding {
	out := make([]AuditFinding, 0, len(lines))
	for _, ln := range lines {
		ln = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(ln), "- "))
		sev, text, ok := splitAuditMarker(ln)
		if !ok || text == "" {
			continue
		}
		out = append(out, NewAuditFinding(sev, text))
	}
	return out
}

// auditReport is the shape of the audit script's `-Json` object that this gate reads. Every
// other field it emits is passthrough telemetry the gate does not decide on.
type auditReport struct {
	Verdict string   `json:"verdict"`
	Reasons []string `json:"reasons"`
}

// ParseAuditReport extracts the verdict and the findings from whatever the audit produced —
// the `-Json` object, or the retained human log when the JSON was never captured. It is
// deliberately tolerant in both directions and conservative in one: raw bytes it cannot read
// at all yield UNREADABLE and no findings, which AuditExitCode turns into a nonzero status.
//
// The text path matters as much as the JSON path: the evidence for this defect was a retained
// 500-line human log, and a gate that can only read the JSON would have scored that log GREEN.
func ParseAuditReport(raw []byte) (AuditVerdict, []AuditFinding) {
	var rep auditReport
	if err := json.Unmarshal(raw, &rep); err == nil && (rep.Verdict != "" || len(rep.Reasons) > 0) {
		return ParseAuditVerdict(rep.Verdict), ParseAuditFindings(rep.Reasons)
	}
	verdict := AuditUnreadable
	lines := strings.Split(strings.ReplaceAll(string(raw), "\r\n", "\n"), "\n")
	for _, ln := range lines {
		if _, rest, ok := strings.Cut(strings.TrimSpace(ln), "VERDICT:"); ok {
			verdict = ParseAuditVerdict(rest)
		}
	}
	findings := ParseAuditFindings(lines)
	if verdict == AuditUnreadable && len(findings) > 0 {
		// No header, but real findings: let the findings alone decide the verdict rather than
		// reporting UNREADABLE over evidence the gate could in fact read.
		verdict = AuditGreen
	}
	return verdict, findings
}

// AuditRecord is one DEDUPLICATED finding's durable history: what it is, who owns it, when it
// was first and last seen, how many audit runs have carried it, whether it has ever come back
// after clearing, and whether it is currently resolved. This is the record that replaces
// appending the same warning to a log forever — one row per distinct fault, aging.
type AuditRecord struct {
	Key      string       `json:"key"`
	Severity AuditVerdict `json:"severity"`
	// Text is the most recent phrasing of the finding (the numbers move; the key does not).
	Text   string     `json:"text"`
	Owner  AuditOwner `json:"owner"`
	Action string     `json:"action,omitempty"`

	FirstSeenUnix int64 `json:"first_seen_unix"`
	LastSeenUnix  int64 `json:"last_seen_unix"`
	// Occurrences is the number of audit runs that carried this finding.
	Occurrences int `json:"occurrences"`
	// Regressions is the number of times the finding came back AFTER being marked resolved —
	// the signal that a remedy did not hold.
	Regressions int `json:"regressions,omitempty"`
	// Resolved is set when an audit run no longer carries the finding; the record is kept (not
	// deleted) so a resolution is reportable and a later regression is detectable.
	Resolved     bool  `json:"resolved,omitempty"`
	ResolvedUnix int64 `json:"resolved_unix,omitempty"`
}

// AgeSeconds is how long this finding has been known, measured from its first sighting. A
// resolved record's age freezes at its resolution.
func (r AuditRecord) AgeSeconds(nowUnix int64) int64 {
	end := nowUnix
	if r.Resolved && r.ResolvedUnix > 0 {
		end = r.ResolvedUnix
	}
	if r.FirstSeenUnix <= 0 || end < r.FirstSeenUnix {
		return 0
	}
	return end - r.FirstSeenUnix
}

// Recurring reports whether more than one audit run has carried this finding — the difference
// between a blip and a standing fault.
func (r AuditRecord) Recurring() bool { return r.Occurrences > 1 }

// FoldAuditLedger folds this run's findings into the prior ledger. Pure and total: it never
// reads a clock (the caller supplies nowUnix), never drops a record, and is idempotent in the
// only sense that matters — a finding repeated within ONE run counts once.
//
// A finding already in the ledger bumps its occurrence count and last-seen (and clears a stale
// resolution, counting a regression); a finding not in the ledger is appended as new; and a
// ledger record this run did NOT carry is marked resolved rather than removed, so the report
// can say a fault cleared and a later run can say it came back.
func FoldAuditLedger(prev []AuditRecord, findings []AuditFinding, nowUnix int64) []AuditRecord {
	out := make([]AuditRecord, len(prev))
	copy(out, prev)
	idx := make(map[string]int, len(out))
	for i := range out {
		if _, dup := idx[out[i].Key]; !dup {
			idx[out[i].Key] = i
		}
	}
	seen := make(map[string]bool, len(findings))
	for _, f := range findings {
		if f.Key == "" || seen[f.Key] {
			continue
		}
		seen[f.Key] = true
		i, ok := idx[f.Key]
		if !ok {
			idx[f.Key] = len(out)
			out = append(out, AuditRecord{
				Key: f.Key, Severity: f.Severity, Text: f.Text, Owner: f.Owner, Action: f.Action,
				FirstSeenUnix: nowUnix, LastSeenUnix: nowUnix, Occurrences: 1,
			})
			continue
		}
		r := out[i]
		r.Severity, r.Text, r.Owner, r.Action = f.Severity, f.Text, f.Owner, f.Action
		r.LastSeenUnix = nowUnix
		r.Occurrences++
		if r.Resolved {
			r.Resolved, r.ResolvedUnix = false, 0
			r.Regressions++
		}
		if r.FirstSeenUnix <= 0 || nowUnix < r.FirstSeenUnix {
			r.FirstSeenUnix = nowUnix
		}
		out[i] = r
	}
	for i := range out {
		if seen[out[i].Key] || out[i].Resolved {
			continue
		}
		out[i].Resolved, out[i].ResolvedUnix = true, nowUnix
	}
	return out
}

// AuditGateResult is the whole gate decision for one audit run: the escalated verdict, the
// exit code the scheduler must record, the deduplicated ledger to persist, and the keys split
// by what this run changed.
type AuditGateResult struct {
	Schema  string       `json:"schema"`
	Verdict AuditVerdict `json:"verdict"`
	// ExitCode is the process status the wrapper must exit with. It is AuditExitCode(Verdict);
	// it is carried on the result so a caller cannot forget to derive it.
	ExitCode int   `json:"exit_code"`
	NowUnix  int64 `json:"now_unix,omitempty"`

	Ledger []AuditRecord `json:"ledger,omitempty"`
	// New / Recurring / Resolved are the keys first seen this run, carried again this run, and
	// cleared this run. NeedsHuman is the open subset whose owner is not the fleet.
	New        []string `json:"new,omitempty"`
	Recurring  []string `json:"recurring,omitempty"`
	Resolved   []string `json:"resolved,omitempty"`
	NeedsHuman []string `json:"needs_human,omitempty"`
}

// AuditGate is the whole decision. It normalizes the reported verdict through the closed
// vocabulary, ESCALATES it with the severity of every finding (so a header that claims GREEN
// over a [RED] reason still exits 3), folds the findings into the durable ledger, and reports
// which keys are new, recurring, resolved, and waiting on a person.
func AuditGate(reported AuditVerdict, findings []AuditFinding, prev []AuditRecord, nowUnix int64) AuditGateResult {
	v := ParseAuditVerdict(string(reported))
	for _, f := range findings {
		v = WorstAuditVerdict(v, f.Severity)
	}
	ledger := FoldAuditLedger(prev, findings, nowUnix)
	res := AuditGateResult{
		Schema:   AuditSchema,
		Verdict:  v,
		ExitCode: AuditExitCode(v),
		NowUnix:  nowUnix,
		Ledger:   ledger,
	}
	for _, r := range ledger {
		switch {
		case r.Resolved:
			if r.ResolvedUnix == nowUnix {
				res.Resolved = append(res.Resolved, r.Key)
			}
			continue
		case r.Occurrences <= 1 && r.FirstSeenUnix == nowUnix:
			res.New = append(res.New, r.Key)
		default:
			res.Recurring = append(res.Recurring, r.Key)
		}
		if r.Owner.NeedsHuman() {
			res.NeedsHuman = append(res.NeedsHuman, r.Key)
		}
	}
	return res
}

// AuditResultSwallowed reports whether an observed scheduler result contradicts the gate — the
// exact defect this leaf closes. A wrapper that runs the audit and then exits 0 regardless
// makes Task Scheduler record success over a RED verdict; feeding the recorded result back
// through this predicate names that as a fault instead of letting it pass as health.
func AuditResultSwallowed(g AuditGateResult, observedSchedulerResult int) bool {
	return g.ExitCode != AuditExitGreen && observedSchedulerResult == AuditExitGreen
}

// AuditorIndependenceFinding folds the n3 question — "is this auditor orthogonal to the
// failure mode it checks?" — into a finding. An auditor scheduled with an Interactive logon is
// RED, not a note: on an RDP/headless box that principal is refused (0x800710E0), so the
// auditor dies exactly the way the watchdogs it audits died, and its silence then reads as
// health. An auditor that is not scheduled at all is AMBER (it runs only when a human
// remembers). An S4U task — or an orthogonal loop, passed as scheduled=false with an empty
// logon type only when no task exists — is clean, and the second return is false.
func AuditorIndependenceFinding(scheduled bool, logonType string) (AuditFinding, bool) {
	if !scheduled {
		return NewAuditFinding(AuditAmber, "n3 GAP: this audit is not itself scheduled or looped -- "+
			"it only runs when a human remembers, and a dead auditor over a dead watchdog is a "+
			"silent double-fault"), true
	}
	if strings.EqualFold(strings.TrimSpace(logonType), "interactive") ||
		strings.EqualFold(strings.TrimSpace(logonType), "interactivetoken") {
		return NewAuditFinding(AuditRed, "n3: the auditor's own task is LogonType=Interactive -- "+
			"it will be refused (0x800710E0) the SAME way the tasks it audits were, so its silence "+
			"cannot be read as health"), true
	}
	return AuditFinding{}, false
}

// AuditReportLines renders the gate's ledger as one line per open or just-resolved finding,
// carrying the four facts the audit log never reported: age, recurrence, resolution, and the
// owner who can clear it. Deterministic and clock-free (the caller supplies nowUnix), so the
// shell only has to print what it is handed.
func AuditReportLines(g AuditGateResult, nowUnix int64) []string {
	out := make([]string, 0, len(g.Ledger))
	for _, r := range g.Ledger {
		if r.Resolved && r.ResolvedUnix != nowUnix {
			continue // already reported as resolved on an earlier run
		}
		state := string(r.Severity)
		if r.Resolved {
			state = "RESOLVED"
		}
		line := state + " x" + strconv.Itoa(r.Occurrences) +
			" age=" + auditAge(r.AgeSeconds(nowUnix)) +
			" owner=" + string(r.Owner)
		if r.Regressions > 0 {
			line += " regressions=" + strconv.Itoa(r.Regressions)
		}
		line += "  " + r.Text
		if !r.Resolved && r.Action != "" {
			line += "  -> " + r.Action
		}
		out = append(out, line)
	}
	return out
}

// auditAge renders a finding's age as a compact duration ("0s" for a first sighting).
func auditAge(seconds int64) string {
	return (time.Duration(seconds) * time.Second).String()
}

// AuditGateSelfcheck is the deterministic, no-I/O proof of the gate, the audit analogue of
// TriageSelfcheck: only GREEN maps to exit 0, a RED finding under a GREEN header still exits
// 3, AMBER is a distinct nonzero status, an unreadable report is nonzero, a repeated finding
// deduplicates into one aging record, a cleared finding is reported resolved, and a scheduler
// result of 0 over a nonzero gate is named as a swallow.
func AuditGateSelfcheck() error {
	for _, v := range []AuditVerdict{AuditAmber, AuditRed, AuditUnreadable, AuditVerdict("bogus")} {
		if AuditExitCode(v) == AuditExitGreen {
			return fmt.Errorf("verdict %q must not map to exit 0", v)
		}
	}
	if AuditExitCode(AuditGreen) != AuditExitGreen {
		return fmt.Errorf("GREEN must map to exit 0, got %d", AuditExitCode(AuditGreen))
	}
	if AuditExitCode(AuditAmber) == AuditExitCode(AuditRed) {
		return fmt.Errorf("AMBER must carry a status distinct from RED")
	}

	// A header that claims GREEN over a [RED] reason escalates: the gate does not believe the
	// header, and the scheduler records 3.
	lying := []byte(`{"verdict":"GREEN","reasons":["[RED] FleetResumeWatchdog DOWN 0x800710E0 (LogonType=Interactive) -- migrate to S4U"]}`)
	v, findings := ParseAuditReport(lying)
	g := AuditGate(v, findings, nil, 1_000)
	if g.Verdict != AuditRed || g.ExitCode != AuditExitRed {
		return fmt.Errorf("a RED reason under a GREEN header must escalate to RED/3, got %s/%d", g.Verdict, g.ExitCode)
	}
	if !AuditResultSwallowed(g, 0) {
		return fmt.Errorf("a scheduler result of 0 over a RED gate must be reported as swallowed")
	}
	if len(g.NeedsHuman) != 1 || g.Ledger[0].Owner != AuditOwnerOperator {
		return fmt.Errorf("a 0x800710E0 DOWN finding must be owned by the operator, got %v", g.Ledger)
	}

	// The same standing fault, re-observed 15 minutes later with a different minute count,
	// deduplicates into ONE record that ages -- instead of a second identical log line.
	stall := func(min string) []AuditFinding {
		return ParseAuditFindings([]string{"[RED] STALL: newest ledger write " + min + " min ago (> 15 min) -- watchdog is not ticking"})
	}
	first := AuditGate(AuditRed, stall("37"), nil, 1_000)
	second := AuditGate(AuditRed, stall("52"), first.Ledger, 1_900)
	if len(second.Ledger) != 1 {
		return fmt.Errorf("a re-observed stall must dedupe into one record, got %d", len(second.Ledger))
	}
	if r := second.Ledger[0]; !r.Recurring() || r.Occurrences != 2 || r.AgeSeconds(1_900) != 900 {
		return fmt.Errorf("a re-observed stall must age and count: %+v", r)
	}
	if len(second.Recurring) != 1 || len(second.New) != 0 {
		return fmt.Errorf("the second sighting must be recurring, not new: %+v", second)
	}

	// The fault clears: the record is kept and reported resolved, and the gate goes green.
	third := AuditGate(AuditGreen, nil, second.Ledger, 2_800)
	if third.ExitCode != AuditExitGreen || len(third.Resolved) != 1 || !third.Ledger[0].Resolved {
		return fmt.Errorf("a cleared fault must be reported resolved and exit 0, got %+v", third)
	}
	// And if it comes back, the regression is counted rather than filed as a brand-new finding.
	fourth := AuditGate(AuditRed, stall("61"), third.Ledger, 3_700)
	if r := fourth.Ledger[0]; r.Regressions != 1 || r.Resolved || r.Occurrences != 3 {
		return fmt.Errorf("a returning fault must count a regression: %+v", r)
	}

	// An auditor scheduled with its own Interactive logon is RED: it dies the way the tasks it
	// audits died, so its silence is not health.
	f, ok := AuditorIndependenceFinding(true, "Interactive")
	if !ok || f.Severity != AuditRed || f.Owner != AuditOwnerOperator {
		return fmt.Errorf("an Interactive-logon auditor must be a RED operator finding, got %+v", f)
	}
	if _, ok := AuditorIndependenceFinding(true, "S4U"); ok {
		return fmt.Errorf("an S4U auditor is orthogonal to the failure mode and must not be a finding")
	}

	// An empty / swallowed report is UNREADABLE, never a pass.
	if uv, uf := ParseAuditReport(nil); uv != AuditUnreadable || len(uf) != 0 {
		return fmt.Errorf("an empty audit report must be UNREADABLE, got %s/%d", uv, len(uf))
	}
	return nil
}
