package sessionaudit

// Behavioral lens (#2365/#2375, ported from tools/session_audit.py BehaviorLens):
// per-transcript stuck/churn detectors computed off the same Analyze() walk.
// All detectors read ONLY what the transcript already carries (tool_use inputs +
// errored tool_results); none re-run anything. The emitted Behavior mirrors the
// Python `behavior` dict field-for-field so `fak session-audit deep --json` and
// `python tools/session_audit.py deep` agree on a shared transcript.

import (
	"encoding/json"
	"math"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

const (
	repeatFailureMin = int64(3) // same tool+args+error this often is a stuck retry loop
	fileChurnMin     = int64(5) // a file mutated this often may be a rewrite/flip-flop loop
	stallGapS        = 300.0    // a gap this long with zero records is a harness/API stall
	successLoopMin   = int64(8) // this many identical SUCCESSFUL calls is a poll loop / storm
)

var (
	shellTools       = map[string]bool{"Bash": true, "PowerShell": true}
	mutationTools    = map[string]bool{"Edit": true, "Write": true, "NotebookEdit": true}
	successLoopTools = map[string]bool{"Read": true, "Glob": true, "Grep": true, "LS": true, "Bash": true, "PowerShell": true}

	editChurnSignatures = []struct{ key, sig string }{
		{"not_read", "File has not been read yet"},
		{"stale_read", "has been modified since read"},
	}

	timeoutKillRE = regexp.MustCompile(`(?i)exit (?:code|status)\W{0,3}143\b|timed out`)
	sleepPollRE   = regexp.MustCompile(`(?i)^\s*(?:sleep|start-sleep)\b`)
	restartMarker = regexp.MustCompile(`(?i)session is being continued from a previous|<command-name>\s*/(?:resume|clear)\b|^\s*resume\s*$`)
)

// Behavior mirrors the Python BehaviorLens.summary() dict plus the stall fields.
type Behavior struct {
	ToolErrors       map[string]int64   `json:"tool_errors"`
	TimeoutKills     int64              `json:"timeout_kills"`
	SleepPolls       int64              `json:"sleep_polls"`
	EditChurn        map[string]int64   `json:"edit_churn"`
	NotReadClasses   map[string]int64   `json:"not_read_classes"`
	RepeatFailures   []RepeatFailureRow `json:"repeat_failures"`
	MaxRepeatFailure int64              `json:"max_repeat_failure"`
	FailureMass      []RepeatFailureRow `json:"failure_mass"`
	MaxFailureMass   int64              `json:"max_failure_mass"`
	FileChurn        []FileChurnRow     `json:"file_churn"`
	MaxFileChurn     int64              `json:"max_file_churn"`
	SuccessLoops     []SuccessLoopRow   `json:"success_loops"`
	MaxSuccessLoop   int64              `json:"max_success_loop"`
	StallGaps        int64              `json:"stall_gaps"`
	MaxGapS          float64            `json:"max_gap_s"`
}

// RepeatFailureRow is one verbatim-repeat / failure-mass offender.
type RepeatFailureRow struct {
	Tool  string `json:"tool"`
	Sig   string `json:"sig"`
	Count int64  `json:"count"`
}

// FileChurnRow is one per-file mutation-churn offender.
type FileChurnRow struct {
	File            string `json:"file"`
	Count           int64  `json:"count"`
	DistinctRegions int64  `json:"distinct_regions"`
	Reverts         int64  `json:"reverts"`
}

// SuccessLoopRow is one successful-identical-call loop (read/glob storm).
type SuccessLoopRow struct {
	Tool   string `json:"tool"`
	Target string `json:"target"`
	Count  int64  `json:"count"`
}

type vKey struct{ tool, ak, sig string }
type mKey struct{ tool, sig string }
type cKey struct{ tool, ak string }
type region struct{ oldH, newH string }

// behaviorLens accumulates the stuck/churn signals across one transcript walk.
// It is fed one tool_use / tool_result at a time (post de-dup), the same order
// the Python lens sees them.
type behaviorLens struct {
	errors         map[string]int64
	timeoutKills   int64
	sleepPolls     int64
	editChurn      map[string]int64
	notReadClasses map[string]int64

	verbatim      map[vKey]int64
	verbatimOrder []vKey
	mass          map[mKey]int64
	massOrder     []mKey

	fileWrites  map[string]int64
	fileOrder   []string
	fileRegions map[string][]region

	callSigs     map[cKey]int64
	callOrder    []cKey
	callLabels   map[cKey]string
	errSigCounts map[cKey]int64

	readPaths    map[string]bool
	mutatedPaths map[string]bool
	sawRestart   bool

	// tool_use id -> attribution, so a later tool_result keys on its call.
	tuName    map[string]string
	tuArgs    map[string]string
	tuHasArgs map[string]bool
	tuPath    map[string]string

	havePrev  bool
	prevDT    time.Time
	stallGaps int64
	maxGapS   float64
}

func newBehaviorLens() *behaviorLens {
	return &behaviorLens{
		errors:         map[string]int64{},
		editChurn:      map[string]int64{},
		notReadClasses: map[string]int64{},
		verbatim:       map[vKey]int64{},
		mass:           map[mKey]int64{},
		fileWrites:     map[string]int64{},
		fileRegions:    map[string][]region{},
		callSigs:       map[cKey]int64{},
		callLabels:     map[cKey]string{},
		errSigCounts:   map[cKey]int64{},
		readPaths:      map[string]bool{},
		mutatedPaths:   map[string]bool{},
		tuName:         map[string]string{},
		tuArgs:         map[string]string{},
		tuHasArgs:      map[string]bool{},
		tuPath:         map[string]string{},
	}
}

// observeRecord runs once per transcript record (any type): it notes a restart /
// compaction marker and accumulates the inter-record stall gap.
func (l *behaviorLens) observeRecord(r transcriptRecord) {
	if isRestartRecord(r) {
		l.sawRestart = true
	}
	if r.Timestamp == "" {
		return
	}
	dt, err := parseTimestamp(r.Timestamp)
	if err != nil {
		return
	}
	if l.havePrev {
		gap := dt.Sub(l.prevDT).Seconds()
		if gap > l.maxGapS {
			l.maxGapS = gap
		}
		if gap >= stallGapS {
			l.stallGaps++
		}
	}
	l.prevDT = dt
	l.havePrev = true
}

// noteToolUse records a tool_use's identity (name/args/path) and feeds the
// sleep-poll, file-churn and successful-call-loop tallies.
func (l *behaviorLens) noteToolUse(id, name string, input json.RawMessage, ak string) {
	if id != "" {
		l.tuName[id] = name
		l.tuArgs[id] = ak
		l.tuHasArgs[id] = true
		l.tuPath[id] = toolPath(input)
	}
	m := rawObj(input)
	if shellTools[name] && !boolField(m, "run_in_background") && sleepPollRE.MatchString(strField(m, "command")) {
		l.sleepPolls++
	}
	if mutationTools[name] {
		if path := firstStr(m, "file_path", "notebook_path"); path != "" {
			l.bumpFile(path)
			oldH := normHead(strField(m, "old_string"), 200)
			newRaw, ok := strFieldOK(m, "new_string")
			if !ok {
				newRaw = strField(m, "content")
			}
			l.fileRegions[path] = append(l.fileRegions[path], region{oldH, normHead(newRaw, 200)})
		}
	}
	if successLoopTools[name] {
		k := cKey{name, ak}
		if _, seen := l.callSigs[k]; !seen {
			l.callOrder = append(l.callOrder, k)
		}
		l.callSigs[k]++
		l.callLabels[k] = toolLabel(input)
	}
}

func (l *behaviorLens) bumpFile(path string) {
	if _, seen := l.fileWrites[path]; !seen {
		l.fileOrder = append(l.fileOrder, path)
	}
	l.fileWrites[path]++
}

// noteToolResult attributes a tool_result back to its tool_use and folds the
// error signals (repeat failures, failure mass, edit churn, timeout kills).
func (l *behaviorLens) noteToolResult(toolUseID string, isError bool, text string) {
	tool := l.tuName[toolUseID]
	if tool == "" {
		tool = "?"
	}
	ak, hasAK := l.tuArgs[toolUseID], l.tuHasArgs[toolUseID]
	path := l.tuPath[toolUseID]

	if !isError {
		if path != "" {
			if tool == "Read" {
				l.readPaths[path] = true
			} else if mutationTools[tool] {
				l.mutatedPaths[path] = true
			}
		}
		return
	}
	l.errors[tool]++
	if successLoopTools[tool] && hasAK {
		l.errSigCounts[cKey{tool, ak}]++
	}
	if shellTools[tool] && timeoutKillRE.MatchString(text) {
		l.timeoutKills++
	}
	for _, ec := range editChurnSignatures {
		if strings.Contains(text, ec.sig) {
			l.editChurn[ec.key]++
			if ec.key == "not_read" {
				l.notReadClasses[l.classifyNotRead(path)]++
			}
		}
	}
	if devindex.RepeatBenign(tool) {
		return
	}
	sig := normHead(text, 160)
	akSlot := ak
	if !hasAK {
		akSlot = "\x00none"
	}
	vk := vKey{tool, akSlot, sig}
	if _, seen := l.verbatim[vk]; !seen {
		l.verbatimOrder = append(l.verbatimOrder, vk)
	}
	l.verbatim[vk]++
	mk := mKey{tool, sig}
	if _, seen := l.mass[mk]; !seen {
		l.massOrder = append(l.massOrder, mk)
	}
	l.mass[mk]++
}

// classifyNotRead splits a not-read edit-churn failure. Precedence: a concrete
// prior write (self_duplicate) beats a prior read + restart (post_resume); a
// never-read edit is the real defect (true_never_read).
func (l *behaviorLens) classifyNotRead(path string) string {
	if path != "" && l.mutatedPaths[path] {
		return "self_duplicate"
	}
	if l.sawRestart && path != "" && l.readPaths[path] {
		return "post_resume"
	}
	return "true_never_read"
}

func (l *behaviorLens) summary() Behavior {
	b := Behavior{
		ToolErrors:     l.errors,
		TimeoutKills:   l.timeoutKills,
		SleepPolls:     l.sleepPolls,
		EditChurn:      l.editChurn,
		NotReadClasses: l.notReadClasses,
		RepeatFailures: []RepeatFailureRow{},
		FailureMass:    []RepeatFailureRow{},
		FileChurn:      []FileChurnRow{},
		SuccessLoops:   []SuccessLoopRow{},
		StallGaps:      l.stallGaps,
		MaxGapS:        round1(l.maxGapS),
	}

	for _, k := range l.verbatimOrder {
		n := l.verbatim[k]
		if n > b.MaxRepeatFailure {
			b.MaxRepeatFailure = n
		}
		if n >= repeatFailureMin {
			b.RepeatFailures = append(b.RepeatFailures, RepeatFailureRow{Tool: k.tool, Sig: k.sig, Count: n})
		}
	}
	sort.SliceStable(b.RepeatFailures, func(i, j int) bool { return b.RepeatFailures[i].Count > b.RepeatFailures[j].Count })
	b.RepeatFailures = topRepeat(b.RepeatFailures, 10)

	for _, k := range l.massOrder {
		n := l.mass[k]
		if n > b.MaxFailureMass {
			b.MaxFailureMass = n
		}
		if n >= repeatFailureMin {
			b.FailureMass = append(b.FailureMass, RepeatFailureRow{Tool: k.tool, Sig: k.sig, Count: n})
		}
	}
	sort.SliceStable(b.FailureMass, func(i, j int) bool { return b.FailureMass[i].Count > b.FailureMass[j].Count })
	b.FailureMass = topRepeat(b.FailureMass, 10)

	for _, f := range l.fileOrder {
		n := l.fileWrites[f]
		if n > b.MaxFileChurn {
			b.MaxFileChurn = n
		}
		if n < fileChurnMin {
			continue
		}
		regions := l.fileRegions[f]
		distinctOld := map[string]bool{}
		for _, rg := range regions {
			distinctOld[rg.oldH] = true
		}
		distinct := int64(len(distinctOld))
		if distinct == 0 {
			distinct = 1
		}
		seenOld := map[string]bool{}
		var reverts int64
		for _, rg := range regions {
			if seenOld[rg.newH] && rg.newH != rg.oldH {
				reverts++
			}
			seenOld[rg.oldH] = true
		}
		// Rewrite loop = edits keep revisiting the same few regions
		// (distinct*2 <= n), OR undo each other WHILE regions are being reused
		// (reverts and distinct < n). A single revert amid all-distinct regions
		// (distinct == n) is a long linear refactor that restores one earlier
		// snippet once — healthy build-out, not thrash. Mirrors the Python
		// _churn_rows gate (kills the b72e2808 false alarm; keeps 5c72b8ba).
		if distinct*2 <= n || (reverts >= 1 && distinct < n) {
			b.FileChurn = append(b.FileChurn, FileChurnRow{File: f, Count: n, DistinctRegions: distinct, Reverts: reverts})
		}
	}
	sort.SliceStable(b.FileChurn, func(i, j int) bool { return b.FileChurn[i].Count > b.FileChurn[j].Count })
	b.FileChurn = topChurn(b.FileChurn, 10)

	for _, k := range l.callOrder {
		if devindex.RepeatBenign(k.tool) {
			continue
		}
		succ := l.callSigs[k] - l.errSigCounts[k]
		if succ > b.MaxSuccessLoop {
			b.MaxSuccessLoop = succ
		}
		if succ >= successLoopMin {
			b.SuccessLoops = append(b.SuccessLoops, SuccessLoopRow{Tool: k.tool, Target: l.callLabels[k], Count: succ})
		}
	}
	sort.SliceStable(b.SuccessLoops, func(i, j int) bool { return b.SuccessLoops[i].Count > b.SuccessLoops[j].Count })
	b.SuccessLoops = topLoop(b.SuccessLoops, 10)

	return b
}

func topRepeat(rows []RepeatFailureRow, n int) []RepeatFailureRow {
	if len(rows) > n {
		return rows[:n]
	}
	return rows
}

func topChurn(rows []FileChurnRow, n int) []FileChurnRow {
	if len(rows) > n {
		return rows[:n]
	}
	return rows
}

func topLoop(rows []SuccessLoopRow, n int) []SuccessLoopRow {
	if len(rows) > n {
		return rows[:n]
	}
	return rows
}

// isRestartRecord reports a session-restart / compaction marker (post_resume
// signal): a compaction summary record, a /resume|/clear command, or a bare
// "Resume" continuation prompt.
func isRestartRecord(r transcriptRecord) bool {
	if r.IsCompactSummary || r.Type == "summary" {
		return true
	}
	if r.LastPrompt != "" && restartMarker.MatchString(r.LastPrompt) {
		return true
	}
	if r.Type == "user" {
		var c string
		if json.Unmarshal(r.Message.Content, &c) == nil && restartMarker.MatchString(c) {
			return true
		}
	}
	return false
}

// canonicalArgs is a stable grouping key for a tool_use input: canonical JSON
// with sorted object keys. Only used as a map key (never emitted), so it need
// only group identical inputs together the way the Python args digest does.
func canonicalArgs(raw json.RawMessage) string {
	if len(raw) == 0 {
		return "{}"
	}
	var v interface{}
	if json.Unmarshal(raw, &v) != nil {
		return string(raw)
	}
	b, err := json.Marshal(v)
	if err != nil {
		return string(raw)
	}
	return string(b)
}

func toolPath(raw json.RawMessage) string {
	return firstStr(rawObj(raw), "file_path", "notebook_path")
}

func toolLabel(raw json.RawMessage) string {
	return normHead(firstStr(rawObj(raw), "file_path", "notebook_path", "pattern", "path", "command"), 120)
}

func rawObj(raw json.RawMessage) map[string]json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	var m map[string]json.RawMessage
	if json.Unmarshal(raw, &m) != nil {
		return nil
	}
	return m
}

func strFieldOK(m map[string]json.RawMessage, key string) (string, bool) {
	v, ok := m[key]
	if !ok {
		return "", false
	}
	var s string
	if json.Unmarshal(v, &s) != nil {
		return "", false
	}
	return s, true
}

func strField(m map[string]json.RawMessage, key string) string {
	s, _ := strFieldOK(m, key)
	return s
}

func firstStr(m map[string]json.RawMessage, keys ...string) string {
	for _, k := range keys {
		if s, ok := strFieldOK(m, k); ok && s != "" {
			return s
		}
	}
	return ""
}

func boolField(m map[string]json.RawMessage, key string) bool {
	v, ok := m[key]
	if !ok {
		return false
	}
	var b bool
	if json.Unmarshal(v, &b) == nil {
		return b
	}
	return false
}

// normHead is the whitespace-collapsed head of s, capped at cap runes — the
// region / signature identity used by the churn and repeat-failure detectors.
func normHead(s string, cap int) string {
	joined := strings.Join(strings.Fields(s), " ")
	if cap <= 0 {
		return ""
	}
	r := []rune(joined)
	if len(r) > cap {
		return string(r[:cap])
	}
	return joined
}

// txtStr flattens a content field (string or list of blocks) to text, capped at
// capN bytes — the tool_result error text the lens signatures match against.
func txtStr(raw json.RawMessage, capN int) string {
	if capN <= 0 || len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		if len(s) > capN {
			return s[:capN]
		}
		return s
	}
	var arr []json.RawMessage
	if json.Unmarshal(raw, &arr) == nil {
		var out strings.Builder
		n := 0
		for _, item := range arr {
			if n >= capN {
				break
			}
			var obj map[string]json.RawMessage
			var part string
			if json.Unmarshal(item, &obj) == nil {
				if c, ok := obj["content"]; ok {
					part = txtStr(c, capN-n)
				} else if t, ok := obj["text"]; ok {
					part = txtStr(t, capN-n)
				}
			} else {
				part = txtStr(item, capN-n)
			}
			out.WriteString(part)
			n += len(part)
		}
		return out.String()
	}
	return ""
}

func round1(x float64) float64 {
	return math.Round(x*10) / 10
}
