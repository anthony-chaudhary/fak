// errorclass.go — the #10668 read-side observability half of the
// error/retry/abort-journaling gap. Codex rollout journals record NO error,
// retry, or stream-failure events at all: terminal provider failures surface
// only as free text inside task_complete (#10634), every turn_aborted reads
// reason:"interrupted" regardless of cause, and event_msg payload types fak's
// readers do not interpret are silently dropped — so the typed error/retry/
// terminal events upstream may add later (#10668) would be invisible to every
// existing consumer. This file makes the readers COUNT what they cannot
// interpret, and classifies the two things journals DO let us derive:
//
//	(a) Unknown-payload census — every event_msg payload type by name, plus
//	    the subset outside the interpreted set. Counting never errors and
//	    never changes any existing signature: consumers that ignore unknown
//	    types keep working unchanged.
//	(b) Abort-cause classification — the recorded reason is preserved
//	    (`interrupted` stays one enum member); only TORN-TAIL evidence (the
//	    rollout's final line failed to parse while the abort is the last
//	    parsed record) upgrades an abort to a process-death tail. Everything
//	    else is classified conservatively: a recorded non-`interrupted`
//	    reason outranks the tail heuristic, and an empty reason stays
//	    unrecorded. Journal timestamps cannot separate a user interrupt from
//	    a provider-stall interrupt, so this fold does not pretend to.
//	(c) Provider error classes — code-anchored status references (5xx / 429 /
//	    400 ONLY) extracted from terminal task_complete free text at READ
//	    time, reduced to class tokens with the text dropped (the
//	    analytics.go privacy contract). Code-anchored on purpose: the live
//	    corpus is full of successful turns whose final message REPORTS on
//	    502s, so a bare-number match would poison the counts. Counts and
//	    classes are correlation aids over what the journal happens to
//	    contain — never proof the turn failed.
package codexlifecycle

import (
	"os"
	"regexp"
	"sort"
	"strings"
)

// RolloutCensus is one rollout's payload census and tail shape. It is
// optional sidecar output: both rollout readers return it from their Census
// variants and leave their original signatures untouched.
type RolloutCensus struct {
	// PayloadTypes counts EVERY event_msg payload type seen, interpreted or
	// not — the event-type inventory the audited day report needed.
	PayloadTypes map[string]int `json:"payload_types,omitempty"`
	// Unknown counts the subset NOT in interpretedPayloadKinds: today's
	// item_completed / thread_settings_applied / thread_goal_updated, and any
	// typed error/retry/terminal event upstream adds later.
	Unknown map[string]int `json:"unknown_payload_types,omitempty"`
	// TornTail reports that the rollout's final non-empty line failed to
	// parse — writer death or a stray line, the process-death evidence the
	// abort classifier consumes.
	TornTail bool `json:"torn_tail,omitempty"`
}

func (c *RolloutCensus) addPayload(payloadType string) {
	if c == nil || payloadType == "" {
		return
	}
	if c.PayloadTypes == nil {
		c.PayloadTypes = map[string]int{}
	}
	c.PayloadTypes[payloadType]++
	if !interpretedPayloadKinds[payloadType] {
		if c.Unknown == nil {
			c.Unknown = map[string]int{}
		}
		c.Unknown[payloadType]++
	}
}

// interpretedPayloadKinds is the closed set of event_msg payload types this
// package's readers decode. Anything else is census-only.
var interpretedPayloadKinds = map[string]bool{
	KindStarted:  true,
	KindComplete: true,
	KindAborted:  true,
	kindTokens:   true,
}

// AbortClass is the closed abort-cause vocabulary (the #10668 read-side
// subset: only what journals let us derive today).
type AbortClass string

const (
	// AbortInterrupted preserves the producer's recorded reason — today every
	// turn_aborted in the corpus reads "interrupted".
	AbortInterrupted AbortClass = "interrupted"
	// AbortProcessDeathTail: the abort is the rollout's last parsed record
	// AND the final line is torn — the writer died mid-write around the
	// abort, which is process-death evidence a clean interrupt lacks.
	AbortProcessDeathTail AbortClass = "process_death_tail"
	// AbortOtherRecorded: a recorded reason other than "interrupted" (future
	// upstream enum values). Counted as recorded, not interpreted.
	AbortOtherRecorded AbortClass = "other_recorded"
	// AbortUnrecorded: no reason on the record and no tail evidence.
	AbortUnrecorded AbortClass = "unrecorded"
)

// ClassifyAbort maps one turn_aborted record's derivable cause. Precedence is
// evidence-strength: a specific recorded reason outranks the tail heuristic,
// which outranks the generic "interrupted" default.
func ClassifyAbort(reason string, atTornTail bool) AbortClass {
	r := strings.ToLower(strings.TrimSpace(reason))
	switch {
	case r != "" && r != string(AbortInterrupted):
		return AbortOtherRecorded
	case atTornTail:
		return AbortProcessDeathTail
	case r == string(AbortInterrupted):
		return AbortInterrupted
	default:
		return AbortUnrecorded
	}
}

// Provider error classes — the closed status-class vocabulary of #10668(c).
const (
	ErrClassHTTP5xx = "http_5xx"
	ErrClassHTTP429 = "http_429"
	ErrClassHTTP400 = "http_400"
)

// Status-class matchers. Every alternative is CODE-ANCHORED: an explicit
// http/status prefix, a well-known code+phrase pair, or the bare class jargon
// token `5xx` (jargon for the class itself, never a lone number). A bare
// \b502\b would match token counts, ports, and agent prose reporting on 502s
// (both abundant in the audited corpus), so it is deliberately absent.
var (
	statusClass5xxRE = regexp.MustCompile(`(?i)\bhttp[ _-]?(?:5\d{2}|5xx)\b` +
		`|\bstatus(?:[ _-]?code)?[\s_:=-]*(?:5\d{2}|5xx)\b` +
		`|\b5xx\b` +
		`|\b(?:502|503|504)[ _-]+(?:bad[ _-]?gateway|gateway[ _-]?timeout|service[ _-]?unavailable|internal[ _-]?server[ _-]?error)\b`)
	statusClass429RE = regexp.MustCompile(`(?i)\bhttp[ _-]?429\b` +
		`|\bstatus(?:[ _-]?code)?[\s_:=-]*429\b`)
	statusClass400RE = regexp.MustCompile(`(?i)\bhttp[ _-]?400\b` +
		`|\bstatus(?:[ _-]?code)?[\s_:=-]*400\b` +
		`|\b400[ _-]+bad[ _-]?request\b`)
)

// ClassifyStatusClasses reduces terminal task_complete free text to status
// class tokens (fixed order, deduped by construction) and drops the text.
func ClassifyStatusClasses(text string) []string {
	if text == "" {
		return nil
	}
	var out []string
	if statusClass5xxRE.MatchString(text) {
		out = append(out, ErrClassHTTP5xx)
	}
	if statusClass429RE.MatchString(text) {
		out = append(out, ErrClassHTTP429)
	}
	if statusClass400RE.MatchString(text) {
		out = append(out, ErrClassHTTP400)
	}
	return out
}

// ErrorClassSchema identifies the report shape.
const ErrorClassSchema = "fak-codex-errorclass/1"

// ErrorClassReport is the corpus-wide census + classification report. Every
// field is a count, a type name, or a closed class token — no paths, session
// ids, prompts, or message content anywhere (the analytics.go privacy
// contract; the root directory is the operator's own --root argument).
type ErrorClassReport struct {
	Schema     string `json:"schema"`
	Root       string `json:"root"`
	Sessions   int    `json:"sessions"`
	Unreadable int    `json:"unreadable,omitempty"`

	// (a) unknown-payload census.
	PayloadTypes        map[string]int `json:"payload_types"`
	UnknownPayloadTypes map[string]int `json:"unknown_payload_types"`
	UnknownPayloadTotal int            `json:"unknown_payload_total"`
	RolloutsWithUnknown int            `json:"rollouts_with_unknown"`
	TornTailRollouts    int            `json:"torn_tail_rollouts"`

	// (b) abort-cause classification over observed turn_aborted records.
	TurnAborted  int            `json:"turn_aborted"`
	AbortClasses map[string]int `json:"abort_classes"`

	// (c) provider error classes over terminal task_complete free text.
	TaskComplete         int            `json:"task_complete"`
	TerminalsWithClass   int            `json:"terminals_with_class"`
	ProviderErrorClasses map[string]int `json:"provider_error_classes"`
}

// ScanErrorClasses folds every rollout under root through the census reader
// and reports the census plus both classifications. Durability and scoping
// match ScanCorpus: unreadable rollouts are counted, never fatal; opt.CWD
// scopes to one repository's sessions; opt.Limit caps files, newest first.
func ScanErrorClasses(root string, opt ScanOptions) (ErrorClassReport, error) {
	rep := ErrorClassReport{
		Schema:               ErrorClassSchema,
		Root:                 root,
		PayloadTypes:         map[string]int{},
		UnknownPayloadTypes:  map[string]int{},
		AbortClasses:         map[string]int{},
		ProviderErrorClasses: map[string]int{},
	}
	paths, err := rolloutPaths(root, opt.Limit)
	if err != nil {
		return rep, err
	}
	for _, p := range paths {
		if _, statErr := os.Stat(p); statErr != nil {
			rep.Unreadable++
			continue
		}
		fh, openErr := os.Open(p)
		if openErr != nil {
			rep.Unreadable++
			continue
		}
		meta, records, census, parseErr := ReadAnalyticsRolloutCensus(fh)
		_ = fh.Close()
		if parseErr != nil {
			rep.Unreadable++
			continue
		}
		if opt.CWD != "" && !sameDir(meta.CWD, opt.CWD) {
			continue
		}
		rep.Sessions++
		mergeCensus(&rep, census)

		// The tail heuristic binds only to the rollout's FINAL parsed record:
		// an abort with records after it has live-session evidence against
		// process death, whatever its reason says.
		for i, r := range records {
			switch r.Kind {
			case KindAborted:
				rep.TurnAborted++
				atTornTail := census.TornTail && i == len(records)-1
				rep.AbortClasses[string(ClassifyAbort(r.Reason, atTornTail))]++
			case KindComplete:
				rep.TaskComplete++
				if len(r.ErrClasses) > 0 {
					rep.TerminalsWithClass++
					for _, cl := range r.ErrClasses {
						rep.ProviderErrorClasses[cl]++
					}
				}
			}
		}
	}
	return rep, nil
}

func mergeCensus(rep *ErrorClassReport, c RolloutCensus) {
	for k, n := range c.PayloadTypes {
		rep.PayloadTypes[k] += n
	}
	for k, n := range c.Unknown {
		rep.UnknownPayloadTypes[k] += n
		rep.UnknownPayloadTotal += n
	}
	if len(c.Unknown) > 0 {
		rep.RolloutsWithUnknown++
	}
	if c.TornTail {
		rep.TornTailRollouts++
	}
}

// CountRow is one deterministic (key, count) row for text rendering.
type CountRow struct {
	Key string
	N   int
}

// SortedCountRows renders a count map in stable key order (encoding/json
// already sorts map keys; text output needs the same determinism).
func SortedCountRows(m map[string]int) []CountRow {
	rows := make([]CountRow, 0, len(m))
	for k, n := range m {
		rows = append(rows, CountRow{Key: k, N: n})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Key < rows[j].Key })
	return rows
}
