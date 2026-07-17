package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/headlesslint"
	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
	"github.com/anthony-chaudhary/fak/internal/resume/transcript"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
)

// guard_stops.go — make the guard Stop hook's session-ending decisions TYPED,
// COUNTABLE, and OBSERVABLE.
//
// The Stop hook (guard_stophook.go) decides, at every turn end, whether to block
// an unchosen end_turn (keep the session working) or allow the stop. Until now
// every one of those outcomes was ephemeral: a single stderr line the harness
// consumes and a bare exit code. That includes the outcomes an operator most wants
// to see — the bounded stand-downs ("give up after N spins") and, worse, the
// FAIL-OPEN allows (gateway unreachable, bad args, gauge missing) that silently let
// a session stop for a reason that has nothing to do with the work being done.
// Those invisible fail-open stops are what read as "the guard is still killing
// sessions": you cannot count what leaves no record.
//
// This file adds three things, all FAIL-OPEN (never change the hook's decision or
// exit code):
//   1. A closed disposition vocabulary (guardStopDisposition) — one typed value per
//      terminal outcome of runGuardStopHook, so a stop is a category, not prose.
//   2. A transcript reader — the Stop event's stdin carries transcript_path; we read
//      a bounded tail of the session JSONL to derive whether the last turn tried to
//      call a tool and whether the agent wrote its sanctioned "no allowed path:"
//      wrap-up, so a give-up can be told apart from a graceful conclusion.
//   3. An append-only JSONL ledger (one row per invocation) plus `fak guard-stops`,
//      which folds the ledger into a tally an operator can read: how many sessions
//      the guard ended, and why.

// ---- disposition vocabulary -------------------------------------------------

// guardStopDisposition is the closed set of terminal outcomes runGuardStopHook can
// reach. Every return site classifies into exactly one of these, so the reason a
// session stopped (or kept going) is a typed fact, not a log line.
type guardStopDisposition string

const (
	// clean stops (the hook allowed a stop it considers legitimate)
	stopDispCleanCompletion guardStopDisposition = "clean_completion" // stage=allow, gauge quiet, handoff gate clear
	stopDispCleanWrapup     guardStopDisposition = "clean_wrapup"     // agent wrote its sanctioned "no allowed path:" note before stopping

	// continues (the hook BLOCKED the stop to keep the session working)
	stopDispToolFeedbackContinue     guardStopDisposition = "tool_feedback_continue"     // block: let the model repair a malformed/misrouted call
	stopDispDenyAllContinue          guardStopDisposition = "deny_all_continue"          // block: blind deny-all ladder (nudge/warn/final)
	stopDispSameIssueContinue        guardStopDisposition = "same_issue_continue"        // block: same-issue deny-all ladder
	stopDispHandoffBlock             guardStopDisposition = "handoff_block"              // block: task-handoff Stop gate held the stop
	stopDispOperatorDirectedContinue guardStopDisposition = "operator_directed_continue" // block: headless turn asked a human; feed the choicetriage remediation back
	stopDispOutputStyleContinue      guardStopDisposition = "output_style_continue"      // block: final turn closed on a prose wall; feed the re-cast remediation back

	// operator-directed non-blocking outcomes (see guard_operator_directed.go): the final
	// turn addressed a human, but the gate allowed the stop.
	stopDispOperatorDirectedEscalate guardStopDisposition = "operator_directed_escalate" // allow: HUMAN_RESIDUAL — a genuine escalation, routed not re-prompted
	stopDispOperatorDirectedWarn     guardStopDisposition = "operator_directed_warn"     // allow: warn soak — remediation printed, stop allowed
	stopDispOperatorDirectedShadow   guardStopDisposition = "operator_directed_shadow"   // allow: shadow — would-enforce decision logged, stop allowed

	// output-style (closing-shape) non-blocking outcomes (see guard_output_style.go): the
	// final turn closed on a prose wall, but the gate allowed the stop.
	stopDispOutputStyleWarn   guardStopDisposition = "output_style_warn"   // allow: warn soak — re-cast remediation printed, stop allowed
	stopDispOutputStyleShadow guardStopDisposition = "output_style_shadow" // allow: shadow — would-enforce decision logged, stop allowed

	// bounded stand-downs (the hook gave up and ALLOWED the stop after the ladder)
	stopDispBlindGiveUp          guardStopDisposition = "blind_give_up"           // stood down past --deny-all-max (blind)
	stopDispSameIssueGiveUp      guardStopDisposition = "same_issue_give_up"      // stood down at the same-issue depth
	stopDispToolFeedbackGiveUp   guardStopDisposition = "tool_feedback_give_up"   // stood down past the tool-feedback continue bound (#A6)
	stopDispHandoffGiveUp        guardStopDisposition = "handoff_give_up"         // stood down: handoff still invalid after a prior block (stop_hook_active) (#A2)
	stopDispHandoffSessionGiveUp guardStopDisposition = "handoff_session_give_up" // stood down: handoff still invalid after the per-session block ceiling (the bound a real worker's late re-stop never reaches via stop_hook_active)

	// non-enforcing outcomes
	stopDispModeOff guardStopDisposition = "mode_off" // --deny-all-continue=off: layer disabled
	stopDispShadow  guardStopDisposition = "shadow"   // shadow mode: would-be decision logged, stop allowed

	// fail-open allows (the hook could not decide, so it allowed the stop) — the
	// class that is otherwise invisible and reads as "still killing sessions".
	stopDispFailOpenBadArgs          guardStopDisposition = "fail_open_bad_args"
	stopDispFailOpenBadMode          guardStopDisposition = "fail_open_bad_mode"
	stopDispFailOpenNoMetricsURL     guardStopDisposition = "fail_open_no_metrics_url"
	stopDispFailOpenGaugeUnavailable guardStopDisposition = "fail_open_gauge_unavailable"
)

// guardStopKind is the coarse grouping a disposition rolls up to, so the summarizer
// can answer the headline question ("how many sessions did the guard end, and were
// those ends chosen or fail-open?") without enumerating every disposition.
type guardStopKind string

const (
	stopKindClean     guardStopKind = "clean"     // allowed a chosen, legitimate stop
	stopKindContinue  guardStopKind = "continue"  // blocked the stop; the session kept working
	stopKindStandDown guardStopKind = "standdown" // bounded give-up: allowed the stop after the ladder
	stopKindFailOpen  guardStopKind = "failopen"  // allowed the stop because the hook could not decide
	stopKindShadow    guardStopKind = "shadow"    // observing only; not enforcing
	stopKindOff       guardStopKind = "off"       // layer disabled
)

// guardStopDispositionKind maps a disposition to its coarse kind. An unknown value
// rolls up to failopen — the conservative reading ("the hook did not cleanly
// decide") rather than silently counting it as a clean stop.
func guardStopDispositionKind(d guardStopDisposition) guardStopKind {
	switch d {
	case stopDispCleanCompletion, stopDispCleanWrapup, stopDispOperatorDirectedEscalate:
		// An operator-directed escalate is a legitimate, routed conclusion — the agent was
		// right to stop on an authority wall — so it rolls up as a clean stop; the
		// OperatorDirected count keeps it separately visible.
		return stopKindClean
	case stopDispToolFeedbackContinue, stopDispDenyAllContinue, stopDispSameIssueContinue, stopDispHandoffBlock, stopDispOperatorDirectedContinue, stopDispOutputStyleContinue:
		return stopKindContinue
	case stopDispBlindGiveUp, stopDispSameIssueGiveUp, stopDispToolFeedbackGiveUp, stopDispHandoffGiveUp, stopDispHandoffSessionGiveUp:
		return stopKindStandDown
	case stopDispModeOff:
		return stopKindOff
	case stopDispShadow, stopDispOperatorDirectedWarn, stopDispOperatorDirectedShadow, stopDispOutputStyleWarn, stopDispOutputStyleShadow:
		return stopKindShadow
	default:
		return stopKindFailOpen
	}
}

// ---- ledger row -------------------------------------------------------------

// guardStopRecordSchema versions the row shape so a reader can evolve.
const guardStopRecordSchema = "fak.guard-stop.v1"

// guardStopRecord is one durable row: a single Stop-hook invocation, with the typed
// disposition, the gauges that drove it, and the transcript-derived context. Ts is
// an injected RFC3339 timestamp so the recorder stays testable — the caller owns the
// clock.
type guardStopRecord struct {
	Schema      string `json:"schema"`
	Ts          string `json:"ts,omitempty"`
	Session     string `json:"session,omitempty"`
	Disposition string `json:"disposition"`
	Kind        string `json:"kind,omitempty"`
	Stage       string `json:"stage,omitempty"`
	Signal      string `json:"signal,omitempty"` // "blind" | "same-issue" | "tool-feedback"
	Mode        string `json:"mode,omitempty"`
	Exit        int    `json:"exit"`
	Blocked     bool   `json:"blocked,omitempty"`
	Depth       int    `json:"depth,omitempty"` // the consecutive count that drove the decision
	Bound       int    `json:"bound,omitempty"` // the give-up bound (max, or same-issue depth)
	// HandoffBlockSeq is how many times the task-handoff Stop gate has blocked THIS
	// session, counting the current decision. It is the per-session sequence the
	// handoff-block ceiling bounds (see guardHandoffBlockCeiling); surfaced so an
	// operator can see a worker riding the ceiling (seq climbing to the bound) instead
	// of the churn being invisible. Zero for any stop the handoff gate did not block.
	HandoffBlockSeq int `json:"handoff_block_seq,omitempty"`

	DenyAllConsecutive      int `json:"deny_all_consecutive,omitempty"`
	DenyAllSameConsecutive  int `json:"deny_all_same_consecutive,omitempty"`
	ToolFeedbackConsecutive int `json:"tool_feedback_consecutive,omitempty"`

	StopHookActive bool                   `json:"stop_hook_active,omitempty"`
	Transcript     *guardStopTranscript   `json:"transcript,omitempty"`
	Next           *sessionctl.NextRecord `json:"next,omitempty"`
	Note           string                 `json:"note,omitempty"`
}

// guardStopTranscript is the bounded read-back of the session transcript at the
// moment of the stop. Read distinguishes "a transcript_path was given and parsed"
// from "no path / unreadable"; Truncated marks that only the tail window was scanned
// (so AssistantTurns is a lower bound).
type guardStopTranscript struct {
	Read               bool   `json:"read"`
	Truncated          bool   `json:"truncated,omitempty"`
	AssistantTurns     int    `json:"assistant_turns,omitempty"`
	LastHadToolUse     bool   `json:"last_had_tool_use,omitempty"`
	LastToolUse        string `json:"last_tool_use,omitempty"`
	NotedNoAllowedPath bool   `json:"noted_no_allowed_path,omitempty"`

	// OperatorDirected records that the FINAL assistant turn ended by addressing a
	// human — "do you want me to push?", "waiting for your confirmation", "please
	// review", "let me know if…". In an autonomous run there is no human to answer,
	// so the question hangs and the work silently stalls. internal/headlesslint (the
	// sensor-side dual of internal/choicetriage) finds it; the fields below carry the
	// top finding's linguistic Class and the choicetriage Disposition/Resolve — what
	// an autonomous worker does INSTEAD of asking (take the action, state the
	// assumption, file a ticket, or emit a typed escalation). Observe-only and
	// fail-open: recording the note never changes the stop decision, and a clean
	// final turn leaves every field zero.
	OperatorDirected            bool   `json:"operator_directed,omitempty"`
	OperatorDirectedClass       string `json:"operator_directed_class,omitempty"`
	OperatorDirectedDisposition string `json:"operator_directed_disposition,omitempty"`
	OperatorDirectedResolve     string `json:"operator_directed_resolve,omitempty"`
	OperatorDirectedCount       int    `json:"operator_directed_count,omitempty"`

	// ClosingProseWall records that the FINAL assistant turn closed on a trailing prose
	// wall — a long paragraph with no scannable bullets, burying the verdict and the next
	// step in text (internal/headlesslint.ScanClosing). It is recorded INDEPENDENTLY of the
	// operator-directed scan: a turn clean of questions can still close on a wall.
	// ClosingResolve carries the re-cast remediation the output-style rung feeds back.
	// Observe-only and fail-open: a scannable close leaves both fields zero.
	ClosingProseWall bool   `json:"closing_prose_wall,omitempty"`
	ClosingResolve   string `json:"closing_resolve,omitempty"`
}

// ---- transcript reading -----------------------------------------------------

// guardStopTranscriptTailBytes bounds the transcript read so hook latency stays flat
// no matter how long the session got. The tail carries the terminal turns, which is
// all the stop decision needs; a bigger file just means AssistantTurns is a lower
// bound (Truncated=true records that).
const guardStopTranscriptTailBytes = 512 * 1024

// parseHookTranscriptPath best-effort extracts transcript_path from a Claude Code
// Stop-hook stdin payload (the same bytes parseHookSessionID reads). "" on any miss.
func parseHookTranscriptPath(payload []byte) string {
	if len(payload) == 0 {
		return ""
	}
	var p struct {
		TranscriptPath string `json:"transcript_path"`
	}
	if json.Unmarshal(payload, &p) != nil {
		return ""
	}
	return strings.TrimSpace(p.TranscriptPath)
}

// readGuardStopTranscript reads a bounded tail of the session transcript and derives
// the stop-relevant signals. It is FAIL-OPEN: an empty path yields nil (no transcript
// context), and a missing/unreadable/empty file yields a record with Read=false
// rather than an error. The signals track the LAST real (non-synthetic) assistant
// turn — whether it tried to call a tool, and whether the agent wrote its sanctioned
// "no allowed path:" wrap-up.
func readGuardStopTranscript(path string) *guardStopTranscript {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil
	}
	sig := &guardStopTranscript{}
	recs := transcript.LoadFileTail(path, guardStopTranscriptTailBytes)
	if len(recs) == 0 {
		return sig // path was given but nothing parsed — Read stays false
	}
	sig.Read = true
	if fi, err := os.Stat(path); err == nil && fi.Size() > guardStopTranscriptTailBytes {
		sig.Truncated = true
	}
	var lastText string
	for _, r := range recs {
		if r.Role() != "assistant" || r.IsSynthetic() {
			continue
		}
		sig.AssistantTurns++
		name := r.LastToolUseName()
		sig.LastHadToolUse = name != ""
		sig.LastToolUse = name
		lastText = r.Text()
		sig.NotedNoAllowedPath = transcriptNotesNoAllowedPath(lastText)
	}
	// Fold the FINAL assistant turn through headlesslint. Gate on the turn having
	// NO tool call: a turn that still tried a tool is not stopping-to-ask — the
	// harness feeds the tool result back and the session keeps going. It is the
	// prose-only end_turn that actually stops the session on an unanswered
	// question, and that is the one worth recording.
	if lastText != "" && !sig.LastHadToolUse {
		applyHeadlessLintSignal(sig, lastText)
		applyClosingSignal(sig, lastText)
	}
	return sig
}

// applyHeadlessLintSignal records the top operator-directed note in the final
// assistant turn, if any. It is the guard-side use of internal/headlesslint: the
// SENSOR that turns "the agent ended by asking a human" into a typed, countable
// fact, sitting beside the sanctioned no-allowed-path wrap-up. The top finding is
// the first in line order (headlesslint yields at most one Finding per line,
// most-specific Class first); its choicetriage Disposition/Resolve name what an
// autonomous worker does instead of asking. Fail-open: a clean scan is a no-op.
func applyHeadlessLintSignal(sig *guardStopTranscript, finalTurn string) {
	rep := headlesslint.Scan(finalTurn)
	if rep.Count == 0 {
		return
	}
	sig.OperatorDirected = true
	sig.OperatorDirectedCount = rep.Count
	top := rep.Findings[0]
	sig.OperatorDirectedClass = string(top.Class)
	sig.OperatorDirectedDisposition = string(top.Disposition)
	sig.OperatorDirectedResolve = top.Resolve
}

// applyClosingSignal records whether the final assistant turn closed on a prose wall
// (internal/headlesslint.ScanClosing) rather than scannable bullets — the SENSOR the
// output-style rung (guard_output_style.go) reads. It is INDEPENDENT of
// applyHeadlessLintSignal: a turn clean of operator-directed questions can still close
// on a wall, so both run over the same final turn. Fail-open: a scannable close is a
// no-op, leaving ClosingProseWall/ClosingResolve zero.
func applyClosingSignal(sig *guardStopTranscript, finalTurn string) {
	rep := headlesslint.ScanClosing(finalTurn, false)
	if !rep.Refused() {
		return
	}
	sig.ClosingProseWall = true
	sig.ClosingResolve = rep.Resolve
}

// transcriptNotesNoAllowedPath reports whether assistant text carries the sanctioned
// wrap-up note ("no allowed path: <reason>") the guard teaches the model to write
// when it concludes there is no permitted way forward. Matching it lets a stop be
// classified as a graceful conclusion rather than a bare give-up.
func transcriptNotesNoAllowedPath(text string) bool {
	return strings.Contains(strings.ToLower(text), "no allowed path")
}

// ---- ledger path + append ---------------------------------------------------

const (
	// guardStopsLedgerEnv is the absolute stops-ledger path the guard installer
	// injects into the Stop hook child. Empty (unset) disables recording entirely,
	// so an unrelated hook test never writes a row into the repo.
	guardStopsLedgerEnv = "FAK_GUARD_STOPS_LEDGER"
	// guardStopsModeEnv opts recording out (value "off") even when a ledger is wired.
	guardStopsModeEnv = "FAK_GUARD_STOPS_MODE"
	// guardStopsLedgerDefaultRel is the repo-root-relative default the installer
	// injects, in the gitignored runtime-state directory, so active hooks never dirty tracked docs.
	guardStopsLedgerDefaultRel = ".fak/guard-stops.jsonl"
)

// guardStopsLedgerConfigured returns the wired ledger path for the WRITER (the Stop
// hook), or "" when recording is disabled (no ledger injected, or mode=off).
func guardStopsLedgerConfigured() string {
	if strings.EqualFold(strings.TrimSpace(os.Getenv(guardStopsModeEnv)), "off") {
		return ""
	}
	return strings.TrimSpace(os.Getenv(guardStopsLedgerEnv))
}

// guardStopsLedgerDefault is the absolute default path the installer injects:
// <repo root>/.fak/guard-stops.jsonl. Empty when the root is unresolvable.
func guardStopsLedgerDefault() string {
	root := repoRoot()
	if strings.TrimSpace(root) == "" {
		return ""
	}
	return filepath.Join(root, filepath.FromSlash(guardStopsLedgerDefaultRel))
}

// guardStopsLedgerResolved is the path the READER (`fak guard-stops`) uses: an
// explicit env override wins, else the repo-root default. Unlike the writer path it
// ignores the off switch — a reader wants to read whatever is on disk.
func guardStopsLedgerResolved() string {
	if p := strings.TrimSpace(os.Getenv(guardStopsLedgerEnv)); p != "" {
		return p
	}
	return guardStopsLedgerDefault()
}

// ---- per-session handoff-block ceiling --------------------------------------

const (
	// guardHandoffBlockCeilingEnv overrides how many times the task-handoff Stop gate
	// may BLOCK a single session's clean stop before it stands down and allows the stop.
	guardHandoffBlockCeilingEnv = "FAK_GUARD_HANDOFF_BLOCK_CEILING"
	// guardHandoffBlockCeilingDefault is that bound's default. The gate's original only
	// stand-down (stop_hook_active) fires ONLY on an immediate harness re-fire; a real
	// worker re-stops many turns later with stop_hook_active=false, so without this
	// ceiling the gate can re-block a worker that has nothing left to hand off every
	// 24-65 turns indefinitely (the observed handoff-loop). Three blocks is enough to
	// demand the handoff firmly without holding a stuck session forever.
	guardHandoffBlockCeilingDefault = 3
	// guardHandoffBlockCounterRel is the subdirectory (beside the stops ledger) holding
	// one tiny per-session counter file, so the ceiling is O(1) and exact per session.
	guardHandoffBlockCounterRel = "handoff-blocks"
)

// guardHandoffBlockCeiling resolves the per-session handoff-block ceiling: the env
// override if a valid non-negative int, else the default. A value <= 0 disables the
// ceiling (the gate reverts to its prior stop_hook_active-only bound).
func guardHandoffBlockCeiling() int {
	v := strings.TrimSpace(os.Getenv(guardHandoffBlockCeilingEnv))
	if v == "" {
		return guardHandoffBlockCeilingDefault
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return guardHandoffBlockCeilingDefault
	}
	if n < 0 {
		return 0
	}
	return n
}

// guardHandoffBlockCounterPath is the per-session counter file, beside the wired stops
// ledger. Empty when no session id is known or no ledger is configured — in which case
// the ceiling is inert and the gate keeps its prior behavior (fail-open). Each claude
// session owns a unique id, so its counter file is never shared and its Stop hooks fire
// serially, so no concurrent writer races on it.
func guardHandoffBlockCounterPath(session string) string {
	session = strings.TrimSpace(session)
	if session == "" {
		return ""
	}
	ledger := guardStopsLedgerConfigured()
	if ledger == "" {
		return ""
	}
	safe := sanitizeSessionForPath(session)
	if safe == "" {
		return ""
	}
	return filepath.Join(filepath.Dir(ledger), guardHandoffBlockCounterRel, safe)
}

// sanitizeSessionForPath maps a session id to a single safe path segment (session ids
// are UUID-like, but be defensive against a path separator smuggled in).
func sanitizeSessionForPath(s string) string {
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			return r
		default:
			return '_'
		}
	}, s)
}

// readSessionHandoffBlocks returns how many times the handoff gate has already blocked
// this session. FAIL-OPEN: any miss (no path, unreadable, unparsable) reads as 0.
func readSessionHandoffBlocks(session string) int {
	p := guardHandoffBlockCounterPath(session)
	if p == "" {
		return 0
	}
	b, err := os.ReadFile(p)
	if err != nil {
		return 0
	}
	n, err := strconv.Atoi(strings.TrimSpace(string(b)))
	if err != nil || n < 0 {
		return 0
	}
	return n
}

// bumpSessionHandoffBlocks increments this session's handoff-block counter by one.
// FAIL-OPEN: a no-op on any error (no path, unwritable dir) — the ceiling then simply
// never engages for this session rather than changing the stop decision.
func bumpSessionHandoffBlocks(session string) {
	p := guardHandoffBlockCounterPath(session)
	if p == "" {
		return
	}
	n := readSessionHandoffBlocks(session) + 1
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return
	}
	_ = os.WriteFile(p, []byte(strconv.Itoa(n)), 0o644)
}

// emitGuardStopRecord appends one row to the wired ledger. FAIL-OPEN: a no-op when no
// ledger is configured, and an append error is advisory (stderr) only — never a
// change to the hook's decision, which has already been made by the time we record.
func emitGuardStopRecord(stderr io.Writer, rec guardStopRecord) {
	ledger := guardStopsLedgerConfigured()
	if ledger == "" {
		return
	}
	if err := appendGuardStopRecord(ledger, rec); err != nil {
		fmt.Fprintf(stderr, "fak guard Stop: stops ledger append skipped (fail-open): %v\n", err)
	}
}

// appendGuardStopRecord writes rec as a single JSONL line (one Write so concurrent
// O_APPEND writers interleave at line granularity), creating parents as needed.
func appendGuardStopRecord(path string, rec guardStopRecord) error {
	if rec.Schema == "" {
		rec.Schema = guardStopRecordSchema
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	b, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	return writeLine(f, b)
}

// writeLine writes b plus a trailing newline in a single Write call.
func writeLine(w io.Writer, b []byte) error {
	_, err := w.Write(append(b, '\n'))
	return err
}

// ---- summary (fak guard-stops) ----------------------------------------------

// guardStopsSummary is the folded value view read back from the ledger.
type guardStopsSummary struct {
	Ledger    string                       `json:"ledger"`
	Total     int                          `json:"total"`
	ByKind    map[guardStopKind]int        `json:"by_kind,omitempty"`
	ByDisp    map[guardStopDisposition]int `json:"by_disposition,omitempty"`
	StandDown int                          `json:"stand_down"`
	FailOpen  int                          `json:"fail_open"`
	// OperatorDirected counts recorded stops whose final turn asked a human instead
	// of acting (headless-directed) — the stops the choicetriage doctrine says an
	// autonomous process should have resolved itself, not paged a person for.
	OperatorDirected int               `json:"operator_directed,omitempty"`
	FirstTs          string            `json:"first_ts,omitempty"`
	LastTs           string            `json:"last_ts,omitempty"`
	Recent           []guardStopRecord `json:"recent,omitempty"`
}

// recordKind resolves a row's coarse kind: the stored Kind if present, else recomputed
// from the disposition (so an older row written before Kind existed still groups).
func recordKind(r guardStopRecord) guardStopKind {
	if k := guardStopKind(r.Kind); k != "" {
		return k
	}
	return guardStopDispositionKind(guardStopDisposition(r.Disposition))
}

// summarizeGuardStops folds the ledger content into counts by kind and disposition,
// plus the most recent guard-ENDED decisions (stand-downs and fail-opens — the rows
// an operator inspects, since those are the stops the guard allowed without the agent
// clearly choosing them). Malformed/foreign lines are skipped.
func summarizeGuardStops(content string, recentN int) guardStopsSummary {
	rows := jsonlledger.Parse(content, func(r guardStopRecord) bool {
		return r.Schema == guardStopRecordSchema
	})
	sum := guardStopsSummary{
		ByKind: map[guardStopKind]int{},
		ByDisp: map[guardStopDisposition]int{},
	}
	var ended []guardStopRecord
	for _, r := range rows {
		sum.Total++
		kind := recordKind(r)
		sum.ByKind[kind]++
		sum.ByDisp[guardStopDisposition(r.Disposition)]++
		if r.Transcript != nil && r.Transcript.OperatorDirected {
			sum.OperatorDirected++
		}
		switch kind {
		case stopKindStandDown:
			sum.StandDown++
			ended = append(ended, r)
		case stopKindFailOpen:
			sum.FailOpen++
			ended = append(ended, r)
		}
		if r.Ts != "" {
			if sum.FirstTs == "" || r.Ts < sum.FirstTs {
				sum.FirstTs = r.Ts
			}
			if r.Ts > sum.LastTs {
				sum.LastTs = r.Ts
			}
		}
	}
	if recentN > 0 && len(ended) > 0 {
		if len(ended) > recentN {
			ended = ended[len(ended)-recentN:]
		}
		sum.Recent = ended
	}
	return sum
}

// guardStopKindOrder is the fixed reading order for the kind rollup: what stopped
// cleanly, what kept going, then the two guard-ended classes the operator cares about.
var guardStopKindOrder = []guardStopKind{
	stopKindClean, stopKindContinue, stopKindStandDown, stopKindFailOpen, stopKindShadow, stopKindOff,
}

// renderGuardStopsSummary formats the folded view for a human reader.
func renderGuardStopsSummary(sum guardStopsSummary) string {
	var b strings.Builder
	if sum.Total == 0 {
		fmt.Fprintf(&b, "fak guard stops: no stop decisions recorded yet.\n")
		fmt.Fprintf(&b, "  ledger: %s\n", sum.Ledger)
		b.WriteString("  The Stop hook records one typed row per turn-end decision once a `fak guard` session runs.")
		return b.String()
	}
	fmt.Fprintf(&b, "fak guard stops: %d decision(s) recorded", sum.Total)
	if sum.FirstTs != "" {
		fmt.Fprintf(&b, " (%s .. %s)", sum.FirstTs, sum.LastTs)
	}
	b.WriteString("\n")
	fmt.Fprintf(&b, "  ledger: %s\n", sum.Ledger)
	for _, k := range guardStopKindOrder {
		if n := sum.ByKind[k]; n > 0 {
			fmt.Fprintf(&b, "  %-10s %d\n", k, n)
		}
	}
	ended := sum.StandDown + sum.FailOpen
	fmt.Fprintf(&b, "  → guard-ended sessions: %d (%d bounded stand-down, %d fail-open)\n", ended, sum.StandDown, sum.FailOpen)
	if sum.FailOpen > 0 {
		b.WriteString("    fail-open stops are NOT completions: the hook could not reach a decision (gateway unreachable, bad args, gauge missing) and allowed the stop. Investigate the wiring.\n")
	}
	if sum.OperatorDirected > 0 {
		fmt.Fprintf(&b, "  → %d stop(s) ended with the agent asking a human instead of acting (headless-directed).\n", sum.OperatorDirected)
		b.WriteString("    An autonomous run has no one to answer; the choicetriage fold says what to do instead (take the action, state the assumption, file a ticket, or escalate). Run `fak headless-lint` on the turn to see the remediation.\n")
		// Enforcement breakdown: how many of those the operator-directed gate actively acted on
		// (blocked to make the agent continue, or routed as a HUMAN_RESIDUAL escalation) vs merely
		// observed (warn soak / shadow). Reads the soak → promote decision straight off the ledger.
		enforced := sum.ByDisp[stopDispOperatorDirectedContinue] + sum.ByDisp[stopDispOperatorDirectedEscalate]
		observed := sum.ByDisp[stopDispOperatorDirectedWarn] + sum.ByDisp[stopDispOperatorDirectedShadow]
		if enforced > 0 || observed > 0 {
			fmt.Fprintf(&b, "    of these: %d enforced (%d auto-continued, %d escalated), %d observed only (warn/shadow soak).\n",
				enforced, sum.ByDisp[stopDispOperatorDirectedContinue], sum.ByDisp[stopDispOperatorDirectedEscalate], observed)
		}
	}

	// per-disposition breakdown, most frequent first
	disps := make([]guardStopDisposition, 0, len(sum.ByDisp))
	for d := range sum.ByDisp {
		disps = append(disps, d)
	}
	sort.Slice(disps, func(i, j int) bool {
		if sum.ByDisp[disps[i]] != sum.ByDisp[disps[j]] {
			return sum.ByDisp[disps[i]] > sum.ByDisp[disps[j]]
		}
		return disps[i] < disps[j]
	})
	for _, d := range disps {
		fmt.Fprintf(&b, "  %-28s %d\n", d, sum.ByDisp[d])
	}

	// recent guard-ended decisions worth a look
	for _, r := range sum.Recent {
		fmt.Fprintf(&b, "  - [%s] %s", firstNonEmpty(r.Ts, "?"), r.Disposition)
		if r.Stage != "" {
			fmt.Fprintf(&b, " stage=%s", r.Stage)
		}
		if r.Depth > 0 {
			fmt.Fprintf(&b, " depth=%d", r.Depth)
			if r.Bound > 0 {
				fmt.Fprintf(&b, "/%d", r.Bound)
			}
		}
		if r.HandoffBlockSeq > 0 {
			fmt.Fprintf(&b, " handoff_seq=%d", r.HandoffBlockSeq)
		}
		if r.Transcript != nil && r.Transcript.NotedNoAllowedPath {
			b.WriteString(" (agent noted no-allowed-path)")
		}
		if r.Transcript != nil && r.Transcript.OperatorDirected {
			fmt.Fprintf(&b, " (asked a human: %s → %s)", r.Transcript.OperatorDirectedClass, r.Transcript.OperatorDirectedDisposition)
		}
		if r.Note != "" {
			fmt.Fprintf(&b, " — %s", r.Note)
		}
		b.WriteString("\n")
	}
	return strings.TrimRight(b.String(), "\n")
}

// cmdGuardStops is the `fak guard-stops` entry point: fold the typed Stop-hook
// decision ledger and print the tally.
func cmdGuardStops(argv []string) {
	os.Exit(runGuardStops(os.Stdout, os.Stderr, argv))
}

func runGuardStops(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("guard-stops", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak guard-stops [--ledger PATH] [--recent N] [--json]")
		fmt.Fprintln(stderr, "  Tally the typed Stop-hook decision ledger: clean stops, bounded")
		fmt.Fprintln(stderr, "  stand-downs, and the fail-open stops that are otherwise invisible.")
		fs.PrintDefaults()
	}
	ledgerFlag := fs.String("ledger", "", "path to the guard stops JSONL ledger (default: $FAK_GUARD_STOPS_LEDGER or <repo>/docs/nightrun/guard-stops.jsonl)")
	recentFlag := fs.Int("recent", 10, "show up to this many recent guard-ended (stand-down/fail-open) decisions")
	jsonFlag := fs.Bool("json", false, "emit the summary as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	ledger := strings.TrimSpace(*ledgerFlag)
	if ledger == "" {
		ledger = guardStopsLedgerResolved()
	}
	if ledger == "" {
		fmt.Fprintln(stderr, "fak guard-stops: no ledger path (pass --ledger, set $FAK_GUARD_STOPS_LEDGER, or run inside a repo)")
		return 1
	}
	content, err := readGuardStopsLedger(ledger)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard-stops: read %s: %v\n", ledger, err)
		return 1
	}
	sum := summarizeGuardStops(content, *recentFlag)
	sum.Ledger = ledger
	if *jsonFlag {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(sum); err != nil {
			fmt.Fprintf(stderr, "fak guard-stops: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintln(stdout, renderGuardStopsSummary(sum))
	return 0
}

// readGuardStopsLedger reads the ledger file. A missing file is a valid empty view
// (a fresh session has recorded nothing), not an error.
func readGuardStopsLedger(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", err
	}
	return string(b), nil
}
