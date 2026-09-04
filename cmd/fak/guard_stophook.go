package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionctl"
	"github.com/anthony-chaudhary/fak/internal/taskmgr"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// guard_stophook.go — the harness half of the deny-all false-stop fix.
//
// The WIRE half is unchanged and correct: when the capability floor refuses EVERY tool call
// in a turn, the gateway must report stop_reason=end_turn (else the client hangs hunting for
// a tool_use block that was dropped — the v0.15.0 contract). But end_turn halts the agent
// loop though the model wanted to act, so the turn ends with the task unfinished — a STOP the
// agent did not choose. This actuator catches that stop and, in enforce mode, RESUMES the
// agent: it is a Claude Code `Stop` hook (installed by `fak guard` into the same --settings
// file as the PreCompact hook) that reads the gateway's fak_guard_deny_all_consecutive gauge
// and, when the most recent turn was a deny-all (and we are under the retry bound), exits 2
// with a continuation instruction on stderr — which Claude Code feeds back to the model so it
// picks an allowed alternative instead of stopping. The bound (max consecutive auto-continues)
// makes it impossible to loop forever on a model that keeps re-proposing refused calls.
//
// It mirrors guard-precompact exactly: a hidden subcommand polling /metrics, fail-open on any
// unavailability (a Stop hook that cannot reach the gateway must never wedge the agent), and a
// shadow mode that logs the would-be decision while always allowing the stop.

const (
	guardStopHookEnvMode       = "FAK_GUARD_DENYALL_MODE"
	guardStopHookEnvMetricsURL = "FAK_GUARD_DENYALL_METRICS_URL"
	guardStopHookEnvMax        = "FAK_GUARD_DENYALL_MAX"
	guardStopHookEnvWarn       = "FAK_GUARD_DENYALL_WARN"
	guardStopHookEnvFinal      = "FAK_GUARD_DENYALL_FINAL"
	// guardStopHookEnvSameStop overrides the same-issue give-up depth: how many consecutive
	// deny-all turns proposing the IDENTICAL refused action (same tool+reason) end the session.
	// This is the ONLY session-ending knob on the new same-issue path; warn/final rungs derive
	// from it. A varied session (fresh block each turn) never reaches it and is never stopped.
	guardStopHookEnvSameStop = "FAK_GUARD_DENYALL_SAME_STOP"
	// guardStopHookEnvToolFeedbackMax overrides the retryable tool-feedback continue bound (#A6):
	// the max consecutive tool-feedback turns to auto-continue past before standing down. It is a
	// SEPARATE, more generous knob than the deny-all --max (a malformed call is model-fixable), but
	// it does bound what was previously an unbounded continue.
	guardStopHookEnvToolFeedbackMax = "FAK_GUARD_TOOL_FEEDBACK_MAX"

	guardTaskHandoffEnvMode = "FAK_GUARD_TASK_HANDOFF_MODE"
	guardTaskHandoffEnvFile = "FAK_GUARD_TASK_HANDOFF_FILE"
	guardTaskHandoffEnvRepo = "FAK_GUARD_TASK_HANDOFF_REPO"
	guardTaskHandoffEnvLive = "FAK_GUARD_TASK_HANDOFF_LIVE"

	// guardTaskHandoffFileEnv is the short, model-facing alias the wrapped agent sees. The
	// hook reads the guard-prefixed env above, but the continuation instruction points agents
	// at this stable name so they do not need to learn the hook's private wiring names.
	guardTaskHandoffFileEnv = "FAK_TASK_HANDOFF_FILE"

	// guardStopHookMetricName is the gateway gauge this hook polls: the count of consecutive
	// deny-all turns ending the most recent served turn (0 on a healthy completion).
	guardStopHookMetricName = "fak_guard_deny_all_consecutive"
	// guardStopHookToolFeedbackMetricName is the sibling gauge for retryable tool-call feedback
	// turns (for example malformed JSON/args). These are NOT session-stop policy input; the hook
	// uses them only to continue the turn so the model can repair the call shape.
	guardStopHookToolFeedbackMetricName = "fak_guard_tool_feedback_consecutive"
	// guardStopHookSameMetricName is the same-issue sibling of guardStopHookMetricName: consecutive
	// deny-all turns proposing the IDENTICAL refused action (same tool + same reason). This is the
	// gauge the hook now keys its give-up on — a true repeated same issue, not a blind deny-all
	// count. An older gateway that does not emit it makes the hook fall back to the blind ladder.
	guardStopHookSameMetricName = "fak_guard_deny_all_same_consecutive"
	// guardStopHookFakVerbCallsMetricName is the cumulative admitted MCP fak-verb call counter.
	// At a clean stop, 0 means fak was present but never used as a substrate (#3093).
	guardStopHookFakVerbCallsMetricName = "fak_mcp_verb_calls_total"

	// guardStopHookUnusedEnvMode gates the unused-substrate advisory: off|shadow (default
	// shadow). It is advisory-only — it NEVER blocks a stop (no enforce rung) — so it rides its
	// own knob rather than the deny-all mode, whose enforce path holds the turn open.
	guardStopHookUnusedEnvMode = "FAK_GUARD_UNUSED_SUBSTRATE_MODE"

	// The graduated back-off ladder. Rather than a single cliff (continue N times, then a hard
	// stop), the auto-continue guidance firms up with the consecutive deny-all depth, and the
	// stand-down moves much later. This gives a confused-but-capable model more room while one
	// that genuinely has no allowed path is guided, early and gently, to say so and wrap up:
	//
	//	consecutive 0	-> ALLOW   (a clean completion is a real stop)
	//	1 .. Warn-1	-> NUDGE   (gentle "pick an allowed alternative")
	//	Warn .. Final-1	-> WARN    (suggest a change of tack; name the clean wrap-up)
	//	Final .. Max	-> FINAL   (last attempts; act now or note the blocker and wrap up)
	//	> Max		-> GIVE-UP (allow the stop, visibly, so it is no longer invisible)
	guardStopHookDefaultWarn  = 3
	guardStopHookDefaultFinal = 7
	guardStopHookDefaultMax   = 9

	// guardStopHookSameStopDefault is the same-issue give-up depth: the number of consecutive
	// deny-all turns proposing the IDENTICAL refused action (same tool+reason) that ends the
	// session. Unlike the blind ladder above, the same-issue ladder has NO ceiling on variety —
	// a session hitting a fresh block each turn pins the same-issue gauge at 1 and rides the NUDGE
	// rung forever. Only a genuine repeat of the same refusal climbs to this depth and stops.
	guardStopHookSameStopDefault = 6

	// guardStopHookToolFeedbackMaxDefault bounds the retryable tool-feedback continue path (#A6).
	// It is DELIBERATELY separate from — and far more generous than — the deny-all ceiling: a
	// malformed/misrouted call is model-fixable, so a handful of repair turns must never be capped
	// by the deny-all --max (a session may legitimately fix a call over several turns). But an
	// UNBOUNDED continue lets a model that emits a malformed call every turn hold the turn open
	// until the harness's own stop_hook_active cutoff. Past this many CONSECUTIVE tool-feedback
	// turns the hook stands down and ALLOWS the stop, visibly, exactly as the deny-all ladder gives
	// up past its bound. 25 straight malformed calls with none landing is unambiguously stuck.
	guardStopHookToolFeedbackMaxDefault = 25
)

// guardStopHookStage is the rung of the graduated deny-all back-off ladder the current
// consecutive count falls in. It drives BOTH the decision (allow / continue / give-up) and the
// firmness of the guidance fed back to the model.
type guardStopHookStage int

const (
	guardStopHookAllow  guardStopHookStage = iota // consecutive 0: a clean completion — allow the stop
	guardStopHookNudge                            // 1 .. Warn-1: gentle "pick an allowed alternative"
	guardStopHookWarn                             // Warn .. Final-1: force a relevance decision
	guardStopHookFinal                            // Final .. Max: last attempts before give-up
	guardStopHookGiveUp                           // > Max: bounded stand-down — allow the stop, visibly
)

func (s guardStopHookStage) String() string {
	switch s {
	case guardStopHookAllow:
		return "allow"
	case guardStopHookNudge:
		return "nudge"
	case guardStopHookWarn:
		return "warn"
	case guardStopHookFinal:
		return "final"
	case guardStopHookGiveUp:
		return "give-up"
	default:
		return "unknown"
	}
}

// guardStopHookContinueReason is the NUDGE-rung instruction fed back to the model (via the Stop
// hook's exit-2 stderr) when fak first resumes the agent past a deny-all stop. The per-call
// refusal detail is already in the transcript (the in-band `[fak] refused …` note on the ended
// turn); this is the gentle nudge to act on it rather than stop. Later rungs of the ladder
// (guardStopHookStageMessage) escalate the firmness and name the sanctioned clean exit.
const guardStopHookContinueReason = "fak guard: heads-up — your previous turn ended before acting because its tool call(s) are waiting on a shape the capability floor can admit (reported upstream as end_turn). You can continue right now. The in-band `[fak]` note on that turn labels each call as `Tool (REASON/DISPOSITION)` — let that reason point the way. Most reasons just invite a small RESHAPING the floor will welcome: for MISROUTE, reach for the tool or argument shape it expects; for SELF_MODIFY, the floor is protecting a guarded write target (VERSION, .dos/, internal/…), so aim the write at an unguarded path, split a compound command to isolate it, or leave the guarded part out; for LEASE_HELD, another agent holds that tree, so narrow your paths or pick up other work; for a preview-confirm pause, re-send the same call with the confirm key it asked for. A few reasons are protected on purpose — a TERMINAL disposition (e.g. SECRET_EXFIL, TRUST_VIOLATION) is a deliberate boundary, so the clean win there is a different task. Choose an ALLOWED alternative and keep the work moving; and if a protected boundary is all that stands between you and the last step, that is a fine, complete place to stop — note it in one line (`no allowed path: <reason>`) and finish cleanly."

func guardStopHookToolFeedbackMessage(consecutive int) string {
	return fmt.Sprintf("fak guard: the previous %d turn(s) ended after retryable tool-call feedback, not a session stop. The proposed tool call(s) were just malformed or otherwise model-fixable, so fak returned per-call feedback and kept the task alive. Fix the JSON/arguments/tool shape and continue — this is a routine retry, so keep going.", consecutive)
}

// guardStopHookToolFeedbackGiveUpMessage is the operator-facing stand-down line (exit 0, NOT fed
// to the model) printed when the retryable tool-feedback continue passes its bound (#A6): the
// model emitted malformed/misrouted calls for more consecutive turns than the ceiling without
// landing one, so the hook stops auto-continuing and allows the stop rather than looping the turn
// open until the harness's own cutoff. Mirrors the deny-all give-up: a visible, logged stand-down.
func guardStopHookToolFeedbackGiveUpMessage(consecutive, bound int) string {
	return fmt.Sprintf("fak guard Stop: standing down — %d consecutive tool-feedback turn(s) exceeded the continue bound (%d). The model kept emitting malformed/misrouted tool calls without landing one, so fak is allowing the stop instead of holding the turn open indefinitely. Raise FAK_GUARD_TOOL_FEEDBACK_MAX to extend the bound.", consecutive, bound)
}

func guardStopHookToolFeedbackMaxFromEnv() int {
	return guardStopHookIntFromEnv(guardStopHookEnvToolFeedbackMax, guardStopHookToolFeedbackMaxDefault)
}

// normalizeDenyAllThresholds makes the ladder a TOTAL, deterministic function of its three
// knobs: it clamps any operator/env misconfiguration into the invariant 1 <= warn <= final <=
// max so a bad flag can never invert the ladder or wedge the hook. A non-positive max falls
// back to the default; warn floors at 1; final is pulled into [warn, max].
func normalizeDenyAllThresholds(warnAt, finalAt, maxN int) (int, int, int) {
	if maxN <= 0 {
		maxN = guardStopHookDefaultMax
	}
	if warnAt < 1 {
		warnAt = 1
	}
	if warnAt > maxN {
		warnAt = maxN
	}
	if finalAt < warnAt {
		finalAt = warnAt
	}
	if finalAt > maxN {
		finalAt = maxN
	}
	return warnAt, finalAt, maxN
}

// guardStopHookStageFor maps a consecutive deny-all count onto its ladder rung. Pure + total;
// thresholds are normalized first so the rung order can never invert.
func guardStopHookStageFor(consecutive, warnAt, finalAt, maxN int) guardStopHookStage {
	warnAt, finalAt, maxN = normalizeDenyAllThresholds(warnAt, finalAt, maxN)
	switch {
	case consecutive <= 0:
		return guardStopHookAllow
	case consecutive > maxN:
		return guardStopHookGiveUp
	case consecutive >= finalAt:
		return guardStopHookFinal
	case consecutive >= warnAt:
		return guardStopHookWarn
	default:
		return guardStopHookNudge
	}
}

// normalizeSameStop clamps the same-issue give-up depth into a sane range and derives the
// warn/final rungs from it. The give-up depth is the single knob; warn/final are stop-3 and
// stop-1 so the guidance firms up over the last few identical repeats before the stand-down.
// A give-up depth below 2 (which would stop on the first deny-all, defeating the "keep going"
// intent) falls back to the default. Returns warn <= final < stop, all >= 1.
func normalizeSameStop(stop int) (warnAt, finalAt, stopN int) {
	if stop < 2 {
		stop = guardStopHookSameStopDefault
	}
	finalAt = stop - 1
	warnAt = stop - 3
	if warnAt < 1 {
		warnAt = 1
	}
	if finalAt < warnAt {
		finalAt = warnAt
	}
	return warnAt, finalAt, stop
}

// guardStopHookSameStageFor maps the same-issue consecutive count (identical refused action turn
// after turn) onto its ladder rung. A clean >= ladder that stands down AT the give-up depth —
// distinct from the blind guardStopHookStageFor (whose > maxN semantics + tests stay untouched),
// so the two paths cannot interfere. Because any deny-all turn has same-issue count >= 1 and a
// varied session pins it at 1, a varied session sits at NUDGE forever and is never given up.
func guardStopHookSameStageFor(sameConsecutive, stop int) guardStopHookStage {
	warnAt, finalAt, stopN := normalizeSameStop(stop)
	switch {
	case sameConsecutive <= 0:
		return guardStopHookAllow
	case sameConsecutive >= stopN:
		return guardStopHookGiveUp
	case sameConsecutive >= finalAt:
		return guardStopHookFinal
	case sameConsecutive >= warnAt:
		return guardStopHookWarn
	default:
		return guardStopHookNudge
	}
}

// guardStopHookSameDecision is the same-issue twin of guardStopHookDecision: given the gateway's
// same-issue consecutive count and the give-up depth, return the exit code, whether it WOULD
// block, and the rung. Mode gating is identical to the blind path (nudge/warn/final block the
// stop; allow and give-up let it through; shadow always allows but reports the would-be block).
func guardStopHookSameDecision(sameConsecutive, stop int, mode string) (exit int, block bool, stage guardStopHookStage) {
	stage = guardStopHookSameStageFor(sameConsecutive, stop)
	if mode == guardPreCompactModeOff {
		return 0, false, stage
	}
	block = stage == guardStopHookNudge || stage == guardStopHookWarn || stage == guardStopHookFinal
	if mode == guardPreCompactModeShadow {
		return 0, block, stage
	}
	if block {
		return 2, true, stage
	}
	return 0, false, stage
}

// guardStopHookStageMessage is the exact stderr text fed back to the model when the hook holds
// the stop (exit 2) at a continue rung. Each rung is firmer than the last, and the WARN/FINAL
// rungs name the clean wrap-up: note the blocker on one line (`no allowed path: <reason>`) and
// stop. (A pure-text turn resets the gateway's consecutive gauge to 0, so that note genuinely
// lets the next stop through — the agent's own choice, not a fak-forced halt.)
func guardStopHookStageMessage(stage guardStopHookStage, consecutive, maxN int) string {
	switch stage {
	case guardStopHookWarn:
		return fmt.Sprintf("fak guard: the last %d turns each closed while the capability floor was still waiting for a shape it can admit, so the same approach keeps returning. Good moment to try a fresh angle: if the remaining work is reachable under this floor, take a different allowed action now — a different tool, a narrower command, or a path the floor welcomes. If a protected boundary is all that's left, note it on one line (`no allowed path: <reason>`) and finish cleanly — that is a good, complete outcome. (Auto-continue %d of %d before fak lets the turn end.)", consecutive, consecutive, maxN)
	case guardStopHookFinal:
		return fmt.Sprintf("fak guard: last auto-continue (%d of %d). After %d turns still waiting on a shape the floor can admit, fak will let the session wrap up. If there is an allowed way forward, take it on this turn; otherwise note what's protecting the last step on one line (`no allowed path: <reason>`) and finish cleanly now, so the stop is your own call — a complete, expected ending.", consecutive, maxN, maxN)
	default:
		return guardStopHookContinueReason
	}
}

// guardStopHookGiveUpMessage is the OPERATOR-facing line printed when the hook gives up and
// allows the stop (exit 0, so it is NOT fed to the model). It makes the previously-invisible
// give-up legible: the residual false-stop the audit named.
func guardStopHookGiveUpMessage(consecutive, maxN int) string {
	return fmt.Sprintf("fak guard Stop: standing down after %d consecutive deny-all turns (every proposed tool call set aside; %d > max %d) — allowing the stop so the loop cannot spin. To keep the agent moving, inspect why the floor sets everything aside (fak guard --dump-policy) or raise --deny-all-continue=max=N; --deny-all-continue off disables this layer.", consecutive, consecutive, maxN)
}

// guardStopHookSameStageMessage is the same-issue twin of guardStopHookStageMessage: the exact
// stderr text fed back to the model (exit 2) at a same-issue continue rung. Unlike the blind
// message it NAMES the repeat — the point is that the model keeps proposing the IDENTICAL refused
// action, so the guidance is "a different angle," not "try again." The NUDGE rung (a shallow or
// just-changed issue) still uses the gentle generic continue reason.
func guardStopHookSameStageMessage(stage guardStopHookStage, sameConsecutive, stop int) string {
	switch stage {
	case guardStopHookWarn:
		return fmt.Sprintf("fak guard: you have now ended %d turns in a row proposing the IDENTICAL refused action — the capability floor is setting aside the very same tool call, for the same reason, each time. Repeating it will not change the verdict. Try a genuinely different angle now: a different tool, a narrower command, or a path the floor welcomes. The in-band `[fak]` note labels the block as `Tool (REASON/DISPOSITION)` — let that reason point the way. If a protected boundary is all that's left, note it on one line (`no allowed path: <reason>`) and finish cleanly — a complete, expected outcome. (Auto-continue %d of %d identical repeats before fak lets the turn end.)", sameConsecutive, sameConsecutive, stop)
	case guardStopHookFinal:
		return fmt.Sprintf("fak guard: last auto-continue (%d of %d). You have proposed the IDENTICAL refused action %d turns running; one more and fak will let the session wrap up. This is a genuine repeat, not exploration — if there is an allowed way forward, take a DIFFERENT action on this turn; otherwise note what is protecting the last step on one line (`no allowed path: <reason>`) and finish cleanly now, so the stop is your own call.", sameConsecutive, stop, sameConsecutive)
	default:
		return guardStopHookContinueReason
	}
}

// guardStopHookSameGiveUpMessage is the OPERATOR-facing line (exit 0, not fed to the model) when
// the hook stands down on a true repeated same issue. It makes explicit that this is a genuine
// spin on ONE refusal — not the old blind count — and that a varied session is never stopped here.
func guardStopHookSameGiveUpMessage(sameConsecutive, stop int) string {
	return fmt.Sprintf("fak guard Stop: standing down after %d turns proposing the IDENTICAL refused action (same tool + same reason; %d >= same-issue give-up %d) — a genuine repeated same issue, not exploration, so allowing the stop keeps the loop from spinning. A session hitting a FRESH block each turn is never stopped here. To keep the agent moving, inspect why the floor sets that same call aside (fak guard --dump-policy) or raise --deny-all-continue=same-stop=N; --deny-all-continue off disables this layer.", sameConsecutive, sameConsecutive, stop)
}

type guardStopHookInstall struct {
	Applied      bool
	Mode         string
	SettingsPath string
	MetricsURL   string
	WarnAt       int
	FinalAt      int
	Max          int
	SameStop     int    // the same-issue give-up depth pinned into the hook (identical refused action, turn after turn)
	StopsLedger  string // absolute path of the typed stop-decision ledger wired into the hook (empty when unresolved)
	Reason       string
}

type guardTaskHandoffConfig struct {
	Mode string
	File string
	Repo string
	Live bool
}

func cmdGuardStopHook(argv []string) {
	os.Exit(runGuardStopHook(os.Stderr, os.Stdin, argv))
}

// runGuardStopHook is the Stop-hook actuator. It returns the process exit code: 2 to BLOCK the
// stop (Claude Code continues the agent with the stderr text as guidance), 0 to allow it. It
// fails OPEN — any bad args, unknown mode, missing metrics URL, or unreachable gateway returns
// 0 so the hook can never wedge the agent — exactly the posture guard-precompact takes.
func runGuardStopHook(stderr io.Writer, stdin io.Reader, argv []string) (exit int) {
	// Capture WIP before any Stop decision; best-effort and tree-preserving.
	_ = runWipAutoCheckpoint(io.Discard, io.Discard, []string{"--reason", "stop"})
	// One typed, countable row per invocation (see guard_stops.go): the disposition is
	// classified at each return below, and a single defer stamps the exit/kind and appends
	// the row. FAIL-OPEN and gated on the wired ledger env, so recording is a no-op for any
	// hook test that does not set it and never changes this hook's decision. The default
	// disposition is the conservative fail-open-bad-args reading until a return reclassifies.
	rec := guardStopRecord{Schema: guardStopRecordSchema, Ts: time.Now().UTC().Format(time.RFC3339), Disposition: string(stopDispFailOpenBadArgs)}
	var nextPayload strings.Builder
	decisionStderr := io.MultiWriter(stderr, &nextPayload)
	defer func() {
		rec.Exit = exit
		rec.Blocked = exit == 2
		if rec.Kind == "" {
			rec.Kind = string(guardStopDispositionKind(guardStopDisposition(rec.Disposition)))
		}
		kind := guardStopDispositionKind(guardStopDisposition(rec.Disposition))
		move := sessionctl.Move{
			Kind: sessionctl.MoveHalt, Render: sessionctl.RenderStop,
			Session: sessionctl.SessionInteractive, Gate: rec.Disposition,
			Source: "guard-stophook", Payload: strings.TrimSpace(nextPayload.String()),
		}
		result := sessionctl.ApplyResult{Applied: true}
		if exit == 2 || kind == stopKindShadow {
			move.Kind, move.Render = sessionctl.MoveContinue, sessionctl.RenderReopen
		}
		if kind == stopKindShadow {
			move.Shadow = true
			result = sessionctl.ApplyResult{Applied: false, Refusal: "shadow: continuation observed but stop allowed"}
		} else if kind == stopKindFailOpen {
			result = sessionctl.ApplyResult{Applied: false, Refusal: "fail-open: stop allowed because the gate could not decide"}
		}
		if next, err := sessionctl.WitnessMove(move, result); err == nil {
			rec.Next = &next
		} else {
			fmt.Fprintf(stderr, "fak guard Stop: next witness skipped (fail-open): %v\n", err)
		}
		emitGuardStopRecord(stderr, rec)
	}()
	stderr = decisionStderr
	fs := flag.NewFlagSet("guard-stophook", flag.ContinueOnError)
	fs.SetOutput(stderr)
	modeFlag := fs.String("mode", os.Getenv(guardStopHookEnvMode), "off|shadow|enforce")
	metricsURLFlag := fs.String("metrics-url", os.Getenv(guardStopHookEnvMetricsURL), "gateway /metrics URL")
	maxFlag := fs.Int("max", guardStopHookMaxFromEnv(), "hard give-up: max consecutive deny-all turns to auto-continue past before letting the turn end")
	warnFlag := fs.Int("warn", guardStopHookWarnFromEnv(), "escalate the continue guidance to a relevance-decision warning at this consecutive deny-all depth")
	finalFlag := fs.Int("final", guardStopHookFinalFromEnv(), "escalate the continue guidance to a final warning at this consecutive deny-all depth")
	sameStopFlag := fs.Int("same-stop", guardStopHookSameStopFromEnv(), "hard give-up depth on the SAME-ISSUE path: end the session after this many consecutive deny-all turns proposing the IDENTICAL refused action (same tool+reason). A varied session never reaches it")
	operatorDirectedFlag := fs.String("operator-directed", os.Getenv(guardStopHookOperatorDirectedEnvMode), "headless 'stopped to ask a human' gate: off|shadow|warn|enforce")
	operatorQuestionFlag := fs.String("operator-question", os.Getenv(guardStopHookOperatorQuestionEnvMode), "headless evidence-first operator-question gate (native ExitPlanMode/AskUserQuestion adjudication): off|shadow|warn|enforce")
	hardwareGateFlag := fs.String("hardware-gate", os.Getenv(guardStopHookHardwareGateEnvMode), "headless 'no local hardware' redirect gate: off|shadow|warn|enforce")
	handoffModeFlag := fs.String("task-handoff-mode", os.Getenv(guardTaskHandoffEnvMode), "completion handoff gate: off|shadow|enforce")
	witnessedDoneFlag := fs.String("witnessed-done", os.Getenv(guardWitnessedDoneModeEnv), "off|shadow|enforce (require narrated completion to name a witnessed stamped commit)")
	handoffFileFlag := fs.String("task-handoff-file", os.Getenv(guardTaskHandoffEnvFile), "path to fak.task-handoff.v1 JSON the agent must write before a clean stop")
	handoffRepoFlag := fs.String("task-handoff-repo", os.Getenv(guardTaskHandoffEnvRepo), "owner/repo passed to fak task handoff --live")
	handoffLiveFlag := fs.Bool("task-handoff-live", guardTaskHandoffLiveFromEnv(), "when true, sync valid next steps to GitHub with fak task handoff --live")
	timeout := fs.Duration("timeout", 500*time.Millisecond, "maximum time to wait for the gateway gauge")
	if err := fs.Parse(argv); err != nil {
		fmt.Fprintf(stderr, "fak guard Stop: allowing stop; bad hook args: %v\n", err)
		return 0
	}
	// Read the hook's stdin payload ONCE (it is not rewindable): stop_hook_active is the
	// defensive shadow-log signal below, session_id stamps the trajctl rows. A parse miss
	// never changes any decision.
	payload := readHookStdin(stdin)
	active := parseStopHookActive(payload)
	// Stamp the stop row's session/re-fire/transcript context from the SAME payload bytes.
	// The transcript read is a bounded, fail-open tail read (guard_stops.go) that lets a
	// give-up be told apart from a graceful "no allowed path:" wrap-up; it never gates.
	rec.Session = parseHookSessionID(payload)
	rec.StopHookActive = active
	transcriptPath := parseHookTranscriptPath(payload)
	if result := refusalRestatementCheck(refusalRestatementInput{
		SessionID:      rec.Session,
		TranscriptPath: transcriptPath,
		StatePath:      refusalRestatementStatePath(rec.Session),
		Head:           guardStopGitHead(),
	}); result.Blocked {
		fmt.Fprintf(stderr, "fak guard stop-hook: %s count=%d; allowing terminal blocked-state without another continuation\n", result.Reason, result.Count)
		rec.Disposition = string(stopDispSameIssueGiveUp)
		rec.Kind = string(stopKindStandDown)
		rec.Stage = "refusal_restatement"
		rec.Signal = "same-issue"
		rec.Depth = result.Count
		rec.Bound = refusalRestatementCap
		rec.Note = result.Reason
		rec.Exit = 0
		rec.Blocked = false
		rec.Transcript = readGuardStopTranscript(transcriptPath)
		emitGuardStopRecord(stderr, rec)
		return 0
	}
	rec.Transcript = readGuardStopTranscript(transcriptPath)
	// #2539: score-at-turn-end — the curve gains a point every turn. Runs the cheap scorer
	// set for the session's open objectives, bounded and fail-open, gated on the guard-wired
	// ledger env. Deliberately BEFORE the deny-all mode gate: sampling cadence is independent
	// of whether auto-continue is off, and it can never change this hook's exit code.
	scoreTurnEndFailOpen(stderr, trajctl.Stamp{SessionID: rec.Session}, time.Now().UnixMilli())
	// #3669: live detour detection — fold the finished turn's tool stream and open/close
	// budgeted detour children on the session ledger (internal/trajctl's detector, #2546).
	// Same guard-wired gate, bounded, and fail-open posture as the sampling above, and
	// likewise BEFORE the deny-all mode gate: it never changes this hook's exit code.
	detectDetoursFailOpen(stderr, transcriptPath, trajctl.Stamp{SessionID: rec.Session}, time.Now().UnixMilli())
	// Managed-context step-advice carryover: capture the last live step-advice decision to
	// a durable per-session stamp while the trace is still alive, so a resume can read what
	// the window pressure WAS (internal/stepbaton, internal/stepbatoncapture). Same
	// guard-wired gate and fail-open/bounded posture as the sampling above, and likewise
	// BEFORE the deny-all mode gate: the carryover is independent of whether auto-continue
	// is off, and can never change this hook's exit code.
	stampStepAdviceFailOpen(stderr, rec.Session, os.Getenv("ANTHROPIC_BASE_URL"))
	// #4118: a Stop may be the session's last breath before a `claude --resume` relaunch, so
	// record the live remaining budget into the transcript-UUID carry store here — while the
	// process still holds the spent-down State. Placed with the other pre-gate fail-open
	// side-effects: independent of the deny-all decision below and never changes the exit code.
	writeDriveCarryFailOpen(time.Now())
	mode, err := normalizeGuardStopHookMode(*modeFlag)
	if err != nil {
		rec.Disposition = string(stopDispFailOpenBadMode)
		fmt.Fprintf(stderr, "fak guard Stop: allowing stop; %v\n", err)
		return 0
	}
	rec.Mode = mode
	if mode == guardPreCompactModeOff {
		rec.Disposition = string(stopDispModeOff)
		return 0
	}
	metricsURL := strings.TrimSpace(*metricsURLFlag)
	if metricsURL == "" {
		metricsURL = guardPreCompactMetricsURLFromBase(os.Getenv("ANTHROPIC_BASE_URL"))
	}
	signals, source, err := fetchGuardStopHookSignalsPreferred(context.Background(), metricsURL, *timeout)
	if err != nil {
		if metricsURL == "" {
			rec.Disposition = string(stopDispFailOpenNoMetricsURL)
		} else {
			rec.Disposition = string(stopDispFailOpenGaugeUnavailable)
		}
		rec.Note = err.Error()
		fmt.Fprintf(stderr, "fak guard Stop: allowing stop; lifecycle signals unavailable: %v\n", err)
		return 0
	}
	rec.Note = "signals=" + source
	rec.DenyAllConsecutive = signals.DenyAllConsecutive
	rec.DenyAllSameConsecutive = signals.DenyAllSameConsecutive
	rec.ToolFeedbackConsecutive = signals.ToolFeedbackConsecutive
	consecutive := signals.DenyAllConsecutive
	feedbackConsecutive := signals.ToolFeedbackConsecutive
	warnAt, finalAt, maxN := normalizeDenyAllThresholds(*warnFlag, *finalFlag, *maxFlag)
	if consecutive <= 0 && feedbackConsecutive > 0 {
		rec.Signal = "tool-feedback"
		rec.Depth = feedbackConsecutive
		feedbackMax := guardStopHookToolFeedbackMaxFromEnv()
		rec.Bound = feedbackMax
		// Bounded give-up (#A6). The tool-feedback continue is DELIBERATELY decoupled from the
		// deny-all --max (a malformed/misrouted call is model-fixable, so a few repair turns must
		// not be capped by the deny-all ceiling), but it is no longer UNBOUNDED. Past its own
		// generous ceiling the hook stands down and ALLOWS the stop — visibly — so a model stuck
		// emitting malformed calls cannot hold the turn open every turn until the harness's own
		// stop_hook_active cutoff.
		if feedbackConsecutive > feedbackMax {
			if mode == guardPreCompactModeShadow {
				rec.Disposition = string(stopDispShadow)
				fmt.Fprintf(stderr, "fak guard Stop: shadow would stand down on tool-feedback past bound (tool_feedback_consecutive=%d bound=%d stop_hook_active=%v)\n", feedbackConsecutive, feedbackMax, active)
				return 0
			}
			rec.Disposition = string(stopDispToolFeedbackGiveUp)
			fmt.Fprintln(stderr, guardStopHookToolFeedbackGiveUpMessage(feedbackConsecutive, feedbackMax))
			return 0
		}
		if mode == guardPreCompactModeShadow {
			rec.Disposition = string(stopDispShadow)
			fmt.Fprintf(stderr, "fak guard Stop: shadow would auto-continue tool-feedback turn(s) (tool_feedback_consecutive=%d stop_hook_active=%v)\n", feedbackConsecutive, active)
			return 0
		}
		rec.Disposition = string(stopDispToolFeedbackContinue)
		fmt.Fprintln(stderr, guardStopHookToolFeedbackMessage(feedbackConsecutive))
		return 2
	}
	// Key the give-up on the SAME-ISSUE gauge when the gateway emits it (a current gateway): only a
	// true repeated same refusal (identical tool+reason turn after turn) ever ends the session, and
	// a varied deny-all session is never stopped. An older gateway that omits the gauge falls back
	// to the legacy blind ladder below, so its behavior is byte-for-byte unchanged.
	useSame := signals.DenyAllSameConsecutiveSeen
	_, _, sameStop := normalizeSameStop(*sameStopFlag)
	// exit is the function's named return (stamped into the stop row by the defer); block and
	// stage are local to this decision. depth/bound select the numbers the shadow log, exit-2
	// guidance, and operator give-up line all speak, so a single decision path drives every
	// message.
	var (
		block bool
		stage guardStopHookStage
	)
	depth, bound := consecutive, maxN
	if useSame {
		depth, bound = signals.DenyAllSameConsecutive, sameStop
		exit, block, stage = guardStopHookSameDecision(signals.DenyAllSameConsecutive, *sameStopFlag, mode)
	} else {
		exit, block, stage = guardStopHookDecision(consecutive, warnAt, finalAt, maxN, mode)
	}
	// Signal names WHICH path actually drove THIS decision, so a folded ledger reads true (#A7).
	// It is NOT merely whether the gateway emitted the same-issue gauge (useSame): a clean
	// completion (depth 0) is "clean" even on a gauge-emitting gateway; only a real repeat
	// (depth > 0) keyed on the same-issue gauge is "same-issue"; a real repeat riding the legacy
	// blind ladder is "blind". depth already disambiguates, so this only corrects the label.
	signalName := "blind"
	switch {
	case depth <= 0:
		signalName = "clean"
	case useSame:
		signalName = "same-issue"
	}
	rec.Signal = signalName
	rec.Stage = stage.String()
	rec.Depth = depth
	rec.Bound = bound
	if mode == guardPreCompactModeShadow {
		rec.Disposition = string(stopDispShadow)
		action := "allow stop"
		switch {
		case block:
			action = "auto-continue (block stop)"
		case stage == guardStopHookGiveUp:
			action = "give up and allow stop"
		}
		fmt.Fprintf(stderr, "fak guard Stop: shadow would %s (stage=%s signal=%s deny_all_consecutive=%d deny_all_same_consecutive=%d same_stop=%d warn=%d final=%d max=%d stop_hook_active=%v)\n", action, stage, signalName, consecutive, signals.DenyAllSameConsecutive, sameStop, warnAt, finalAt, maxN, active)
		return 0
	}
	if exit == 2 {
		// Exit 2 blocks the stop; stderr is shown to Claude as the reason to continue. The text
		// escalates with the ladder rung (nudge -> warn -> final); the same-issue path names the repeat.
		if useSame {
			rec.Disposition = string(stopDispSameIssueContinue)
			fmt.Fprintln(stderr, guardStopHookSameStageMessage(stage, depth, bound))
		} else {
			rec.Disposition = string(stopDispDenyAllContinue)
			fmt.Fprintln(stderr, guardStopHookStageMessage(stage, depth, bound))
		}
		return 2
	}
	// Allowed (exit 0). A clean completion (stage allow) is silent; a bounded stand-down is the
	// residual false-stop — make it operator-visible (it is NOT fed to the model).
	if stage == guardStopHookGiveUp {
		recordGuardStopHookGiveUp(stderr, &rec, useSame, depth, bound)
		return 0
	}
	if stage == guardStopHookAllow {
		// Advisory-only (never blocks the stop): a clean completion that used ZERO fak verbs is
		// the unused-substrate pathology — fak present as a passive guard but never engaged as a
		// substrate (#3093). Emit before the handoff gate; the stop is still allowed regardless.
		emitUnusedSubstrateAdvisory(stderr, signals)
		// Advisory-only (never blocks the stop): a clean completion whose modified trees intersect
		// a LIVE known-bad signature — surface the fleet's recorded pre-existing blocker so the
		// agent recognises the red as not-its-own and does not loop re-fixing it. Fails open silent.
		emitKnownFrictionAdvisory(stderr, transcriptPath)
		// Hardware-gate rung: a clean stop whose final turn declared a LOCAL-hardware blocker as
		// terminal ("no GPU here", "can't run without CUDA"). On a headless run enforce BLOCKS it and
		// feeds the sanctioned-compute-node redirect back so the agent dispatches to the fleet instead
		// of stopping at the local boundary. Fires BEFORE the operator-directed rung because it is the
		// more specific misroute — a concrete "wrong machine" error with a fixed remedy — so its
		// redirect wins over the generic "act on your own question" guidance.
		hgExit, hgDisp, hgFired := runGuardHardwareGateGate(stderr, *hardwareGateFlag, transcriptPath)
		if hgFired && hgExit == 2 {
			rec.Disposition = string(hgDisp)
			return 2
		}
		// Operator-directed rung: a clean stop whose final turn asked a human. On a headless run
		// enforce BLOCKS it (feed the choicetriage remediation back so the agent acts instead of
		// asking) — precedes the handoff gate, mirroring deny-all-precedes-handoff, so the more
		// specific "act on your own question" guidance wins over the generic handoff demand.
		odExit, odDisp, odFired := runGuardOperatorDirectedGate(stderr, *operatorDirectedFlag, rec.Transcript)
		if odFired {
			routeGuardOperatorEscalationFailOpen(rec.Session, odDisp, rec.Transcript)
		}
		if odFired && odExit == 2 {
			rec.Disposition = string(odDisp)
			return 2
		}
		// Evidence-first operator-question rung: consumes native tool inputs from the
		// transcript and runs after the linguistic operator-directed sensor. It is
		// harness-agnostic and carries its OWN dial (--operator-question /
		// FAK_GUARD_OPERATOR_QUESTION_MODE), which defaults at install to inherit the resolved
		// operator-directed posture (and thus its operator-absence cap) but can be tuned apart.
		var oqAnswer string
		oqExit, oqDisp, oqHarness, oqFired := runGuardOperatorQuestionGate(decisionStderr, *operatorQuestionFlag, transcriptPath, rec.Session, &oqAnswer)
		if oqFired {
			rec.Disposition = string(oqDisp)
			rec.Kind = "operator-question:" + oqHarness
			rec.Signal = string(oqDisp)
		}
		if oqFired && oqExit == 2 {
			if oqDisp == stopDispOperatorQuestionResolved && strings.TrimSpace(oqAnswer) != "" {
				nextPayload.Reset()
				_, _ = fmt.Fprintf(decisionStderr, "Resolved operator answer: %s", strings.TrimSpace(oqAnswer))
			}
			return 2
		}
		wdExit, wdDisp, wdReason, wdFired := runGuardWitnessedDoneGate(stderr, *witnessedDoneFlag, transcriptPath, repoRoot(), rec.Session, guardWitnessedDoneMaxFromEnv())
		if wdFired {
			rec.Disposition = string(wdDisp)
			rec.Kind = string(guardWitnessedDoneKind(wdDisp))
			rec.Signal = wdReason
		}
		if wdFired && wdExit == 2 {
			return 2
		}
		handoffExit, handoffDisp := runGuardTaskHandoffGate(stderr, active, guardTaskHandoffConfig{
			Mode: *handoffModeFlag,
			File: *handoffFileFlag,
			Repo: *handoffRepoFlag,
			Live: *handoffLiveFlag,
		})
		switch {
		case handoffDisp != "":
			// The handoff gate reached its own terminal disposition — a block (exit 2) or a
			// stop_hook_active stand-down (exit 0, #A2). Record it verbatim rather than folding a
			// give-up into a bare clean stop.
			rec.Disposition = string(handoffDisp)
		case hgFired:
			// The hardware-gate rung fired but allowed the stop (warn/shadow) and the handoff gate
			// did not itself block: record the hardware-gate disposition, not a bare clean stop. It
			// precedes odFired for the same reason it fires first — the more specific signal wins.
			rec.Disposition = string(hgDisp)
		case wdFired:
			// The witnessed-done rung fired but allowed the stop (confirmed, shadow, or bounded
			// stand-down) and handoff did not override it: preserve its typed disposition.
			rec.Disposition = string(wdDisp)
		case odFired:
			// The gate fired but allowed the stop (warn/shadow/escalate) and the handoff gate did
			// not itself block: record the operator-directed disposition, not a bare clean stop.
			rec.Disposition = string(odDisp)
		case rec.Transcript != nil && rec.Transcript.NotedNoAllowedPath:
			rec.Disposition = string(stopDispCleanWrapup)
		default:
			rec.Disposition = string(stopDispCleanCompletion)
		}
		return handoffExit
	}
	rec.Disposition = string(stopDispCleanCompletion)
	return 0
}

// recordGuardStopHookGiveUp stamps the stop row's give-up disposition and prints the
// operator-facing stand-down line (exit 0 path, so it is NOT fed to the model). A give-up
// whose transcript shows the agent already wrote its sanctioned "no allowed path:" wrap-up
// is a graceful conclusion, not a bare stand-down — record it as such (the operator
// messages below are unchanged either way).
func recordGuardStopHookGiveUp(stderr io.Writer, rec *guardStopRecord, useSame bool, depth, bound int) {
	switch {
	case rec.Transcript != nil && rec.Transcript.NotedNoAllowedPath:
		rec.Disposition = string(stopDispCleanWrapup)
	case useSame:
		rec.Disposition = string(stopDispSameIssueGiveUp)
	default:
		rec.Disposition = string(stopDispBlindGiveUp)
	}
	if useSame {
		fmt.Fprintln(stderr, guardStopHookSameGiveUpMessage(depth, bound))
	} else {
		fmt.Fprintln(stderr, guardStopHookGiveUpMessage(depth, bound))
	}
}

// emitUnusedSubstrateAdvisory prints a one-line advisory (never blocks the stop) when a
// session reaches a clean completion having called ZERO fak verbs — the #3093 pathology
// where fak is present but inert. It fires only when the gateway actually reported the
// counter (FakVerbCallsSeen), so an older gateway that omits the metric stays silent, and
// only in shadow mode (the default; `off` suppresses it). There is deliberately no enforce
// rung: a run legitimately may not need fak, so this is visibility, not a gate.
func emitUnusedSubstrateAdvisory(stderr io.Writer, signals guardStopHookSignals) {
	mode := strings.ToLower(strings.TrimSpace(os.Getenv(guardStopHookUnusedEnvMode)))
	if mode == "" {
		mode = guardPreCompactModeShadow // default: advise
	}
	if mode == guardPreCompactModeOff {
		return
	}
	if !signals.FakVerbCallsSeen || signals.FakVerbCalls > 0 {
		return // metric absent, or fak was used — nothing to advise.
	}
	fmt.Fprintln(stderr,
		"fak guard Stop: heads-up — this session is ending clean having called ZERO fak verbs "+
			"(fak_mcp_verb_calls_total=0). fak was present as a guard but never used as a substrate: "+
			"no fak_capabilities / fak_admit / fak_adjudicate / fak_memory_run. If this run was meant to "+
			"leverage fak, check the MCP server is wired to THIS workspace and reach for the fak verbs "+
			"(fak_capabilities to discover task-scoped tools; fak_admit before a write). Advisory only — the stop is allowed.")
}

// runGuardTaskHandoffGate is the task-handoff Stop rung. It returns (exit, disposition): the
// disposition is non-empty ONLY when the gate reached a terminal decision the caller must record
// verbatim — a block (exit 2, stopDispHandoffBlock) or a stop_hook_active stand-down (exit 0,
// stopDispHandoffGiveUp, #A2). An empty disposition means "no override" (gate off, shadow, or the
// handoff is valid) and the caller classifies the clean stop itself.
//
// The stopHookActive give-up is the bound (#A2): when the harness is ALREADY re-firing this Stop
// hook because we blocked last turn, we have demanded the handoff once and it is still not valid —
// blocking again only spins the harness with no new information, so we stand down and allow the
// stop, visibly, exactly as the deny-all ladder gives up past its bound.
func runGuardTaskHandoffGate(stderr io.Writer, stopHookActive bool, cfg guardTaskHandoffConfig) (int, guardStopDisposition) {
	mode, err := normalizeGuardTaskHandoffMode(cfg.Mode)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard Stop: allowing stop; %v\n", err)
		return 0, ""
	}
	if mode == guardPreCompactModeOff {
		return 0, ""
	}
	file := strings.TrimSpace(cfg.File)
	if file == "" {
		fmt.Fprintln(stderr, "fak guard Stop: allowing stop; task handoff gate enabled but no handoff file configured")
		return 0, ""
	}
	handoff, review, err := readAndReviewGuardTaskHandoff(file)
	if err != nil || !review.OK {
		msg := guardTaskHandoffRequiredMessage(file, review, err, cfg.Live, cfg.Repo)
		if mode == guardPreCompactModeShadow {
			fmt.Fprintf(stderr, "fak guard Stop: shadow would block clean stop for task handoff: %s\n", strings.TrimSpace(msg))
			return 0, ""
		}
		if stopHookActive {
			fmt.Fprintf(stderr, "fak guard Stop: task handoff still missing/invalid after a prior block (stop_hook_active) — standing down and allowing the stop. Write a valid `%s` to `%s` to hand off next steps next time.\n", taskmgr.SchemaHandoff, file)
			return 0, stopDispHandoffGiveUp
		}
		fmt.Fprintln(stderr, msg)
		return 2, stopDispHandoffBlock
	}
	if cfg.Live && len(handoff.NextSteps) > 0 {
		var out, errb bytes.Buffer
		args := []string{"--file", file, "--live", "--json"}
		if repo := strings.TrimSpace(cfg.Repo); repo != "" {
			args = append(args, "--repo", repo)
		}
		code := runTaskHandoff(&out, &errb, args)
		if code != 0 {
			msg := fmt.Sprintf("fak guard Stop: task handoff is valid, but live GitHub issue sync failed (exit %d): %s", code, strings.TrimSpace(errb.String()))
			if mode == guardPreCompactModeShadow {
				fmt.Fprintln(stderr, "fak guard Stop: shadow would block clean stop: "+msg)
				return 0, ""
			}
			if stopHookActive {
				fmt.Fprintln(stderr, msg)
				fmt.Fprintln(stderr, "This Stop hook already blocked once (stop_hook_active) and the live GitHub sync is still failing — standing down and allowing the stop. Use --task-handoff-live=false to require only the validated handoff artifact.")
				return 0, stopDispHandoffGiveUp
			}
			fmt.Fprintln(stderr, msg)
			fmt.Fprintln(stderr, "Fix the handoff or GitHub sync, then stop again; use --task-handoff-live=false to require only the validated handoff artifact.")
			return 2, stopDispHandoffBlock
		}
	}
	return 0, ""
}

func readAndReviewGuardTaskHandoff(file string) (taskmgr.Handoff, taskmgr.HandoffReview, error) {
	b, err := os.ReadFile(file)
	if err != nil {
		return taskmgr.Handoff{}, taskmgr.HandoffReview{}, err
	}
	var h taskmgr.Handoff
	if err := json.Unmarshal(b, &h); err != nil {
		return taskmgr.Handoff{}, taskmgr.HandoffReview{}, err
	}
	return h, taskmgr.ReviewHandoff(h), nil
}

func guardTaskHandoffRequiredMessage(file string, review taskmgr.HandoffReview, err error, live bool, repo string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak guard Stop: task handoff required before a clean stop. Write a valid `%s` JSON record to `%s` and stop again.\n", taskmgr.SchemaHandoff, file)
	if err != nil {
		fmt.Fprintf(&b, "Current handoff read failed: %v\n", err)
	} else if len(review.Reasons) > 0 {
		fmt.Fprintf(&b, "Current handoff was refused: %s\n", strings.Join(review.Reasons, ", "))
	}
	fmt.Fprintf(&b, "Required fields: task.state=`%s`, task.witness.verified_state=`%s`, current_state, and either 1-2 next_steps or no_next_step_reason.\n", taskmgr.StateDone, taskmgr.VerifiedDone)
	fmt.Fprintln(&b, "When follow-up work is reasonable, prefer 1-2 next_steps with stable key/title/body/reason so `fak task handoff` can create or update GitHub issues.")
	if live {
		fmt.Fprintln(&b, "This hook is in live mode: after the JSON validates it will run `fak task handoff --live` before allowing the stop.")
	} else {
		cmd := "fak task handoff --file \"" + file + "\" --live"
		if strings.TrimSpace(repo) != "" {
			cmd += " --repo " + strings.TrimSpace(repo)
		}
		fmt.Fprintf(&b, "To sync issues yourself before stopping, run `%s`; otherwise the validated handoff artifact is the stop witness.\n", cmd)
	}
	fmt.Fprintf(&b, "The path is also exposed to the agent as `$%s`.\n", guardTaskHandoffFileEnv)
	return strings.TrimRight(b.String(), "\n")
}

// guardStopHookDecision is the PURE decision behind the hook: given the gateway's consecutive
// deny-all count, the graduated thresholds, and the mode, return the exit code, whether it
// WOULD block, and the ladder rung (drives the guidance text + the shadow log). Side-effect-free
// so the policy is unit-tested without an HTTP gateway. The continue rungs (nudge/warn/final)
// block the stop (1..max); rung 0 is a clean completion and rung > max is the bounded give-up —
// both ALLOW the stop, so a stuck model cannot loop forever.
func guardStopHookDecision(consecutive, warnAt, finalAt, maxN int, mode string) (exit int, block bool, stage guardStopHookStage) {
	stage = guardStopHookStageFor(consecutive, warnAt, finalAt, maxN)
	if mode == guardPreCompactModeOff {
		return 0, false, stage
	}
	block = stage == guardStopHookNudge || stage == guardStopHookWarn || stage == guardStopHookFinal
	if mode == guardPreCompactModeShadow {
		return 0, block, stage // shadow always allows the stop (exit 0) but reports the would-be block
	}
	if block {
		return 2, true, stage
	}
	return 0, false, stage
}

// readStopHookActive parses the stop_hook_active flag from Claude's Stop-hook stdin JSON. A nil
// reader, an empty body, or a parse miss returns false — it is advisory only.
func readStopHookActive(stdin io.Reader) bool {
	return parseStopHookActive(readHookStdin(stdin))
}

// parseStopHookActive is the bytes form of readStopHookActive, for callers that already
// drained the one-shot hook stdin.
func parseStopHookActive(payload []byte) bool {
	if len(payload) == 0 {
		return false
	}
	var p struct {
		StopHookActive bool `json:"stop_hook_active"`
	}
	if json.Unmarshal(payload, &p) != nil {
		return false
	}
	return p.StopHookActive
}

func normalizeGuardStopHookMode(mode string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", guardPreCompactModeEnforce:
		// Default ENFORCE: the false stop is a real defect, so the fix is on by default. It is
		// bounded (guardStopHookDefaultMax) and fully observable (the deny-all metrics + exit
		// summary), and `--deny-all-continue off` opts out. This differs deliberately from the
		// PreCompact hook's shadow default, whose enforce path can break harness context management.
		return guardPreCompactModeEnforce, nil
	case guardPreCompactModeShadow:
		return guardPreCompactModeShadow, nil
	case guardPreCompactModeOff:
		return guardPreCompactModeOff, nil
	default:
		return "", fmt.Errorf("invalid --deny-all-continue mode %q (want off, shadow, or enforce)", mode)
	}
}

func normalizeGuardTaskHandoffMode(mode string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", guardPreCompactModeOff:
		return guardPreCompactModeOff, nil
	case guardPreCompactModeShadow:
		return guardPreCompactModeShadow, nil
	case guardPreCompactModeEnforce:
		return guardPreCompactModeEnforce, nil
	default:
		return "", fmt.Errorf("invalid --task-handoff mode %q (want off, shadow, or enforce)", mode)
	}
}

func guardStopHookMaxFromEnv() int {
	return guardStopHookIntFromEnv(guardStopHookEnvMax, guardStopHookDefaultMax)
}

func guardStopHookWarnFromEnv() int {
	return guardStopHookIntFromEnv(guardStopHookEnvWarn, guardStopHookDefaultWarn)
}

func guardStopHookFinalFromEnv() int {
	return guardStopHookIntFromEnv(guardStopHookEnvFinal, guardStopHookDefaultFinal)
}

func guardStopHookSameStopFromEnv() int {
	return guardStopHookIntFromEnv(guardStopHookEnvSameStop, guardStopHookSameStopDefault)
}

// guardStopHookIntFromEnv reads a positive int env override, falling back to def on any unset,
// blank, unparseable, or non-positive value (normalization clamps the rest).
func guardStopHookIntFromEnv(name string, def int) int {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return def
}

func guardTaskHandoffLiveFromEnv() bool {
	v := strings.TrimSpace(os.Getenv(guardTaskHandoffEnvLive))
	return v == "1" || strings.EqualFold(v, "true") || strings.EqualFold(v, "yes")
}

func guardTaskHandoffEnv(cfg guardTaskHandoffConfig) [][2]string {
	mode, err := normalizeGuardTaskHandoffMode(cfg.Mode)
	if err != nil || mode == guardPreCompactModeOff {
		return nil
	}
	file := strings.TrimSpace(cfg.File)
	if file == "" {
		return nil
	}
	env := [][2]string{
		{guardTaskHandoffEnvMode, mode},
		{guardTaskHandoffEnvFile, file},
		{guardTaskHandoffFileEnv, file},
	}
	if repo := strings.TrimSpace(cfg.Repo); repo != "" {
		env = append(env, [2]string{guardTaskHandoffEnvRepo, repo})
	}
	if cfg.Live {
		env = append(env, [2]string{guardTaskHandoffEnvLive, "1"})
	}
	return env
}

func guardTaskHandoffConfigOrZero(configs []guardTaskHandoffConfig) guardTaskHandoffConfig {
	if len(configs) == 0 {
		return guardTaskHandoffConfig{}
	}
	return configs[0]
}

// installGuardStopHook installs the Claude Code Stop hook for a guard session. When the
// PreCompact hook already wrote a --settings file (existingSettingsPath non-empty), the Stop
// hook is MERGED into it so a single --settings carries both (--settings is a single-value flag;
// injecting it twice clobbers rather than merges). Otherwise it writes its own settings file and
// injects --settings. Off mode or a non-claude child is a no-op (command returned unchanged).
func installGuardStopHook(command []string, mode, gwURL, existingSettingsPath string, warnAt, finalAt, maxN, sameStop int, operatorDirectedMode string, handoffConfig ...guardTaskHandoffConfig) ([]string, [][2]string, guardStopHookInstall, error) {
	normalized, err := normalizeGuardStopHookMode(mode)
	if err != nil {
		return command, nil, guardStopHookInstall{}, err
	}
	install := guardStopHookInstall{Mode: normalized, WarnAt: warnAt, FinalAt: finalAt, Max: maxN, SameStop: sameStop}
	if normalized == guardPreCompactModeOff {
		install.Reason = "disabled"
		return command, nil, install, nil
	}
	if !guardPreCompactIsClaudeCommand(command) {
		install.Reason = "non-claude-child"
		return command, nil, install, nil
	}
	fakBin, err := os.Executable()
	if err != nil || strings.TrimSpace(fakBin) == "" {
		fakBin = "fak"
	}
	dir := ""
	if strings.TrimSpace(existingSettingsPath) == "" {
		// Allocate through the creation seam so the name carries this guard's PID
		// (fak-guard-stophook-<pid>-*). A raw os.MkdirTemp here produced a pid-less
		// name that guardTempDirOwner refuses, so guardReapStaleTempDirs could never
		// claim the dir even though "stophook" is already in the reaped hook set —
		// the same defect as #5527's task-handoff leak, at a third call site (#5535).
		// Nothing removes this dir on the happy path either (there is no Close() to
		// lean on, unlike the lifecycle server), so the dead-owner sweep is the only
		// bound this family has. See guard_tempreap.go.
		dir, err = guardSessionTempDir("stophook")
		if err != nil {
			return command, nil, guardStopHookInstall{}, err
		}
	}
	return installGuardStopHookAt(command, mode, gwURL, fakBin, dir, existingSettingsPath, warnAt, finalAt, maxN, sameStop, operatorDirectedMode, handoffConfig...)
}

func installGuardStopHookAt(command []string, mode, gwURL, fakBin, dir, existingSettingsPath string, warnAt, finalAt, maxN, sameStop int, operatorDirectedMode string, handoffConfig ...guardTaskHandoffConfig) ([]string, [][2]string, guardStopHookInstall, error) {
	normalized, err := normalizeGuardStopHookMode(mode)
	if err != nil {
		return command, nil, guardStopHookInstall{}, err
	}
	// Normalize once so the install record, the banner, and the injected env all carry the SAME
	// effective ladder the hook will use — a misconfigured flag can never present one ladder and
	// run another. The same-issue give-up depth is normalized on its own scale (warn/final derived).
	warnAt, finalAt, maxN = normalizeDenyAllThresholds(warnAt, finalAt, maxN)
	_, _, sameStop = normalizeSameStop(sameStop)
	install := guardStopHookInstall{Mode: normalized, WarnAt: warnAt, FinalAt: finalAt, Max: maxN, SameStop: sameStop}
	if normalized == guardPreCompactModeOff {
		install.Reason = "disabled"
		return command, nil, install, nil
	}
	if !guardPreCompactIsClaudeCommand(command) {
		install.Reason = "non-claude-child"
		return command, nil, install, nil
	}

	var settingsPath string
	if strings.TrimSpace(existingSettingsPath) != "" {
		// Merge the Stop hook INTO the file the PreCompact hook already wrote + injected.
		if err := mergeGuardStopHookIntoSettings(existingSettingsPath, fakBin); err != nil {
			return command, nil, install, err
		}
		settingsPath = existingSettingsPath
		// command already carries --settings; do NOT inject it again.
	} else {
		if strings.TrimSpace(dir) == "" {
			return command, nil, install, errors.New("empty Stop hook settings directory")
		}
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return command, nil, install, err
		}
		settingsPath = filepath.Join(dir, "claude-stophook-settings.json")
		if err := writeGuardStopHookSettings(settingsPath, fakBin); err != nil {
			return command, nil, install, err
		}
		command = appendClaudeSettingsArg(command, settingsPath)
	}
	metricsURL := guardPreCompactMetricsURLFromBase(gwURL)
	install.Applied = true
	install.SettingsPath = settingsPath
	install.MetricsURL = metricsURL
	env := [][2]string{
		{guardStopHookEnvMode, normalized},
		{guardStopHookEnvMetricsURL, metricsURL},
		{guardStopHookEnvWarn, strconv.Itoa(warnAt)},
		{guardStopHookEnvFinal, strconv.Itoa(finalAt)},
		{guardStopHookEnvMax, strconv.Itoa(maxN)},
		// Pin the same-issue give-up depth (default 6, deny-all policy same-stop) so the installed Stop hook keys its
		// stand-down on a true repeated same refusal rather than the blind deny-all count.
		{guardStopHookEnvSameStop, strconv.Itoa(sameStop)},
		// Pin the RESOLVED operator-directed gate mode. The operator-absent cap
		// (guardOperatorDirectedEffectiveMode) has already run at the call site, so this value is
		// authoritative — and it is pinned even when it resolves to `off` (attended interactive) so
		// the hook never falls back to the warn default the flag would otherwise supply.
		{guardStopHookOperatorDirectedEnvMode, guardOperatorDirectedNormalizedOrWarn(operatorDirectedMode)},
		// Pin the hardware-gate mode. It INHERITS the resolved operator-directed posture — both are
		// headless-only enforcement gates capped by operator-absence, so the same effective mode is
		// the right default without threading a second install param. An operator who wants to tune
		// it independently sets FAK_GUARD_HARDWARE_GATE_MODE / --hardware-gate on the child.
		{guardStopHookHardwareGateEnvMode, guardHardwareGateNormalizedOrWarn(operatorDirectedMode)},
		// Pin the operator-question mode. Same rationale as the hardware gate: the evidence-first
		// ExitPlanMode/AskUserQuestion rung is an operator-absence-capped headless gate, so it
		// INHERITS the resolved operator-directed posture as its default — a session that split out
		// this dial but tuned neither behaves exactly as before. An operator who wants the
		// plan/question rung on a different posture than the prose sensor sets
		// FAK_GUARD_OPERATOR_QUESTION_MODE / --operator-question on the child.
		{guardStopHookOperatorQuestionEnvMode, guardOperatorQuestionNormalizedOrWarn(operatorDirectedMode)},
	}
	// #2539: wire the trajctl ledger into the hook children so the Stop hook's turn-end
	// scorers and the PreCompact boundary twin have a curve to append to. Unresolvable repo
	// root injects nothing and the sampling stays a total no-op.
	if ledger := guardTrajctlLedgerDefault(); ledger != "" {
		env = append(env, [2]string{guardTrajctlEnvLedger, ledger})
	}
	// Wire the typed stop-decision ledger into the hook child so every turn-end decision
	// (clean stops, bounded stand-downs, and the fail-open stops that are otherwise
	// invisible) lands as a countable row `fak guard-stops` can fold. Unresolvable repo
	// root injects nothing and recording stays a total no-op.
	if ledger := guardStopsLedgerDefault(); ledger != "" {
		env = append(env, [2]string{guardStopsLedgerEnv, ledger})
		install.StopsLedger = ledger
	}
	env = append(env, guardTaskHandoffEnv(guardTaskHandoffConfigOrZero(handoffConfig))...)
	return command, env, install, nil
}

// guardStopHookMatchers builds the Stop-hook settings entry. The Stop event takes NO matcher
// (matchers are for tool-scoped events), so the Matcher field is left empty (omitempty drops it).
func guardStopHookMatchers(fakBin string) []guardPreCompactClaudeMatcher {
	return []guardPreCompactClaudeMatcher{{
		Hooks: []guardPreCompactClaudeCommand{{
			Type:    "command",
			Command: guardPreCompactHookCommand(fakBin),
			Args:    []string{"guard-stophook"},
		}},
	}}
}

func guardGoalQuestionMatchers(fakBin string) guardPreCompactClaudeMatcher {
	return guardPreCompactClaudeMatcher{
		Matcher: "AskUserQuestion",
		Hooks: []guardPreCompactClaudeCommand{{
			Type:    "command",
			Command: guardPreCompactHookCommand(fakBin),
			Args:    []string{"guard-goal-question"},
		}},
	}
}
func guardCommitGateMatchers(fakBin string) []guardPreCompactClaudeMatcher {
	return []guardPreCompactClaudeMatcher{{
		Matcher: "Bash",
		Hooks: []guardPreCompactClaudeCommand{{
			Type:    "command",
			Command: guardPreCompactHookCommand(fakBin),
			Args:    []string{"guard-commit-gate"},
		}},
	}}
}

func writeGuardStopHookSettings(path, fakBin string) error {
	settings := guardPreCompactClaudeSettings{
		Hooks: map[string][]guardPreCompactClaudeMatcher{
			"Stop":       guardStopHookMatchers(fakBin),
			"PreToolUse": append(guardCommitGateMatchers(fakBin), guardGoalQuestionMatchers(fakBin)),
		},
	}
	data, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return writeGuardSettingsFileAtomic(path, data)
}

// mergeGuardStopHookIntoSettings adds (or replaces) the Stop hook in an existing guard settings
// file, preserving every other key (e.g. the PreCompact hook), so a single --settings carries both.
func mergeGuardStopHookIntoSettings(path, fakBin string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var settings guardPreCompactClaudeSettings
	if err := json.Unmarshal(raw, &settings); err != nil {
		return fmt.Errorf("parse existing hook settings %s: %w", path, err)
	}
	if settings.Hooks == nil {
		settings.Hooks = map[string][]guardPreCompactClaudeMatcher{}
	}
	settings.Hooks["Stop"] = guardStopHookMatchers(fakBin)
	settings.Hooks["PreToolUse"] = append(guardCommitGateMatchers(fakBin), guardGoalQuestionMatchers(fakBin))
	data, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return writeGuardSettingsFileAtomic(path, data)
}

type guardStopHookSignals struct {
	DenyAllConsecutive      int
	ToolFeedbackConsecutive int
	// DenyAllSameConsecutive is the consecutive deny-all turns proposing the IDENTICAL refused
	// action (fak_guard_deny_all_same_consecutive). This is the gauge the hook keys its give-up
	// on. Absence is tolerated (older gateway): DenyAllSameConsecutiveSeen distinguishes "gauge
	// reported 0" from "gauge not on the scrape", so the hook falls back to the blind ladder only
	// when the gateway genuinely does not emit it.
	DenyAllSameConsecutive     int
	DenyAllSameConsecutiveSeen bool
	// FakVerbCalls is the cumulative admitted MCP fak-verb call count this process
	// (fak_mcp_verb_calls_total). 0 at a clean stop means fak was present but never used as a
	// substrate — the #3093 unused-substrate pathology. Absence is tolerated (older gateway):
	// FakVerbCallsSeen distinguishes "counter reported 0" from "counter not on the scrape".
	FakVerbCalls     int
	FakVerbCallsSeen bool
}

func fetchGuardStopHookSignalsPreferred(ctx context.Context, metricsURL string, timeout time.Duration) (guardStopHookSignals, string, error) {
	socketPath := strings.TrimSpace(os.Getenv(guardLifecycleSocketEnv))
	token := strings.TrimSpace(os.Getenv(guardLifecycleTokenEnv))
	if socketPath != "" || token != "" {
		if socketPath == "" || token == "" {
			return guardStopHookSignals{}, "ipc", errors.New("in-process lifecycle IPC environment incomplete")
		}
		in, err := fetchGuardLifecycleSignals(socketPath, token, timeout)
		if err != nil {
			// A supervisor-provisioned IPC endpoint is authoritative. Do not silently
			// degrade an enforce gate to a socket-shaped HTTP fail-open window.
			return guardStopHookSignals{}, "ipc", fmt.Errorf("in-process lifecycle IPC: %w", err)
		}
		return guardStopHookSignals{
			DenyAllConsecutive:         in.DenyAllConsecutive,
			DenyAllSameConsecutive:     in.DenyAllSameConsecutive,
			DenyAllSameConsecutiveSeen: in.DenyAllSameConsecutiveSeen,
			ToolFeedbackConsecutive:    in.ToolFeedbackConsecutive,
			FakVerbCalls:               in.FakVerbCalls,
			FakVerbCallsSeen:           in.FakVerbCallsSeen,
		}, "ipc", nil
	}
	if strings.TrimSpace(metricsURL) == "" {
		return guardStopHookSignals{}, "", errors.New("no lifecycle IPC or metrics URL configured")
	}
	out, err := fetchGuardStopHookSignals(ctx, metricsURL, timeout)
	return out, "http", err
}
func fetchGuardStopHookSignals(ctx context.Context, metricsURL string, timeout time.Duration) (guardStopHookSignals, error) {
	if timeout <= 0 {
		timeout = 500 * time.Millisecond
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, metricsURL, nil)
	if err != nil {
		return guardStopHookSignals{}, err
	}
	client := &http.Client{Timeout: timeout}
	resp, err := client.Do(req)
	if err != nil {
		return guardStopHookSignals{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return guardStopHookSignals{}, fmt.Errorf("metrics returned HTTP %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return guardStopHookSignals{}, err
	}
	return parseGuardStopHookSignals(string(body))
}

// parseGuardStopHookConsecutive extracts the unlabeled fak_guard_deny_all_consecutive gauge
// value from a Prometheus scrape. Not-found is an error so the caller fails open rather than
// silently treating a missing gauge as 0 (which would never auto-continue).
func parseGuardStopHookConsecutive(metrics string) (int, error) {
	signals, err := parseGuardStopHookSignals(metrics)
	return signals.DenyAllConsecutive, err
}

func parseGuardStopHookSignals(metrics string) (guardStopHookSignals, error) {
	var out guardStopHookSignals
	foundDenyAll := false
	for _, line := range strings.Split(metrics, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		// The Prometheus sample name is fields[0] up to any label set ("name{labels}"). Strip the
		// labels and match BEFORE parsing a value, so this scan only ever reads a value for a line it
		// consumes. This /metrics endpoint carries the gateway's WHOLE scrape (labeled families,
		// histograms), and a label value can hold a space — promQuote escapes \, ", and newline but
		// NOT spaces — which strings.Fields would split, pushing a non-numeric token into the middle
		// of the line. Parsing every line's value (the old fields[1]) would then hard-fail the entire
		// scan on one unrelated series and silently fail-open the deny-all governor. Stripping the
		// label set also lets a future LABELED emission of one of our own gauges still match.
		name := fields[0]
		if brace := strings.IndexByte(name, '{'); brace >= 0 {
			name = name[:brace]
		}
		switch name {
		case guardStopHookMetricName, guardStopHookSameMetricName,
			guardStopHookToolFeedbackMetricName, guardStopHookFakVerbCallsMetricName:
		default:
			continue // not one of ours — its value shape can never gate the scan
		}
		// The value is the LAST field (Prometheus puts it after the optional label set), so this is
		// correct whether or not the line carries labels.
		raw := fields[len(fields)-1]
		value, err := strconv.ParseFloat(raw, 64)
		if err != nil {
			return guardStopHookSignals{}, fmt.Errorf("parse %s value %q: %w", name, raw, err)
		}
		switch name {
		case guardStopHookMetricName:
			out.DenyAllConsecutive = int(value)
			foundDenyAll = true
		case guardStopHookSameMetricName:
			out.DenyAllSameConsecutive = int(value)
			out.DenyAllSameConsecutiveSeen = true
		case guardStopHookToolFeedbackMetricName:
			out.ToolFeedbackConsecutive = int(value)
		case guardStopHookFakVerbCallsMetricName:
			out.FakVerbCalls = int(value)
			out.FakVerbCallsSeen = true
		}
	}
	if !foundDenyAll {
		return guardStopHookSignals{}, fmt.Errorf("metric %s not found", guardStopHookMetricName)
	}
	return out, nil
}

func refusalRestatementStatePath(sessionID string) string {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return ""
	}
	base := strings.TrimSpace(os.Getenv("FAK_GUARD_REFUSAL_STATE_DIR"))
	if base == "" {
		base = filepath.Join(os.TempDir(), "fak-guard-refusal-state")
	}
	return filepath.Join(base, refusalSafeSessionID(sessionID)+".json")
}

func guardStopGitHead() string {
	return gitHeadFromCommand(exec.Command("git", "rev-parse", "HEAD"))
}
func refusalSafeSessionID(s string) string {
	var b strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			b.WriteRune(r)
		}
	}
	if b.Len() == 0 {
		return "session"
	}
	return b.String()
}
