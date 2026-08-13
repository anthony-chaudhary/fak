package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

// guard_cap_park.go — the mid-session CAP-CRASH auto-recover (#2256), the rate-limit sibling
// of guardMaybeRecoverAuthCrash (guard_child.go). Where the auth path answers "did this crash
// happen because the subscription token EXPIRED, and has a fresh login landed?", this path
// answers "did the wrapped agent die because the account hit a usage/session/weekly CAP whose
// reset is known, and if so, can we ride out the window and relaunch --continue in-process?"
//
// Why inline, when `fak resume` already does supervisor-side scan→diagnose→next→relaunch: the
// external sweep is a SEPARATE process that runs post-hoc. Inside a live `fak guard -- claude`
// the child has just written its own transcript naming the cap; the guard is still up holding
// the gateway. Recovering HERE — witness the cap from the child's own trail, park to the reset,
// relaunch --continue — closes the loop without waiting for an out-of-band sweep to notice, the
// same way guardMaybeRecoverAuthCrash closes the auth loop without a manual re-run. It reuses
// the SAME closed vocabulary the resume verbs fold (resume.Diagnose / resume.FoldNextAction /
// the published SessionLimit/WeeklyLimit reset windows) so the inline and supervisor paths can
// never disagree about what a cap is or how long its reset takes.
//
// Field signature this rides out (2026-07-01, cited on #2256): a headless dispatch session hits
// a usage-cap 429 whose reset is ~1h out; the wrapped Claude Code times its own request out at
// ~300s, twelve futile cycles burn ~50min, then the session is LOST even though the reset
// instant was known. With the companion gateway fix (#2258, upstream_retry_ceiling) the child no
// longer sleeps in-handler past the client; it exits with the cap named — and THIS path is what
// turns that clean exit into a parked-then-resumed session instead of a dead one.

// guardCapParkDecision is the pure verdict of the cap-crash classifier: given a completed child
// exit and the cap witnessed from its transcript, should the guard park, and if so until when
// (relative), then relaunch. It carries no clock and does no I/O so the whole precedence is
// unit-tested without standing guard up (mirroring guardClassifyAuthCrash's purity).
type guardCapParkDecision struct {
	// Park is true iff the crash correlates with a recoverable wall-clock cap whose reset has
	// NOT surely elapsed yet — the one case where waiting can make a relaunch stick. A crash
	// that is not a cap, or a cap whose window already passed, or a non-zero-but-clean exit, is
	// Park=false: the caller's existing report/exit path proceeds unchanged.
	Park bool
	// LimitReason is the closed cap token (session_limit / weekly_limit / usage_limit) driving
	// the wait — empty when Park is false. rate_limited (a 529/429 burst) is deliberately NOT a
	// park case here: that is a per-source admission concern the resume host-admission gate owns,
	// not a per-session wall-clock reset, so it falls through to today's report path.
	LimitReason string
	// Wait is how long to park before a relaunch can possibly stick — the published reset window
	// for the cap, floored by any budget the caller already spent. Zero when Park is false.
	Wait time.Duration
	// Relaunch is the command to re-run after the park (the original command with the agent's
	// safe resume flag appended, never stacked twice). Nil when Park is false or the agent has no
	// resume flag fak knows is safe (an unrecognized binary is never auto-relaunched — same
	// discipline as guardContinueFlagForAgent on the auth path).
	Relaunch []string
	// Reason is the human one-liner for the park/no-park verdict, surfaced on the park status line.
	Reason string
}

// guardCapParkResetWindow maps a closed cap token to the wall-clock window a relaunch must wait
// out, reusing the published cap semantics resume already encodes (a 5h rolling session cap, a
// 7d weekly cap; usage_limit has no published window so it takes the conservative session floor).
// It returns 0 for rate_limited (a burst — admission, not a reset, handles it) and for any
// non-cap token, so those never drive a park here. Kept as a thin adapter over the resume
// constants rather than duplicating the numbers, so the inline and supervisor paths share ONE
// source of truth for how long a cap takes to reset.
func guardCapParkResetWindow(limitReason string) time.Duration {
	switch limitReason {
	case resume.LimitSession, resume.LimitUsage:
		return time.Duration(resume.SessionLimitResetSeconds) * time.Second
	case resume.LimitWeekly:
		return time.Duration(resume.WeeklyLimitResetSeconds) * time.Second
	default: // rate_limited (burst) or a non-cap token: no wall-clock reset to park on
		return 0
	}
}

// guardClassifyCapCrash decides whether a completed child exit correlates with a recoverable
// usage/session/weekly cap whose reset window has not surely elapsed, and if so how long to
// park before relaunching the SAME command with the agent's safe resume flag. It is the pure
// counterpart of guardClassifyAuthCrash: no clock, no I/O, no exit — the witness (diag, from
// resume.Diagnose over the child's transcript) and the already-idle seconds are caller-supplied.
//
// Precedence (first binding constraint wins), deliberately parallel to FoldNextAction:
//  1. A nil/success/zero-code exit, or a crash the witness did not classify as a rate-limit cap
//     (CrashRateLimit), is not our case → Park=false (the caller reports/exits as today).
//  2. A cap token with no wall-clock reset (rate_limited burst) → Park=false: admission, not a
//     per-session reset, owns that; parking a whole guarded session on a burst would be wrong.
//  3. A cap whose reset window has SURELY elapsed already (idle >= window) → Park=false but the
//     caller MAY relaunch immediately; we still return the Relaunch command with Park=false and
//     Wait=0 so the caller can re-run without a wait. (This is the "reset already passed while
//     the child was dying" tail — waiting zero is correct, not a refusal.)
//  4. An agent with no fak-known safe resume flag → Park=false, Relaunch=nil: fak never guesses a
//     foreign tool's continuation syntax (same rule as the auth path).
//  5. Otherwise → Park=true, Wait=(window - idle) clamped to [0, window], Relaunch set.
//
// idleSeconds is how long the session has ALREADY been idle since its last real turn (from the
// transcript's last timestamp); a negative value means "unknown", which is conservatively NOT
// treated as elapsed (we wait the full window). command/agentName mirror the auth path.
func guardClassifyCapCrash(diag resume.Diagnosis, idleSeconds int64, command []string, agentName string) guardCapParkDecision {
	if diag.Crash != resume.CrashRateLimit || diag.LimitReason == "" {
		return guardCapParkDecision{Reason: "not a rate-limit cap crash — no park"}
	}
	window := guardCapParkResetWindow(diag.LimitReason)
	if window <= 0 {
		return guardCapParkDecision{
			Reason: fmt.Sprintf("%s is a burst, not a wall-clock cap — admission handles it, no park", diag.LimitReason),
		}
	}
	flag, known := guardContinueFlagForAgent(agentName)
	if !known {
		return guardCapParkDecision{
			LimitReason: diag.LimitReason,
			Reason:      fmt.Sprintf("crashed on %s but %q has no resume flag fak can safely auto-inject — no park", diag.LimitReason, agentName),
		}
	}
	relaunch := guardAppendContinueFlag(command, flag)

	// The reset has surely elapsed already (idle known and past the window): no wait needed, but
	// the command is still relaunch-eligible — return it with Park=false / Wait=0.
	if idleSeconds >= 0 && time.Duration(idleSeconds)*time.Second >= window {
		return guardCapParkDecision{
			LimitReason: diag.LimitReason,
			Relaunch:    relaunch,
			Reason:      fmt.Sprintf("%s reset window (%s) already elapsed while the child was dying — relaunch now", diag.LimitReason, window),
		}
	}

	wait := window
	if idleSeconds > 0 {
		wait = window - time.Duration(idleSeconds)*time.Second
		if wait < 0 {
			wait = 0
		}
	}
	return guardCapParkDecision{
		Park:        true,
		LimitReason: diag.LimitReason,
		Wait:        wait,
		Relaunch:    relaunch,
		Reason:      fmt.Sprintf("crashed on %s; reset window %s, ~%s remaining — park then relaunch --continue", diag.LimitReason, window, wait.Round(time.Second)),
	}
}

// guardCapParkWait blocks for the decision's Wait (parking the guarded session on a known cap
// reset), printing ONE park line up front and ONE outcome line, so an operator tailing the log
// sees a parked session as parked — never a silent hang. It mirrors guardParkForRelogin's
// observable, injectable-clock shape. budget clamps the wait DOWN (a fat-fingered / very-long
// weekly window never wedges a session past the operator's declared ceiling); a budget <= 0
// means "no extra ceiling beyond the reset window itself". now/sleep default to the real clock
// and are injectable so tests never sleep wall-clock time. It returns the actual elapsed park.
//
// Unlike the relogin park this does NOT re-probe a file: the reset instant is provider-published
// (the cap window), so the wait is a single bounded sleep, then the caller relaunches. If the
// relaunch immediately re-hits the cap (the window estimate was short), the SAME recovery fires
// again next exit — bounded by the caller's overall park budget / --max-duration, exactly like
// the auth path's repeated-cycle handling.
func guardCapParkWait(dec guardCapParkDecision, budget time.Duration, now func() time.Time, sleep func(time.Duration), stderr io.Writer) time.Duration {
	if !dec.Park || dec.Wait <= 0 {
		return 0
	}
	if now == nil {
		now = time.Now
	}
	if sleep == nil {
		sleep = time.Sleep
	}
	wait := dec.Wait
	if budget > 0 && wait > budget {
		wait = budget
	}
	if stderr != nil {
		until := now().Add(wait)
		fmt.Fprintf(stderr, "fak guard: parked until %s — %s: riding out the reset for %s before relaunching `%s` (FAK_GUARD_PARK_BUDGET clamps this; 0-length disables the cap park).\n",
			until.Format("15:04"), dec.LimitReason, wait.Round(time.Second), strings.Join(dec.Relaunch, " "))
	}
	start := now()
	sleep(wait)
	elapsed := now().Sub(start)
	if stderr != nil {
		fmt.Fprintf(stderr, "fak guard: %s reset window rode out after %s — relaunching to resume this session.\n",
			dec.LimitReason, elapsed.Round(time.Second))
	}
	return elapsed
}

// guardCapRecoverFromEvents is the pure witness→decision fold: given the child transcript's
// ordered events, the last-record wall-clock unix (for idle) and the current time, it runs
// resume.Diagnose (the SAME classifier the supervisor sweep uses) and hands the verdict to
// guardClassifyCapCrash. Split out from the I/O so the classification path is unit-tested over
// synthetic event slices with no files. nowUnix<=0 or lastUnix<=0 yields idle=-1 ("unknown"),
// which guardClassifyCapCrash conservatively treats as not-yet-elapsed (park the full window).
func guardCapRecoverFromEvents(events []resume.Event, lastUnix, nowUnix int64, command []string, agentName string) guardCapParkDecision {
	idle := int64(-1)
	if lastUnix > 0 && nowUnix > 0 && nowUnix >= lastUnix {
		idle = nowUnix - lastUnix
	}
	diag := resume.Diagnose(events, resume.Input{IdleSeconds: idle})
	return guardClassifyCapCrash(diag, idle, command, agentName)
}

// guardCapWitnessTranscript is the transcript-locating I/O the witness needs: the freshest
// `.jsonl` under the guarded Claude project whose mtime is AT OR AFTER childStarted — i.e. the
// transcript THIS session just wrote, never a stale sibling. It returns "" (no witness) when the
// store is unreadable, empty, or the newest transcript predates the child launch (so a session
// with no fresh transcript is never mis-recovered off an old one). storeRoot is normally
// guardClaudeConfigDir()/projects; childStarted is when the just-exited child was launched.
func guardCapWitnessTranscript(storeRoot string, childStarted time.Time) string {
	paths, err := findTranscripts(storeRoot)
	if err != nil || len(paths) == 0 {
		return ""
	}
	// findTranscripts returns lexically sorted; re-rank by mtime so "freshest" is by write time,
	// not by name (session UIDs are not time-ordered).
	type stamped struct {
		path string
		mod  time.Time
	}
	var live []stamped
	for _, p := range paths {
		fi, err := os.Stat(p)
		if err != nil {
			continue
		}
		if fi.ModTime().Before(childStarted) {
			continue // predates this child — a stale sibling session, not ours
		}
		live = append(live, stamped{p, fi.ModTime()})
	}
	if len(live) == 0 {
		return ""
	}
	sort.Slice(live, func(i, j int) bool { return live[i].mod.After(live[j].mod) })
	return live[0].path
}

// guardCapParkEnabled reports whether the inline cap park is on. It is ON by default (the field
// signature on #2256 is a session LOST to a known-reset cap — the safe default is to ride it
// out), disabled only by FAK_GUARD_CAP_PARK=0/false, matching the FAK_GUARD_PARK_BUDGET=0 escape
// hatch the relogin park already documents. Reusing guardParkBudget() as the ceiling keeps ONE
// knob governing how long any guard park may wait.
func guardCapParkEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_GUARD_CAP_PARK"))) {
	case "0", "false", "off", "no":
		return false
	default:
		return true
	}
}

// guardMaybeRecoverCapCrash is the cap-crash counterpart of guardMaybeRecoverAuthCrash, wired at
// the SAME two recovery sites (runGuardChildAndReport / runGuardChildSupervisedAndReport). After
// the wrapped agent exits non-zero it: locates the transcript this session just wrote, folds it
// through resume.Diagnose + guardClassifyCapCrash, and — on a recoverable cap whose reset has not
// elapsed — parks (bounded by guardParkBudget()) then returns the --continue relaunch command
// with ok=true. Every other case (no fresh transcript, a non-cap crash, a burst, an unrecognized
// agent, the park disabled) returns ok=false and the caller's existing report/exit path proceeds
// unchanged. runErr must be the child's completed exit error; a nil/success exit never matches.
// childStarted scopes the transcript search to this session; now/sleep are injectable for tests.
func guardMaybeRecoverCapCrash(runErr error, command []string, agentName string, childStarted time.Time, quiet bool, maxWait time.Duration, now func() time.Time, sleep func(time.Duration), stderr io.Writer) (relaunch []string, ok bool) {
	if runErr == nil || maxWait < 0 || !guardCapParkEnabled() {
		return nil, false
	}
	if now == nil {
		now = time.Now
	}
	storeRoot := filepath.Join(guardClaudeConfigDir(), "projects")
	transcript := guardCapWitnessTranscript(storeRoot, childStarted)
	if transcript == "" {
		return nil, false
	}
	f, err := os.Open(transcript)
	if err != nil {
		return nil, false
	}
	events, _, lastUnix, _ := scanTranscriptToEvents(f)
	f.Close()
	dec := guardCapRecoverFromEvents(events, lastUnix, now().Unix(), command, agentName)
	if !dec.Park {
		// Not a park case. If the reset already elapsed (Relaunch set, Wait 0) the caller could
		// relaunch immediately — but that is the auth path's job to decide; here we only own the
		// wait-then-relaunch. A no-park verdict falls through to today's report/exit path.
		return nil, false
	}
	if !quiet && stderr != nil {
		fmt.Fprintf(stderr, "fak guard: %s exited on a %s — %s\n", agentName, dec.LimitReason, dec.Reason)
	}
	budget := guardParkBudget()
	if maxWait > 0 && (budget <= 0 || maxWait < budget) {
		budget = maxWait
	}
	guardCapParkWait(dec, budget, now, sleep, stderr)
	return dec.Relaunch, true
}
