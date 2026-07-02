// sleepwait.go — the long-foreground-sleep rung of the repo-guard PreToolUse
// hook (#2366).
//
// A long foreground timer (`sleep 1500`, `Start-Sleep -Seconds 300`, a
// `for ...; do sleep 300; probe; done` poll loop) holds an agent turn open
// doing nothing: the 2026-07-01 trajectory audit found single sessions burning
// ~75 minutes of held-open turns this way while background waits (Monitor,
// run_in_background, ScheduleWakeup) exist for exactly this. This rung is
// ADVISORY, not a refusal: legitimate long waits stay possible, but the agent
// gets a structured pointer at the background alternatives before the turn is
// wasted.
//
// Curation principle: flag only a sleep invocation whose OWN duration is
// provably at or over the threshold. An unresolvable duration (`sleep $T`)
// passes — the guard never guesses. Short sleeps inside loops pass too: the
// per-invocation duration is the thing this rung can prove. Pure: no
// filesystem, no clock, hermetically testable like the rest of the core.
package repoguard

import (
	"fmt"
	"strconv"
	"strings"
)

// ReasonForegroundSleep is the structured advisory token for long foreground
// sleep timers.
const ReasonForegroundSleep = "FOREGROUND_SLEEP"

// DefaultForegroundSleepThresholdS is the advisory threshold: a single sleep
// invocation at or above this many seconds is flagged.
const DefaultForegroundSleepThresholdS = 120.0

// foregroundSleepFix pre-fills the background-wait alternatives, mirroring the
// Fix convention of the INTERACTIVE_HANG rung.
const foregroundSleepFix = "wait in the background instead: Bash run_in_background with an until-loop, the Monitor tool, or ScheduleWakeup"

// shellFlowKeywords are the POSIX flow-control words that can prefix the real
// verb inside a split segment (`do sleep 300`, `while sleep 300`), stripped so
// the sleep underneath is still seen.
var shellFlowKeywords = setOf("do", "then", "else", "elif", "while", "until", "time")

// IsAdvisoryReason reports whether a violation reason is advisory-only: the
// hook surfaces it on stderr but never denies the tool call for it.
func IsAdvisoryReason(reason string) bool {
	return reason == ReasonForegroundSleep
}

// ClassifySleepWait returns FOREGROUND_SLEEP advisories for long foreground
// sleep timers in a shell command. Pure string work.
func ClassifySleepWait(command string) []Violation {
	return classifySleepWait(command)
}

func classifySleepWait(command string) []Violation {
	// A command that backgrounds anything (`sleep 300 &`) does not provably
	// hold the turn — splitSegments erases the `&`, so check the raw text and
	// skip the whole command. Advisory rung: prefer the false negative.
	if hasBackgroundAmpersand(command) {
		return nil
	}
	var out []Violation
	for _, seg := range splitSegments(command) {
		toks, ok := shlexSplit(seg)
		if !ok {
			toks = strings.Fields(seg)
		}
		verb, operands, _ := stripEnvAndEnvVerb(toks)
		for shellFlowKeywords[verb] {
			verb, operands, _ = stripEnvAndEnvVerb(operands)
		}
		seconds, known := sleepSeconds(verb, operands)
		if !known || seconds < DefaultForegroundSleepThresholdS {
			continue
		}
		out = append(out, Violation{
			Reason:   ReasonForegroundSleep,
			Op:       fmt.Sprintf("%s %.0fs", verb, seconds),
			Target:   strings.TrimSpace(seg),
			Resolved: "<foreground>",
			Why:      fmt.Sprintf("holds this turn open ~%.0fs doing nothing in the foreground", seconds),
			Fix:      foregroundSleepFix,
		})
	}
	return out
}

// sleepSeconds resolves the total duration of one sleep invocation. known is
// false when the segment is not a sleep, or when any duration operand is
// unresolvable (a shell variable, a glob) — the guard never guesses.
func sleepSeconds(verb string, operands []string) (seconds float64, known bool) {
	switch strings.ToLower(verb) {
	case "sleep":
		return coreutilsSleepSeconds(operands)
	case "start-sleep":
		return startSleepSeconds(operands)
	}
	return 0, false
}

// coreutilsSleepSeconds sums `sleep` duration operands: bare seconds plus the
// GNU s/m/h/d suffixes, multiple operands added together (`sleep 1m 30` = 90s).
func coreutilsSleepSeconds(operands []string) (float64, bool) {
	var total float64
	var sawDuration bool
	for _, op := range operands {
		if strings.HasPrefix(op, "-") {
			continue // --help etc.: flags never carry a duration
		}
		unit := 1.0
		num := op
		if n := len(op); n > 0 {
			switch op[n-1] {
			case 's':
				num = op[:n-1]
			case 'm':
				num, unit = op[:n-1], 60
			case 'h':
				num, unit = op[:n-1], 3600
			case 'd':
				num, unit = op[:n-1], 86400
			}
		}
		v, err := strconv.ParseFloat(num, 64)
		if err != nil {
			return 0, false // $VAR, $(cmd), a filename — unresolvable
		}
		total += v * unit
		sawDuration = true
	}
	return total, sawDuration
}

// startSleepSeconds resolves a PowerShell Start-Sleep invocation: -Seconds /
// -Milliseconds (PowerShell prefix-matches parameter names, so -s / -sec / -ms
// count), the -Seconds:300 colon form, and the bare positional-number form.
func startSleepSeconds(operands []string) (float64, bool) {
	var total float64
	var sawDuration bool
	for i := 0; i < len(operands); i++ {
		op := operands[i]
		if strings.HasPrefix(op, "-") {
			name := strings.ToLower(strings.TrimPrefix(op, "-"))
			value := ""
			if eq := strings.IndexByte(name, ':'); eq >= 0 {
				name, value = name[:eq], name[eq+1:]
			} else if i+1 < len(operands) {
				i++
				value = operands[i]
			}
			unit := 0.0
			switch {
			case name != "" && strings.HasPrefix("seconds", name):
				unit = 1
			case len(name) >= 2 && strings.HasPrefix("milliseconds", name):
				unit = 1.0 / 1000
			default:
				continue // an unrelated or ambiguous flag
			}
			v, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return 0, false // $var — unresolvable
			}
			total += v * unit
			sawDuration = true
			continue
		}
		v, err := strconv.ParseFloat(op, 64)
		if err != nil {
			return 0, false
		}
		total += v // positional form is seconds
		sawDuration = true
	}
	return total, sawDuration
}

// hasBackgroundAmpersand reports a job-control `&`: not the `&&` chain
// operator and not part of a redirection (`2>&1`, `&>`, `>&`, `<&`).
func hasBackgroundAmpersand(command string) bool {
	r := []rune(command)
	for i, c := range r {
		if c != '&' {
			continue
		}
		if i > 0 && (r[i-1] == '&' || r[i-1] == '>' || r[i-1] == '<') {
			continue
		}
		if i+1 < len(r) && (r[i+1] == '&' || r[i+1] == '>') {
			continue
		}
		return true
	}
	return false
}

func renderSleepReason(violations []Violation) string {
	parts := make([]string, len(violations))
	for i, v := range violations {
		parts[i] = v.Op + " (" + v.Target + ") " + v.Why + " — fix: " + v.Fix
	}
	return ReasonForegroundSleep + ": a long foreground timer holds this turn open doing nothing. " +
		strings.Join(parts, "; ") + "."
}
