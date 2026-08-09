package fleetaccounts

import (
	"regexp"
	"strings"
	"time"
)

// weekdayPrefixRE strips a leading weekday token ("Mon ", "tue ", ...) from a reset
// string before format matching, mirroring fleet_accounts._reset_is_future's
// `^(mon|tue|wed|thu|fri|sat|sun)\s+` scrub. Claude stamps its DATED weekly resets as
// "Mon Jun 25 at 1pm"; without this strip the weekday makes the whole string unparseable,
// which resetIsFuture reports as nil (unknown) and throttleIsActive then treats fail-closed
// as a still-active weekly cap — walling a healthy seat indefinitely.
var weekdayPrefixRE = regexp.MustCompile(`^(?i:(mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?))\s+(?:at\s+)?`)

var weekdayNumbers = map[string]time.Weekday{
	"mon": time.Monday, "monday": time.Monday,
	"tue": time.Tuesday, "tuesday": time.Tuesday,
	"wed": time.Wednesday, "wednesday": time.Wednesday,
	"thu": time.Thursday, "thursday": time.Thursday,
	"fri": time.Friday, "friday": time.Friday,
	"sat": time.Saturday, "saturday": time.Saturday,
	"sun": time.Sunday, "sunday": time.Sunday,
}

// soonness.go — the shared reset-time core behind resetIsFuture, plus the ResetSoonness
// signal the config-plane rotation uses to break ties AMONG walled buckets (order the
// soonest-to-reset account first). Both read Claude's messy reset strings through ONE
// parser (resetTime) so the format handling — parenthetical stripping, dated vs
// time-only layouts, the daily-reset rollover slack — lives in a single place and is not
// re-invented per caller. Mirrors fleet_accounts._reset_is_future's anchoring.

// resetSoonWindow is the horizon over which a still-future reset is projected onto a [0,1)
// soonness score: a reset AT now scores ~1 (about to free up), a reset resetSoonWindow or
// more away scores ~0. It is a coarse tie-break horizon, not a promise about the exact
// free-up moment — the input strings only carry minute resolution and no explicit date for
// the time-only forms. A day comfortably brackets Claude's 5-hour rolling and daily windows.
const resetSoonWindow = 24 * time.Hour

// resetTime parses one of Claude's reset strings into the concrete UTC-anchored instant it
// refers to, using the SAME format handling resetIsFuture relies on. ok is false for an
// empty or unparseable string. The returned instant may be in the past (an expired reset);
// callers that care about future-ness compare against now themselves. now anchors the
// undated time-only forms (which carry no date) and the dated forms' year rollover.
func resetTime(reset string, now time.Time) (time.Time, bool) {
	if reset == "" {
		return time.Time{}, false
	}
	loc := now.Location()
	if strings.Contains(reset, "America/Los_Angeles") {
		if la, err := time.LoadLocation("America/Los_Angeles"); err == nil {
			loc = la
		}
	}
	nowInLoc := now.In(loc)
	raw := parenTail.ReplaceAllString(reset, "")
	raw = strings.TrimSpace(raw)
	raw = parenAll.ReplaceAllString(raw, "")
	raw = strings.TrimSpace(raw)
	raw = strings.ToLower(wsRun.ReplaceAllString(raw, " "))

	// A weekday plus only a time names the next occurrence of that weekday.
	// Resolve it before the historical weekday scrub so "Monday at 9am" does
	// not become today's 09:00. Explicit dates still win: "Mon Jun 3 at 9am"
	// falls through to the dated layouts after its weekday is removed.
	if match := weekdayPrefixRE.FindStringSubmatch(raw); match != nil {
		rest := strings.TrimSpace(raw[len(match[0]):])
		if tod, ok := parseResetClock(rest, loc); ok {
			weekday := weekdayNumbers[strings.ToLower(match[1])]
			candidate := time.Date(nowInLoc.Year(), nowInLoc.Month(), nowInLoc.Day(), tod.Hour(), tod.Minute(), 0, 0, loc)
			days := (int(weekday) - int(nowInLoc.Weekday()) + 7) % 7
			candidate = candidate.AddDate(0, 0, days)
			if !candidate.After(nowInLoc) {
				candidate = candidate.AddDate(0, 0, 7)
			}
			if candidate.Sub(nowInLoc) <= 8*24*time.Hour {
				return candidate, true
			}
			return time.Time{}, false
		}
		raw = strings.TrimSpace(raw[len(match[0]):])
		if raw == "" {
			return time.Time{}, false
		}
	}

	type fmtSpec struct {
		layout string
		dated  bool
	}
	// Order mirrors fleet_accounts._reset_is_future: the DATED weekly forms first (both
	// the "at" and comma separators Claude emits — "Jun 25 at 1pm" / "Jun 25, 1pm"), then
	// the BARE 5-hour rolling forms. Minute-bearing layouts precede their hour-only twin so
	// "1:30pm" is not eaten by the "1pm" layout.
	specs := []fmtSpec{
		{"Jan 2 at 3:04pm", true},
		{"Jan 2 at 3pm", true},
		{"Jan 2, 3:04pm", true},
		{"Jan 2, 3pm", true},
		{"3:04pm", false},
		{"3pm", false},
	}
	for _, sp := range specs {
		parsed, err := time.Parse(sp.layout, raw)
		if err != nil {
			continue
		}
		if sp.dated {
			cand := time.Date(nowInLoc.Year(), parsed.Month(), parsed.Day(),
				parsed.Hour(), parsed.Minute(), 0, 0, loc)
			if cand.Before(now) && now.Sub(cand) > 180*24*time.Hour {
				cand = cand.AddDate(1, 0, 0)
			}
			return cand, true
		}
		cand := time.Date(nowInLoc.Year(), nowInLoc.Month(), nowInLoc.Day(),
			parsed.Hour(), parsed.Minute(), 0, 0, loc)
		// An undated time-only reset that has already passed today rolls to tomorrow when it
		// still falls inside the daily-reset slack, exactly as resetIsFuture treats it as a
		// future reset — so soonness projects onto the same instant that future-ness reports.
		if !cand.After(now) {
			tomorrow := cand.Add(24 * time.Hour)
			if tomorrow.Sub(now) <= dailyResetWindow {
				cand = tomorrow
			}
		}
		return cand, true
	}
	return time.Time{}, false
}

func parseResetClock(raw string, loc *time.Location) (time.Time, bool) {
	for _, layout := range []string{"3:04pm", "3pm"} {
		if parsed, err := time.ParseInLocation(layout, strings.TrimSpace(raw), loc); err == nil {
			return parsed, true
		}
	}
	return time.Time{}, false
}

// ResetInstant parses one of Claude's reset strings ("7:10pm (America/Los_Angeles)",
// "Dec 31, 1pm") into the concrete instant it refers to, anchored on now — the exported
// face of resetTime for callers that need the moment itself (e.g. the resume resolver's
// WAIT_RESET countdown), not a soonness score. ok is false for an empty or unparseable
// string; the instant may be in the past (an expired reset) — callers compare against now.
func ResetInstant(reset string, now time.Time) (time.Time, bool) {
	return resetTime(reset, now)
}

// ResetSoonness scores how SOON a still-future reset string will free an account up, on a
// [0,1) scale where SOONER is HIGHER: a reset at (or just after) now scores near 1, a reset
// resetSoonWindow or further out scores near 0. ok is false when the string is empty,
// unparseable, or names an already-expired reset (no soonness to report — the account is not
// waiting on that reset). It is the tie-break the config-plane rotation folds into the walled
// tier so a rotate onto an all-walled fleet lands on the account that recovers first, rather
// than on an arbitrary name. Pure over the injected now for deterministic tests.
func ResetSoonness(reset string, now time.Time) (float64, bool) {
	t, ok := resetTime(reset, now)
	if !ok {
		return 0, false
	}
	wait := t.Sub(now)
	if wait < 0 {
		return 0, false // already reset; nothing to wait on
	}
	if wait >= resetSoonWindow {
		return 0, true // very far out — future, but no sooner than the horizon
	}
	// Linear: wait==0 -> 1, wait==resetSoonWindow -> 0. Strictly within [0,1).
	return 1 - wait.Seconds()/resetSoonWindow.Seconds(), true
}
