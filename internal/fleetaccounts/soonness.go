package fleetaccounts

import (
	"strings"
	"time"
)

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
	raw := parenTail.ReplaceAllString(reset, "")
	raw = strings.TrimSpace(raw)
	raw = parenAll.ReplaceAllString(raw, "")
	raw = strings.TrimSpace(raw)
	raw = strings.ToLower(wsRun.ReplaceAllString(raw, " "))

	type fmtSpec struct {
		layout string
		dated  bool
	}
	specs := []fmtSpec{
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
			cand := time.Date(now.Year(), parsed.Month(), parsed.Day(),
				parsed.Hour(), parsed.Minute(), 0, 0, now.Location())
			if cand.Before(now) && now.Sub(cand) > 180*24*time.Hour {
				cand = cand.AddDate(1, 0, 0)
			}
			return cand, true
		}
		cand := time.Date(now.Year(), now.Month(), now.Day(),
			parsed.Hour(), parsed.Minute(), 0, 0, now.Location())
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
