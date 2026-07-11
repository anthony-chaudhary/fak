package main

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/watchdoghealth"
)

// info_watchdog.go — the resume/heal watchdog's health, rendered in the `fak info` pane (#3802/#3803).
//
// The watchdog is the layer that resumes stranded agents and restarts a dead monitor. Until now its
// only live surface was the durable status ledger on disk — so an operator watching the split pane
// had no at-a-glance answer to "is the resume layer even alive?". This panel projects the gateway's
// /debug/vars "watchdog" block (the process-global resumemetrics counters) into gutter rows so the
// pane shows, in one place, whether the watchdog is ALIVE (ticking), HEALTHY (the cross-monitor
// rollup verdict), and DOING WORK (launch/skip verdicts, autoheal restarts, resumes proven to have
// produced a real post-launch turn). The `fak watchdog status --post-slack` digest is the push
// twin; this is the pull view #3802 asks for.
//
// Pure by construction: a *guardInfoWatchdog (resumemetrics.Snapshot) in, strings out. It leans on
// internal/watchdoghealth for the CLOSED status vocabulary and the attention-floor gate, so the
// pane's notion of "needs attention" can never drift from the digest's.

// watchdog health-chip glyphs. The status word leads every cell, so the glyph is a redundant
// at-a-glance cue, not the sole signal — but it maps to the watchdoghealth severity bands: ● alive
// and healthy, ○ deliberately not installed (a no-op, never attention), ◐ mid-recovery, ⚠ actionable
// (down / probe-unknown), ✗ the machine gave up (a human is owed).
const (
	guardWatchdogChipHealthy   = "●"
	guardWatchdogChipAbsent    = "○"
	guardWatchdogChipHealing   = "◐"
	guardWatchdogChipAttention = "⚠"
	guardWatchdogChipGaveUp    = "✗"
)

// guardInfoWatchdogActive reports whether the snapshot carries any watchdog signal worth a panel.
// It mirrors resumemetrics.Active() so a present-but-all-zero snapshot renders nothing, the same
// nil-when-empty contract the other optional panels keep (the gateway already omits the block on a
// cold process, so in practice a nil pointer is the common "no watchdog here" case).
func guardInfoWatchdogActive(wd *guardInfoWatchdog) bool {
	if wd == nil {
		return false
	}
	return wd.Ticks > 0 || wd.ProgressWitnessed > 0 || strings.TrimSpace(wd.HealthRollup) != "" ||
		len(wd.Actions) > 0 || len(wd.AutohealResults) > 0 || len(wd.MonitorStatus) > 0
}

// guardInfoWatchdogPanelRows is the watchdog sub-pane. Full form is the rollup head plus the
// actions / per-monitor / autoheal breakdowns; mini keeps just the rollup head. Silent (nil) when
// no watchdog signal is present — a silent panel costs zero rows.
func guardInfoWatchdogPanelRows(ctx guardInfoPanelCtx, level guardInfoPanelLevel) []string {
	wd := ctx.v.Watchdog
	if !guardInfoWatchdogActive(wd) {
		return nil
	}
	if level == guardPanelMini {
		return []string{" watchdog " + guardInfoWatchdogMiniText(wd)}
	}
	rows := []string{" watchdog " + guardInfoWatchdogHeadText(wd)}
	if actions := guardInfoWatchdogCountMap(wd.Actions, 4); actions != "" {
		rows = append(rows, guardInfoWatchdogSubRow("actions", actions))
	}
	if monitors := guardInfoWatchdogMonitorsText(wd.MonitorStatus, 3); monitors != "" {
		rows = append(rows, guardInfoWatchdogSubRow("monitors", monitors))
	}
	if heal := guardInfoWatchdogCountMap(wd.AutohealResults, 4); heal != "" {
		rows = append(rows, guardInfoWatchdogSubRow("autoheal", heal))
	}
	return rows
}

// guardInfoWatchdogSubRow lays a labeled breakdown under the head row: the 10-space gutter, an
// 8-wide sub-label, then the value — so actions / monitors / autoheal values align in a column.
func guardInfoWatchdogSubRow(label, value string) string {
	return fmt.Sprintf("          %-8s %s", label, value)
}

// guardInfoWatchdogHeadText is the head value: the rollup glyph + verdict word, the tick count
// (the alive proof — zero in-process ticks means the watchdog never ran here), and the resume
// count when any resume has been proven to produce a real turn.
func guardInfoWatchdogHeadText(wd *guardInfoWatchdog) string {
	parts := []string{watchdogGlyph(wd.HealthRollup) + " " + watchdogRollupWord(wd.HealthRollup)}
	parts = append(parts, fmt.Sprintf("%s ticks", guardInfoShortCount(int(wd.Ticks))))
	if wd.ProgressWitnessed > 0 {
		parts = append(parts, fmt.Sprintf("%s resumed", guardInfoShortCount(int(wd.ProgressWitnessed))))
	}
	return strings.Join(parts, " · ")
}

// guardInfoWatchdogMiniText is the one-row fold: the rollup verdict, the tick count, and how many
// monitors (or the rollup itself) are at or above the attention floor.
func guardInfoWatchdogMiniText(wd *guardInfoWatchdog) string {
	s := watchdogGlyph(wd.HealthRollup) + " " + watchdogRollupWord(wd.HealthRollup)
	s += fmt.Sprintf(" · %s ticks", guardInfoShortCount(int(wd.Ticks)))
	if n := guardInfoWatchdogAttentionCount(wd.MonitorStatus); n > 0 {
		s += fmt.Sprintf(" · %d need attention", n)
	} else if watchdogNeedsAttention(wd.HealthRollup) {
		s += " · needs attention"
	}
	return s
}

// guardInfoWatchdogText is the compact status-line fragment (line mode + `fak info --once`): the
// rollup verdict, the tick/resume counts, and an attention tail. Empty when no watchdog signal is
// present so the common passthrough line stays clean.
func guardInfoWatchdogText(wd *guardInfoWatchdog) string {
	if !guardInfoWatchdogActive(wd) {
		return ""
	}
	s := "watchdog: " + watchdogRollupWord(wd.HealthRollup)
	s += fmt.Sprintf(" (%s ticks", guardInfoShortCount(int(wd.Ticks)))
	if wd.ProgressWitnessed > 0 {
		s += fmt.Sprintf(", %s resumed", guardInfoShortCount(int(wd.ProgressWitnessed)))
	}
	s += ")"
	if n := guardInfoWatchdogAttentionCount(wd.MonitorStatus); n > 0 {
		s += fmt.Sprintf(" — %d need attention", n)
	} else if watchdogNeedsAttention(wd.HealthRollup) {
		s += " — needs attention"
	}
	return s
}

// guardInfoWatchdogAttentionCount counts the monitors whose last folded status is at or above the
// watchdoghealth attention floor (DOWN / UNKNOWN / GAVE_UP) — the same gate the digest's
// NeedsAttention uses, so the pane and the `--check` exit code agree.
func guardInfoWatchdogAttentionCount(m map[string]string) int {
	n := 0
	for _, st := range m {
		if watchdogNeedsAttention(st) {
			n++
		}
	}
	return n
}

// guardInfoWatchdogMonitorsText renders the per-monitor statuses as glyphed chips, attention-first
// then by name, capped at limit with a "+K more" tail so a wide monitor set can never scroll the
// pane. Empty for an empty map.
func guardInfoWatchdogMonitorsText(m map[string]string, limit int) string {
	if len(m) == 0 {
		return ""
	}
	type mon struct{ name, status string }
	rows := make([]mon, 0, len(m))
	for name, st := range m {
		rows = append(rows, mon{name: name, status: st})
	}
	sort.SliceStable(rows, func(i, j int) bool {
		ai, aj := watchdogNeedsAttention(rows[i].status), watchdogNeedsAttention(rows[j].status)
		if ai != aj {
			return ai // monitors needing attention lead
		}
		return rows[i].name < rows[j].name
	})
	shown := rows
	extra := 0
	if limit > 0 && len(rows) > limit {
		shown = rows[:limit]
		extra = len(rows) - limit
	}
	parts := make([]string, 0, len(shown)+1)
	for _, r := range shown {
		parts = append(parts, watchdogGlyph(r.status)+" "+r.name)
	}
	if extra > 0 {
		parts = append(parts, fmt.Sprintf("+%d more", extra))
	}
	return strings.Join(parts, "  ")
}

// guardInfoWatchdogCountMap renders a count map (verdict → n) as the top-`limit` entries by count
// desc, then key asc for a stable order regardless of Go's map iteration, each as "key n", with a
// "+K more" fold past the cap. Empty for an empty/all-zero map.
func guardInfoWatchdogCountMap(m map[string]int64, limit int) string {
	if len(m) == 0 {
		return ""
	}
	type kv struct {
		k string
		n int64
	}
	rows := make([]kv, 0, len(m))
	for k, n := range m {
		if n == 0 {
			continue
		}
		rows = append(rows, kv{k: k, n: n})
	}
	if len(rows) == 0 {
		return ""
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].n != rows[j].n {
			return rows[i].n > rows[j].n
		}
		return rows[i].k < rows[j].k
	})
	shown := rows
	extra := 0
	if limit > 0 && len(rows) > limit {
		shown = rows[:limit]
		extra = len(rows) - limit
	}
	parts := make([]string, 0, len(shown)+1)
	for _, r := range shown {
		parts = append(parts, fmt.Sprintf("%s %d", r.k, r.n))
	}
	if extra > 0 {
		parts = append(parts, fmt.Sprintf("+%d more", extra))
	}
	return strings.Join(parts, " · ")
}

// watchdogStatus normalizes a snapshot status token (lowercased by resumemetrics.norm) back to the
// watchdoghealth.Status vocabulary. An empty token stays empty (no verdict yet); any other token is
// upper-cased so it matches a StatusX const, and an unrecognized one flows through as-is for the
// callers, which fail it safe to the attention band.
func watchdogStatus(token string) watchdoghealth.Status {
	token = strings.TrimSpace(token)
	if token == "" {
		return ""
	}
	return watchdoghealth.Status(strings.ToUpper(token))
}

// watchdogGlyph maps a watchdog status token to its health chip. It fails safe to the attention
// glyph for any unrecognized non-empty token — an unknown status is a reason to look, never to
// relax — and to a neutral chip for an empty (no-verdict) token.
func watchdogGlyph(token string) string {
	switch watchdogStatus(token) {
	case watchdoghealth.StatusHealthy:
		return guardWatchdogChipHealthy
	case watchdoghealth.StatusNotInstalled:
		return guardWatchdogChipAbsent
	case watchdoghealth.StatusHealing:
		return guardWatchdogChipHealing
	case watchdoghealth.StatusGaveUp:
		return guardWatchdogChipGaveUp
	case watchdoghealth.StatusDown, watchdoghealth.StatusUnknown:
		return guardWatchdogChipAttention
	case "":
		return guardWatchdogChipAbsent
	default:
		return guardWatchdogChipAttention
	}
}

// watchdogRollupWord renders a status token as the plain word the pane shows. GAVE_UP carries its
// "needs a human" note inline so the mini form alone tells the operator to act; an empty token is
// the honest "no verdict yet".
func watchdogRollupWord(token string) string {
	switch watchdogStatus(token) {
	case watchdoghealth.StatusHealthy:
		return "healthy"
	case watchdoghealth.StatusNotInstalled:
		return "not installed"
	case watchdoghealth.StatusHealing:
		return "healing"
	case watchdoghealth.StatusDown:
		return "down"
	case watchdoghealth.StatusGaveUp:
		return "gave up (needs a human)"
	case watchdoghealth.StatusUnknown:
		return "unknown"
	case "":
		return "no verdict yet"
	default:
		return strings.ToLower(strings.TrimSpace(token))
	}
}

// watchdogNeedsAttention reports whether a status token is at or above the watchdoghealth attention
// floor. It defers to watchdoghealth.Health.NeedsAttention so the pane's gate is exactly the
// digest's; an empty token is not attention (no verdict is not a known-bad state).
func watchdogNeedsAttention(token string) bool {
	s := watchdogStatus(token)
	if s == "" {
		return false
	}
	return watchdoghealth.Health{Status: s}.NeedsAttention()
}
