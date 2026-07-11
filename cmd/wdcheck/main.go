// Command wdcheck is a THROWAWAY verifier for the info_watchdog.go pure mappers, run only
// because cmd/fak currently cannot build (unrelated in-flight `wip land` work). It copies the
// pure functions verbatim and exercises them against the REAL watchdoghealth + resumemetrics
// types, so the watchdoghealth attention-floor integration and the fold ordering are genuinely
// checked. Delete after use.
package main

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/resumemetrics"
	"github.com/anthony-chaudhary/fak/internal/watchdoghealth"
)

const (
	guardWatchdogChipHealthy   = "●"
	guardWatchdogChipAbsent    = "○"
	guardWatchdogChipHealing   = "◐"
	guardWatchdogChipAttention = "⚠"
	guardWatchdogChipGaveUp    = "✗"
)

func watchdogStatus(token string) watchdoghealth.Status {
	token = strings.TrimSpace(token)
	if token == "" {
		return ""
	}
	return watchdoghealth.Status(strings.ToUpper(token))
}

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

func watchdogNeedsAttention(token string) bool {
	s := watchdogStatus(token)
	if s == "" {
		return false
	}
	return watchdoghealth.Health{Status: s}.NeedsAttention()
}

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
			return ai
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

func main() {
	fails := 0
	check := func(name, got, want string) {
		if got != want {
			fmt.Printf("FAIL %s: got %q want %q\n", name, got, want)
			fails++
		}
	}
	checkBool := func(name string, got, want bool) {
		if got != want {
			fmt.Printf("FAIL %s: got %v want %v\n", name, got, want)
			fails++
		}
	}

	// Glyph vocabulary, incl. empty (neutral) and unknown-token fail-safe (attention).
	check("glyph healthy", watchdogGlyph("healthy"), guardWatchdogChipHealthy)
	check("glyph not_installed", watchdogGlyph("not_installed"), guardWatchdogChipAbsent)
	check("glyph healing", watchdogGlyph("healing"), guardWatchdogChipHealing)
	check("glyph down", watchdogGlyph("down"), guardWatchdogChipAttention)
	check("glyph unknown", watchdogGlyph("unknown"), guardWatchdogChipAttention)
	check("glyph gave_up", watchdogGlyph("gave_up"), guardWatchdogChipGaveUp)
	check("glyph empty", watchdogGlyph(""), guardWatchdogChipAbsent)
	check("glyph bogus", watchdogGlyph("bogus"), guardWatchdogChipAttention)

	// Rollup words.
	check("word gave_up", watchdogRollupWord("gave_up"), "gave up (needs a human)")
	check("word empty", watchdogRollupWord(""), "no verdict yet")
	check("word healthy", watchdogRollupWord("healthy"), "healthy")

	// Attention gate == watchdoghealth attention floor (DOWN/UNKNOWN/GAVE_UP).
	checkBool("attn down", watchdogNeedsAttention("down"), true)
	checkBool("attn unknown", watchdogNeedsAttention("unknown"), true)
	checkBool("attn gave_up", watchdogNeedsAttention("gave_up"), true)
	checkBool("attn healthy", watchdogNeedsAttention("healthy"), false)
	checkBool("attn not_installed", watchdogNeedsAttention("not_installed"), false)
	checkBool("attn healing", watchdogNeedsAttention("healing"), false)
	checkBool("attn empty", watchdogNeedsAttention(""), false)

	// Monitor fold: attention-first, then name; "+K more" cap.
	mon := guardInfoWatchdogMonitorsText(map[string]string{
		"a-healthy": "healthy", "b-healthy": "healthy", "c-down": "down", "d-gaveup": "gave_up",
	}, 2)
	if !strings.HasPrefix(mon, guardWatchdogChipAttention+" c-down") || !strings.Contains(mon, "+2 more") {
		fmt.Printf("FAIL monitors fold: %q\n", mon)
		fails++
	}

	// Count fold: count-desc then key-asc, zero dropped, "+K more".
	cm := guardInfoWatchdogCountMap(map[string]int64{"a": 1, "b": 5, "c": 5, "d": 2, "e": 9, "zero": 0}, 3)
	if !strings.HasPrefix(cm, "e 9 · b 5 · c 5") || !strings.Contains(cm, "+2 more") || strings.Contains(cm, "zero") {
		fmt.Printf("FAIL count fold: %q\n", cm)
		fails++
	}

	// Prove the wire shape decodes into resumemetrics.Snapshot as the alias assumes.
	var _ *resumemetrics.Snapshot = &resumemetrics.Snapshot{
		Ticks: 1, ProgressWitnessed: 1, HealthRollup: "down",
		Actions: map[string]int64{"launch": 1}, AutohealResults: map[string]int64{"restarted": 1},
		MonitorStatus: map[string]string{"m": "down"},
	}

	if fails == 0 {
		fmt.Println("ALL WATCHDOG MAPPER CHECKS PASSED")
	} else {
		fmt.Printf("%d CHECK(S) FAILED\n", fails)
		os.Exit(1)
	}
}
