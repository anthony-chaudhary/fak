package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/resumemetrics"
)

// watchdogFixture builds a snapshot in the wire shape the gateway emits: the tokens are the
// watchdoghealth vocabulary AS resumemetrics.norm stores them (lower-cased), because the real
// producer writes string(digest.Rollup)/string(h.Status) through norm (watchdog_cli.go).
func watchdogFixture() *guardInfoWatchdog {
	return &resumemetrics.Snapshot{
		Ticks:             142,
		ProgressWitnessed: 3,
		HealthRollup:      "down",
		Actions:           map[string]int64{"launch": 3, "skip": 138, "defer": 1},
		AutohealResults:   map[string]int64{"restarted": 1, "healthy": 2},
		MonitorStatus:     map[string]string{"fleet-resume-watchdog": "healthy", "drain-steward": "down"},
	}
}

func TestGuardInfoWatchdogPanelFull(t *testing.T) {
	ctx := guardInfoPanelCtx{v: guardInfoVars{Watchdog: watchdogFixture()}, width: 120}
	rows := guardInfoWatchdogPanelRows(ctx, guardPanelFull)
	joined := strings.Join(rows, "\n")
	// Head: rollup glyph+word, the tick alive-proof, and the proven-resume count.
	if !strings.Contains(joined, guardWatchdogChipAttention+" down") {
		t.Fatalf("head missing rollup glyph+word:\n%s", joined)
	}
	if !strings.Contains(joined, "142 ticks") || !strings.Contains(joined, "3 resumed") {
		t.Fatalf("head missing tick/resume counts:\n%s", joined)
	}
	// Actions row: top-by-count desc, "key n".
	if !strings.Contains(joined, "actions") || !strings.Contains(joined, "skip 138 · launch 3 · defer 1") {
		t.Fatalf("actions row missing/misordered:\n%s", joined)
	}
	// Autoheal row.
	if !strings.Contains(joined, "autoheal") || !strings.Contains(joined, "healthy 2 · restarted 1") {
		t.Fatalf("autoheal row missing/misordered:\n%s", joined)
	}
	// Monitors row: the attention-needing monitor (down) leads the healthy one.
	monLine := ""
	for _, r := range rows {
		if strings.Contains(r, "monitors") {
			monLine = r
		}
	}
	if monLine == "" {
		t.Fatalf("no monitors row:\n%s", joined)
	}
	if !strings.Contains(monLine, guardWatchdogChipAttention+" drain-steward") ||
		!strings.Contains(monLine, guardWatchdogChipHealthy+" fleet-resume-watchdog") {
		t.Fatalf("monitors row missing glyphed chips: %q", monLine)
	}
	if strings.Index(monLine, "drain-steward") > strings.Index(monLine, "fleet-resume-watchdog") {
		t.Fatalf("attention-needing monitor must lead: %q", monLine)
	}
}

func TestGuardInfoWatchdogPanelMini(t *testing.T) {
	ctx := guardInfoPanelCtx{v: guardInfoVars{Watchdog: watchdogFixture()}, width: 80}
	rows := guardInfoWatchdogPanelRows(ctx, guardPanelMini)
	if len(rows) != 1 {
		t.Fatalf("mini watchdog = %d rows, want exactly 1", len(rows))
	}
	line := rows[0]
	if !strings.Contains(line, guardWatchdogChipAttention+" down") || !strings.Contains(line, "142 ticks") {
		t.Fatalf("mini line missing rollup/ticks: %q", line)
	}
	if !strings.Contains(line, "1 need attention") {
		t.Fatalf("mini line missing attention fold: %q", line)
	}
}

func TestGuardInfoWatchdogPanelSilentWhenAbsent(t *testing.T) {
	// No block at all is silent.
	ctx := guardInfoPanelCtx{v: guardInfoVars{}, width: 80}
	if rows := guardInfoWatchdogPanelRows(ctx, guardPanelFull); rows != nil {
		t.Fatalf("watchdog panel with no block = %v, want nil (silent)", rows)
	}
	// A present-but-all-zero snapshot is also silent (mirrors resumemetrics.Active()).
	ctx = guardInfoPanelCtx{v: guardInfoVars{Watchdog: &resumemetrics.Snapshot{}}, width: 80}
	if rows := guardInfoWatchdogPanelRows(ctx, guardPanelFull); rows != nil {
		t.Fatalf("watchdog panel with empty snapshot = %v, want nil (silent)", rows)
	}
}

// TestGuardInfoWatchdogMiniRollupAttention proves the mini form still calls out a bad rollup even
// when no per-monitor status was published (an empty MonitorStatus map).
func TestGuardInfoWatchdogMiniRollupAttention(t *testing.T) {
	wd := &resumemetrics.Snapshot{Ticks: 5, HealthRollup: "gave_up"}
	line := guardInfoWatchdogMiniText(wd)
	if !strings.Contains(line, "gave up (needs a human)") {
		t.Fatalf("mini missing gave-up wording: %q", line)
	}
	if !strings.Contains(line, "needs attention") {
		t.Fatalf("mini must flag rollup attention with no monitors: %q", line)
	}
}

func TestGuardInfoWatchdogLineText(t *testing.T) {
	if got := guardInfoWatchdogText(nil); got != "" {
		t.Fatalf("nil watchdog line = %q, want empty", got)
	}
	got := guardInfoWatchdogText(watchdogFixture())
	for _, want := range []string{"watchdog: down", "142 ticks", "3 resumed", "1 need attention"} {
		if !strings.Contains(got, want) {
			t.Fatalf("line %q missing %q", got, want)
		}
	}
	// A healthy watchdog with no attention reads clean — no "needs attention" tail.
	healthy := &resumemetrics.Snapshot{Ticks: 10, HealthRollup: "healthy", ProgressWitnessed: 0,
		MonitorStatus: map[string]string{"m": "healthy"}}
	if got := guardInfoWatchdogText(healthy); strings.Contains(got, "attention") || strings.Contains(got, "resumed") {
		t.Fatalf("healthy line should be clean: %q", got)
	}
}

// TestWatchdogGlyphVocabulary pins the glyph for every closed status, that an empty token is the
// neutral chip, and that an unrecognized token fails safe to the attention chip (never healthy).
func TestWatchdogGlyphVocabulary(t *testing.T) {
	cases := map[string]string{
		"healthy":       guardWatchdogChipHealthy,
		"not_installed": guardWatchdogChipAbsent,
		"healing":       guardWatchdogChipHealing,
		"down":          guardWatchdogChipAttention,
		"unknown":       guardWatchdogChipAttention,
		"gave_up":       guardWatchdogChipGaveUp,
		"":              guardWatchdogChipAbsent,
		"bogus-token":   guardWatchdogChipAttention,
	}
	for tok, want := range cases {
		if got := watchdogGlyph(tok); got != want {
			t.Fatalf("watchdogGlyph(%q) = %q, want %q", tok, got, want)
		}
	}
}

// TestWatchdogNeedsAttention pins the attention gate against the watchdoghealth attention floor:
// DOWN / UNKNOWN / GAVE_UP owe attention; HEALTHY / NOT_INSTALLED / HEALING and an empty token do
// not. This is the same floor the digest's --check exit code uses, so pane and check agree.
func TestWatchdogNeedsAttention(t *testing.T) {
	attention := []string{"down", "unknown", "gave_up"}
	calm := []string{"healthy", "not_installed", "healing", ""}
	for _, tok := range attention {
		if !watchdogNeedsAttention(tok) {
			t.Fatalf("watchdogNeedsAttention(%q) = false, want true", tok)
		}
	}
	for _, tok := range calm {
		if watchdogNeedsAttention(tok) {
			t.Fatalf("watchdogNeedsAttention(%q) = true, want false", tok)
		}
	}
}

// TestWatchdogCountMapFold pins the count-map renderer: count-desc then key-asc ordering, and a
// "+K more" fold past the cap.
func TestWatchdogCountMapFold(t *testing.T) {
	m := map[string]int64{"a": 1, "b": 5, "c": 5, "d": 2, "e": 9, "zero": 0}
	got := guardInfoWatchdogCountMap(m, 3)
	// e(9) leads; b/c tie at 5 break by key (b before c); zero-count entry dropped; two folded.
	if !strings.HasPrefix(got, "e 9 · b 5 · c 5") {
		t.Fatalf("count map order = %q, want e/b/c prefix", got)
	}
	if !strings.Contains(got, "+2 more") {
		t.Fatalf("count map missing fold tail: %q", got)
	}
	if strings.Contains(got, "zero") {
		t.Fatalf("count map must drop zero-count entries: %q", got)
	}
	if guardInfoWatchdogCountMap(nil, 3) != "" || guardInfoWatchdogCountMap(map[string]int64{"x": 0}, 3) != "" {
		t.Fatalf("empty/all-zero count map must render empty")
	}
}

// TestWatchdogMonitorsFold pins the "+K more" fold and the attention-first ordering across a wide
// monitor set.
func TestWatchdogMonitorsFold(t *testing.T) {
	m := map[string]string{
		"a-healthy": "healthy", "b-healthy": "healthy",
		"c-down": "down", "d-gaveup": "gave_up",
	}
	got := guardInfoWatchdogMonitorsText(m, 2)
	// Both attention monitors lead (order by name among equal attention): c-down, d-gaveup.
	if !strings.HasPrefix(got, guardWatchdogChipAttention+" c-down") {
		t.Fatalf("attention monitor must lead: %q", got)
	}
	if !strings.Contains(got, "+2 more") {
		t.Fatalf("monitors missing fold tail: %q", got)
	}
	if guardInfoWatchdogMonitorsText(nil, 2) != "" {
		t.Fatalf("empty monitor map must render empty")
	}
}
