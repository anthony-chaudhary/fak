package main

// guard_defer_inert_test.go — the banner half of the #3621 DEFER_ENABLED_BUT_INERT watchdog
// (the /debug/vars half is witnessed in internal/gateway/defer_inert_test.go). The exit banner
// is the artifact an operator actually reads after a session, so an armed-but-never-bit
// cold-tool-defer lever has to name itself THERE, not only on a scrape nobody pulled.

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func TestFormatAuditSummaryDeferEnabledButInert(t *testing.T) {
	inert := formatAuditSummary(gateway.AdjudicationSummary{
		Total: 6, Allowed: 6,
		DeferStandDownTurns:   4,
		DeferStandDownReasons: map[string]uint64{"no_cold_tools": 3, "already_deferred": 1},
	})
	for _, want := range []string{
		"cold-tool deferral", "DEFER_ENABLED_BUT_INERT",
		"0 cold def(s) deferred across 4 eligible turn(s)",
		"stood down: already_deferred", "stood down: no_cold_tools",
		"cache_attribution.fak_defer_finding",
	} {
		if !strings.Contains(inert, want) {
			t.Errorf("inert-defer banner missing %q:\n%s", want, inert)
		}
	}

	// A healthy defer session prints the SHED row and never the alarm — the done-condition's
	// negative half. One deferred def clears the finding however many stand-downs surround it.
	healthy := formatAuditSummary(gateway.AdjudicationSummary{
		Total: 6, Allowed: 6,
		DeferColdTurns: 2, DeferColdCount: 9,
		DeferStandDownTurns:   4,
		DeferStandDownReasons: map[string]uint64{"no_cold_tools": 4},
	})
	if strings.Contains(healthy, "DEFER_ENABLED_BUT_INERT") {
		t.Errorf("banner raised the finding on a session that deferred 9 cold defs:\n%s", healthy)
	}
	if !strings.Contains(healthy, "9 cold tool def(s)") {
		t.Errorf("healthy session lost its deferral shed row:\n%s", healthy)
	}

	// Below the eligible-turn floor a zero-defer session is short, not inert.
	short := formatAuditSummary(gateway.AdjudicationSummary{
		Total: 2, Allowed: 2,
		DeferStandDownTurns:   2,
		DeferStandDownReasons: map[string]uint64{"no_cold_tools": 2},
	})
	if strings.Contains(short, "DEFER_ENABLED_BUT_INERT") {
		t.Errorf("banner raised the finding below the eligible-turn floor:\n%s", short)
	}

	// A session that never armed the lever books no eligible turns at all, so the banner stays
	// quiet — the fail-safe that keeps the default-off / ablated / non-Anthropic case silent.
	unarmed := formatAuditSummary(gateway.AdjudicationSummary{Total: 9, Allowed: 9})
	if strings.Contains(unarmed, "DEFER_ENABLED_BUT_INERT") {
		t.Errorf("banner raised the finding on a session with no eligible defer turns:\n%s", unarmed)
	}
}
