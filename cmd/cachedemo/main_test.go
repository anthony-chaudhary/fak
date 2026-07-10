// Tests for cachedemo's pure helpers: the thousands-grouping formatters, the
// ratio/reason formatters, the since-date ledger filters, the spine picker, and
// the substring-level contract of the narrative renderers (per-fire honesty
// labels, nil-spine fallback, since-window label). No ledger files are read and
// main() is never invoked — every fixture is constructed in-memory.
package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

func TestGroupThousands(t *testing.T) {
	cases := []struct{ in, want string }{
		{"0", "0"},
		{"7", "7"},
		{"999", "999"},
		{"1000", "1,000"},
		{"12345", "12,345"},
		{"123456", "123,456"},
		{"1234567", "1,234,567"},
		{"-12", "-12"},
		{"-1234", "-1,234"},
		{"-1234567", "-1,234,567"},
	}
	for _, c := range cases {
		if got := groupThousands(c.in); got != c.want {
			t.Errorf("groupThousands(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestCommasAndCommasF(t *testing.T) {
	if got := commas(0); got != "0" {
		t.Errorf("commas(0) = %q, want %q", got, "0")
	}
	if got := commas(9876543); got != "9,876,543" {
		t.Errorf("commas(9876543) = %q, want %q", got, "9,876,543")
	}
	// commasF renders %.0f then groups: fractional input rounds to an integer string.
	if got := commasF(1234567.0); got != "1,234,567" {
		t.Errorf("commasF(1234567.0) = %q, want %q", got, "1,234,567")
	}
	if got := commasF(999.4); got != "999" {
		t.Errorf("commasF(999.4) = %q, want %q", got, "999")
	}
}

func TestRatio(t *testing.T) {
	if got := ratio(10, 4); got != 2.5 {
		t.Errorf("ratio(10, 4) = %v, want 2.5", got)
	}
	if got := ratio(0, 5); got != 0 {
		t.Errorf("ratio(0, 5) = %v, want 0", got)
	}
	// Division-by-zero guard: a zero denominator must yield 0, not +Inf/NaN.
	if got := ratio(7, 0); got != 0 {
		t.Errorf("ratio(7, 0) = %v, want 0", got)
	}
}

func TestFmtReasons(t *testing.T) {
	// Keys must come out sorted so the rendered line is deterministic.
	got := fmtReasons(map[string]uint64{"under_budget": 3, "burst_unprofitable": 1, "anchor": 2})
	want := "anchor=2 burst_unprofitable=1 under_budget=3"
	if got != want {
		t.Errorf("fmtReasons = %q, want %q", got, want)
	}
	if got := fmtReasons(nil); got != "" {
		t.Errorf("fmtReasons(nil) = %q, want empty", got)
	}
}

func TestFilterTrack1Since(t *testing.T) {
	rows := []cachevalueledger.Row{
		{Date: "2026-07-01"},
		{Date: "2026-07-04"},
		{Date: "2026-07-05"},
	}
	got := filterTrack1Since(rows, "2026-07-04")
	if len(got) != 2 {
		t.Fatalf("filterTrack1Since len = %d, want 2 (boundary date inclusive)", len(got))
	}
	if got[0].Date != "2026-07-04" || got[1].Date != "2026-07-05" {
		t.Errorf("filterTrack1Since kept %q, %q; want the on-or-after rows in order", got[0].Date, got[1].Date)
	}
	// The input slice must not be mutated (out is built on a fresh backing array).
	if rows[0].Date != "2026-07-01" || rows[1].Date != "2026-07-04" || rows[2].Date != "2026-07-05" {
		t.Errorf("filterTrack1Since mutated its input: %+v", rows)
	}
	if all := filterTrack1Since(rows, ""); len(all) != 3 {
		t.Errorf("filterTrack1Since with empty since kept %d rows, want all 3", len(all))
	}
	if none := filterTrack1Since(rows, "2026-08-01"); len(none) != 0 {
		t.Errorf("filterTrack1Since with a future since kept %d rows, want 0", len(none))
	}
}

func TestFilterTrack2Since(t *testing.T) {
	rows := []cachevaluereport.SavingsRow{
		{Date: "2026-06-30"},
		{Date: "2026-07-04"},
	}
	got := filterTrack2Since(rows, "2026-07-04")
	if len(got) != 1 || got[0].Date != "2026-07-04" {
		t.Fatalf("filterTrack2Since = %+v, want exactly the boundary row", got)
	}
	if all := filterTrack2Since(rows, ""); len(all) != 2 {
		t.Errorf("filterTrack2Since with empty since kept %d rows, want all 2", len(all))
	}
}

func spineFixtureRows() []gatewayusageledger.Row {
	return []gatewayusageledger.Row{
		// Rich serve session: must be skipped, only guard sessions qualify.
		{SessionType: "serve", PID: 100, Counters: gatewayusageledger.Counters{
			CachedTurns: 50, CompactionFired: 50, CompactionShedTokens: 1000, CachedPromptTokens: 1000,
		}},
		// Qualifying guard, score = 5 + 1 = 6.
		{SessionType: "guard", PID: 200, Counters: gatewayusageledger.Counters{
			CachedTurns: 5, CompactionFired: 1, CompactionShedTokens: 100, CachedPromptTokens: 10,
		}},
		// Qualifying guard, score = 3 + 10 = 13 — the auto-pick winner.
		{SessionType: "guard", PID: 300, Counters: gatewayusageledger.Counters{
			CachedTurns: 3, CompactionFired: 10, CompactionShedTokens: 100, CachedPromptTokens: 10,
		}},
		// Guard with a huge score but CachedTurns < 2: disqualified from auto-pick.
		{SessionType: "guard", PID: 400, Counters: gatewayusageledger.Counters{
			CachedTurns: 1, CompactionFired: 100, CompactionShedTokens: 100, CachedPromptTokens: 10,
		}},
		// Guard with zero cache/compaction activity: only reachable via a pin.
		{SessionType: "guard", PID: 777, Counters: gatewayusageledger.Counters{}},
	}
}

func TestPickSpine(t *testing.T) {
	rows := spineFixtureRows()

	t.Run("empty ledger yields nil", func(t *testing.T) {
		if got := pickSpine(nil, 0); got != nil {
			t.Errorf("pickSpine(nil, 0) = %+v, want nil", got)
		}
	})

	t.Run("auto-pick takes the highest cached-turns+fired score among qualifying guards", func(t *testing.T) {
		got := pickSpine(rows, 0)
		if got == nil {
			t.Fatal("pickSpine auto-pick = nil, want the pid-300 row")
		}
		if got.PID != 300 {
			t.Errorf("pickSpine auto-pick chose pid %d, want 300", got.PID)
		}
		if got != &rows[2] {
			t.Errorf("pickSpine must return a pointer into the input slice, not a copy")
		}
	})

	t.Run("pin bypasses the qualification gate", func(t *testing.T) {
		got := pickSpine(rows, 777)
		if got == nil || got.PID != 777 {
			t.Fatalf("pickSpine(rows, 777) = %+v, want the pinned zero-activity guard row", got)
		}
	})

	t.Run("pin to an absent pid yields nil", func(t *testing.T) {
		if got := pickSpine(rows, 99999); got != nil {
			t.Errorf("pickSpine(rows, 99999) = %+v, want nil", got)
		}
	})

	t.Run("pin never matches a non-guard session", func(t *testing.T) {
		if got := pickSpine(rows, 100); got != nil {
			t.Errorf("pickSpine(rows, 100) = %+v, want nil (pid 100 is a serve row)", got)
		}
	})
}

func TestRenderDemoNilSpine(t *testing.T) {
	out := renderDemo(cachevaluereport.FleetBenefitReport{}, nil, "")
	if !strings.Contains(out, "fak — shared cache savings, multi-turn demo") {
		t.Errorf("renderDemo missing the header banner:\n%s", out)
	}
	if !strings.Contains(out, "PER-TURN SPINE: no qualifying multi-turn guard session") {
		t.Errorf("renderDemo with nil spine must print the no-session fallback:\n%s", out)
	}
	if strings.Contains(out, "savings rows since") {
		t.Errorf("renderDemo with empty since must not print a since window:\n%s", out)
	}
}

func TestRenderSpinePerFireHonesty(t *testing.T) {
	row := gatewayusageledger.Row{
		SessionType: "guard",
		PID:         4242,
		GeneratedAt: "2026-07-04T00:00:00Z",
		UptimeSecs:  600,
		Counters: gatewayusageledger.Counters{
			CachedTurns:           4,
			Total:                 20,
			CachedPromptTokens:    50000,
			CacheCreationTokens:   10000,
			CompactionFired:       3,
			CompactionShedTokens:  9000,
			CompactionBailReasons: map[string]uint64{"under_budget": 2, "burst_unprofitable": 1},
		},
	}
	var b strings.Builder
	renderSpine(&b, &row)
	out := b.String()

	// The honest headline is shed PER FIRE = cumulative / fires (9,000 / 3 = 3,000),
	// with the cumulative sum shown only under its explicit label.
	if !strings.Contains(out, "shed PER FIRE ............ ~3,000 tokens") {
		t.Errorf("renderSpine missing the per-fire headline:\n%s", out)
	}
	if !strings.Contains(out, "9,000 cumulative ÷ 3 fires") {
		t.Errorf("renderSpine missing the labeled cumulative sum:\n%s", out)
	}
	// read:write ratio = 50000/10000 = 5.0.
	if !strings.Contains(out, "read : write = 5.0 : 1") {
		t.Errorf("renderSpine missing the read:write ratio:\n%s", out)
	}
	// Bail reasons render sorted and deterministic.
	if !strings.Contains(out, "bail reasons ............. burst_unprofitable=1 under_budget=2") {
		t.Errorf("renderSpine missing the sorted bail reasons:\n%s", out)
	}
	// Fired > 0 and multi-turn: the multi-turn effect paragraph must appear.
	if !strings.Contains(out, "Multi-turn effect: fak fired compaction 3 times over 4 turns") {
		t.Errorf("renderSpine missing the multi-turn effect paragraph:\n%s", out)
	}
}

func TestRenderSpineNoFires(t *testing.T) {
	row := gatewayusageledger.Row{
		SessionType: "guard",
		PID:         1,
		Counters: gatewayusageledger.Counters{
			CachedTurns:          2,
			CompactionShedTokens: 500,
		},
	}
	var b strings.Builder
	renderSpine(&b, &row)
	out := b.String()
	if !strings.Contains(out, "shed (cumulative) ........ 500 tokens") {
		t.Errorf("renderSpine with zero fires must fall back to the cumulative label:\n%s", out)
	}
	if strings.Contains(out, "shed PER FIRE") {
		t.Errorf("renderSpine with zero fires must not print a per-fire line:\n%s", out)
	}
	if strings.Contains(out, "Multi-turn effect") {
		t.Errorf("renderSpine with zero fires must not print the multi-turn paragraph:\n%s", out)
	}
}

func TestRenderFleetSinceAndShare(t *testing.T) {
	share := 12.3
	rep := cachevaluereport.FleetBenefitReport{
		UsageRows:    7,
		ExitSessions: 5,
		FakSharePct:  &share,
	}
	var b strings.Builder
	renderFleet(&b, rep, "2026-07-04")
	out := b.String()
	if !strings.Contains(out, "CUMULATIVE OWNER SPLIT (savings rows since 2026-07-04)") {
		t.Errorf("renderFleet missing the since window label:\n%s", out)
	}
	if !strings.Contains(out, "sessions folded .......... 7 usage rows, 5 exit sessions") {
		t.Errorf("renderFleet missing the folded-sessions line:\n%s", out)
	}
	if !strings.Contains(out, "fak share (count-axis) ... 12.3%") {
		t.Errorf("renderFleet missing the fak-share line for a non-nil share:\n%s", out)
	}

	// A nil share must suppress the share line entirely.
	var b2 strings.Builder
	renderFleet(&b2, cachevaluereport.FleetBenefitReport{}, "")
	if strings.Contains(b2.String(), "fak share (count-axis)") {
		t.Errorf("renderFleet with nil FakSharePct must omit the share line:\n%s", b2.String())
	}
}
