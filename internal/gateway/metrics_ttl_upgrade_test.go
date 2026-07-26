package gateway

import (
	"strings"
	"testing"
	"time"
)

// The managed-cache 1h TTL upgrade family must witness every attempt outcome (upgraded +
// each bail reason) and must emit the "upgraded" row even at zero, so an ACTIVE lever with
// no eligible stable head reads as visible-zero rather than an absent panel.
func TestCacheTTLUpgradeMetrics(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeCacheTTLUpgrade("")                     // an actual upgrade
	m.observeCacheTTLUpgrade("")                     // second turn re-upgrades
	m.observeCacheTTLUpgrade("no_stable_breakpoint") // bail
	m.observeCacheTTLUpgrade("volatile_head")        // bail

	var b strings.Builder
	m.writeCompactionMetrics(&b)
	got := b.String()
	for _, want := range []string{
		`fak_gateway_cache_ttl_upgrade_total{outcome="upgraded"} 2`,
		`fak_gateway_cache_ttl_upgrade_total{outcome="no_stable_breakpoint"} 1`,
		`fak_gateway_cache_ttl_upgrade_total{outcome="volatile_head"} 1`,
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("compaction metrics missing %q:\n%s", want, got)
		}
	}
}

// Zero attempts: the "upgraded" row still renders at 0 (panel exists pre-first-attempt),
// and no phantom reason rows appear.
func TestCacheTTLUpgradeMetricsZero(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	var b strings.Builder
	m.writeCompactionMetrics(&b)
	got := b.String()
	if !strings.Contains(got, `fak_gateway_cache_ttl_upgrade_total{outcome="upgraded"} 0`) {
		t.Fatalf("zero-state upgraded row missing:\n%s", got)
	}
	if strings.Count(got, "fak_gateway_cache_ttl_upgrade_total{") != 1 {
		t.Fatalf("unexpected extra ttl-upgrade rows in zero state:\n%s", got)
	}
}

// The TTL-upgrade outcomes must fold into AdjudicationSummary — the exported bridge the
// guard exit banner and the gateway-usage ledger fill read — split into the upgraded
// count and the per-refusal-reason map, so the durable row can carry the same family
// the in-process fak_gateway_cache_ttl_upgrade_total counter witnesses (#1844 C6).
func TestCacheTTLUpgradeFoldsIntoAdjudicationSummary(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeCacheTTLUpgrade("")
	m.observeCacheTTLUpgrade("")
	m.observeCacheTTLUpgrade("no_stable_breakpoint")
	m.observeCacheTTLUpgrade("volatile_head")
	m.observeCacheTTLUpgrade("volatile_head")

	sum := m.adjudicationSummary()
	if sum.CacheTTLUpgraded != 2 {
		t.Fatalf("CacheTTLUpgraded = %d, want 2", sum.CacheTTLUpgraded)
	}
	if got := sum.CacheTTLUpgradeReasons["no_stable_breakpoint"]; got != 1 {
		t.Fatalf("CacheTTLUpgradeReasons[no_stable_breakpoint] = %d, want 1", got)
	}
	if got := sum.CacheTTLUpgradeReasons["volatile_head"]; got != 2 {
		t.Fatalf("CacheTTLUpgradeReasons[volatile_head] = %d, want 2", got)
	}
	if _, present := sum.CacheTTLUpgradeReasons["upgraded"]; present {
		t.Fatalf("upgraded leaked into the refusal-reason map: %v", sum.CacheTTLUpgradeReasons)
	}
}

// A composed place-then-upgrade (#2175) is an AUTHORING outcome, not a refusal: it must
// fold into CacheTTLUpgraded alongside plain "upgraded" and must NEVER appear in the
// refusal-reason map. Booking it as a bail undercounted the authored count and
// double-penalized the fired-rate the cache-health digest folds.
func TestCacheTTLUpgradePlacedAndUpgradedCountsAsAuthored(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeCacheTTLUpgrade("")                               // an existing breakpoint upgraded
	m.observeCacheTTLUpgrade(cacheTTLUpgradePlacedAndUpgraded) // composed place-then-upgrade fire
	m.observeCacheTTLUpgrade(cacheTTLUpgradePlacedAndUpgraded) // a second composed fire
	m.observeCacheTTLUpgrade("volatile_head")                  // a genuine refusal

	sum := m.adjudicationSummary()
	if sum.CacheTTLUpgraded != 3 { // 1 upgraded + 2 placed_and_upgraded
		t.Fatalf("CacheTTLUpgraded = %d, want 3 (upgraded + placed_and_upgraded authored)", sum.CacheTTLUpgraded)
	}
	if _, present := sum.CacheTTLUpgradeReasons[cacheTTLUpgradePlacedAndUpgraded]; present {
		t.Fatalf("placed_and_upgraded leaked into the refusal-reason map: %v", sum.CacheTTLUpgradeReasons)
	}
	if got := sum.CacheTTLUpgradeReasons["volatile_head"]; got != 1 {
		t.Fatalf("CacheTTLUpgradeReasons[volatile_head] = %d, want 1", got)
	}
}

// The #3620 UPGRADE_NEVER_FIRED live watchdog over synthetic outcome counters: an ACTIVE
// session whose every upgrade attempt is refused must raise the finding once the attempt
// floor accrues, and a single fired upgrade must clear it — the running upgraded-vs-refused
// ratio the cache-verify loop keys on.
func TestUpgradeNeverFiredWatchdog(t *testing.T) {
	m := newGatewayMetrics(time.Now())

	// Below the floor: a short session's first refused turns are not an alarm.
	m.observeCacheTTLUpgrade("volatile_head")
	m.observeCacheTTLUpgrade("no_stable_breakpoint")
	if sum := m.adjudicationSummary(); sum.UpgradeNeverFired() {
		t.Fatalf("watchdog fired below the attempt floor: attempts=%d", sum.TTLUpgradeAttempts())
	}

	// At the floor with zero upgrades: the finding fires.
	m.observeCacheTTLUpgrade("volatile_head")
	sum := m.adjudicationSummary()
	if got := sum.TTLUpgradeAttempts(); got != 3 {
		t.Fatalf("TTLUpgradeAttempts = %d, want 3", got)
	}
	if !sum.UpgradeNeverFired() {
		t.Fatalf("watchdog silent at %d refused attempts with 0 upgrades", sum.TTLUpgradeAttempts())
	}

	// One fired upgrade clears it, however many refusals surround it.
	m.observeCacheTTLUpgrade("")
	m.observeCacheTTLUpgrade("volatile_head")
	sum = m.adjudicationSummary()
	if sum.UpgradeNeverFired() {
		t.Fatalf("watchdog still raised after an upgraded outcome: upgraded=%d attempts=%d",
			sum.CacheTTLUpgraded, sum.TTLUpgradeAttempts())
	}
}

// A cold process (lever off or nothing eligible ever attempted) has no outcomes at all and
// must never raise the finding — zero attempts is "unproven", not "never fired".
func TestUpgradeNeverFiredQuietWhenCold(t *testing.T) {
	if sum := newGatewayMetrics(time.Now()).adjudicationSummary(); sum.UpgradeNeverFired() {
		t.Fatal("watchdog fired on a cold session with no upgrade outcomes")
	}
}

// A lever-off session (nothing observed) folds to a zero count and an ABSENT reason map,
// so the ledger row's omitempty keeps the JSON key out and OFF stays distinguishable from
// ON-but-ineligible.
func TestCacheTTLUpgradeSummaryZeroStateAbsent(t *testing.T) {
	sum := newGatewayMetrics(time.Now()).adjudicationSummary()
	if sum.CacheTTLUpgraded != 0 {
		t.Fatalf("CacheTTLUpgraded = %d, want 0", sum.CacheTTLUpgraded)
	}
	if sum.CacheTTLUpgradeReasons != nil {
		t.Fatalf("CacheTTLUpgradeReasons = %v, want nil (absent under omitempty)", sum.CacheTTLUpgradeReasons)
	}
}
