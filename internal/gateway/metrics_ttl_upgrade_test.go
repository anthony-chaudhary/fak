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
