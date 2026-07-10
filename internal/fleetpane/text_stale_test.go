package fleetpane

import (
	"strings"
	"testing"
	"time"
)

func staleTestDoc(generated string, staleMin float64) FleetDoc {
	return FleetDoc{
		Schema:       FleetSchema,
		GeneratedUTC: generated,
		StaleMin:     staleMin,
		Verdict:      "OK",
		States:       map[string]int{"OK": 1},
		Totals:       map[string]int{},
		Machines:     []map[string]any{{"id": "peer", "state": "OK"}},
	}
}

func TestFleetTextAtFreshSnapshotShowsAge(t *testing.T) {
	base := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	doc := staleTestDoc(base.Format(time.RFC3339), 15)
	text := FleetTextAt(doc, base.Add(12*time.Second))
	if !strings.Contains(text, "updated 12s ago") {
		t.Fatalf("fresh snapshot missing age suffix:\n%s", text)
	}
	if strings.Contains(text, "STALE_SNAPSHOT") {
		t.Fatalf("fresh snapshot must not flag STALE_SNAPSHOT:\n%s", text)
	}
}

func TestFleetTextAtStaleBoundary(t *testing.T) {
	base := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	doc := staleTestDoc(base.Format(time.RFC3339), 15) // threshold = 15m * 2 = 30m

	stale := FleetTextAt(doc, base.Add(31*time.Minute))
	if !strings.Contains(stale, "STALE_SNAPSHOT") {
		t.Fatalf("31m-old snapshot (threshold 30m) must flag STALE_SNAPSHOT:\n%s", stale)
	}
	if !strings.Contains(stale, "verdict: STALE_SNAPSHOT (frozen: OK)") {
		t.Fatalf("stale verdict line must carry the frozen original verdict:\n%s", stale)
	}

	fresh := FleetTextAt(doc, base.Add(29*time.Minute))
	if strings.Contains(fresh, "STALE_SNAPSHOT") {
		t.Fatalf("29m-old snapshot (threshold 30m) must not flag STALE_SNAPSHOT:\n%s", fresh)
	}
}

func TestFleetTextAtIdleFleetIsNotStale(t *testing.T) {
	base := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	doc := FleetDoc{
		Schema:       FleetSchema,
		GeneratedUTC: base.Format(time.RFC3339),
		StaleMin:     15,
		Verdict:      "OK",
	}
	text := FleetTextAt(doc, base.Add(5*time.Second))
	if strings.Contains(text, "STALE_SNAPSHOT") {
		t.Fatalf("a quiet-but-fresh fleet must never render STALE_SNAPSHOT:\n%s", text)
	}
	if !strings.Contains(text, "updated 5s ago") {
		t.Fatalf("idle fresh snapshot missing age suffix:\n%s", text)
	}
}

func TestFleetSnapshotAgeUnparseable(t *testing.T) {
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	for _, generated := range []string{"", "not-a-time"} {
		doc := staleTestDoc(generated, 15)
		if _, stale, ok := FleetSnapshotAge(doc, now); ok || stale {
			t.Fatalf("GeneratedUTC=%q: want ok=false stale=false, got ok=%v stale=%v", generated, ok, stale)
		}
		text := FleetTextAt(doc, now)
		if strings.Contains(text, "updated") {
			t.Fatalf("GeneratedUTC=%q: unparseable timestamp must omit the age suffix:\n%s", generated, text)
		}
		if strings.Contains(text, "STALE_SNAPSHOT") {
			t.Fatalf("GeneratedUTC=%q: unparseable timestamp must not flag STALE_SNAPSHOT:\n%s", generated, text)
		}
	}
}

func TestFleetSnapshotAgeStaleMinFallback(t *testing.T) {
	base := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	doc := staleTestDoc(base.Format(time.RFC3339), 0) // fallback horizon 15m → threshold 30m
	if _, stale, ok := FleetSnapshotAge(doc, base.Add(31*time.Minute)); !ok || !stale {
		t.Fatalf("31m with fallback horizon: want ok=true stale=true, got ok=%v stale=%v", ok, stale)
	}
	if _, stale, ok := FleetSnapshotAge(doc, base.Add(29*time.Minute)); !ok || stale {
		t.Fatalf("29m with fallback horizon: want ok=true stale=false, got ok=%v stale=%v", ok, stale)
	}
}

func TestHumanizeAge(t *testing.T) {
	cases := []struct {
		d    time.Duration
		want string
	}{
		{12 * time.Second, "12s"},
		{6 * time.Minute, "6m"},
		{2 * time.Hour, "2h"},
		{59 * time.Second, "59s"},
		{90 * time.Second, "1m"},
		{119 * time.Minute, "1h"},
	}
	for _, c := range cases {
		if got := humanizeAge(c.d); got != c.want {
			t.Fatalf("humanizeAge(%v) = %q, want %q", c.d, got, c.want)
		}
	}
}
