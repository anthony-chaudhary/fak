package loopscore

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// base is a fixed clock so the fold is deterministic. Events stamped "old" relative
// to it are dark; events stamped "recent" are live.
var base = time.Date(2026, 6, 29, 12, 0, 0, 0, time.UTC)

func ev(loop string, kind loopmgr.EventKind, ts time.Time, status loopmgr.RunStatus, metrics map[string]int64) loopmgr.Event {
	return loopmgr.Event{
		LoopID:     loop,
		Kind:       kind,
		TSUnixNano: ts.UnixNano(),
		Status:     status,
		Metrics:    metrics,
	}
}

func job(id string, interval int64, state loopmgr.JobState) loopmgr.Job {
	return loopmgr.Job{
		Schedule: loopmgr.Schedule{JobID: id, IntervalSeconds: interval},
		State:    state,
	}
}

func reg(jobs ...loopmgr.Job) *loopmgr.Registry {
	r := loopmgr.Registry{Schema: loopmgr.SchemaRegistry, Jobs: map[string]loopmgr.Job{}}
	for _, j := range jobs {
		r.Jobs[j.JobID()] = j
	}
	return &r
}

func buildWith(events []loopmgr.Event, registry *loopmgr.Registry) ScorecardPayload {
	return Build(Options{Now: base, useInputs: true, events: events, registry: registry})
}

func hasKPI(rows []KPIResult, key string) (KPIResult, bool) {
	for _, r := range rows {
		if r.Key == key {
			return r, true
		}
	}
	return KPIResult{}, false
}

func mustKPI(t *testing.T, p ScorecardPayload, key string) KPIResult {
	t.Helper()
	for _, group := range [][]KPIResult{p.Durability, p.SelfReport, p.Dogfood} {
		if r, ok := hasKPI(group, key); ok {
			return r
		}
	}
	t.Fatalf("kpi %q not found", key)
	return KPIResult{}
}

// TestFragileCorpusReds: a fleet of unregistered fire-and-forget loops + a registered
// job that never ticked + a run that ran with --no-guard reds across all three axes —
// the honest shape this tree starts from.
func TestFragileCorpusReds(t *testing.T) {
	old := base.Add(-10 * 24 * time.Hour) // far past the dark horizon
	var events []loopmgr.Event
	// A firing dispatcher, unregistered, mostly fire-and-forget (4 fires, 1 end), no guard.
	for i := 0; i < 4; i++ {
		events = append(events, ev("dispatch/x", loopmgr.EventFire, old, "", nil))
		events = append(events, ev("dispatch/x", loopmgr.EventAdmit, old, loopmgr.StatusAdmitted, nil))
	}
	events = append(events, ev("dispatch/x", loopmgr.EventEnd, old, loopmgr.StatusClaimedDone, nil))
	// A smoke loop that DID run through `fak loop run` (argc) but with --no-guard (guard_enabled=0).
	runMetrics := map[string]int64{"argc": 3, "guard_enabled": 0}
	for i := 0; i < 2; i++ {
		events = append(events, ev("smoke/x", loopmgr.EventFire, old, "", runMetrics))
		events = append(events, ev("smoke/x", loopmgr.EventStart, old, loopmgr.StatusRunning, runMetrics))
		events = append(events, ev("smoke/x", loopmgr.EventEnd, old, loopmgr.StatusClaimedDone, runMetrics))
	}
	// A registered job that has never been observed ticking -> dark.
	registry := reg(job("freshness", 86400, loopmgr.JobArmed))

	p := buildWith(events, registry)

	if p.OK {
		t.Fatalf("fragile corpus must red, got OK; reason=%s", p.Reason)
	}
	debt := anyInt(p.Corpus["loopscore_debt"])
	if debt < 3 {
		t.Fatalf("fragile corpus debt = %d, want >=3 (durability+self-report+dogfood gaps)", debt)
	}
	for _, key := range []string{"firing_loops_registered", "no_dark_loop", "outcome_recorded", "guard_wrapped"} {
		if r := mustKPI(t, p, key); r.Passed {
			t.Errorf("expected HARD gap %q to fail on the fragile corpus, but it passed: %s", key, r.Detail)
		}
	}
	if ev := p.Evidence; ev.FiredRegistered != 0 || ev.Dark < 3 || ev.GuardWrapped != 0 {
		t.Errorf("fragile evidence unexpected: fired_registered=%d dark=%d guard_wrapped=%d", ev.FiredRegistered, ev.Dark, ev.GuardWrapped)
	}
}

// TestDurableCorpusGreens: every firing loop is registered+armed, fires recently,
// runs under guard, records ends, and at least one witnesses — zero hard debt.
func TestDurableCorpusGreens(t *testing.T) {
	recent := base.Add(-1 * time.Minute)
	guard := map[string]int64{"argc": 2, "guard_enabled": 1}
	var events []loopmgr.Event
	for i := 0; i < 3; i++ {
		events = append(events, ev("dispatch/x", loopmgr.EventFire, recent, "", guard))
		events = append(events, ev("dispatch/x", loopmgr.EventStart, recent, loopmgr.StatusRunning, guard))
		events = append(events, ev("dispatch/x", loopmgr.EventEnd, recent, loopmgr.StatusClaimedDone, guard))
	}
	events = append(events, ev("dispatch/x", loopmgr.EventHeartbeat, recent, "", guard))
	events = append(events, ev("dispatch/x", loopmgr.EventWitness, recent, loopmgr.StatusWitnessedDone, guard))
	for i := 0; i < 2; i++ {
		events = append(events, ev("smoke/x", loopmgr.EventFire, recent, "", guard))
		events = append(events, ev("smoke/x", loopmgr.EventStart, recent, loopmgr.StatusRunning, guard))
		events = append(events, ev("smoke/x", loopmgr.EventEnd, recent, loopmgr.StatusClaimedDone, guard))
	}
	events = append(events, ev("smoke/x", loopmgr.EventNotify, recent, "", guard))
	registry := reg(job("dispatch/x", 300, loopmgr.JobArmed), job("smoke/x", 600, loopmgr.JobArmed))

	p := buildWith(events, registry)

	if !p.OK {
		t.Fatalf("durable corpus must green, got debt=%d reason=%s", anyInt(p.Corpus["loopscore_debt"]), p.Reason)
	}
	if score := anyInt(p.Corpus["score"]); score < 90 {
		t.Errorf("durable composite = %d, want >=90 (grade A)", score)
	}
	if ev := p.Evidence; ev.Dark != 0 || ev.GuardWrapped != 2 || ev.Witnessed != 1 {
		t.Errorf("durable evidence unexpected: dark=%d guard_wrapped=%d witnessed=%d", ev.Dark, ev.GuardWrapped, ev.Witnessed)
	}
}

// TestThreeXLiftDetected: Compare reports a >=3x composite lift fragile -> durable.
func TestThreeXLiftDetected(t *testing.T) {
	old := base.Add(-10 * 24 * time.Hour)
	fragileEvents := []loopmgr.Event{
		ev("dispatch/x", loopmgr.EventFire, old, "", nil),
		ev("dispatch/x", loopmgr.EventFire, old, "", nil),
		ev("smoke/x", loopmgr.EventFire, old, "", map[string]int64{"argc": 2, "guard_enabled": 0}),
		ev("smoke/x", loopmgr.EventEnd, old, loopmgr.StatusClaimedDone, map[string]int64{"argc": 2, "guard_enabled": 0}),
	}
	fragile := buildWith(fragileEvents, reg(job("freshness", 86400, loopmgr.JobArmed)))

	recent := base.Add(-1 * time.Minute)
	guard := map[string]int64{"argc": 2, "guard_enabled": 1}
	durableEvents := []loopmgr.Event{
		ev("dispatch/x", loopmgr.EventFire, recent, "", guard),
		ev("dispatch/x", loopmgr.EventEnd, recent, loopmgr.StatusClaimedDone, guard),
		ev("dispatch/x", loopmgr.EventWitness, recent, loopmgr.StatusWitnessedDone, guard),
		ev("dispatch/x", loopmgr.EventNotify, recent, "", guard),
	}
	durable := buildWith(durableEvents, reg(job("dispatch/x", 300, loopmgr.JobArmed)))

	baseline := map[string]any{"corpus": fragile.Corpus}
	out := Compare(durable, baseline)
	if !strings.Contains(out, "3x composite lift") && !strings.Contains(out, "all loopscore debt retired") {
		t.Errorf("expected a >=3x lift (or full debt retirement) verdict, got:\n%s", out)
	}
}

// TestAbsentLedgerIsNotAFalsePass: an empty ledger (no events) reds on the canonical
// ledger KPI rather than scoring a clean tree.
func TestAbsentLedgerIsNotAFalsePass(t *testing.T) {
	p := buildWith(nil, reg())
	if p.OK {
		t.Fatalf("absent ledger must not be a false pass: %s", p.Reason)
	}
	if r := mustKPI(t, p, "canonical_ledger"); r.Passed {
		t.Errorf("canonical_ledger must fail when the ledger has no events: %s", r.Detail)
	}
}
