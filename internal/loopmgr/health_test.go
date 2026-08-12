package loopmgr

import (
	"reflect"
	"testing"
	"time"
)

// buildLedger appends a sequence of events to an in-memory chain via Summarize's
// input contract. We construct events directly (not through Append, which needs a
// file) and run them through Summarize, exactly as the production fold does.
func tick(loopID string, kind EventKind, status RunStatus, tsUnixNano int64) Event {
	return Event{
		Schema:     SchemaEvent,
		LoopID:     loopID,
		Kind:       kind,
		Status:     status,
		TSUnixNano: tsUnixNano,
	}
}

func TestFoldHealth_PerLoopRowsAndRates(t *testing.T) {
	now := time.Unix(1_000_000, 0).UTC()
	secAgo := func(s int64) int64 { return now.Add(-time.Duration(s) * time.Second).UnixNano() }

	// "active" — fired recently (10s ago), 2 ended runs, 1 witnessed -> keep rate 0.5,
	// and within its 60s registered cadence -> LIVE.
	// "dark-ledger" — has a ledger entry but last tick is 10000s ago, far past its
	// 60s default horizon -> DARK.
	// "dark-registered" — registered with a 60s cadence but NEVER ticked -> DARK.
	events := []Event{
		tick("active", EventStart, StatusRunning, secAgo(40)),
		tick("active", EventEnd, StatusClaimedDone, secAgo(35)),
		tick("active", EventWitness, StatusWitnessedDone, secAgo(34)),
		tick("active", EventStart, StatusRunning, secAgo(20)),
		tick("active", EventEnd, StatusClaimedDone, secAgo(12)),
		tick("active", EventWitness, StatusWitnessRefused, secAgo(11)),
		{LoopID: "active", Kind: EventFire, TSUnixNano: secAgo(10), Metrics: map[string]int64{MetricLearningDebt: 7}},
		tick("dark-ledger", EventStart, StatusRunning, secAgo(10_050)),
		tick("dark-ledger", EventEnd, StatusClaimedDone, secAgo(10_000)),
	}
	st := Summarize(events, now)

	reg := Registry{Jobs: map[string]Job{}}
	mustPut := func(r *Registry, id string, interval int64) {
		t.Helper()
		if err := r.Put(Job{
			Schedule: Schedule{JobID: id, IntervalSeconds: interval, MissedRun: MissedSkip},
			State:    JobArmed,
		}, now); err != nil {
			t.Fatalf("registry.Put(%s): %v", id, err)
		}
	}
	mustPut(&reg, "active", 60)
	mustPut(&reg, "dark-registered", 60)

	rep := FoldHealth(st, reg, now, HealthThresholds{})

	if rep.Schema != HealthSchema {
		t.Fatalf("schema = %q, want %q", rep.Schema, HealthSchema)
	}
	if rep.TSUnixNano != now.UnixNano() {
		t.Fatalf("ts = %d, want %d", rep.TSUnixNano, now.UnixNano())
	}

	rows := map[string]HealthRow{}
	for _, r := range rep.Rows {
		rows[r.LoopID] = r
	}
	if len(rep.Rows) != 3 {
		t.Fatalf("rows = %d, want 3 (union of ledger+registry): %+v", len(rep.Rows), rep.Rows)
	}

	// Rows must be in stable sorted loop-id order.
	wantOrder := []string{"active", "dark-ledger", "dark-registered"}
	gotOrder := make([]string, len(rep.Rows))
	for i, r := range rep.Rows {
		gotOrder[i] = r.LoopID
	}
	if !reflect.DeepEqual(gotOrder, wantOrder) {
		t.Fatalf("row order = %v, want %v", gotOrder, wantOrder)
	}

	// --- active: LIVE, 2 runs, keep rate 0.5, registered cadence, not dark.
	a := rows["active"]
	if a.State != HealthLive {
		t.Errorf("active.State = %q, want %q", a.State, HealthLive)
	}
	if a.Dark {
		t.Errorf("active.Dark = true, want false (active loop must not be dark)")
	}
	if a.Runs != 2 {
		t.Errorf("active.Runs = %d, want 2 (two ended runs)", a.Runs)
	}
	if a.Witnessed != 1 {
		t.Errorf("active.Witnessed = %d, want 1", a.Witnessed)
	}
	if a.KeepRate != 0.5 {
		t.Errorf("active.KeepRate = %v, want 0.5 (1 witnessed / 2 runs)", a.KeepRate)
	}
	// The claimed-vs-witnessed surfacing on the same loop: 2 ended, 1 witnessed-done, 1
	// witness-refused -> gap 1, refused 1, unavailable 0. Exactly half witnessed
	// (Witnessed*2 == Runs) is the boundary — NOT a majority-unwitnessed collapse.
	if a.WitnessGap != 1 {
		t.Errorf("active.WitnessGap = %d, want 1 (2 ended - 1 witnessed)", a.WitnessGap)
	}
	if a.WitnessRefused != 1 || a.WitnessUnavailable != 0 {
		t.Errorf("active witness refused=%d unavailable=%d, want 1/0", a.WitnessRefused, a.WitnessUnavailable)
	}
	if a.WitnessCollapse {
		t.Errorf("active.WitnessCollapse = true, want false (half-witnessed is the boundary, not a collapse)")
	}
	if a.CadenceSource != "registry" || a.CadenceSeconds != 60 {
		t.Errorf("active cadence = %ds from %q, want 60s from registry", a.CadenceSeconds, a.CadenceSource)
	}
	if !a.Registered || !a.Ledgered {
		t.Errorf("active registered=%v ledgered=%v, want both true", a.Registered, a.Ledgered)
	}
	if a.AgeSeconds != 10 {
		t.Errorf("active.AgeSeconds = %d, want 10 (last tick 10s ago)", a.AgeSeconds)
	}
	if a.LearningDebt == nil || *a.LearningDebt != 7 {
		t.Errorf("active.LearningDebt = %v, want 7", a.LearningDebt)
	}

	// --- dark-ledger: has a ledger entry but stale far past horizon -> DARK.
	dl := rows["dark-ledger"]
	if dl.State != HealthDark || !dl.Dark {
		t.Errorf("dark-ledger.State = %q dark=%v, want dark/true (stale last tick)", dl.State, dl.Dark)
	}
	if !dl.Ledgered || dl.Registered {
		t.Errorf("dark-ledger ledgered=%v registered=%v, want ledgered, not registered", dl.Ledgered, dl.Registered)
	}
	if dl.CadenceSource != "default" {
		t.Errorf("dark-ledger.CadenceSource = %q, want default (no registry cadence)", dl.CadenceSource)
	}
	if dl.AgeSeconds != 10_000 {
		t.Errorf("dark-ledger.AgeSeconds = %d, want 10000", dl.AgeSeconds)
	}

	// --- dark-registered: registered cadence but NEVER ticked -> DARK, age 0.
	dr := rows["dark-registered"]
	if dr.State != HealthDark || !dr.Dark {
		t.Errorf("dark-registered.State = %q dark=%v, want dark/true (never ticked)", dr.State, dr.Dark)
	}
	if !dr.Registered || dr.Ledgered {
		t.Errorf("dark-registered registered=%v ledgered=%v, want registered, not ledgered", dr.Registered, dr.Ledgered)
	}
	if dr.LastTickUnixNano != 0 || dr.AgeSeconds != 0 {
		t.Errorf("dark-registered lastTick=%d age=%d, want both 0 (never ticked)", dr.LastTickUnixNano, dr.AgeSeconds)
	}
	if dr.Runs != 0 || dr.KeepRate != -1 {
		t.Errorf("dark-registered runs=%d keepRate=%v, want 0 runs and -1 keep rate (no denominator)", dr.Runs, dr.KeepRate)
	}

	// --- roll-up
	// WitnessGap: active 1 + dark-ledger 1 (ended, never witnessed) + dark-registered 0.
	// WitnessCollapse: only dark-ledger (0 of 1 witnessed is majority-unwitnessed); active
	// is exactly half (boundary, not collapse) and dark-registered has no ended runs.
	// Utility partition (#6497): all three ended runs completed but declared neither a
	// useful effect nor a typed no-fuel result, so they land in Unattributed — and no
	// failure counter fires.
	want := HealthRollup{Loops: 3, Live: 1, Stale: 0, Dark: 2, Unknown: 0, Registered: 2, Ledgered: 2, WitnessGap: 2, WitnessCollapse: 1, Runs: 3, Unattributed: 3}
	if rep.Rollup != want {
		t.Errorf("rollup = %+v, want %+v", rep.Rollup, want)
	}
}

// TestFoldHealth_WitnessGapAndCollapse pins the claimed-vs-witnessed surfacing: a LIVE
// loop that keeps ending runs it cannot prove done (every witness refused/unavailable)
// shows a full WitnessGap and WitnessCollapse, while an honestly-witnessed loop shows
// neither — so the worst-first walk that reads this fold can tell "talking, not proven
// done" apart from a loop with honest failures, which the keep rate alone conflated.
func TestFoldHealth_WitnessGapAndCollapse(t *testing.T) {
	now := time.Unix(5_000_000, 0).UTC()
	secAgo := func(s int64) int64 { return now.Add(-time.Duration(s) * time.Second).UnixNano() }

	// "talker": 3 ended runs, ZERO witnessed-done (2 refused, 1 unavailable), fired 5s
	// ago so it is LIVE — the exact case the keep rate hid: not dark, not stale, yet not
	// proven done. "honest": 2 ended runs, both witnessed-done — no gap, no collapse.
	events := []Event{
		tick("talker", EventEnd, StatusClaimedDone, secAgo(40)),
		tick("talker", EventEnd, StatusClaimedDone, secAgo(38)),
		tick("talker", EventEnd, StatusClaimedDone, secAgo(36)),
		tick("talker", EventWitness, StatusWitnessRefused, secAgo(35)),
		tick("talker", EventWitness, StatusWitnessRefused, secAgo(34)),
		tick("talker", EventWitness, StatusWitnessUnavailable, secAgo(33)),
		tick("talker", EventFire, "", secAgo(5)),
		tick("honest", EventEnd, StatusClaimedDone, secAgo(20)),
		tick("honest", EventEnd, StatusClaimedDone, secAgo(18)),
		tick("honest", EventWitness, StatusWitnessedDone, secAgo(17)),
		tick("honest", EventWitness, StatusWitnessedDone, secAgo(16)),
		tick("honest", EventFire, "", secAgo(5)),
	}
	st := Summarize(events, now)

	reg := Registry{Jobs: map[string]Job{}}
	for _, id := range []string{"talker", "honest"} {
		if err := reg.Put(Job{
			Schedule: Schedule{JobID: id, IntervalSeconds: 60, MissedRun: MissedSkip},
			State:    JobArmed,
		}, now); err != nil {
			t.Fatalf("Put(%s): %v", id, err)
		}
	}

	rep := FoldHealth(st, reg, now, HealthThresholds{})
	rows := map[string]HealthRow{}
	for _, r := range rep.Rows {
		rows[r.LoopID] = r
	}

	// talker: LIVE (fired 5s ago) yet majority-unwitnessed — the seam this fold closes.
	tk := rows["talker"]
	if tk.State != HealthLive {
		t.Fatalf("talker.State = %q, want live — the point is a LIVE loop that is not proven done", tk.State)
	}
	if tk.Runs != 3 || tk.Witnessed != 0 {
		t.Fatalf("talker runs=%d witnessed=%d, want 3/0", tk.Runs, tk.Witnessed)
	}
	if tk.WitnessGap != 3 {
		t.Errorf("talker.WitnessGap = %d, want 3 (3 ended, 0 witnessed-done)", tk.WitnessGap)
	}
	if tk.WitnessRefused != 2 || tk.WitnessUnavailable != 1 {
		t.Errorf("talker refused=%d unavailable=%d, want 2/1", tk.WitnessRefused, tk.WitnessUnavailable)
	}
	if !tk.WitnessCollapse {
		t.Errorf("talker.WitnessCollapse = false, want true (0 of 3 witnessed is majority-unwitnessed)")
	}
	if tk.KeepRate != 0 {
		t.Errorf("talker.KeepRate = %v, want 0 (no witnessed run) — a keep rate a naive pane could still read as 'just failing'", tk.KeepRate)
	}

	// honest: fully witnessed — neither field fires.
	hn := rows["honest"]
	if hn.WitnessGap != 0 {
		t.Errorf("honest.WitnessGap = %d, want 0 (both runs witnessed)", hn.WitnessGap)
	}
	if hn.WitnessCollapse {
		t.Errorf("honest.WitnessCollapse = true, want false (all witnessed)")
	}
	if hn.WitnessRefused != 0 || hn.WitnessUnavailable != 0 {
		t.Errorf("honest refused=%d unavailable=%d, want 0/0", hn.WitnessRefused, hn.WitnessUnavailable)
	}

	// Roll-up: total gap 3 (talker) + 0 (honest); one collapsed loop.
	if rep.Rollup.WitnessGap != 3 {
		t.Errorf("rollup.WitnessGap = %d, want 3", rep.Rollup.WitnessGap)
	}
	if rep.Rollup.WitnessCollapse != 1 {
		t.Errorf("rollup.WitnessCollapse = %d, want 1 (only talker collapsed)", rep.Rollup.WitnessCollapse)
	}
}

// TestFoldHealth_StaleVsLiveVsDarkBoundaries pins the age classification against a
// cadence so a slipping loop reads STALE, not yet DARK, and the dark boundary is at
// DarkMultiple cadences.
func TestFoldHealth_StaleVsLiveVsDarkBoundaries(t *testing.T) {
	now := time.Unix(2_000_000, 0).UTC()
	secAgo := func(s int64) int64 { return now.Add(-time.Duration(s) * time.Second).UnixNano() }

	// cadence 100s, DarkMultiple 2 (default) -> dark boundary at 200s.
	//   live:  ticked 50s ago   (<=100)            -> live
	//   stale: ticked 150s ago  (100<age<=200)     -> stale
	//   dark:  ticked 250s ago  (>200)             -> dark
	events := []Event{
		tick("live", EventFire, "", secAgo(50)),
		tick("stale", EventFire, "", secAgo(150)),
		tick("dark", EventFire, "", secAgo(250)),
	}
	st := Summarize(events, now)

	reg := Registry{Jobs: map[string]Job{}}
	for _, id := range []string{"live", "stale", "dark"} {
		if err := reg.Put(Job{
			Schedule: Schedule{JobID: id, IntervalSeconds: 100, MissedRun: MissedSkip},
			State:    JobArmed,
		}, now); err != nil {
			t.Fatalf("Put(%s): %v", id, err)
		}
	}

	rep := FoldHealth(st, reg, now, HealthThresholds{})
	got := map[string]HealthState{}
	for _, r := range rep.Rows {
		got[r.LoopID] = r.State
	}
	want := map[string]HealthState{"live": HealthLive, "stale": HealthStale, "dark": HealthDark}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("states = %v, want %v", got, want)
	}
	if rep.Rollup.Live != 1 || rep.Rollup.Stale != 1 || rep.Rollup.Dark != 1 {
		t.Fatalf("rollup = %+v, want 1 live / 1 stale / 1 dark", rep.Rollup)
	}
}

// TestFoldHealth_PureRead proves the fold mutates neither input and is deterministic:
// two folds of the same inputs produce DeepEqual reports, and the input Status /
// Registry are byte-identical before and after.
func TestFoldHealth_PureRead(t *testing.T) {
	now := time.Unix(3_000_000, 0).UTC()
	events := []Event{
		tick("l", EventStart, StatusRunning, now.Add(-30*time.Second).UnixNano()),
		tick("l", EventEnd, StatusClaimedDone, now.Add(-20*time.Second).UnixNano()),
		tick("l", EventWitness, StatusWitnessedDone, now.Add(-19*time.Second).UnixNano()),
	}
	st := Summarize(events, now)
	reg := Registry{Jobs: map[string]Job{}}
	if err := reg.Put(Job{
		Schedule: Schedule{JobID: "l", IntervalSeconds: 60, MissedRun: MissedCatchUp},
		State:    JobArmed,
	}, now); err != nil {
		t.Fatalf("Put: %v", err)
	}

	stBefore := deepCopyStatus(st)
	regBefore := deepCopyRegistry(reg)

	r1 := FoldHealth(st, reg, now, HealthThresholds{})
	r2 := FoldHealth(st, reg, now, HealthThresholds{})

	if !reflect.DeepEqual(r1, r2) {
		t.Fatalf("fold not deterministic:\n r1=%+v\n r2=%+v", r1, r2)
	}
	if !reflect.DeepEqual(st, stBefore) {
		t.Errorf("FoldHealth mutated its Status input:\n before=%+v\n after =%+v", stBefore, st)
	}
	if !reflect.DeepEqual(reg, regBefore) {
		t.Errorf("FoldHealth mutated its Registry input:\n before=%+v\n after =%+v", regBefore, reg)
	}

	// And it actually folded something real: one live, witnessed, keep-rate-1 loop.
	if len(r1.Rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(r1.Rows))
	}
	row := r1.Rows[0]
	if row.State != HealthLive || row.KeepRate != 1 || row.Runs != 1 || row.Witnessed != 1 {
		t.Errorf("row = %+v, want live, keepRate 1, 1 run, 1 witnessed", row)
	}
}

// TestFoldHealth_UnknownWhenNoCadenceAndNoTick covers the decline-to-judge branch: a
// loop with neither a usable last tick nor a cadence is UNKNOWN, distinct from DARK.
// TestFoldHealth_RetiredJobIsNotDark pins the stopped/disabled honesty edge: a
// registered job the operator put DOWN (stopped or disabled — not armed) is not
// expected to tick, so it reads RETIRED, never DARK. Pre-fix it folded DARK,
// reddening `loop health --check` (which gates on rollup.Dark) and inflating the
// dark tally for a loop that is intentionally quiet. The still-armed never-ticked
// job is the one that must stay DARK.
func TestFoldHealth_RetiredJobIsNotDark(t *testing.T) {
	now := time.Unix(2_000_000, 0).UTC()
	st := Summarize(nil, now) // no ledger: every job has never ticked (dark if armed)

	reg := Registry{Jobs: map[string]Job{}}
	for _, j := range []struct {
		id    string
		state JobState
	}{
		{"stopped-loop", JobStopped},
		{"disabled-loop", JobDisabled},
		{"armed-loop", JobArmed},
	} {
		if err := reg.Put(Job{
			Schedule: Schedule{JobID: j.id, IntervalSeconds: 3600, MissedRun: MissedSkip},
			State:    j.state,
		}, now); err != nil {
			t.Fatalf("registry.Put(%s): %v", j.id, err)
		}
	}

	rep := FoldHealth(st, reg, now, HealthThresholds{})

	if got := rowFor(t, rep, "stopped-loop"); got.State != HealthRetired || got.Dark {
		t.Errorf("stopped job: state=%q dark=%v, want retired/false", got.State, got.Dark)
	}
	if got := rowFor(t, rep, "disabled-loop"); got.State != HealthRetired || got.Dark {
		t.Errorf("disabled job: state=%q dark=%v, want retired/false", got.State, got.Dark)
	}
	if got := rowFor(t, rep, "armed-loop"); got.State != HealthDark || !got.Dark {
		t.Errorf("armed never-ticked job: state=%q dark=%v, want dark/true", got.State, got.Dark)
	}

	if rep.Rollup.Dark != 1 {
		t.Errorf("rollup.Dark = %d, want 1 (only the armed never-ticked loop)", rep.Rollup.Dark)
	}
	if rep.Rollup.Retired != 2 {
		t.Errorf("rollup.Retired = %d, want 2 (stopped + disabled)", rep.Rollup.Retired)
	}
}

func TestFoldHealth_UnknownWhenNoCadenceAndNoTick(t *testing.T) {
	now := time.Unix(4_000_000, 0).UTC()
	// A loop whose only events carry no usable timestamp AND a zero default horizon.
	events := []Event{tick("ghost", EventFire, "", 0)}
	st := Summarize(events, now)
	rep := FoldHealth(st, Registry{Jobs: map[string]Job{}}, now, HealthThresholds{DefaultCadenceSeconds: -1, DarkMultiple: 1})
	if len(rep.Rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(rep.Rows))
	}
	// DefaultCadenceSeconds: -1 is rejected by the helper (<=0) and replaced with the
	// real default, so this still resolves to a cadence; force the no-cadence path by
	// asserting the helper's own contract directly.
	if got := (HealthThresholds{}).defaultCadenceSeconds(); got != DefaultHealthCadenceSeconds {
		t.Fatalf("defaultCadenceSeconds() = %d, want %d", got, DefaultHealthCadenceSeconds)
	}
	if got := DeriveState(0, 0, now, HealthThresholds{}); got != HealthUnknown {
		t.Fatalf("DeriveState(no tick, no cadence) = %q, want %q", got, HealthUnknown)
	}
	if got := DeriveState(now.UnixNano(), 0, now, HealthThresholds{}); got != HealthUnknown {
		t.Fatalf("DeriveState(ticked, no cadence) = %q, want %q", got, HealthUnknown)
	}
}

func deepCopyStatus(s Status) Status {
	out := s
	out.Loops = append([]LoopSnapshot(nil), s.Loops...)
	for i := range out.Loops {
		if s.Loops[i].LastRun != nil {
			lr := *s.Loops[i].LastRun
			out.Loops[i].LastRun = &lr
		}
		if s.Loops[i].Metrics != nil {
			out.Loops[i].Metrics = map[string]int64{}
			for k, v := range s.Loops[i].Metrics {
				out.Loops[i].Metrics[k] = v
			}
		}
	}
	return out
}

func deepCopyRegistry(r Registry) Registry {
	out := Registry{Schema: r.Schema, Jobs: map[string]Job{}}
	for k, v := range r.Jobs {
		out.Jobs[k] = v
	}
	return out
}
