package main

// guard_fleetbus_test.go — the acceptance witness for #5953 (epic #5599).
//
// The reported gap was not "guards lack a feature". It was that a fleet of ~16 live
// `fak guard` gateways was INVISIBLE to its own control plane, and the second half of
// that gap is subtler: a control plane that could see them would then have promised
// things they cannot do. So every arm here is about one of the two honest answers —
// the guard shows up, and it says up front what it cannot apply — plus the third,
// which is that what it CAN apply is a real write with a real witness.

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetbus"
	"github.com/anthony-chaudhary/fak/internal/goalpark"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
)

func guardTestApplier(t *testing.T, tbl *session.Table, seats guardSeatRefresher) *guardBusApplier {
	t.Helper()
	if tbl == nil {
		tbl = &session.Table{}
	}
	return &guardBusApplier{
		sessions: &fleetBusApplier{tbl: tbl, native: false, ctx: context.Background()},
		seats:    seats,
	}
}

func guardTestCensus(usable, pool int) func() (guardSeatCensus, error) {
	return func() (guardSeatCensus, error) {
		return guardSeatCensus{Usable: usable, Pool: pool, Total: pool + 22, Source: "registry.json"}, nil
	}
}

func guardTestPark(t *testing.T, s goalpark.Store, goal, account string, now time.Time, wall time.Duration) {
	t.Helper()
	err := s.Park(goalpark.Record{
		Schema:      goalpark.Schema,
		Goal:        goal,
		Account:     account,
		Reason:      "LONG_RETRY_AFTER",
		ParkedAt:    now.Unix(),
		ParkedUntil: now.Add(wall).Unix(),
		Command:     []string{"fak", "guard", "--", "claude"},
	})
	if err != nil {
		t.Fatalf("seed park %s: %v", goal, err)
	}
}

// --- criterion 1: a guard with NO fleet flags shows up ---------------------- //

// TestGuardJoinsTheBusWithNoFleetFlags is the reported gap itself. A guard started the
// way every guard on the fleet is started — no --fleet-bus, no --fleet-bus-dir —
// must be addressable from the control point within one TTL, carrying a role that can
// be selected on and the capability claims an operator reads before fanning anything.
func TestGuardJoinsTheBusWithNoFleetFlags(t *testing.T) {
	busDir := t.TempDir()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	// on=true is what a guard gets with NO flags at all: the flag's default is true,
	// unlike serve's. resolveGuardFleetBusDir is pinned separately below.
	stop := startGuardFleetBus(ctx, true, busDir, "guard-test-1", time.Hour)
	defer stop()

	bus, err := fleetbus.OpenDir(busDir)
	if err != nil {
		t.Fatalf("OpenDir: %v", err)
	}
	// The FIRST announce is synchronous, so no tick has to elapse: a boot window in
	// which `send` refuses FLEETBUS_NO_TARGET against a fleet that is up would be a
	// refusal correct about the roster and wrong about the world.
	roster, err := bus.Instances(time.Now(), fleetbus.DefaultInstanceTTL)
	if err != nil {
		t.Fatalf("Instances: %v", err)
	}
	if len(roster) != 1 {
		t.Fatalf("roster = %d instances, want the guard to have announced itself", len(roster))
	}
	inst := roster[0]
	if inst.ID != "guard-test-1" || inst.Role != guardFleetBusRole {
		t.Fatalf("instance = %s/%s, want guard-test-1/%s", inst.ID, inst.Role, guardFleetBusRole)
	}
	if !fleetbus.PublishTargets(fleetbus.Selector{Role: []string{"guard"}}, roster)[0].DeclaresOp(guardBusSeatRefresh) {
		t.Fatalf("--role guard did not address a guard declaring %s: %v", guardBusSeatRefresh, inst.Ops)
	}
	if !inst.DeclaresUnsupported(fleetbus.Op(sessionctl.OpSteer)) {
		t.Fatalf("instance unsupported = %v, want it to declare steer unsupported", inst.Unsupported)
	}
	if inst.DeclaresOp(fleetbus.Op(sessionctl.OpSteer)) {
		t.Fatalf("instance ops = %v, want steer absent — a guard must not claim it", inst.Ops)
	}
	for _, op := range sessionctl.BroadcastableOps() {
		if !inst.DeclaresOp(fleetbus.Op(op)) {
			t.Fatalf("instance ops = %v, want the lifecycle op %q it really can apply", inst.Ops, op)
		}
	}
	// The whole roster answer an operator reads BEFORE sending anything.
	if got := fleetbus.CapabilityFor(roster, fleetbus.Op(sessionctl.OpSteer)); got.Declared != 0 || got.Unsupported != 1 {
		t.Fatalf("steer capability = %+v, want 0 of 1 declaring it and 1 declaring it unsupported", got)
	}
}

// TestGuardFleetBusDefaultsToOnAndIDIsAnnounceable — the two resolutions the arming
// path depends on. A default id that fleetbus.Announce would refuse means a guard that
// silently never joins, which is the bug this issue reports arrived at by another road.
func TestGuardFleetBusDefaultsToOnAndIDIsAnnounceable(t *testing.T) {
	if got := resolveGuardFleetBusDir(true, "  /tmp/some-bus  "); got != "/tmp/some-bus" {
		t.Fatalf("resolveGuardFleetBusDir = %q, want the trimmed explicit dir", got)
	}
	if got := resolveGuardFleetBusDir(true, ""); strings.TrimSpace(got) == "" {
		t.Fatal("a guard with no --fleet-bus-dir resolved to no bus at all — it would never announce")
	}
	id := resolveGuardFleetBusID("")
	if !strings.HasPrefix(id, "guard-") {
		t.Fatalf("default instance id = %q, want a guard- prefix so a mixed roster is readable", id)
	}
	if !fleetbusIDIsAToken(id) {
		t.Fatalf("default instance id %q is not a bus token — Announce would refuse it", id)
	}
	if got := resolveGuardFleetBusID("  my-guard  "); got != "my-guard" {
		t.Fatalf("explicit --fleet-bus-id = %q, want the trimmed name", got)
	}
}

// --- criterion 4: the opt-out is TOTAL -------------------------------------- //

// TestGuardFleetBusOptOutTouchesNothing. This flag defaults ON, so the operator who
// says no is precisely the one who has to be able to PROVE nothing happened — an
// empty bus directory left behind by a disarmed guard would be a control plane
// half-created by a binary that was told not to.
func TestGuardFleetBusOptOutTouchesNothing(t *testing.T) {
	busDir := filepath.Join(t.TempDir(), "bus")
	stop := startGuardFleetBus(context.Background(), false, busDir, "guard-test-off", time.Hour)
	if stop == nil {
		t.Fatal("startGuardFleetBus returned a nil stop func")
	}
	stop()
	if _, err := os.Stat(busDir); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("stat(%s) = %v, want the disarmed guard to have created nothing", busDir, err)
	}
	if got := resolveGuardFleetBusDir(false, busDir); got != "" {
		t.Fatalf("resolveGuardFleetBusDir with --fleet-bus=false = %q, want \"\" so the no-op is total", got)
	}
}

// --- criterion 3: steer refuses with a real denominator --------------------- //

// TestGuardRefusesSteerAgainstARealDenominator is the case the issue calls out by
// name. Sixteen loop-less guards addressed by `--op steer --all` must fold to
// "16 targeted, 16 refused, 0 applied" with the closed reason on every row — never
// FLEETBUS_NO_TARGET (which would say nobody was there) and never a false applied.
func TestGuardRefusesSteerAgainstARealDenominator(t *testing.T) {
	busDir := t.TempDir()
	bus, err := fleetbus.OpenDir(busDir)
	if err != nil {
		t.Fatalf("OpenDir: %v", err)
	}
	now := time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC)
	const fleet = 16
	var selves []fleetbus.Instance
	for i := 0; i < fleet; i++ {
		id := "guard-" + string(rune('a'+i))
		inst, r := fleetbus.NewInstance(id, "box-a", guardFleetBusRole, 100+i, "", guardFleetBusAdvertisedOps(), now)
		if r != nil {
			t.Fatalf("NewInstance %s: %v", id, r)
		}
		inst = inst.WithUnsupportedOps(guardFleetBusUnsupportedOps())
		if err := bus.Announce(inst); err != nil {
			t.Fatalf("Announce %s: %v", id, err)
		}
		selves = append(selves, inst)
	}
	roster, err := bus.Instances(now, fleetbus.DefaultInstanceTTL)
	if err != nil {
		t.Fatalf("Instances: %v", err)
	}
	targets := fleetbus.PublishTargets(fleetbus.Selector{All: true}, roster)
	if len(targets) != fleet {
		t.Fatalf("PublishTargets = %d, want %d — declaring steer unsupported must NOT remove an instance from the fan-out", len(targets), fleet)
	}

	d, refusal := fleetbus.NewDirective("op", fleetbus.Op(sessionctl.OpSteer), "go",
		fleetbus.Selector{All: true}, 5*time.Minute, "", now)
	if refusal != nil {
		t.Fatalf("NewDirective: %v", refusal)
	}
	d = d.WithTargets(targets)
	if err := bus.Publish(d); err != nil {
		t.Fatalf("Publish: %v", err)
	}
	for _, self := range selves {
		ap := guardTestApplier(t, nil, guardSeatRefresher{census: guardTestCensus(3, 14)})
		if _, err := fleetbus.Drain(bus, self, ap, now); err != nil {
			t.Fatalf("Drain %s: %v", self.ID, err)
		}
	}
	acks, err := bus.Acks(d.ID)
	if err != nil {
		t.Fatalf("Acks: %v", err)
	}
	rep := fleetbus.Fold(d, roster, acks, now)
	if rep.Targeted != fleet || rep.Refused != fleet || rep.Applied != 0 || rep.Outstanding != 0 {
		t.Fatalf("fold = targeted %d / refused %d / applied %d / outstanding %d, want %d/%d/0/0",
			rep.Targeted, rep.Refused, rep.Applied, rep.Outstanding, fleet, fleet)
	}
	if !rep.Complete || rep.AffectedTotal != 0 {
		t.Fatalf("fold complete=%v affected=%d, want a complete answer that moved nothing", rep.Complete, rep.AffectedTotal)
	}
	for _, row := range rep.Rows {
		if row.Reason != fleetbus.ApplyRefused {
			t.Fatalf("%s: reason = %q, want the closed token %q", row.Instance, row.Reason, fleetbus.ApplyRefused)
		}
		if !strings.Contains(row.Detail, "STEER_NO_OWNED_LOOP") {
			t.Fatalf("%s: detail = %q, want it to carry STEER_NO_OWNED_LOOP", row.Instance, row.Detail)
		}
		// serve's refusal PRESCRIBES --native ("start it with --native"). A guard has no
		// such flag, and a remedy the operator cannot act on is worse than none — so the
		// guard must not carry serve's wording, and must name a remedy that exists.
		if strings.Contains(row.Detail, "start it with --native") {
			t.Fatalf("%s: detail = %q, want a guard-shaped remedy, not serve's prescription", row.Instance, row.Detail)
		}
		if !strings.Contains(row.Detail, "fak session steer") {
			t.Fatalf("%s: detail = %q, want it to name the remedy that actually exists for a guard", row.Instance, row.Detail)
		}
	}
}

// --- criterion 2 + 5: seat-refresh is a real applier ------------------------ //

// TestGuardSeatRefreshRefusesWhileTheRosterIsStillWalled. Releasing a park into a
// fleet that still has no usable seat costs a worker unit per park and teaches nobody
// anything, so "the fix has not landed yet" must be a REFUSAL under a closed token —
// and it must leave every park exactly where it found it.
func TestGuardSeatRefreshRefusesWhileTheRosterIsStillWalled(t *testing.T) {
	now := time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC)
	park := goalpark.Store{Dir: t.TempDir()}
	guardTestPark(t, park, "lane-alpha", "acct-a", now, 20*time.Hour)

	ap := guardTestApplier(t, nil, guardSeatRefresher{
		census:   guardTestCensus(0, 14),
		park:     park,
		instance: "guard-1",
		now:      func() time.Time { return now },
	})
	out := ap.Apply(fleetBusDirective(string(guardBusSeatRefresh), fleetbus.Selector{All: true}))
	if out.Status != fleetbus.AckRefused || out.Reason != fleetbus.ApplyRefused {
		t.Fatalf("outcome = %+v, want a refusal under a closed token", out)
	}
	if !strings.Contains(out.Detail, "SEAT_ROSTER_STILL_WALLED") {
		t.Fatalf("detail = %q, want SEAT_ROSTER_STILL_WALLED", out.Detail)
	}
	if out.Affected != 0 {
		t.Fatalf("affected = %d, want 0 — a refusal must never claim work", out.Affected)
	}
	rec, err := park.Load("lane-alpha")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !rec.Blocks("acct-a", now) {
		t.Fatal("a refused seat-refresh released the park anyway")
	}
}

// TestGuardSeatRefreshReleasesParkedWorkersOnceASeatIsUsable is the issue's fifth
// acceptance criterion: after a seat is enrolled, ONE control-point command releases
// the workers that were sitting auth-blocked. The witness must name what was observed
// — the seat count and the parks retired — and Affected must count only what moved.
func TestGuardSeatRefreshReleasesParkedWorkersOnceASeatIsUsable(t *testing.T) {
	now := time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC)
	park := goalpark.Store{Dir: t.TempDir()}
	guardTestPark(t, park, "lane-alpha", "acct-a", now, 20*time.Hour)
	guardTestPark(t, park, "lane-beta", "acct-b", now, 18*time.Hour)
	// A park whose wall already lifted is NOT holding anybody, so it must not inflate
	// the affected count: the operator came for the number of workers actually freed.
	guardTestPark(t, park, "lane-old", "acct-c", now.Add(-30*time.Hour), 2*time.Hour)

	ap := guardTestApplier(t, nil, guardSeatRefresher{
		census:   guardTestCensus(3, 14),
		park:     park,
		instance: "guard-1",
		now:      func() time.Time { return now },
	})
	d := fleetBusDirective(string(guardBusSeatRefresh), fleetbus.Selector{All: true})
	d.Reason = "enrolled seat work-7"
	out := ap.Apply(d)
	if out.Status != fleetbus.AckApplied {
		t.Fatalf("outcome = %+v, want applied", out)
	}
	if out.Affected != 2 {
		t.Fatalf("affected = %d, want 2 (the holding parks only)", out.Affected)
	}
	for _, want := range []string{"3/14", "lane-alpha", "lane-beta", "released 2/2"} {
		if !strings.Contains(out.Witness, want) {
			t.Fatalf("witness = %q, want it to name %q", out.Witness, want)
		}
	}
	if strings.Contains(out.Witness, "lane-old") {
		t.Fatalf("witness = %q, want the already-lifted park left out of the release", out.Witness)
	}
	for _, goal := range []string{"lane-alpha", "lane-beta"} {
		rec, err := park.Load(goal)
		if err != nil {
			t.Fatalf("Load %s: %v", goal, err)
		}
		if rec.Blocks(rec.Account, now) {
			t.Fatalf("%s still blocks %s after a seat-refresh applied", goal, rec.Account)
		}
		if !strings.Contains(rec.NextAction, "enrolled seat work-7") {
			t.Fatalf("%s next_legal_action = %q, want the directive's reason carried through", goal, rec.NextAction)
		}
	}

	// A SECOND instance draining the same broadcast must not re-count the same
	// releases. This is what keeps the fleet-wide affected total honest.
	peer := guardTestApplier(t, nil, guardSeatRefresher{
		census:   guardTestCensus(3, 14),
		park:     park,
		instance: "guard-2",
		now:      func() time.Time { return now },
	})
	second := peer.Apply(d)
	if second.Status != fleetbus.AckApplied || second.Affected != 0 {
		t.Fatalf("peer outcome = %+v, want applied with 0 affected (the parks were already released)", second)
	}
}

// TestGuardSeatRefreshFoldsAcrossTheFleetExactlyOnce is the issue's second acceptance
// criterion end to end: one `send --op seat-refresh --all` against N guards folds to
// N addressed / N acked with a per-instance outcome. The number that has to be right
// is AffectedTotal — the guards on a box share ONE park store, so a fleet-wide total
// that summed each instance's view would report 3 guards x 2 parks = 6 workers freed
// when 2 were. The O_EXCL claim is what makes the sum equal the truth.
func TestGuardSeatRefreshFoldsAcrossTheFleetExactlyOnce(t *testing.T) {
	bus, err := fleetbus.OpenDir(t.TempDir())
	if err != nil {
		t.Fatalf("OpenDir: %v", err)
	}
	now := time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC)
	park := goalpark.Store{Dir: t.TempDir()}
	guardTestPark(t, park, "lane-alpha", "acct-a", now, 20*time.Hour)
	guardTestPark(t, park, "lane-beta", "acct-b", now, 18*time.Hour)

	const fleet = 3
	var selves []fleetbus.Instance
	for i := 0; i < fleet; i++ {
		id := "guard-" + string(rune('a'+i))
		inst, r := fleetbus.NewInstance(id, "box-a", guardFleetBusRole, 200+i, "", guardFleetBusAdvertisedOps(), now)
		if r != nil {
			t.Fatalf("NewInstance %s: %v", id, r)
		}
		inst = inst.WithUnsupportedOps(guardFleetBusUnsupportedOps())
		if err := bus.Announce(inst); err != nil {
			t.Fatalf("Announce %s: %v", id, err)
		}
		selves = append(selves, inst)
	}
	roster, err := bus.Instances(now, fleetbus.DefaultInstanceTTL)
	if err != nil {
		t.Fatalf("Instances: %v", err)
	}
	d, refusal := fleetbus.NewDirective("op", guardBusSeatRefresh, "",
		fleetbus.Selector{All: true}, 5*time.Minute, "enrolled seat work-7", now)
	if refusal != nil {
		t.Fatalf("NewDirective: %v", refusal)
	}
	d = d.WithTargets(fleetbus.PublishTargets(fleetbus.Selector{All: true}, roster))
	if err := bus.Publish(d); err != nil {
		t.Fatalf("Publish: %v", err)
	}
	for _, self := range selves {
		ap := guardTestApplier(t, nil, guardSeatRefresher{
			census:   guardTestCensus(3, 14),
			park:     park,
			instance: self.ID,
			now:      func() time.Time { return now },
		})
		if _, err := fleetbus.Drain(bus, self, ap, now); err != nil {
			t.Fatalf("Drain %s: %v", self.ID, err)
		}
	}
	acks, err := bus.Acks(d.ID)
	if err != nil {
		t.Fatalf("Acks: %v", err)
	}
	rep := fleetbus.Fold(d, roster, acks, now)
	if rep.Targeted != fleet || rep.Applied != fleet || rep.Outstanding != 0 || !rep.Complete {
		t.Fatalf("fold = targeted %d / applied %d / outstanding %d / complete %v, want %d/%d/0/true",
			rep.Targeted, rep.Applied, rep.Outstanding, rep.Complete, fleet, fleet)
	}
	if rep.AffectedTotal != 2 {
		t.Fatalf("affected_total = %d, want 2 — the fleet released 2 parks, not %d x 2", rep.AffectedTotal, fleet)
	}
	// Every instance carries its OWN outcome, and the ones that lost the race say so
	// rather than reporting a release they did not perform.
	var releasers, peers int
	for _, row := range rep.Rows {
		if row.Witness == "" {
			t.Fatalf("%s: applied with no witness — an ack must name what it observed", row.Instance)
		}
		if !strings.Contains(row.Witness, "3/14") {
			t.Fatalf("%s: witness = %q, want the observed seat count", row.Instance, row.Witness)
		}
		switch {
		case row.Affected > 0:
			releasers += row.Affected
		case strings.Contains(row.Witness, "released by a peer instance first") || strings.Contains(row.Witness, "already released early by a peer instance"):
			peers++
		default:
			t.Fatalf("%s: witness = %q, want it to say either what it released or that a peer won", row.Instance, row.Witness)
		}
	}
	if releasers != 2 || peers != fleet-1 {
		t.Fatalf("rows = %d released / %d deferred to a peer, want 2 and %d", releasers, peers, fleet-1)
	}
}

// TestGuardSeatRefreshAppliesWithZeroAffectedWhenNothingIsHolding — the same shape
// applyLifecycle uses for an all-no-op fan. The op ran, the observation is real, and
// nothing needed moving; refusing here would raise a false alarm about a box that is
// exactly where it should be.
func TestGuardSeatRefreshAppliesWithZeroAffectedWhenNothingIsHolding(t *testing.T) {
	now := time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC)
	ap := guardTestApplier(t, nil, guardSeatRefresher{
		census:   guardTestCensus(9, 14),
		park:     goalpark.Store{Dir: t.TempDir()},
		instance: "guard-1",
		now:      func() time.Time { return now },
	})
	out := ap.Apply(fleetBusDirective(string(guardBusSeatRefresh), fleetbus.Selector{All: true}))
	if out.Status != fleetbus.AckApplied || out.Affected != 0 {
		t.Fatalf("outcome = %+v, want applied with 0 affected", out)
	}
	if !strings.Contains(out.Witness, "no park was holding") || !strings.Contains(out.Witness, "9/14") {
		t.Fatalf("witness = %q, want it to say what it observed and that nothing was holding", out.Witness)
	}
}

// TestGuardSeatRefreshRefusesWhenItCannotReadTheRoster — an unreadable roster must
// never be read as "no seats" (which would refuse forever) or as "seats exist" (which
// would release a fleet into a wall). It gets its own token.
func TestGuardSeatRefreshRefusesWhenItCannotReadTheRoster(t *testing.T) {
	ap := guardTestApplier(t, nil, guardSeatRefresher{
		census: func() (guardSeatCensus, error) {
			return guardSeatCensus{}, errors.New("registry.json: permission denied")
		},
		park: goalpark.Store{Dir: t.TempDir()},
	})
	out := ap.Apply(fleetBusDirective(string(guardBusSeatRefresh), fleetbus.Selector{All: true}))
	if out.Status != fleetbus.AckRefused || !strings.Contains(out.Detail, "SEAT_ROSTER_UNREADABLE") {
		t.Fatalf("outcome = %+v, want a SEAT_ROSTER_UNREADABLE refusal", out)
	}
}

// --- the lifecycle half is not a stub --------------------------------------- //

// TestGuardAppliesLifecycleOpsForReal. Declaring steer unsupported would be worth
// nothing if the guard then refused everything else too: the point of the declaration
// is that the REST of the vocabulary is real. A guard shares the process-wide session
// table its gateway populates, so a fanned pause is the same Table.Transition write
// the single-session verb rides.
func TestGuardAppliesLifecycleOpsForReal(t *testing.T) {
	tbl := &session.Table{}
	seedFleetBusSession(t, tbl, "guarded-1", sessionctl.BroadcastMeta{})
	ap := guardTestApplier(t, tbl, guardSeatRefresher{census: guardTestCensus(3, 14)})

	out := ap.Apply(fleetBusDirective("pause", fleetbus.Selector{All: true}))
	if out.Status != fleetbus.AckApplied || out.Affected != 1 {
		t.Fatalf("pause outcome = %+v, want applied with 1 affected", out)
	}
	if snap := tbl.Snapshot(); len(snap) != 1 || snap[0].Run != session.Paused {
		t.Fatalf("table after pause = %+v, want the session actually paused", snap)
	}
	// ...and a re-pause is an apply that moved nothing, never a second affected count.
	if again := ap.Apply(fleetBusDirective("pause", fleetbus.Selector{All: true})); again.Status != fleetbus.AckApplied || again.Affected != 0 {
		t.Fatalf("re-pause outcome = %+v, want applied with 0 affected", again)
	}
}

// TestGuardAdvertisesExactlyWhatItCanApply is the anti-drift arm on the two claims.
// Every advertised op must get a real attempt (never FLEETBUS_UNKNOWN_OP — an op an
// instance claims and then does not recognise is a roster that lies), and every
// declared-unsupported op must actually refuse.
func TestGuardAdvertisesExactlyWhatItCanApply(t *testing.T) {
	if err := validateFleetBusVocabularies(); err != nil {
		t.Fatalf("guard/gateway/session op vocabularies collide: %v", err)
	}
	tbl := &session.Table{}
	seedFleetBusSession(t, tbl, "guarded-1", sessionctl.BroadcastMeta{})
	ap := guardTestApplier(t, tbl, guardSeatRefresher{
		census:   guardTestCensus(3, 14),
		park:     goalpark.Store{Dir: t.TempDir()},
		instance: "guard-1",
	})
	advertised := guardFleetBusAdvertisedOps()
	if len(advertised) == 0 {
		t.Fatal("a guard advertises no ops at all")
	}
	for _, op := range advertised {
		d := fleetBusDirective(string(op), fleetbus.Selector{All: true})
		d.Payload = "go"
		if out := ap.Apply(d); out.Reason == fleetbus.UnknownOp {
			t.Errorf("guard advertises %q but refuses it as unknown", op)
		}
	}
	for _, op := range guardFleetBusUnsupportedOps() {
		for _, adv := range advertised {
			if adv == op {
				t.Fatalf("op %q is advertised AND declared unsupported — one instance, two answers", op)
			}
		}
		out := ap.Apply(fleetBusDirective(string(op), fleetbus.Selector{All: true}))
		if out.Status == fleetbus.AckApplied {
			t.Fatalf("op %q was declared unsupported but applied anyway: %+v", op, out)
		}
		if out.Reason == fleetbus.UnknownOp {
			t.Fatalf("op %q was declared unsupported but refused as UNKNOWN — the two are different facts", op)
		}
	}
}

func TestStartGuardFleetBusQuietSuppressesSuccessfulArming(t *testing.T) {
	busDir := t.TempDir()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	old := os.Stderr
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stderr = w
	stop := startGuardFleetBus(ctx, true, busDir, "guard-quiet-1", time.Hour, true)
	stop()
	w.Close()
	os.Stderr = old
	out, err := io.ReadAll(r)
	r.Close()
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 0 {
		t.Fatalf("quiet attended arming wrote pre-child bytes: %q", out)
	}
	if _, err := os.Stat(filepath.Join(busDir, "instances", "guard-quiet-1.json")); err != nil {
		t.Fatalf("quiet mode must still arm fleet bus: %v", err)
	}
}

func TestGuardResumeDirectiveWakesPausedSessions(t *testing.T) {
	tbl := &session.Table{}
	tbl.Transition("sess-1", session.Paused, "paused for test")
	tbl.Transition("sess-2", session.Paused, "paused for test")
	tbl.Transition("sess-running", session.Running, "")

	ap := guardTestApplier(t, tbl, guardSeatRefresher{})

	// 1. Resume all: transitions paused sessions to running, returns affected=2
	out := ap.Apply(fleetBusDirective("resume", fleetbus.Selector{All: true}))
	if out.Status != fleetbus.AckApplied || out.Affected != 2 {
		t.Fatalf("outcome = %+v, want AckApplied with affected=2", out)
	}
	if !strings.Contains(out.Witness, "resumed 2 session(s)") {
		t.Fatalf("witness = %q, want it to contain 'resumed 2 session(s)'", out.Witness)
	}
	if cur := tbl.Get("sess-1"); cur.Run != session.Running {
		t.Errorf("sess-1 run state = %v, want Running", cur.Run)
	}
	if cur := tbl.Get("sess-2"); cur.Run != session.Running {
		t.Errorf("sess-2 run state = %v, want Running", cur.Run)
	}
	if cur := tbl.Get("sess-running"); cur.Run != session.Running {
		t.Errorf("sess-running run state = %v, want Running", cur.Run)
	}

	// 2. Second resume when no sessions are paused returns applied with affected=0
	second := ap.Apply(fleetBusDirective("resume", fleetbus.Selector{All: true}))
	if second.Status != fleetbus.AckApplied || second.Affected != 0 {
		t.Fatalf("second outcome = %+v, want AckApplied with affected=0", second)
	}
	if !strings.Contains(second.Witness, "resumed 0 session(s)") {
		t.Fatalf("witness = %q, want 'resumed 0 session(s)'", second.Witness)
	}

	// 3. Idle daemon with empty table returns applied with affected=0, never refuses
	emptyAp := guardTestApplier(t, &session.Table{}, guardSeatRefresher{})
	emptyOut := emptyAp.Apply(fleetBusDirective("resume", fleetbus.Selector{All: true}))
	if emptyOut.Status != fleetbus.AckApplied || emptyOut.Affected != 0 {
		t.Fatalf("empty daemon outcome = %+v, want AckApplied with affected=0", emptyOut)
	}
}
