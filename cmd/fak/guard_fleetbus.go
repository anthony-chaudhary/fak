package main

// guard_fleetbus.go — the GUARD's half of the fleet control bus (#5953, epic #5599).
//
// serve_fleetbus.go made a `fak serve` reachable from one control point. The fleet is
// not made of serves. It is made of ~16 live `fak guard` gateways, and every one of
// them was invisible to `fak dev fleet control` — so the control plane's own
// instances list showed an empty fleet while sixteen gateways were up, and the first
// honest thing an operator could learn about them was nothing at all.
//
// Three decisions carry this file, and each of them is a refusal to fake something:
//
//	A guard announces BY DEFAULT. serve's --fleet-bus is off by default because
//	arming a control plane is a thing an operator states; a guard is different in the
//	one way that matters here — it is the process that is ALREADY running, in bulk,
//	unattended, and a fleet-control instance nobody remembered to arm is exactly as
//	useful as no fleet control. Opting out (--fleet-bus=false) is still a total no-op:
//	no announce, no directory, no filesystem touch at all.
//
//	A guard DECLARES what it cannot do instead of accepting and then refusing. It
//	owns no session loop of its own — there is no --native to turn on, because a guard
//	wraps somebody else's agent — so it announces steer as UNSUPPORTED
//	(fleetbus.Instance.Unsupported). The operator then reads "16 instances, 0 can
//	steer" from the roster BEFORE fanning anything out, which is the honest answer.
//	Declaring it does not exempt the guard from being addressed: a steer aimed at it
//	still lands, still gets a real attempt, and still draws a real refusal ack with
//	the closed token — because the roster's claim is not a witness.
//
//	A guard applies what it CAN, for real. The lifecycle ops are not stubs here: a
//	guard shares the package-level session table (main_session_control.go's
//	serveSessions) that its gateway's decideSession populates at every request
//	boundary, so pause/resume/cancel/terminate/throttle are the same Table.Transition
//	write the single-session verbs ride. And seat-refresh is this file's own op: the
//	one thing an operator actually needs from a guard that a serve cannot offer.

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/fleetbus"
	"github.com/anthony-chaudhary/fak/internal/goalpark"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
)

// guardFleetBusRole is the instance record's role token for a guard. It is what the
// Selector's --role axis addresses, so `--role guard` reaches exactly the wrapped
// gateways and not the serves.
const guardFleetBusRole = "guard"

// guardBusSeatRefresh is the guard's own op: re-read the seat roster from disk and,
// if it now shows a seat that can serve, retire the goal parks that were holding this
// box's workers off their goals.
//
// It exists because the fleet's actual stuck state is not a session-lifecycle problem
// at all. Workers sit auth-blocked for days after the operator has already fixed the
// cause — enrolled a seat, re-logged in — because nothing on the box re-reads the
// roster and nothing retires a park from outside the parked process (goalpark's three
// internal clearing paths are a timer, a probe slot, and a write-time clamp; none of
// them can hear "the wall is gone"). This is the fourth path's trigger.
const guardBusSeatRefresh fleetbus.Op = "seat-refresh"

// guardBusOwnedOps names the ops whose meaning lives in THIS file, so
// validateFleetBusVocabularies can prove they collide with neither sessionctl's
// vocabulary nor the gateway's. One op with two owners is a dispatch that silently
// picks one; the check makes that a startup error instead.
func guardBusOwnedOps() []fleetbus.Op { return []fleetbus.Op{guardBusSeatRefresh} }

// guardFleetBusAdvertisedOps is what a guard SAYS it can apply. Display only — the
// ack is the witness — but it is what makes `fak dev fleet control instances` a
// briefing rather than a list of pids.
//
// Deliberately NOT fleetBusAdvertisedOps(): that set is serve's, and it claims steer.
func guardFleetBusAdvertisedOps() []fleetbus.Op {
	ops := []fleetbus.Op{guardBusSeatRefresh}
	for _, op := range sessionctl.BroadcastableOps() {
		ops = append(ops, fleetbus.Op(op))
	}
	slices.Sort(ops)
	return ops
}

// guardFleetBusUnsupportedOps is the other half of the same claim: what this process
// structurally cannot do, for the whole time it runs.
//
// steer is the entry, and it is a STRUCTURAL no rather than a configuration one. A
// serve refuses steer when it was started without --native, which is a flag away from
// yes. A guard has no such flag: it wraps a child agent and the loop that could
// consume an operator turn belongs to that child, reachable through the child's own
// gateway (`fak session steer`), never through the guard. Saying so on the presence
// record is what turns a fan-out of sixteen refusals into one answer read up front.
func guardFleetBusUnsupportedOps() []fleetbus.Op {
	return []fleetbus.Op{fleetbus.Op(sessionctl.OpSteer)}
}

// --- the applier ----------------------------------------------------------- //

// guardBusApplier is the guard's Applier. It delegates session-lifecycle meaning to
// the SAME fleetBusApplier a serve uses — a fanned pause must be the identical write
// on both roles, or the fleet has two pauses — and owns exactly the two things a
// guard answers differently: steer (a structural refusal in its own words) and
// seat-refresh (its own op).
type guardBusApplier struct {
	sessions *fleetBusApplier
	seats    guardSeatRefresher
}

func (a *guardBusApplier) Apply(d fleetbus.Directive) fleetbus.Outcome {
	if d.Op == guardBusSeatRefresh {
		return a.seats.apply(d)
	}
	if sessionctl.ControlOp(strings.TrimSpace(string(d.Op))) == sessionctl.OpSteer {
		// Intercepted rather than left to fleetBusApplier.applySteer, whose refusal
		// names a serve and prescribes --native. Both are wrong here, and a refusal
		// that prescribes a flag the operator cannot pass is worse than no advice.
		return fleetbus.OutcomeRefused(fleetbus.ApplyRefused,
			"STEER_NO_OWNED_LOOP: a guard wraps somebody else's agent and owns no session loop, so there is no --native to turn on; "+
				"an enqueued steer would sit in a mailbox nothing drains. Steer the wrapped session through its own gateway "+
				"(`fak session steer`), or address a --role serve instance started with --native. This instance announced steer as unsupported.")
	}
	if a.sessions == nil {
		return fleetbus.OutcomeRefused(fleetbus.ApplyRefused, "this guard has no session applier wired")
	}
	if sessionctl.ControlOp(strings.TrimSpace(string(d.Op))) == sessionctl.OpResume {
		return a.applyResume(d)
	}
	return a.sessions.Apply(d)
}

func (a *guardBusApplier) applyResume(d fleetbus.Directive) fleetbus.Outcome {
	if a.sessions.tbl == nil {
		return fleetbus.OutcomeRefused(fleetbus.ApplyRefused, "this guard has no session table to write")
	}
	reason := d.Reason
	if reason == "" {
		reason = "fleet control resume"
	}
	ctx := a.sessions.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	var count int
	if !d.Selector.NarrowsSessions() {
		resumed := a.sessions.tbl.ResumeAll(reason)
		count = len(resumed)
		if a.sessions.durability != nil && a.sessions.durability.registry != nil {
			for _, st := range resumed {
				if err := a.sessions.durability.writeThrough(ctx, st.TraceID, st); err != nil {
					return fleetbus.OutcomeRefused(fleetbus.ApplyRefused,
						"resume applied in memory for %s but durable mirror failed: %v", st.TraceID, err)
				}
			}
		}
	} else {
		matched := a.sessions.matchedStates(d.Selector)
		for _, prev := range matched {
			if prev.Run != session.Paused {
				continue
			}
			st, took := a.sessions.tbl.Transition(prev.TraceID, session.Running, reason)
			if !took {
				continue
			}
			count++
			if a.sessions.durability != nil && a.sessions.durability.registry != nil {
				if err := a.sessions.durability.writeThrough(ctx, prev.TraceID, st); err != nil {
					return fleetbus.OutcomeRefused(fleetbus.ApplyRefused,
						"resume applied in memory for %s but durable mirror failed: %v", prev.TraceID, err)
				}
			}
		}
	}
	witness := fmt.Sprintf("resumed %d session(s)", count)
	if a.sessions.durability != nil && a.sessions.durability.registry != nil {
		witness = "durable:" + witness
	} else {
		witness = "memory-only:" + witness
	}
	return fleetbus.OutcomeApplied(witness, count)
}

// --- seat-refresh ---------------------------------------------------------- //

// guardSeatCensus is one re-read of the seat roster. Three numbers rather than one
// because "3 of 14" and "3 of 36" are different fleets and only the first is the
// working pool: Pool excludes the terminal seats (tombstoned, disabled, no config
// dir) that a 36-seat registry is mostly made of, and Total is kept so the witness
// can say which denominator it used.
type guardSeatCensus struct {
	Usable int
	Pool   int
	Total  int
	Source string
}

// guardSeatRefresher is seat-refresh's whole implementation. The roster read is a
// function seam so a test can state a census without a real ~/.claude-accounts; the
// park store is a plain value because it is already a directory a test can point at.
type guardSeatRefresher struct {
	census func() (guardSeatCensus, error)
	park   goalpark.Store
	// instance names this guard in the park's claim sidecar, so an operator reading a
	// released record can tell WHICH instance on the box won the race.
	instance string
	now      func() time.Time
}

func (r guardSeatRefresher) clock() time.Time {
	if r.now != nil {
		return r.now()
	}
	return time.Now()
}

// apply is seat-refresh. It refuses far more readily than it releases, and the order
// of the checks is the point: the roster is re-read FIRST, and a roster that is still
// walled stops the op before a single park is touched. Releasing into a wall costs a
// worker unit per park and teaches nobody anything, so "the fix has not landed yet"
// has to be a refusal with a closed token, not a release with an empty witness.
func (r guardSeatRefresher) apply(d fleetbus.Directive) fleetbus.Outcome {
	if r.census == nil {
		return fleetbus.OutcomeRefused(fleetbus.ApplyRefused,
			"SEAT_REFRESH_UNWIRED: this instance advertised %s but has no roster reader", guardBusSeatRefresh)
	}
	now := r.clock()
	census, err := r.census()
	if err != nil {
		return fleetbus.OutcomeRefused(fleetbus.ApplyRefused,
			"SEAT_ROSTER_UNREADABLE: could not re-read the seat roster: %v", err)
	}
	if census.Usable == 0 {
		return fleetbus.OutcomeRefused(fleetbus.ApplyRefused,
			"SEAT_ROSTER_STILL_WALLED: re-read %s and found 0 of %d pool seat(s) able to serve (%d in the registry incl. terminal seats); "+
				"releasing a park now would only walk the worker back into the same wall. Enroll or re-login a seat "+
				"(`fak accounts add` / `fak accounts login`), then re-send.",
			dashIfEmpty(census.Source), census.Pool, census.Total)
	}
	records, err := r.park.List()
	if err != nil {
		return fleetbus.OutcomeRefused(fleetbus.ApplyRefused,
			"PARK_STORE_UNREADABLE: seat roster is healthy (%d/%d) but the goal-park store could not be listed: %v",
			census.Usable, census.Pool, err)
	}
	// A park is HOLDING if it is unclaimed and its announced wait has not elapsed —
	// the same condition Record.Blocks tests, minus the account scoping. Account
	// scoping is deliberately dropped here: Blocks answers "may THIS account work
	// this goal", and seat-refresh is the operator asserting a roster-wide change, so
	// the subject is every park still standing on this box.
	//
	// cutShort is the same op's own footprint: a park CLAIMED while its wall was still
	// standing was retired early, which only Release does. Every guard on a box shares
	// one park store, so by the time the second instance drains this broadcast the
	// first has usually emptied it — and without this bucket that instance would
	// witness "no park was holding", which is true of the instant it looked and reads
	// as "there was never anything to do here". The fold is what an operator believes;
	// "nothing was stuck" and "a peer just freed two workers" must not print the same.
	var holding, cutShort []goalpark.Record
	for _, rec := range records {
		if now.Unix() >= rec.ParkedUntil {
			continue // the wall lifted on its own; this op did not and need not touch it
		}
		if rec.ClaimedAt == 0 {
			holding = append(holding, rec)
			continue
		}
		cutShort = append(cutShort, rec)
	}
	rosterWitness := fmt.Sprintf("seat roster re-read: %d/%d pool seat(s) can serve (%d total, %s)",
		census.Usable, census.Pool, census.Total, dashIfEmpty(census.Source))
	if len(holding) == 0 {
		// APPLIED with Affected 0, the same shape applyLifecycle uses for an all-no-op
		// fan: the op ran, the observation is real, and nothing needed moving. Refusing
		// here would raise a false alarm about a box that is exactly where it should be.
		if len(cutShort) > 0 {
			return fleetbus.OutcomeApplied(fmt.Sprintf(
				"%s; nothing left to release — %d park(s) on this box were already released early by a peer instance",
				rosterWitness, len(cutShort)), 0)
		}
		return fleetbus.OutcomeApplied(rosterWitness+"; no park was holding a goal on this box", 0)
	}
	sort.Slice(holding, func(i, j int) bool { return holding[i].Goal < holding[j].Goal })

	reason := strings.TrimSpace(d.Reason)
	if reason == "" {
		reason = fmt.Sprintf("fleet seat-refresh: %d/%d pool seat(s) can serve", census.Usable, census.Pool)
	}
	var released, contended, failed []string
	for _, rec := range holding {
		_, relErr := r.park.Release(rec.Goal, r.instance, reason, now)
		switch {
		case relErr == nil:
			released = append(released, fmt.Sprintf("%s (account %s, %s of wall remaining)",
				rec.Goal, dashIfEmpty(rec.Account), guardParkRemaining(rec, now)))
		case errors.Is(relErr, goalpark.ErrClaimed):
			// Another instance on this box won the O_EXCL claim. That is the exactly-once
			// guarantee working, not an error: the park IS released, just not by us, and
			// counting it as Affected here would double-count one release across the fold.
			contended = append(contended, rec.Goal)
		default:
			failed = append(failed, rec.Goal+": "+relErr.Error())
		}
	}
	if len(released) == 0 && len(contended) == 0 {
		return fleetbus.OutcomeRefused(fleetbus.ApplyRefused,
			"PARK_RELEASE_FAILED: %s; all %d holding park(s) refused release: %s",
			rosterWitness, len(holding), strings.Join(failed, "; "))
	}
	witness := fmt.Sprintf("%s; released %d/%d holding park(s): %s",
		rosterWitness, len(released), len(holding), strings.Join(released, ", "))
	if len(released) == 0 {
		witness = fmt.Sprintf("%s; all %d holding park(s) were released early by a peer instance first", rosterWitness, len(contended))
	} else if len(contended) > 0 {
		witness += fmt.Sprintf("; %d released early by a peer instance first (%s)", len(contended), strings.Join(contended, ", "))
	}
	if len(failed) > 0 {
		// A partial release is still an apply, but the report must not hide the tail.
		witness += "; refused: " + strings.Join(failed, "; ")
	}
	return fleetbus.OutcomeApplied(witness, len(released))
}

// guardParkRemaining renders how much of a park's announced wall is left at now, so
// the witness says what was actually cut short rather than just that something was.
// It reads the injected now rather than the wall clock: the whole outcome is folded
// from one instant, and a witness measured against a second, later instant could
// report a different remaining wall than the release it is describing.
func guardParkRemaining(rec goalpark.Record, now time.Time) string {
	left := time.Unix(rec.ParkedUntil, 0).Sub(now).Round(time.Minute)
	if left < 0 {
		left = 0
	}
	return left.String()
}

// guardSeatCensusFromRegistry is the production roster read: the same registry
// `fak accounts status` reads, with the SAME cooldown overlay, so a seat sitting in a
// usage-limit window is not counted as able to serve. Being strict here is the safe
// direction — an under-count refuses the release and costs a re-send, an over-count
// releases a fleet back into a wall.
func guardSeatCensusFromRegistry() (guardSeatCensus, error) {
	path := guardSeatRegistryPath()
	if strings.TrimSpace(path) == "" {
		return guardSeatCensus{}, errors.New("no accounts registry path (set FAK_ACCOUNTS_REGISTRY, or a home dir fak can resolve)")
	}
	reg, err := accounts.LoadRegistry(path)
	if err != nil {
		return guardSeatCensus{Source: path}, err
	}
	// A cooldown store that will not load must not manufacture capacity: fall back to
	// the overlay-free report only when the store is genuinely absent, which
	// LoadCooldownStore already reports as an empty store and no error.
	cd, cdErr := accounts.LoadCooldownStore(defaultCooldownStorePath())
	if cdErr != nil {
		return guardSeatCensus{Source: path}, fmt.Errorf("cooldown store %s: %w", defaultCooldownStorePath(), cdErr)
	}
	rep := reg.Refresh().LoginReportAt(cd, time.Now())
	return guardSeatCensus{
		Usable: rep.Summary.CanServe,
		Pool:   rep.Summary.ActiveStyleSeats,
		Total:  rep.Summary.Total,
		Source: path,
	}, nil
}

// guardSeatRegistryPath mirrors `fak accounts`' own default derivation (its
// --registry default), so the guard re-reads the file the operator just enrolled into
// rather than a second registry nobody writes.
func guardSeatRegistryPath() string {
	if p := strings.TrimSpace(os.Getenv("FAK_ACCOUNTS_REGISTRY")); p != "" {
		return p
	}
	home, err := os.UserHomeDir()
	if err != nil || strings.TrimSpace(home) == "" {
		return ""
	}
	return filepath.Join(home, ".claude-accounts", "registry.json")
}

// --- arming ---------------------------------------------------------------- //

// resolveGuardFleetBusDir returns the bus directory to arm, or "" when --fleet-bus is
// off. Returning "" for the off case is what makes the opt-out TOTAL, exactly as it is
// for serve: startFleetBusInstance treats an empty dir as a byte-for-byte no-op, so a
// guard started with --fleet-bus=false never creates, opens or touches a bus
// directory. That matters more for a default-on flag than a default-off one — the
// operator who says no is the one who has to be able to prove nothing happened.
func resolveGuardFleetBusDir(on bool, dir string) string {
	if !on {
		return ""
	}
	if strings.TrimSpace(dir) != "" {
		return strings.TrimSpace(dir)
	}
	return defaultFleetBusDir()
}

func resolveGuardFleetBusID(id string) string {
	if strings.TrimSpace(id) != "" {
		return sanitizeBusToken(strings.TrimSpace(id))
	}
	return sanitizeBusToken(fmt.Sprintf("guard-%s-%d", fleetBusMachine(), os.Getpid()))
}

// startGuardFleetBus arms this guard as a bus instance and returns the stop func. It
// is the guard-shaped call onto startFleetBusInstance; everything role-specific — who
// it says it is, what it claims, what it declares it cannot do, and which applier owns
// the meaning — is stated here and nowhere else.
// startGuardFleetBus preserves bus arming while optionally suppressing successful
// initialization chatter for compact attended launch. Errors still spill.
func startGuardFleetBus(ctx context.Context, on bool, dir, id string, interval time.Duration, quiet ...bool) func() {
	suppressSuccess := len(quiet) > 0 && quiet[0]
	busDir := resolveGuardFleetBusDir(on, dir)
	if busDir == "" {
		return func() {}
	}
	instanceID := resolveGuardFleetBusID(id)
	return startFleetBusInstance(ctx, fleetBusArming{
		busDir:      busDir,
		instanceID:  instanceID,
		role:        guardFleetBusRole,
		logPrefix:   "fak guard",
		quiet:       suppressSuccess,
		interval:    interval,
		ops:         guardFleetBusAdvertisedOps(),
		unsupported: guardFleetBusUnsupportedOps(),
		applier: &guardBusApplier{
			// native is false and there is no flag that could make it true — see
			// guardFleetBusUnsupportedOps. The steer path never reaches this applier
			// anyway (guardBusApplier intercepts it), but the field is set honestly so a
			// future op that reads it cannot be misled.
			sessions: &fleetBusApplier{tbl: serveSessions, native: false, durability: serveSessionDurability, ctx: ctx},
			seats: guardSeatRefresher{
				census:   guardSeatCensusFromRegistry,
				park:     goalParkStore(),
				instance: instanceID,
			},
		},
	})
}
