package main

// serve_fleetbus.go — the INSTANCE half of the fleet control bus (#5600, epic #5599).
// `fak fleet control send` publishes; this is what makes a directive true.
//
// The bus itself declares no ops (internal/fleetbus is tier 1 and cannot see
// sessionctl); op MEANING lives here, at the only layer that owns the live
// session.Table and the a2achan steer bus. So this file is the whole binding between
// "the fleet was told" and "this process did it":
//
//	steer                             -> steerSession per matched session (needs --native)
//	pause/resume/cancel/terminate/throttle -> the same Table.Transition write the
//	                                         single-session verbs ride
//	redirect / budget / priority      -> refused: payload/value ops a bare fan cannot name
//	anything else                     -> FLEETBUS_UNKNOWN_OP, acked, never dropped
//
// Two properties this file must not break:
//
// The ack reports what was OBSERVED, never what was enqueued. Every Outcome's witness
// names the local write that actually happened and Affected counts the sessions it
// landed on; an instance that matched nothing REFUSES rather than acking a hollow
// "applied", because a fleet-wide report of "everybody applied, 0 affected" is the
// accepted-but-never-applied phantom wearing a success token.
//
// A fanned op refuses exactly as it would alone. `steer` on a non-native serve carries
// the gateway's own STEER_NO_OWNED_LOOP through in the ack detail — the fan-out does
// not soften a local refusal, and it does not invent one either.

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/fleetbus"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
)

// DefaultFleetBusInterval is how often an armed instance announces itself and drains.
// It is well under fleetbus.DefaultInstanceTTL so a live instance never flickers out of
// the roster (which would shrink the denominator a control point measures against).
const DefaultFleetBusInterval = 5 * time.Second

// fleetBusApplier is the serve process's Applier. It holds only the two things op
// meaning needs: the drive table it can write, and whether this serve owns a loop a
// steer could ever reach.
type fleetBusApplier struct {
	tbl        *session.Table
	durability *sessionDurability
	// native mirrors gateway.Server.ownsSessionLoop (unexported there): a non-native
	// serve has no owned loop to deliver an operator turn to, so a steer that reached
	// it would sit in a mailbox nothing drains.
	native bool
	// ctx bounds the steer sends to the serve lifetime.
	ctx context.Context
}

func (a *fleetBusApplier) Apply(d fleetbus.Directive) fleetbus.Outcome {
	op := sessionctl.ControlOp(strings.TrimSpace(string(d.Op)))
	if _, known := sessionctl.Spec(op); !known {
		return fleetbus.OutcomeRefused(fleetbus.UnknownOp,
			"op %q is outside this instance's control vocabulary %v", d.Op, sessionctl.Ops())
	}
	if a.tbl == nil {
		return fleetbus.OutcomeRefused(fleetbus.ApplyRefused, "this instance has no session table to write")
	}
	if op == sessionctl.OpSteer {
		return a.applySteer(d)
	}
	if run, lifecycle := sessionctl.BroadcastRunState(op); lifecycle {
		return a.applyLifecycle(d, op, run)
	}
	// redirect carries an objective, budget/priority carry values. Refusing them here
	// is the same fence sessionctl.Broadcast sets, for the same reason: a bare fan-out
	// cannot name the payload, and guessing one would be worse than saying no.
	return fleetbus.OutcomeRefused(fleetbus.ApplyRefused,
		"op %q is not fannable without a per-session payload (fannable: steer, %v)", op, sessionctl.BroadcastableOps())
}

// applySteer delivers the directive's payload as an operator turn to every matched
// session. It refuses on the SAME grounds the single-session gateway route refuses a
// steer, carrying that route's own token, so an operator debugging a fanned refusal
// reads the local reason rather than a fleet-level paraphrase.
func (a *fleetBusApplier) applySteer(d fleetbus.Directive) fleetbus.Outcome {
	if !a.native {
		return fleetbus.OutcomeRefused(fleetbus.ApplyRefused,
			"STEER_NO_OWNED_LOOP: this serve does not own a session loop (start it with --native); an enqueued steer would sit in a mailbox nothing drains")
	}
	if strings.TrimSpace(d.Payload) == "" {
		return fleetbus.OutcomeRefused(fleetbus.ApplyRefused, "steer needs a --payload (the text to deliver)")
	}
	// Screen the append BEFORE any loop can see it, in the same position the
	// single-session route screens it (#2402: owned-loop gate, THEN the context
	// screen, then the send). Skipping it here would make the fan-out strictly more
	// permissive than the one-session route it claims to mirror — and that is the one
	// direction a fleet path must never differ, because the text an operator cannot
	// push into a single session would instead land in every session on every
	// instance, acked back as a witnessed success.
	if code, held := ctxmmu.ScreenBytes([]byte(d.Payload)); held {
		return fleetbus.OutcomeRefused(fleetbus.ApplyRefused,
			"STEER_QUARANTINED: the append tripped the context screen (%s); it is held as a quarantine stub and never reaches a loop",
			abi.ReasonName(code))
	}
	traces := a.matchedSessions(d.Selector)
	if len(traces) == 0 {
		return fleetbus.OutcomeRefused(fleetbus.ApplyRefused, "%s", a.noMatchDetail(d.Selector))
	}
	issuer := "fleet-control"
	if d.Issuer != "" {
		issuer = "fleet-control:" + d.Issuer
	}
	var steered int
	var failures []string
	for _, trace := range traces {
		if err := steerSession(a.ctx, trace, issuer, d.Payload); err != nil {
			failures = append(failures, trace+": "+err.Error())
			continue
		}
		steered++
	}
	if steered == 0 {
		return fleetbus.OutcomeRefused(fleetbus.ApplyRefused,
			"every steer refused (%d session(s)): %s", len(traces), strings.Join(failures, "; "))
	}
	witness := fmt.Sprintf("steered %d/%d session(s) at their next turn boundary", steered, len(traces))
	if len(failures) > 0 {
		// A partial fan is still an apply, but the report must not hide the tail.
		witness += "; refused: " + strings.Join(failures, "; ")
	}
	return fleetbus.OutcomeApplied(witness, steered)
}

// applyLifecycle fans one lifecycle op onto the drive table. With a session axis stated
// it goes through sessionctl.Broadcast (the selector resolver that already owns the
// lane/wave/label tag registry); with none stated it fans over every live session on
// this instance — Broadcast refuses a zero session selector by design, and the
// affirmative "which instances" was already stated one level up, which is the gate that
// makes the widening deliberate rather than accidental.
func (a *fleetBusApplier) applyLifecycle(d fleetbus.Directive, op sessionctl.ControlOp, run session.RunState) fleetbus.Outcome {
	reason := d.Reason
	if reason == "" {
		reason = "fleet control " + string(op)
	}
	matched := a.matchedSessions(d.Selector)
	if len(matched) == 0 {
		return fleetbus.OutcomeRefused(fleetbus.ApplyRefused, "%s", a.noMatchDetail(d.Selector))
	}
	ctx := a.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	applied := 0
	for _, traceID := range matched {
		st, took := a.tbl.Transition(traceID, run, reason)
		if !took {
			continue
		}
		if a.durability != nil && a.durability.registry != nil {
			if err := a.durability.writeThrough(ctx, traceID, st); err != nil {
				return fleetbus.OutcomeRefused(fleetbus.ApplyRefused,
					"%s applied in memory for %s but durable mirror failed: %v", op, traceID, err)
			}
		}
		applied++
	}
	if applied == 0 {
		return fleetbus.OutcomeRefused(fleetbus.ApplyRefused,
			"all %d matched session(s) refused %s (see the drive table's own tokens)", len(matched), op)
	}
	if a.durability == nil || a.durability.registry == nil {
		out := fleetbus.OutcomeApplied(
			fmt.Sprintf("%s: %d/%d session(s) took it in memory only (durability disabled)", op, applied, len(matched)),
			applied)
		out.Witness = "memory-only:" + out.Witness
		return out
	}
	out := fleetbus.OutcomeApplied(
		fmt.Sprintf("%s: %d/%d session(s) took it and reached the durable mirror", op, applied, len(matched)),
		applied)
	out.Witness = "durable:" + out.Witness
	return out
}

func (a *fleetBusApplier) matchedSessions(sel fleetbus.Selector) []string {
	var out []string
	narrow := sel.NarrowsSessions()
	bsel := sessionctl.BroadcastSelector{Lane: sel.Lane, Wave: sel.Wave, Label: sel.Label}
	for _, st := range a.tbl.Snapshot() {
		if !steerable(st.Run) {
			continue
		}
		if narrow {
			meta, tagged := sessionctl.SessionTag(st.TraceID)
			if !tagged || !bsel.Matches(meta) {
				continue
			}
		}
		out = append(out, st.TraceID)
	}
	return out
}

// steerable reports whether a session can still consume an operator turn. A steer is
// delivered AT THE NEXT TURN BOUNDARY, so this is the steer path's version of the
// terminal check Table.Transition already makes for a lifecycle op — without it, the
// drive table's snapshot (a retained per-trace record, not a live-loop registry) hands
// back sessions that have no next boundary, every enqueue into their dead mailboxes
// counts as Affected, and the "matched nothing => REFUSE" fence can never fire.
//
// Stopped is terminal. Draining and Terminating are ending — each advances at most one
// more step to take the stop, and neither will ever pick up new operator input, so an
// append to them is the enqueued-but-never-applied phantom by another route.
func steerable(run session.RunState) bool {
	switch run {
	case session.Running, session.Throttled, session.Paused:
		// Paused counts: a pause holds the session at a boundary rather than ending
		// it, and the steer lands when it resumes.
		return true
	default:
		return false
	}
}

// noMatchDetail explains WHICH kind of nothing this was. "This instance holds no
// sessions" and "it holds sessions, none tagged into your lane" are different operator
// problems with different fixes, and an ack that blurs them wastes the round trip.
func (a *fleetBusApplier) noMatchDetail(sel fleetbus.Selector) string {
	snapshot := a.tbl.Snapshot()
	var live int
	for _, st := range snapshot {
		if steerable(st.Run) {
			live++
		}
	}
	if live == 0 {
		if retained := len(snapshot); retained > 0 {
			// The distinction matters: a serve that has handled traffic retains a
			// record per trace long after those turns ended, so "0 live" against a
			// non-empty table is a finished fleet, not a misconfigured one.
			return fmt.Sprintf("no session on this instance can still take a turn (%d retained record(s), all stopped/draining)", retained)
		}
		return "no live session on this instance to apply to"
	}
	if sel.NarrowsSessions() {
		return fmt.Sprintf("none of this instance's %d live session(s) are tagged %s (an untagged session never matches a selector)",
			live, strings.TrimSpace(sessionctl.BroadcastSelector{Lane: sel.Lane, Wave: sel.Wave, Label: sel.Label}.String()))
	}
	return fmt.Sprintf("none of this instance's %d live session(s) matched", live)
}

// --- the drain loop -------------------------------------------------------- //

// startFleetBusLoop arms this serve as a bus INSTANCE: announce presence, then drain
// every interval for the lifetime of ctx. It mirrors startGatewayUsageSnapshotLoop's
// shape — an empty dir is a byte-for-byte no-op, the returned stop func cancels, and the
// loop also exits on ctx so a caller that forgets stop cannot leak the goroutine.
//
// The FIRST announce is synchronous, before the goroutine starts. A boot that announced
// only on the first tick would leave a window where `fak fleet control send` refuses
// FLEETBUS_NO_TARGET against a fleet that is actually up — a refusal that is correct
// about the roster and wrong about the world.
func startFleetBusLoop(ctx context.Context, busDir, instanceID string, interval time.Duration, tbl *session.Table, native bool) func() {
	if strings.TrimSpace(busDir) == "" {
		return func() {}
	}
	if interval <= 0 {
		interval = DefaultFleetBusInterval
	}
	bus, err := fleetbus.OpenDir(busDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak serve: --fleet-bus disabled: %v\n", err)
		return func() {}
	}
	machine, pid, ops := fleetBusMachine(), os.Getpid(), fleetBusAdvertisedOps()
	stamp := func(now time.Time) (fleetbus.Instance, *fleetbus.Refusal) {
		return fleetbus.NewInstance(instanceID, machine, "serve", pid, "", ops, now)
	}
	self, refusal := stamp(time.Now())
	if refusal != nil {
		fmt.Fprintf(os.Stderr, "fak serve: --fleet-bus disabled: %v\n", refusal)
		return func() {}
	}
	ap := &fleetBusApplier{tbl: tbl, native: native, durability: serveSessionDurability, ctx: ctx}

	// announce re-stamps and republishes presence, returning the record the drain then
	// matches against. Re-stamping through NewInstance rather than poking SeenUTC keeps
	// the wire format owned by exactly one place.
	announce := func(now time.Time, prev fleetbus.Instance) fleetbus.Instance {
		inst, r := stamp(now)
		if r != nil {
			return prev
		}
		if err := bus.Announce(inst); err != nil {
			fmt.Fprintf(os.Stderr, "fak serve: fleet-bus announce failed (non-fatal): %v\n", err)
		}
		return inst
	}
	self = announce(time.Now(), self)
	fmt.Fprintf(os.Stderr, "fak serve: fleet-bus armed as %s on %s (drain every %s)\n", self.ID, bus.Root, interval)

	loopCtx, cancel := context.WithCancel(ctx)
	go func() {
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-loopCtx.Done():
				return
			case now := <-t.C:
				self = announce(now, self)
				rep, err := fleetbus.Drain(bus, self, ap, now)
				if err != nil {
					fmt.Fprintf(os.Stderr, "fak serve: fleet-bus drain failed (non-fatal): %v\n", err)
					continue
				}
				for _, msg := range rep.Errors {
					fmt.Fprintf(os.Stderr, "fak serve: fleet-bus drain: %s\n", msg)
				}
				if rep.Applied+rep.Refused+rep.Expired > 0 {
					fmt.Fprintf(os.Stderr, "fak serve: fleet-bus %s: applied=%d refused=%d expired=%d\n",
						self.ID, rep.Applied, rep.Refused, rep.Expired)
				}
			}
		}
	}()
	return cancel
}

// resolveFleetBusDir returns the bus directory to arm, or "" when --fleet-bus is off.
// Returning "" for the off case is what makes startFleetBusLoop's no-op total: the
// disabled path never touches the filesystem, so an operator who did not ask for a
// control plane cannot end up with a half-created one on disk.
func resolveFleetBusDir(sf *serveFlags) string {
	if sf.fleetBus == nil || !*sf.fleetBus {
		return ""
	}
	if sf.fleetBusDir != nil && strings.TrimSpace(*sf.fleetBusDir) != "" {
		return strings.TrimSpace(*sf.fleetBusDir)
	}
	return defaultFleetBusDir()
}

func resolveFleetBusID(sf *serveFlags) string {
	if sf.fleetBusID != nil && strings.TrimSpace(*sf.fleetBusID) != "" {
		return sanitizeBusToken(strings.TrimSpace(*sf.fleetBusID))
	}
	return defaultFleetBusInstanceID()
}

// defaultFleetBusInstanceID names this process on the bus. Host+pid is stable for the
// process lifetime and unique across a box, which is exactly the scope of an instance
// record; an operator who wants a stable name across restarts passes --fleet-bus-id.
func defaultFleetBusInstanceID() string {
	return sanitizeBusToken(fmt.Sprintf("serve-%s-%d", fleetBusMachine(), os.Getpid()))
}

func fleetBusMachine() string {
	host, err := os.Hostname()
	if err != nil || strings.TrimSpace(host) == "" {
		return "unknown-host"
	}
	return sanitizeBusToken(host)
}

// fleetBusAdvertisedOps is what this instance SAYS it can apply. Display only — nothing
// routes on it (the roster's claim is not a witness, the ack is), but it lets
// `fak fleet control instances` show an operator what a send could ask for.
func fleetBusAdvertisedOps() []fleetbus.Op {
	ops := []fleetbus.Op{fleetbus.Op(sessionctl.OpSteer)}
	for _, op := range sessionctl.BroadcastableOps() {
		ops = append(ops, fleetbus.Op(op))
	}
	return ops
}
