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
// names the local write that actually happened and Affected counts the sessions that
// actually MOVED — a write that leaves the run state where it already was is reported,
// never counted; an instance that matched nothing REFUSES rather than acking a hollow
// "applied", because a fleet-wide report of "everybody applied, 0 affected" is the
// accepted-but-never-applied phantom wearing a success token.
//
// A fanned op refuses exactly as it would alone. `steer` on a non-native serve carries
// the gateway's own STEER_NO_OWNED_LOOP through in the ack detail — the fan-out does
// not soften a local refusal, and it does not invent one either.

import (
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/fleetbus"
	"github.com/anthony-chaudhary/fak/internal/gateway"
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
	gateway    gwBusApplier
	// native mirrors gateway.Server.ownsSessionLoop (unexported there): a non-native
	// serve has no owned loop to deliver an operator turn to, so a steer that reached
	// it would sit in a mailbox nothing drains.
	native bool
	// ctx bounds the steer sends to the serve lifetime.
	ctx context.Context
}

// gwBusApplier is the narrow, gateway-scoped control surface. Its vocabulary is
// deliberately disjoint from sessionctl: routing dispatches one token to one owner.
type gwBusApplier interface {
	ReloadRoute() (witness string, changed bool, err error)
}

const gwBusReloadRoute fleetbus.Op = "gateway-route-reload"

type gwBusOpSpec struct {
	Capability string
	Boundary   string
	Witness    string
}

var gwBusOps = map[fleetbus.Op]gwBusOpSpec{
	gwBusReloadRoute: {
		Capability: "gateway.route.reload",
		Boundary:   "configured route-manifest watcher",
		Witness:    "route reload event (source, generation, result)",
	},
}

func isGwBusOp(op fleetbus.Op) bool {
	_, ok := gwBusOps[op]
	return ok
}

func validateFleetBusVocabularies() error {
	for op := range gwBusOps {
		if _, overlap := sessionctl.Spec(sessionctl.ControlOp(op)); overlap {
			return fmt.Errorf("fleet-bus op %q is ambiguous between gateway and session appliers", op)
		}
	}
	// The guard adds a third vocabulary (guard_fleetbus.go). Checking it here rather
	// than in that file keeps ONE place that can answer "does this op have exactly one
	// owner", which is the only form of the question dispatch actually asks.
	for _, op := range guardBusOwnedOps() {
		if _, overlap := sessionctl.Spec(sessionctl.ControlOp(op)); overlap {
			return fmt.Errorf("fleet-bus op %q is ambiguous between guard and session appliers", op)
		}
		if isGwBusOp(op) {
			return fmt.Errorf("fleet-bus op %q is ambiguous between guard and gateway appliers", op)
		}
	}
	return nil
}

func (a *fleetBusApplier) applyGwBus(d fleetbus.Directive) fleetbus.Outcome {
	if a.gateway == nil {
		return fleetbus.OutcomeRefused(fleetbus.ApplyRefused, "%s is not configured on this gateway", d.Op)
	}
	witness, changed, err := a.gateway.ReloadRoute()
	if err != nil {
		return fleetbus.OutcomeRefused(fleetbus.ApplyRefused, "%s refused: %v", d.Op, err)
	}
	if strings.TrimSpace(witness) == "" {
		return fleetbus.OutcomeRefused(fleetbus.ApplyRefused, "%s returned no effect witness", d.Op)
	}
	if !changed {
		return fleetbus.OutcomeApplied("gateway-route-unchanged:"+witness, 0)
	}
	return fleetbus.OutcomeApplied("gateway-route-reloaded:"+witness, 1)
}

func (a *fleetBusApplier) Apply(d fleetbus.Directive) fleetbus.Outcome {
	if isGwBusOp(d.Op) {
		return a.applyGwBus(d)
	}
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
	// The PRE-write run state has to come from this one snapshot walk. Re-reading the
	// table after the transition would race a concurrent turn and attribute its move to
	// this directive; Table.Transition's own bool cannot answer it either, because that
	// bool means "not terminal", never "the write mattered" (internal/session/table.go).
	matched := a.matchedStates(d.Selector)
	if len(matched) == 0 {
		return fleetbus.OutcomeRefused(fleetbus.ApplyRefused, "%s", a.noMatchDetail(d.Selector))
	}
	ctx := a.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	changed, already := 0, 0
	for _, prev := range matched {
		st, took := a.tbl.Transition(prev.TraceID, run, reason)
		if !took {
			continue
		}
		if a.durability != nil && a.durability.registry != nil {
			if err := a.durability.writeThrough(ctx, prev.TraceID, st); err != nil {
				return fleetbus.OutcomeRefused(fleetbus.ApplyRefused,
					"%s applied in memory for %s but durable mirror failed: %v", op, prev.TraceID, err)
			}
		}
		// The write was taken either way — it is still mirrored, and the session is
		// still where the operator asked. Only a run-state CHANGE counts as Affected.
		if prev.Run == run {
			already++
			continue
		}
		changed++
	}
	if changed+already == 0 {
		return fleetbus.OutcomeRefused(fleetbus.ApplyRefused,
			"all %d matched session(s) refused %s (see the drive table's own tokens)", len(matched), op)
	}
	mirror, prefix := "in memory only (durability disabled)", "memory-only:"
	if a.durability != nil && a.durability.registry != nil {
		mirror, prefix = "and reached the durable mirror", "durable:"
	}
	// An all-no-op fan stays APPLIED with Affected 0 — a legal outcome the bus documents
	// (internal/fleetbus/fleetbus.go's Affected doc), and the one an operator can act on:
	// converting it to a refusal would trade an overcount for a false alarm about a fleet
	// that is exactly where they asked it to be. What it must NOT do is say "took it".
	var witness string
	switch {
	case changed == 0:
		witness = fmt.Sprintf("%s: no run-state change — all %d matched session(s) were already %s (%s)",
			op, already, run, mirror)
	case already == 0:
		witness = fmt.Sprintf("%s: %d/%d session(s) took it %s", op, changed, len(matched), mirror)
	default:
		witness = fmt.Sprintf("%s: %d/%d session(s) took it %s; %d were already %s",
			op, changed, len(matched), mirror, already, run)
	}
	out := fleetbus.OutcomeApplied(witness, changed)
	out.Witness = prefix + out.Witness
	return out
}

// matchedStates resolves the selector against ONE snapshot and keeps the whole record,
// not just the trace id. The run state it carries is the pre-write one a lifecycle fan
// needs to tell a change from a no-op write; dropping it here is what forced the old
// applier to read Transition's "not terminal" bool as if it meant "changed" (#5822).
func (a *fleetBusApplier) matchedStates(sel fleetbus.Selector) []session.State {
	var out []session.State
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
		out = append(out, st)
	}
	return out
}

func (a *fleetBusApplier) matchedSessions(sel fleetbus.Selector) []string {
	states := a.matchedStates(sel)
	if len(states) == 0 {
		return nil
	}
	out := make([]string, 0, len(states))
	for _, st := range states {
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

type serveGwBusApplier struct {
	srv  *gateway.Server
	addr string
}

func (a serveGwBusApplier) ReloadRoute() (string, bool, error) {
	if a.srv == nil || a.srv.RouteWatcher() == nil {
		return "", false, errors.New("route reload is not configured")
	}
	ev := a.srv.RouteWatcher().Reload()
	if ev.Err != nil {
		return "", false, ev.Err
	}
	return fmt.Sprintf("source=%s reloads=%d rejects=%d", ev.Path, ev.Reloads, ev.Rejects), ev.Reloaded, nil
}

// fleetBusArming is everything that differs between one bus instance and the next.
// The announce/drain loop below is identical for every binary that joins the bus —
// what a serve and a guard disagree about is only who they say they are, what they
// claim they can and cannot do, and which applier owns the meaning — so those are
// parameters and the loop is written once. Forking the loop per role is how the two
// would drift into announcing on different schedules or logging refusals only on one
// side, which is the failure the whole return path exists to prevent.
type fleetBusArming struct {
	// busDir is the directory transport root. Empty means "not armed" and is a
	// byte-for-byte no-op — see resolveFleetBusDir.
	busDir     string
	instanceID string
	// addr is the configured transport address published on Instance. Empty is
	// valid for roles/transports with no address.
	addr string
	// role is the instance record's role token ("serve", "guard"). Display only.
	role string
	// logPrefix is what this binary calls itself on stderr ("fak serve", "fak guard"),
	// so an operator reading a mixed log can tell which process refused.
	logPrefix string
	interval  time.Duration
	// ops / unsupported are the two capability CLAIMS on the presence record. Neither
	// routes; see fleetbus.Instance.
	ops         []fleetbus.Op
	unsupported []fleetbus.Op
	// applier owns op meaning for this role.
	applier fleetbus.Applier
}

// startFleetBusLoop arms this serve as a bus INSTANCE. It is the serve-shaped call
// onto startFleetBusInstance and stays positional because it is what serve's stages
// and its tests already call.
func startFleetBusLoop(ctx context.Context, busDir, instanceID string, interval time.Duration, tbl *session.Table, native bool, gwAppliers ...gwBusApplier) func() {
	ap := &fleetBusApplier{tbl: tbl, native: native, durability: serveSessionDurability, ctx: ctx}
	var addr string
	if len(gwAppliers) != 0 {
		ap.gateway = gwAppliers[0]
		// The serve gateway applier already crosses this call boundary. Carry the
		// transport address on that existing argument rather than inventing a
		// second loop entry point solely for one presence-record field.
		if serve, ok := gwAppliers[0].(serveGwBusApplier); ok {
			addr = serve.addr
		}
	}
	return startFleetBusInstance(ctx, fleetBusArming{
		busDir:     busDir,
		instanceID: instanceID,
		addr:       strings.TrimSpace(addr),
		role:       "serve",
		logPrefix:  "fak serve",
		interval:   interval,
		ops:        fleetBusAdvertisedOps(),
		applier:    ap,
	})
}

// startFleetBusInstance arms one process as a bus INSTANCE: announce presence, then
// drain every interval for the lifetime of ctx. It mirrors startGatewayUsageSnapshotLoop's
// shape — an empty dir is a byte-for-byte no-op, the returned stop func cancels, and the
// loop also exits on ctx so a caller that forgets stop cannot leak the goroutine.
//
// The FIRST announce is synchronous, before the goroutine starts. A boot that announced
// only on the first tick would leave a window where `fak fleet control send` refuses
// FLEETBUS_NO_TARGET against a fleet that is actually up — a refusal that is correct
// about the roster and wrong about the world.
func startFleetBusInstance(ctx context.Context, arm fleetBusArming) func() {
	if strings.TrimSpace(arm.busDir) == "" {
		return func() {}
	}
	prefix := strings.TrimSpace(arm.logPrefix)
	if prefix == "" {
		prefix = "fak"
	}
	if arm.interval <= 0 {
		arm.interval = DefaultFleetBusInterval
	}
	if arm.applier == nil {
		fmt.Fprintf(os.Stderr, "%s: --fleet-bus disabled: no applier was wired for role %q\n", prefix, arm.role)
		return func() {}
	}
	if err := validateFleetBusVocabularies(); err != nil {
		fmt.Fprintf(os.Stderr, "%s: --fleet-bus disabled: %v\n", prefix, err)
		return func() {}
	}
	bus, err := fleetbus.OpenDir(arm.busDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: --fleet-bus disabled: %v\n", prefix, err)
		return func() {}
	}
	machine, pid := fleetBusMachine(), fleetBusPID()
	stamp := func(now time.Time) (fleetbus.Instance, *fleetbus.Refusal) {
		inst, r := fleetbus.NewInstance(arm.instanceID, machine, arm.role, pid, arm.addr, arm.ops, now)
		if r != nil {
			return fleetbus.Instance{}, r
		}
		return inst.WithUnsupportedOps(arm.unsupported), nil
	}
	self, refusal := stamp(time.Now())
	if refusal != nil {
		fmt.Fprintf(os.Stderr, "%s: --fleet-bus disabled: %v\n", prefix, refusal)
		return func() {}
	}

	// announce re-stamps and republishes presence, returning the record the drain then
	// matches against. Re-stamping through NewInstance rather than poking SeenUTC keeps
	// the wire format owned by exactly one place.
	announce := func(now time.Time, prev fleetbus.Instance) fleetbus.Instance {
		inst, r := stamp(now)
		if r != nil {
			return prev
		}
		if err := bus.Announce(inst); err != nil {
			fmt.Fprintf(os.Stderr, "%s: fleet-bus announce failed (non-fatal): %v\n", prefix, err)
		}
		return inst
	}
	self = announce(time.Now(), self)
	fmt.Fprintf(os.Stderr, "%s: fleet-bus armed as %s on %s (drain every %s)\n", prefix, self.ID, bus.Root, arm.interval)
	if len(self.Unsupported) > 0 {
		// Say the closed half out loud at arm time. An instance that declares an op
		// unsupported is still addressed by it (that is deliberate — see
		// fleetbus.Instance.Unsupported), so the operator's first evidence would
		// otherwise be a refusal ack, long after the boot log could have told them.
		fmt.Fprintf(os.Stderr, "%s: fleet-bus %s declares unsupported: %v (it will be addressed and will refuse, never silently skipped)\n",
			prefix, self.ID, self.Unsupported)
	}

	loopCtx, cancel := context.WithCancel(ctx)
	go func() {
		t := time.NewTicker(arm.interval)
		defer t.Stop()
		for {
			select {
			case <-loopCtx.Done():
				return
			case now := <-t.C:
				self = announce(now, self)
				rep, err := fleetbus.Drain(bus, self, arm.applier, now)
				if err != nil {
					fmt.Fprintf(os.Stderr, "%s: fleet-bus drain failed (non-fatal): %v\n", prefix, err)
					continue
				}
				for _, msg := range rep.Errors {
					fmt.Fprintf(os.Stderr, "%s: fleet-bus drain: %s\n", prefix, msg)
				}
				if rep.Applied+rep.Refused+rep.Expired > 0 {
					fmt.Fprintf(os.Stderr, "%s: fleet-bus %s: applied=%d refused=%d expired=%d\n",
						prefix, self.ID, rep.Applied, rep.Refused, rep.Expired)
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

func resolveFleetBusIdentity(sf *serveFlags) fleetbus.ServeIdentity {
	var explicit, addr string
	var stdio bool
	if sf.fleetBusID != nil && strings.TrimSpace(*sf.fleetBusID) != "" {
		explicit = sanitizeBusToken(strings.TrimSpace(*sf.fleetBusID))
	}
	if sf.addr != nil {
		addr = strings.TrimSpace(*sf.addr)
	}
	if sf.stdio != nil {
		stdio = *sf.stdio
	}
	return fleetbus.ResolveServeIdentity(fleetbus.ServeIdentityRequest{
		ExplicitID: explicit,
		Machine:    fleetBusMachine(),
		Addr:       addr,
		PID:        fleetBusPID(),
		Stdio:      stdio,
	})
}

func resolveFleetBusID(sf *serveFlags) string {
	return resolveFleetBusIdentity(sf).ID
}

// defaultFleetBusInstanceID retains the old no-flags helper for address-less callers.
// The real serve path calls resolveFleetBusIdentity with its configured transport; an
// address-less caller necessarily gets the named process-local fallback.
func defaultFleetBusInstanceID() string {
	return fleetbus.ResolveServeIdentity(fleetbus.ServeIdentityRequest{
		Machine: fleetBusMachine(),
		PID:     fleetBusPID(),
	}).ID
}

var fleetBusPID = os.Getpid

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
	for op := range gwBusOps {
		ops = append(ops, op)
	}
	slices.Sort(ops)
	return ops
}
