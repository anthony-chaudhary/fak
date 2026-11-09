package servelaunch

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// StopLadder is the teardown half of the launch lifecycle: an escalating,
// idempotent stop ladder that takes a launched unit from "running" to
// "confirmed gone" and refuses to report success on an unconfirmed exit.
//
// The ladder is data (StopConfig.Steps), not literals: each rung names what
// it delivers — a signal to the target's own pid, or a port-kill safety net
// that discovers pids by port — and how long to poll for a confirmed exit
// before the next rung fires. The shipped plan (DefaultSteps) mirrors the
// pinned study's graceful-teardown contract: SIGTERM, a bounded kill-0-style
// poll window, SIGKILL, then the port-kill safety net for holders invisible
// to the launched pid. Callers copy DefaultSteps and edit it; nothing here
// hardcodes the order or the windows.
//
// Every observation goes through an injectable seam: Signal delivers, Exited
// confirms (the production adapter is a kill-0 probe), PortKill discovers
// and signals by port, and Clock/NewTicker bound the polls exactly like the
// readiness wait does. No code path here sleeps; tests drive fake ticks by
// hand.

// Closed outcome sentinels for the stop ladder. Callers branch on these with
// errors.Is, never on error prose: the prose is free to change, the identity
// is the contract.
var (
	// ErrAlreadyStopped reports that the target was already gone when the
	// ladder started: stopping is idempotent and this is a typed no-op, not
	// a failure.
	ErrAlreadyStopped = errors.New("servelaunch: target already stopped")
	// ErrNotStopped reports that the full ladder ran and the target is still
	// alive: every rung fired, none confirmed an exit.
	ErrNotStopped = errors.New("servelaunch: target still alive after full stop ladder")
)

// Signal is the closed vocabulary of escalation signals a rung may deliver.
type Signal int

const (
	// SignalTerm requests a graceful stop (SIGTERM-class).
	SignalTerm Signal = iota
	// SignalKill forces the stop (SIGKILL-class).
	SignalKill
)

// RungKind selects what one ladder rung acts on.
type RungKind int

const (
	// RungSignal delivers Step.Signal to the target's own pid.
	RungSignal RungKind = iota
	// RungPortKill discovers the pids holding the target's port and signals
	// them: the safety net for child processes and rebinders that the
	// launched pid alone does not cover.
	RungPortKill
)

// Step is one rung of the escalation ladder, as data: what it delivers and
// how long to poll for a confirmed exit before the next rung fires. A
// non-positive WaitForExit means "check once, then escalate".
type Step struct {
	Name        string
	Kind        RungKind
	Signal      Signal
	WaitForExit time.Duration
}

// Defaults keep a zero-ish StopConfig usable; they are the shipped plan's
// numbers, exposed so callers can read or override them.
const (
	// DefaultTermGrace bounds the kill-0-style poll after a graceful stop
	// request before the ladder escalates.
	DefaultTermGrace = 30 * time.Second
	// DefaultKillGrace bounds the poll after a forced stop.
	DefaultKillGrace = 5 * time.Second
	// DefaultPortKillGrace bounds the poll after the port-kill safety net.
	DefaultPortKillGrace = 5 * time.Second
	// DefaultPollInterval is the exit-poll cadence between Exited checks.
	DefaultPollInterval = 500 * time.Millisecond
)

// DefaultSteps is the shipped escalation plan, as data: SIGTERM, bounded
// kill-0-style poll, SIGKILL, then the port-kill safety net. Copy and edit
// it rather than re-implementing the order.
func DefaultSteps() []Step {
	return []Step{
		{Name: "sigterm", Kind: RungSignal, Signal: SignalTerm, WaitForExit: DefaultTermGrace},
		{Name: "sigkill", Kind: RungSignal, Signal: SignalKill, WaitForExit: DefaultKillGrace},
		{Name: "port-kill", Kind: RungPortKill, Signal: SignalKill, WaitForExit: DefaultPortKillGrace},
	}
}

// StopConfig wires one stop attempt. The seams are injectable so no test
// touches a real process: Signal delivers to the target pid, Exited is the
// kill-0-style liveness probe, PortKill discovers and signals by port, and
// Clock/NewTicker bound the polls exactly like the readiness wait.
type StopConfig struct {
	// Steps is the escalation ladder, in firing order. Required.
	Steps []Step
	// Port is the port the target claimed; required by any RungPortKill step.
	Port int
	// Signal delivers one signal to the target's own pid. Required by any
	// RungSignal step.
	Signal func(sig Signal) error
	// Exited reports whether the target has exited (production: a kill-0
	// probe). Required: without a confirmation seam the ladder cannot report
	// anything true.
	Exited func() bool
	// PortKill discovers the pids holding Port and signals them; it returns
	// the pids it signalled. Required by any RungPortKill step — a missing
	// function is a config refusal, never a silently skipped safety net.
	PortKill func(port int, sig Signal) ([]int, error)
	// PollInterval is the exit-poll cadence.
	PollInterval time.Duration
	// Clock and NewTicker are the injectable time seams (see readiness.go).
	Clock     Clock
	NewTicker TickerFactory
}

// StopResult records which rungs fired and which one confirmed the exit.
type StopResult struct {
	// StoppedBy is the name of the rung whose effect was confirmed; "" when
	// the target is still alive.
	StoppedBy string
	// RungsFired lists the rungs that delivered something, in order.
	RungsFired []string
	// PortKilledPIDs are the pids the port-kill safety net signalled.
	PortKilledPIDs []int
	// Elapsed is measured at tick granularity, like the readiness wait.
	Elapsed time.Duration
}

// StopLadder runs the escalation ladder until the target's exit is
// confirmed, then stops escalating. It returns (*StopResult, nil) on a
// confirmed exit, ErrAlreadyStopped when the target was already gone up
// front (the idempotent no-op), and an error wrapping ErrNotStopped when the
// last rung's window closes with the target still alive. Delivery failures
// not explained by a confirmed exit, and context aborts, are returned
// wrapped and match none of the ladder sentinels.
func StopLadder(ctx context.Context, cfg StopConfig) (*StopResult, error) {
	if len(cfg.Steps) == 0 {
		return nil, errors.New("servelaunch: stop ladder needs at least one step")
	}
	if cfg.Exited == nil {
		return nil, errors.New("servelaunch: stop ladder requires an Exited probe")
	}
	for _, s := range cfg.Steps {
		switch s.Kind {
		case RungSignal:
			if cfg.Signal == nil {
				return nil, fmt.Errorf("servelaunch: step %q requires a Signal func", s.Name)
			}
		case RungPortKill:
			if cfg.PortKill == nil {
				return nil, fmt.Errorf("servelaunch: step %q requires a PortKill func", s.Name)
			}
			if cfg.Port <= 0 {
				return nil, fmt.Errorf("servelaunch: step %q requires a positive Port", s.Name)
			}
		default:
			return nil, fmt.Errorf("servelaunch: step %q has unknown kind %d", s.Name, s.Kind)
		}
	}
	cfg = normalizeStop(cfg)

	res := &StopResult{}
	if cfg.Exited() {
		return res, ErrAlreadyStopped
	}

	start := cfg.Clock.Now()
	now := start
	ticker := cfg.NewTicker(cfg.PollInterval)
	defer ticker.Stop()

	// waitExit polls at the configured cadence until the target is confirmed
	// exited or the step's window elapses; it reports a confirmed exit seen
	// on the deadline tick itself.
	waitExit := func(window time.Duration) (bool, error) {
		deadline := now.Add(window)
		for {
			if cfg.Exited() {
				return true, nil
			}
			select {
			case tv := <-ticker.Channel():
				now = tv
				if !now.Before(deadline) {
					return cfg.Exited(), nil
				}
			case <-ctx.Done():
				return false, ctx.Err()
			}
		}
	}

	lastFired := ""
	for _, step := range cfg.Steps {
		if err := ctx.Err(); err != nil {
			return res, fmt.Errorf("servelaunch: stop aborted: %w", err)
		}
		if cfg.Exited() {
			// A prior rung's effect landed late: credit it, escalate nothing.
			res.StoppedBy = lastFired
			res.Elapsed = now.Sub(start)
			return res, nil
		}

		switch step.Kind {
		case RungSignal:
			if err := cfg.Signal(step.Signal); err != nil {
				if cfg.Exited() {
					res.StoppedBy = lastFired
					res.Elapsed = now.Sub(start)
					return res, nil
				}
				return res, fmt.Errorf("servelaunch: signal delivery failed on step %q: %w", step.Name, err)
			}
		case RungPortKill:
			pids, err := cfg.PortKill(cfg.Port, step.Signal)
			if err != nil && !cfg.Exited() {
				return res, fmt.Errorf("servelaunch: port-kill failed on step %q: %w", step.Name, err)
			}
			if err == nil {
				res.PortKilledPIDs = append(res.PortKilledPIDs, pids...)
			}
		}
		res.RungsFired = append(res.RungsFired, step.Name)
		lastFired = step.Name

		confirmed, err := waitExit(step.WaitForExit)
		if err != nil {
			return res, fmt.Errorf("servelaunch: stop aborted: %w", err)
		}
		if confirmed {
			res.StoppedBy = step.Name
			res.Elapsed = now.Sub(start)
			return res, nil
		}
	}

	if cfg.Exited() {
		res.StoppedBy = lastFired
		res.Elapsed = now.Sub(start)
		return res, nil
	}
	res.Elapsed = now.Sub(start)
	return res, fmt.Errorf("%w: fired rungs %v", ErrNotStopped, res.RungsFired)
}

func normalizeStop(cfg StopConfig) StopConfig {
	if cfg.PollInterval <= 0 {
		cfg.PollInterval = DefaultPollInterval
	}
	if cfg.Clock == nil {
		cfg.Clock = realClock{}
	}
	if cfg.NewTicker == nil {
		cfg.NewTicker = DefaultTickerFactory
	}
	return cfg
}

// Unit is one launched unit of a multi-unit launch, listed in start order.
type Unit struct {
	// Name identifies the unit in rollback errors.
	Name string
	// Stop stops this unit; typically wraps StopLadder with the unit's own
	// StopConfig.
	Stop func(ctx context.Context) error
}

// Rollback stops already-started units in reverse start order: the last unit
// started is the first stopped, undoing a partial launch the way it
// happened. It keeps going after a failing unit so one wedged unit cannot
// strand the earlier ones, and joins the per-unit errors. An
// ErrAlreadyStopped from an already-gone unit is not a rollback failure.
func Rollback(ctx context.Context, units []Unit) error {
	var errs []error
	for i := len(units) - 1; i >= 0; i-- {
		u := units[i]
		if u.Stop == nil {
			continue
		}
		err := u.Stop(ctx)
		if err == nil || errors.Is(err, ErrAlreadyStopped) {
			continue
		}
		errs = append(errs, fmt.Errorf("unit %q: %w", u.Name, err))
	}
	return errors.Join(errs...)
}
