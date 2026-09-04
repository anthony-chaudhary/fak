package agentqueue

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
)

var controllerReconciled sync.Map // map[string]*atomic.Bool

type Controller struct {
	Store            Store
	FakPath          string
	Runner           CommandRunner
	Interval         time.Duration
	ReconcileOnStart bool
	Liveness         ProcessLivenessChecker
	RestartOptions   RestartOptions

	reconciled *atomic.Bool
}

func (c Controller) getReconciledBool() *atomic.Bool {
	if c.reconciled != nil {
		return c.reconciled
	}
	if c.Store.Path != "" {
		val, _ := controllerReconciled.LoadOrStore(c.Store.Path, &atomic.Bool{})
		return val.(*atomic.Bool)
	}
	return nil
}

func (c Controller) isReconciled() bool {
	b := c.getReconciledBool()
	return b != nil && b.Load()
}

func (c Controller) markReconciled() {
	b := c.getReconciledBool()
	if b != nil {
		b.Store(true)
	}
}

// WithReconcileOnStart enables startup restart reconciliation with the given
// liveness checker and options.
func (c Controller) WithReconcileOnStart(liveness ProcessLivenessChecker, opts RestartOptions) Controller {
	c.ReconcileOnStart = true
	c.Liveness = liveness
	c.RestartOptions = opts
	if c.reconciled == nil {
		c.reconciled = &atomic.Bool{}
	}
	return c
}

// ReconcileStartup executes restart reconciliation over the controller store.
func (c Controller) ReconcileStartup(ctx context.Context) (RestartReconciliation, Snapshot, error) {
	liveness := c.Liveness
	if liveness == nil {
		liveness = processalive.Check
	}
	rec, snap, err := c.Store.ReconcileRestart(ctx, liveness, c.RestartOptions)
	if err != nil {
		return RestartReconciliation{}, Snapshot{}, err
	}
	c.markReconciled()
	return rec, snap, nil
}

type TickReceipt struct {
	Generation string                 `json:"generation"`
	Plan       Receipt                `json:"plan"`
	Launches   []LaunchReceipt        `json:"launches"`
	Restart    *RestartReconciliation `json:"restart,omitempty"`
}

// Tick performs one fenced reserve-and-actuate cycle. Reservations are durable
// before any worker launch, so duplicate controllers and retries cannot exceed
// the pool maximum even when launch observation lags.
func (c Controller) Tick(ctx context.Context) (TickReceipt, error) {
	var restartRec *RestartReconciliation
	if c.ReconcileOnStart && !c.isReconciled() {
		liveness := c.Liveness
		if liveness == nil {
			liveness = processalive.Check
		}
		rec, _, err := c.Store.ReconcileRestart(ctx, liveness, c.RestartOptions)
		if err != nil {
			return TickReceipt{}, err
		}
		restartRec = &rec
		c.markReconciled()
	}

	observed, err := c.Store.Load()
	if err != nil {
		return TickReceipt{}, err
	}
	plan, reserved, err := c.Store.Reserve(ctx, observed.Generation)
	if err != nil {
		return TickReceipt{}, err
	}
	runner := c.Runner
	if runner == nil {
		runner = ExecRunner{}
	}
	launches, err := Actuate(ctx, c.FakPath, reserved, plan.Start, runner)
	if err != nil {
		return TickReceipt{Generation: reserved.Generation, Plan: plan, Launches: launches, Restart: restartRec}, err
	}
	return TickReceipt{Generation: reserved.Generation, Plan: plan, Launches: launches, Restart: restartRec}, nil
}

// Run sustains reconciliation until cancellation. It ticks immediately, then
// at Interval; cancellation is a normal stop rather than an error.
func (c Controller) Run(ctx context.Context, observe func(TickReceipt)) error {
	if c.Interval <= 0 {
		return errors.New("agentqueue: controller interval must be positive")
	}
	if err := ctx.Err(); err != nil {
		if errors.Is(err, context.Canceled) {
			return nil
		}
		return err
	}
	if c.reconciled == nil {
		c.reconciled = &atomic.Bool{}
	}
	for {
		receipt, err := c.Tick(ctx)
		if err != nil {
			if errors.Is(err, context.Canceled) {
				return nil
			}
			return err
		}
		if observe != nil {
			observe(receipt)
		}
		timer := time.NewTimer(c.Interval)
		select {
		case <-ctx.Done():
			if !timer.Stop() {
				<-timer.C
			}
			return nil
		case <-timer.C:
		}
	}
}
