package agentqueue

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
)

// defaultLaunchWindow bounds how long a fenced launching attempt may wait for
// the guarded dispatch wrapper to register its process identity. A launch that
// is never registered is left for fenced restart reconciliation rather than
// being silently relaunched.
const defaultLaunchWindow = 2 * time.Minute

var controllerReconciled sync.Map // map[string]*atomic.Bool

type Controller struct {
	Store            Store
	FakPath          string
	Runner           CommandRunner
	Interval         time.Duration
	ReconcileOnStart bool
	Liveness         ProcessLivenessChecker
	RestartOptions   RestartOptions
	LaunchWindow     time.Duration

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

func (c Controller) launchWindow() time.Duration {
	if c.LaunchWindow > 0 {
		return c.LaunchWindow
	}
	return defaultLaunchWindow
}

// launchNonce derives the launch nonce from the reserved attempt and the
// generation it was reserved against. It is fresh for each reservation round
// and stable across retries of the same round, so a repeated tick re-reads the
// winning nonce while a competing controller is fenced.
func launchNonce(attemptID, generation string) string {
	sum := sha256.Sum256([]byte("agentqueue:launch\x00" + attemptID + "\x00" + generation))
	return "nonce:" + hex.EncodeToString(sum[:16])
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
	// Fence every reserved attempt as launching BEFORE any process is created,
	// then hand the persisted attempt id, nonce, and absolute state path to the
	// guarded dispatch wrapper. A reservation whose fence fails is never
	// launched.
	statePath, err := c.Store.StatePath()
	if err != nil {
		return TickReceipt{Generation: reserved.Generation, Plan: plan, Restart: restartRec}, err
	}
	deadline := time.Now().UTC().Add(c.launchWindow())
	handoffs := make(map[string]LaunchHandoff, len(plan.Start))
	for _, start := range plan.Start {
		nonce := launchNonce(start.IdempotencyKey, observed.Generation)
		if _, _, err := c.Store.BeginLaunching(ctx, start.IdempotencyKey, nonce, deadline); err != nil {
			return TickReceipt{Generation: reserved.Generation, Plan: plan, Restart: restartRec}, fmt.Errorf("agentqueue: fence attempt %q for launch: %w", start.IdempotencyKey, err)
		}
		handoffs[start.IdempotencyKey] = LaunchHandoff{AttemptID: start.IdempotencyKey, Nonce: nonce, StatePath: statePath}
	}
	runner := c.Runner
	if runner == nil {
		runner = ExecRunner{}
	}
	launches, err := Actuate(ctx, c.FakPath, reserved, plan.Start, handoffs, runner)
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
