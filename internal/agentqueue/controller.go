package agentqueue

import (
	"context"
	"errors"
	"time"
)

type Controller struct {
	Store    Store
	FakPath  string
	Runner   CommandRunner
	Interval time.Duration
}

type TickReceipt struct {
	Generation string          `json:"generation"`
	Plan       Receipt         `json:"plan"`
	Launches   []LaunchReceipt `json:"launches"`
}

// Tick performs one fenced reserve-and-actuate cycle. Reservations are durable
// before any worker launch, so duplicate controllers and retries cannot exceed
// the pool maximum even when launch observation lags.
func (c Controller) Tick(ctx context.Context) (TickReceipt, error) {
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
		return TickReceipt{Generation: reserved.Generation, Plan: plan, Launches: launches}, err
	}
	return TickReceipt{Generation: reserved.Generation, Plan: plan, Launches: launches}, nil
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
