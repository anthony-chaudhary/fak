package gateway

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"
)

// This executes the production lifecycle primitive, not a replacement gateway.
// HTTP/readiness integration remains the responsibility of gateway package tests.
func TestDecodeConfirmLifecycleInterrupted(t *testing.T) {
	for _, mode := range []string{"timeout", "cancel"} {
		t.Run(mode, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				var life warmupProbeLifecycle
				if err := life.begin(); err != nil {
					t.Fatal(err)
				}
				held := true
				defer func() {
					if held {
						life.end()
					}
				}()
				release := make(chan struct{})
				defer close(release)
				started := make(chan context.Context, 1)
				result := make(chan error, 1)
				var finished atomic.Bool
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				go func() {
					result <- life.run(ctx, time.Second, func(probeCtx context.Context) {
						started <- probeCtx
						<-release // deliberately ignore cancellation until released
						finished.Store(true)
					})
				}()
				probeCtx := <-started
				want := ErrWarmupTimeout
				if mode == "cancel" {
					if err := life.begin(); !errors.Is(err, ErrWarmupInProgress) {
						t.Fatalf("concurrent begin = %v", err)
					}
					want = context.Canceled
					cancel()
				}
				if err := <-result; !errors.Is(err, want) {
					t.Fatalf("lifecycle returned %v while worker active; want %v", err, want)
				}
				if finished.Load() {
					t.Fatal("fake worker finished before release")
				}
				if probeCtx.Err() == nil {
					t.Fatal("probe child was not canceled")
				}
				if mode == "timeout" && ctx.Err() != nil {
					t.Fatal("timeout canceled parent")
				}
				life.end()
				held = false
				// Even a retry that would disable measurement cannot acquire the
				// warmup reservation until the previous backend acknowledges.
				if err := life.begin(); !errors.Is(err, ErrWarmupTimeout) {
					t.Fatalf("retry before acknowledgment = %v", err)
				}
				var extra atomic.Bool
				if err := life.run(context.Background(), time.Second, func(context.Context) { extra.Store(true) }); !errors.Is(err, ErrWarmupTimeout) {
					t.Fatalf("direct overlapping probe = %v", err)
				}
				if extra.Load() {
					t.Fatal("overlapping probe entered backend")
				}
				release <- struct{}{}
				synctest.Wait()
				if !finished.Load() {
					t.Fatal("late worker did not finish")
				}
				if err := life.begin(); err != nil {
					t.Fatalf("acknowledged retry: %v", err)
				}
				held = true
				if err := life.run(context.Background(), time.Second, func(context.Context) { extra.Store(true) }); err != nil {
					t.Fatalf("new probe after acknowledgment: %v", err)
				}
				if !extra.Load() {
					t.Fatal("acknowledged retry did not run")
				}
			})
		})
	}
}

func TestDecodeConfirmLifecycleCompleted(t *testing.T) {
	for _, measurementErr := range []error{nil, errors.New("measurement failed")} {
		name := "success"
		if measurementErr != nil {
			name = "measurement-error"
		}
		t.Run(name, func(t *testing.T) {
			var life warmupProbeLifecycle
			var probeCtx context.Context
			var observed error
			if err := life.begin(); err != nil {
				t.Fatal(err)
			}
			defer life.end()
			err := life.run(context.Background(), 0, func(ctx context.Context) {
				probeCtx = ctx
				observed = measurementErr
			})
			if err != nil || observed != measurementErr {
				t.Fatalf("completed measurement lifecycle=%v measurement=%v", err, observed)
			}
			if probeCtx == nil || probeCtx.Err() == nil {
				t.Fatal("completed child context not released")
			}
		})
	}
}

// Passing on both baseline and candidate rules out a blanket failure that merely
// refuses every probe, including a normally returned measurement.
func TestDecodeConfirmLifecycleAcknowledgesCompletion(t *testing.T) {
	var life warmupProbeLifecycle
	var completed bool
	err := life.run(context.Background(), time.Second, func(context.Context) { completed = true })
	if err != nil || !completed {
		t.Fatalf("normal completion lifecycle=%v completed=%v", err, completed)
	}
}

func TestDecodeConfirmLifecycleAlreadyCanceled(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var life warmupProbeLifecycle
		var called atomic.Bool
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		err := life.run(ctx, time.Second, func(context.Context) { called.Store(true) })
		synctest.Wait()
		if !errors.Is(err, context.Canceled) || called.Load() {
			t.Fatalf("already canceled lifecycle=%v called=%v", err, called.Load())
		}
	})
}

func TestDecodeConfirmLifecycleCompletedCancellation(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var life warmupProbeLifecycle
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		var measurement error
		err := life.run(ctx, time.Second, func(context.Context) {
			cancel()
			measurement = context.Canceled
		})
		synctest.Wait()
		if !errors.Is(err, context.Canceled) || !errors.Is(measurement, context.Canceled) {
			t.Fatalf("completed canceled callback lifecycle=%v measurement=%v", err, measurement)
		}
		// Positive termination acknowledgment permits a new attempt; it does not
		// retroactively turn the canceled attempt into a successful startup.
		if err := life.begin(); err != nil {
			t.Fatalf("completed callback was not acknowledged: %v", err)
		}
		life.end()
	})
}
