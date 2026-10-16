package compute

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fak-test:runtime medium est=1s
func TestDeviceRecoveryIncompleteCompletion(t *testing.T) {
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, stage := range []string{"rebuild", "validate"} {
				for _, fault := range []string{"nil", "error", "goexit"} {
					for _, budget := range []int{1, 3} {
						t.Run(fmt.Sprintf("%s/%s/budget=%d", stage, fault, budget), func(t *testing.T) {
							latch := NewDeviceFaultLatch("cuda", budget)
							latch.Observe(DeviceFaultContext, "original-fault", 31)
							started, release, ownerDone := make(chan struct{}), make(chan struct{}), make(chan struct{})
							var releaseOnce sync.Once
							var returned atomic.Bool
							var rebuilds, validations atomic.Int32
							ownerResult := make(chan error, 1)
							const waiters = 3
							results := make(chan error, waiters)
							// On a failing historical implementation, release only this test's
							// stranded waiters after the failure has already been observed.
							t.Cleanup(func() {
								releaseOnce.Do(func() { close(release) })
								select {
								case <-ownerDone:
								case <-time.After(3 * time.Second):
									return
								}
								latch.mu.Lock()
								if attempt := latch.recovery; attempt != nil {
									attempt.result = errors.New("test cleanup: incomplete recovery")
									latch.recovery = nil
									close(attempt.done)
								}
								latch.mu.Unlock()
							})
							fail := func() error {
								close(started)
								<-release
								switch fault {
								case "nil":
									panic(nil)
								case "error":
									panic(errors.New("selected recovery panic"))
								case "goexit":
									runtime.Goexit()
								}
								return nil
							}
							go func() {
								defer close(ownerDone)
								err := latch.Reconstruct(func() error {
									rebuilds.Add(1)
									if stage == "rebuild" {
										return fail()
									}
									return nil
								}, func() error {
									validations.Add(1)
									if stage == "validate" {
										return fail()
									}
									return nil
								})
								returned.Store(true)
								ownerResult <- err
							}()
							select {
							case <-started:
							case <-time.After(3 * time.Second):
								t.Fatal("owner did not reach selected callback")
							}
							for i := 0; i < waiters; i++ {
								go func() { results <- latch.Reconstruct(nil, nil) }()
							}
							waitForDeviceRecoveryWaiters(t, latch, waiters)
							wantSite, wantCode := "original-fault", 31
							if fault == "goexit" && budget == 3 {
								wantSite, wantCode = "newer-fault", 79
								latch.Observe(DeviceFaultExecution, wantSite, wantCode)
							}

							releaseOnce.Do(func() { close(release) })
							select {
							case <-ownerDone:
							case <-time.After(3 * time.Second):
								t.Fatal("owner did not finish")
							}
							if returned.Load() != (fault != "goexit") {
								t.Fatalf("fault=%s owner returned=%t", fault, returned.Load())
							}
							var first error
							for i := 0; i < waiters; i++ {
								got := receiveRecoveryResult(t, results)
								var refusal *DeviceFaultError
								if !errors.As(got, &refusal) || refusal.Site != wantSite || refusal.Code != wantCode || refusal.Attempts != 1 {
									t.Fatalf("incomplete recovery admitted waiter: %v", got)
								}
								if i == 0 {
									first = got
								} else if got != first {
									t.Fatal("waiters did not receive one shared result")
								}
							}
							if fault != "goexit" && <-ownerResult != first {
								t.Fatal("owner and waiters received different results")
							}
							wantValidations := int32(0)
							if stage == "validate" {
								wantValidations = 1
							}
							if rebuilds.Load() != 1 || validations.Load() != wantValidations {
								t.Fatalf("callbacks rebuild=%d validate=%d", rebuilds.Load(), validations.Load())
							}
							snapshot := latch.Snapshot()
							wantHealth := DevicePoisoned
							if budget == 1 {
								wantHealth = DeviceUnrecoverable
							}
							if snapshot.Health != wantHealth || snapshot.Attempts != 1 || snapshot.Recoveries != 0 || latch.Admit("next-operation") == nil {
								t.Fatalf("incomplete recovery reopened serving: %+v", snapshot)
							}
							retryCalls := 0
							retry := latch.Reconstruct(func() error { retryCalls++; return nil }, func() error { retryCalls++; return nil })
							if budget == 1 {
								if retry == nil || retryCalls != 0 {
									t.Fatal("exhausted attempt retried")
								}
							} else if retry != nil || retryCalls != 2 || latch.Admit("after-validated-retry") != nil || latch.Snapshot().Recoveries != 1 {
								t.Fatalf("completed later retry failed: err=%v calls=%d", retry, retryCalls)
							}
						})
					}
				}
			}
		})
	}
}

// fak-test:runtime fast est=1ms
func TestDeviceRecoveryStepPreservesErrors(t *testing.T) {
	primary := errors.New("primary")
	if got := runDeviceRecoveryStep(func() error { return primary }); got != primary {
		t.Fatalf("returned primary changed: %v", got)
	}
	if got := runDeviceRecoveryStep(func() error { panic(primary) }); got != primary {
		t.Fatalf("panic primary changed: %v", got)
	}
	if got := runDeviceRecoveryStep(nil); got != nil {
		t.Fatalf("nil step changed: %v", got)
	}
	if got := runDeviceRecoveryStep(func() error { return nil }); got != nil {
		t.Fatalf("successful step changed: %v", got)
	}
}
