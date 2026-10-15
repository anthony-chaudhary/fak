package model

import (
	"context"
	"errors"
	"os"
	"runtime"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestV41EligibilityFinishCompleted(t *testing.T) {
	primary, secondary := errors.New("request failed"), errors.New("retirement failed")
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, tc := range []struct{ name, fault, cleanup string }{
				{"success", "", ""},
				{"cleanup-nil", "", "nil"},
				{"cleanup-nonnil", "", "nonnil"},
				{"error-cleanup-nil", "error", "nil"},
				{"error-cleanup-nonnil", "error", "nonnil"},
				{"primary-nil", "nil", "nonnil"},
				{"primary-nonnil", "nonnil", "nil"},
				{"goexit", "goexit", "nonnil"},
			} {
				t.Run(tc.name, func(t *testing.T) {
					m, be := v41EligibilityFixture(t)
					e, err := m.AcquireV41BatchEligibility(context.Background(), be, 4)
					if err != nil || e == nil {
						t.Fatalf("acquisition: %v", err)
					}
					retirements := 0
					m.SetWeightCloser(v41EligibilityCloser(func() error {
						retirements++
						if tc.cleanup == "nil" {
							panic(nil)
						}
						if tc.cleanup == "nonnil" {
							panic(secondary)
						}
						return nil
					}))
					_ = m.CloseWeights()
					var got error
					var payload any
					returned, boundaryReturned := false, false
					done := make(chan struct{})
					go func() {
						defer close(done)
						func() {
							defer func() { payload = recover() }()
							got = func() (err error) {
								complete := false
								defer func() { e.FinishCompleted(&err, complete) }()
								switch tc.fault {
								case "nil":
									panic(nil)
								case "nonnil":
									panic(primary)
								case "goexit":
									runtime.Goexit()
								case "error":
									err = primary
								}
								complete = true
								return err
							}()
							returned = true
						}()
						boundaryReturned = true
					}()
					<-done
					wantReturn := tc.fault == "error" || (tc.fault == "" && tc.cleanup == "")
					if returned != wantReturn || boundaryReturned == (tc.fault == "goexit") {
						t.Fatalf("returned=%t boundaryReturned=%t fault=%q cleanup=%q", returned, boundaryReturned, tc.fault, tc.cleanup)
					}
					wantPanic := tc.fault
					if wantPanic == "" {
						wantPanic = tc.cleanup
					}
					if wantPanic == "nil" && mode == "0" {
						if _, ok := payload.(*runtime.PanicNilError); !ok {
							t.Fatalf("panic=%T", payload)
						}
					} else {
						var want any
						if wantPanic == "nonnil" {
							want = primary
							if tc.fault == "" {
								want = secondary
							}
						}
						if payload != want {
							t.Fatalf("panic=%v want=%v", payload, want)
						}
					}
					if (tc.fault == "error" && got != primary) || (tc.fault != "error" && got != nil) {
						t.Fatalf("error=%v", got)
					}
					e.FinishCompleted(&got, true)
					if retirements != 1 || m.weightCloser.sessions != 0 {
						t.Fatalf("retirements=%d pins=%d", retirements, m.weightCloser.sessions)
					}
				})
			}
		})
	}
}
