package model

import (
	"context"
	"errors"
	"os"
	"runtime"
	"testing"
)

type eligibilityCompletionContext struct {
	context.Context
	fail func() error
}

func (c eligibilityCompletionContext) Err() error { return c.fail() }

// fak-test:runtime fast est=100ms lane=default
func TestV41EligibilityAcquirePreservesIncompleteUnwind(t *testing.T) {
	primary := errors.New("eligibility probe failed")
	secondary := errors.New("eligibility retirement failed")
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, deny := range []bool{false, true} {
				name := map[bool]string{false: "success", true: "fallback"}[deny]
				t.Run(name, func(t *testing.T) {
					m, be := v41EligibilityFixture(t)
					be.denyShared = deny
					retirements := 0
					m.SetWeightCloser(v41EligibilityCloser(func() error { retirements++; return nil }))
					e, err := m.AcquireV41BatchEligibility(context.Background(), be, 4)
					if err != nil || e == nil || e.Eligible() == deny || m.weightCloser.sessions != 1 || retirements != 0 {
						t.Fatal("successful acquisition did not transfer one live pin")
					}
					var active *WeightSessionsActiveError
					if !errors.As(m.CloseWeights(), &active) || retirements != 0 {
						t.Fatal("transferred eligibility did not retain weights")
					}
					e.Finish(&err)
					e.Finish(&err)
					if retirements != 1 || m.weightCloser.sessions != 0 {
						t.Fatal("eligibility did not release its pin exactly once")
					}
				})
			}
			for _, fault := range []string{"nil", "nonnil", "goexit", "error"} {
				for _, cleanup := range []string{"nil", "nonnil"} {
					t.Run(fault+"/cleanup-"+cleanup, func(t *testing.T) {
						m, be := v41EligibilityFixture(t)
						retirements := 0
						retiredUnderOwner := false
						m.SetWeightCloser(v41EligibilityCloser(func() error {
							retirements++
							if be.owner.TryLock() {
								be.owner.Unlock()
							} else {
								retiredUnderOwner = true
							}
							if cleanup == "nil" {
								panic(nil)
							}
							panic(secondary)
						}))
						be.probe = func() {
							_ = m.CloseWeights()
							switch fault {
							case "nil":
								panic(nil)
							case "nonnil":
								panic(primary)
							case "goexit":
								runtime.Goexit()
							}
						}
						ctx := context.Context(context.Background())
						if fault == "error" {
							ctx = eligibilityCompletionContext{Context: ctx, fail: func() error {
								_ = m.CloseWeights()
								return context.Canceled
							}}
						}
						var eligibility *V41BatchEligibility
						var err error
						var payload any
						returned, boundaryReturned := false, false
						done := make(chan struct{})
						go func() {
							defer close(done)
							func() {
								defer func() { payload = recover() }()
								eligibility, err = m.AcquireV41BatchEligibility(ctx, be, 4)
								returned = true
							}()
							boundaryReturned = true
						}()
						<-done
						if returned != (fault == "error") || boundaryReturned == (fault == "goexit") {
							t.Fatalf("acquisition returned=%t recovery boundary returned=%t; fault=%q", returned, boundaryReturned, fault)
						}
						if fault == "nil" && mode == "0" {
							if _, ok := payload.(*runtime.PanicNilError); !ok {
								t.Fatalf("nil panic payload=%T, want *runtime.PanicNilError", payload)
							}
						} else {
							var want any
							if fault == "nonnil" {
								want = primary
							}
							if payload != want {
								t.Fatalf("panic payload=%v, want %v", payload, want)
							}
						}
						var wantErr error
						wantProbes := 1
						if fault == "error" {
							wantErr, wantProbes = context.Canceled, 0
						}
						if eligibility != nil || err != wantErr || be.probes != wantProbes || retirements != 1 || m.weightCloser.sessions != 0 || retiredUnderOwner {
							t.Fatalf("eligibility=%p err=%v probes=%d retirements=%d pins=%d owned=%t", eligibility, err, be.probes, retirements, m.weightCloser.sessions, retiredUnderOwner)
						}
						if !be.owner.TryLock() {
							t.Fatal("acquisition left backend owner locked")
						}
						be.owner.Unlock()
					})
				}
			}
		})
	}
}
