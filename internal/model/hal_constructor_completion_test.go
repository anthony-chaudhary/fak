package model

import (
	"errors"
	"os"
	"runtime"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type constructorCompletionBackend struct {
	compute.Backend
	model      *Model
	stage      string
	fault      func()
	cleanup    func()
	kvCalls    int
	flushes    int
	pinnedAtKV bool
}

func (b *constructorCompletionBackend) NewKV(cfg compute.KVConfig) compute.KVStore {
	b.kvCalls++
	b.pinnedAtKV = b.model.weightCloser != nil && b.model.weightCloser.sessions == 1
	if b.stage == "before-transfer" {
		_ = b.model.CloseWeights()
		b.fault()
	}
	return b.Backend.NewKV(cfg)
}

func (b *constructorCompletionBackend) GraphReset() {
	if b.stage != "success" {
		_ = b.model.CloseWeights()
	}
	if b.stage == "after-transfer" {
		b.fault()
	}
}

func (b *constructorCompletionBackend) BeginBatch() {}
func (b *constructorCompletionBackend) FlushBatch() {
	b.flushes++
	if b.cleanup != nil {
		b.cleanup()
	}
}

type constructorCompletionOwnedBackend struct {
	*constructorCompletionBackend
	owner sync.Mutex
}

func (b *constructorCompletionOwnedBackend) ForwardExecutionOwner() *sync.Mutex { return &b.owner }

// fak-test:runtime medium est=1s lane=default
func TestBackendConstructorPreservesIncompleteUnwind(t *testing.T) {
	primary := errors.New("constructor operation failed")
	secondary := errors.New("constructor cleanup failed")
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, owned := range []bool{false, true} {
				ownerName := map[bool]string{false: "no-owner", true: "owned"}[owned]
				cases := []struct{ name, stage, fault string }{
					{"before-transfer-nil", "before-transfer", "nil"},
					{"before-transfer-nonnil", "before-transfer", "nonnil"},
					{"before-transfer-goexit", "before-transfer", "goexit"},
					{"after-transfer-nil", "after-transfer", "nil"},
					{"after-transfer-nonnil", "after-transfer", "nonnil"},
					{"after-transfer-goexit", "after-transfer", "goexit"},
					{"returned-error", "refusal", ""},
					{"success", "success", ""},
					{"wrapper-nil", "after-transfer", "nil"},
				}
				for _, tc := range cases {
					t.Run(ownerName+"/"+tc.name, func(t *testing.T) {
						m := v41ReducedModel(t)
						base := &constructorCompletionBackend{Backend: compute.Default(), model: m, stage: tc.stage}
						wrapper := &constructorCompletionOwnedBackend{constructorCompletionBackend: base}
						var be compute.Backend = base
						if owned {
							be = wrapper
						}
						base.fault = func() {
							switch tc.fault {
							case "nil":
								panic(nil)
							case "nonnil":
								panic(primary)
							case "goexit":
								runtime.Goexit()
							}
						}
						if tc.stage != "success" {
							base.cleanup = func() {
								if tc.fault == "nil" {
									panic(secondary)
								}
								panic(nil)
							}
						}
						if tc.stage == "refusal" {
							m.sharedExpertRings = map[compute.Backend]*SharedExpertRing{
								be: {m: m, be: be, closed: true},
							}
						}
						retirements := 0
						retiredUnderOwner := false
						m.SetWeightCloser(v41EligibilityCloser(func() error {
							retirements++
							if owned {
								if wrapper.owner.TryLock() {
									wrapper.owner.Unlock()
								} else {
									retiredUnderOwner = true
								}
							}
							if tc.stage != "success" {
								if tc.stage == "before-transfer" && tc.fault == "nil" {
									panic(nil)
								}
								panic(secondary)
							}
							return nil
						}))
						var session *Session
						var err error
						var payload any
						returned, boundaryReturned := false, false
						done := make(chan struct{})
						go func() {
							defer close(done)
							func() {
								defer func() { payload = recover() }()
								if tc.name == "wrapper-nil" {
									session = m.NewBackendSession(be)
								} else {
									session, err = m.NewBackendSessionChecked(be)
								}
								returned = true
							}()
							boundaryReturned = true
						}()
						<-done
						if returned != (tc.fault == "") || boundaryReturned == (tc.fault == "goexit") {
							t.Fatalf("constructor returned=%t recovery boundary returned=%t; fault=%q", returned, boundaryReturned, tc.fault)
						}
						if tc.fault == "nil" && mode == "0" {
							if _, ok := payload.(*runtime.PanicNilError); !ok {
								t.Fatalf("nil panic payload=%T, want *runtime.PanicNilError", payload)
							}
						} else {
							var want any
							if tc.fault == "nonnil" {
								want = primary
							}
							if payload != want {
								t.Fatalf("panic payload=%v, want %v", payload, want)
							}
						}
						if tc.stage == "refusal" {
							if session != nil || !errors.Is(err, ErrSharedRingClosed) {
								t.Fatalf("construction refusal: session=%p err=%v", session, err)
							}
						} else if err != nil || (session != nil) != (tc.stage == "success") {
							t.Fatalf("construction result: session=%p err=%v", session, err)
						}
						if tc.stage == "success" {
							if m.weightCloser.sessions != 1 || retirements != 0 || base.flushes != 0 {
								t.Fatal("successful construction did not transfer one live pin")
							}
							var active *WeightSessionsActiveError
							if !errors.As(m.CloseWeights(), &active) {
								t.Fatal("successful session did not retain model weights")
							}
							session.Close()
							session.Close()
						}
						wantFlushes := 1
						if tc.stage == "before-transfer" {
							wantFlushes = 0
						}
						if !base.pinnedAtKV || base.kvCalls != 1 || base.flushes != wantFlushes || retirements != 1 || m.weightCloser.sessions != 0 || retiredUnderOwner {
							t.Fatalf("pin/rollback: pinned=%t KV=%d flushes=%d want=%d retirements=%d pins=%d owned=%t", base.pinnedAtKV, base.kvCalls, base.flushes, wantFlushes, retirements, m.weightCloser.sessions, retiredUnderOwner)
						}
						if owned {
							if !wrapper.owner.TryLock() {
								t.Fatal("constructor retained execution ownership")
							}
							wrapper.owner.Unlock()
						}
					})
				}
			}
		})
	}
}
