package model

import (
	"errors"
	"net"
	"os"
	"runtime"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type closeCompletionBackend struct {
	compute.Backend
	owner   sync.Mutex
	flush   func()
	flushes int
}

func (b *closeCompletionBackend) ForwardExecutionOwner() *sync.Mutex { return &b.owner }
func (b *closeCompletionBackend) BeginBatch()                        {}
func (b *closeCompletionBackend) FlushBatch() {
	b.flushes++
	b.flush()
}

type closeCompletionConn struct {
	net.Conn
	closes int
	fault  func()
}

func (c *closeCompletionConn) Close() error {
	c.closes++
	if c.fault != nil {
		c.fault()
	}
	return nil
}

// fak-test:runtime fast est=10ms lane=default
func TestSessionClosePreservesIncompleteUnwind(t *testing.T) {
	primary := errors.New("close body failed")
	secondary := errors.New("close retirement failed")
	for _, panicnil := range []string{"0", "1"} {
		t.Run("panicnil="+panicnil, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+panicnil)
			for _, tc := range []struct {
				name       string
				body       func()
				retirement func()
				abort      func()
				wantPanic  any
				nilPanic   bool
				goexit     bool
			}{
				{name: "success"},
				{name: "primary", body: func() { panic(primary) }, wantPanic: primary},
				{name: "nil-primary", body: func() { panic(nil) }, nilPanic: true},
				{name: "goexit", body: runtime.Goexit, goexit: true},
				{name: "retirement", retirement: func() { panic(secondary) }, wantPanic: secondary},
				{name: "primary-over-retirement", body: func() { panic(primary) }, retirement: func() { panic(secondary) }, wantPanic: primary},
				{name: "nil-primary-over-retirement", body: func() { panic(nil) }, retirement: func() { panic(secondary) }, nilPanic: true},
				{name: "primary-over-abort", body: func() { panic(primary) }, abort: func() { panic(secondary) }, wantPanic: primary},
				{name: "nil-primary-over-nil-abort", body: func() { panic(nil) }, abort: func() { panic(nil) }, nilPanic: true},
				{name: "goexit-over-cleanup", body: runtime.Goexit, retirement: func() { panic(secondary) }, abort: func() { panic(secondary) }, goexit: true},
			} {
				t.Run(tc.name, func(t *testing.T) {
					be := &closeCompletionBackend{Backend: compute.Default()}
					conn := &closeCompletionConn{fault: tc.abort}
					coord := &EPDecodeCoordinator{g: &DistComm{coord: conn}}
					m := &Model{Cfg: Config{ModelType: "deepseek41"}, epCoord: coord}
					s := &Session{M: m, Backend: be}
					coord.v41Owner = s
					retirements := 0
					be.flush = func() {
						if tc.retirement != nil {
							s.backendExecutionScope.afterRelease(func() {
								retirements++
								tc.retirement()
							})
						}
						if tc.body != nil {
							tc.body()
						}
					}
					var payload any
					returned, boundaryReturned := false, false
					done := make(chan struct{})
					go func() {
						defer close(done)
						func() {
							defer func() { payload = recover() }()
							s.Close()
							returned = true
						}()
						boundaryReturned = true
					}()
					<-done
					incomplete := tc.body != nil || tc.retirement != nil
					if returned == incomplete || boundaryReturned == tc.goexit {
						t.Fatalf("Close returned=%t recovery boundary returned=%t; incomplete=%t Goexit=%t", returned, boundaryReturned, incomplete, tc.goexit)
					}
					if tc.nilPanic && panicnil == "0" {
						if _, ok := payload.(*runtime.PanicNilError); !ok {
							t.Fatalf("nil panic payload=%T, want *runtime.PanicNilError", payload)
						}
					} else if payload != tc.wantPanic {
						t.Fatalf("panic payload=%v, want %v", payload, tc.wantPanic)
					}
					if !be.owner.TryLock() {
						t.Fatal("Close retained backend execution ownership")
					}
					be.owner.Unlock()
					if !coord.mu.TryLock() {
						t.Fatal("Close retained EP coordination ownership")
					}
					coord.mu.Unlock()
					wantAborts := 0
					if incomplete {
						wantAborts = 1
					}
					if s.backendExecutionScope != nil || coord.v41Owner != nil || conn.closes != wantAborts {
						t.Fatalf("retirement: scope=%p EP owner=%p aborts=%d, want %d", s.backendExecutionScope, coord.v41Owner, conn.closes, wantAborts)
					}
					s.Close()
					wantRetirements := 0
					if tc.retirement != nil {
						wantRetirements = 1
					}
					if be.flushes != 1 || retirements != wantRetirements || conn.closes != wantAborts {
						t.Fatalf("repeated Close: flushes=%d retirements=%d aborts=%d", be.flushes, retirements, conn.closes)
					}
				})
			}
		})
	}
}
