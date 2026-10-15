package model

import (
	"errors"
	"os"
	"runtime"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=10ms lane=default
func TestBackendRetirementPanicPresence(t *testing.T) {
	primary := errors.New("primary operation failure")
	secondary := errors.New("secondary retirement failure")
	returnedError := errors.New("primary returned error")
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, tc := range []struct {
				name       string
				surface    string
				callbacks  []string
				body       string
				want       string
				legacyWant string
				goexit     bool
				attempts   int
			}{
				{name: "close-nil-then-nonnil", surface: "close", callbacks: []string{"nil", "secondary"}, want: "nil", attempts: 2},
				{name: "legacy-nil-then-nonnil", surface: "legacy", callbacks: []string{"nil", "secondary"}, want: "nil", legacyWant: "secondary", attempts: 2},
				{name: "close-nonnil-then-nil", surface: "close", callbacks: []string{"secondary", "nil"}, want: "secondary", attempts: 2},
				{name: "legacy-nonnil-then-nil", surface: "legacy", callbacks: []string{"secondary", "nil"}, want: "secondary", attempts: 2},
				{name: "close-multiple-nil", surface: "close", callbacks: []string{"nil", "nil"}, want: "nil", attempts: 2},
				{name: "legacy-multiple-nil", surface: "legacy", callbacks: []string{"nil", "nil"}, want: "nil", attempts: 2},
				{name: "returned-error", surface: "error", callbacks: []string{"nil", "secondary"}, attempts: 2},
				{name: "close-primary-nil", surface: "close", callbacks: []string{"nil", "secondary"}, body: "nil", want: "nil", attempts: 2},
				{name: "close-primary-nonnil", surface: "close", callbacks: []string{"nil", "secondary"}, body: "primary", want: "primary", attempts: 2},
				{name: "legacy-primary-nonnil", surface: "legacy", callbacks: []string{"nil", "secondary"}, body: "primary", want: "primary", attempts: 2},
				{name: "panic-over-returned-error", surface: "error", callbacks: []string{"nil", "secondary"}, body: "primary", want: "primary", attempts: 2},
				{name: "close-body-goexit", surface: "close", callbacks: []string{"nil", "secondary"}, body: "goexit", goexit: true, attempts: 2},
				{name: "close-callback-goexit", surface: "close", callbacks: []string{"normal", "goexit", "secondary"}, goexit: true, attempts: 2},
				{name: "legacy-callback-goexit", surface: "legacy", callbacks: []string{"normal", "goexit", "secondary"}, goexit: true, attempts: 2},
				{name: "close-success", surface: "close", callbacks: []string{"normal"}, attempts: 1},
			} {
				t.Run(tc.name, func(t *testing.T) {
					be := &closeCompletionBackend{Backend: compute.Default()}
					s := &Session{Backend: be}
					var conn *closeCompletionConn
					var coord *EPDecodeCoordinator
					if tc.surface == "close" {
						conn = &closeCompletionConn{}
						coord = &EPDecodeCoordinator{g: &DistComm{coord: conn}, v41Owner: s}
						s.M = &Model{Cfg: Config{ModelType: "deepseek41"}, epCoord: coord}
					}
					attempts := 0
					ownedDuringCallback := false
					invoke := func(kind string) {
						switch kind {
						case "nil":
							panic(nil)
						case "primary":
							panic(primary)
						case "secondary":
							panic(secondary)
						case "goexit":
							runtime.Goexit()
						}
					}
					queue := func(scope *backendExecutionScope) {
						for _, kind := range tc.callbacks {
							scope.afterRelease(func() {
								attempts++
								if be.owner.TryLock() {
									be.owner.Unlock()
								} else {
									ownedDuringCallback = true
								}
								if s.backendExecutionScope != nil {
									ownedDuringCallback = true
								}
								invoke(kind)
							})
						}
					}
					be.flush = func() {
						queue(s.backendExecutionScope)
						invoke(tc.body)
					}
					var payload any
					var err error
					returned, boundaryReturned := false, false
					done := make(chan struct{})
					go func() {
						defer close(done)
						func() {
							defer func() { payload = recover() }()
							if tc.surface == "close" {
								s.Close()
							} else {
								func() {
									scope := s.beginBackendExecution()
									if tc.surface == "error" {
										err = returnedError
										defer scope.finishError(&err)
									} else {
										defer scope.finish()
									}
									queue(scope)
									invoke(tc.body)
								}()
							}
							returned = true
						}()
						boundaryReturned = true
					}()
					<-done
					want := tc.want
					if tc.surface == "legacy" && want == "nil" && mode == "1" {
						want = tc.legacyWant
					}
					if returned != (want == "" && !tc.goexit) || boundaryReturned == tc.goexit {
						t.Fatalf("returned=%t boundaryReturned=%t; want panic=%q Goexit=%t", returned, boundaryReturned, want, tc.goexit)
					}
					var wantPayload any
					switch want {
					case "primary":
						wantPayload = primary
					case "secondary":
						wantPayload = secondary
					}
					if want == "nil" && mode == "0" {
						if _, ok := payload.(*runtime.PanicNilError); !ok {
							t.Fatalf("nil panic payload=%T, want *runtime.PanicNilError", payload)
						}
					} else if payload != wantPayload {
						t.Fatalf("panic payload=%v, want %v", payload, wantPayload)
					}
					if tc.surface == "error" && err != returnedError {
						t.Fatalf("returned error changed: %v", err)
					}
					if attempts != tc.attempts || ownedDuringCallback || s.backendExecutionScope != nil {
						t.Fatalf("callback attempts=%d want=%d owned=%t scope=%p", attempts, tc.attempts, ownedDuringCallback, s.backendExecutionScope)
					}
					if !be.owner.TryLock() {
						t.Fatal("retirement retained execution ownership")
					}
					be.owner.Unlock()
					if tc.surface == "close" {
						if !coord.mu.TryLock() {
							t.Fatal("retirement retained EP coordination ownership")
						}
						coord.mu.Unlock()
						wantAborts := 0
						if want != "" || tc.goexit {
							wantAborts = 1
						}
						if coord.v41Owner != nil || conn.closes != wantAborts {
							t.Fatalf("EP retirement: owner=%p abort attempts=%d, want %d", coord.v41Owner, conn.closes, wantAborts)
						}
						s.Close()
						if be.flushes != 1 || attempts != tc.attempts || conn.closes != wantAborts {
							t.Fatalf("repeated Close: flushes=%d callback attempts=%d EP abort attempts=%d", be.flushes, attempts, conn.closes)
						}
					}
				})
			}
		})
	}
}
