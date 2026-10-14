package model

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"net"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=10ms lane=default
func TestEPFollowerCleanupContract(t *testing.T) {
	t.Run("successful-reset-and-shutdown", func(t *testing.T) {
		conn := newEPFollowerCleanupConn(epOpPrefill, epOpPrefill, epOpShutdown)
		var backends []*epFollowerCleanupBackend
		err := RunEPFollower(&DistComm{rank: 1, size: 3, coord: conn}, func() *Session {
			be := &epFollowerCleanupBackend{}
			backends = append(backends, be)
			return &Session{Backend: be}
		})
		if err != nil || len(backends) != 2 || conn.closes != 0 {
			t.Fatalf("successful loop changed: err=%v factories=%d aborts=%d", err, len(backends), conn.closes)
		}
		for _, be := range backends {
			be.wantRetired(t)
		}
	})
	t.Run("normal-error-identity-and-unwrapping", func(t *testing.T) {
		original := errors.New("ordinary failure")
		if got := epFollowerAppendFailure(original, epFollowerCleanup("clean", func() {})); got != original {
			t.Fatal("clean retirement replaced an ordinary returned error")
		}
		conn := newEPFollowerCleanupConn(epOpPrefill)
		conn.readErr = original
		be := &epFollowerCleanupBackend{}
		err := RunEPFollower(&DistComm{rank: 1, size: 3, coord: conn}, func() *Session { return &Session{Backend: be} })
		var carrier *EPFollowerPanicError
		if !errors.Is(err, original) || errors.As(err, &carrier) || conn.closes != 1 {
			t.Fatalf("ordinary loop failure misclassified: %v aborts=%d", err, conn.closes)
		}
		be.wantRetired(t)
	})
	t.Run("reset-clears-slot-before-close", func(t *testing.T) {
		closeValue := &struct{ label string }{"reset Close"}
		conn := newEPFollowerCleanupConn(epOpPrefill, epOpPrefill, epOpShutdown)
		be := &epFollowerCleanupBackend{closePanic: closeValue}
		factories := 0
		err := RunEPFollower(&DistComm{rank: 1, size: 3, coord: conn}, func() *Session {
			factories++
			return &Session{Backend: be}
		})
		var carrier *EPFollowerPanicError
		if !errors.As(err, &carrier) || carrier.PanicValue() != closeValue || factories != 1 || conn.closes != 1 {
			t.Fatalf("reset failure changed ownership: %v factories=%d aborts=%d", err, factories, conn.closes)
		}
		// Each public Prefill/Close consults the execution owner twice. A
		// duplicate public Close is visible even though closeOnce skips Flush.
		be.wantRetired(t)
	})
	t.Run("shutdown-close-failure-aborts", func(t *testing.T) {
		closeErr, abortErr := errors.New("final Close"), errors.New("abort Close")
		conn := newEPFollowerCleanupConn(epOpPrefill, epOpShutdown)
		conn.closePanic = abortErr
		be := &epFollowerCleanupBackend{closePanic: closeErr}
		err := RunEPFollower(&DistComm{rank: 1, size: 3, coord: conn}, func() *Session { return &Session{Backend: be} })
		if !errors.Is(err, closeErr) || !errors.Is(err, abortErr) || conn.closes != 1 {
			t.Fatalf("SHUTDOWN hid retirement failure: %v aborts=%d", err, conn.closes)
		}
		wantEPFollowerPanicValues(t, err, []any{closeErr, abortErr})
		be.wantRetired(t)
	})
	for _, primary := range []struct {
		name  string
		value any
	}{
		{"error", errors.New("forward error panic")},
		{"pointer", &struct{ label string }{"forward pointer panic"}},
		{"string", "forward string panic"},
	} {
		t.Run("primary-"+primary.name+"-survives-cleanup-panics", func(t *testing.T) {
			abortValue, closeValue := errors.New("abort panic"), errors.New("Session Close panic")
			conn := newEPFollowerCleanupConn(epOpPrefill)
			conn.readPanic, conn.closePanic = primary.value, abortValue
			be := &epFollowerCleanupBackend{closePanic: closeValue}
			err := RunEPFollower(&DistComm{rank: 1, size: 3, coord: conn}, func() *Session { return &Session{Backend: be} })
			var carrier *EPFollowerPanicError
			if !errors.As(err, &carrier) || carrier.PanicValue() != primary.value || conn.closes != 1 ||
				!errors.Is(err, abortValue) || !errors.Is(err, closeValue) {
				t.Fatalf("primary identity lost: %v aborts=%d", err, conn.closes)
			}
			if cause, ok := primary.value.(error); ok && !errors.Is(err, cause) {
				t.Fatal("error-valued primary panic lost errors.Is")
			}
			wantEPFollowerPanicValues(t, err, []any{primary.value, abortValue, closeValue})
			be.wantRetired(t)
		})
	}
	t.Run("factory-panic-has-no-returned-session", func(t *testing.T) {
		value := &struct{ label string }{"factory"}
		conn := newEPFollowerCleanupConn(epOpPrefill)
		err := RunEPFollower(&DistComm{rank: 2, size: 3, coord: conn}, func() *Session { panic(value) })
		wantEPFollowerPanicValues(t, err, []any{value})
		if conn.closes != 1 {
			t.Fatal("factory panic did not abort")
		}
	})
}

func wantEPFollowerPanicValues(t *testing.T, err error, want []any) {
	t.Helper()
	var got []any
	var visit func(error)
	visit = func(e error) {
		switch e := e.(type) {
		case *EPFollowerPanicError:
			got = append(got, e.PanicValue())
		case interface{ Unwrap() []error }:
			for _, child := range e.Unwrap() {
				visit(child)
			}
		case interface{ Unwrap() error }:
			visit(e.Unwrap())
		}
	}
	visit(err)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("panic ordering got=%v want=%v", got, want)
	}
}

// Only existing ownership/batch hooks are implemented. The embedded nil
// backend makes any unexpected compute call fail; empty Prefill touches no
// weights or math. Each public operation consults this owner twice.
type epFollowerCleanupBackend struct {
	compute.Backend
	owner              sync.Mutex
	ownerCalls, closes int
	closePanic         any
}

func (b *epFollowerCleanupBackend) ForwardExecutionOwner() *sync.Mutex {
	b.ownerCalls++
	return &b.owner
}
func (*epFollowerCleanupBackend) BeginBatch() { panic("unexpected compute batch") }
func (b *epFollowerCleanupBackend) FlushBatch() {
	b.closes++
	if b.closePanic != nil {
		panic(b.closePanic)
	}
}
func (b *epFollowerCleanupBackend) wantRetired(t *testing.T) {
	t.Helper()
	if b.closes != 1 || b.ownerCalls != 4 {
		t.Fatalf("public Session retirement was not once: closes=%d ownerCalls=%d", b.closes, b.ownerCalls)
	}
	if !b.owner.TryLock() {
		t.Fatal("Session execution owner remained locked")
	}
	b.owner.Unlock()
}

// Independently construct at most three tiny empty-token reply frames. This
// exercises the public RunEPFollower loop over in-memory bytes, never sockets.
func newEPFollowerCleanupConn(ops ...epDecodeOp) *epFollowerCleanupConn {
	if len(ops) > 3 {
		panic("fixture frame cap exceeded")
	}
	var wire []byte
	for _, op := range ops {
		frame := make([]byte, 14)
		binary.LittleEndian.PutUint32(frame[:4], 10)
		frame[4] = 0 // Successful broadcast reply.
		frame[5] = byte(op)
		// pos and token count are zero.
		wire = append(wire, frame...)
	}
	return &epFollowerCleanupConn{reader: bytes.NewReader(wire)}
}

type epFollowerCleanupConn struct {
	reader                *bytes.Reader
	readErr               error
	readPanic, closePanic any
	closes, written       int
}

func (c *epFollowerCleanupConn) Read(p []byte) (int, error) {
	if c.reader.Len() == 0 {
		if c.readPanic != nil {
			panic(c.readPanic)
		}
		if c.readErr != nil {
			return 0, c.readErr
		}
		return 0, io.EOF
	}
	return c.reader.Read(p)
}
func (c *epFollowerCleanupConn) Write(p []byte) (int, error) {
	c.written += len(p)
	if c.written > 64 {
		panic("fixture announcement cap exceeded")
	}
	return len(p), nil
}
func (c *epFollowerCleanupConn) Close() error {
	c.closes++
	if c.closePanic != nil {
		panic(c.closePanic)
	}
	return nil
}
func (*epFollowerCleanupConn) LocalAddr() net.Addr              { return nil }
func (*epFollowerCleanupConn) RemoteAddr() net.Addr             { return nil }
func (*epFollowerCleanupConn) SetDeadline(time.Time) error      { return nil }
func (*epFollowerCleanupConn) SetReadDeadline(time.Time) error  { return nil }
func (*epFollowerCleanupConn) SetWriteDeadline(time.Time) error { return nil }

// fak-test:runtime fast est=10ms lane=default
func TestEPFollowerRetirementResult(t *testing.T) {
	t.Run("reset-and-final-close-count-all-accepted-sessions", func(t *testing.T) {
		conn := newEPFollowerCleanupConn(epOpPrefill, epOpPrefill, epOpShutdown)
		var backends []*epFollowerCleanupBackend
		out := RunEPFollowerWithResult(&DistComm{rank: 1, size: 3, coord: conn}, func() *Session {
			be := &epFollowerCleanupBackend{}
			backends = append(backends, be)
			return &Session{Backend: be}
		})
		want := EPFollowerRetirement{AcceptedSessions: 2, CloseAttempts: 2, CloseCompletions: 2, Available: true}
		if out.Err != nil || out.Retirement != want || len(backends) != 2 {
			t.Fatalf("reset retirement: %+v factories=%d", out, len(backends))
		}
		for _, be := range backends {
			be.wantRetired(t)
		}
	})
	for _, reset := range []bool{false, true} {
		name := "final-close-panic"
		ops := []epDecodeOp{epOpPrefill, epOpShutdown}
		if reset {
			name, ops = "reset-close-panic", []epDecodeOp{epOpPrefill, epOpPrefill}
		}
		t.Run(name, func(t *testing.T) {
			primary := errors.New("Close did not return")
			be := &epFollowerCleanupBackend{closePanic: primary}
			conn := newEPFollowerCleanupConn(ops...)
			factories := 0
			out := RunEPFollowerWithResult(&DistComm{rank: 1, size: 3, coord: conn}, func() *Session {
				factories++
				return &Session{Backend: be}
			})
			want := EPFollowerRetirement{AcceptedSessions: 1, CloseAttempts: 1, Available: true}
			var carrier *EPFollowerPanicError
			if !errors.Is(out.Err, primary) || !errors.As(out.Err, &carrier) || carrier.PanicValue() != primary ||
				out.Retirement != want || factories != 1 || conn.closes != 1 {
				t.Fatalf("failed Close was credited: %+v factories=%d aborts=%d", out, factories, conn.closes)
			}
			be.wantRetired(t) // Exactly one public attempt, not proof of completion.
		})
	}
	t.Run("execution-error-with-completed-deferred-close", func(t *testing.T) {
		primary := errors.New("execution canceled")
		conn := newEPFollowerCleanupConn(epOpPrefill)
		conn.readErr = primary
		be := &epFollowerCleanupBackend{}
		out := RunEPFollowerWithResult(&DistComm{rank: 2, size: 3, coord: conn}, func() *Session { return &Session{Backend: be} })
		want := EPFollowerRetirement{AcceptedSessions: 1, CloseAttempts: 1, CloseCompletions: 1, Available: true}
		if !errors.Is(out.Err, primary) || out.Retirement != want {
			t.Fatalf("execution error erased completed retirement: %+v", out)
		}
		be.wantRetired(t)
	})
	t.Run("factory-panic-after-successful-reset-close", func(t *testing.T) {
		primary := &struct{ label string }{"replacement factory"}
		conn := newEPFollowerCleanupConn(epOpPrefill, epOpPrefill)
		be := &epFollowerCleanupBackend{}
		calls := 0
		out := RunEPFollowerWithResult(&DistComm{rank: 1, size: 3, coord: conn}, func() *Session {
			calls++
			if calls == 2 {
				panic(primary)
			}
			return &Session{Backend: be}
		})
		want := EPFollowerRetirement{AcceptedSessions: 1, CloseAttempts: 1, CloseCompletions: 1, Available: true}
		var carrier *EPFollowerPanicError
		if !errors.As(out.Err, &carrier) || carrier.PanicValue() != primary || out.Retirement != want || calls != 2 {
			t.Fatalf("later panic changed accepted-session evidence: %+v factories=%d", out, calls)
		}
		be.wantRetired(t)
	})
	t.Run("no-session-is-observed-zero", func(t *testing.T) {
		conn := newEPFollowerCleanupConn(epOpShutdown)
		out := RunEPFollowerWithResult(&DistComm{rank: 1, size: 3, coord: conn}, func() *Session { panic("unexpected factory") })
		if out.Err != nil || out.Retirement != (EPFollowerRetirement{Available: true}) {
			t.Fatalf("zero ownership became unknown: %+v", out)
		}
	})
	t.Run("each-counter-overflow-is-sticky-unavailable", func(t *testing.T) {
		for field := 0; field < 3; field++ {
			r := EPFollowerRetirement{Available: true}
			counters := []*uint64{&r.AcceptedSessions, &r.CloseAttempts, &r.CloseCompletions}
			*counters[field] = ^uint64(0)
			r.increment(counters[field])
			if r.Available || *counters[field] != ^uint64(0) {
				t.Fatalf("counter %d overflow wrapped or stayed available: %+v", field, r)
			}
			r.increment(counters[(field+1)%3])
			if r.Available {
				t.Fatal("later observation restored overflow availability")
			}
		}
	})
}
