package localbench

import (
	"context"
	"errors"
	"io"
	"net"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fak-test:runtime fast est=20ms lane=default
func TestV41EPOwnerTransportContract(t *testing.T) {
	now := time.Unix(100, 0)
	total, phase := now.Add(time.Hour), now.Add(time.Minute)
	newTransport := func(t *testing.T) *v41EPOwnerTransport {
		t.Helper()
		transport, err := newV41EPOwnerTransport(now, total, phase)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { transport.stop(nil) })
		return transport
	}
	admit := func(t *testing.T, transport *v41EPOwnerTransport, raw *v41EPFakeConn) *v41EPOwnerConn {
		t.Helper()
		conn, err := transport.ownConn(raw)
		if err != nil {
			t.Fatal(err)
		}
		return conn.(*v41EPOwnerConn)
	}
	t.Run("finite-constructor", func(t *testing.T) {
		for _, bounds := range [][3]time.Time{
			{{}, total, phase}, {now, {}, phase}, {now, now, phase},
			{now, now.Add(-time.Second), phase}, {now, now.Add(V41EPOwnerMaxTotalTime + 1), phase},
			{now, total, {}}, {now, total, now}, {now, total, total.Add(1)},
		} {
			if _, err := newV41EPOwnerTransport(bounds[0], bounds[1], bounds[2]); err == nil {
				t.Fatalf("accepted invalid bounds %v", bounds)
			}
		}
		newTransport(t)
	})
	t.Run("independent-deadline-caps", func(t *testing.T) {
		transport, raw := newTransport(t), &v41EPFakeConn{}
		conn := admit(t, transport, raw)
		raw.wantDeadline(t, "both", phase)
		for _, setter := range []struct {
			kind string
			set  func(time.Time) error
		}{{"both", conn.SetDeadline}, {"read", conn.SetReadDeadline}, {"write", conn.SetWriteDeadline}} {
			for _, requested := range []time.Time{total.Add(time.Hour), {}} {
				if err := setter.set(requested); err != nil {
					t.Fatal(err)
				}
				raw.wantDeadline(t, setter.kind, phase)
			}
		}
		early, later := now.Add(10*time.Second), now.Add(5*time.Minute)
		if err := conn.SetReadDeadline(early); err != nil {
			t.Fatal(err)
		}
		if err := conn.SetWriteDeadline(later); err != nil {
			t.Fatal(err)
		}
		if err := transport.setPhaseDeadline(now.Add(10 * time.Minute)); err != nil {
			t.Fatal(err)
		}
		raw.wantDeadline(t, "read", early)
		raw.wantDeadline(t, "write", later)
		tight := now.Add(2 * time.Second)
		if err := transport.setPhaseDeadline(tight); err != nil {
			t.Fatal(err)
		}
		raw.wantDeadline(t, "read", tight)
		raw.wantDeadline(t, "write", tight)
		if err := transport.setPhaseDeadline(total); err != nil {
			t.Fatal(err)
		}
		if err := conn.SetDeadline(time.Time{}); err != nil {
			t.Fatal(err)
		}
		raw.wantDeadline(t, "both", total)
		for _, bad := range []time.Time{{}, total.Add(1)} {
			if err := transport.setPhaseDeadline(bad); err == nil {
				t.Fatal("invalid phase accepted")
			}
		}
		raw.wantDeadline(t, "both", total)
	})
	t.Run("exclusive-topology-and-cleanup", func(t *testing.T) {
		transport := newTransport(t)
		primary, closeErr, closePanic := errors.New("primary"), errors.New("close"), errors.New("close panic")
		listener := &v41EPFakeListener{closeFn: func() error { return closeErr }}
		ownedListener, err := transport.ownListener(listener)
		if err != nil {
			t.Fatal(err)
		}
		first := &v41EPFakeConn{closeFn: func() error { panic(closePanic) }}
		second := &v41EPFakeConn{}
		ownedFirst, ownedSecond := admit(t, transport, first), admit(t, transport, second)
		refusedListener, refusedConn := &v41EPFakeListener{}, &v41EPFakeConn{}
		if _, err := transport.ownListener(refusedListener); err == nil || refusedListener.closes.Load() != 1 {
			t.Fatalf("listener cap did not consume ownership: %v", err)
		}
		if _, err := transport.ownConn(refusedConn); err == nil || refusedConn.closes.Load() != 1 {
			t.Fatalf("connection cap did not consume ownership: %v", err)
		}
		if _, err := transport.ownConn(nil); err == nil {
			t.Fatal("nil connection accepted")
		}
		if _, err := transport.ownListener(nil); err == nil {
			t.Fatal("nil listener accepted")
		}
		err = transport.stop(primary)
		for _, cause := range []error{primary, closeErr, closePanic} {
			if !errors.Is(err, cause) {
				t.Fatalf("cleanup lost cause %v: %v", cause, err)
			}
		}
		if repeated := transport.stop(errors.New("later")); repeated != err {
			t.Fatal("later stop replaced first result")
		}
		retired, available := transport.retirementAfterJoin()
		if !available || errors.Is(retired, primary) || !errors.Is(retired, closeErr) || !errors.Is(retired, closePanic) {
			t.Fatalf("retirement confused primary and release failures: %v %v", retired, available)
		}
		ownedListener.Close()
		ownedFirst.Close()
		ownedSecond.Close()
		if listener.closes.Load() != 1 || first.closes.Load() != 1 || second.closes.Load() != 1 {
			t.Fatal("owned raw handle not closed exactly once")
		}
		lateListener, lateConn := &v41EPFakeListener{}, &v41EPFakeConn{}
		if _, err := transport.ownListener(lateListener); !errors.Is(err, primary) || lateListener.closes.Load() != 1 {
			t.Fatalf("late listener: %v", err)
		}
		if _, err := transport.ownConn(lateConn); !errors.Is(err, primary) || lateConn.closes.Load() != 1 {
			t.Fatalf("late connection: %v", err)
		}
		if err := transport.setPhaseDeadline(phase); !errors.Is(err, primary) {
			t.Fatalf("late phase: %v", err)
		}
	})
	t.Run("failed-initial-cap-retires-ownership", func(t *testing.T) {
		transport := newTransport(t)
		failure := errors.New("initial deadline")
		raw := &v41EPFakeConn{deadlineFn: func(string, time.Time) error { return failure }}
		if conn, err := transport.ownConn(raw); conn != nil || !errors.Is(err, failure) || raw.closes.Load() != 1 {
			t.Fatalf("failed admission conn=%v err=%v closes=%d", conn, err, raw.closes.Load())
		}
	})
	t.Run("failed-phase-attempts-all-handles", func(t *testing.T) {
		transport := newTransport(t)
		readErr, writeErr := errors.New("read cap"), errors.New("write cap")
		first := &v41EPFakeConn{deadlineFn: func(kind string, _ time.Time) error {
			switch kind {
			case "read":
				return readErr
			case "write":
				return writeErr
			}
			return nil
		}}
		second := &v41EPFakeConn{}
		admit(t, transport, first)
		admit(t, transport, second)
		tight := now.Add(time.Second)
		err := transport.setPhaseDeadline(tight)
		if !errors.Is(err, readErr) || !errors.Is(err, writeErr) {
			t.Fatalf("phase errors lost: %v", err)
		}
		for _, raw := range []*v41EPFakeConn{first, second} {
			raw.wantDeadline(t, "read", tight)
			raw.wantDeadline(t, "write", tight)
			if raw.closes.Load() != 1 {
				t.Fatal("phase failure left raw handle open")
			}
		}
	})
	for _, operation := range []string{"initial", "phase"} {
		t.Run("normal-stop-preserves-"+operation+"-failure", func(t *testing.T) {
			transport, raw := newTransport(t), &v41EPFakeConn{}
			if operation == "phase" {
				admit(t, transport, raw)
			}
			barrier := newV41EPBarrier(t)
			failure := errors.New("deadline installation")
			raw.deadlineFn = func(kind string, _ time.Time) error {
				if operation == "initial" || kind == "read" {
					barrier.block()
					return failure
				}
				return nil
			}
			done := make(chan error, 1)
			go func() {
				if operation == "phase" {
					done <- transport.setPhaseDeadline(now.Add(time.Second))
					return
				}
				_, err := transport.ownConn(raw)
				done <- err
			}()
			v41EPAwait(t, barrier.entered)
			if err := transport.stop(nil); err != nil {
				t.Fatal(err)
			}
			barrier.unblock()
			if err := v41EPAwait(t, done); !errors.Is(err, failure) {
				t.Fatalf("normal stop erased operation failure: %v", err)
			}
			if err := transport.stop(errors.New("later stop")); err != nil {
				t.Fatalf("operation failure changed fixed stop result: %v", err)
			}
			if raw.closes.Load() != 1 {
				t.Fatal("failed operation repeated raw close")
			}
		})
	}
	t.Run("phase-cannot-lose-to-in-flight-setter", func(t *testing.T) {
		transport, raw := newTransport(t), &v41EPFakeConn{}
		conn := admit(t, transport, raw)
		barrier := newV41EPBarrier(t)
		raw.deadlineFn = func(kind string, _ time.Time) error {
			if kind == "both" {
				barrier.block()
			}
			return nil
		}
		setterDone, phaseDone := make(chan error, 1), make(chan error, 1)
		go func() { setterDone <- conn.SetDeadline(time.Time{}) }()
		v41EPAwait(t, barrier.entered)
		tight := now.Add(time.Second)
		go func() { phaseDone <- transport.setPhaseDeadline(tight) }()
		barrier.unblock()
		if err := v41EPAwait(t, setterDone); err != nil {
			t.Fatal(err)
		}
		if err := v41EPAwait(t, phaseDone); err != nil {
			t.Fatal(err)
		}
		raw.wantDeadline(t, "read", tight)
		raw.wantDeadline(t, "write", tight)
		if err := conn.SetDeadline(time.Time{}); err != nil {
			t.Fatal(err)
		}
		raw.wantDeadline(t, "both", tight)
	})
	for _, operation := range []string{"registration", "phase", "setter"} {
		t.Run("stop-interrupts-blocked-"+operation, func(t *testing.T) {
			transport := newTransport(t)
			barrier := newV41EPBarrier(t)
			closed := make(chan struct{})
			raw := &v41EPFakeConn{closeFn: func() error { close(closed); return nil }}
			var conn *v41EPOwnerConn
			if operation != "registration" {
				conn = admit(t, transport, raw)
			}
			raw.deadlineFn = func(kind string, _ time.Time) error {
				if operation != "phase" || kind == "read" {
					barrier.block()
				}
				return nil
			}
			finished := make(chan error, 1)
			go func() {
				switch operation {
				case "registration":
					_, err := transport.ownConn(raw)
					finished <- err
				case "phase":
					finished <- transport.setPhaseDeadline(now.Add(time.Second))
				default:
					finished <- conn.SetWriteDeadline(time.Time{})
				}
			}()
			v41EPAwait(t, barrier.entered)
			primary := errors.New("canceled")
			stopped := make(chan error, 1)
			go func() { stopped <- transport.stop(primary) }()
			v41EPAwait(t, closed)
			if err := v41EPAwait(t, stopped); !errors.Is(err, primary) {
				t.Fatal(err)
			}
			if _, available := transport.snapshotAfterJoin(); available {
				t.Fatal("accepted unsettled deadline or registration")
			}
			if _, available := transport.retirementAfterJoin(); available {
				t.Fatal("raw Close alone acquired joined retirement evidence")
			}
			cleanupCap := make(chan error, 1)
			go func() { cleanupCap <- transport.setCleanupDeadline(total) }()
			if err := v41EPAwait(t, cleanupCap); !errors.Is(err, errV41EPOwnerRetirementUnsettled) {
				t.Fatalf("stopped unsettled cleanup cap did not refuse promptly: %v", err)
			}
			barrier.unblock()
			v41EPAwait(t, finished)
			if counters, available := transport.snapshotAfterJoin(); !available || counters != (V41EPOwnerSocketCounters{}) {
				t.Fatalf("settled zero counters: %+v %v", counters, available)
			}
			if err, available := transport.retirementAfterJoin(); !available || err != nil {
				t.Fatalf("settled retirement: %v %v", err, available)
			}
			if raw.closes.Load() != 1 {
				t.Fatal("blocked handle closed more than once")
			}
		})
	}
	t.Run("stop-and-refused-registration-settle", func(t *testing.T) {
		transport := newTransport(t)
		barrier := newV41EPBarrier(t)
		listener := &v41EPFakeListener{closeFn: func() error { barrier.block(); return nil }}
		if _, err := transport.ownListener(listener); err != nil {
			t.Fatal(err)
		}
		primary := errors.New("first stop")
		first, second := make(chan error, 1), make(chan error, 1)
		go func() { first <- transport.stop(primary) }()
		v41EPAwait(t, barrier.entered)
		go func() { second <- transport.stop(errors.New("second stop")) }()
		lateBarrier := newV41EPBarrier(t)
		lateErr := errors.New("refused handle Close")
		late := &v41EPFakeConn{closeFn: func() error { lateBarrier.block(); return lateErr }}
		lateDone := make(chan error, 1)
		go func() { _, err := transport.ownConn(late); lateDone <- err }()
		v41EPAwait(t, lateBarrier.entered)
		if _, available := transport.snapshotAfterJoin(); available {
			t.Fatal("accepted unfinished stop")
		}
		barrier.unblock()
		if a, b := v41EPAwait(t, first), v41EPAwait(t, second); a != b || !errors.Is(a, primary) {
			t.Fatalf("concurrent stops differ: %v / %v", a, b)
		}
		if _, available := transport.snapshotAfterJoin(); available {
			t.Fatal("accepted unfinished refused registration")
		}
		if _, available := transport.retirementAfterJoin(); available {
			t.Fatal("refused handle still closing acquired retirement evidence")
		}
		lateBarrier.unblock()
		if err := v41EPAwait(t, lateDone); !errors.Is(err, primary) || !errors.Is(err, lateErr) {
			t.Fatal(err)
		}
		if err, available := transport.retirementAfterJoin(); !available || !errors.Is(err, lateErr) || errors.Is(err, primary) {
			t.Fatalf("late release failure lost before settled publication: %v %v", err, available)
		}
		if errors.Is(transport.stop(nil), lateErr) {
			t.Fatal("late failure rewrote the fixed first-stop result")
		}
		if listener.closes.Load() != 1 || late.closes.Load() != 1 {
			t.Fatal("registration/stop race duplicated a close")
		}
	})
	t.Run("partial-io-counts-and-join-boundary", func(t *testing.T) {
		transport := newTransport(t)
		barrier := newV41EPBarrier(t)
		writeErr := errors.New("partial write")
		raw := &v41EPFakeConn{
			readFn:  func([]byte) (int, error) { barrier.block(); return 2, io.EOF },
			writeFn: func([]byte) (int, error) { return 3, writeErr },
		}
		conn := admit(t, transport, raw)
		if _, available := transport.snapshotAfterJoin(); available {
			t.Fatal("accepted live snapshot")
		}
		readDone := make(chan error, 1)
		go func() { _, err := conn.Read(make([]byte, 4)); readDone <- err }()
		v41EPAwait(t, barrier.entered)
		if n, err := conn.Write(make([]byte, 4)); n != 3 || !errors.Is(err, writeErr) {
			t.Fatalf("partial write changed: %d %v", n, err)
		}
		transport.stop(nil)
		if _, available := transport.snapshotAfterJoin(); available {
			t.Fatal("accepted in-flight raw read")
		}
		if _, available := transport.retirementAfterJoin(); available {
			t.Fatal("in-flight raw read acquired retirement evidence")
		}
		barrier.unblock()
		if err := v41EPAwait(t, readDone); !errors.Is(err, io.EOF) {
			t.Fatal(err)
		}
		want := V41EPOwnerSocketCounters{ReadBytes: 2, WrittenBytes: 3, ReadErrors: 1, WriteErrors: 1}
		if got, available := transport.snapshotAfterJoin(); !available || got != want {
			t.Fatalf("partial counters got=%+v available=%v", got, available)
		}
		if _, err := conn.Read(nil); !errors.Is(err, net.ErrClosed) {
			t.Fatal("read admitted after close")
		}
		if _, err := conn.Write(nil); !errors.Is(err, net.ErrClosed) {
			t.Fatal("write admitted after close")
		}
		if got, available := transport.snapshotAfterJoin(); !available || got != want {
			t.Fatal("synthetic closed errors changed raw counters")
		}
	})
	t.Run("unavailable-is-not-zero", func(t *testing.T) {
		for _, mode := range []string{"byte-overflow", "error-overflow", "aggregate-overflow", "panic", "invalid-count"} {
			t.Run(mode, func(t *testing.T) {
				transport := newTransport(t)
				panicValue := &struct{ label string }{"read panic"}
				raw := &v41EPFakeConn{readFn: func([]byte) (int, error) {
					if mode == "panic" {
						panic(panicValue)
					}
					if mode == "invalid-count" {
						return 2, nil
					}
					return 1, io.EOF
				}}
				conn := admit(t, transport, raw)
				switch mode {
				case "byte-overflow", "aggregate-overflow":
					conn.counters.ReadBytes = ^uint64(0)
				case "error-overflow":
					conn.counters.ReadErrors = ^uint64(0)
				}
				if mode == "aggregate-overflow" {
					second := admit(t, transport, &v41EPFakeConn{})
					second.counters.ReadBytes = 1
				} else if mode == "panic" {
					func() {
						defer func() {
							if got := recover(); got != panicValue {
								t.Fatalf("panic identity changed: %v", got)
							}
						}()
						conn.Read(make([]byte, 1))
					}()
				} else {
					conn.Read(make([]byte, 1))
				}
				transport.stop(nil)
				if got, available := transport.snapshotAfterJoin(); available || got != (V41EPOwnerSocketCounters{}) {
					t.Fatalf("invalid work acquired exact counters: %+v %v", got, available)
				}
				if err, available := transport.retirementAfterJoin(); !available || err != nil {
					t.Fatalf("counter unavailability misclassified successful retirement: %v %v", err, available)
				}
			})
		}
		transport := newTransport(t)
		transport.stop(nil)
		if got, available := transport.snapshotAfterJoin(); !available || got != (V41EPOwnerSocketCounters{}) {
			t.Fatalf("observed zero unavailable: %+v %v", got, available)
		}
	})
	t.Run("cleanup-stop-separates-execution-cause", func(t *testing.T) {
		for _, closeFailure := range []bool{false, true} {
			transport := newTransport(t)
			var closeErr error
			if closeFailure {
				closeErr = errors.Join(errors.New("actual Close failure"), context.Canceled)
			}
			raw := &v41EPFakeConn{closeFn: func() error { return closeErr }}
			admit(t, transport, raw)
			got := transport.stopIO(context.Canceled)
			if (got != nil) != closeFailure || closeFailure && !errors.Is(got, closeErr) {
				t.Fatalf("Close cause was stripped or invented: %v", got)
			}
			if !errors.Is(transport.stop(nil), context.Canceled) {
				t.Fatal("cleanup-only result replaced the execution primary")
			}
			if err := transport.setCleanupDeadline(total); err != nil {
				t.Fatal(err)
			}
			if err := transport.setPhaseDeadline(total); !errors.Is(err, context.Canceled) {
				t.Fatalf("ordinary stopped phase was reopened: %v", err)
			}
			if err, available := transport.retirementAfterJoin(); !available || (err != nil) != closeFailure || closeFailure && !errors.Is(err, closeErr) {
				t.Fatalf("final retirement result: %v %v", err, available)
			}
		}
	})
	t.Run("bounded-refused-handle-errors", func(t *testing.T) {
		transport := newTransport(t)
		fixed := transport.stop(context.Canceled)
		first := errors.New("first refused release")
		for i := 0; i < v41EPOwnerMaxCloseErrors+8; i++ {
			raw := &v41EPFakeListener{closeFn: func() error { return first }}
			if _, err := transport.ownListener(raw); !errors.Is(err, first) || raw.closes.Load() != 1 {
				t.Fatalf("refused handle ownership: %v", err)
			}
		}
		if len(transport.closeErrors) != v41EPOwnerMaxCloseErrors || !transport.closeOverflow {
			t.Fatal("release error collector was not bounded")
		}
		if err, available := transport.retirementAfterJoin(); !available || !errors.Is(err, first) || !strings.Contains(err.Error(), "additional raw Close failures omitted") {
			t.Fatalf("bounded failure evidence: %v %v", err, available)
		}
		if transport.stop(nil) != fixed {
			t.Fatal("refused releases rewrote the fixed first-stop result")
		}
	})
	t.Run("cleanup-cap-open-and-stopped-validation", func(t *testing.T) {
		transport, raw := newTransport(t), &v41EPFakeConn{}
		admit(t, transport, raw)
		if err := transport.setCleanupDeadline(total); err != nil {
			t.Fatal(err)
		}
		raw.wantDeadline(t, "read", total)
		raw.wantDeadline(t, "write", total)
		for _, stopped := range []bool{false, true} {
			if stopped {
				transport.stop(nil)
			}
			for _, bad := range []time.Time{{}, total.Add(1)} {
				if err := transport.setCleanupDeadline(bad); err == nil {
					t.Fatal("cleanup accepted missing or extended immutable cap")
				}
			}
		}
	})
	t.Run("cleanup-waiting-on-phase-lock-is-tracked", func(t *testing.T) {
		transport, raw := newTransport(t), &v41EPFakeConn{}
		admit(t, transport, raw)
		transport.phaseMu.Lock()
		locked := true
		joined := false
		done := make(chan error, 1)
		defer func() {
			if locked {
				transport.phaseMu.Unlock()
			}
			if !joined {
				<-done
			}
		}()
		go func() { done <- transport.setCleanupDeadline(total) }()
		until := time.Now().Add(2 * time.Second)
		for {
			transport.mu.Lock()
			reserved := transport.activeWork == 1
			transport.mu.Unlock()
			if reserved {
				break
			}
			if time.Now().After(until) {
				t.Fatal("waiting cleanup call was not reserved")
			}
			runtime.Gosched()
		}
		if err := transport.stopIO(context.Canceled); err != nil {
			t.Fatal(err)
		}
		if _, available := transport.retirementAfterJoin(); available {
			t.Fatal("waiting cleanup setter was invisible to retirement")
		}
		refused := make(chan error, 1)
		go func() { refused <- transport.setCleanupDeadline(total) }()
		if err := v41EPAwait(t, refused); !errors.Is(err, errV41EPOwnerRetirementUnsettled) {
			t.Fatalf("stopped cleanup waited behind phaseMu or accepted: %v", err)
		}
		transport.phaseMu.Unlock()
		locked = false
		err := v41EPAwait(t, done)
		joined = true
		if err != nil {
			t.Fatal(err)
		}
		if err, available := transport.retirementAfterJoin(); !available || err != nil {
			t.Fatalf("queued cleanup did not recheck stopped state: %v %v", err, available)
		}
		if raw.closes.Load() != 1 {
			t.Fatal("queued cleanup repeated retirement")
		}
	})
}

type v41EPFakeConn struct {
	mu         sync.Mutex
	deadlines  map[string]time.Time
	closes     atomic.Int32
	deadlineFn func(string, time.Time) error
	closeFn    func() error
	readFn     func([]byte) (int, error)
	writeFn    func([]byte) (int, error)
}

func (c *v41EPFakeConn) Read(p []byte) (int, error) {
	if c.readFn != nil {
		return c.readFn(p)
	}
	return 0, io.EOF
}
func (c *v41EPFakeConn) Write(p []byte) (int, error) {
	if c.writeFn != nil {
		return c.writeFn(p)
	}
	return len(p), nil
}
func (c *v41EPFakeConn) Close() error {
	c.closes.Add(1)
	if c.closeFn != nil {
		return c.closeFn()
	}
	return nil
}
func (c *v41EPFakeConn) LocalAddr() net.Addr                { return v41EPFakeAddr("local") }
func (c *v41EPFakeConn) RemoteAddr() net.Addr               { return v41EPFakeAddr("remote") }
func (c *v41EPFakeConn) SetDeadline(d time.Time) error      { return c.setDeadline("both", d) }
func (c *v41EPFakeConn) SetReadDeadline(d time.Time) error  { return c.setDeadline("read", d) }
func (c *v41EPFakeConn) SetWriteDeadline(d time.Time) error { return c.setDeadline("write", d) }
func (c *v41EPFakeConn) setDeadline(kind string, deadline time.Time) error {
	c.mu.Lock()
	if c.deadlines == nil {
		c.deadlines = make(map[string]time.Time)
	}
	c.deadlines[kind] = deadline
	c.mu.Unlock()
	if c.deadlineFn != nil {
		return c.deadlineFn(kind, deadline)
	}
	return nil
}
func (c *v41EPFakeConn) wantDeadline(t *testing.T, kind string, want time.Time) {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	if got := c.deadlines[kind]; !got.Equal(want) {
		t.Fatalf("%s deadline got %v want %v", kind, got, want)
	}
}

type v41EPFakeListener struct {
	closes  atomic.Int32
	closeFn func() error
}

func (*v41EPFakeListener) Accept() (net.Conn, error) { panic("test must not accept") }
func (*v41EPFakeListener) Addr() net.Addr            { return v41EPFakeAddr("listener") }
func (l *v41EPFakeListener) Close() error {
	l.closes.Add(1)
	if l.closeFn != nil {
		return l.closeFn()
	}
	return nil
}

type v41EPFakeAddr string

func (a v41EPFakeAddr) Network() string { return "fake" }
func (a v41EPFakeAddr) String() string  { return string(a) }

type v41EPBarrier struct {
	entered, release chan struct{}
	enter, leave     sync.Once
}

func newV41EPBarrier(t *testing.T) *v41EPBarrier {
	b := &v41EPBarrier{entered: make(chan struct{}), release: make(chan struct{})}
	t.Cleanup(b.unblock)
	return b
}
func (b *v41EPBarrier) block() {
	b.enter.Do(func() { close(b.entered) })
	<-b.release
}
func (b *v41EPBarrier) unblock() { b.leave.Do(func() { close(b.release) }) }

func v41EPAwait[T any](t *testing.T, ch <-chan T) T {
	t.Helper()
	select {
	case value := <-ch:
		return value
	case <-time.After(2 * time.Second):
		t.Fatal("fake-handle operation did not settle")
		var zero T
		return zero
	}
}
