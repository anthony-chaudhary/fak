package localbench

import (
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"time"
)

const (
	v41EPOwnerMaxConnections = 2
	v41EPOwnerMaxCloseErrors = 4
)

var errV41EPOwnerRetirementUnsettled = errors.New("v41 EP owner transport: raw retirement is not settled after caller join")

// This registry owns supplied handles only. It does not dial, listen, accept,
// create a group, or start a cancellation watcher. Callers transfer exclusive
// ownership even on refusal and must never reuse a transferred raw handle.
type v41EPOwnerTransport struct {
	mu            sync.Mutex
	phaseMu       sync.Mutex
	totalDeadline time.Time
	phaseDeadline time.Time
	listener      *v41EPOwnerListener
	connections   []*v41EPOwnerConn
	stopped       bool
	primary       error
	stopErr       error
	stopDone      chan struct{}
	activeWork    int
	closeErrors   []error // Bounded independently of arbitrarily many refused handles.
	closeOverflow bool
}

func newV41EPOwnerTransport(now, totalDeadline, phaseDeadline time.Time) (*v41EPOwnerTransport, error) {
	if now.IsZero() || totalDeadline.IsZero() || !totalDeadline.After(now) || totalDeadline.Sub(now) > V41EPOwnerMaxTotalTime ||
		phaseDeadline.IsZero() || !phaseDeadline.After(now) || phaseDeadline.After(totalDeadline) {
		return nil, errors.New("v41 EP owner transport: finite positive total and phase deadlines are required")
	}
	return &v41EPOwnerTransport{totalDeadline: totalDeadline, phaseDeadline: phaseDeadline, stopDone: make(chan struct{})}, nil
}

func (t *v41EPOwnerTransport) ownListener(raw net.Listener) (net.Listener, error) {
	t.beginWork()
	defer t.endWork()
	if raw == nil {
		return nil, errors.New("v41 EP owner transport: nil listener")
	}
	listener := &v41EPOwnerListener{Listener: raw, handle: v41EPOwnerHandle{closer: raw}}
	t.mu.Lock()
	var err error
	switch {
	case t.stopped:
		err = t.closedCause()
	case t.listener != nil:
		err = errors.New("v41 EP owner transport: listener already owned")
	default:
		t.listener = listener
	}
	t.mu.Unlock()
	if err != nil {
		closeErr := listener.Close()
		t.recordCloseError(closeErr)
		return nil, errors.Join(err, closeErr)
	}
	return listener, nil
}

func (t *v41EPOwnerTransport) ownConn(raw net.Conn) (net.Conn, error) {
	t.beginWork()
	defer t.endWork()
	if raw == nil {
		return nil, errors.New("v41 EP owner transport: nil connection")
	}
	c := &v41EPOwnerConn{raw: raw, handle: v41EPOwnerHandle{closer: raw}, countersAvailable: true}
	t.mu.Lock()
	var err error
	switch {
	case t.stopped:
		err = t.closedCause()
	case len(t.connections) >= v41EPOwnerMaxConnections:
		err = errors.New("v41 EP owner transport: connection cap exceeded")
	default:
		c.totalDeadline, c.phaseDeadline = t.totalDeadline, t.phaseDeadline
		t.connections = append(t.connections, c)
	}
	t.mu.Unlock()
	if err != nil {
		closeErr := c.Close()
		t.recordCloseError(closeErr)
		return nil, errors.Join(err, closeErr)
	}
	// Registration precedes external calls, so stop can close this raw handle
	// even if installation blocks. A concurrent phase update changes c's phase
	// under its deadline mutex; this zero-clear reads that latest value.
	if err := c.SetDeadline(time.Time{}); err != nil {
		failure := fmt.Errorf("v41 EP owner transport: initial deadline: %w", err)
		return nil, errors.Join(failure, t.stop(failure))
	}
	t.mu.Lock()
	if t.stopped {
		err = t.closedCause()
	}
	t.mu.Unlock()
	if err != nil {
		return nil, err
	}
	return c, nil
}

// setPhaseDeadline is a caller-owned execution boundary. Phase transitions are
// serialized, while individual connection setters share a deadline mutex. stop
// takes neither mutex and can interrupt an external deadline call via raw Close.
func (t *v41EPOwnerTransport) setPhaseDeadline(deadline time.Time) error {
	t.beginWork()
	defer t.endWork()
	if deadline.IsZero() || deadline.After(t.totalDeadline) {
		return errors.New("v41 EP owner transport: phase exceeds immutable total deadline")
	}
	t.phaseMu.Lock()
	defer t.phaseMu.Unlock()
	t.mu.Lock()
	if t.stopped {
		err := t.closedCause()
		t.mu.Unlock()
		return err
	}
	t.phaseDeadline = deadline
	connections := append([]*v41EPOwnerConn(nil), t.connections...)
	t.mu.Unlock()
	return t.installPhaseDeadline(connections, deadline)
}

// setCleanupDeadline keeps ordinary phase refusal separate from retirement.
// Open calls reserve work before waiting on phaseMu. Stopped calls never wait on
// that mutex: only an already settled registry can need no further installation.
// The caller must have joined all users before accepting stopped retirement.
func (t *v41EPOwnerTransport) setCleanupDeadline(deadline time.Time) error {
	if deadline.IsZero() || deadline.After(t.totalDeadline) {
		return errors.New("v41 EP owner transport: cleanup phase exceeds immutable total deadline")
	}
	t.mu.Lock()
	if t.stopped {
		settled := t.retirementSettledLocked()
		t.mu.Unlock()
		if !settled {
			return errV41EPOwnerRetirementUnsettled
		}
		return nil
	}
	t.activeWork++
	t.mu.Unlock()
	reserved := true
	defer func() {
		if reserved {
			t.endWork()
		}
	}()
	t.phaseMu.Lock()
	defer t.phaseMu.Unlock()
	t.mu.Lock()
	if t.stopped {
		// This call will perform no more external work. Retire its reservation
		// under the same lock as the observation; other queued calls still count.
		t.activeWork--
		reserved = false
		settled := t.retirementSettledLocked()
		t.mu.Unlock()
		if !settled {
			return errV41EPOwnerRetirementUnsettled
		}
		return nil
	}
	t.phaseDeadline = deadline
	connections := append([]*v41EPOwnerConn(nil), t.connections...)
	t.mu.Unlock()
	return t.installPhaseDeadline(connections, deadline)
}

// The caller holds phaseMu and a tracked work reservation, never t.mu.
func (t *v41EPOwnerTransport) installPhaseDeadline(connections []*v41EPOwnerConn, deadline time.Time) error {
	var failures []error
	for _, c := range connections {
		if err := c.setPhaseDeadline(deadline); err != nil {
			failures = append(failures, err)
		}
	}
	if err := errors.Join(failures...); err != nil {
		failure := fmt.Errorf("v41 EP owner transport: phase deadlines: %w", err)
		return errors.Join(failure, t.stop(failure))
	}
	return nil
}

func (t *v41EPOwnerTransport) beginWork() {
	t.mu.Lock()
	t.activeWork++
	t.mu.Unlock()
}

func (t *v41EPOwnerTransport) endWork() {
	t.mu.Lock()
	t.activeWork--
	t.mu.Unlock()
}

// closedCause requires t.mu. A normal stop has no primary failure but still
// refuses subsequent handle registration and phase changes.
func (t *v41EPOwnerTransport) closedCause() error {
	if t.primary != nil {
		return t.primary
	}
	return net.ErrClosed
}

// stop attempts every owned close exactly once. It touches raw handles only,
// never a group connection table, Session, model, or backend. The first caller's
// primary cause is retained through all cleanup errors and concurrent stops.
func (t *v41EPOwnerTransport) stop(cause error) error {
	t.mu.Lock()
	if t.stopped {
		done := t.stopDone
		t.mu.Unlock()
		<-done
		return t.stopErr
	}
	t.stopped, t.primary = true, cause
	listener := t.listener
	connections := append([]*v41EPOwnerConn(nil), t.connections...)
	t.mu.Unlock()
	failures := []error{cause}
	if listener != nil {
		closeErr := listener.Close()
		t.recordCloseError(closeErr)
		failures = append(failures, closeErr)
	}
	for _, c := range connections {
		closeErr := c.Close()
		t.recordCloseError(closeErr)
		failures = append(failures, closeErr)
	}
	t.mu.Lock()
	t.stopErr = errors.Join(failures...)
	err := t.stopErr
	t.mu.Unlock()
	close(t.stopDone)
	return err
}

// stopIO reports actual Close failures without reclassifying the execution
// cause as a release failure. This only completes the stop operation; it does
// not join active registration, setters, I/O or caller-owned execution. A final
// retirementAfterJoin observation is mandatory before group-table teardown.
func (t *v41EPOwnerTransport) stopIO(cause error) error {
	_ = t.stop(cause)
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.closeResultLocked()
}

func (t *v41EPOwnerTransport) recordCloseError(err error) {
	if err == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if len(t.closeErrors) < v41EPOwnerMaxCloseErrors {
		t.closeErrors = append(t.closeErrors, err)
	} else {
		t.closeOverflow = true
	}
}

func (t *v41EPOwnerTransport) closeResultLocked() error {
	if t.closeOverflow {
		return errors.Join(errors.Join(t.closeErrors...), errors.New("v41 EP owner transport: additional raw Close failures omitted"))
	}
	return errors.Join(t.closeErrors...)
}

// retirementAfterJoin requires the caller to have joined every user, including
// listener Accept. It cannot prevent a caller from violating that premise by
// starting another registration afterward. Available evidence is independent of
// socket counter overflow; a non-nil result records actual release failures.
func (t *v41EPOwnerTransport) retirementAfterJoin() (error, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.retirementSettledLocked() {
		return nil, false
	}
	return t.closeResultLocked(), true
}

// Requires t.mu. Connection I/O and setters never acquire t.mu while holding
// ioMu, so this gives one settlement observation without a lock inversion.
func (t *v41EPOwnerTransport) retirementSettledLocked() bool {
	if !t.stopped || t.activeWork != 0 {
		return false
	}
	select {
	case <-t.stopDone:
	default:
		return false
	}
	for _, c := range t.connections {
		c.ioMu.Lock()
		settled := c.activeIO == 0 && c.activeSetters == 0 && c.handle.closed.Load()
		c.ioMu.Unlock()
		if !settled {
			return false
		}
	}
	return true
}

// snapshotAfterJoin must be called only after the caller's execution join. It
// additionally requires a completed stop and settled registration, phase,
// deadline, and raw I/O calls, and refuses any counter overflow. These checks do
// not establish that a backend goroutine joined. The caller must join all users.
func (t *v41EPOwnerTransport) snapshotAfterJoin() (V41EPOwnerSocketCounters, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.retirementSettledLocked() {
		return V41EPOwnerSocketCounters{}, false
	}
	var result V41EPOwnerSocketCounters
	for _, c := range t.connections {
		c.ioMu.Lock()
		observed, available := c.counters, c.countersAvailable && c.activeIO == 0 && c.activeSetters == 0 && c.handle.closed.Load()
		c.ioMu.Unlock()
		if !available || !v41EPOwnerAddCounter(&result.ReadBytes, observed.ReadBytes) ||
			!v41EPOwnerAddCounter(&result.WrittenBytes, observed.WrittenBytes) ||
			!v41EPOwnerAddCounter(&result.ReadErrors, observed.ReadErrors) ||
			!v41EPOwnerAddCounter(&result.WriteErrors, observed.WriteErrors) {
			return V41EPOwnerSocketCounters{}, false
		}
	}
	return result, true
}

type v41EPOwnerHandle struct {
	closer io.Closer
	once   sync.Once
	closed atomic.Bool
	err    error
}

func (h *v41EPOwnerHandle) Close() error {
	h.closed.Store(true)
	h.once.Do(func() {
		// A malformed supplied handle must not prevent the remaining owned
		// handles from closing. Preserve an error-valued panic's error chain.
		defer func() {
			if cause := recover(); cause != nil {
				if err, ok := cause.(error); ok {
					h.err = fmt.Errorf("v41 EP owner transport: close panic: %w", err)
				} else {
					h.err = fmt.Errorf("v41 EP owner transport: close panic: %v", cause)
				}
			}
		}()
		h.err = h.closer.Close()
	})
	return h.err
}

type v41EPOwnerListener struct {
	net.Listener
	handle v41EPOwnerHandle
}

func (l *v41EPOwnerListener) Close() error { return l.handle.Close() }

type v41EPOwnerConn struct {
	raw               net.Conn
	handle            v41EPOwnerHandle
	deadlineMu        sync.Mutex
	totalDeadline     time.Time
	phaseDeadline     time.Time
	requestedRead     time.Time
	requestedWrite    time.Time
	ioMu              sync.Mutex
	activeIO          int
	activeSetters     int
	counters          V41EPOwnerSocketCounters
	countersAvailable bool
}

func (c *v41EPOwnerConn) Close() error         { return c.handle.Close() }
func (c *v41EPOwnerConn) LocalAddr() net.Addr  { return c.raw.LocalAddr() }
func (c *v41EPOwnerConn) RemoteAddr() net.Addr { return c.raw.RemoteAddr() }

func (c *v41EPOwnerConn) capDeadline(request time.Time) time.Time {
	deadline := c.totalDeadline
	if c.phaseDeadline.Before(deadline) {
		deadline = c.phaseDeadline
	}
	if !request.IsZero() && request.Before(deadline) {
		deadline = request
	}
	return deadline
}

func (c *v41EPOwnerConn) SetDeadline(deadline time.Time) error {
	if !c.beginSetter() {
		return net.ErrClosed
	}
	defer c.endSetter()
	c.deadlineMu.Lock()
	defer c.deadlineMu.Unlock()
	if c.handle.closed.Load() {
		return net.ErrClosed
	}
	c.requestedRead, c.requestedWrite = deadline, deadline
	return c.raw.SetDeadline(c.capDeadline(deadline))
}

func (c *v41EPOwnerConn) SetReadDeadline(deadline time.Time) error {
	if !c.beginSetter() {
		return net.ErrClosed
	}
	defer c.endSetter()
	c.deadlineMu.Lock()
	defer c.deadlineMu.Unlock()
	if c.handle.closed.Load() {
		return net.ErrClosed
	}
	c.requestedRead = deadline
	return c.raw.SetReadDeadline(c.capDeadline(deadline))
}

func (c *v41EPOwnerConn) SetWriteDeadline(deadline time.Time) error {
	if !c.beginSetter() {
		return net.ErrClosed
	}
	defer c.endSetter()
	c.deadlineMu.Lock()
	defer c.deadlineMu.Unlock()
	if c.handle.closed.Load() {
		return net.ErrClosed
	}
	c.requestedWrite = deadline
	return c.raw.SetWriteDeadline(c.capDeadline(deadline))
}

func (c *v41EPOwnerConn) setPhaseDeadline(deadline time.Time) error {
	if !c.beginSetter() {
		return net.ErrClosed
	}
	defer c.endSetter()
	c.deadlineMu.Lock()
	defer c.deadlineMu.Unlock()
	if c.handle.closed.Load() {
		return net.ErrClosed
	}
	c.phaseDeadline = deadline
	readErr := c.raw.SetReadDeadline(c.capDeadline(c.requestedRead))
	writeErr := c.raw.SetWriteDeadline(c.capDeadline(c.requestedWrite))
	return errors.Join(readErr, writeErr)
}

func (c *v41EPOwnerConn) beginSetter() bool {
	c.ioMu.Lock()
	defer c.ioMu.Unlock()
	if c.handle.closed.Load() {
		return false
	}
	c.activeSetters++
	return true
}

func (c *v41EPOwnerConn) endSetter() {
	c.ioMu.Lock()
	c.activeSetters--
	c.ioMu.Unlock()
}

func (c *v41EPOwnerConn) Read(buffer []byte) (int, error)  { return c.transfer(buffer, true) }
func (c *v41EPOwnerConn) Write(buffer []byte) (int, error) { return c.transfer(buffer, false) }

func (c *v41EPOwnerConn) transfer(buffer []byte, read bool) (n int, err error) {
	c.ioMu.Lock()
	if c.handle.closed.Load() {
		c.ioMu.Unlock()
		return 0, net.ErrClosed
	}
	c.activeIO++
	c.ioMu.Unlock()
	returned := false
	defer func() {
		c.ioMu.Lock()
		defer c.ioMu.Unlock()
		c.activeIO--
		if !returned || n < 0 || n > len(buffer) {
			c.countersAvailable = false
			return
		}
		bytes, failures := &c.counters.WrittenBytes, &c.counters.WriteErrors
		if read {
			bytes, failures = &c.counters.ReadBytes, &c.counters.ReadErrors
		}
		if !v41EPOwnerAddCounter(bytes, uint64(n)) || err != nil && !v41EPOwnerAddCounter(failures, 1) {
			c.countersAvailable = false
		}
	}()
	if read {
		n, err = c.raw.Read(buffer)
	} else {
		n, err = c.raw.Write(buffer)
	}
	returned = true
	return n, err
}

func v41EPOwnerAddCounter(counter *uint64, amount uint64) bool {
	if amount > ^uint64(0)-*counter {
		return false
	}
	*counter += amount
	return true
}
