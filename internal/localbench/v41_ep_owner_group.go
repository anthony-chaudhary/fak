package localbench

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/netip"
	"slices"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// The production constructor does no network I/O. Its returned Join operation
// performs one local listen or dial only after the lifecycle arms cancellation.
// This factory is not wired into the reserved command or an artifact issuer.
func newV41EPOwnerGroup(ctx context.Context, req V41EPOwnerRequest, rank int, total time.Time) (*v41EPOwnerLifecycleGroup, error) {
	return newV41EPOwnerGroupWithIO(time.Now(), ctx, req, rank, total, v41EPOwnerGroupIO{
		Listen: (&net.ListenConfig{}).Listen,
		Dial:   (&net.Dialer{}).DialContext,
	})
}

// Factories transfer every non-nil returned handle, including alongside error.
// They own anything not returned, including resources allocated before a panic.
// Neither the factory nor the adapter retries a failed invocation.
type v41EPOwnerGroupIO struct {
	Listen func(context.Context, string, string) (net.Listener, error)
	Dial   func(context.Context, string, string) (net.Conn, error)
}

type v41EPOwnerGroupState struct {
	mu                        sync.Mutex
	attempted, active, closed bool
	closeOnce                 sync.Once
	closeErr                  error
	group                     *model.DistComm
	bundle                    *v41EPOwnerLifecycleGroup
	transport                 *v41EPOwnerTransport
	bootstrap                 *v41EPOwnerBootstrap
	network, address          string
	rank                      int
	io                        v41EPOwnerGroupIO
}

func newV41EPOwnerGroupWithIO(now time.Time, ctx context.Context, req V41EPOwnerRequest, rank int, total time.Time, supplied v41EPOwnerGroupIO) (*v41EPOwnerLifecycleGroup, error) {
	if ctx == nil {
		return nil, errors.New("v41 EP owner group: phase context is required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	req.Shards, req.Owners, req.PromptTokenIDs = slices.Clone(req.Shards), slices.Clone(req.Owners), slices.Clone(req.PromptTokenIDs)
	if _, err := V41EPOwnerRequestDigest(req); err != nil {
		return nil, err
	}
	if rank < 0 || rank > 2 || rank == 0 && supplied.Listen == nil || rank != 0 && supplied.Dial == nil {
		return nil, errors.New("v41 EP owner group: local rank and explicit I/O factory are required")
	}
	// Literal admission is deliberately stricter than request-only diagnostics:
	// no DNS, multi-address fallback, wildcard destination or address rewriting.
	endpoint, err := netip.ParseAddrPort(req.CoordinatorAddress)
	if err != nil || endpoint.String() != req.CoordinatorAddress || endpoint.Port() == 0 ||
		endpoint.Addr().IsUnspecified() || endpoint.Addr().IsMulticast() ||
		endpoint.Addr() == netip.AddrFrom4([4]byte{255, 255, 255, 255}) ||
		endpoint.Addr().Is4In6() || endpoint.Addr().Zone() != "" {
		return nil, errors.New("v41 EP owner group: exact canonical unicast literal endpoint is required")
	}
	network := "tcp6"
	if endpoint.Addr().Is4() {
		network = "tcp4"
	}
	deadline, ok := ctx.Deadline()
	if !ok {
		return nil, errors.New("v41 EP owner group: finite Join deadline is required")
	}
	if total.After(now.Add(req.Budgets.TotalNS)) || deadline.After(now.Add(req.Budgets.JoinNS)) {
		return nil, errors.New("v41 EP owner group: supplied deadlines exceed the validated request budgets")
	}
	transport, err := newV41EPOwnerTransport(now, total, deadline)
	if err != nil {
		return nil, err
	}
	bootstrap, err := newV41EPOwnerBootstrap(req, rank, transport)
	if err != nil {
		return nil, err // No external handle has been created.
	}
	state := &v41EPOwnerGroupState{transport: transport, bootstrap: bootstrap, network: network,
		address: req.CoordinatorAddress, rank: rank, io: supplied}
	state.bundle = &v41EPOwnerLifecycleGroup{Rank: -1,
		Join: state.join, SetPhaseDeadline: transport.setPhaseDeadline,
		SetCleanupDeadline: transport.setCleanupDeadline, StopIO: transport.stopIO,
		Close: state.close, SocketCounts: state.socketCounts,
	}
	return state.bundle, nil
}

func (s *v41EPOwnerGroupState) join(ctx context.Context) (err error) {
	s.mu.Lock()
	if s.attempted || s.closed {
		s.mu.Unlock()
		return errors.New("v41 EP owner group: Join is single-use and terminal")
	}
	s.attempted, s.active = true, true
	s.mu.Unlock()
	// Track Listen/Dial, Accept, admission, model Join and the entire error/panic
	// unwind. Stop takes no Join lock and can close newly registered raw handles.
	defer func() {
		primaryPanic := recover()
		cause := err
		if primaryPanic != nil {
			if e, ok := primaryPanic.(error); ok {
				cause = fmt.Errorf("v41 EP owner group: Join panic: %w", e)
			} else {
				cause = fmt.Errorf("v41 EP owner group: Join panic: %v", primaryPanic)
			}
		}
		if cause != nil {
			releaseErr, _ := v41EPOwnerInvoke(func() error { return s.transport.stopIO(cause) })
			err = errors.Join(err, releaseErr)
		}
		s.mu.Lock()
		s.active = false
		s.mu.Unlock()
		if primaryPanic != nil {
			panic(primaryPanic)
		}
	}()
	if ctx == nil {
		return errors.New("v41 EP owner group: Join context is required")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.transport.mu.Lock()
	stopped := s.transport.stopped
	phaseDeadline := s.transport.phaseDeadline
	var stoppedErr error
	if stopped {
		stoppedErr = s.transport.closedCause()
	}
	s.transport.mu.Unlock()
	if stopped {
		return stoppedErr
	}
	// A caller cannot extend constructor/lifecycle phase admission by passing
	// a later or unbounded context to this deferred operation.
	ctx, cancel := context.WithDeadline(ctx, phaseDeadline)
	defer cancel()
	if err := ctx.Err(); err != nil {
		return err
	}
	var group *model.DistComm
	if s.rank == 0 {
		raw, factoryErr := s.io.Listen(ctx, s.network, s.address)
		var listener net.Listener
		var ownErr error
		if raw != nil {
			listener, ownErr = s.transport.ownListener(raw)
		}
		if err := errors.Join(factoryErr, ownErr, ctx.Err()); err != nil {
			return err
		}
		if listener == nil {
			return errors.New("v41 EP owner group: listen returned no handle")
		}
		group, err = model.Coordinate(&v41EPOwnerAdmissionListener{Listener: listener, state: s, ctx: ctx}, 3)
	} else {
		raw, factoryErr := s.io.Dial(ctx, s.network, s.address)
		var owned net.Conn
		var ownErr error
		if raw != nil {
			owned, ownErr = s.transport.ownConn(raw)
		}
		if err := errors.Join(factoryErr, ownErr, ctx.Err()); err != nil {
			return err
		}
		if owned == nil {
			return errors.New("v41 EP owner group: dial returned no handle")
		}
		if err := s.bootstrap.exchangeWithRoot(owned.(*v41EPOwnerConn)); err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		group, err = model.Join(&v41EPOwnerStrictWriteConn{Conn: owned}, s.rank, 3)
	}
	if group != nil {
		s.mu.Lock()
		s.group = group
		s.bundle.Value, s.bundle.Rank, s.bundle.Size = group, group.Rank(), group.Size()
		s.mu.Unlock()
	}
	return errors.Join(err, ctx.Err())
}

// The lifecycle must also join forward users before calling Close; tracking Join
// here cannot establish that a separately running model executor has returned.
func (s *v41EPOwnerGroupState) close() error {
	s.mu.Lock()
	if s.active {
		s.mu.Unlock()
		return errors.New("v41 EP owner group: Join has not returned")
	}
	s.closed = true
	group := s.group
	s.mu.Unlock()
	retired, available := s.transport.retirementAfterJoin()
	if !available {
		return errV41EPOwnerRetirementUnsettled
	}
	s.closeOnce.Do(func() {
		s.closeErr = retired
		if group != nil {
			s.closeErr = errors.Join(s.closeErr, group.Close())
		}
	})
	return s.closeErr
}

func (s *v41EPOwnerGroupState) socketCounts() (V41EPOwnerSocketCounters, bool) {
	s.mu.Lock()
	active := s.active
	s.mu.Unlock()
	if active {
		return V41EPOwnerSocketCounters{}, false
	}
	// As with Close, separately running forward users remain caller-joined.
	return s.transport.snapshotAfterJoin()
}

type v41EPOwnerAdmissionListener struct {
	net.Listener
	state   *v41EPOwnerGroupState
	ctx     context.Context
	accepts int
}

func (l *v41EPOwnerAdmissionListener) Accept() (net.Conn, error) {
	if l.accepts >= v41EPOwnerMaxConnections {
		return nil, errors.New("v41 EP owner group: root admission cap exceeded")
	}
	l.accepts++
	if err := l.ctx.Err(); err != nil {
		return nil, err
	}
	raw, acceptErr := l.Listener.Accept()
	var owned net.Conn
	var ownErr error
	if raw != nil {
		owned, ownErr = l.state.transport.ownConn(raw)
	}
	if err := errors.Join(acceptErr, ownErr, l.ctx.Err()); err != nil {
		return nil, err
	}
	if owned == nil {
		return nil, errors.New("v41 EP owner group: Accept returned no handle")
	}
	replay, _, err := l.state.bootstrap.admitFollower(owned.(*v41EPOwnerConn))
	if err != nil {
		return nil, err
	}
	return &v41EPOwnerStrictWriteConn{Conn: replay}, nil
}

// model.Join and ordinary frame writers do not all validate short nil-error
// writes. Refuse them above the counted connection, without retrying any bytes.
type v41EPOwnerStrictWriteConn struct{ net.Conn }

func (c *v41EPOwnerStrictWriteConn) Write(p []byte) (int, error) {
	n, err := c.Conn.Write(p)
	if err == nil && n != len(p) {
		err = io.ErrShortWrite
	}
	return n, err
}
