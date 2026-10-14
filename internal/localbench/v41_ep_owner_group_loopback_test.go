package localbench

import (
	"context"
	"errors"
	"net"
	"net/netip"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime integration est=600ms lane=default
func TestV41EPOwnerGroupLoopback(t *testing.T) {
	t.Run("three-rank-host-sum-and-exact-wire-counts", func(t *testing.T) {
		f := newV41EPGroupTCPFixture(t, 3)
		var joined [3]<-chan v41EPOwnerPhaseResult
		joined[0] = f.startJoin(0, f.req.Budgets.JoinNS)
		joined[2] = f.startJoin(2, f.req.Budgets.JoinNS)
		// Force arrival order 2 then 1; the reduction must still use rank order.
		if result := f.await(t, joined[2]); result.err() != nil || result.panic != nil {
			f.fail(t, "rank 2 Join: %v panic=%v", result.err(), result.panic)
		}
		joined[1] = f.startJoin(1, f.req.Budgets.JoinNS)
		for _, rank := range []int{0, 1} {
			if result := f.await(t, joined[rank]); result.err() != nil || result.panic != nil {
				f.fail(t, "rank %d Join: %v panic=%v", rank, result.err(), result.panic)
			}
			if f.groups[rank].Rank != rank || f.groups[rank].Size != 3 {
				f.fail(t, "actual group topology differs at rank %d", rank)
			}
		}
		if f.groups[2].Rank != 2 || f.groups[2].Size != 3 {
			f.fail(t, "actual group topology differs at rank 2")
		}
		parts := [3][]float32{{1e20, 1}, {-1e20, 2}, {3, 4}}
		var reduced [3][]float32
		var forward [3]<-chan v41EPOwnerPhaseResult
		for rank := range forward {
			forward[rank] = f.startPhase(rank, f.req.Budgets.ForwardNS, func(context.Context) error {
				group, ok := f.groups[rank].Value.(*model.DistComm)
				if !ok {
					return errors.New("actual DistComm unavailable")
				}
				var err error
				reduced[rank], err = group.AllReduceSum(parts[rank])
				return err
			})
		}
		for rank := range forward {
			if result := f.await(t, forward[rank]); result.err() != nil || result.panic != nil || !slices.Equal(reduced[rank], []float32{3, 7}) {
				f.fail(t, "rank %d host sum=%v err=%v panic=%v", rank, reduced[rank], result.err(), result.panic)
			}
		}
		f.retire(t)
		// Independent test encoding establishes the actual preamble size. The
		// collective request/response is length4 + tag1 + count4 + two float32s8.
		preamble, frame := uint64(len(v41EPTestPreamble(t, f.req, 0))), uint64(17)
		for rank, group := range f.groups {
			want := V41EPOwnerSocketCounters{ReadBytes: preamble + frame, WrittenBytes: preamble + 4 + frame}
			if rank == 0 {
				want = V41EPOwnerSocketCounters{ReadBytes: 2 * (preamble + 4 + frame), WrittenBytes: 2 * (preamble + frame)}
			}
			if got, ok := group.SocketCounts(); !ok || got != want {
				t.Fatalf("rank %d raw counts=%+v available=%v want=%+v", rank, got, ok, want)
			}
		}
		f.assertRetired(t, 2)
	})
	for _, mode := range []string{"cancel-accept", "cancel-partial-preamble", "deadline-accept"} {
		t.Run(mode, func(t *testing.T) {
			f := newV41EPGroupTCPFixture(t, 1)
			budget := f.req.Budgets.JoinNS
			if mode == "deadline-accept" {
				budget = 500 * time.Millisecond
			}
			joined := f.startJoin(0, budget)
			f.awaitSignal(t, f.listener.acceptEntered)
			if mode == "cancel-partial-preamble" {
				peer, err := (&net.Dialer{}).DialContext(f.ctx, "tcp4", f.req.CoordinatorAddress)
				if peer != nil {
					f.peer = peer // Fixture owns every non-nil result, even with error.
				}
				if err != nil || peer == nil {
					f.fail(t, "partial-preamble peer dial: %v", err)
				}
				if err := peer.SetDeadline(f.total); err != nil {
					f.fail(t, "peer deadline: %v", err)
				}
				wire := v41EPTestPreamble(t, f.req, 1)
				if n, err := peer.Write(wire[:2]); err != nil || n != 2 {
					f.fail(t, "partial-preamble write n=%d err=%v", n, err)
				}
				// The observer signals only when a later real Read starts after
				// consuming both bytes. No exact TCP Read-call count is assumed.
				f.awaitSignal(t, f.listener.readAfterPartial)
			}
			select {
			case result := <-joined:
				f.fail(t, "Join returned before intended cancellation/expiry: %v", result.err())
			default:
			}
			want := context.Canceled
			if mode == "deadline-accept" {
				want = context.DeadlineExceeded
			} else {
				f.cancel()
			}
			result := f.await(t, joined)
			if !errors.Is(result.err(), want) || result.panic != nil || f.groups[0].Value != nil {
				f.fail(t, "Join failure=%v panic=%v, want %v without group publication", result.err(), result.panic, want)
			}
			f.retire(t)
			wantCounts, accepted := V41EPOwnerSocketCounters{}, 0
			if mode == "cancel-partial-preamble" {
				wantCounts.ReadBytes, wantCounts.ReadErrors, accepted = 2, 1, 1
			}
			if got, ok := f.groups[0].SocketCounts(); !ok || got != wantCounts {
				t.Fatalf("incomplete raw counts=%+v available=%v want=%+v", got, ok, wantCounts)
			}
			f.assertRetired(t, accepted)
		})
	}
}

// These fixtures test real loopback sockets and the actual phase watcher. They
// do not construct a Model/Session, invoke native compute, mint a supervisor
// handoff, or emit benchmark receipts. The root's production Listen callback is
// deliberately not exercised: retaining an ephemeral listener avoids any
// close/rebind race while the exact endpoint is placed in the request digest.
type v41EPGroupTCPFixture struct {
	ctx      context.Context
	cancel   context.CancelFunc
	total    time.Time
	req      V41EPOwnerRequest
	listener *v41EPGroupTCPListener
	peer     net.Conn
	groups   []*v41EPOwnerLifecycleGroup
	given    atomic.Bool
	users    sync.WaitGroup
	once     sync.Once
	closeErr error
}

func newV41EPGroupTCPFixture(t *testing.T, ranks int) *v41EPGroupTCPFixture {
	t.Helper()
	now := time.Now()
	f := &v41EPGroupTCPFixture{req: v41EPOwnerRequestFixture(), total: now.Add(12 * time.Second)}
	f.ctx, f.cancel = context.WithDeadline(context.Background(), f.total)
	// Installed before any I/O: pre-transfer failures/panics remain fixture-owned.
	t.Cleanup(func() {
		f.close()
		if f.closeErr != nil {
			t.Errorf("loopback fixture retirement: %v", f.closeErr)
		}
	})
	f.req.Budgets = V41EPOwnerBudgets{JoinNS: 4 * time.Second, LoadNS: time.Second, ForwardNS: 4 * time.Second, CleanupNS: time.Second, TotalNS: 12 * time.Second}
	f.req.Supervisor.TerminateNS = f.req.Budgets.TotalNS
	raw, err := (&net.ListenConfig{}).Listen(f.ctx, "tcp4", "127.0.0.1:0")
	if raw != nil {
		f.listener = &v41EPGroupTCPListener{Listener: raw, acceptEntered: make(chan struct{}), readAfterPartial: make(chan struct{})}
	}
	if err != nil || raw == nil {
		f.fail(t, "held loopback listen: %v", err)
	}
	address, err := netip.ParseAddrPort(raw.Addr().String())
	if err != nil || !address.Addr().Is4() || !address.Addr().IsLoopback() || address.Port() == 0 {
		f.fail(t, "held listener is not a concrete tcp4 loopback endpoint: %v", raw.Addr())
	}
	f.req.CoordinatorAddress = address.String()
	phase, cancel := context.WithDeadline(f.ctx, now.Add(f.req.Budgets.JoinNS))
	defer cancel()
	for rank := 0; rank < ranks; rank++ {
		var group *v41EPOwnerLifecycleGroup
		if rank == 0 {
			group, err = newV41EPOwnerGroupWithIO(time.Now(), phase, f.req, rank, f.total, v41EPOwnerGroupIO{
				Listen: func(_ context.Context, network, endpoint string) (net.Listener, error) {
					if network != "tcp4" || endpoint != address.String() || f.given.Swap(true) {
						return nil, errors.New("held listener transfer disagrees or repeats")
					}
					return f.listener, nil
				},
			})
		} else {
			group, err = newV41EPOwnerGroup(phase, f.req, rank, f.total)
		}
		if group != nil {
			f.groups = append(f.groups, group)
		}
		if err != nil || group == nil {
			f.fail(t, "rank %d group constructor: %v", rank, err)
		}
	}
	return f
}

func (f *v41EPGroupTCPFixture) startJoin(rank int, budget time.Duration) <-chan v41EPOwnerPhaseResult {
	return f.startPhase(rank, budget, f.groups[rank].Join)
}

func (f *v41EPGroupTCPFixture) startPhase(rank int, budget time.Duration, work func(context.Context) error) <-chan v41EPOwnerPhaseResult {
	done := make(chan v41EPOwnerPhaseResult, 1)
	f.users.Add(1)
	go func() {
		defer f.users.Done()
		done <- v41EPOwnerRunPhase(f.ctx, f.total, budget, time.Now, false, func(ctx context.Context, watch func(func(error) error) *v41EPOwnerPhaseWatcher) error {
			group := f.groups[rank]
			watch(group.StopIO) // Armed before any setter or synchronous Join/I/O.
			if err := v41EPOwnerSetPhase(group, ctx); err != nil {
				return err
			}
			return work(ctx)
		})
	}()
	return done
}

func (f *v41EPGroupTCPFixture) await(t *testing.T, done <-chan v41EPOwnerPhaseResult) v41EPOwnerPhaseResult {
	t.Helper()
	timer := time.NewTimer(5 * time.Second)
	defer timer.Stop()
	select {
	case result := <-done:
		return result
	case <-timer.C:
		f.fail(t, "phase did not return within its bounded wait")
		return v41EPOwnerPhaseResult{} // Fatal exits only after joined retirement.
	}
}

func (f *v41EPGroupTCPFixture) awaitSignal(t *testing.T, signal <-chan struct{}) {
	t.Helper()
	timer := time.NewTimer(5 * time.Second)
	defer timer.Stop()
	select {
	case <-signal:
	case <-timer.C:
		f.fail(t, "intended blocking operation was not observed")
	}
}

func (f *v41EPGroupTCPFixture) fail(t *testing.T, format string, args ...any) {
	t.Helper()
	f.close() // Cancel/close/join precede Fatal, including every failure-only timeout.
	t.Fatalf(format+"; retirement=%v", append(args, f.closeErr)...)
}

func (f *v41EPGroupTCPFixture) close() {
	f.once.Do(func() {
		f.cancel()
		for _, group := range f.groups {
			f.closeErr = errors.Join(f.closeErr, group.StopIO(context.Canceled))
		}
		if f.peer != nil {
			f.closeErr = errors.Join(f.closeErr, f.peer.Close())
		}
		// Stop never waits for Join. Late handles retire upon registration. All
		// phase calls include watcher join before this explicit operation join.
		f.users.Wait()
		if f.listener != nil && !f.given.Load() {
			f.closeErr = errors.Join(f.closeErr, f.listener.Close())
		}
		for _, group := range f.groups {
			f.closeErr = errors.Join(f.closeErr, group.Close())
		}
	})
}

func (f *v41EPGroupTCPFixture) retire(t *testing.T) {
	t.Helper()
	f.close()
	if f.closeErr != nil {
		t.Fatalf("joined loopback retirement: %v", f.closeErr)
	}
}

func (f *v41EPGroupTCPFixture) assertRetired(t *testing.T, accepted int) {
	t.Helper()
	if !f.given.Load() || f.listener.closes.Load() != 1 {
		t.Fatal("held listener did not transfer and close exactly once")
	}
	f.listener.mu.Lock()
	defer f.listener.mu.Unlock()
	if len(f.listener.conns) != accepted {
		t.Fatalf("accepted %d connections, want %d", len(f.listener.conns), accepted)
	}
	for _, conn := range f.listener.conns {
		if conn.closes.Load() != 1 {
			t.Fatal("accepted socket did not close exactly once")
		}
	}
	if conn, err := f.listener.Listener.Accept(); conn != nil || !errors.Is(err, net.ErrClosed) {
		if conn != nil {
			conn.Close()
		}
		t.Fatalf("owned listener remains live: %v", err)
	}
}

type v41EPGroupTCPListener struct {
	net.Listener
	acceptEntered, readAfterPartial chan struct{}
	acceptOnce, readOnce            sync.Once
	closes                          atomic.Int32
	mu                              sync.Mutex
	conns                           []*v41EPGroupTCPConn
}

func (l *v41EPGroupTCPListener) Accept() (net.Conn, error) {
	l.acceptOnce.Do(func() { close(l.acceptEntered) })
	raw, err := l.Listener.Accept()
	if raw == nil {
		return nil, err
	}
	conn := &v41EPGroupTCPConn{Conn: raw, listener: l}
	l.mu.Lock()
	l.conns = append(l.conns, conn)
	l.mu.Unlock()
	return conn, err
}

func (l *v41EPGroupTCPListener) Close() error {
	l.closes.Add(1)
	return l.Listener.Close()
}

type v41EPGroupTCPConn struct {
	net.Conn
	listener *v41EPGroupTCPListener
	read     uint64 // A rank performs one synchronous reader at a time.
	closes   atomic.Int32
}

func (c *v41EPGroupTCPConn) Read(p []byte) (int, error) {
	if c.read >= 2 {
		c.listener.readOnce.Do(func() { close(c.listener.readAfterPartial) })
	}
	n, err := c.Conn.Read(p)
	c.read += uint64(n)
	return n, err
}

func (c *v41EPGroupTCPConn) Close() error {
	c.closes.Add(1)
	return c.Conn.Close()
}
