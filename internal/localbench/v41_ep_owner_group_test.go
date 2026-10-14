package localbench

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"net"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=30ms lane=default
func TestV41EPOwnerGroupContract(t *testing.T) {
	newGroup := func(t *testing.T, req V41EPOwnerRequest, rank int, supplied v41EPOwnerGroupIO) (*v41EPOwnerLifecycleGroup, context.Context) {
		t.Helper()
		now := time.Now()
		ctx, cancel := context.WithDeadline(context.Background(), now.Add(req.Budgets.JoinNS))
		t.Cleanup(cancel)
		group, err := newV41EPOwnerGroupWithIO(now, ctx, req, rank, now.Add(req.Budgets.TotalNS), supplied)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { group.StopIO(nil); group.Close() })
		return group, ctx
	}
	peer := func(t *testing.T, req V41EPOwnerRequest, rank int, announce bool) (*v41EPFakeConn, *bytes.Buffer, int) {
		t.Helper()
		wire := v41EPTestPreamble(t, req, rank)
		if announce {
			var prefix [4]byte
			binary.LittleEndian.PutUint32(prefix[:], uint32(rank))
			wire = append(wire, prefix[:]...)
		}
		reader, written := bytes.NewReader(wire), &bytes.Buffer{}
		return &v41EPFakeConn{readFn: reader.Read, writeFn: written.Write}, written, len(wire)
	}
	t.Run("literal-endpoint-and-finite-construction", func(t *testing.T) {
		req := v41EPOwnerRequestFixture()
		now := time.Now()
		ctx, cancel := context.WithDeadline(context.Background(), now.Add(req.Budgets.JoinNS))
		defer cancel()
		calls := 0
		supplied := v41EPOwnerGroupIO{Listen: func(context.Context, string, string) (net.Listener, error) { calls++; return nil, nil }}
		for _, address := range []string{"localhost:1234", "0.0.0.0:1234", "[::]:1234", "[::ffff:127.0.0.1]:1234", "[fe80::1%eth0]:1234", "224.0.0.1:1234", "255.255.255.255:1234"} {
			bad := req
			bad.CoordinatorAddress = address
			if _, err := newV41EPOwnerGroupWithIO(now, ctx, bad, 0, now.Add(req.Budgets.TotalNS), supplied); err == nil {
				t.Fatalf("nonliteral or remapped endpoint accepted: %s", address)
			}
		}
		for _, invalid := range []context.Context{nil, context.Background()} {
			if _, err := newV41EPOwnerGroupWithIO(now, invalid, req, 0, now.Add(req.Budgets.TotalNS), supplied); err == nil {
				t.Fatal("missing phase context/deadline accepted")
			}
		}
		for _, bound := range []struct {
			phase, total time.Time
			valid        bool
		}{
			{now.Add(req.Budgets.JoinNS), now.Add(req.Budgets.TotalNS), true},
			{now.Add(req.Budgets.JoinNS), now.Add(req.Budgets.TotalNS + 1), false},
			{now.Add(req.Budgets.JoinNS + 1), now.Add(req.Budgets.TotalNS), false},
		} {
			phase, cancelPhase := context.WithDeadline(context.Background(), bound.phase)
			group, err := newV41EPOwnerGroupWithIO(now, phase, req, 0, bound.total, supplied)
			cancelPhase()
			if (err == nil) != bound.valid {
				t.Fatalf("request deadline admission: valid=%v err=%v", bound.valid, err)
			}
			if group != nil {
				group.StopIO(nil)
				group.Close()
			}
		}
		if calls != 0 {
			t.Fatal("construction performed I/O")
		}
	})
	t.Run("root-real-coordinate-distinct-ranks-and-snapshot", func(t *testing.T) {
		req := v41EPOwnerRequestFixture()
		first, firstWrites, firstRead := peer(t, req, 2, true)
		second, secondWrites, secondRead := peer(t, req, 1, true)
		listener := &v41EPGroupFakeListener{results: []v41EPGroupAcceptResult{{conn: first}, {conn: second}}}
		listens := 0
		address := req.CoordinatorAddress
		group, ctx := newGroup(t, req, 0, v41EPOwnerGroupIO{Listen: func(ctx context.Context, network, got string) (net.Listener, error) {
			listens++
			if network != "tcp4" || got != address {
				return nil, errors.New("endpoint remapped")
			}
			if _, ok := ctx.Deadline(); !ok {
				return nil, errors.New("unbounded listen")
			}
			return listener, nil
		}})
		if listens != 0 || group.Value != nil || group.Rank != -1 {
			t.Fatal("factory performed Join or claimed actual group")
		}
		// Mutation after construction must not change peer agreement or endpoint.
		req.CoordinatorAddress = "127.0.0.1:1"
		req.PromptTokenIDs[0]++
		req.Owners[0].Backend = "other"
		if err := group.Join(ctx); err != nil {
			t.Fatal(err)
		}
		actual, ok := group.Value.(*model.DistComm)
		if !ok || actual.Rank() != 0 || actual.Size() != 3 || group.Rank != 0 || group.Size != 3 || listens != 1 || listener.accepts != 2 {
			t.Fatal("actual Coordinate topology not published")
		}
		if err := group.Join(ctx); err == nil || listens != 1 {
			t.Fatal("second Join performed I/O")
		}
		if err := group.Close(); !errors.Is(err, errV41EPOwnerRetirementUnsettled) {
			t.Fatalf("open raw transport allowed teardown: %v", err)
		}
		if err := group.StopIO(nil); err != nil {
			t.Fatal(err)
		}
		if err := group.Close(); err != nil {
			t.Fatal(err)
		}
		counts, available := group.SocketCounts()
		if !available || counts.ReadBytes != uint64(firstRead+secondRead) || counts.WrittenBytes != uint64(firstWrites.Len()+secondWrites.Len()) {
			t.Fatalf("replayed Join bytes changed raw counters: %+v %v", counts, available)
		}
		if first.closes.Load() != 1 || second.closes.Load() != 1 || listener.closes.Load() != 1 {
			t.Fatal("root ownership was not closed once")
		}
	})
	for _, rank := range []int{1, 2} {
		t.Run("follower-actual-join-rank-"+string(rune('0'+rank)), func(t *testing.T) {
			req := v41EPOwnerRequestFixture()
			req.CoordinatorAddress = "[::1]:12345"
			raw, written, wantRead := peer(t, req, 0, false)
			dials := 0
			group, ctx := newGroup(t, req, rank, v41EPOwnerGroupIO{Dial: func(_ context.Context, network, address string) (net.Conn, error) {
				dials++
				if network != "tcp6" || address != req.CoordinatorAddress {
					return nil, errors.New("IPv6 endpoint changed")
				}
				return raw, nil
			}})
			if err := group.Join(ctx); err != nil {
				t.Fatal(err)
			}
			if group.Rank != rank || group.Size != 3 || dials != 1 {
				t.Fatal("follower actual topology wrong")
			}
			wire := written.Bytes()
			if len(wire) < 4 || binary.LittleEndian.Uint32(wire[len(wire)-4:]) != uint32(rank) {
				t.Fatal("model.Join did not announce exact rank once")
			}
			if err := group.Join(ctx); err == nil || dials != 1 {
				t.Fatal("follower Join repeated")
			}
			group.StopIO(nil)
			if err := group.Close(); err != nil {
				t.Fatal(err)
			}
			counts, available := group.SocketCounts()
			if !available || counts.ReadBytes != uint64(wantRead) || counts.WrittenBytes != uint64(len(wire)) || raw.closes.Load() != 1 {
				t.Fatalf("follower raw accounting changed: %+v %v", counts, available)
			}
		})
	}
	for _, mode := range []string{"listen", "dial", "accept"} {
		t.Run("partial-"+mode+"-return-transfers-ownership", func(t *testing.T) {
			req := v41EPOwnerRequestFixture()
			primary, release := errors.New("factory failed"), errors.New("raw Close failed")
			raw := &v41EPFakeConn{closeFn: func() error { return release }}
			listener := &v41EPGroupFakeListener{results: []v41EPGroupAcceptResult{{conn: raw, err: primary}}}
			rank := 0
			supplied := v41EPOwnerGroupIO{Listen: func(context.Context, string, string) (net.Listener, error) {
				if mode == "listen" {
					return listener, primary
				}
				return listener, nil
			}}
			if mode == "listen" {
				listener.closeErr = release
			}
			if mode == "dial" {
				rank = 1
				supplied.Dial = func(context.Context, string, string) (net.Conn, error) { return raw, primary }
			}
			group, ctx := newGroup(t, req, rank, supplied)
			err := group.Join(ctx)
			if !errors.Is(err, primary) || !errors.Is(err, release) || group.Value != nil {
				t.Fatalf("partial factory failure lost: %v", err)
			}
			if mode != "listen" && raw.closes.Load() != 1 || mode != "dial" && listener.closes.Load() != 1 {
				t.Fatal("partial handle leaked")
			}
			if err := group.Close(); !errors.Is(err, release) || errors.Is(err, primary) {
				t.Fatalf("retirement confused primary with release failure: %v", err)
			}
		})
	}
	t.Run("failed-second-root-admission-retires-all", func(t *testing.T) {
		req := v41EPOwnerRequestFixture()
		first, _, _ := peer(t, req, 1, true)
		second, _, _ := peer(t, req, 1, true)
		listener := &v41EPGroupFakeListener{results: []v41EPGroupAcceptResult{{conn: first}, {conn: second}}}
		group, ctx := newGroup(t, req, 0, v41EPOwnerGroupIO{Listen: func(context.Context, string, string) (net.Listener, error) { return listener, nil }})
		if err := group.Join(ctx); err == nil {
			t.Fatal("duplicate second rank admitted")
		}
		if err := group.Close(); err != nil {
			t.Fatal(err)
		}
		if listener.accepts != 2 || listener.closes.Load() != 1 || first.closes.Load() != 1 || second.closes.Load() != 1 {
			t.Fatal("failed second admission leaked or repeated ownership")
		}
	})
	t.Run("short-join-announcement-is-terminal", func(t *testing.T) {
		req := v41EPOwnerRequestFixture()
		raw, written, _ := peer(t, req, 0, false)
		writes := 0
		raw.writeFn = func(p []byte) (int, error) {
			writes++
			if len(p) == 4 {
				written.Write(p[:2])
				return 2, nil
			}
			return written.Write(p)
		}
		group, ctx := newGroup(t, req, 1, v41EPOwnerGroupIO{Dial: func(context.Context, string, string) (net.Conn, error) { return raw, nil }})
		if err := group.Join(ctx); !errors.Is(err, io.ErrShortWrite) || writes != 2 || group.Value != nil {
			t.Fatalf("short Join write accepted/retried: %v writes=%d", err, writes)
		}
		if err := group.Close(); err != nil {
			t.Fatal(err)
		}
		counts, available := group.SocketCounts()
		if !available || counts.WrittenBytes != uint64(written.Len()) || raw.closes.Load() != 1 {
			t.Fatal("partial announcement accounting changed")
		}
	})
	// The lifecycle owns the cancellation watcher. These leaf controls invoke
	// its StopIO boundary explicitly; the group does not start its own watcher.
	for _, mode := range []string{"accept", "handshake", "join-write", "dial"} {
		t.Run("cancel-blocked-"+mode+"-joins-before-close", func(t *testing.T) {
			req := v41EPOwnerRequestFixture()
			raw, _, _ := peer(t, req, 0, false)
			barrier := newV41EPBarrier(t)
			listener := &v41EPGroupFakeListener{}
			rank := 1
			supplied := v41EPOwnerGroupIO{Dial: func(context.Context, string, string) (net.Conn, error) { return raw, nil }}
			switch mode {
			case "accept":
				rank = 0
				listener.acceptFn = func() (net.Conn, error) { barrier.block(); return nil, context.Canceled }
				supplied.Listen = func(context.Context, string, string) (net.Listener, error) { return listener, nil }
			case "handshake":
				raw.readFn = func([]byte) (int, error) { barrier.block(); return 0, net.ErrClosed }
			case "join-write":
				raw.writeFn = func(p []byte) (int, error) {
					if len(p) == 4 {
						barrier.block()
						return 0, net.ErrClosed
					}
					return len(p), nil
				}
			case "dial":
				supplied.Dial = func(ctx context.Context, _, _ string) (net.Conn, error) { barrier.block(); return raw, ctx.Err() }
			}
			group, phase := newGroup(t, req, rank, supplied)
			ctx, cancel := context.WithCancel(phase)
			done := make(chan error, 1)
			joined := false
			defer func() {
				cancel()
				group.StopIO(context.Canceled)
				barrier.unblock()
				if !joined {
					<-done
				}
			}()
			go func() { done <- group.Join(ctx) }()
			v41EPAwait(t, barrier.entered)
			cancel()
			if err := group.StopIO(context.Canceled); err != nil {
				t.Fatal(err)
			}
			if err := group.Close(); err == nil || !strings.Contains(err.Error(), "Join has not returned") {
				t.Fatalf("group teardown raced Join: %v", err)
			}
			if _, available := group.SocketCounts(); available {
				t.Fatal("active Join acquired final socket evidence")
			}
			barrier.unblock()
			err := v41EPAwait(t, done)
			joined = true
			if err == nil {
				t.Fatal("canceled Join succeeded")
			}
			if err := group.Close(); err != nil {
				t.Fatal(err)
			}
			if mode == "accept" {
				if listener.closes.Load() != 1 {
					t.Fatal("blocked listener not retired")
				}
			} else if raw.closes.Load() != 1 {
				t.Fatal("blocked/late raw connection not retired")
			}
		})
	}
	t.Run("join-panic-preserves-identity-and-joins-unwind", func(t *testing.T) {
		req := v41EPOwnerRequestFixture()
		value := &struct{ label string }{"Accept panic"}
		listener := &v41EPGroupFakeListener{acceptFn: func() (net.Conn, error) { panic(value) }}
		group, ctx := newGroup(t, req, 0, v41EPOwnerGroupIO{Listen: func(context.Context, string, string) (net.Listener, error) { return listener, nil }})
		func() {
			defer func() {
				if got := recover(); got != value {
					t.Fatalf("Join panic identity changed: %v", got)
				}
			}()
			group.Join(ctx)
		}()
		if err := group.Close(); err != nil || listener.closes.Load() != 1 {
			t.Fatalf("panic unwind not joined/retired: %v", err)
		}
	})
}

type v41EPGroupAcceptResult struct {
	conn net.Conn
	err  error
}
type v41EPGroupFakeListener struct {
	results  []v41EPGroupAcceptResult
	acceptFn func() (net.Conn, error)
	accepts  int
	closes   atomic.Int32
	closeErr error
}

func (l *v41EPGroupFakeListener) Accept() (net.Conn, error) {
	l.accepts++
	if l.acceptFn != nil {
		return l.acceptFn()
	}
	if l.accepts > len(l.results) {
		return nil, errors.New("fake Accept cap exhausted")
	}
	r := l.results[l.accepts-1]
	return r.conn, r.err
}
func (l *v41EPGroupFakeListener) Close() error { l.closes.Add(1); return l.closeErr }
func (*v41EPGroupFakeListener) Addr() net.Addr { return v41EPFakeAddr("group-listener") }
