package model

import (
	"encoding/binary"
	"errors"
	"io"
	"net"
	"testing"
	"time"
)

// dist_transport_test.go — pins the DistTransport control-plane vtable
// (dist_transport.go), the loopback-TCP adaptation of TensorFold's cluster
// transport contract: the payload bound (oversize refuses BEFORE send as
// ErrTooLarge), per-peer link up/down state, per-link FIFO ordering (in order
// or not at all), and the closed failure classes (peer timeout/down -> one
// retryable ErrPeerTimeout; anything else -> ErrCorrupt).

// fak-test:runtime fast est=1ms lane=default
func TestDistTransportFailedLinkDropsQueuedControls(t *testing.T) {
	transport := &distTransport{
		g:     &DistComm{size: 3, rank: 0},
		links: []DistLink{{Peer: 0}, {Peer: 1, Up: true}, {Peer: 2, Up: true}},
		inbox: []chan []byte{nil, make(chan []byte, 2), make(chan []byte, 2)},
	}
	transport.inbox[1] <- []byte("stale-1")
	transport.inbox[1] <- []byte("stale-2")
	transport.inbox[2] <- []byte("live-1")
	transport.inbox[2] <- []byte("live-2")
	transport.failLink(1, ErrPeerTimeout)
	transport.failLink(1, ErrCorrupt) // The first failure class remains authoritative.
	buf := make([]byte, 32)
	for _, want := range []string{"live-1", "live-2"} {
		peer, n, ok := transport.Recv(buf)
		if !ok || peer != 2 || string(buf[:n]) != want {
			t.Fatalf("failed link leaked or healthy FIFO changed: peer=%d n=%d ok=%v payload=%q", peer, n, ok, buf[:n])
		}
	}
	if _, _, ok := transport.Recv(buf); ok {
		t.Fatal("queued control from failed link was delivered")
	}
	links := transport.Links()
	if links[0].Peer != 1 || links[0].Up || !errors.Is(links[0].Last, ErrPeerTimeout) || !links[1].Up {
		t.Fatalf("failure classification changed: %+v", links)
	}
}

// distFormGroup forms a size-rank DistComm over a loopback TCP listener and
// returns every rank's group plus the workers' raw connections (so a test can
// inject raw frames or drop a link). The caller owns both ends.
func distFormGroup(t *testing.T, size int) ([]*DistComm, []net.Conn) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	addr := ln.Addr().String()

	type coordRes struct {
		g   *DistComm
		err error
	}
	coordCh := make(chan coordRes, 1)
	go func() {
		g, cerr := Coordinate(ln, size)
		coordCh <- coordRes{g, cerr}
	}()

	groups := make([]*DistComm, size)
	conns := make([]net.Conn, size) // conns[0] is nil; conns[r] is worker r's raw connection
	for r := 1; r < size; r++ {
		conn, derr := net.Dial("tcp", addr)
		if derr != nil {
			t.Fatalf("dial rank %d: %v", r, derr)
		}
		g, jerr := Join(conn, r, size)
		if jerr != nil {
			conn.Close()
			t.Fatalf("join rank %d: %v", r, jerr)
		}
		groups[r] = g
		conns[r] = conn
	}
	res := <-coordCh
	if res.err != nil {
		t.Fatalf("coordinate: %v", res.err)
	}
	groups[0] = res.g
	ln.Close()
	return groups, conns
}

// distPollUntil waits (up to 5 s) for cond to hold, sleeping 2 ms between polls;
// link readers report state asynchronously, so a tight poll is the deterministic
// witness on loopback.
func distPollUntil(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(2 * time.Millisecond)
	}
}

// TestDistTransportPayloadLimitLinkState pins the borrowed transport contract:
// bounded payloads, per-peer link state, per-link ordering, and the closed
// failure classes.
func TestDistTransportPayloadLimitLinkState(t *testing.T) {
	t.Run("ClassifyMapsClosedFailureClasses", func(t *testing.T) {
		if got := ClassifyDistTransportError(nil); got != nil {
			t.Fatalf("ClassifyDistTransportError(nil) = %v, want nil", got)
		}
		if got := ClassifyDistTransportError(io.EOF); !errors.Is(got, ErrPeerTimeout) {
			t.Fatalf("EOF classified as %v, want ErrPeerTimeout", got)
		}
		if got := ClassifyDistTransportError(net.ErrClosed); !errors.Is(got, ErrPeerTimeout) {
			t.Fatalf("closed connection classified as %v, want ErrPeerTimeout", got)
		}
		opErr := &net.OpError{Op: "read", Err: errors.New("connection reset by peer")}
		if got := ClassifyDistTransportError(opErr); !errors.Is(got, ErrPeerTimeout) {
			t.Fatalf("socket I/O failure classified as %v, want ErrPeerTimeout", got)
		}
		if got := ClassifyDistTransportError(errors.New("bad frame contents")); !errors.Is(got, ErrCorrupt) {
			t.Fatalf("non-socket failure classified as %v, want ErrCorrupt", got)
		}
		if got := ClassifyDistTransportError(ErrCorrupt); !errors.Is(got, ErrCorrupt) {
			t.Fatalf("ErrCorrupt classified as %v, want ErrCorrupt", got)
		}
	})

	t.Run("PayloadBoundRefusesBeforeSend", func(t *testing.T) {
		groups, _ := distFormGroup(t, 2)
		t0 := NewDistTransport(groups[0])
		defer t0.Close()
		t1 := NewDistTransport(groups[1])
		defer t1.Close()

		const expectedLimit = 1 << 20
		if t0.Limit() != expectedLimit || t1.Limit() != expectedLimit {
			t.Fatalf("Limit() = %d/%d, want %d on both ends", t0.Limit(), t1.Limit(), expectedLimit)
		}

		// Oversize refuses BEFORE any byte reaches the wire.
		oversize := make([]byte, distTransportLimit+1)
		if err := t0.Send(1, oversize); !errors.Is(err, ErrTooLarge) {
			t.Fatalf("Send(%d bytes) = %v, want ErrTooLarge", len(oversize), err)
		}
		buf := make([]byte, 64)
		if _, _, ok := t1.Recv(buf); ok {
			t.Fatalf("worker Recv after a refused oversize send delivered a message; the refusal must precede the write")
		}
		links := t0.Links()
		if len(links) != 1 || links[0].Peer != 1 || !links[0].Up {
			t.Fatalf("Links() after a refused send = %+v, want peer 1 still up", links)
		}

		// Exactly-at-limit sends and round-trips intact.
		exact := make([]byte, expectedLimit)
		for i := range exact {
			exact[i] = byte(i*31 + i/251)
		}
		if err := t0.Send(1, exact); err != nil {
			t.Fatalf("Send(%d bytes) = %v, want nil", len(exact), err)
		}
		bigBuf := make([]byte, distTransportLimit)
		distPollUntil(t, "at-limit delivery", func() bool {
			peer, n, ok := t1.Recv(bigBuf)
			if !ok {
				return false
			}
			if peer != 0 || n != len(exact) || string(bigBuf[:n]) != string(exact) {
				t.Fatalf("at-limit delivery not byte-intact: peer=%d bytes=%d", peer, n)
			}
			return true
		})

		// Self and out-of-range peers refuse as ErrLinkDown.
		if err := t0.Send(0, []byte{'x'}); !errors.Is(err, ErrLinkDown) {
			t.Fatalf("Send(self) = %v, want ErrLinkDown", err)
		}
		if err := t0.Send(5, []byte{'x'}); !errors.Is(err, ErrLinkDown) {
			t.Fatalf("Send(out of range) = %v, want ErrLinkDown", err)
		}
	})

	t.Run("PerLinkOrderingPreserved", func(t *testing.T) {
		const perLink = 40
		groups, _ := distFormGroup(t, 3)
		transports := make([]DistTransport, 3)
		for r, g := range groups {
			transports[r] = NewDistTransport(g)
			defer transports[r].Close()
		}

		// Interleave distinct payloads across both links from the coordinator.
		payload := func(link, i int) []byte {
			b := make([]byte, 8)
			binary.BigEndian.PutUint16(b[0:2], uint16(link))
			binary.BigEndian.PutUint32(b[2:6], uint32(i))
			binary.BigEndian.PutUint16(b[6:8], uint16(len(b)))
			return b
		}
		for i := 0; i < perLink; i++ {
			if err := transports[0].Send(1, payload(1, i)); err != nil {
				t.Fatalf("coordinator Send(1, %d) = %v", i, err)
			}
			if err := transports[0].Send(2, payload(2, i)); err != nil {
				t.Fatalf("coordinator Send(2, %d) = %v", i, err)
			}
		}

		// Each worker must see ITS link's payloads in exact send order.
		for peer := 1; peer <= 2; peer++ {
			var got [][]byte
			buf := make([]byte, 8)
			distPollUntil(t, "all ordered deliveries", func() bool {
				src, n, ok := transports[peer].Recv(buf)
				if !ok {
					return len(got) == perLink
				}
				if src != 0 || n != len(buf) {
					t.Fatalf("worker %d Recv = (peer %d, n %d), want (peer 0, n %d)", peer, src, n, len(buf))
				}
				got = append(got, append([]byte(nil), buf...))
				return len(got) == perLink
			})
			if len(got) != perLink {
				t.Fatalf("worker %d received %d messages, want %d", peer, len(got), perLink)
			}
			for i, m := range got {
				want := payload(peer, i)
				if string(m) != string(want) {
					t.Fatalf("worker %d message %d = % x, want % x (per-link order broken)", peer, i, m, want)
				}
			}
		}
	})

	t.Run("LinkStateUpDownAndFailureClasses", func(t *testing.T) {
		// A corrupt frame (an empty delivery: every protocol frame carries at
		// least its tag byte) marks the link down as ErrCorrupt.
		groups, conns := distFormGroup(t, 2)
		t0 := NewDistTransport(groups[0])
		defer t0.Close()
		t1 := NewDistTransport(groups[1])
		defer t1.Close()

		links := t0.Links()
		if len(links) != 1 || links[0].Peer != 1 || !links[0].Up || links[0].Last != nil {
			t.Fatalf("initial Links() = %+v, want peer 1 up with no last error", links)
		}
		wlinks := t1.Links()
		if len(wlinks) != 1 || wlinks[0].Peer != 0 || !wlinks[0].Up {
			t.Fatalf("worker initial Links() = %+v, want peer 0 up", wlinks)
		}

		// Inject an empty frame from the worker's raw connection.
		var hdr [4]byte // length 0: a complete but empty frame
		if _, err := conns[1].Write(hdr[:]); err != nil {
			t.Fatalf("inject corrupt frame: %v", err)
		}
		distPollUntil(t, "corrupt link down", func() bool {
			l := t0.Links()[0]
			return !l.Up && errors.Is(l.Last, ErrCorrupt)
		})
		if err := t0.Send(1, []byte{'x'}); !errors.Is(err, ErrLinkDown) {
			t.Fatalf("Send to a corrupt-down link = %v, want ErrLinkDown", err)
		}

		// A peer that goes away (clean close) marks the link down as the one
		// RETRYABLE class, ErrPeerTimeout.
		groups2, conns2 := distFormGroup(t, 2)
		u0 := NewDistTransport(groups2[0])
		defer u0.Close()
		u1 := NewDistTransport(groups2[1])
		defer u1.Close()
		if err := conns2[1].Close(); err != nil {
			t.Fatalf("drop worker link: %v", err)
		}
		distPollUntil(t, "downed link reported", func() bool {
			l := u0.Links()[0]
			return !l.Up && errors.Is(l.Last, ErrPeerTimeout)
		})
		if err := u0.Send(1, []byte{'x'}); !errors.Is(err, ErrLinkDown) {
			t.Fatalf("Send to a downed link = %v, want ErrLinkDown", err)
		}
	})
}
