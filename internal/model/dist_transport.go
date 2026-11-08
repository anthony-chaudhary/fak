package model

import (
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"sync"
)

// dist_transport.go — the control-plane transport vtable for a DistComm process
// group. ADAPTED from TensorFold's cluster transport contract (Apache-2.0):
// ashhart/TensorFold zig/src/cluster/transport.zig @9e8c368b0b4c2c7892a0c9f0f302d7c164016209
// plus its loopback-adjacent realization and error classification in
// zig/src/cluster/fabric_link.zig:79. The upstream idea: what a control plane
// needs from a link layer is exactly four things — a bounded send, a poll of
// the next message from any link, every link's up/down state, and the largest
// payload a link carries — with messages on one link arriving IN ORDER OR NOT
// AT ALL, and link failures mapped to closed classes (peer timeout/down is one
// RETRYABLE class; anything else is corrupt). No upstream code is copied; this
// is an idiomatic Go re-expression over fak's existing framed-connection seams
// (the same writeFrame/readFrame framing writeRequest/readRequest use).
// Loopback/stream TCP only; physical RDMA stays open.

// Closed transport error classes. ErrTooLarge and ErrLinkDown are the vtable's
// own refusals (a payload the link cannot carry; a send with no live link).
// ErrPeerTimeout and ErrCorrupt are the OBSERVED link-failure classes: a peer
// that timed out or went down is one retryable class; anything else that broke
// a frame is Corrupt.
var (
	ErrTooLarge    = errors.New("model: dist transport payload exceeds the link limit")
	ErrLinkDown    = errors.New("model: dist transport link down")
	ErrPeerTimeout = errors.New("model: dist transport peer timeout or down (retryable)")
	ErrCorrupt     = errors.New("model: dist transport corrupt frame")
)

// DistLink is one peer link's state as observed by this rank.
type DistLink struct {
	Peer int
	Up   bool
	Last error // nil while healthy; ErrPeerTimeout or ErrCorrupt once the link failed
}

// DistTransport is the control-plane view of a formed DistComm group: the four
// operations a membership/control loop needs from the link layer — bounded
// send, poll receive, per-peer link state, and the link payload limit.
// Messages on one link arrive in order or not at all: each link is a FIFO
// mailbox, and a link failure drops the rest of its queue rather than
// delivering out of order.
type DistTransport interface {
	// Send queues payload for the peer rank's link. It refuses BEFORE sending:
	// ErrLinkDown when the peer is self, out of range, or down; ErrTooLarge
	// when payload exceeds Limit().
	Send(peer int, payload []byte) error
	// Recv copies the next waiting message from ANY link into buf and reports
	// its source peer and length; ok is false when no message waits (a
	// non-blocking poll). A message larger than buf is dropped, matching the
	// upstream mailbox contract.
	Recv(buf []byte) (peer int, n int, ok bool)
	// Links reports every peer link with its up/down state and last observed
	// failure class.
	Links() []DistLink
	// Limit is the largest payload a link carries.
	Limit() int
	// Close stops the link readers by closing the group's links; it is safe to
	// call more than once, and a later DistComm.Close stays safe (closing an
	// already-closed connection just returns an error). Frames already claimed
	// off the wire remain available to the collective read paths.
	Close() error
}

// distTransportLimit is the largest control payload a link carries — the
// mailbox size of the stream-TCP adaptation (upstream sizes it from the
// registered-window layout; its control-plane default is 1 MiB).
const distTransportLimit = 1 << 20

// ClassifyDistTransportError maps a raw link-layer failure onto the closed
// observed-failure classes: a peer that timed out, reset, or went away (any
// socket-level I/O failure, EOF, or close) is the single RETRYABLE
// ErrPeerTimeout class; anything else is ErrCorrupt.
func ClassifyDistTransportError(err error) error {
	if err == nil {
		return nil
	}
	var opErr *net.OpError
	switch {
	case errors.Is(err, ErrCorrupt):
		return ErrCorrupt
	case errors.As(err, &opErr):
		return ErrPeerTimeout
	case errors.Is(err, io.EOF), errors.Is(err, net.ErrClosed), os.IsTimeout(err):
		return ErrPeerTimeout
	default:
		return ErrCorrupt
	}
}

// distTransport is the DistTransport over one formed DistComm's links: one
// reader goroutine per peer rides the group's existing framed connections.
// Control frames tagged opTransportCtrl land in that link's FIFO inbox; every
// other frame is parked on the group (enqueuePending) so the collective read
// paths drain it in arrival order — a wrapped group keeps running its
// collectives correctly.
type distTransport struct {
	g     *DistComm
	limit int

	mu     sync.Mutex
	links  []DistLink     // index = peer rank; links[rank].Up tracks liveness
	inbox  []chan []byte  // index = peer rank; nil where no link
	stop   chan struct{}  // closed once (under mu) by Close
	wg     sync.WaitGroup // one entry per live link reader
	cursor int            // round-robin start for Recv, guarded by mu
}

// NewDistTransport wraps a formed group as the control-plane transport
// vtable. The group's one-collective-at-a-time discipline still applies to
// WRITES: never mix a Send with a collective round on the same rank. Physical
// RDMA stays open; this adaptation is loopback/stream TCP only.
func NewDistTransport(g *DistComm) DistTransport {
	if g == nil {
		return nil
	}
	t := &distTransport{
		g:     g,
		limit: distTransportLimit,
		links: make([]DistLink, g.size),
		inbox: make([]chan []byte, g.size),
		stop:  make(chan struct{}),
	}
	for r := 0; r < g.size; r++ {
		if r == g.rank {
			continue
		}
		conn := g.linkConn(r)
		if conn == nil {
			t.links[r] = DistLink{Peer: r}
			continue
		}
		t.links[r] = DistLink{Peer: r, Up: true}
		t.inbox[r] = make(chan []byte, 8)
		t.wg.Add(1)
		go t.pump(r, conn)
	}
	return t
}

// pump reads framed payloads off one link until it fails or the transport
// closes. A terminal read failure marks the link down with the classified
// cause; an empty frame (every protocol frame on these links carries at least
// its tag byte) is a corrupt delivery, not a quiet peer.
func (t *distTransport) pump(peer int, conn net.Conn) {
	defer t.wg.Done()
	for {
		select {
		case <-t.stop:
			return
		default:
		}
		frame, err := readFrame(conn)
		if err != nil {
			t.failLink(peer, ClassifyDistTransportError(err))
			return
		}
		if len(frame) == 0 {
			t.failLink(peer, ErrCorrupt)
			return
		}
		if distOp(frame[0]) == opTransportCtrl {
			t.deliverControl(peer, frame[1:])
			continue
		}
		t.g.enqueuePending(frame)
	}
}

// deliverControl hands a control frame to that link's FIFO inbox, blocking on
// a full inbox (backpressure) unless the transport is closing.
func (t *distTransport) deliverControl(peer int, data []byte) {
	select {
	case t.inbox[peer] <- data:
	case <-t.stop:
	}
}

// failLink records the first observed failure on a link: down, with the
// classified cause.
func (t *distTransport) failLink(peer int, cause error) {
	t.mu.Lock()
	if t.links[peer].Up {
		t.links[peer].Up = false
		t.links[peer].Last = cause
	}
	t.mu.Unlock()
}

// Send implements DistTransport. The oversize refusal happens before any byte
// reaches the wire.
func (t *distTransport) Send(peer int, payload []byte) error {
	if peer == t.g.rank || peer < 0 || peer >= t.g.size {
		return fmt.Errorf("model: dist transport send to peer %d of %d ranks: %w", peer, t.g.size, ErrLinkDown)
	}
	t.mu.Lock()
	up := t.links[peer].Up
	t.mu.Unlock()
	if !up {
		return fmt.Errorf("model: dist transport send to peer %d: %w", peer, ErrLinkDown)
	}
	if len(payload) > t.limit {
		return fmt.Errorf("model: dist transport send to peer %d: payload %d bytes exceeds link limit %d: %w",
			peer, len(payload), t.limit, ErrTooLarge)
	}
	if err := writeRequestRaw(t.g.linkConn(peer), opTransportCtrl, payload); err != nil {
		t.failLink(peer, ClassifyDistTransportError(err))
		return fmt.Errorf("model: dist transport send to peer %d: %w", peer, ErrLinkDown)
	}
	return nil
}

// Recv implements DistTransport: a non-blocking round-robin poll across every
// link's FIFO inbox.
func (t *distTransport) Recv(buf []byte) (int, int, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	// Linearize polling against failLink: after a failure is recorded, queued
	// controls from that peer cannot escape through an otherwise healthy inbox.
	size := t.g.size
	start := t.cursor
	for i := 0; i < size; i++ {
		peer := (start + i) % size
		ch := t.inbox[peer]
		if ch == nil || !t.links[peer].Up {
			continue
		}
		select {
		case data := <-ch:
			t.cursor = peer + 1
			if len(data) > len(buf) {
				// Mailbox contract: a message that does not fit is dropped,
				// not truncated — keep polling the remaining links.
				continue
			}
			return peer, copy(buf, data), true
		default:
		}
	}
	return 0, 0, false
}

// Links implements DistTransport.
func (t *distTransport) Links() []DistLink {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make([]DistLink, 0, t.g.size-1)
	for r, l := range t.links {
		if r == t.g.rank {
			continue
		}
		out = append(out, l)
	}
	return out
}

// Limit implements DistTransport.
func (t *distTransport) Limit() int { return t.limit }

// Close implements DistTransport. Closing the links unblocks any reader
// parked in readFrame; the wait guarantees no reader outlives the call.
func (t *distTransport) Close() error {
	t.mu.Lock()
	select {
	case <-t.stop:
		t.mu.Unlock()
		return nil
	default:
		close(t.stop)
	}
	t.mu.Unlock()
	var first error
	for r := 0; r < t.g.size; r++ {
		if c := t.g.linkConn(r); c != nil {
			if err := c.Close(); err != nil && first == nil {
				first = err
			}
		}
	}
	t.wg.Wait()
	return first
}

// distTransport satisfies the DistTransport vtable.
var _ DistTransport = (*distTransport)(nil)
