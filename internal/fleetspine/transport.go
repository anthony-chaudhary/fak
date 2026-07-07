package fleetspine

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"sync"
)

// Transport is the swappable wire for heartbeats: Advertise sends one, Listen delivers every
// received one to onHeartbeat until ctx is cancelled. Keeping it an interface lets the
// UDP-multicast transport be the production path while tests use an in-memory channel fake
// (newChanTransport) with no real socket — so unit tests never touch the network.
type Transport interface {
	// Advertise marshals hb and sends it once. A transient send error is returned for the
	// caller to log-and-continue; it is never fatal to the guard.
	Advertise(ctx context.Context, hb Heartbeat) error
	// Listen blocks, decoding datagrams and calling onHeartbeat for each well-formed one,
	// until ctx is cancelled or the socket errors unrecoverably. Malformed/oversized
	// datagrams are dropped silently (never a panic, never a callback).
	Listen(ctx context.Context, onHeartbeat func(Heartbeat)) error
}

// maxDatagram bounds a single read. A heartbeat is a handful of short strings — well under 1KB
// — so 8KB is generous headroom while capping what one packet can allocate. A datagram larger
// than this is truncated by UDP and will simply fail to parse, so it is dropped like any other
// malformed packet.
const maxDatagram = 8 << 10

// udpMulticastTransport is the production LAN transport, stdlib-net only (no golang.org/x/net):
// it receives on a joined multicast group via net.ListenMulticastUDP and sends to the group
// via the same socket's WriteToUDP. Default multicast TTL 1 keeps traffic to the local subnet
// — exactly the "the LAN" scope this feature targets.
type udpMulticastTransport struct {
	group *net.UDPAddr

	mu   sync.Mutex
	conn *net.UDPConn // lazily bound on first Listen; reused for Advertise
}

// NewUDPMulticastTransport resolves group:port and returns a transport bound to it. It returns
// an error if the address does not resolve; the actual socket is opened on first use so a
// caller can construct the transport cheaply and let Listen surface a bind failure (which the
// guard treats as "degrade to disk-only", never fatal).
func NewUDPMulticastTransport(group string, port int) (Transport, error) {
	if group == "" {
		return nil, fmt.Errorf("fleetspine: empty multicast group")
	}
	addr, err := net.ResolveUDPAddr("udp4", fmt.Sprintf("%s:%d", group, port))
	if err != nil {
		return nil, fmt.Errorf("fleetspine: resolve %s:%d: %w", group, port, err)
	}
	if addr.IP == nil || !addr.IP.IsMulticast() {
		return nil, fmt.Errorf("fleetspine: %s is not a multicast address", group)
	}
	return &udpMulticastTransport{group: addr}, nil
}

// listener lazily joins the multicast group and returns the shared receive socket. The joined
// socket is also used to send (WriteToUDP to the group), so Advertise and Listen share one fd.
func (t *udpMulticastTransport) listener() (*net.UDPConn, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.conn != nil {
		return t.conn, nil
	}
	conn, err := net.ListenMulticastUDP("udp4", nil, t.group)
	if err != nil {
		return nil, err
	}
	_ = conn.SetReadBuffer(maxDatagram * 4)
	t.conn = conn
	return conn, nil
}

func (t *udpMulticastTransport) Advertise(ctx context.Context, hb Heartbeat) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	conn, err := t.listener()
	if err != nil {
		return err
	}
	buf, err := json.Marshal(hb)
	if err != nil {
		return err
	}
	_, err = conn.WriteToUDP(buf, t.group)
	return err
}

func (t *udpMulticastTransport) Listen(ctx context.Context, onHeartbeat func(Heartbeat)) error {
	conn, err := t.listener()
	if err != nil {
		return err
	}
	// Unblock the blocking ReadFromUDP when ctx is cancelled by closing the socket; the
	// pending read then returns an error and the loop exits.
	go func() {
		<-ctx.Done()
		t.mu.Lock()
		if t.conn != nil {
			_ = t.conn.Close()
			t.conn = nil
		}
		t.mu.Unlock()
	}()
	buf := make([]byte, maxDatagram)
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		n, _, err := conn.ReadFromUDP(buf)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err() // cancellation closed the socket — expected, not an error
			}
			return err
		}
		hb, ok := decodeHeartbeat(buf[:n])
		if !ok {
			continue // malformed/oversized — drop, never a callback or a panic
		}
		onHeartbeat(hb)
	}
}

// decodeHeartbeat unmarshals one datagram into a Heartbeat, rejecting anything that does not
// parse or lacks a usable id. It is the single decode gate both the UDP and fake transports
// funnel through, so the malformed-packet contract is enforced in one place.
func decodeHeartbeat(b []byte) (Heartbeat, bool) {
	var hb Heartbeat
	if err := json.Unmarshal(b, &hb); err != nil {
		return Heartbeat{}, false
	}
	if hb.ID == "" {
		return Heartbeat{}, false
	}
	return hb, true
}

// chanTransport is an in-memory Transport backed by a buffered channel, for tests: Advertise
// pushes a marshalled datagram, Listen drains it through the same decodeHeartbeat gate. It
// exercises the encode/decode path (including malformed-drop) with no socket.
type chanTransport struct {
	ch chan []byte
}

// newChanTransport builds an in-memory transport with the given buffer depth.
func newChanTransport(buffer int) *chanTransport {
	if buffer <= 0 {
		buffer = 16
	}
	return &chanTransport{ch: make(chan []byte, buffer)}
}

func (t *chanTransport) Advertise(ctx context.Context, hb Heartbeat) error {
	buf, err := json.Marshal(hb)
	if err != nil {
		return err
	}
	select {
	case t.ch <- buf:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (t *chanTransport) Listen(ctx context.Context, onHeartbeat func(Heartbeat)) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case buf := <-t.ch:
			if hb, ok := decodeHeartbeat(buf); ok {
				onHeartbeat(hb)
			}
		}
	}
}

// inject pushes a raw datagram (possibly malformed) directly onto a chanTransport, so a test
// can assert the decode gate drops bad packets.
func (t *chanTransport) inject(b []byte) {
	t.ch <- b
}
