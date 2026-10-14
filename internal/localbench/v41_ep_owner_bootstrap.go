package localbench

import (
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
)

const (
	v41EPOwnerPeerMagic      = "FK41EP01"
	v41EPOwnerPeerFixedBytes = 108
	v41EPOwnerPeerMaxPayload = v41EPOwnerPeerFixedBytes + V41EPOwnerMaxRunIDBytes
	v41EPOwnerPeerMaxFrame   = 4 + v41EPOwnerPeerMaxPayload
)

type v41EPOwnerPeerIdentity struct {
	runID                  string
	request, source, shard [32]byte
}

// Bootstrap establishes agreement on a caller-trusted request, not peer
// authentication or source/binary attestation. It uses supplied owned handles
// only. The runner must create their registry from the request's time budgets,
// join all bootstrap calls before final observation, and stop it on failure.
type v41EPOwnerBootstrap struct {
	mu        sync.Mutex
	transport *v41EPOwnerTransport
	identity  v41EPOwnerPeerIdentity
	rank      byte
	used      map[*v41EPOwnerConn]bool
	seen      [3]bool
	failed    error
}

func newV41EPOwnerBootstrap(req V41EPOwnerRequest, rank int, transport *v41EPOwnerTransport) (*v41EPOwnerBootstrap, error) {
	digest, err := V41EPOwnerRequestDigest(req)
	if err != nil {
		return nil, err
	}
	if rank < 0 || rank >= 3 || transport == nil {
		return nil, errors.New("v41 EP owner bootstrap: rank 0, 1, or 2 and an owned transport are required")
	}
	transport.mu.Lock()
	defer transport.mu.Unlock()
	if transport.stopDone == nil || transport.totalDeadline.IsZero() || transport.phaseDeadline.IsZero() || transport.phaseDeadline.After(transport.totalDeadline) {
		return nil, errors.New("v41 EP owner bootstrap: initialized bounded registry is required")
	}
	if transport.stopped {
		return nil, transport.closedCause()
	}
	identity := v41EPOwnerPeerIdentity{runID: req.RunID}
	// Request validation has already checked all three exact SHA-256 strings.
	for _, item := range []struct {
		source string
		target *[32]byte
	}{{digest, &identity.request}, {req.SourceSHA256, &identity.source}, {req.ExpectedShardSetSHA256, &identity.shard}} {
		raw, err := hex.DecodeString(item.source)
		if err != nil || len(raw) != len(item.target) {
			return nil, errors.New("v41 EP owner bootstrap: invalid identity digest")
		}
		copy(item.target[:], raw)
	}
	return &v41EPOwnerBootstrap{transport: transport, identity: identity, rank: byte(rank), used: make(map[*v41EPOwnerConn]bool)}, nil
}

// admitFollower consumes one peer preamble and the existing four-byte Join rank
// announcement. Its returned connection replays that prefix above raw counters;
// it preserves all following bytes for the caller's later Coordinate call.
func (b *v41EPOwnerBootstrap) admitFollower(conn *v41EPOwnerConn) (net.Conn, int, error) {
	return b.exchange(conn, true)
}

// exchangeWithRoot leaves the existing Join announcement to the caller. Each
// follower bootstrap is single-use, even if another connection is supplied.
func (b *v41EPOwnerBootstrap) exchangeWithRoot(conn *v41EPOwnerConn) error {
	_, _, err := b.exchange(conn, false)
	return err
}

func (b *v41EPOwnerBootstrap) exchange(conn *v41EPOwnerConn, root bool) (replay net.Conn, rank int, err error) {
	b.transport.beginWork()
	defer b.transport.endWork()
	b.mu.Lock()
	defer b.mu.Unlock()
	owned, admissionErr := b.member(conn)
	defer func() {
		panicked := recover()
		if panicked != nil {
			if cause, ok := panicked.(error); ok {
				err = fmt.Errorf("v41 EP owner bootstrap panic: %w", cause)
			} else {
				err = fmt.Errorf("v41 EP owner bootstrap panic: %v", panicked)
			}
		}
		if err != nil {
			if owned {
				err = errors.Join(err, conn.Close())
			}
			if b.failed == nil {
				b.failed = err
			}
		}
		if panicked != nil {
			panic(panicked)
		}
	}()
	if b.failed != nil {
		return nil, 0, errors.Join(b.failed, admissionErr)
	}
	if admissionErr != nil {
		return nil, 0, admissionErr
	}
	if b.used[conn] || !root && len(b.used) != 0 {
		return nil, 0, errors.New("v41 EP owner bootstrap: connection or follower exchange already used")
	}
	b.used[conn] = true
	if root != (b.rank == 0) {
		return nil, 0, errors.New("v41 EP owner bootstrap: wrong local rank for role")
	}
	if !root {
		if err := b.writePeer(conn); err != nil {
			return nil, 0, err
		}
		peer, err := b.readPeer(conn)
		if err != nil {
			return nil, 0, err
		}
		if peer != 0 {
			return nil, 0, errors.New("v41 EP owner bootstrap: follower requires rank zero")
		}
		if err := b.finish(conn, peer); err != nil {
			return nil, 0, err
		}
		return nil, 0, nil
	}
	peer, err := b.readPeer(conn)
	if err != nil {
		return nil, 0, err
	}
	if peer < 1 || peer > 2 || b.seen[peer] {
		return nil, 0, errors.New("v41 EP owner bootstrap: root requires distinct ranks one and two")
	}
	if err := b.writePeer(conn); err != nil {
		return nil, 0, err
	}
	var join [4]byte
	if err := v41EPOwnerReadExact(conn, join[:]); err != nil {
		return nil, 0, fmt.Errorf("v41 EP owner bootstrap: read Join rank: %w", err)
	}
	if binary.LittleEndian.Uint32(join[:]) != uint32(peer) {
		return nil, 0, errors.New("v41 EP owner bootstrap: Join rank disagrees with preamble")
	}
	if err := b.finish(conn, peer); err != nil {
		return nil, 0, err
	}
	return &v41EPOwnerJoinReplay{v41EPOwnerConn: conn, prefix: join}, int(peer), nil
}

// member reports ownership separately from liveness. A refused foreign handle
// stays with its actual registry; a refused member is retired exactly once.
func (b *v41EPOwnerBootstrap) member(conn *v41EPOwnerConn) (bool, error) {
	b.transport.mu.Lock()
	defer b.transport.mu.Unlock()
	return b.memberLocked(conn)
}

// finish makes successful admission linearize under the same mutex as stop.
func (b *v41EPOwnerBootstrap) finish(conn *v41EPOwnerConn, peer byte) error {
	b.transport.mu.Lock()
	defer b.transport.mu.Unlock()
	if _, err := b.memberLocked(conn); err != nil {
		return err
	}
	if b.rank == 0 {
		b.seen[peer] = true
	}
	return nil
}

func (b *v41EPOwnerBootstrap) memberLocked(conn *v41EPOwnerConn) (bool, error) {
	owned := false
	for _, candidate := range b.transport.connections {
		if candidate == conn {
			owned = true
			break
		}
	}
	if conn == nil || !owned {
		return false, errors.New("v41 EP owner bootstrap: connection is not owned by this registry")
	}
	if b.transport.stopped {
		return true, b.transport.closedCause()
	}
	if conn.handle.closed.Load() {
		return true, net.ErrClosed
	}
	return true, nil
}

func (b *v41EPOwnerBootstrap) writePeer(conn *v41EPOwnerConn) error {
	var frame [v41EPOwnerPeerMaxFrame]byte
	payload := frame[4:]
	copy(payload[:8], v41EPOwnerPeerMagic)
	payload[8], payload[9] = b.rank, 3
	binary.BigEndian.PutUint16(payload[10:12], uint16(len(b.identity.runID)))
	copy(payload[12:44], b.identity.request[:])
	copy(payload[44:76], b.identity.source[:])
	copy(payload[76:108], b.identity.shard[:])
	copy(payload[108:], b.identity.runID)
	size := v41EPOwnerPeerFixedBytes + len(b.identity.runID)
	binary.BigEndian.PutUint32(frame[:4], uint32(size))
	n, err := conn.Write(frame[:4+size])
	if err != nil {
		return fmt.Errorf("v41 EP owner bootstrap: write preamble: %w", err)
	}
	if n != 4+size {
		return fmt.Errorf("v41 EP owner bootstrap: write preamble: %w", io.ErrShortWrite)
	}
	return nil
}

func (b *v41EPOwnerBootstrap) readPeer(conn *v41EPOwnerConn) (byte, error) {
	var header [4]byte
	if err := v41EPOwnerReadExact(conn, header[:]); err != nil {
		return 0, fmt.Errorf("v41 EP owner bootstrap: read length: %w", err)
	}
	size := binary.BigEndian.Uint32(header[:])
	if size <= v41EPOwnerPeerFixedBytes || size > v41EPOwnerPeerMaxPayload {
		return 0, errors.New("v41 EP owner bootstrap: invalid bounded preamble length")
	}
	var buffer [v41EPOwnerPeerMaxPayload]byte
	payload := buffer[:int(size)]
	if err := v41EPOwnerReadExact(conn, payload); err != nil {
		return 0, fmt.Errorf("v41 EP owner bootstrap: read preamble: %w", err)
	}
	runBytes := int(binary.BigEndian.Uint16(payload[10:12]))
	if string(payload[:8]) != v41EPOwnerPeerMagic || payload[9] != 3 || payload[8] > 2 ||
		runBytes == 0 || runBytes > V41EPOwnerMaxRunIDBytes || len(payload) != v41EPOwnerPeerFixedBytes+runBytes {
		return 0, errors.New("v41 EP owner bootstrap: invalid preamble version, world, rank, or run length")
	}
	var identity v41EPOwnerPeerIdentity
	identity.runID = string(payload[108:])
	copy(identity.request[:], payload[12:44])
	copy(identity.source[:], payload[44:76])
	copy(identity.shard[:], payload[76:108])
	if identity != b.identity {
		return 0, errors.New("v41 EP owner bootstrap: peer request identity disagrees")
	}
	return payload[8], nil
}

// A positive-length read must make progress. At most len(buffer) successful
// reads are possible; partial n plus any error remains a failed exchange.
func v41EPOwnerReadExact(conn *v41EPOwnerConn, buffer []byte) error {
	for len(buffer) > 0 {
		n, err := conn.Read(buffer)
		if n < 0 || n > len(buffer) {
			return errors.New("v41 EP owner bootstrap: invalid read count")
		}
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrNoProgress
		}
		buffer = buffer[n:]
	}
	return nil
}

type v41EPOwnerJoinReplay struct {
	*v41EPOwnerConn
	mu     sync.Mutex
	prefix [4]byte
	used   int
}

func (c *v41EPOwnerJoinReplay) Read(buffer []byte) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.handle.closed.Load() {
		return 0, net.ErrClosed
	}
	if len(buffer) == 0 {
		return 0, nil
	}
	if c.used < len(c.prefix) {
		n := copy(buffer, c.prefix[c.used:])
		c.used += n
		return n, nil
	}
	return c.v41EPOwnerConn.Read(buffer)
}
