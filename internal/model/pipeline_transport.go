package model

import (
	"encoding/binary"
	"fmt"
	"io"
	"net"
)

// pipeline_transport.go — a REAL network StageTransport.
//
// pipeline.go defines the StageTransport seam and LocalTransport (the in-process
// MarshalHidden->UnmarshalHidden stand-in). LocalTransport alone leaves "the
// NCCL/RPC wire is a swap underneath the contract" an aspiration — there is no
// implementation that actually moves the bytes between processes. TCPTransport is
// that implementation: it marshals the hidden state, ships the frame over a TCP
// connection to a peer, reads the reply frame, and unmarshals it. On loopback this
// is a genuine cross-process send that is provably byte-identical to LocalTransport
// (TestTCPTransportMatchesLocal), so the pipeline is demonstrably transport-agnostic
// on hardware that exists — a real fleet swaps in NCCL/gRPC the same way.
//
// TCPTransport owns only the move-bytes-across-the-wire half of a boundary. The peer
// on the other end runs an echo (a forwarding worker would instead run its band and
// reply with its own output frame); the seam abstracts exactly the transport, so the
// echo peer is sufficient to prove the wire is interchangeable with the in-process
// path.

// TCPTransport implements StageTransport over a TCP connection. The caller owns the
// connection lifecycle (dial/accept/close); this frames each Send over it.
type TCPTransport struct {
	conn net.Conn
}

// NewTCPTransport wraps an established connection as a StageTransport.
func NewTCPTransport(conn net.Conn) *TCPTransport {
	return &TCPTransport{conn: conn}
}

// Send marshals the hidden state to a frame, writes it to the peer, reads the reply
// frame, and unmarshals it back to (hidden, nextLo) — the StageTransport contract
// realized over a real socket. The on-wire bytes are exactly MarshalHidden's output,
// so a correct echo/forward peer returns a bit-identical hidden state.
func (t *TCPTransport) Send(hidden [][]float32, nextLo, dstStage int) ([][]float32, int, error) {
	if t.conn == nil {
		return nil, 0, fmt.Errorf("model: TCPTransport stage %d has no connection", dstStage)
	}
	payload, err := MarshalHidden(hidden, nextLo)
	if err != nil {
		return nil, 0, fmt.Errorf("model: TCPTransport marshal into stage %d: %w", dstStage, err)
	}
	if err := writeFrame(t.conn, payload); err != nil {
		return nil, 0, fmt.Errorf("model: TCPTransport send into stage %d: %w", dstStage, err)
	}
	reply, err := readFrame(t.conn)
	if err != nil {
		return nil, 0, fmt.Errorf("model: TCPTransport recv into stage %d: %w", dstStage, err)
	}
	recv, gotLo, err := UnmarshalHidden(reply)
	if err != nil {
		return nil, 0, fmt.Errorf("model: TCPTransport unmarshal into stage %d: %w", dstStage, err)
	}
	return recv, gotLo, nil
}

// writeFrame writes a 4-byte little-endian length prefix then the payload, so the
// reader knows the exact frame size on a stream socket (TCP has no message
// boundaries). The hidden-state frame already carries its own seq/hidden header;
// this length prefix only delimits frames on the wire.
func writeFrame(w io.Writer, payload []byte) error {
	var hdr [4]byte
	binary.LittleEndian.PutUint32(hdr[:], uint32(len(payload)))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	if len(payload) > 0 {
		if _, err := w.Write(payload); err != nil {
			return err
		}
	}
	return nil
}

// readFrame reads a length-prefixed frame written by writeFrame, using io.ReadFull
// so a short read on the stream does not truncate the frame.
func readFrame(r io.Reader) ([]byte, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, err
	}
	n := binary.LittleEndian.Uint32(hdr[:])
	buf := make([]byte, n)
	if n > 0 {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, err
		}
	}
	return buf, nil
}

// EchoFrames serves as the peer end for TCPTransport in tests and as a reference
// forwarding loop: it reads each frame and writes it straight back, until the
// connection closes (io.EOF). A real forwarding worker would run its band between
// read and write; the echo is the identity peer that proves the transport itself
// preserves the payload bit-for-bit.
func EchoFrames(conn net.Conn) error {
	for {
		frame, err := readFrame(conn)
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		if err := writeFrame(conn, frame); err != nil {
			return err
		}
	}
}
