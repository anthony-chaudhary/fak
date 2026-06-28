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
// TCPTransport owns only the move-bytes-across-the-wire half of a boundary. The
// reference peer EchoFrames runs an echo, which proves the transport itself is
// interchangeable with the in-process path. ServeBand (below) is the REAL peer: it
// runs the worker's resident band and replies with its own output frame, so a
// >=2-stage run over TCPTransport is a genuine multi-worker pipeline rather than a
// substrate exercised against an identity echo.

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
// preserves the payload bit-for-bit. ServeBand is that real forwarding worker.
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

// replyResumeLayer is the sentinel resume-layer a last stage stamps on its logits
// reply frame. A logits frame is the pipeline's terminal output, not a hidden state
// handed to a downstream band, so it has no real next-band start; -1 is distinct
// from every valid band start (which is >= 0), so a logits reply can never be
// mistaken for (or pass) a hidden-state boundary check.
const replyResumeLayer = -1

// ServeBand is the band-running worker serve loop — the peer that replaces
// EchoFrames on the other end of a TCPTransport. Where EchoFrames returns each
// inbound frame unchanged, ServeBand decodes the frame to a hidden state, runs
// ForwardBand over this worker's resident band [stage.Spec.Lo,stage.Spec.Hi), and
// replies with its OWN output frame:
//   - the last stage marshals the per-position logits it produced (final norm + LM
//     head) under the replyResumeLayer sentinel;
//   - an interior stage forwards its output hidden state to the next worker over
//     downstream and relays that worker's reply (ultimately the last stage's logits)
//     back upstream unchanged.
//
// It serves frames until the connection closes (io.EOF), so one worker handles every
// token/step of a generation over a persistent connection. The caller owns the
// connection lifecycle (accept/close), matching TCPTransport.
//
// The inbound frame's resume-layer must equal this stage's band start — the same
// boundary-integrity check handoff makes on the driver side — so a frame routed to
// the wrong worker fails closed rather than running a band against a mismatched
// hidden state. A nil downstream on an interior stage is a configuration error,
// surfaced rather than deadlocked. ForwardBand itself still rejects a band that
// begins on a GLM shared-indexer layer, so no top-k state is ever marshalled across
// a stage even if a malformed plan reaches a worker.
func ServeBand(conn net.Conn, stage PipelineStage, downstream StageTransport) error {
	if conn == nil {
		return fmt.Errorf("model: ServeBand stage [%d,%d) has no connection", stage.Spec.Lo, stage.Spec.Hi)
	}
	if stage.Model == nil {
		return fmt.Errorf("model: ServeBand stage [%d,%d) has nil model", stage.Spec.Lo, stage.Spec.Hi)
	}
	for {
		frame, err := readFrame(conn)
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return fmt.Errorf("model: ServeBand stage [%d,%d) recv: %w", stage.Spec.Lo, stage.Spec.Hi, err)
		}
		hidden, lo, err := UnmarshalHidden(frame)
		if err != nil {
			return fmt.Errorf("model: ServeBand stage [%d,%d) unmarshal: %w", stage.Spec.Lo, stage.Spec.Hi, err)
		}
		if lo != stage.Spec.Lo {
			return fmt.Errorf("model: ServeBand frame resumes at layer %d but stage band starts at %d", lo, stage.Spec.Lo)
		}
		out, logits, err := stage.Model.ForwardBand(hidden, stage.Spec.Lo, stage.Spec.Hi, stage.Spec.Last)
		if err != nil {
			return fmt.Errorf("model: ServeBand stage [%d,%d) ForwardBand: %w", stage.Spec.Lo, stage.Spec.Hi, err)
		}
		var reply []byte
		if stage.Spec.Last {
			// Terminal stage: the logits ARE the pipeline output. Marshal them with the
			// hidden codec (logits are a uniform [][]float32 exactly like a hidden state)
			// under the sentinel resume-layer and reply straight up the chain.
			reply, err = MarshalHidden(logits, replyResumeLayer)
			if err != nil {
				return fmt.Errorf("model: ServeBand stage [%d,%d) marshal logits: %w", stage.Spec.Lo, stage.Spec.Hi, err)
			}
		} else {
			if downstream == nil {
				return fmt.Errorf("model: ServeBand interior stage [%d,%d) has no downstream transport", stage.Spec.Lo, stage.Spec.Hi)
			}
			// Hand the output hidden to the next worker (resuming at this band's end) and
			// relay whatever it returns — the last stage's logits, propagated back up.
			recv, gotLo, derr := downstream.Send(out, stage.Spec.Hi, stage.Spec.Hi)
			if derr != nil {
				return fmt.Errorf("model: ServeBand stage [%d,%d) downstream: %w", stage.Spec.Lo, stage.Spec.Hi, derr)
			}
			reply, err = MarshalHidden(recv, gotLo)
			if err != nil {
				return fmt.Errorf("model: ServeBand stage [%d,%d) relay marshal: %w", stage.Spec.Lo, stage.Spec.Hi, err)
			}
		}
		if err := writeFrame(conn, reply); err != nil {
			return fmt.Errorf("model: ServeBand stage [%d,%d) reply: %w", stage.Spec.Lo, stage.Spec.Hi, err)
		}
	}
}

// RunPipelineAcrossWorkers drives a genuine multi-worker pipeline: it runs ONLY the
// FIRST stage in this process (embed + its band) and delegates every later stage to
// a remote ServeBand worker reached over downstream. It returns the last stage's
// per-position logits, relayed back up the worker chain.
//
// This is the multi-node counterpart of RunPipelineWith. RunPipelineWith runs every
// band in-process and uses the transport only as a marshalling boundary (its peer is
// an echo); here the remote bands run ONLY inside the worker serve loops, so a
// >=2-stage run is a real cross-process pipeline whose logits are bit-identical to
// the monolithic Forward (the shipped correctness contract, now carried across the
// wire rather than across a slice). At temperature 0 the argmax over these logits is
// therefore token-identical to monolithic generation.
//
// It fails closed on a misconfigured head: the first stage must be marked First, and
// a multi-stage run (First not also Last) needs a non-nil downstream transport.
func RunPipelineAcrossWorkers(ids []int, first PipelineStage, downstream StageTransport) ([][]float32, error) {
	if first.Model == nil {
		return nil, fmt.Errorf("model: RunPipelineAcrossWorkers first stage has nil model")
	}
	if !first.Spec.First {
		return nil, fmt.Errorf("model: RunPipelineAcrossWorkers first stage band [%d,%d) is not marked First", first.Spec.Lo, first.Spec.Hi)
	}
	x := first.Model.embedBand(ids)
	hidden, logits, err := first.Model.ForwardBand(x, first.Spec.Lo, first.Spec.Hi, first.Spec.Last)
	if err != nil {
		return nil, fmt.Errorf("model: RunPipelineAcrossWorkers first band [%d,%d): %w", first.Spec.Lo, first.Spec.Hi, err)
	}
	if first.Spec.Last {
		return logits, nil // degenerate single-stage pipeline: nothing crosses the wire
	}
	if downstream == nil {
		return nil, fmt.Errorf("model: RunPipelineAcrossWorkers multi-stage run needs a downstream transport")
	}
	recv, _, err := downstream.Send(hidden, first.Spec.Hi, 1)
	if err != nil {
		return nil, fmt.Errorf("model: RunPipelineAcrossWorkers handoff to stage 1: %w", err)
	}
	return recv, nil // recv is the last stage's logits, relayed up the worker chain
}
