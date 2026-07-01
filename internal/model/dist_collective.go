package model

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"net"
)

// dist_collective.go — DistComm, the first REAL cross-process collective on fak: a
// coordinator-rooted process group that performs AllReduceSum / AllGather over a real
// wire, with each rank holding ONLY its own part. It is to tensor_parallel.go's
// Collective seam exactly what pipeline_transport.go's TCPTransport is to pipeline.go's
// StageTransport seam — the implementation that actually moves the bytes between
// processes, proven byte-identical to the in-process default on hardware that exists.
//
// WHERE THIS SITS IN THE NATIVE-753B PLAN. Pillar-3 (multi-GPU) had a proven, bit-exact
// in-process tensor-parallel decomposition over LocalCollective (host []float32) and a
// device-tensor compute.CollectiveBackend HAL seam — but every collective was IN-PROCESS:
// no implementation crossed a process boundary, so "the real NCCL/RDMA collective is a
// swap underneath the contract" stayed an aspiration. DistComm crosses that boundary at
// the HOST layer — it is the distributed twin of LocalCollective the way TCPTransport is
// the distributed twin of LocalTransport — and de-risks the architecture the multi-GPU
// pillar stands on: rank coordination, a wire protocol, rank-order reduction across the
// wire, and the fail-closed contract, all on one box.
//
// It is NOT the plan's device rung and must not be read as filling it. That rung ("P3 Real
// cross-process") requires a 2-GPU all-reduce of a DEVICE tensor through a non-cpu-ref
// compute.CollectiveBackend (the NCCL/RCCL backend) — only after which "multi-GPU" may be
// claimed (collective_bridge.go and the staged plan bind the "TCP transport" phrase to that
// device-tensor backend). DistComm reduces HOST float32, so it proves the distributed
// plumbing ABOVE the device line; the device line is the next, GPU-node rung — and DistComm
// is what makes that rung a wire swap, not a redesign, correct exactly when it reproduces
// these bytes.
//
// THE FAITHFUL SHAPE (why a new type, not a model.Collective). LocalCollective and
// BackendCollective implement model.Collective, whose AllReduceSum(parts [][]float32)
// takes EVERY rank's part in one process — the in-process orchestrator shape the wired
// ForwardTP uses today. A genuine distributed collective is the opposite: rank r holds
// only its own part and a connection to the group, which is precisely WHY real NCCL
// serving runs N processes. DistComm therefore has the natural distributed signature
// (each rank passes only its own data) and is a process group ("communicator", NCCL's
// own word), not a model.Collective. Wiring ForwardTP to run per-rank across a DistComm
// is the separate multi-process serving-topology lever (it co-arrives with MLA-aware TP);
// this file lands and PINS the cross-process collective primitive that lever stands on,
// the same way tensor_parallel.go landed the TP primitive before it was wired live.
//
// TOPOLOGY + PROTOCOL. A star rooted at rank 0 (the coordinator): ranks 1..P-1 dial in
// and announce their rank; the coordinator owns one framed connection per worker. Each
// collective is one gather→reduce→scatter round:
//   - every worker sends an op-tagged request frame carrying its part;
//   - the coordinator reads each worker's part FROM THAT WORKER'S connection (so the part
//     is placed at its rank regardless of arrival order), prepends its own rank-0 part, and
//     reduces/concatenates through LocalCollective — the SAME rank-order spec the in-process
//     gate pins, so the result is byte-identical by construction;
//   - the coordinator broadcasts a status-tagged response frame (the reduced vector, or the
//     fail-closed error) back to every worker.
// The op tag lets the coordinator reject a rank desync (a peer calling a DIFFERENT
// collective); the status tag lets a fail-closed reduction error unblock every worker
// instead of deadlocking them on a response that never comes. The coordinator reads ALL
// requests before writing ANY response, so no rank blocks a peer (a classic deadlock-free
// gather-then-scatter). Single rank (P==1) is the identity: no wire, the lone part routed
// through LocalCollective so it inherits the same width validation.
//
// HONESTY. This is a cross-PROCESS collective over HOST float32 — it is NOT multi-GPU and
// is NOT NCCL. "Multi-GPU" stays unclaimable until a non-cpu-ref compute.CollectiveBackend
// (the NCCL/RCCL device backend) all-reduces a DEVICE tensor across 2 GPUs and matches
// cpu-ref on the GPU server. DistComm proves the distributed architecture above the device
// line; the device line is the next, GPU-node rung. Following the repo's own TCPTransport
// precedent, the gate runs the ranks as goroutines over a loopback socket — a genuine
// cross-process send, verifiable on one box.

// distOp tags which collective a request frame belongs to, so the coordinator fails closed
// on a rank desync (a peer that called a different collective in the same round) rather
// than silently reducing mismatched buffers.
type distOp byte

const (
	opAllReduceSum distOp = 1
	opAllGather    distOp = 2
)

func (o distOp) String() string {
	switch o {
	case opAllReduceSum:
		return "AllReduceSum"
	case opAllGather:
		return "AllGather"
	default:
		return fmt.Sprintf("distOp(%d)", byte(o))
	}
}

// DistComm is one rank's handle to a coordinator-rooted process group. Rank 0 is the
// coordinator and holds one connection per worker (workers[r], r>=1); a worker rank holds
// its single connection to the coordinator (coord). It is NOT safe for concurrent use by
// multiple goroutines on one rank — a process group runs one collective at a time, in the
// same order on every rank, exactly like an MPI/NCCL communicator.
type DistComm struct {
	rank    int
	size    int
	workers []net.Conn // rank 0 only: len==size, workers[0]==nil (self), workers[r] for r>=1
	coord   net.Conn   // rank r>0 only: the connection to the coordinator
}

// Rank returns this handle's rank in [0,Size).
func (g *DistComm) Rank() int { return g.rank }

// Size returns the number of ranks in the group.
func (g *DistComm) Size() int { return g.size }

// Coordinate forms the rank-0 end of a process group of `size` ranks: it accepts size-1
// worker connections on ln, reading each worker's announced rank so the connection is
// bound to its rank (not its arrival order). It fails closed on a bad/duplicate announced
// rank or an accept error, closing any connections taken so far. size==1 is the
// single-rank group (no workers accepted), the identity case.
func Coordinate(ln net.Listener, size int) (*DistComm, error) {
	if size < 1 {
		return nil, fmt.Errorf("model: Coordinate size = %d, want >= 1", size)
	}
	g := &DistComm{rank: 0, size: size, workers: make([]net.Conn, size)}
	for i := 1; i < size; i++ {
		conn, err := ln.Accept()
		if err != nil {
			g.Close()
			return nil, fmt.Errorf("model: Coordinate accept worker %d/%d: %w", i, size-1, err)
		}
		var hb [4]byte
		if _, err := io.ReadFull(conn, hb[:]); err != nil {
			conn.Close()
			g.Close()
			return nil, fmt.Errorf("model: Coordinate read rank announce: %w", err)
		}
		r := int(binary.LittleEndian.Uint32(hb[:]))
		if r < 1 || r >= size {
			conn.Close()
			g.Close()
			return nil, fmt.Errorf("model: Coordinate worker announced rank %d, want [1,%d)", r, size)
		}
		if g.workers[r] != nil {
			conn.Close()
			g.Close()
			return nil, fmt.Errorf("model: Coordinate rank %d announced twice", r)
		}
		g.workers[r] = conn
	}
	return g, nil
}

// Join forms a worker (rank>=1) end of a process group: it announces its rank to the
// coordinator over an already-established connection. The caller owns conn's lifecycle
// beyond Close. rank must be in [1,size).
func Join(conn net.Conn, rank, size int) (*DistComm, error) {
	if size < 2 {
		return nil, fmt.Errorf("model: Join size = %d, want >= 2 (rank 0 coordinates)", size)
	}
	if rank < 1 || rank >= size {
		return nil, fmt.Errorf("model: Join rank = %d, want [1,%d)", rank, size)
	}
	if conn == nil {
		return nil, fmt.Errorf("model: Join got a nil connection")
	}
	var hb [4]byte
	binary.LittleEndian.PutUint32(hb[:], uint32(rank))
	if _, err := conn.Write(hb[:]); err != nil {
		return nil, fmt.Errorf("model: Join announce rank %d: %w", rank, err)
	}
	return &DistComm{rank: rank, size: size, coord: conn}, nil
}

// Close closes the connections this rank owns. The coordinator closes every worker
// connection; a worker closes its connection to the coordinator. It is safe to call more
// than once and on a partially-formed group.
func (g *DistComm) Close() error {
	var first error
	if g.rank == 0 {
		for r, c := range g.workers {
			if c == nil {
				continue
			}
			if err := c.Close(); err != nil && first == nil {
				first = err
			}
			g.workers[r] = nil
		}
	} else if g.coord != nil {
		first = g.coord.Close()
		g.coord = nil
	}
	return first
}

// AllReduceSum returns, on EVERY rank, the element-wise sum of all ranks' equal-length
// parts, added in rank order (parts[0] then += parts[r]) — the fixed order LocalCollective
// and sumPartialsRankOrder use, so the result is byte-identical to the in-process reduce.
// It fails closed (on every rank, deadlock-free) on ragged parts, mirroring LocalCollective.
func (g *DistComm) AllReduceSum(myPart []float32) ([]float32, error) {
	if g.size == 1 {
		return LocalCollective{}.AllReduceSum([][]float32{myPart})
	}
	if g.rank == 0 {
		return g.coordinate(opAllReduceSum, myPart, TPPlan{})
	}
	return g.worker(opAllReduceSum, myPart)
}

// AllGather returns, on EVERY rank, the rank-ordered concatenation of all ranks' shards
// (parts[0]‖parts[1]‖…), after the SAME per-rank width validation LocalCollective.AllGather
// applies against plan, so a mis-sized rank is rejected at the boundary rather than
// shifting every downstream feature. Every rank passes the identical plan (a process group
// agrees on its tiling); the coordinator validates against it.
func (g *DistComm) AllGather(myPart []float32, plan TPPlan) ([]float32, error) {
	if g.size == 1 {
		return LocalCollective{}.AllGather([][]float32{myPart}, plan)
	}
	if g.rank == 0 {
		return g.coordinate(opAllGather, myPart, plan)
	}
	return g.worker(opAllGather, myPart)
}

// coordinate runs the rank-0 gather→reduce→scatter for one collective: read each worker's
// op-tagged request from ITS connection (placing the part at its rank), reduce/concatenate
// through LocalCollective with rank 0's own part first, then broadcast a status-tagged
// response to every worker. On a reduction error or a worker op mismatch it broadcasts the
// error so no worker is left blocked, then returns it. Read-all-then-write-all is
// deadlock-free.
func (g *DistComm) coordinate(op distOp, myPart []float32, plan TPPlan) ([]float32, error) {
	parts := make([][]float32, g.size)
	parts[0] = myPart
	var gatherErr error
	for r := 1; r < g.size; r++ {
		gotOp, part, err := readRequest(g.workers[r])
		if err != nil {
			// A transport error means we cannot trust the round; record it but keep draining
			// every connection so a later writeResponse cannot block a half-read peer.
			if gatherErr == nil {
				gatherErr = fmt.Errorf("model: AllReduce/AllGather read rank %d request: %w", r, err)
			}
			continue
		}
		if gotOp != op {
			if gatherErr == nil {
				gatherErr = fmt.Errorf("model: rank %d called %s but coordinator is running %s (process-group desync)", r, gotOp, op)
			}
			continue
		}
		parts[r] = part
	}

	var result []float32
	err := gatherErr
	if err == nil {
		switch op {
		case opAllReduceSum:
			result, err = LocalCollective{}.AllReduceSum(parts)
		case opAllGather:
			result, err = LocalCollective{}.AllGather(parts, plan)
		default:
			err = fmt.Errorf("model: coordinator got unknown op %s", op)
		}
	}

	// Broadcast the verdict to every worker (the result on success, the error otherwise) so
	// every rank unblocks and returns the same outcome.
	for r := 1; r < g.size; r++ {
		if g.workers[r] == nil {
			continue
		}
		if werr := writeResponse(g.workers[r], result, err); werr != nil && err == nil {
			err = fmt.Errorf("model: write response to rank %d: %w", r, werr)
		}
	}
	if err != nil {
		return nil, err
	}
	return result, nil
}

// worker runs a non-coordinator rank's half of one collective: send the op-tagged request
// with its part, then read the status-tagged response (the reduced vector, or the
// coordinator's fail-closed error verbatim).
func (g *DistComm) worker(op distOp, myPart []float32) ([]float32, error) {
	if err := writeRequest(g.coord, op, myPart); err != nil {
		return nil, fmt.Errorf("model: rank %d send %s request: %w", g.rank, op, err)
	}
	out, err := readResponse(g.coord)
	if err != nil {
		return nil, fmt.Errorf("model: rank %d recv %s response: %w", g.rank, op, err)
	}
	return out, nil
}

// ---- wire format -----------------------------------------------------------------
//
// Each frame is delimited by writeFrame/readFrame (the 4-byte length prefix from
// pipeline_transport.go). A request payload is [op:1][f32-vector]; a response payload is
// [status:1] then, on ok, the f32-vector, else a UTF-8 error message. A float32 vector is
// [count:4 LE] then count words via math.Float32bits — the identity bit pattern, so the
// round-trip preserves signed zero and NaN and cannot perturb a single bit (the same
// guarantee MarshalHidden documents). This is why the cross-process reduce is BIT-exact
// vs the in-process one and not merely close.

const (
	statusOK  byte = 0
	statusErr byte = 1
)

// encodeF32 serializes a float32 vector as [count:4 LE][count words via Float32bits].
func encodeF32(v []float32) []byte {
	buf := make([]byte, 4+len(v)*4)
	binary.LittleEndian.PutUint32(buf[0:4], uint32(len(v)))
	off := 4
	for _, f := range v {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(f))
		off += 4
	}
	return buf
}

// decodeF32 reads a vector written by encodeF32 from the front of b and returns it with the
// number of bytes consumed. It fails closed on a truncated buffer rather than reading past
// the end.
func decodeF32(b []byte) ([]float32, int, error) {
	if len(b) < 4 {
		return nil, 0, fmt.Errorf("model: dist vector header %d bytes, want >= 4", len(b))
	}
	n := int(binary.LittleEndian.Uint32(b[0:4]))
	// `n > (len(b)-4)/4` rather than `4+n*4 > len(b)`: the latter overflows int on a 32-bit
	// GOARCH (n*4 wraps, the guard wrongly passes, and the read loop then runs off the buffer),
	// which would defeat the fail-closed contract on untrusted wire input. This form is
	// overflow-safe on every word size — the same discipline mmap_unix.go uses for 32-bit hosts.
	if n < 0 || n > (len(b)-4)/4 {
		return nil, 0, fmt.Errorf("model: dist vector claims %d words but only %d bytes after header", n, len(b)-4)
	}
	v := make([]float32, n)
	off := 4
	for i := 0; i < n; i++ {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[off : off+4]))
		off += 4
	}
	return v, 4 + n*4, nil
}

// writeRequest frames an op-tagged request carrying the rank's part.
func writeRequest(conn net.Conn, op distOp, v []float32) error {
	payload := append([]byte{byte(op)}, encodeF32(v)...)
	return writeFrame(conn, payload)
}

// readRequest decodes a frame written by writeRequest.
func readRequest(conn net.Conn) (distOp, []float32, error) {
	payload, err := readFrame(conn)
	if err != nil {
		return 0, nil, err
	}
	if len(payload) < 1 {
		return 0, nil, fmt.Errorf("model: dist request frame is empty")
	}
	v, _, err := decodeF32(payload[1:])
	if err != nil {
		return 0, nil, err
	}
	return distOp(payload[0]), v, nil
}

// writeResponse frames the coordinator's verdict: a status byte then, on success, the
// result vector, else the error text. A nil opErr with a nil result still writes a valid
// empty-vector ok frame.
func writeResponse(conn net.Conn, result []float32, opErr error) error {
	if opErr != nil {
		return writeFrame(conn, append([]byte{statusErr}, []byte(opErr.Error())...))
	}
	return writeFrame(conn, append([]byte{statusOK}, encodeF32(result)...))
}

// readResponse decodes a frame written by writeResponse, surfacing the coordinator's
// fail-closed error as an error on this rank so a refusal propagates to every rank.
func readResponse(conn net.Conn) ([]float32, error) {
	payload, err := readFrame(conn)
	if err != nil {
		return nil, err
	}
	if len(payload) < 1 {
		return nil, fmt.Errorf("model: dist response frame is empty")
	}
	switch payload[0] {
	case statusErr:
		return nil, fmt.Errorf("model: coordinator refused the collective: %s", string(payload[1:]))
	case statusOK:
		v, _, err := decodeF32(payload[1:])
		return v, err
	default:
		return nil, fmt.Errorf("model: dist response status byte = %d, want 0(ok)/1(err)", payload[0])
	}
}

// distCommCollective adapts a DistComm (this rank's handle to the process group) to the model
// Collective seam so the SHARDED EP forward can reduce through it. The impedance match is the
// number of parts: model.Collective.AllReduceSum takes EVERY rank's part in one process (the
// in-process orchestrator shape LocalCollective/BackendCollective serve), while a genuine
// distributed collective has each rank hold ONLY its own part — DistComm's natural signature. So
// this adapter accepts exactly ONE part (this rank's) and forwards it to DistComm.AllReduceSum,
// which gathers the peers over the wire and reduces in rank order — byte-identical to the
// in-process reduce (dist_collective_test.go pins DistComm == LocalCollective at max|Δ|=0). At
// size 1 DistComm is the identity, so a one-process sharded EP still reduces correctly.
//
// The rank-local EP forward only ever all-reduces its [H] routed partial; it never gathers bands
// (AllGather recombines a column-parallel matmul's output, which EP does not do). AllGather
// therefore refuses rather than silently mis-reduce — the fail-closed contract every collective
// here keeps.
type distCommCollective struct{ g *DistComm }

// NewDistCommCollective wraps a rank's DistComm handle as a model Collective for the sharded EP
// decode path (SetExpertParallelCollective). The caller owns g's lifecycle (Close on serve exit).
func NewDistCommCollective(g *DistComm) Collective { return distCommCollective{g: g} }

func (d distCommCollective) AllReduceSum(parts [][]float32) ([]float32, error) {
	if len(parts) != 1 {
		return nil, fmt.Errorf("model: distCommCollective.AllReduceSum expects this rank's single part, got %d (a distributed collective holds only its own part)", len(parts))
	}
	return d.g.AllReduceSum(parts[0])
}

func (d distCommCollective) AllGather(parts [][]float32, p TPPlan) ([]float32, error) {
	return nil, fmt.Errorf("model: distCommCollective does not implement AllGather (expert-parallel reduces [H] partials through AllReduceSum, it never gathers expert bands)")
}

// distCommCollective is a drop-in for the model Collective seam.
var _ Collective = distCommCollective{}
