package model

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"time"
)

// This op is deliberately distinct from generic AllGather/AllReduce. Mixing
// protocol generations or collective kinds is a terminal group error.
const opV41EPReduce distOp = 4

// v41EPReduction is private to the exact V4.1 path. The root invokes sum on
// its calling goroutine, inside the Session's existing backend execution scope.
// DistComm does not acquire a backend owner or invoke any follower callback.
type v41EPReduction struct {
	owners TPPlan
	digest [32]byte
	layer  int
	picks  []routePick
	width  int
	sum    v41RoutedSumFunc
}

func (c *v41ExpertParallelCollective) reduce(part []float32, round v41EPReduction) (out []float32, err error) {
	if c.failure != nil {
		return nil, c.failure
	}
	opened := time.Now()
	defer func() {
		if err != nil {
			c.failure = fmt.Errorf("%w: root-ordered round: %w", ErrV41ExpertParallel, err)
			c.g.abortV41EP()
			out, err = nil, c.failure
			var rootPanic *v41EPRootPanic
			if errors.As(err, &rootPanic) {
				panic(rootPanic.value)
			}
		}
	}()
	plan, err := v41EPGatherPlan(round.picks, round.owners, round.width)
	if err != nil {
		return nil, err
	}
	if len(plan.Shards) != c.g.Size() || c.g.Rank() < 0 || c.g.Rank() >= len(plan.Shards) || len(part) != plan.Shards[c.g.Rank()].Width() {
		return nil, fmt.Errorf("invalid exact round rank or local width")
	}
	timeout := c.timeout
	if timeout <= 0 {
		timeout = v41EPRoundTimeout
	}
	if err = c.g.v41EPDeadline(opened.Add(timeout)); err != nil {
		return nil, err
	}
	if c.g.Rank() == 0 {
		out, err = c.reduceRoot(part, plan, round)
	} else {
		if err = writeRequest(c.g.coord, opV41EPReduce, part); err == nil {
			var reply []float32
			reply, err = readV41EPVector(c.g.coord, v41EPHeaderWords+round.width, false)
			if err == nil {
				out, err = v41EPReducedRow(reply, round.digest, round.width)
			}
		}
	}
	if err != nil {
		return nil, err
	}
	if err = c.g.v41EPDeadline(time.Time{}); err != nil {
		return nil, err
	}
	// Count completed application frames only, including length/op/vector and
	// identity/status headers. TCP/IP overhead and failed partial I/O are absent.
	replyBytes := uint64(9 + 4*(v41EPHeaderWords+round.width))
	if c.g.Rank() == 0 {
		c.sentBytes.Add(uint64(c.g.Size()-1) * replyBytes)
		c.receivedBytes.Add(uint64(9*(c.g.Size()-1) + 4*(plan.Dim-len(part))))
	} else {
		c.sentBytes.Add(uint64(9 + 4*len(part)))
		c.receivedBytes.Add(replyBytes)
	}
	c.gatherNanos.Add(uint64(time.Since(opened)))
	c.rounds.Add(1)
	return out, nil
}

func (c *v41ExpertParallelCollective) reduceRoot(part []float32, plan TPPlan, round v41EPReduction) ([]float32, error) {
	gathered := make([]float32, plan.Dim)
	copy(gathered, part)
	var cause error
	for rank := 1; rank < c.g.Size(); rank++ {
		shard, conn := plan.Shards[rank], c.g.workers[rank]
		row, err := readV41EPVector(conn, shard.Width(), true)
		if err != nil {
			// A rejected frame may be only partly consumed. Close that socket
			// immediately so its sender cannot remain blocked on the remainder.
			_ = conn.Close()
			if cause == nil {
				cause = fmt.Errorf("owner %d frame: %w", rank, err)
			}
			continue
		}
		copy(gathered[shard.Lo:shard.Hi], row)
	}
	var result []float32
	if cause == nil {
		result, cause = round.reduce(gathered, plan)
	}
	// Both verdicts bind the entire round identity. Failure carries no result
	// and no backend error/panic text; the original cause stays on its owner.
	words := v41EPHeaderWords
	if cause == nil {
		words += round.width
	}
	reply := make([]float32, words)
	v41EPHeader(reply, round.digest, cause != nil)
	if cause == nil {
		copy(reply[v41EPHeaderWords:], result)
	}
	for rank := 1; rank < c.g.Size(); rank++ {
		if err := writeRequest(c.g.workers[rank], opV41EPReduce, reply); err != nil && cause == nil {
			cause = fmt.Errorf("owner %d response: %w", rank, err)
			// No later peer receives success after a broadcast write fails.
			reply = make([]float32, v41EPHeaderWords)
			v41EPHeader(reply, round.digest, true)
		}
	}
	if cause != nil {
		return nil, cause
	}
	return result, nil
}

// Keep panic identity private and intact while the root first sends a generic
// failure verdict. The owning reduce boundary aborts/latches, then rethrows it.
type v41EPRootPanic struct{ value any }

func (*v41EPRootPanic) Error() string { return "exact root sum panicked" }
func (p *v41EPRootPanic) Unwrap() error {
	cause, _ := p.value.(error)
	return cause
}

func (r v41EPReduction) reduce(gathered []float32, plan TPPlan) (row []float32, err error) {
	// A selected root backend can return or panic. Either must reach peers as
	// a failure verdict before sockets close, without replay or another sum.
	complete := false
	defer func() {
		if value := recover(); !complete {
			row, err = nil, &v41EPRootPanic{value: value}
		}
	}()
	row, err = func() ([]float32, error) {
		outputs, err := v41EPUnpack(gathered, plan, r.owners, r.digest, r.picks, r.width)
		if err != nil {
			return nil, err
		}
		row, err = v41FullRoutedSumWith(r.layer, r.picks, outputs, r.width, r.sum)
		if err != nil {
			return nil, fmt.Errorf("exact sum: %w", err)
		}
		if len(row) != r.width {
			return nil, fmt.Errorf("invalid exact sum width")
		}
		for _, value := range row {
			if !finite32(value) {
				return nil, fmt.Errorf("nonfinite exact sum")
			}
		}
		return row, nil
	}()
	complete = true
	return row, err
}

// The generic frame readers intentionally retain their existing semantics.
// Only this typed protocol rejects oversized, short and trailing-byte frames
// before allocation and requires exactly the advertised vector length.
func readV41EPVector(conn net.Conn, words int, exact bool) ([]float32, error) {
	if words < v41EPHeaderWords || words > v41EPMaxWords {
		return nil, fmt.Errorf("invalid exact frame bound")
	}
	var header [4]byte
	if _, err := io.ReadFull(conn, header[:]); err != nil {
		return nil, err
	}
	n := uint64(binary.LittleEndian.Uint32(header[:]))
	limit := uint64(5 + 4*words)
	if n > limit || n < 5+4*v41EPHeaderWords || (exact && n != limit) {
		return nil, fmt.Errorf("invalid exact frame length %d", n)
	}
	payload := make([]byte, int(n))
	if _, err := io.ReadFull(conn, payload); err != nil {
		return nil, err
	}
	if distOp(payload[0]) != opV41EPReduce {
		return nil, fmt.Errorf("exact round collective kind mismatch")
	}
	row, consumed, err := decodeF32(payload[1:])
	if err != nil {
		return nil, err
	}
	if consumed != len(payload)-1 || (exact && len(row) != words) {
		return nil, fmt.Errorf("exact vector length mismatch")
	}
	return row, nil
}

func v41EPReducedRow(reply []float32, digest [32]byte, width int) ([]float32, error) {
	if len(reply) < v41EPHeaderWords {
		return nil, fmt.Errorf("short exact response header")
	}
	want := make([]float32, v41EPHeaderWords)
	v41EPHeader(want, digest, false)
	for i, value := range want {
		if i != 1 && math.Float32bits(reply[i]) != math.Float32bits(value) {
			return nil, fmt.Errorf("exact response identity or version differs")
		}
	}
	switch math.Float32bits(reply[1]) {
	case 1:
		if len(reply) != v41EPHeaderWords {
			return nil, fmt.Errorf("failed exact response contains a result")
		}
		return nil, fmt.Errorf("root refused exact reduction")
	case 0:
		if len(reply)-v41EPHeaderWords != width {
			return nil, fmt.Errorf("exact response width differs")
		}
	default:
		return nil, fmt.Errorf("invalid exact response status")
	}
	for _, value := range reply[v41EPHeaderWords:] {
		if !finite32(value) {
			return nil, fmt.Errorf("nonfinite exact response")
		}
	}
	return reply[v41EPHeaderWords:], nil
}
