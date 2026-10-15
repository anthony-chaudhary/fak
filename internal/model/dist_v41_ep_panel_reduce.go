package model

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"time"
)

// A panel is one physical exchange of independently identified logical rows.
// Scalar Step and singleton Prefill retain opcode 4 and its original protocol.
const opV41EPPanelReduce distOp = 5
const v41EPPanelWireVersion = uint32(0x56343103)
const v41EPPanelHeaderWords = 11
const v41EPPanelLiveBytes = 64 << 20

type v41EPPanelReduction struct {
	owners  TPPlan
	layer   int
	width   int
	picks   [][]routePick
	digests [][32]byte
	sum     v41RoutedSumFunc
}

type v41EPPanelLayout struct {
	requests     []int
	payloadWords int
	replyWords   int
	liveBytes    uint64
}

// v41EPPanelPlan materializes the per-owner request lengths only for callers
// that consume them. Window selection uses the same complete admission checks.
func v41EPPanelPlan(owners TPPlan, picks [][]routePick, width int) (v41EPPanelLayout, error) {
	plan, err := v41EPPanelAdmitted(owners, picks, width)
	if err != nil {
		return plan, err
	}
	base := v41EPPanelHeaderWords + len(picks)*v41EPHeaderWords
	plan.requests = make([]int, len(owners.Shards))
	for rank, band := range owners.Shards {
		words := base
		for _, row := range picks {
			for _, pick := range row {
				if pick.expert >= band.Lo && pick.expert < band.Hi {
					words += width
				}
			}
		}
		plan.requests[rank] = words
	}
	return plan, nil
}

// Admission is entirely common geometry: even Qmax scans ALL owner bands.
// No allocation, owner math or I/O precedes the byte checks. All arithmetic
// below is uint64 with operands first bounded by small protocol constants, so
// neither machine-int nor uint64 overflow can authorize an allocation.
func v41EPPanelAdmitted(owners TPPlan, picks [][]routePick, width int) (v41EPPanelLayout, error) {
	var plan v41EPPanelLayout
	n, world := len(picks), len(owners.Shards)
	if owners.Validate() != nil || n < 2 || n > v41EPOwnerPanelTokens || width <= 0 || width > v41EPMaxWords || world > v41EPMaxWords/v41EPHeaderWords {
		return plan, fmt.Errorf("invalid panel geometry")
	}
	k, maxPicks := 0, 0
	for _, row := range picks {
		if len(row) > v41EPOwnerPanelPicks-k {
			return plan, fmt.Errorf("panel pick bound exceeded")
		}
		k += len(row)
		maxPicks = max(maxPicks, len(row))
		for _, pick := range row {
			if pick.expert < 0 || pick.expert >= owners.Dim || !finite32(pick.weight) {
				return plan, fmt.Errorf("invalid panel pick")
			}
		}
	}
	N, W, H, K := uint64(n), uint64(world), uint64(width), uint64(k)
	P := H * K
	if 4*P > v41EPOwnerPanelMaxBytes {
		return plan, fmt.Errorf("panel owner payload bound exceeded")
	}
	base := uint64(v41EPPanelHeaderWords) + N*v41EPHeaderWords
	A, Qmax := W*base+P, base
	metadata := 64*(N*W+K+N+W) + 4096
	if 4*A+metadata > v41EPPanelLiveBytes {
		return plan, fmt.Errorf("panel minimum transport allowance exceeded")
	}
	for _, band := range owners.Shards {
		owned := uint64(0)
		for _, row := range picks {
			for _, pick := range row {
				if pick.expert >= band.Lo && pick.expert < band.Hi {
					owned++
				}
			}
		}
		Qmax = max(Qmax, base+H*owned)
	}
	Gmax := W*v41EPHeaderWords + H*uint64(maxPicks)
	F := uint64(v41EPPanelHeaderWords) + N*(v41EPHeaderWords+H)
	if 5+4*Qmax > v41EPMaxFrameBytes || 5+4*F > v41EPMaxFrameBytes {
		return plan, fmt.Errorf("panel frame bound exceeded")
	}
	// Live explicit owner/transport allowance, not RSS or device peak. Root:
	// borrowed owner rows P, decoded requests A (including the local copy),
	// in-flight encoded read Qmax, private exact-capacity unpack slab P,
	// response F, encoded response F, current backend result H. Reserve an
	// additional Gmax + F, and all slice/header/plan metadata below. The extra
	// F also covers simultaneous cached success and small failure encodings
	// during a response-write failure. Follower:
	// owner rows P, request + encoding <= 3*Qmax, reply + decoding <= 3*F.
	// Borrowed normalized inputs and backend scratch are excluded. Earlier
	// completed windows retain caller-owned results plus their bounded wire
	// headers, outside this current-window allowance. This is not a bound on
	// full-history Prefill storage; cumulative output accounting is checked
	// separately by v41EPPrefillResultStorageBound. No append-grown payload
	// arrays exist here.
	peak := max(4*(2*P+A+Qmax+Gmax+3*F+H), 4*(P+3*Qmax+3*F))
	peak += metadata
	if peak > v41EPPanelLiveBytes {
		return plan, fmt.Errorf("panel live transport allowance exceeded")
	}
	return v41EPPanelLayout{payloadWords: int(P), replyWords: int(F), liveBytes: peak}, nil
}

func (r v41EPPanelReduction) identity() [32]byte {
	h := sha256.New()
	_, _ = h.Write([]byte("fak-v41-exact-prefill-panel-v1"))
	var word [8]byte
	put := func(n int) { binary.LittleEndian.PutUint64(word[:], uint64(n)); _, _ = h.Write(word[:]) }
	put(len(r.picks))
	put(r.layer)
	put(r.width)
	put(r.owners.Dim)
	put(len(r.owners.Shards))
	for _, band := range r.owners.Shards {
		put(band.Rank)
		put(band.Lo)
		put(band.Hi)
	}
	for _, digest := range r.digests {
		_, _ = h.Write(digest[:])
	}
	var digest [32]byte
	copy(digest[:], h.Sum(nil))
	return digest
}

func v41EPPanelHeader(part []float32, digest [32]byte, rows int, failed bool) {
	v41EPHeader(part, digest, failed)
	part[0] = math.Float32frombits(v41EPPanelWireVersion)
	part[10] = math.Float32frombits(uint32(rows))
}

func v41EPCheckPanelHeader(part []float32, digest [32]byte, rows int) (bool, error) {
	if len(part) < v41EPPanelHeaderWords {
		return false, fmt.Errorf("short panel envelope")
	}
	var want [v41EPPanelHeaderWords]float32
	v41EPPanelHeader(want[:], digest, rows, false)
	for i, value := range want {
		if i != 1 && math.Float32bits(part[i]) != math.Float32bits(value) {
			return false, fmt.Errorf("panel identity, row count or version differs")
		}
	}
	status := math.Float32bits(part[1])
	if status > 1 {
		return false, fmt.Errorf("invalid panel status")
	}
	return status == 1, nil
}

func (c *v41ExpertParallelCollective) reducePanel(ownerRows [][]float32, r v41EPPanelReduction) (out [][]float32, err error) {
	if c.failure != nil {
		return nil, c.failure
	}
	opened := time.Now()
	defer func() {
		if err != nil {
			c.failure = fmt.Errorf("%w: root-ordered panel: %w", ErrV41ExpertParallel, err)
			c.g.abortV41EP()
			out, err = nil, c.failure
			var rootPanic *v41EPRootPanic
			if errors.As(err, &rootPanic) {
				panic(rootPanic.value)
			}
		}
	}()
	plan, err := v41EPPanelPlan(r.owners, r.picks, r.width)
	if err != nil {
		return nil, err
	}
	if c.g.Size() != len(plan.requests) || c.g.Rank() < 0 || c.g.Rank() >= len(plan.requests) || len(r.digests) != len(r.picks) || len(ownerRows) != len(r.picks) {
		return nil, fmt.Errorf("invalid panel rank or identity count")
	}
	digest := r.identity()
	var part []float32
	var request []byte
	if c.g.Rank() == 0 {
		part, err = r.pack(ownerRows, plan, 0, digest)
	} else {
		request, err = r.encodeOwnerRequest(ownerRows, plan, c.g.Rank(), digest)
	}
	if err != nil {
		return nil, err
	}
	timeout := c.timeout
	if timeout <= 0 {
		timeout = v41EPRoundTimeout
	}
	if err = c.g.v41EPDeadline(opened.Add(timeout)); err != nil {
		return nil, err
	}
	if c.g.Rank() == 0 {
		out, err = c.reducePanelRoot(part, plan, r, digest)
	} else if err = writeFrame(c.g.coord, request); err == nil {
		var reply []float32
		reply, err = readV41EPPanelVector(c.g.coord, plan.replyWords, false)
		if err == nil {
			out, err = r.response(reply, digest)
		}
	}
	if err != nil {
		return nil, err
	}
	if err = c.g.v41EPDeadline(time.Time{}); err != nil {
		return nil, err
	}
	replyBytes := uint64(9 + 4*plan.replyWords)
	if c.g.Rank() == 0 {
		c.sentBytes.Add(uint64(c.g.Size()-1) * replyBytes)
		for _, words := range plan.requests[1:] {
			c.receivedBytes.Add(uint64(9 + 4*words))
		}
	} else {
		c.sentBytes.Add(uint64(9 + 4*plan.requests[c.g.Rank()]))
		c.receivedBytes.Add(replyBytes)
	}
	c.gatherNanos.Add(uint64(time.Since(opened)))
	// Never credit partial root math. Like scalar completion, this is local:
	// an earlier peer may have accepted success before a later write fails.
	c.panelExtraRows.Add(uint64(len(r.picks) - 1))
	c.rounds.Add(1)
	return out, nil
}

func (r v41EPPanelReduction) pack(rows [][]float32, plan v41EPPanelLayout, rank int, digest [32]byte) ([]float32, error) {
	band := r.owners.Shards[rank]
	// Validate the entire local contribution before allocating its duplicate.
	for token, picks := range r.picks {
		words := 0
		for _, pick := range picks {
			if pick.expert >= band.Lo && pick.expert < band.Hi {
				words += r.width
			}
		}
		if len(rows[token]) != words {
			return nil, fmt.Errorf("invalid panel owner row %d width", token)
		}
		for _, value := range rows[token] {
			if !finite32(value) {
				return nil, fmt.Errorf("nonfinite panel owner row %d", token)
			}
		}
	}
	part := make([]float32, plan.requests[rank])
	v41EPPanelHeader(part, digest, len(rows), false)
	offset := v41EPPanelHeaderWords
	for token, row := range rows {
		v41EPHeader(part[offset:], r.digests[token], false)
		offset += v41EPHeaderWords
		copy(part[offset:], row)
		offset += len(row)
	}
	return part, nil
}

// Followers need only a wire snapshot. Preserve pack's full local validation
// and refusal order before allocating it, without the intermediate F32 part.
// Source rows own their storage under Session execution ownership; this copy
// is independent and complete before deadline installation or any I/O. The
// existing conservative panel allowance and deterministic windows are unchanged.
func (r v41EPPanelReduction) encodeOwnerRequest(rows [][]float32, plan v41EPPanelLayout, rank int, digest [32]byte) ([]byte, error) {
	band := r.owners.Shards[rank]
	for token, picks := range r.picks {
		words := 0
		for _, pick := range picks {
			if pick.expert >= band.Lo && pick.expert < band.Hi {
				words += r.width
			}
		}
		if len(rows[token]) != words {
			return nil, fmt.Errorf("invalid panel owner row %d width", token)
		}
		for _, value := range rows[token] {
			if !finite32(value) {
				return nil, fmt.Errorf("nonfinite panel owner row %d", token)
			}
		}
	}
	words := plan.requests[rank]
	if words < v41EPPanelHeaderWords || words > v41EPMaxWords {
		return nil, fmt.Errorf("invalid panel write bound")
	}
	payload := make([]byte, 5+4*words)
	payload[0] = byte(opV41EPPanelReduce)
	binary.LittleEndian.PutUint32(payload[1:5], uint32(words))
	at := 5
	put := func(values []float32) {
		for _, value := range values {
			binary.LittleEndian.PutUint32(payload[at:], math.Float32bits(value))
			at += 4
		}
	}
	var header [v41EPPanelHeaderWords]float32
	v41EPPanelHeader(header[:], digest, len(rows), false)
	put(header[:])
	for token, row := range rows {
		v41EPHeader(header[:], r.digests[token], false)
		put(header[:v41EPHeaderWords])
		put(row)
	}
	return payload, nil
}

func (c *v41ExpertParallelCollective) reducePanelRoot(part []float32, plan v41EPPanelLayout, r v41EPPanelReduction, digest [32]byte) ([][]float32, error) {
	parts := make([][]float32, len(plan.requests))
	parts[0] = part
	var cause error
	for rank := 1; rank < c.g.Size(); rank++ {
		row, err := readV41EPPanelVector(c.g.workers[rank], plan.requests[rank], true)
		if err != nil {
			_ = c.g.workers[rank].Close()
			if cause == nil {
				cause = fmt.Errorf("owner %d panel frame: %w", rank, err)
			}
			continue
		}
		parts[rank] = row
	}
	var reply []float32
	if cause == nil {
		reply, cause = r.reduce(parts, plan, digest)
	}
	if cause != nil {
		reply = make([]float32, v41EPPanelHeaderWords)
		v41EPPanelHeader(reply, digest, len(r.picks), true)
	}
	// Encode lazily so a zero-peer root retains its original behavior. All
	// writes are synchronous; peers share immutable bytes, never the F32 slab.
	var payload []byte
	for rank := 1; rank < c.g.Size(); rank++ {
		var err error
		if payload == nil {
			payload, err = encodeV41EPPanelVector(reply)
		}
		if err == nil {
			err = writeFrame(c.g.workers[rank], payload)
		}
		if err != nil && cause == nil {
			cause = fmt.Errorf("owner %d panel response: %w", rank, err)
			// Preserve the existing partial-write asymmetry. No later peer
			// receives success, but an earlier completed response is irrevocable.
			// Discard the success encoding before visiting any later peer.
			payload = nil
			reply = make([]float32, v41EPPanelHeaderWords)
			v41EPPanelHeader(reply, digest, len(r.picks), true)
		}
	}
	if cause != nil {
		return nil, cause
	}
	return r.response(reply, digest)
}

func (r v41EPPanelReduction) reduce(parts [][]float32, plan v41EPPanelLayout, digest [32]byte) (reply []float32, err error) {
	defer func() {
		if value := recover(); value != nil {
			reply, err = nil, &v41EPRootPanic{value: value}
		}
	}()
	// Exact-capacity private contribution storage, independent of wire buffers
	// and caller-owned rows. Every token/owner is validated before any root sum.
	flat := make([]float32, plan.payloadWords)
	outputs := make([][][]float32, len(r.picks))
	offset := 0
	for token, picks := range r.picks {
		outputs[token] = make([][]float32, len(picks))
		for slot := range picks {
			outputs[token][slot] = flat[offset : offset+r.width : offset+r.width]
			offset += r.width
		}
	}
	for rank, part := range parts {
		if len(part) != plan.requests[rank] {
			return nil, fmt.Errorf("owner %d panel width differs", rank)
		}
		failed, err := v41EPCheckPanelHeader(part, digest, len(r.picks))
		if err != nil || failed {
			return nil, fmt.Errorf("owner %d refused or panel envelope differs", rank)
		}
		at, band := v41EPPanelHeaderWords, r.owners.Shards[rank]
		for token, picks := range r.picks {
			var want [v41EPHeaderWords]float32
			v41EPHeader(want[:], r.digests[token], false)
			for i, value := range want {
				if math.Float32bits(part[at+i]) != math.Float32bits(value) {
					return nil, fmt.Errorf("owner %d row %d refused or identity differs", rank, token)
				}
			}
			at += v41EPHeaderWords
			for slot, pick := range picks {
				if pick.expert < band.Lo || pick.expert >= band.Hi {
					continue
				}
				row := part[at : at+r.width]
				for _, value := range row {
					if !finite32(value) {
						return nil, fmt.Errorf("nonfinite owner %d row %d contribution", rank, token)
					}
				}
				copy(outputs[token][slot], row)
				at += r.width
			}
		}
	}
	reply = make([]float32, plan.replyWords)
	v41EPPanelHeader(reply, digest, len(r.picks), false)
	offset = v41EPPanelHeaderWords
	for token, picks := range r.picks {
		row, cause := v41FullRoutedSumWith(r.layer, picks, outputs[token], r.width, r.sum)
		if cause != nil {
			return nil, fmt.Errorf("panel row %d exact sum: %w", token, cause)
		}
		if len(row) != r.width {
			return nil, fmt.Errorf("panel row %d sum width differs", token)
		}
		for _, value := range row {
			if !finite32(value) {
				return nil, fmt.Errorf("nonfinite panel row %d sum", token)
			}
		}
		v41EPHeader(reply[offset:], r.digests[token], false)
		offset += v41EPHeaderWords
		copy(reply[offset:], row)
		offset += r.width
	}
	return reply, nil
}

func (r v41EPPanelReduction) response(reply []float32, digest [32]byte) ([][]float32, error) {
	failed, err := v41EPCheckPanelHeader(reply, digest, len(r.picks))
	if err != nil {
		return nil, err
	}
	if failed {
		if len(reply) != v41EPPanelHeaderWords {
			return nil, fmt.Errorf("failed panel response contains results")
		}
		return nil, fmt.Errorf("root refused exact panel reduction")
	}
	if len(reply) != v41EPPanelHeaderWords+len(r.picks)*(v41EPHeaderWords+r.width) {
		return nil, fmt.Errorf("panel response width differs")
	}
	rows := make([][]float32, len(r.picks))
	at := v41EPPanelHeaderWords
	for token := range rows {
		last := at + v41EPHeaderWords + r.width
		row, err := v41EPReducedRow(reply[at:last:last], r.digests[token], r.width)
		if err != nil {
			return nil, fmt.Errorf("panel response row %d: %w", token, err)
		}
		rows[token], at = row, last
	}
	return rows, nil
}

// These typed frame helpers never use append-grown encoding buffers. Both the
// trusted bound and untrusted advertised span are checked before allocation.
func writeV41EPPanelVector(conn net.Conn, row []float32) error {
	payload, err := encodeV41EPPanelVector(row)
	if err != nil {
		return err
	}
	return writeFrame(conn, payload)
}

func encodeV41EPPanelVector(row []float32) ([]byte, error) {
	if len(row) < v41EPPanelHeaderWords || len(row) > v41EPMaxWords {
		return nil, fmt.Errorf("invalid panel write bound")
	}
	payload := make([]byte, 5+4*len(row))
	payload[0] = byte(opV41EPPanelReduce)
	binary.LittleEndian.PutUint32(payload[1:5], uint32(len(row)))
	for i, value := range row {
		binary.LittleEndian.PutUint32(payload[5+4*i:], math.Float32bits(value))
	}
	return payload, nil
}

func readV41EPPanelVector(conn net.Conn, words int, exact bool) ([]float32, error) {
	if words < v41EPPanelHeaderWords || words > v41EPMaxWords {
		return nil, fmt.Errorf("invalid panel read bound")
	}
	var header [4]byte
	if _, err := io.ReadFull(conn, header[:]); err != nil {
		return nil, err
	}
	n := uint64(binary.LittleEndian.Uint32(header[:]))
	limit := uint64(5 + 4*words)
	if n > limit || n < 5+4*v41EPPanelHeaderWords || exact && n != limit {
		return nil, fmt.Errorf("invalid panel frame length %d", n)
	}
	payload := make([]byte, int(n))
	if _, err := io.ReadFull(conn, payload); err != nil {
		return nil, err
	}
	if distOp(payload[0]) != opV41EPPanelReduce || uint64(binary.LittleEndian.Uint32(payload[1:5]))*4+5 != n {
		return nil, fmt.Errorf("panel collective kind or vector length differs")
	}
	row, _, err := decodeF32(payload[1:])
	return row, err
}
