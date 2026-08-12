package model

import (
	"encoding/binary"
	"fmt"
	"sync"
)

// ep_decode_coord.go — the coordinated expert-parallel decode protocol (#4835).
//
// The temporary EP serve topology (internal/gateway/http_epfanout.go) mirrors the WHOLE
// chat request to every rank over HTTP, so all N ranks redundantly tokenize, prefill,
// decode, AND sample the same prompt — only rank 0's body is returned and the followers'
// responses are discarded. That mirror exists solely so every rank reaches the per-MoE-
// layer AllReduce in lockstep, but it pays for it by running the entire request N times
// (the 8-GPU witness: 0.0406 tok/s, slower than the ~0.2 tok/s scalar baseline) and by
// running N independent samplers whose token histories can diverge.
//
// This is the coordinated replacement: a rank-0-rooted decode frame stream carried on the
// SAME DistComm as the forward's collectives, in the same deterministic order on every
// rank. Rank 0 OWNS tokenization input, sampling, and the token history; followers
// contribute ONLY their local expert compute + collectives — they never tokenize, never
// sample, never build a response, and their per-step logits are discarded.
//
// The seam is Session.Prefill/Session.Step, NOT a bespoke decode driver. That placement is
// load-bearing: the live serve decodes through the in-kernel planner's own real loop in
// internal/agent (prefix cache, temperature/penalty sampling, stop sequences, batched lanes),
// so a driver that re-implemented greedy Generate could never BE the serve path. Hooking the
// two forward entry points instead means whatever forward rank 0 actually runs is the forward
// the followers replay, with rank 0's sampler left exactly as it is.
//
// Ordering safety: a DistComm runs one collective at a time in the same order on every
// rank (dist_collective.go). Rank 0 broadcasts a frame and THEN runs its forward; each
// follower receives the frame and THEN runs the SAME forward (same token ids, same
// sequence position => same routing => same number and order of AllReduces). So the
// broadcast rounds and the forward's internal AllReduces interleave identically on all
// ranks — no new deadlock class. Because every rank sees the same ids at the same
// positions, the per-layer AllReduce keeps all hidden states aligned automatically, and
// rank 0's sampled tokens are, by construction, bit-identical to an uncoordinated
// single-process decode of the same session.
//
// Not covered here (tracked on #4835): the cmd/fak + internal/gateway serve wiring that
// parks follower ranks in RunEPFollower and stops mirroring the HTTP request, and the
// end-to-end tok/s artifact on the 8-GPU checkpoint, which stays GPU-gated. Concurrency >1
// is deliberately SERIALIZED by the coordinator mutex rather than interleaved: overlapping
// requests would need a coordinated request id in the frame, which this slice does not add.

// epDecodeOp tags a coordinated-decode frame. The wire form is
// [op:1][pos:4 LE int32][count:4 LE uint32][count x uint32 LE token id]: PREFILL carries the
// prompt ids, STEP carries exactly one accepted id, SHUTDOWN carries none. pos is the
// sequence position the forward is about to run AT — the follower refuses a frame whose pos
// does not match its own mirrored session length (see RunEPFollower).
type epDecodeOp byte

const (
	epOpPrefill  epDecodeOp = 1
	epOpStep     epDecodeOp = 2
	epOpShutdown epDecodeOp = 3
)

// encodeEPDecodeFrame serializes a coordinated-decode frame. Token ids and the position are
// written as little-endian uint32 (round-tripped through int32 on decode so a negative
// sentinel survives), matching the fixed-width, fail-closed discipline of the DistComm wire
// codecs.
func encodeEPDecodeFrame(op epDecodeOp, pos int, ids []int) []byte {
	buf := make([]byte, 9+4*len(ids))
	buf[0] = byte(op)
	binary.LittleEndian.PutUint32(buf[1:5], uint32(int32(pos)))
	binary.LittleEndian.PutUint32(buf[5:9], uint32(len(ids)))
	for i, id := range ids {
		binary.LittleEndian.PutUint32(buf[9+4*i:13+4*i], uint32(int32(id)))
	}
	return buf
}

// decodeEPDecodeFrame parses a frame produced by encodeEPDecodeFrame, failing closed on a
// truncated buffer, an unknown op, a negative position, or an op/count mismatch — a desynced
// follower must error rather than silently mis-drive its forward.
func decodeEPDecodeFrame(payload []byte) (epDecodeOp, int, []int, error) {
	if len(payload) < 9 {
		return 0, 0, nil, fmt.Errorf("model: EP decode frame too short: %d bytes, want >=9", len(payload))
	}
	op := epDecodeOp(payload[0])
	pos := int(int32(binary.LittleEndian.Uint32(payload[1:5])))
	count := int(binary.LittleEndian.Uint32(payload[5:9]))
	if pos < 0 {
		return 0, 0, nil, fmt.Errorf("model: EP decode frame pos=%d, want >=0", pos)
	}
	if count < 0 || 9+4*count != len(payload) {
		return 0, 0, nil, fmt.Errorf("model: EP decode frame len=%d, want %d for count=%d", len(payload), 9+4*count, count)
	}
	switch op {
	case epOpPrefill:
	case epOpStep:
		if count != 1 {
			return 0, 0, nil, fmt.Errorf("model: EP STEP frame count=%d, want 1", count)
		}
	case epOpShutdown:
		if count != 0 {
			return 0, 0, nil, fmt.Errorf("model: EP SHUTDOWN frame count=%d, want 0", count)
		}
	default:
		return 0, 0, nil, fmt.Errorf("model: EP decode frame unknown op %d", op)
	}
	ids := make([]int, count)
	for i := range ids {
		ids[i] = int(int32(binary.LittleEndian.Uint32(payload[9+4*i : 13+4*i])))
	}
	return op, pos, ids, nil
}

// EPDecodeCoordinator is rank 0's half of the coordinated EP decode. Installing it on a Model
// (SetEPDecodeCoordinator) makes every Prefill/Step that model's sessions run announce itself
// to the follower ranks first, so rank 0's ORDINARY decode loop — whatever sampler, prefix
// cache, or stop rule it uses — drives the group. It must be built on rank 0; every other rank
// runs RunEPFollower. Call Shutdown after the last request to release the followers.
type EPDecodeCoordinator struct {
	g *DistComm
	// mu serializes the (broadcast frame, run forward) critical section. Holding it ACROSS the
	// forward is the point: two concurrent rank-0 requests would otherwise interleave their
	// frames and their AllReduces, and followers — which see one flat frame stream with no
	// request id — could not tell the two apart. Serializing is the honest bound for this
	// slice; real concurrency >1 needs a request id in the frame (#4835, still open).
	mu sync.Mutex
}

// NewEPDecodeCoordinator builds rank 0's coordinator over the process group that already
// carries the forward's collectives. It fails closed off rank 0 — a follower that installed a
// coordinator would broadcast frames INTO the root and desync the whole group.
func NewEPDecodeCoordinator(g *DistComm) (*EPDecodeCoordinator, error) {
	if g == nil {
		return nil, fmt.Errorf("model: NewEPDecodeCoordinator needs a process group, got nil")
	}
	if g.Rank() != 0 {
		return nil, fmt.Errorf("model: NewEPDecodeCoordinator must run on rank 0, got rank %d", g.Rank())
	}
	return &EPDecodeCoordinator{g: g}, nil
}

// announce broadcasts one frame and returns the release func the caller must run after its
// forward completes. A transport failure here is unrecoverable by construction: the followers
// are now at a different point in the frame stream than rank 0, so the very next per-layer
// AllReduce would block forever. Panicking turns that silent hang into a loud, honest
// boundary — the same discipline requirePreNorm uses for an unimplemented topology.
func (c *EPDecodeCoordinator) announce(op epDecodeOp, pos int, ids []int) func() {
	c.mu.Lock()
	if _, err := c.g.BroadcastFromRoot(encodeEPDecodeFrame(op, pos, ids)); err != nil {
		c.mu.Unlock()
		panic(fmt.Sprintf("model: EP decode coordinator broadcast (op=%d pos=%d n=%d) failed, the process group is desynced and every following collective would hang: %v", op, pos, len(ids), err))
	}
	return c.mu.Unlock
}

// Shutdown releases every RunEPFollower by broadcasting a single SHUTDOWN frame. It is safe to
// call once, after the last request the group will serve.
func (c *EPDecodeCoordinator) Shutdown() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, err := c.g.BroadcastFromRoot(encodeEPDecodeFrame(epOpShutdown, 0, nil)); err != nil {
		return fmt.Errorf("model: EP decode coordinator shutdown broadcast: %w", err)
	}
	return nil
}

// SetEPDecodeCoordinator installs (or with nil clears) rank 0's coordinated-decode driver on
// this model. When set, every Prefill/Step on a session of this model first broadcasts the
// forward it is about to run to the follower ranks, so the followers replay it and reach the
// same collectives. nil — the default, and every non-sharded serve — leaves the forward
// entry points byte-identical to today: the hook reads one nil field and returns.
func (m *Model) SetEPDecodeCoordinator(c *EPDecodeCoordinator) { m.epCoord = c }

// EPDecodeCoordinated reports whether this model decodes through a coordinated EP group —
// whether every Prefill/Step on its sessions is announced to follower ranks that replay it.
// It reads the same field epAnnounce does, so it can never disagree with what the forward
// actually did; it does not latch, so clearing the coordinator clears the answer.
//
// The decode driver above this package keys KV PREFIX REUSE off it (#5553). Restoring a
// cached prefix means rank 0 runs ZERO forwards over the matched tokens, so its next PREFILL
// is announced at a position the followers' fresh mirrors never computed and epCheckMirror
// fails closed — correctly, because letting it through would reduce partials computed from
// different contexts. Reuse must therefore be off exactly while this is true.
//
// It is nil-safe on purpose: a planner built before a model finishes loading asks the
// question with a nil *Model, and answering "coordinated" there would drop prefix reuse for
// every ordinary single-process serve — which installs no coordinator and reports false.
func (m *Model) EPDecodeCoordinated() bool { return m != nil && m.epCoord != nil }

// epSeqLen is the sequence position this session's next forward runs at. A device (HAL)
// session decodes from halKV, every other session from the host KVCache — the coordinated
// frame must carry whichever one the forward will actually append to, because that is the
// number the follower's mirror is checked against.
func (s *Session) epSeqLen() int {
	if s.halKV != nil {
		return s.halKV.Len()
	}
	if s.Cache == nil {
		return 0
	}
	return s.Cache.Len()
}

// epAnnounce is the Session-side coordination hook called at the top of Prefill and Step. It
// returns nil (and does nothing at all) unless a coordinator is installed, which is the case
// for every ordinary serve; on rank 0 of a coordinated sharded serve it broadcasts the frame
// and returns the release the caller defers until after its forward.
func (s *Session) epAnnounce(op epDecodeOp, ids []int) func() {
	if s == nil || s.M == nil || s.M.epCoord == nil {
		return nil
	}
	return s.M.epCoord.announce(op, s.epSeqLen(), ids)
}

// RunEPFollower runs a follower rank's (rank>0) half of the coordinated decode: it receives
// broadcast frames from rank 0 and contributes ONLY its local expert compute + collectives. It
// never tokenizes, never samples, never builds a response, and discards every Prefill/Step
// logit — the follower structurally cannot own the token history. newSession mints this rank's
// mirror session and MUST match the kind of session rank 0 decodes with (a device serve passes
// a backend session), because the mirror has to run the same forward.
//
// It returns nil when rank 0 broadcasts SHUTDOWN. It fails closed on any frame or transport
// error and, critically, on a POSITION MISMATCH: if rank 0's session resumed from a restored
// prefix-cache snapshot it will announce a prefill at a position the follower's fresh mirror
// never computed, and the follower's hidden states would then be wrong at every layer — the
// AllReduce would sum a garbage partial into a plausible-looking answer. That is the one
// failure this protocol must never take silently, so it is an error, not a warning. The serve
// wiring must therefore disable prefix reuse on the coordinated path (#4835).
func RunEPFollower(g *DistComm, newSession func() *Session) error {
	if g == nil {
		return fmt.Errorf("model: RunEPFollower needs a process group, got nil")
	}
	if g.Rank() == 0 {
		return fmt.Errorf("model: RunEPFollower must run on a follower rank (>0), got rank 0")
	}
	if newSession == nil {
		return fmt.Errorf("model: RunEPFollower needs a session factory, got nil")
	}
	var sess *Session
	for {
		payload, err := g.BroadcastFromRoot(nil)
		if err != nil {
			return fmt.Errorf("model: EP follower rank %d recv frame: %w", g.Rank(), err)
		}
		op, pos, ids, err := decodeEPDecodeFrame(payload)
		if err != nil {
			return err
		}
		switch op {
		case epOpPrefill:
			// pos 0 opens a new request: rank 0 started a fresh session, so the mirror does too.
			if pos == 0 {
				sess = newSession()
			}
			if err := epCheckMirror(g, sess, op, pos); err != nil {
				return err
			}
			_ = sess.Prefill(ids)
		case epOpStep:
			if err := epCheckMirror(g, sess, op, pos); err != nil {
				return err
			}
			_ = sess.Step(ids[0])
		case epOpShutdown:
			return nil
		}
	}
}

// epCheckMirror is the follower's fail-closed guard that its mirror session sits at exactly the
// position rank 0 is about to run at. A mismatch means the two ranks would feed different
// hidden states into the same AllReduce.
func epCheckMirror(g *DistComm, sess *Session, op epDecodeOp, pos int) error {
	if sess == nil {
		return fmt.Errorf("model: EP follower rank %d got op=%d at pos %d before any PREFILL opened a mirror session", g.Rank(), op, pos)
	}
	if got := sess.epSeqLen(); got != pos {
		return fmt.Errorf("model: EP follower rank %d mirror desync — rank 0 runs op=%d at pos %d but this rank's mirror is at pos %d; the reduce would sum a partial computed from a different context (rank 0 must not resume a coordinated session from a restored prefix cache)", g.Rank(), op, pos, got)
	}
	return nil
}
