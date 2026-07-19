package model

import (
	"encoding/binary"
	"fmt"
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
// Ordering safety: a DistComm runs one collective at a time in the same order on every
// rank (dist_collective.go). Rank 0 broadcasts a frame and THEN runs its forward; each
// follower receives the frame and THEN runs the SAME forward (same token ids => same
// routing => same number and order of AllReduces). So the broadcast rounds and the
// forward's internal AllReduces interleave identically on all ranks — no new deadlock
// class. Because every rank sees the same ids in the same order, the per-layer AllReduce
// keeps all hidden states aligned automatically, and rank 0's sampled tokens are, by
// construction, bit-identical to Session.Generate on the same session.
//
// Slice 1 (#4835) is this protocol + its regression witness, entirely in internal/model.
// The live-serve wiring (follower ranks run epFollowerDecode instead of the HTTP
// mirror; rank 0 drives epFrontDecode) is a separate slice in cmd/fak + internal/gateway.
// The end-to-end tok/s artifact on the 8-GPU checkpoint stays GPU-gated.

// epDecodeOp tags a coordinated-decode frame. The wire form is
// [op:1][count:4 LE uint32][count x uint32 LE token id]: PREFILL carries the prompt ids,
// STEP carries exactly one sampled id, SHUTDOWN carries none.
type epDecodeOp byte

const (
	epOpPrefill  epDecodeOp = 1
	epOpStep     epDecodeOp = 2
	epOpShutdown epDecodeOp = 3
)

// encodeEPDecodeFrame serializes a coordinated-decode frame. Token ids are written as
// little-endian uint32 (round-tripped through int32 on decode so a negative sentinel
// survives), matching the fixed-width, fail-closed discipline of the DistComm wire codecs.
func encodeEPDecodeFrame(op epDecodeOp, ids []int) []byte {
	buf := make([]byte, 5+4*len(ids))
	buf[0] = byte(op)
	binary.LittleEndian.PutUint32(buf[1:5], uint32(len(ids)))
	for i, id := range ids {
		binary.LittleEndian.PutUint32(buf[5+4*i:9+4*i], uint32(int32(id)))
	}
	return buf
}

// decodeEPDecodeFrame parses a frame produced by encodeEPDecodeFrame, failing closed on a
// truncated buffer, an unknown op, or an op/count mismatch — a desynced follower must
// error rather than silently mis-drive its forward.
func decodeEPDecodeFrame(payload []byte) (epDecodeOp, []int, error) {
	if len(payload) < 5 {
		return 0, nil, fmt.Errorf("model: EP decode frame too short: %d bytes, want >=5", len(payload))
	}
	op := epDecodeOp(payload[0])
	count := int(binary.LittleEndian.Uint32(payload[1:5]))
	if count < 0 || 5+4*count != len(payload) {
		return 0, nil, fmt.Errorf("model: EP decode frame len=%d, want %d for count=%d", len(payload), 5+4*count, count)
	}
	switch op {
	case epOpPrefill:
	case epOpStep:
		if count != 1 {
			return 0, nil, fmt.Errorf("model: EP STEP frame count=%d, want 1", count)
		}
	case epOpShutdown:
		if count != 0 {
			return 0, nil, fmt.Errorf("model: EP SHUTDOWN frame count=%d, want 0", count)
		}
	default:
		return 0, nil, fmt.Errorf("model: EP decode frame unknown op %d", op)
	}
	ids := make([]int, count)
	for i := range ids {
		ids[i] = int(int32(binary.LittleEndian.Uint32(payload[5+4*i : 9+4*i])))
	}
	return op, ids, nil
}

// epFrontDecode runs rank 0's half of the coordinated EP decode: it owns sampling and the
// token history and drives every follower through the same forward via broadcast frames on
// g. It mirrors Session.Generate (greedy argmax, EOS stop) exactly, inserting one
// BroadcastFromRoot before each collective-bearing forward step so the followers'
// AllReduces stay in lockstep with rank 0's. The returned tokens are bit-identical to
// s.Generate(prompt, n) on the same session. It must run only on rank 0; every other rank
// runs epFollowerDecode. Call epShutdownFollowers after the last request to release
// the followers.
func epFrontDecode(g *DistComm, s *Session, prompt []int, n int) ([]int, error) {
	if g.Rank() != 0 {
		return nil, fmt.Errorf("model: epFrontDecode must run on rank 0, got rank %d", g.Rank())
	}
	if _, err := g.BroadcastFromRoot(encodeEPDecodeFrame(epOpPrefill, prompt)); err != nil {
		return nil, fmt.Errorf("model: epFrontDecode broadcast prefill: %w", err)
	}
	logits := s.Prefill(prompt)
	out := make([]int, 0, n)
	for i := 0; i < n; i++ {
		next := argmaxF32(logits)
		out = append(out, next)
		if s.M.Cfg.IsEOS(next) {
			break
		}
		if _, err := g.BroadcastFromRoot(encodeEPDecodeFrame(epOpStep, []int{next})); err != nil {
			return out, fmt.Errorf("model: epFrontDecode broadcast step: %w", err)
		}
		logits = s.Step(next)
	}
	return out, nil
}

// epFollowerDecode runs a follower rank's (rank>0) half of the coordinated decode: it
// receives broadcast frames from rank 0 and contributes ONLY its local expert compute +
// collectives. It never tokenizes, never samples, never builds a response, and discards
// every Prefill/Step logit — the follower structurally cannot own the token history. It
// returns nil when rank 0 broadcasts a SHUTDOWN frame (epShutdownFollowers); a STEP before
// any PREFILL, or any frame/transport error, fails closed.
func epFollowerDecode(g *DistComm, m *Model) error {
	if g.Rank() == 0 {
		return fmt.Errorf("model: epFollowerDecode must run on a follower rank (>0), got rank 0")
	}
	var sess *Session
	for {
		payload, err := g.BroadcastFromRoot(nil)
		if err != nil {
			return fmt.Errorf("model: epFollowerDecode recv frame: %w", err)
		}
		op, ids, err := decodeEPDecodeFrame(payload)
		if err != nil {
			return err
		}
		switch op {
		case epOpPrefill:
			sess = m.NewSession()
			_ = sess.Prefill(ids)
		case epOpStep:
			if sess == nil {
				return fmt.Errorf("model: epFollowerDecode got STEP before PREFILL")
			}
			_ = sess.Step(ids[0])
		case epOpShutdown:
			return nil
		}
	}
}

// epShutdownFollowers is called on rank 0 to release every epFollowerDecode: it
// broadcasts a single SHUTDOWN frame that terminates each follower's loop.
func epShutdownFollowers(g *DistComm) error {
	if g.Rank() != 0 {
		return fmt.Errorf("model: epShutdownFollowers must run on rank 0, got rank %d", g.Rank())
	}
	if _, err := g.BroadcastFromRoot(encodeEPDecodeFrame(epOpShutdown, nil)); err != nil {
		return fmt.Errorf("model: epShutdownFollowers broadcast: %w", err)
	}
	return nil
}
