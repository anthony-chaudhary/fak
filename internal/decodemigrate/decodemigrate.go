// Package decodemigrate is the pure, CPU-witnessable core of cross-machine
// deterministic decode migration (#4307, parent #4296): move an in-flight
// generation from one instance to another mid-stream and resume it BIT-EXACTLY,
// so the post-migration continuation is identical to a run that never migrated.
//
// The whole model/GPU/network layer is out of scope here. What this package
// isolates is the one contract that makes such a hop safe: a decode step's
// resumable state (the pinned seed, the running PRNG stream state, a fold of the
// entire prior draw history, and the step index) can be CAPTURED into a
// checkpoint, serialized to a deterministic byte string, teleported to a fresh
// instance, and RESTORED so continued decoding reproduces the exact same token
// stream. The carried-history fold (acc) is load-bearing: a migration that drops
// any checkpoint field decodes differently from the first post-migration step,
// which is exactly the corruption class a live drain/teleport must never hit.
//
// Everything here is deterministic and wall-clock-free: identical seed in,
// byte-identical stream out, on any machine.
package decodemigrate

import (
	"encoding/binary"
	"fmt"
)

// vocabSize is the token space the mock decoder emits ids from. It is small and
// fixed so a stream is easy to eyeball, while the carried-history fold keeps every
// token dependent on the whole prior sequence rather than just (seed, step).
const vocabSize = 128

// splitmixGamma is the golden-ratio odd increment the stream state advances by
// each step (the standard splitmix64 stride).
const splitmixGamma uint64 = 0x9e3779b97f4a7c15

// historyPrime is the multiplier the running fold uses so each new draw mixes the
// entire prior history back in (an FNV-style rolling accumulator).
const historyPrime uint64 = 0x100000001b3

// DecodeStream is one in-flight generation's resumable decode state on a single
// instance. It is a deterministic mock of a real sampler loop: a splitmix64 stream
// advanced by a fixed stride each step, a running fold of every prior draw so the
// next token depends on the full history, and the step counter. No model, GPU, or
// network — just the state a migration must carry intact.
type DecodeStream struct {
	seed int64  // pinned request param; reproduces the step-zero state
	rng  uint64 // splitmix64 stream state, advanced each step
	acc  uint64 // running fold of every prior draw (carried history)
	step uint64 // steps taken so far
}

// NewDecodeStream returns the decode state at step zero for a pinned seed. Two
// calls with the same seed produce byte-identical streams forever after.
func NewDecodeStream(seed int64) *DecodeStream {
	return &DecodeStream{
		seed: seed,
		rng:  uint64(seed)*splitmixGamma + 0x243f6a8885a308d3,
	}
}

// Next advances the stream one step and returns the emitted token id in
// [0,vocabSize). The stream state advances by the golden stride, mixes with the
// carried history fold, the splitmix finalizer scrambles it, and the finalized
// draw both selects the token and folds back into the history accumulator.
func (s *DecodeStream) Next() int {
	s.rng += splitmixGamma
	z := s.rng ^ s.acc
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	z ^= z >> 31
	s.acc = s.acc*historyPrime + z
	s.step++
	return int(z % vocabSize)
}

// Step reports how many tokens this stream has drawn so far.
func (s *DecodeStream) Step() uint64 { return s.step }

// Checkpoint is the complete, serializable resumable state of a DecodeStream at an
// instant — the payload a migration teleports to another instance. Every field is
// required: omitting any one produces a different restored stream.
type Checkpoint struct {
	Seed int64
	Rng  uint64
	Acc  uint64
	Step uint64
}

// Capture snapshots the live decode state into a Checkpoint. The source stream is
// left untouched, so an instance can checkpoint without pausing (the migration
// decides when the source stops advancing).
func (s *DecodeStream) Capture() Checkpoint {
	return Checkpoint{Seed: s.seed, Rng: s.rng, Acc: s.acc, Step: s.step}
}

// RestoreDecodeStream rehydrates a FRESH decode instance from a checkpoint — the
// destination side of the hop. The returned stream shares no memory with the
// source; only what round-tripped through the checkpoint survives, as in a real
// process on another box.
func RestoreDecodeStream(cp Checkpoint) *DecodeStream {
	return &DecodeStream{seed: cp.Seed, rng: cp.Rng, acc: cp.Acc, step: cp.Step}
}

// checkpointBytes is the exact serialized length of a Checkpoint: four 64-bit
// fields, big-endian.
const checkpointBytes = 32

// Marshal serializes a checkpoint to a deterministic, fixed-length byte string —
// the wire form the teleport carries. Big-endian and field-ordered so the same
// checkpoint always produces the same bytes on any machine.
func (cp Checkpoint) Marshal() []byte {
	b := make([]byte, checkpointBytes)
	binary.BigEndian.PutUint64(b[0:8], uint64(cp.Seed))
	binary.BigEndian.PutUint64(b[8:16], cp.Rng)
	binary.BigEndian.PutUint64(b[16:24], cp.Acc)
	binary.BigEndian.PutUint64(b[24:32], cp.Step)
	return b
}

// UnmarshalCheckpoint reconstructs a checkpoint from its wire bytes. It refuses a
// payload of the wrong length rather than guessing — a truncated teleport restored
// as zeroes would silently reproduce the reseed-from-scratch corruption.
func UnmarshalCheckpoint(b []byte) (Checkpoint, error) {
	if len(b) != checkpointBytes {
		return Checkpoint{}, fmt.Errorf("decodemigrate: checkpoint is %d bytes, want %d", len(b), checkpointBytes)
	}
	return Checkpoint{
		Seed: int64(binary.BigEndian.Uint64(b[0:8])),
		Rng:  binary.BigEndian.Uint64(b[8:16]),
		Acc:  binary.BigEndian.Uint64(b[16:24]),
		Step: binary.BigEndian.Uint64(b[24:32]),
	}, nil
}

// Migrate performs the full hop for a source stream: capture its live state,
// serialize it to the wire form, teleport (here: the bytes themselves), and
// rehydrate a fresh destination instance from those bytes. The returned stream is
// the one another box would resume decoding on; continued Next calls reproduce the
// source's would-be stream bit-for-bit.
func Migrate(src *DecodeStream) (*DecodeStream, error) {
	wire := src.Capture().Marshal()
	cp, err := UnmarshalCheckpoint(wire)
	if err != nil {
		return nil, err
	}
	return RestoreDecodeStream(cp), nil
}
