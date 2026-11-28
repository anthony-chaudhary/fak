package compute

// Lamport -0.0 sentinel slot all-reduce push-plane protocol.
//
// ADAPT of sgl-project/sglang@f8aa131db5885f8c331909b000b6edf56a682a05
// (python/sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh:145-205,
// Apache-2.0): producers encode +0.0 payload words to -0.0 (kNegZeroF16=0x8000,
// kNegZeroF32=0x80000000) so written zeros are distinguishable from untouched
// all-zero slot words without a separate arrival flags buffer; consumers poll
// slots until no empty sentinel word remains, reduce across ranks, and
// canonicalize -0.0 back to +0.0 on output.
//
// Distinct from doorbell_allreduce.go (flag-word doorbell) and fused GEMM+AR:
// this is the sentinel-in-payload 3-slot Lamport ring protocol.
// Clean-room host contract and concurrent atomic oracle only ([SW-VERIFIED]):
// no P2P hardware use or speed claim.

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync/atomic"
	"time"
)

const (
	// KNegZeroF16 is the IEEE-754 binary16 / bfloat16 -0.0 bit pattern used as the
	// canonical non-sentinel representation of zero in 16-bit Lamport slots.
	KNegZeroF16 uint16 = 0x8000
	// KNegZeroF32 is the IEEE-754 binary32 -0.0 bit pattern used as the
	// canonical non-sentinel representation of zero in 32-bit Lamport slots.
	KNegZeroF32 uint32 = 0x80000000
	// LamportEmptySentinelF16 is the untouched/cleared 16-bit slot word (+0.0 bits).
	LamportEmptySentinelF16 uint16 = 0x0000
	// LamportEmptySentinelF32 is the untouched/cleared 32-bit slot word (+0.0 bits).
	LamportEmptySentinelF32 uint32 = 0x00000000
	// LamportRingSlots is the 3-slot Lamport ring depth per (src, dst) peer link.
	LamportRingSlots = 3
)

var (
	// ErrLamportInvalidArgs is the closed refusal for invalid rank count, rank ID,
	// epoch, element count, or ragged rank payload lengths.
	ErrLamportInvalidArgs = errors.New("compute: lamport allreduce invalid rank or shape")
	// ErrLamportSlotIncomplete is the closed refusal when polling times out or a
	// peer slot still contains an untouched empty sentinel word.
	ErrLamportSlotIncomplete = errors.New("compute: lamport allreduce slot still contains empty sentinel")
)

// EncodeLamportF32 encodes a float32 payload word for a Lamport slot: +0.0
// (0x00000000) becomes -0.0 (KNegZeroF32 = 0x80000000) so no written word
// matches the empty sentinel LamportEmptySentinelF32.
func EncodeLamportF32(v float32) uint32 {
	bits := math.Float32bits(v)
	if bits == LamportEmptySentinelF32 {
		return KNegZeroF32
	}
	return bits
}

// DecodeLamportF32 decodes a Lamport slot word. If the word is the untouched
// empty sentinel (0x00000000), empty is true. Otherwise -0.0 canonicalizes back
// to +0.0 and empty is false.
func DecodeLamportF32(word uint32) (val float32, empty bool) {
	if word == LamportEmptySentinelF32 {
		return 0, true
	}
	if word == KNegZeroF32 {
		return 0.0, false
	}
	return math.Float32frombits(word), false
}

// CanonicalizeLamportF32 maps -0.0 to +0.0 while leaving every other float32 bit
// pattern (including NaN payloads) untouched.
func CanonicalizeLamportF32(v float32) float32 {
	if math.Float32bits(v) == KNegZeroF32 {
		return 0.0
	}
	return v
}

// EncodeLamportF16 encodes a 16-bit (FP16/BF16) payload word for a Lamport slot,
// rewriting 0x0000 (+0.0) to KNegZeroF16 (0x8000).
func EncodeLamportF16(bits uint16) uint16 {
	if bits == LamportEmptySentinelF16 {
		return KNegZeroF16
	}
	return bits
}

// DecodeLamportF16 decodes a 16-bit Lamport slot word, reporting empty=true when
// word == LamportEmptySentinelF16 and canonicalizing KNegZeroF16 back to 0x0000.
func DecodeLamportF16(word uint16) (bits uint16, empty bool) {
	if word == LamportEmptySentinelF16 {
		return 0, true
	}
	if word == KNegZeroF16 {
		return 0x0000, false
	}
	return word, false
}

// LamportSlotBuffer stores one peer's payload in atomic uint32 slot words with
// NO separate arrival-flag buffer: an all-zero word marks an unwritten slot.
type LamportSlotBuffer struct {
	words []atomic.Uint32
}

// NewLamportSlotBuffer allocates an untouched slot buffer of length count,
// initialized to LamportEmptySentinelF32.
func NewLamportSlotBuffer(count int) (*LamportSlotBuffer, error) {
	if count <= 0 {
		return nil, fmt.Errorf("%w: slot count=%d", ErrLamportInvalidArgs, count)
	}
	return &LamportSlotBuffer{words: make([]atomic.Uint32, count)}, nil
}

// Len returns the number of element slots in the buffer.
func (b *LamportSlotBuffer) Len() int {
	if b == nil {
		return 0
	}
	return len(b.words)
}

// WriteF32 encodes src (+0.0 -> -0.0) and stores every element word atomically.
func (b *LamportSlotBuffer) WriteF32(src []float32) error {
	if b == nil || len(src) != len(b.words) {
		got := 0
		if b != nil {
			got = len(b.words)
		}
		return fmt.Errorf("%w: write len=%d want %d", ErrLamportInvalidArgs, len(src), got)
	}
	for i, v := range src {
		b.words[i].Store(EncodeLamportF32(v))
	}
	return nil
}

// WriteEncodedWord stores one raw encoded uint32 word at index idx (exposed for
// testing partial-write / incomplete-peer detection).
func (b *LamportSlotBuffer) WriteEncodedWord(idx int, word uint32) error {
	if b == nil || idx < 0 || idx >= len(b.words) {
		return fmt.Errorf("%w: index %d out of bounds", ErrLamportInvalidArgs, idx)
	}
	b.words[idx].Store(word)
	return nil
}

// Ready reports true iff every word in the slot buffer is non-sentinel (!= 0).
func (b *LamportSlotBuffer) Ready() bool {
	if b == nil || len(b.words) == 0 {
		return false
	}
	for i := range b.words {
		if b.words[i].Load() == LamportEmptySentinelF32 {
			return false
		}
	}
	return true
}

// ReadF32 reads and canonicalizes all elements when no sentinel word remains,
// returning ErrLamportSlotIncomplete if any word still holds the empty sentinel.
func (b *LamportSlotBuffer) ReadF32() ([]float32, error) {
	if b == nil || len(b.words) == 0 {
		return nil, fmt.Errorf("%w: empty buffer", ErrLamportInvalidArgs)
	}
	out := make([]float32, len(b.words))
	for i := range b.words {
		w := b.words[i].Load()
		val, empty := DecodeLamportF32(w)
		if empty {
			return nil, fmt.Errorf("%w: word %d still holds empty sentinel", ErrLamportSlotIncomplete, i)
		}
		out[i] = val
	}
	return out, nil
}

// Reset clears every slot word back to LamportEmptySentinelF32 for ring reuse.
func (b *LamportSlotBuffer) Reset() {
	if b == nil {
		return
	}
	for i := range b.words {
		b.words[i].Store(LamportEmptySentinelF32)
	}
}

// LamportCommunicator models a multi-rank 3-slot Lamport push-plane ring
// without separate arrival flags or NCCL barriers.
type LamportCommunicator struct {
	ranks int
	count int
	// slots[dstRank][ringSlot][srcRank] holds srcRank's pushed payload for dstRank.
	slots [][][]*LamportSlotBuffer
}

// NewLamportCommunicator initializes a Lamport ring for ranks peers and count
// float32 elements per all-reduce vector.
func NewLamportCommunicator(ranks, count int) (*LamportCommunicator, error) {
	if ranks <= 0 || count <= 0 {
		return nil, fmt.Errorf("%w: ranks=%d count=%d", ErrLamportInvalidArgs, ranks, count)
	}
	slots := make([][][]*LamportSlotBuffer, ranks)
	for dst := 0; dst < ranks; dst++ {
		slots[dst] = make([][]*LamportSlotBuffer, LamportRingSlots)
		for s := 0; s < LamportRingSlots; s++ {
			slots[dst][s] = make([]*LamportSlotBuffer, ranks)
			for src := 0; src < ranks; src++ {
				buf, err := NewLamportSlotBuffer(count)
				if err != nil {
					return nil, err
				}
				slots[dst][s][src] = buf
			}
		}
	}
	return &LamportCommunicator{
		ranks: ranks,
		count: count,
		slots: slots,
	}, nil
}

func (c *LamportCommunicator) checkRankEpoch(rank, epoch int) (int, error) {
	if c == nil || rank < 0 || rank >= c.ranks || epoch < 0 {
		return 0, fmt.Errorf("%w: rank=%d epoch=%d", ErrLamportInvalidArgs, rank, epoch)
	}
	return epoch % LamportRingSlots, nil
}

// InboundSlot returns the underlying slot buffer where srcRank writes for dstRank
// at epoch, enabling partial-write observation in unit tests.
func (c *LamportCommunicator) InboundSlot(dstRank, srcRank, epoch int) (*LamportSlotBuffer, error) {
	slotIdx, err := c.checkRankEpoch(dstRank, epoch)
	if err != nil {
		return nil, err
	}
	if srcRank < 0 || srcRank >= c.ranks {
		return nil, fmt.Errorf("%w: srcRank=%d", ErrLamportInvalidArgs, srcRank)
	}
	return c.slots[dstRank][slotIdx][srcRank], nil
}

// Push writes srcRank's payload (+0.0 -> -0.0 encoded) directly into every
// destination rank's ring slot for epoch.
func (c *LamportCommunicator) Push(srcRank, epoch int, payload []float32) error {
	slotIdx, err := c.checkRankEpoch(srcRank, epoch)
	if err != nil {
		return err
	}
	if len(payload) != c.count {
		return fmt.Errorf("%w: payload len=%d want %d", ErrLamportInvalidArgs, len(payload), c.count)
	}
	for dst := 0; dst < c.ranks; dst++ {
		if err := c.slots[dst][slotIdx][srcRank].WriteF32(payload); err != nil {
			return err
		}
	}
	return nil
}

// PollReady reports whether all peer inbound slots for (dstRank, epoch) contain
// non-sentinel words.
func (c *LamportCommunicator) PollReady(dstRank, epoch int) bool {
	slotIdx, err := c.checkRankEpoch(dstRank, epoch)
	if err != nil {
		return false
	}
	for src := 0; src < c.ranks; src++ {
		if !c.slots[dstRank][slotIdx][src].Ready() {
			return false
		}
	}
	return true
}

// ConsumeAllReduce polls dstRank's inbound slots for epoch until all peers have
// written non-sentinel words (or timeout expires), performs rank-order sum
// reduction, canonicalizes -0.0 -> +0.0 on output, and resets the consumed slots
// back to the empty sentinel for future ring epochs.
func (c *LamportCommunicator) ConsumeAllReduce(dstRank, epoch int, timeout time.Duration) ([]float32, error) {
	slotIdx, err := c.checkRankEpoch(dstRank, epoch)
	if err != nil {
		return nil, err
	}
	deadline := time.Now().Add(timeout)
	for !c.PollReady(dstRank, epoch) {
		if timeout <= 0 || time.Now().After(deadline) {
			return nil, fmt.Errorf("%w: dstRank=%d epoch=%d timed out", ErrLamportSlotIncomplete, dstRank, epoch)
		}
		runtime.Gosched()
	}

	acc := make([]float32, c.count)
	for src := 0; src < c.ranks; src++ {
		vals, err := c.slots[dstRank][slotIdx][src].ReadF32()
		if err != nil {
			return nil, err
		}
		if src == 0 {
			copy(acc, vals)
		} else {
			for i := 0; i < c.count; i++ {
				acc[i] += vals[i]
			}
		}
	}
	for i := range acc {
		acc[i] = CanonicalizeLamportF32(acc[i])
	}
	for src := 0; src < c.ranks; src++ {
		c.slots[dstRank][slotIdx][src].Reset()
	}
	return acc, nil
}

// LamportAllreduceF32 executes the full push-and-consume Lamport sentinel slot
// protocol across len(parts) ranks on one fixed epoch and returns the canonicalized
// rank-order sum.
func LamportAllreduceF32(parts [][]float32) ([]float32, error) {
	if len(parts) == 0 || len(parts[0]) == 0 {
		return nil, fmt.Errorf("%w: empty parts", ErrLamportInvalidArgs)
	}
	n := len(parts[0])
	for r, p := range parts {
		if len(p) != n {
			return nil, fmt.Errorf("%w: rank %d len=%d want %d", ErrLamportInvalidArgs, r, len(p), n)
		}
	}
	if len(parts) == 1 {
		out := make([]float32, n)
		for i, v := range parts[0] {
			out[i] = CanonicalizeLamportF32(v)
		}
		return out, nil
	}

	comm, err := NewLamportCommunicator(len(parts), n)
	if err != nil {
		return nil, err
	}
	for r, p := range parts {
		if err := comm.Push(r, 0, p); err != nil {
			return nil, err
		}
	}
	return comm.ConsumeAllReduce(0, 0, time.Second)
}

// LamportAllreduceF16Bits executes the 16-bit (-0.0 = KNegZeroF16 = 0x8000)
// Lamport sentinel slot protocol over IEEE-754 binary16 payloads, accumulating in
// float32 in rank order and returning canonicalized binary16 words (-0.0 -> +0.0).
func LamportAllreduceF16Bits(parts [][]uint16) ([]uint16, error) {
	if len(parts) == 0 || len(parts[0]) == 0 {
		return nil, fmt.Errorf("%w: empty f16 parts", ErrLamportInvalidArgs)
	}
	ranks := len(parts)
	n := len(parts[0])
	for r, p := range parts {
		if len(p) != n {
			return nil, fmt.Errorf("%w: rank %d f16 len=%d want %d", ErrLamportInvalidArgs, r, len(p), n)
		}
	}

	// Simulate per-peer 16-bit slot buffer initialized to LamportEmptySentinelF16 (0x0000).
	slots := make([][]uint16, ranks)
	for r := 0; r < ranks; r++ {
		slots[r] = make([]uint16, n)
		for i, bits := range parts[r] {
			slots[r][i] = EncodeLamportF16(bits)
		}
	}

	out := make([]uint16, n)
	for i := 0; i < n; i++ {
		var sum float32
		for r := 0; r < ranks; r++ {
			bits, empty := DecodeLamportF16(slots[r][i])
			if empty {
				return nil, fmt.Errorf("%w: f16 rank %d idx %d empty", ErrLamportSlotIncomplete, r, i)
			}
			sum += fp16ToFloat32(bits)
		}
		encoded := k8v4Float32ToFP16RN(CanonicalizeLamportF32(sum))
		if encoded == KNegZeroF16 {
			encoded = 0x0000
		}
		out[i] = encoded
	}
	return out, nil
}
