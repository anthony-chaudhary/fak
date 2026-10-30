//go:build cuda && cgo

package compute

import (
	"encoding/binary"
	"math"
	"sort"
	"testing"
)

// v41GreedyHeadFixture is a random Q6_K head with finite non-power-of-two half
// scales, so neighbouring rows do not tie.
func v41GreedyHeadFixture(out, in int, seed uint32) []byte {
	const blockBytes = 210
	raw := make([]byte, out*(in/256)*blockBytes)
	next := func() uint32 {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		return seed
	}
	for block := 0; block < len(raw)/blockBytes; block++ {
		b := raw[block*blockBytes : (block+1)*blockBytes]
		for i := 0; i < 192; i++ {
			b[i] = byte(next())
		}
		for i := 0; i < 16; i++ {
			b[192+i] = byte(int8(next()%61) - 30)
		}
		binary.LittleEndian.PutUint16(b[208:], uint16(0x1c00+next()%0x3ff))
	}
	return raw
}

// TestCUDAV41GreedyHeadParity witnesses the device greedy head at the pinned
// Flash geometry (Q6_K output.weight [129280,5120]): the resident Q6_K GEMV
// followed by the device argmax returns the host reference's greedy id, the
// top-5 logits match it, and only the 4-byte id crosses host-ward.
// fak-test:runtime slow est=20s lane=cuda
func TestCUDAV41GreedyHeadParity(t *testing.T) {
	cb := cudaOrSkip(t)
	const vocab, hidden = 129280, 5120
	raw := v41GreedyHeadFixture(vocab, hidden, 0x9e3779b9)
	w := cb.Upload(NewQ6K(Default(), []int{vocab, hidden}, raw), Q6_K)
	defer cb.Free(w)
	rowBytes := hidden / 256 * 210
	deq := make([]float32, hidden)
	for trial := 0; trial < 3; trial++ {
		x := q6KSharedHeadInput(trial+1, hidden)[trial*hidden:]
		ref := make([]float64, vocab)
		for r := 0; r < vocab; r++ {
			for blk := 0; blk < hidden/256; blk++ {
				dequantQ6K(deq[blk*256:(blk+1)*256], raw[r*rowBytes+blk*210:r*rowBytes+(blk+1)*210])
			}
			var acc float64
			for c, v := range deq {
				acc += float64(v) * float64(x[c])
			}
			ref[r] = acc
		}
		order := make([]int, vocab)
		for i := range order {
			order[i] = i
		}
		sort.SliceStable(order, func(a, b int) bool { return ref[order[a]] > ref[order[b]] })
		if gap := ref[order[0]] - ref[order[1]]; gap < 1e-3*math.Abs(ref[order[0]]) {
			t.Fatalf("trial %d: fixture top-1 gap %g too small for an F32 parity claim", trial, gap)
		}

		dx := mkResident(cb, []int{hidden}, x)
		logits := cb.MatMul(w, dx)
		cb.ResetHostXfer()
		id := cb.Argmax(logits)
		if moved := cb.HostXferBytes(); moved != 4 {
			t.Fatalf("trial %d: argmax moved %d bytes host-ward, want 4", trial, moved)
		}
		if id != order[0] {
			t.Fatalf("trial %d: device greedy id %d, host reference %d", trial, id, order[0])
		}
		got := cb.Read(logits) // diagnostic read for the top-5 check, outside the measured window
		for k := 0; k < 5; k++ {
			r := order[k]
			if d := math.Abs(float64(got[r]) - ref[r]); d > 1e-4*math.Max(1, math.Abs(ref[r])) {
				t.Fatalf("trial %d: top-%d row %d logit %g, host %g (|d|=%g)", trial, k+1, r, got[r], ref[r], d)
			}
		}
		t.Logf("trial %d: greedy id %d logit %.6f (host %.6f), top-5 %v", trial, id, got[id], ref[id], order[:5])
		cb.Free(logits)
		cb.Free(dx)
	}
}
