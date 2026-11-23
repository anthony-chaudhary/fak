package compute

import (
	"errors"
	"math"
	"slices"
	"testing"
)

const kvCompressFusedEps = float32(1e-6)

func kvCompressFusedKV(pos, d int) float32 {
	return float32(2 * math.Sin(float64(7*pos+3*d)+0.5))
}

func kvCompressFusedScore(pos, d int) float32 {
	return float32(3 * math.Cos(float64(5*pos+11*d)))
}

func kvCompressFusedGain(width int) []float32 {
	gain := make([]float32, width)
	for d := range gain {
		gain[d] = 0.75 + 0.125*float32(d%5)
	}
	return gain
}

// newKVCompressFusedRing returns a ring whose every slot is poison, so a read
// of a slot the stream never wrote shows up in the output.
func newKVCompressFusedRing(ratio, width, capacity int) KVCompressFusedRing {
	ring := KVCompressFusedRing{Ratio: ratio, Width: width, Capacity: capacity,
		KV: make([]float32, capacity*width), Scores: make([]float32, capacity*width)}
	for i := range ring.KV {
		ring.KV[i], ring.Scores[i] = 1e30, -1e30
	}
	return ring
}

// writeKVCompressFusedRing stores positions [lo, hi) with an explicit wrapping
// cursor rather than Slot, so the test does not trust the code under test to
// place rows.
func writeKVCompressFusedRing(ring KVCompressFusedRing, lo, hi int) {
	slot := lo % ring.Capacity
	for pos := lo; pos < hi; pos++ {
		for d := 0; d < ring.Width; d++ {
			ring.KV[slot*ring.Width+d] = kvCompressFusedKV(pos, d)
			ring.Scores[slot*ring.Width+d] = kvCompressFusedScore(pos, d)
		}
		if slot++; slot == ring.Capacity {
			slot = 0
		}
	}
}

// kvCompressFusedTwoStage is the unfused reference: stage one copies the
// chunk's aligned span out of the ring into linear staging, stage two pools
// each completed group into its own row with a materialized softmax and the
// allocating V41CompressorNorm, and the rows are then copied into the output.
func kvCompressFusedTwoStage(t *testing.T, ring KVCompressFusedRing, start, length int, gain []float32) (first int, rows []float32) {
	t.Helper()
	r, w := ring.Ratio, ring.Width
	alignedStart := start - start%r
	end := start + length
	n := end - alignedStart
	stageKV := make([]float32, n*w)
	stageScore := make([]float32, n*w)
	slot := alignedStart % ring.Capacity
	for i := 0; i < n; i++ {
		copy(stageKV[i*w:(i+1)*w], ring.KV[slot*w:(slot+1)*w])
		copy(stageScore[i*w:(i+1)*w], ring.Scores[slot*w:(slot+1)*w])
		if slot++; slot == ring.Capacity {
			slot = 0
		}
	}
	var compressed [][]float32
	for g := 0; (g+1)*r <= n; g++ {
		pooled := make([]float32, w)
		for d := 0; d < w; d++ {
			if r == 1 {
				pooled[d] = stageKV[g*w+d]
				continue
			}
			weights := make([]float32, r)
			m := float32(math.Inf(-1))
			for tok := 0; tok < r; tok++ {
				m = max(m, stageScore[(g*r+tok)*w+d])
			}
			var denom float32
			for tok := 0; tok < r; tok++ {
				weights[tok] = float32(math.Exp(float64(stageScore[(g*r+tok)*w+d] - m)))
				denom += weights[tok]
			}
			var acc float32
			for tok := 0; tok < r; tok++ {
				acc += float32(stageKV[(g*r+tok)*w+d] * (weights[tok] / denom))
			}
			pooled[d] = acc
		}
		normed, err := V41CompressorNorm(pooled, gain, kvCompressFusedEps)
		if err != nil {
			t.Fatalf("reference norm: %v", err)
		}
		compressed = append(compressed, normed)
	}
	rows = make([]float32, 0, len(compressed)*w)
	for _, row := range compressed {
		rows = append(rows, row...)
	}
	return alignedStart / r, rows
}

// kvCompressFusedFloat64 is an approximate oracle from the generator alone:
// float64 softmax pooling and RMSNorm over the absolute group, no ring.
func kvCompressFusedFloat64(ratio, width, group int, gain []float32) []float64 {
	pooled := make([]float64, width)
	for d := 0; d < width; d++ {
		var denom, acc float64
		for tok := 0; tok < ratio; tok++ {
			e := 1.0
			if ratio > 1 {
				e = math.Exp(float64(kvCompressFusedScore(group*ratio+tok, d)))
			}
			denom += e
			acc += e * float64(kvCompressFusedKV(group*ratio+tok, d))
		}
		pooled[d] = acc / denom
	}
	var ms float64
	for _, v := range pooled {
		ms += v * v
	}
	rstd := 1 / math.Sqrt(ms/float64(width)+float64(kvCompressFusedEps))
	for d := range pooled {
		pooled[d] *= rstd * float64(gain[d])
	}
	return pooled
}

func checkKVCompressFusedApprox(t *testing.T, chunk KVCompressFusedChunk, ratio, width int, gain []float32) {
	t.Helper()
	for g := 0; g < chunk.Groups; g++ {
		want := kvCompressFusedFloat64(ratio, width, chunk.FirstGroup+g, gain)
		for d, wv := range want {
			got := float64(chunk.Rows[g*width+d])
			// Two BF16 roundings bound the relative error by about 2^-7.
			if math.Abs(got-wv) > 1.5e-2*math.Abs(wv)+1e-3 {
				t.Fatalf("group %d dim %d: got %v, float64 oracle %v", chunk.FirstGroup+g, d, got, wv)
			}
		}
	}
}

func TestKVCompressFusedMatchesTwoStageOnChunkFixtures(t *testing.T) {
	cases := []struct {
		name                                  string
		ratio, width, capacity, start, length int
	}{
		{"ratio2 aligned", 2, 8, 16, 0, 8},
		{"ratio2 odd length leaves one pending", 2, 8, 16, 4, 7},
		{"ratio2 starts mid group", 2, 8, 16, 5, 6},
		{"ratio2 wraps the ring", 2, 8, 10, 26, 9},
		{"ratio2 chunk starts inside the straddling group", 2, 4, 6, 11, 5},
		{"ratio2 straddling group wraps from the last slot to slot zero", 2, 4, 7, 21, 4},
		{"ratio4 non-multiple length wraps", 4, 16, 12, 37, 10},
		{"ratio4 group wraps inside a ring that is no multiple of the ratio", 4, 8, 10, 49, 5},
		{"ratio4 too short to close a group", 4, 4, 8, 13, 2},
		{"ratio1 wraps, every position is a row", 1, 8, 5, 3, 5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ring := newKVCompressFusedRing(tc.ratio, tc.width, tc.capacity)
			end := tc.start + tc.length
			writeKVCompressFusedRing(ring, max(0, end-tc.capacity), end)
			before := slices.Clone(ring.KV)
			gain := kvCompressFusedGain(tc.width)

			chunk, err := KVCompressFused(ring, tc.start, tc.length, gain, kvCompressFusedEps)
			if err != nil {
				t.Fatalf("KVCompressFused: %v", err)
			}
			wantFirst, wantRows := kvCompressFusedTwoStage(t, ring, tc.start, tc.length, gain)
			if chunk.FirstGroup != wantFirst {
				t.Fatalf("FirstGroup %d, want %d", chunk.FirstGroup, wantFirst)
			}
			if !slices.Equal(chunk.Rows, wantRows) {
				t.Fatalf("fused rows differ from two-stage reference:\n got %v\nwant %v", chunk.Rows, wantRows)
			}
			// Group accounting from first principles: a group belongs to the
			// chunk holding its last position.
			closing := 0
			for pos := tc.start; pos < end; pos++ {
				if (pos+1)%tc.ratio == 0 {
					closing++
				}
			}
			if chunk.Groups != closing || len(chunk.Rows) != closing*tc.width || cap(chunk.Rows) != len(chunk.Rows) {
				t.Fatalf("groups=%d len=%d cap=%d, want %d closing groups of width %d", chunk.Groups, len(chunk.Rows), cap(chunk.Rows), closing, tc.width)
			}
			if wantPending := end - (end/tc.ratio)*tc.ratio; chunk.Pending != wantPending {
				t.Fatalf("Pending %d, want %d", chunk.Pending, wantPending)
			}
			if !slices.Equal(ring.KV, before) {
				t.Fatal("fused pass wrote into the ring")
			}
			checkKVCompressFusedApprox(t, chunk, tc.ratio, tc.width, gain)
		})
	}
}

// Streaming chunks through a wrapping ring must emit every group of the
// sequence exactly once, in order, equal to compressing the whole sequence.
func TestKVCompressFusedChunkStreamEqualsWholeSequence(t *testing.T) {
	cases := []struct {
		ratio, width, capacity int
		lengths                []int
	}{
		{2, 8, 8, []int{3, 5, 1, 4, 2, 7, 6}},
		{4, 4, 8, []int{3, 5, 2, 6, 1, 4, 7}},
		{1, 4, 8, []int{3, 5, 8, 1}},
		{2, 4, 7, []int{3, 5, 1, 4, 2, 6, 5}},
		{4, 4, 9, []int{3, 5, 2, 6, 1, 4, 5}},
	}
	for _, tc := range cases {
		ring := newKVCompressFusedRing(tc.ratio, tc.width, tc.capacity)
		gain := kvCompressFusedGain(tc.width)
		var streamed []float32
		pos, nextGroup, chunks := 0, 0, 0
		for _, length := range tc.lengths {
			writeKVCompressFusedRing(ring, pos, pos+length)
			chunk, err := KVCompressFused(ring, pos, length, gain, kvCompressFusedEps)
			if err != nil {
				t.Fatalf("ratio %d chunk at %d: %v", tc.ratio, pos, err)
			}
			chunks++
			if chunk.Groups > 0 && chunk.FirstGroup != nextGroup {
				t.Fatalf("ratio %d chunk at %d starts group %d, want %d", tc.ratio, pos, chunk.FirstGroup, nextGroup)
			}
			nextGroup += chunk.Groups
			streamed = append(streamed, chunk.Rows...)
			pos += length
		}
		if chunks != len(tc.lengths) {
			t.Fatalf("ran %d chunks, want %d", chunks, len(tc.lengths))
		}

		linear := newKVCompressFusedRing(tc.ratio, tc.width, pos)
		writeKVCompressFusedRing(linear, 0, pos)
		_, whole := kvCompressFusedTwoStage(t, linear, 0, pos, gain)
		if nextGroup != pos/tc.ratio {
			t.Fatalf("ratio %d emitted %d groups over %d positions, want %d", tc.ratio, nextGroup, pos, pos/tc.ratio)
		}
		if !slices.Equal(streamed, whole) {
			t.Fatalf("ratio %d streamed rows differ from whole-sequence compression", tc.ratio)
		}
	}
}

func TestKVCompressFusedProducesOneBufferPerChunk(t *testing.T) {
	ring := newKVCompressFusedRing(2, 16, 12)
	writeKVCompressFusedRing(ring, 21, 33)
	gain := kvCompressFusedGain(16)
	var chunk KVCompressFusedChunk
	allocs := testing.AllocsPerRun(20, func() {
		var err error
		if chunk, err = KVCompressFused(ring, 22, 11, gain, kvCompressFusedEps); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 1 {
		t.Fatalf("fused pass made %v allocations per chunk, want exactly the one output buffer", allocs)
	}
	if chunk.Groups < 2 || len(chunk.Rows) != chunk.Groups*16 || cap(chunk.Rows) != len(chunk.Rows) {
		t.Fatalf("groups=%d len=%d cap=%d: rows are not one contiguous buffer", chunk.Groups, len(chunk.Rows), cap(chunk.Rows))
	}
}

func TestKVCompressFusedHandGate(t *testing.T) {
	ring := KVCompressFusedRing{Ratio: 2, Width: 2, Capacity: 2,
		KV:     []float32{1, 5, 3, 7},
		Scores: []float32{0, 0, 0, 40},
	}
	chunk, err := KVCompressFused(ring, 0, 2, []float32{1, 1}, kvCompressFusedEps)
	if err != nil {
		t.Fatal(err)
	}
	// Equal scores average dim 0 to 2; a 40-nat gap selects token 1 (7) on dim 1.
	rms := math.Sqrt((2*2 + 7*7) / 2.0)
	want := []float64{2 / rms, 7 / rms}
	for d, wv := range want {
		if math.Abs(float64(chunk.Rows[d])-wv) > 1e-2 {
			t.Fatalf("dim %d: got %v, want about %v", d, chunk.Rows[d], wv)
		}
	}
}

func TestKVCompressFusedRefusals(t *testing.T) {
	gain := kvCompressFusedGain(4)
	ring := newKVCompressFusedRing(2, 4, 6)
	writeKVCompressFusedRing(ring, 0, 6)

	if _, err := KVCompressFused(ring, 1, 6, gain, kvCompressFusedEps); !errors.Is(err, ErrV41CompressorRingGeometry) {
		t.Fatalf("lapped straddling group: got %v, want geometry refusal", err)
	}
	if _, err := KVCompressFused(ring, 0, 6, gain, kvCompressFusedEps); err != nil {
		t.Fatalf("span exactly at capacity must be accepted: %v", err)
	}
	if _, err := KVCompressFused(ring, -1, 2, gain, kvCompressFusedEps); !errors.Is(err, ErrV41CompressorRingGeometry) {
		t.Fatalf("negative start: got %v", err)
	}
	if _, err := KVCompressFused(ring, 0, 2, gain[:3], kvCompressFusedEps); !errors.Is(err, ErrV41CompressorNormDomain) {
		t.Fatalf("short gain: got %v", err)
	}
	short := ring
	short.Scores = short.Scores[:len(short.Scores)-1]
	if _, err := KVCompressFused(short, 0, 2, gain, kvCompressFusedEps); !errors.Is(err, ErrV41CompressorRingGeometry) {
		t.Fatalf("short score ring: got %v", err)
	}

	// A non-finite score inside a completed group withholds the chunk and
	// names its dimension; the same value in the pending tail is never read.
	poisoned := newKVCompressFusedRing(2, 4, 6)
	writeKVCompressFusedRing(poisoned, 0, 5)
	poisoned.Scores[3*4+2] = float32(math.NaN())
	var nf *V41CompressorRingNonFiniteError
	if _, err := KVCompressFused(poisoned, 0, 5, gain, kvCompressFusedEps); !errors.As(err, &nf) || nf.Dim != 2 || nf.Status != V41CompressorRingScoreNonFinite {
		t.Fatalf("NaN score in group 1: got %v", err)
	}
	poisoned.Scores[3*4+2] = kvCompressFusedScore(3, 2)
	poisoned.KV[4*4+1] = float32(math.Inf(1))
	chunk, err := KVCompressFused(poisoned, 0, 5, gain, kvCompressFusedEps)
	if err != nil || chunk.Groups != 2 || chunk.Pending != 1 {
		t.Fatalf("Inf in pending position 4 must not be read: chunk=%+v err=%v", chunk, err)
	}
}
