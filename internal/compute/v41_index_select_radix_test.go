package compute

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"sort"
	"strings"
	"testing"
)

// The unchanged rank reference is independent of the radix/partition model.
// Source checks bind that model to the embedded shader, not to a device run.
// fak-test:runtime fast est=1s lane=default
func TestV41IndexSelectRadix(t *testing.T) {
	source := strings.Join(strings.Fields(v41IndexSelectFullCoverageSource), " ")
	for _, fragment := range []string{
		"shared uint radixHistogram[256];", "shared uvec2 partitionPrefix[256];",
		"for (int shift = 24; shift >= 0; shift -= 8)",
		"atomicAdd(radixHistogram[(key >> uint(shift)) & 255u], 1u);",
		"uint tieAllowance = radixRank + 1u;",
		"uint begin = lane * quotient + min(lane, remainder);",
		"uint slot = before.x + min(before.y, tieAllowance);",
		"if (rows >= 1024u) { selectRadix(rows, uint(pc.topK), masked, keepBase, blockSize); return; }",
	} {
		if !strings.Contains(source, fragment) {
			t.Errorf("embedded shader missing radix source contract %q", fragment)
		}
	}
	nanAt := strings.Index(source, "if (Status[0] != 0u) return;")
	fullAt := strings.Index(source, "if (uint(pc.topK) >= rows)")
	radixAt := strings.Index(source, "if (rows >= 1024u)")
	if nanAt < 0 || !(nanAt < fullAt && fullAt < radixAt) {
		t.Error("radix branch must follow complete NaN refusal and full coverage")
	}

	check := func(scores []float32, k, blocks, size int, large bool) {
		t.Helper()
		var want []int32
		var wantErr error
		if large {
			want = v41IndexRadixSortOracle(scores, k) // Unmasked large cases only.
		} else {
			want, wantErr = V41IndexSelectReference(scores, k, blocks, size)
		}
		got, work, err := v41IndexRadixModel(scores, k, blocks, size)
		if (err != nil) != (wantErr != nil) || err != nil && err.Error() != wantErr.Error() || !reflect.DeepEqual(got, want) {
			t.Fatalf("N=%d K=%d mask=%d/%d got=%v/%v want=%v/%v", len(scores), k, blocks, size, got, err, want, wantErr)
		}
		wantWork := 0
		if err == nil && len(scores) >= 1024 && k < len(scores) {
			wantWork = 6 * len(scores)
		}
		if work != wantWork {
			t.Fatalf("N=%d K=%d radix key evaluations=%d want=%d", len(scores), k, work, wantWork)
		}
	}
	rng := rand.New(rand.NewSource(13294))
	values := []float32{float32(math.Inf(-1)), math.Float32frombits(0x80000000), 0, 1, 1, -2, float32(math.Inf(1)), math.Float32frombits(1), math.Float32frombits(0x80000001)}
	for _, n := range []int{1023, 1024, 1025, 1031} {
		scores := make([]float32, n)
		for i := range scores {
			scores[i] = values[rng.Intn(len(values))]
		}
		for _, k := range []int{1, 512, n - 1, n, n + 1} {
			check(scores, k, 0, 0, false)
		}
		for _, mask := range [][2]int{{1, 8}, {2, 17}, {(n-1)/8 + 1, 8}} {
			check(scores, 512, mask[0], mask[1], false)
		}
	}
	for _, value := range []float32{0, float32(math.Inf(-1)), float32(math.Inf(1))} {
		scores := make([]float32, 1025)
		for i := range scores {
			scores[i] = value
			if value == 0 && i%2 == 0 {
				scores[i] = math.Float32frombits(0x80000000)
			}
		}
		for _, k := range []int{1, 511, 512, 513, 1024} {
			check(scores, k, 0, 0, false)
		}
		check(scores, 512, 1, 8, false) // Includes selected masked sentinel ties.
	}
	for _, n := range []int{4099, 16384} {
		scores := make([]float32, n)
		for i := range scores {
			bits := rng.Uint32()
			if bits&0x7f800000 == 0x7f800000 { // Finite arbitrary bit patterns.
				bits &^= 0x00800000
			}
			scores[i] = math.Float32frombits(bits)
		}
		for _, k := range []int{1, 512, n - 1} {
			check(scores, k, 0, 0, true)
		}
	}
	for _, at := range []int{0, 511, 1024} {
		scores := make([]float32, 1025)
		scores[at] = math.Float32frombits(0xffc01234)
		check(scores, 512, 1, 8, false) // NaN is refused even outside kept blocks.
		check(scores, 1025, 0, 0, false)
	}
	// Maximum geometry is checked arithmetically without a64MiB test allocation.
	for _, n := range []uint32{1, 255, 256, 257, 1025, V41IndexSelectMaxRows - 1, V41IndexSelectMaxRows} {
		previous := uint32(0)
		for lane := uint32(0); lane < 256; lane++ {
			begin, end := v41IndexRadixPartition(n, lane)
			if begin != previous || end < begin || end > n {
				t.Fatalf("partition gap/overlap/overflow N=%d lane=%d [%d,%d) after%d", n, lane, begin, end, previous)
			}
			previous = end
		}
		if previous != n {
			t.Fatalf("partitions end=%d want=%d", previous, n)
		}
	}
	t.Log("N16384 partial selection:98304 radix key evaluations versus268435456 legacy row comparisons, plus unchanged NaN/mask and bounded shared work; CPU model counts only")
}

func v41IndexRadixPartition(n, lane uint32) (uint32, uint32) {
	q, rem := n/256, n%256
	begin := lane*q + min(lane, rem)
	end := begin + q
	if lane < rem {
		end++
	}
	return begin, end
}

func v41IndexRadixModel(scores []float32, k, blockK, blockSize int) ([]int32, int, error) {
	n := len(scores)
	if _, err := validateV41IndexSelect(Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{n}}, n, k, blockK, blockSize); err != nil {
		return nil, 0, err
	}
	keys := make([]uint32, n)
	for i, value := range scores {
		bits := math.Float32bits(value)
		if bits&0x7f800000 == 0x7f800000 && bits&0x007fffff != 0 {
			return nil, 0, fmt.Errorf("compute: V4.1 index select score %d is NaN", i)
		}
		if bits == 0x80000000 {
			bits = 0
		}
		keys[i] = bits | 0x80000000
		if bits&0x80000000 != 0 {
			keys[i] = ^bits
		}
	}
	if n < 1024 || k >= n {
		out, _, err := v41IndexSelectFullCoverageModel(scores, k, blockK, blockSize)
		return out, 0, err
	}
	const negInf, posInf = uint32(0x007fffff), uint32(0xff800000)
	if blockK > 0 {
		blocks := (n-1)/blockSize + 1
		if blockK < blocks {
			best := make([]uint32, blocks)
			for b := range best {
				best[b] = negInf
				for i := b * blockSize; i < min((b+1)*blockSize, n); i++ {
					best[b] = max(best[b], keys[i])
				}
			}
			best[blocks-1] = posInf
			for b, mine := range best {
				rank := 0
				for other, key := range best {
					if key > mine || key == mine && other < b {
						rank++
					}
				}
				if rank >= blockK || mine == negInf {
					for i := b * blockSize; i < min((b+1)*blockSize, n); i++ {
						keys[i] = negInf
					}
				}
			}
		}
	}
	prefix, mask, rank, work := uint32(0), uint32(0), uint32(k-1), 0
	for shift := 24; shift >= 0; shift -= 8 {
		var histogram [256]uint32
		for lane := 0; lane < 256; lane++ {
			for i := lane; i < n; i += 256 {
				work++
				if keys[i]&mask == prefix {
					histogram[keys[i]>>uint(shift)&255]++
				}
			}
		}
		for bucket := 255; bucket >= 0; bucket-- {
			if rank < histogram[bucket] {
				prefix |= uint32(bucket) << uint(shift)
				break
			}
			rank -= histogram[bucket]
		}
		mask |= 255 << uint(shift)
	}
	allowance := rank + 1
	var scan [256][2]uint32
	for lane := uint32(0); lane < 256; lane++ {
		begin, end := v41IndexRadixPartition(uint32(n), lane)
		for i := begin; i < end; i++ {
			work++
			if keys[i] > prefix {
				scan[lane][0]++
			}
			if keys[i] == prefix {
				scan[lane][1]++
			}
		}
	}
	for offset := 1; offset < 256; offset *= 2 {
		prior := scan // Every lane reads the completed previous shared phase.
		for lane := offset; lane < 256; lane++ {
			scan[lane][0] += prior[lane-offset][0]
			scan[lane][1] += prior[lane-offset][1]
		}
	}
	out, writes := make([]int32, k), make([]bool, k)
	for lane := uint32(0); lane < 256; lane++ {
		before := [2]uint32{}
		if lane > 0 {
			before = scan[lane-1]
		}
		slot, equals := before[0]+min(before[1], allowance), before[1]
		begin, end := v41IndexRadixPartition(uint32(n), lane)
		for i := begin; i < end; i++ {
			work++
			selected := keys[i] > prefix || keys[i] == prefix && equals < allowance
			if keys[i] == prefix {
				equals++
			}
			if selected {
				if slot >= uint32(k) || writes[slot] {
					return nil, work, fmt.Errorf("invalid radix output slot%d", slot)
				}
				out[slot], writes[slot] = int32(i), true
				if keys[i] == negInf {
					out[slot] = -1
				}
				slot++
			}
		}
	}
	for _, written := range writes {
		if !written {
			return nil, work, fmt.Errorf("radix output hole")
		}
	}
	return out, work, nil
}

// Independent float comparator, stable position tie, then position sort. Large
// cases are finite and unmasked; the untouched rank oracle covers masks/NaNs.
func v41IndexRadixSortOracle(scores []float32, k int) []int32 {
	positions := make([]int, len(scores))
	for i := range positions {
		positions[i] = i
	}
	sort.Slice(positions, func(i, j int) bool {
		a, b := positions[i], positions[j]
		return scores[a] > scores[b] || scores[a] == scores[b] && a < b
	})
	positions = positions[:min(k, len(positions))]
	sort.Ints(positions)
	out := make([]int32, len(positions))
	for i, pos := range positions {
		out[i] = int32(pos)
		if math.IsInf(float64(scores[pos]), -1) {
			out[i] = -1
		}
	}
	return out
}
