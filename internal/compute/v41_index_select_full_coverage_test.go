package compute

import (
	"crypto/sha256"
	_ "embed"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
)

// Bind the source contract to the same build inputs as this test. In particular,
// a Go source overlay must also supply the shader as an embedded build input.
//
//go:embed shaders/v41_index_select.comp
var v41IndexSelectFullCoverageSource string

// Refs #13668/#13294. Counts are CPU shader-equivalent loop work, not GPU time.
// The independent full-rank reference is deliberately unchanged.
// fak-test:runtime fast est=500ms lane=default
func TestV41IndexSelectFullCoverage(t *testing.T) {
	const parentSHA = "b7855ccb6d241213bfb1dbeada14c4b55a75a78e0b3fb517fa9ffad796013f0f"
	const candidateSHA = "eea9fc242fb08c5b89c27657cfd9c0726ac13375ce37abc135ab372cda8c09cd"
	shaderSHA := fmt.Sprintf("%x", sha256.Sum256([]byte(v41IndexSelectFullCoverageSource)))
	t.Logf("embedded shader SHA256=%s candidate=%s parent=%s", shaderSHA, candidateSHA, parentSHA)
	if shaderSHA != candidateSHA {
		t.Errorf("embedded shader is not the reviewed full-coverage candidate (parent RED must identify %s)", parentSHA)
	}
	source := strings.Join(strings.Fields(v41IndexSelectFullCoverageSource), " ")
	for _, fragment := range []string{
		"if (masked && uint(pc.topKBlocks) >= blocks) masked = false;",
		"if (uint(pc.topK) >= rows) {",
		"Out[i] = mine == KEY_NEG_INF ? -1.0 : float(i);",
		"uint selectedBase = 2u * blocks;",
		"for (uint o = 0u; o < blocks; ++o)",
		"for (uint j = 0u; j < rows; ++j)",
		"for (uint j = 0u; j < i; ++j)",
	} {
		if !strings.Contains(source, fragment) {
			t.Errorf("full-coverage source contract missing %q", fragment)
		}
	}
	nanAt := strings.Index(source, "if (Status[0] != 0u) return;")
	blockAt := strings.Index(source, "if (masked && uint(pc.topKBlocks) >= blocks)")
	fullAt := strings.Index(source, "if (uint(pc.topK) >= rows)")
	rankAt := strings.Index(source, "for (uint j = 0u; j < rows; ++j)")
	if nanAt < 0 || !(nanAt < blockAt && blockAt < fullAt && fullAt < rankAt) {
		t.Error("coverage branches must follow complete NaN refusal and precede row ranking")
	}
	check := func(scores []float32, k, blocks, size int) [3]int64 {
		t.Helper()
		want, wantErr := V41IndexSelectReference(scores, k, blocks, size)
		got, work, err := v41IndexSelectFullCoverageModel(scores, k, blocks, size)
		if (err != nil) != (wantErr != nil) || (err != nil && err.Error() != wantErr.Error()) || !reflect.DeepEqual(got, want) {
			t.Fatalf("N=%d K=%d blocks=%d/%d got=%v/%v want=%v/%v", len(scores), k, blocks, size, got, err, want, wantErr)
		}
		if err == nil {
			n := int64(len(scores))
			if k >= len(scores) && (work[1] != 0 || work[2] != 0) {
				t.Fatalf("full row coverage still ranked/prefix-scanned: %v", work)
			}
			if k < len(scores) && work[1] != n*n {
				t.Fatalf("partial row coverage changed its full-rank body: %v", work)
			}
			if blocks > 0 {
				b := int64((len(scores)-1)/size + 1)
				wantBlockWork := b * b
				if int64(blocks) >= b {
					wantBlockWork = 0
				}
				if work[0] != wantBlockWork {
					t.Fatalf("block coverage work=%d want=%d", work[0], wantBlockWork)
				}
			}
		}
		return work
	}
	negInf := float32(math.Inf(-1))
	values := []float32{negInf, math.Float32frombits(0x80000000), 0, 1, 1, -2, float32(math.Inf(1)), negInf}
	for _, n := range []int{1, 7, 255, 256, 257, 511, 512, 513} {
		scores := make([]float32, n)
		for i := range scores {
			scores[i] = values[(i*13+i/7)%len(values)]
		}
		blocks := (n-1)/8 + 1
		for _, k := range []int{n - 1, n, n + 1} {
			for _, mask := range [][2]int{{0, 0}, {1, 8}, {blocks, 8}, {blocks + 1, 8}} {
				check(scores, k, mask[0], mask[1])
			}
		}
		// Published budgets: row coverage ends after512keys; block coverage
		// remains complete through2048blocks. Partial ranking stays unchanged.
		check(scores, 512, 2048, 8)
	}
	for _, tc := range []struct {
		scores          []float32
		k, blocks, size int
		want            []int32
	}{
		{[]float32{negInf, 9, 8, negInf, 7}, 5, 1, 2, []int32{-1, -1, -1, -1, 4}},
		{[]float32{negInf, 9, negInf, 8}, 5, 2, 2, []int32{-1, 1, -1, 3}},
		{[]float32{negInf, negInf, 3, -2, negInf, negInf}, 2, 3, 2, []int32{2, 3}},
		{[]float32{negInf, negInf, negInf}, 3, 1, 2, []int32{-1, -1, -1}},
	} {
		check(tc.scores, tc.k, tc.blocks, tc.size)
		got, _, _ := v41IndexSelectFullCoverageModel(tc.scores, tc.k, tc.blocks, tc.size)
		if !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("fixed position/sentinel control=%v want=%v", got, tc.want)
		}
	}
	for _, at := range []int{0, 256} {
		scores := make([]float32, 257)
		scores[at] = math.Float32frombits(0xffc01234)
		for _, g := range [][3]int{{257, 33, 8}, {258, 1, 8}, {256, 33, 8}} {
			if work := check(scores, g[0], g[1], g[2]); work != [3]int64{} {
				t.Fatal("NaN refusal ran selection work")
			}
		}
	}
	check(nil, 1, 0, 0)
	for _, g := range [][3]int{{0, 0, 0}, {-1, 0, 0}, {1, -1, 0}, {1, 1, 0}} {
		check([]float32{1, 2}, g[0], g[1], g[2])
	}
	t.Log("covered geometry performs zero block/row ranking and row-prefix comparisons; partial-coverage loops retain full scans (CPU operation counts only)")
}

// Work records block-rank, row-rank and output-prefix comparisons. Row flags
// and publication preserve the shader's256-lane assignment and input order.
func v41IndexSelectFullCoverageModel(scores []float32, topK, topKBlocks, blockSize int) ([]int32, [3]int64, error) {
	var work [3]int64
	rows := len(scores)
	if _, err := validateV41IndexSelect(Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{rows}}, rows, topK, topKBlocks, blockSize); err != nil {
		return nil, work, err
	}
	const negInf, posInf = uint32(0x007fffff), uint32(0xff800000)
	keys := make([]uint32, rows)
	for i, value := range scores {
		bits := math.Float32bits(value)
		if bits&0x7f800000 == 0x7f800000 && bits&0x007fffff != 0 {
			return nil, work, fmt.Errorf("compute: V4.1 index select score %d is NaN", i)
		}
		if bits == 0x80000000 {
			bits = 0
		}
		if bits&0x80000000 != 0 {
			keys[i] = ^bits
		} else {
			keys[i] = bits | 0x80000000
		}
	}
	outranks := func(a uint32, ia int, b uint32, ib int) bool { return a > b || (a == b && ia < ib) }
	if topKBlocks > 0 {
		blocks := (rows-1)/blockSize + 1
		if topKBlocks < blocks {
			best, keep := make([]uint32, blocks), make([]bool, blocks)
			for b := range best {
				best[b] = negInf
				for p := b * blockSize; p < min((b+1)*blockSize, rows); p++ {
					best[b] = max(best[b], keys[p])
				}
			}
			best[blocks-1] = posInf
			for b, mine := range best {
				rank := 0
				for o, other := range best {
					work[0]++
					if outranks(other, o, mine, b) {
						rank++
					}
				}
				keep[b] = rank < topKBlocks && mine != negInf
			}
			for i := range keys {
				if !keep[i/blockSize] {
					keys[i] = negInf
				}
			}
		}
	}
	if topK >= rows {
		out := make([]int32, rows)
		for lane := 0; lane < 256; lane++ {
			for i := lane; i < rows; i += 256 {
				out[i] = int32(i)
				if keys[i] == negInf {
					out[i] = -1
				}
			}
		}
		return out, work, nil
	}
	selected := make([]bool, rows)
	for lane := 0; lane < 256; lane++ {
		for i := lane; i < rows; i += 256 {
			rank := 0
			for j, other := range keys {
				work[1]++
				if outranks(other, j, keys[i], i) {
					rank++
				}
			}
			selected[i] = rank < topK
		}
	}
	out := make([]int32, min(topK, rows))
	for i, keep := range selected {
		if !keep {
			continue
		}
		slot := 0
		for j := 0; j < i; j++ {
			work[2]++
			if selected[j] {
				slot++
			}
		}
		out[slot] = int32(i)
		if keys[i] == negInf {
			out[slot] = -1
		}
	}
	return out, work, nil
}
