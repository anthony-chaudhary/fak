package compute

import (
	"errors"
	"math/rand"
	"os"
	"slices"
	"strings"
	"testing"
)

// verticalSlashDenseOracle derives one row block's metadata from a dense key
// column mask instead of the kernel's streaming two-pointer merge. A slash
// diagonal s seen from the row block ending at endM spans the 64 key columns
// ending at e = max(endM-s, 64). The first diagonal of a run anchors a 64-wide
// grid at e-64; later diagonals light the grid cell holding their last column,
// and a diagonal landing more than one cell past the lit run starts a new grid.
// Stray verticals are those below the final lit column that no cell covers.
func verticalSlashDenseOracle(seqlen int32, m int, vertical, slash []int32) (blocks, columns []int32) {
	const tile = 64
	startM := m * tile
	if startM >= int(seqlen) {
		return nil, nil
	}
	endM := startM + tile
	covered := make([]bool, endM+2*tile)
	anchor, maxCell, lastEnd := 0, -1, 0
	for _, s := range slash {
		if int(s) >= endM {
			continue
		}
		e := max(endM-int(s), tile)
		if maxCell < 0 || e > anchor+(maxCell+2)*tile {
			anchor, maxCell = e-tile, -1
		}
		cell := (e-anchor+tile-1)/tile - 1
		if cell <= maxCell {
			continue
		}
		maxCell = cell
		first := anchor + cell*tile
		blocks = append(blocks, int32(first))
		for c := first; c < first+tile; c++ {
			covered[c] = true
		}
		lastEnd = first + tile
	}
	if maxCell < 0 {
		return nil, nil
	}
	for _, v := range vertical {
		if int(v) < lastEnd && !covered[v] {
			columns = append(columns, v)
		}
	}
	return blocks, columns
}

func verticalSlashRow(idx *VerticalSlashIndex, b, h, m int) (blocks, columns []int32) {
	row := (b*idx.Heads+h)*idx.Rows + m
	blocks = idx.BlockOffset[row*idx.NNZS : (row+1)*idx.NNZS]
	columns = idx.ColumnIndex[row*idx.NNZV : (row+1)*idx.NNZV]
	return blocks[:idx.BlockCount[row]], columns[:idx.ColumnCount[row]]
}

func TestVerticalSlashIndexBuilder(t *testing.T) {
	t.Run("hand derived", func(t *testing.T) {
		// Expected values are worked by hand from the upstream kernel loop.
		cases := []struct {
			name            string
			context         int
			seqlen          int32
			vertical, slash []int32
			blockCount      []int32
			blockOffset     []int32
			columnCount     []int32
			columnIndex     []int32
		}{
			{
				// Row 1 extends one range across both diagonals; every vertical
				// lands inside a block or past the last one.
				name: "merged range", context: 128, seqlen: 128,
				vertical: []int32{0, 3, 70, 100}, slash: []int32{100, 64, 0},
				blockCount:  []int32{1, 2},
				blockOffset: []int32{0, 0, 0 /**/, 0, 64, 0},
				columnCount: []int32{0, 0},
				columnIndex: []int32{0, 0, 0, 0 /**/, 0, 0, 0, 0},
			},
			{
				// Unaligned block offsets, split ranges, duplicate verticals
				// kept, and a live final row block past seqlen's last full tile.
				name: "split ranges and stray columns", context: 256, seqlen: 200,
				vertical: []int32{5, 5, 70, 130, 250}, slash: []int32{130, 10, 0},
				blockCount:  []int32{1, 2, 3, 3},
				blockOffset: []int32{0, 0, 0 /**/, 54, 118, 0 /**/, 0, 118, 182 /**/, 62, 182, 246},
				columnCount: []int32{0, 2, 1, 3},
				columnIndex: []int32{
					0, 0, 0, 0, 0,
					5, 5, 0, 0, 0,
					70, 0, 0, 0, 0,
					5, 5, 130, 0, 0,
				},
			},
			{
				// Row blocks at or past seqlen stay zero.
				name: "seqlen gates rows", context: 256, seqlen: 64,
				vertical: []int32{5, 5, 70, 130, 250}, slash: []int32{130, 10, 0},
				blockCount:  []int32{1, 0, 0, 0},
				blockOffset: []int32{0, 0, 0 /**/, 0, 0, 0 /**/, 0, 0, 0 /**/, 0, 0, 0},
				columnCount: []int32{0, 0, 0, 0},
				columnIndex: make([]int32, 20),
			},
			{
				// Row 0 has no diagonal below its end: the port's one deviation
				// leaves it empty. Row 1 clips its only diagonal to [0, 64).
				name: "no slash below row end", context: 128, seqlen: 128,
				vertical: []int32{1, 90}, slash: []int32{200, 100},
				blockCount:  []int32{0, 1},
				blockOffset: []int32{0, 0 /**/, 0, 0},
				columnCount: []int32{0, 0},
				columnIndex: []int32{0, 0 /**/, 0, 0},
			},
		}
		for _, tc := range cases {
			got, err := BuildVerticalSlashIndexHost([]int32{tc.seqlen}, tc.vertical, tc.slash, 1, 1, tc.context)
			if err != nil {
				t.Fatalf("%s: %v", tc.name, err)
			}
			if !slices.Equal(got.BlockCount, tc.blockCount) || !slices.Equal(got.BlockOffset, tc.blockOffset) ||
				!slices.Equal(got.ColumnCount, tc.columnCount) || !slices.Equal(got.ColumnIndex, tc.columnIndex) {
				t.Errorf("%s:\n block_count  %v want %v\n block_offset %v want %v\n column_count %v want %v\n column_index %v want %v",
					tc.name, got.BlockCount, tc.blockCount, got.BlockOffset, tc.blockOffset,
					got.ColumnCount, tc.columnCount, got.ColumnIndex, tc.columnIndex)
			}
		}
	})

	t.Run("randomized against dense oracle", func(t *testing.T) {
		rng := rand.New(rand.NewSource(0x5ca1ab1e))
		for iter := 0; iter < 400; iter++ {
			batch, heads := 1+rng.Intn(2), 1+rng.Intn(3)
			context := 1 + rng.Intn(420)
			nnzV, nnzS := 1+rng.Intn(8), 1+rng.Intn(6)
			seqlens := make([]int32, batch)
			for i := range seqlens {
				seqlens[i] = int32(rng.Intn(context + 1))
			}
			vertical := make([]int32, batch*heads*nnzV)
			slash := make([]int32, batch*heads*nnzS)
			for bh := 0; bh < batch*heads; bh++ {
				v, s := vertical[bh*nnzV:(bh+1)*nnzV], slash[bh*nnzS:(bh+1)*nnzS]
				for i := range v {
					v[i] = int32(rng.Intn(context + 100))
				}
				for i := range s {
					s[i] = int32(rng.Intn(context + 50))
				}
				slices.Sort(v)
				slices.Sort(s)
				slices.Reverse(s)
			}
			got, err := BuildVerticalSlashIndexHost(seqlens, vertical, slash, batch, heads, context)
			if err != nil {
				t.Fatal(err)
			}
			if got.Rows != (context+63)/64 || got.NNZV != nnzV || got.NNZS != nnzS {
				t.Fatalf("iter %d: shape rows=%d nnzV=%d nnzS=%d", iter, got.Rows, got.NNZV, got.NNZS)
			}
			for b := 0; b < batch; b++ {
				for h := 0; h < heads; h++ {
					bh := b*heads + h
					v, s := vertical[bh*nnzV:(bh+1)*nnzV], slash[bh*nnzS:(bh+1)*nnzS]
					for m := 0; m < got.Rows; m++ {
						blocks, columns := verticalSlashRow(got, b, h, m)
						wantBlocks, wantColumns := verticalSlashDenseOracle(seqlens[b], m, v, s)
						if !slices.Equal(blocks, wantBlocks) || !slices.Equal(columns, wantColumns) {
							t.Fatalf("iter %d b=%d h=%d row=%d seqlen=%d vertical=%v slash=%v:\n blocks  %v want %v\n columns %v want %v",
								iter, b, h, m, seqlens[b], v, s, blocks, wantBlocks, columns, wantColumns)
						}
						// The metadata must cover every exact diagonal cell of
						// the row block: query row i attends key column i-s.
						for _, sv := range s {
							for i := m * 64; i < (m+1)*64 && len(blocks) > 0; i++ {
								col := int32(i) - sv
								if col < 0 {
									continue
								}
								hit := false
								for _, off := range blocks {
									hit = hit || (col >= off && col < off+64)
								}
								if !hit {
									t.Fatalf("iter %d b=%d h=%d row=%d: diagonal %d column %d uncovered by blocks %v",
										iter, b, h, m, sv, col, blocks)
								}
							}
						}
					}
				}
			}
		}
	})

	t.Run("refused shapes", func(t *testing.T) {
		one := []int32{0}
		for name, call := range map[string]func() (*VerticalSlashIndex, error){
			"seqlens per batch": func() (*VerticalSlashIndex, error) {
				return BuildVerticalSlashIndexHost([]int32{64}, []int32{0, 0}, []int32{0, 0}, 2, 1, 64)
			},
			"empty vertical": func() (*VerticalSlashIndex, error) {
				return BuildVerticalSlashIndexHost([]int32{64}, nil, one, 1, 1, 64)
			},
			"ragged slash": func() (*VerticalSlashIndex, error) {
				return BuildVerticalSlashIndexHost([]int32{64}, []int32{0, 0}, []int32{2, 1, 0}, 1, 2, 64)
			},
			"negative slash": func() (*VerticalSlashIndex, error) {
				return BuildVerticalSlashIndexHost([]int32{64}, one, []int32{-1}, 1, 1, 64)
			},
			"zero context": func() (*VerticalSlashIndex, error) {
				return BuildVerticalSlashIndexHost([]int32{0}, one, one, 1, 1, 0)
			},
		} {
			if _, err := call(); !errors.Is(err, ErrVerticalSlashIndexShape) {
				t.Errorf("%s: err = %v, want ErrVerticalSlashIndexShape", name, err)
			}
		}
	})

	t.Run("kernel source contract", func(t *testing.T) {
		raw, err := os.ReadFile("cuda/minference_vertical_slash_index.cuh")
		if err != nil {
			t.Fatal(err)
		}
		kernel := strings.Join(strings.Fields(string(raw)), " ")
		for _, want := range []string{
			"microsoft/MInference csrc/vertical_slash_index.cu",
			"29ef1974fcb5a1440ccf92d7d7996900230fddcf",
			"Copyright (c) Microsoft Corporation.",
			"MIT License",
			"__global__ void vertical_slash_index_kernel(",
			`extern "C" int fcuda_vertical_slash_index_64x64(`,
			"const int BLOCK_SIZE_M = 64; const int BLOCK_SIZE_N = 64; const int N_THREADS = 64;",
			"const dim3 dimGrid(n_heads, batch_size, (n_rows + N_THREADS - 1) / N_THREADS);",
			// The merge the host mirror transcribes.
			"if (start_m >= seqlen) { return; }",
			"while (s_idx >= end_m) { if (s >= NNZ_S) { return; } s_idx = slash_indexes[s++]; }",
			"s_idx = max(end_m - s_idx, BLOCK_SIZE_M); int range_start = s_idx - BLOCK_SIZE_M, range_end = s_idx;",
			"if (v_idx < range_start) { column_index[tmp_col_cnt++] = v_idx; }",
			"v_idx = end_m + BLOCK_SIZE_M;",
			"s_idx = max(end_m - slash_indexes[s++], BLOCK_SIZE_M);",
			"if (s_idx > range_end + BLOCK_SIZE_M) {",
			"} else if (s_idx > range_end) { range_end += BLOCK_SIZE_M; }",
			"for (int idx = range_start; idx < range_end; idx += block_size) { block_offset[block_count++] = idx; }",
		} {
			if !strings.Contains(kernel, want) {
				t.Errorf("kernel source lost %q", want)
			}
		}
	})
}
