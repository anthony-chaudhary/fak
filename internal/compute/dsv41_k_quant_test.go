package compute

import (
	"bytes"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default

// TestDSv41KQuantCacheRoundTrip pins the V4.1 mxfp8 K-plane borrow from
// vLLM deepseek_v41 cache_utils.py @ ae065918640a198a56f6f17eb45e251c3229fe41
// (Apache-2.0):
//
//	(a) quantize -> dequant-gather round trip stays within the mxfp8
//	    quantization bound (16 * tileScale per dim) vs the f32 oracle on
//	    adversarial magnitudes, with exact record bytes on hand-computed
//	    tiles;
//	(b) slot addressing matches the paged_kv_write_map scatter (same
//	    pos -> (page, offset) decomposition as paged_kv_write_offsets) for
//	    non-contiguous block tables;
//	(c) the one-pass and per-tile gather variants agree element-wise.
//
// CPU/in-memory results only: [SW-VERIFIED], no GPU or throughput claim.
func TestDSv41KQuantCacheRoundTrip(t *testing.T) {
	t.Parallel()

	t.Run("round_trip_bound", func(t *testing.T) {
		t.Parallel()
		row := make([]float32, DSV41KHeadDim)
		// Tile 0: fp8-max boundary. amax = 448 -> raw = 1 exactly ->
		// exponent 0, encoded scale 127.
		row[0], row[1], row[3] = 448, -448, 1
		// Tile 1: all-zero. amax floors to 1e-4 -> exponent -22, encoded
		// scale 105 (upstream's floor keeps a zero tile on a finite scale).
		// Tile 2: sub-f32-normal inputs. amax = 3e-30 < 1e-4 -> same floor.
		row[64], row[65], row[66] = 1e-30, -1e-30, 3e-30
		// Tile 3: adversarial large magnitudes. amax = 1e30 ->
		// exponent 91, encoded scale 218.
		row[96], row[97], row[98], row[99] = 1e30, -1e30, 5e29, 447.9
		// Tile 4: exact power-of-two boundary. amax = 0.4375 = 448*2^-10
		// -> raw = 2^-10 exactly -> exponent -10, encoded scale 117.
		row[128], row[129], row[130] = 0.4375, -0.4375, 0.21875
		// Tiles 5..15: deterministic mixed-sign/magnitude fill spanning
		// 1e-2..1e3.
		for tt := 5; tt < DSV41KNumScales; tt++ {
			for d := 0; d < DSV41KQuantBlock; d++ {
				mag := math.Pow(10, float64(d%6)-2)
				row[tt*DSV41KQuantBlock+d] =
					float32(math.Sin(float64(tt*37+d)*0.7) * mag)
			}
		}

		rec, err := DSV41KQuantizeRow(row)
		if err != nil {
			t.Fatalf("DSV41KQuantizeRow returned %v, want nil", err)
		}
		wantScales := []byte{127, 105, 105, 218, 117}
		if !bytes.Equal(rec[DSV41KHeadDim:DSV41KHeadDim+5], wantScales) {
			t.Errorf("scale bytes = %v, want %v",
				rec[DSV41KHeadDim:DSV41KHeadDim+5], wantScales)
		}
		// Hand-computed payload bytes for the five pinned tiles.
		var wantPayload [5 * DSV41KQuantBlock]byte
		wantPayload[0], wantPayload[1], wantPayload[3] = 0x7E, 0xFE, 0x38
		// -1e-30 / 2^-22 rounds to signed zero in e4m3: 0x80, not 0x00.
		wantPayload[65] = 0x80
		wantPayload[96], wantPayload[97], wantPayload[98] = 0x7D, 0xFD, 0x75
		wantPayload[128], wantPayload[129], wantPayload[130] = 0x7E, 0xFE, 0x76
		if !bytes.Equal(rec[:5*DSV41KQuantBlock], wantPayload[:]) {
			t.Errorf("payload bytes diverge from the hand-computed mxfp8 record")
		}

		deq, err := DSV41KDequantizeRow(rec)
		if err != nil {
			t.Fatalf("DSV41KDequantizeRow returned %v, want nil", err)
		}
		// Exact round trip on the exactly-representable fixture values.
		exactDims := []int{0, 1, 2, 3, 128, 129, 130}
		exactWants := []float32{448, -448, 0, 1, 0.4375, -0.4375, 0.21875}
		for d, want := range exactWants {
			dim := exactDims[d]
			if got := deq[dim]; got != want {
				t.Errorf("dequant[%d] = %v, want %v", dim, got, want)
			}
		}
		// Closed bound: |dequant - original| <= 16 * tileScale per dim.
		for d := 0; d < DSV41KHeadDim; d++ {
			tile := d / DSV41KQuantBlock
			exp := int(rec[DSV41KHeadDim+tile]) - 127
			bound := DSV41KMaxTileError(math.Ldexp(1, exp))
			if got := math.Abs(float64(deq[d] - row[d])); got > bound {
				t.Fatalf("dim %d: |dequant-original| = %g exceeds the mxfp8 bound %g",
					d, got, bound)
			}
		}
		// Independent scale cross-check on the random tiles: the stored
		// exponent must equal ceil(log2(amax/448)) recomputed in float64.
		for tt := 5; tt < DSV41KNumScales; tt++ {
			var amax float64
			for _, v := range row[tt*DSV41KQuantBlock : (tt+1)*DSV41KQuantBlock] {
				if a := math.Abs(float64(v)); a > amax {
					amax = a
				}
			}
			if amax < 1e-4 {
				amax = 1e-4
			}
			wantExp := int(math.Ceil(math.Log2(amax / 448.0)))
			if gotExp := int(rec[DSV41KHeadDim+tt]) - 127; gotExp != wantExp {
				t.Errorf("tile %d: stored exponent %d, want %d", tt, gotExp, wantExp)
			}
		}
	})

	t.Run("encoder_rne_closure", func(t *testing.T) {
		t.Parallel()
		// Closed round values pinning RNE + saturation of the e4m3 encoder.
		cases := []struct {
			in   float32
			want byte
		}{
			{448, 0x7E}, {-448, 0xFE}, {448.0001, 0x7E}, {-448.0001, 0xFE},
			{0, 0x00}, {math.Float32frombits(0x80000000), 0x80}, // -0.0 -> signed zero
			{1, 0x38}, {0.5, 0x30}, {224, 0x76}, {447.9, 0x7E},
			{float32(math.Ldexp(1, -13)), 0x01}, // smallest subnormal
			{float32(math.Ldexp(1, -14)), 0x00}, // exact tie rounds to even (0)
			{float32(math.Ldexp(3, -14)), 0x02}, // exact tie rounds to even (2)
		}
		for _, c := range cases {
			if got := dsv41kEncodeE4M3(c.in); got != c.want {
				t.Errorf("encode(%v) = 0x%02X, want 0x%02X", c.in, got, c.want)
			}
		}
	})

	t.Run("slot_addressing_matches_write_map", func(t *testing.T) {
		t.Parallel()
		// Non-contiguous, unordered physical pages: logical pages 0,1,2
		// live in physical blocks 3, 7, 10 of a 12-block arena.
		const capacityBlocks = 12
		blocks := []int32{3, 7, 10}
		const firstOffset, tokens = 5, 38 // positions 5..42
		arena := make([]byte, capacityBlocks*DSV41KPageBytes)
		for i := range arena {
			arena[i] = 0xAB
		}
		k := make([]float32, tokens*DSV41KHeadDim)
		for tok := 0; tok < tokens; tok++ {
			for d := 0; d < DSV41KHeadDim; d++ {
				k[tok*DSV41KHeadDim+d] = float32(tok)*1000 + float32(d) + 1
			}
		}
		if err := DSV41KQuantInsert(k, tokens, firstOffset, blocks,
			capacityBlocks, arena); err != nil {
			t.Fatalf("DSV41KQuantInsert returned %v, want nil", err)
		}
		// Hand-computed record bases through the paged_kv_write_offsets
		// decomposition (pos = firstOffset + token; page =
		// blocks[pos/16]; offset = pos%16; base = page*8448 + offset*528):
		//   pos 5  -> block 3,  off 5  -> 3*8448 + 5*528  = 27984
		//   pos 16 -> block 7,  off 0  -> 7*8448          = 59136
		//   pos 42 -> block 10, off 10 -> 10*8448 + 10*528 = 89760
		handComputed := map[int]int64{0: 27984, 11: 59136, 37: 89760}
		for tok, base := range handComputed {
			if got := dsv41kRecordSlot(int(blocks[(firstOffset+tok)/DSV41KPageSize]),
				firstOffset+tok); got != base {
				t.Fatalf("record base for token %d = %d, want %d", tok, got, base)
			}
		}
		// Every inserted record sits at its computed slot, byte for byte.
		for tok := 0; tok < tokens; tok++ {
			pos := firstOffset + tok
			base := dsv41kRecordSlot(int(blocks[pos/DSV41KPageSize]), pos)
			wantRec, err := DSV41KQuantizeRow(k[tok*DSV41KHeadDim : (tok+1)*DSV41KHeadDim])
			if err != nil {
				t.Fatalf("quantize row %d: %v", tok, err)
			}
			if !bytes.Equal(arena[base:base+DSV41KBytesPerToken], wantRec) {
				t.Fatalf("token %d: record at base %d does not match the quantized row",
					tok, base)
			}
		}
		// Untouched slots keep the sentinel: block 3 owns offsets 5..15,
		// block 7 owns 0..15, block 10 owns 0..10.
		written := map[int32][]int{3: {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
			7:  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
			10: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
		sentinel := bytes.Repeat([]byte{0xAB}, DSV41KBytesPerToken)
		for _, blk := range blocks {
			for off := 0; off < DSV41KPageSize; off++ {
				isWritten := false
				for _, w := range written[blk] {
					if w == off {
						isWritten = true
					}
				}
				if isWritten {
					continue
				}
				base := int64(blk)*int64(DSV41KPageBytes) + int64(off)*int64(DSV41KBytesPerToken)
				if !bytes.Equal(arena[base:base+DSV41KBytesPerToken], sentinel) {
					t.Fatalf("block %d offset %d: untouched slot was overwritten",
						blk, off)
				}
			}
		}
		// Gather back through the same non-contiguous table: the scatter and
		// the gather must agree record for record.
		seqLen := firstOffset + tokens
		out := make([]float32, seqLen*DSV41KHeadDim)
		if err := DSV41KDequantGatherOnePass(out, arena, blocks, seqLen, tokens,
			firstOffset); err != nil {
			t.Fatalf("gather returned %v, want nil", err)
		}
		for tok := 0; tok < tokens; tok++ {
			wantRec, _ := DSV41KQuantizeRow(k[tok*DSV41KHeadDim : (tok+1)*DSV41KHeadDim])
			wantRow, _ := DSV41KDequantizeRow(wantRec)
			gotRow := out[(firstOffset+tok)*DSV41KHeadDim : (firstOffset+tok+1)*DSV41KHeadDim]
			if !reflect.DeepEqual(gotRow, wantRow) {
				t.Fatalf("token %d: gathered row diverges from the oracle dequant", tok)
			}
		}
	})

	t.Run("gather_variants_agree", func(t *testing.T) {
		t.Parallel()
		// Two-page sequence with distinct, out-of-order physical pages, plus
		// edge windows: a full single page, a one-token tail, and an offset
		// window.
		const capacityBlocks = 8
		blocks := []int32{5, 1} // deliberately out of order
		seqLen := 32            // pages: 0..15 -> block 5, 16..31 -> block 1
		arena := make([]byte, capacityBlocks*DSV41KPageBytes)
		k := make([]float32, seqLen*DSV41KHeadDim)
		for p := 0; p < seqLen; p++ {
			for d := 0; d < DSV41KHeadDim; d++ {
				k[p*DSV41KHeadDim+d] = float32(p*7 + d)
			}
		}
		if err := DSV41KQuantInsert(k, seqLen, 0, blocks, capacityBlocks, arena); err != nil {
			t.Fatalf("insert returned %v, want nil", err)
		}
		windows := []struct {
			gatherLen, offset int
		}{
			{32, 0}, {16, 0}, {1, 0}, {32, 4},
		}
		for _, w := range windows {
			a := make([]float32, (w.offset+w.gatherLen)*DSV41KHeadDim)
			b := make([]float32, (w.offset+w.gatherLen)*DSV41KHeadDim)
			if err := DSV41KDequantGatherOnePass(a, arena, blocks, seqLen,
				w.gatherLen, w.offset); err != nil {
				t.Fatalf("one-pass gather (%d,%d): %v", w.gatherLen, w.offset, err)
			}
			if err := DSV41KDequantGatherPerBlock(b, arena, blocks, seqLen,
				w.gatherLen, w.offset); err != nil {
				t.Fatalf("per-block gather (%d,%d): %v", w.gatherLen, w.offset, err)
			}
			if !reflect.DeepEqual(a, b) {
				t.Fatalf("one-pass and per-block gathers disagree for window (%d,%d)",
					w.gatherLen, w.offset)
			}
			// Both must also match the record-level oracle.
			start := seqLen - w.gatherLen
			for i := 0; i < w.gatherLen; i++ {
				pos := start + i
				base := dsv41kRecordSlot(int(blocks[pos/DSV41KPageSize]), pos)
				wantRow, _ := DSV41KDequantizeRow(
					arena[base : base+DSV41KBytesPerToken])
				gotRow := a[(w.offset+i)*DSV41KHeadDim : (w.offset+i+1)*DSV41KHeadDim]
				if !reflect.DeepEqual(gotRow, wantRow) {
					t.Fatalf("position %d: gather diverges from the record oracle", pos)
				}
			}
		}
	})

	t.Run("sentinel_errors", func(t *testing.T) {
		t.Parallel()
		arena := make([]byte, 4*DSV41KPageBytes)
		checks := []struct {
			name     string
			err      error
			sentinel error
		}{
			{"empty_tokens", dsv41kInsertErr(nil, 0, 0, []int32{0}, 4, arena), ErrDSV41KEmpty},
			{"short_k", dsv41kInsertErr(make([]float32, 100), 1, 0, []int32{0}, 4, arena), ErrDSV41KShape},
			{"wrong_page_count", dsv41kInsertErr(make([]float32, DSV41KHeadDim), 1, 0, []int32{0, 1}, 4, arena), ErrDSV41KPageMap},
			{"duplicate_page", dsv41kInsertErr(make([]float32, 32*DSV41KHeadDim), 32, 0, []int32{1, 1}, 4, arena), ErrDSV41KPageMap},
			{"page_out_of_range", dsv41kInsertErr(make([]float32, DSV41KHeadDim), 1, 0, []int32{4}, 4, arena), ErrDSV41KPageMap},
			{"negative_page", dsv41kInsertErr(make([]float32, DSV41KHeadDim), 1, 0, []int32{-1}, 4, arena), ErrDSV41KPageMap},
			{"bad_first_offset", dsv41kInsertErr(make([]float32, DSV41KHeadDim), 1, 16, []int32{0}, 4, arena), ErrDSV41KPageMap},
			{"small_arena", dsv41kInsertErr(make([]float32, DSV41KHeadDim), 1, 0, []int32{0}, 4, arena[:DSV41KPageBytes]), ErrDSV41KShape},
			{"row_width", dsv41kRowErr(make([]float32, DSV41KHeadDim-1)), ErrDSV41KShape},
			{"nan_row", dsv41kRowErr(nanRow()), ErrDSV41KNonFinite},
			{"short_record", dsv41kRecordErr(make([]byte, DSV41KBytesPerToken-1)), ErrDSV41KRecord},
			{"gather_past_seq", dsv41kGatherErr(arena, []int32{0}, 16, 17, 0), ErrDSV41KShape},
			{"gather_offset_overflow", dsv41kGatherErr(arena, []int32{0}, 16, 16, 64), ErrDSV41KShape},
			{"short_block_table", dsv41kGatherErr(arena, []int32{}, 33, 33, 0), ErrDSV41KPageMap},
			{"negative_block_table", dsv41kGatherErr(arena, []int32{-1, 0, 1}, 33, 33, 0), ErrDSV41KPageMap},
			{"arena_extent", dsv41kGatherErr(arena, []int32{0, 4}, 33, 33, 0), ErrDSV41KPageMap},
		}
		for _, c := range checks {
			if !errors.Is(c.err, c.sentinel) {
				t.Errorf("%s: errors.Is(%v, %v) = false, want true",
					c.name, c.err, c.sentinel)
			}
		}
	})

	t.Run("source_contract", func(t *testing.T) {
		t.Parallel()
		cu, err := os.ReadFile(filepath.FromSlash("dsv41_k_quant.cu"))
		if err != nil {
			t.Fatal(err)
		}
		cuText := string(bytes.ReplaceAll(cu, []byte("\r\n"), []byte("\n")))
		cuCode := stripCComments(cuText)
		for token, want := range map[string]int{
			"__global__ void k_dsv41_kq_":            3,
			`extern "C" int fcuda_dsv41_kq_`:         2,
			`#include "cuda/dsv41_k_quant_vllm.cuh"`: 1,
		} {
			if got := strings.Count(cuCode, token); got != want {
				t.Errorf("dsv41_k_quant.cu has %d occurrences of %q, want %d",
					got, token, want)
			}
		}
		if !strings.Contains(cuText, "ae065918640a198a56f6f17eb45e251c3229fe41") {
			t.Error("dsv41_k_quant.cu lacks the upstream vLLM pin citation")
		}
		cuh, err := os.ReadFile(filepath.FromSlash("cuda/dsv41_k_quant_vllm.cuh"))
		if err != nil {
			t.Fatal(err)
		}
		cuhText := string(cuh)
		for sym := range map[string]bool{
			"dsv41_kq_ue8m0_exponent": true,
			"dsv41_kq_e4m3_encode":    true,
			"dsv41_kq_e4m3_decode":    true,
			"dsv41_kq_record_slot":    true,
			"dsv41_kq_insert_slot":    true,
			"dsv41_kq_gather_slot":    true,
		} {
			if !strings.Contains(cuhText, sym) {
				t.Errorf("dsv41_k_quant_vllm.cuh missing shared helper %q", sym)
			}
		}
		if !strings.Contains(cuhText, `#include "paged_kv_write_map.h"`) {
			t.Error("dsv41_k_quant_vllm.cuh must address through the existing paged write map seam")
		}
		if !strings.Contains(cuhText, "ae065918640a198a56f6f17eb45e251c3229fe41") {
			t.Error("dsv41_k_quant_vllm.cuh lacks the upstream vLLM pin citation")
		}
	})
}

func dsv41kInsertErr(k []float32, numTokens, firstOffset int, blocks []int32,
	capacityBlocks int, arena []byte) error {
	return DSV41KQuantInsert(k, numTokens, firstOffset, blocks, capacityBlocks, arena)
}

func dsv41kRowErr(row []float32) error {
	_, err := DSV41KQuantizeRow(row)
	return err
}

func dsv41kRecordErr(rec []byte) error {
	_, err := DSV41KDequantizeRow(rec)
	return err
}

func dsv41kGatherErr(arena []byte, blockTable []int32, seqLen, gatherLen, offset int) error {
	out := make([]float32, 64*DSV41KHeadDim)
	return DSV41KDequantGatherOnePass(out, arena, blockTable, seqLen, gatherLen, offset)
}

func nanRow() []float32 {
	row := make([]float32, DSV41KHeadDim)
	row[0] = math.Float32frombits(0x7f800000) // float32 NaN
	return row
}
