package compute

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"os"
	"testing"
)

// nvfp4_golden_fixture_test.go — layout-guard campaign leaf (flow ticket lg-golden-fixtures):
// pinned input-to-swizzled byte fixtures for SwizzleNVFP4Scales, the 128x4 block-scale tile
// layout of NVIDIA TensorRT-LLM, zero padding included. Unlike the round-trip test in
// nvfp4_scale_layout_test.go (which cannot see an offset bug that Unswizzle shares), these
// fixtures pin the ABSOLUTE destination of every scale byte against the pinned upstream
// formula, so a stride/tile drift in nvfp4ScaleLayoutOffset fails here even when the
// swizzle/unswizzle pair still inverts itself.
//
// PROVENANCE. Pinned at NVIDIA/TensorRT-LLM @ ecae3c97e2b646580415518a45a72bc5a5a67075,
// fetched 2026-11-03:
//   - cpp/tensorrt_llm/kernels/quantization.h (sha256 4a64db27f63ba18c790087bcae314ad9b3eec3f9ca1907cfe314faaf0c3594d7)
//     — QuantizationSFLayout::SWIZZLED contract: 512-byte blocks of 128x4 FP8 values, rows
//     padded to 128, columns to 4.
//   - cpp/tensorrt_llm/kernels/quantization.cu (sha256 141e4a222b0511bd568aebd0b72d3ff3d3aa6340f676eb215502aa512621318f)
//     — block_scale_interleave_kernel: out-of-range row/col writes zero, and the exact
//     offset math (numSfTilesK = PadUp(cols,4)/4; tileOffset = ((r/128)*numSfTilesK + c/4)*512;
//     dst = tileOffset + (r%32)*16 + ((r%128)/32)*4 + c%4).
// The expected bytes below come from nvfp4GoldenSwizzle, which applies that pinned formula
// through nvfp4ScaleGoldenOffset (the deliberately duplicated upstream formula already in
// nvfp4_scale_layout_test.go) — never from SwizzleNVFP4Scales, so the fixture cannot be
// laundered by production drift. See testdata/README.md for the full record.
//
// DERIVATION. Inputs are linear[i] = (37*i+11) mod 256 (gcd(37,256)=1, so the stream covers
// all 256 byte values and includes zeros inside the data region — only the PADDING region
// must be all zero). Regenerate with FAK_UPDATE_GOLDEN=1 (repo convention); verify with a
// plain run.

const nvfp4SwizzleGoldenFixturePath = "testdata/nvfp4_swizzle_golden.json"

const nvfp4SwizzleGoldenSchema = "fak.nvfp4-swizzle-golden.v1"

type nvfp4SwizzleProvenance struct {
	Reference     string `json:"reference"`
	SourceSHA     string `json:"source_sha"`
	HeaderSHA256  string `json:"header_sha256"`
	KernelSHA256  string `json:"kernel_sha256"`
	Fetched       string `json:"fetched"`
	Formula       string `json:"formula"`
	Padding       string `json:"padding_rule"`
	DerivationCmd string `json:"derivation_cmd"`
}

type nvfp4SwizzleCase struct {
	Name      string `json:"name"`
	Rows      int    `json:"rows"`
	ScaleCols int    `json:"scale_cols"`
	InputHex  string `json:"input_hex"`
	PackedHex string `json:"packed_hex"`
}

type nvfp4SwizzleFixture struct {
	Schema     string                 `json:"schema"`
	Provenance nvfp4SwizzleProvenance `json:"provenance"`
	Cases      []nvfp4SwizzleCase     `json:"cases"`
}

// nvfp4SwizzleCaseShapes is the pinned coverage set: the minimal shape (maximum padding
// fraction), exactly one 128x4 tile (no padding anywhere), and both axes padded at once
// (row pad into the second tile row, column pad inside every tile).
var nvfp4SwizzleCaseShapes = []struct {
	name       string
	rows, cols int
}{
	{"single-scale-padded-to-tile", 1, 1},
	{"exact-single-tile", 128, 4},
	{"both-axes-padding", 130, 6},
}

func nvfp4GoldenInput(rows, cols int) []byte {
	linear := make([]byte, rows*cols)
	for i := range linear {
		linear[i] = byte((i*37 + 11) % 256)
	}
	return linear
}

// nvfp4GoldenSwizzle applies the pinned upstream formula to build the canonical 128x4
// tiled buffer: every in-range scale lands at its pinned offset, and every out-of-range
// (padded) slot stays zero. Structurally it is the upstream kernel's scatter loop, not the
// production function.
func nvfp4GoldenSwizzle(rows, cols int, linear []byte) []byte {
	paddedRows := ((rows + 127) / 128) * 128
	paddedCols := ((cols + 3) / 4) * 4
	out := make([]byte, paddedRows*paddedCols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[nvfp4ScaleGoldenOffset(r, c, paddedCols/4)] = linear[r*cols+c]
		}
	}
	return out
}

// buildNVFP4SwizzleFixture derives the whole fixture from the pinned upstream formula
// alone. This is the derivation command's body.
func buildNVFP4SwizzleFixture() nvfp4SwizzleFixture {
	cases := make([]nvfp4SwizzleCase, 0, len(nvfp4SwizzleCaseShapes))
	for _, sh := range nvfp4SwizzleCaseShapes {
		input := nvfp4GoldenInput(sh.rows, sh.cols)
		packed := nvfp4GoldenSwizzle(sh.rows, sh.cols, input)
		cases = append(cases, nvfp4SwizzleCase{
			Name:      sh.name,
			Rows:      sh.rows,
			ScaleCols: sh.cols,
			InputHex:  hex.EncodeToString(input),
			PackedHex: hex.EncodeToString(packed),
		})
	}
	return nvfp4SwizzleFixture{
		Schema: nvfp4SwizzleGoldenSchema,
		Provenance: nvfp4SwizzleProvenance{
			Reference: "NVIDIA/TensorRT-LLM @ ecae3c97e2b646580415518a45a72bc5a5a67075, " +
				"cpp/tensorrt_llm/kernels/quantization.h (QuantizationSFLayout::SWIZZLED) + " +
				"cpp/tensorrt_llm/kernels/quantization.cu (block_scale_interleave_kernel)",
			SourceSHA:     "ecae3c97e2b646580415518a45a72bc5a5a67075",
			HeaderSHA256:  "4a64db27f63ba18c790087bcae314ad9b3eec3f9ca1907cfe314faaf0c3594d7",
			KernelSHA256:  "141e4a222b0511bd568aebd0b72d3ff3d3aa6340f676eb215502aa512621318f",
			Fetched:       "2026-11-03",
			Formula:       "tilesK=PadUp(cols,4)/4; dst=((r/128)*tilesK+c/4)*512+(r%32)*16+((r%128)/32)*4+c%4",
			Padding:       "out-of-range row or col writes zero; buffer is PadUp(rows,128)*PadUp(cols,4)",
			DerivationCmd: "FAK_UPDATE_GOLDEN=1 go test ./internal/compute -run TestNVFP4SwizzleGoldenFixtures -count=1",
		},
		Cases: cases,
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestNVFP4SwizzleGoldenFixtures(t *testing.T) {
	want := buildNVFP4SwizzleFixture()

	if os.Getenv("FAK_UPDATE_GOLDEN") == "1" {
		data, err := json.MarshalIndent(want, "", "  ")
		if err != nil {
			t.Fatalf("marshal golden fixture: %v", err)
		}
		if err := os.WriteFile(nvfp4SwizzleGoldenFixturePath, append(data, '\n'), 0o644); err != nil {
			t.Fatalf("write golden fixture: %v", err)
		}
		t.Logf("updated %s (%d cases)", nvfp4SwizzleGoldenFixturePath, len(want.Cases))
		return
	}

	gotData, err := os.ReadFile(nvfp4SwizzleGoldenFixturePath)
	if err != nil {
		t.Fatalf("read golden fixture %s: %v (regenerate with FAK_UPDATE_GOLDEN=1)", nvfp4SwizzleGoldenFixturePath, err)
	}
	var got nvfp4SwizzleFixture
	if err := json.Unmarshal(gotData, &got); err != nil {
		t.Fatalf("unmarshal golden fixture: %v", err)
	}
	if got.Schema != nvfp4SwizzleGoldenSchema {
		t.Fatalf("fixture schema = %q, want %q", got.Schema, nvfp4SwizzleGoldenSchema)
	}
	if len(got.Cases) != len(want.Cases) {
		t.Fatalf("fixture has %d cases, derived contract has %d — coverage changed; review before regenerating",
			len(got.Cases), len(want.Cases))
	}

	sawNoPadding, sawBothPadded := false, false
	for i, c := range got.Cases {
		w := want.Cases[i]
		if c.Name != w.Name || c.Rows != w.Rows || c.ScaleCols != w.ScaleCols {
			t.Fatalf("case %d drifted from the pinned shape set: committed %+v, derived %+v", i, c, w)
		}
		if c.Rows%128 == 0 && c.ScaleCols%4 == 0 {
			sawNoPadding = true
		}
		if c.Rows%128 != 0 && c.ScaleCols%4 != 0 {
			sawBothPadded = true
		}

		input, err := hex.DecodeString(c.InputHex)
		if err != nil {
			t.Fatalf("case %s: bad input hex: %v", c.Name, err)
		}
		packed, err := hex.DecodeString(c.PackedHex)
		if err != nil {
			t.Fatalf("case %s: bad packed hex: %v", c.Name, err)
		}
		if !bytes.Equal(input, nvfp4GoldenInput(c.Rows, c.ScaleCols)) {
			t.Fatalf("case %s: committed input no longer matches the documented derivation — the fixture was hand-edited", c.Name)
		}

		// Fixture integrity: committed bytes must equal the pinned upstream formula applied
		// to the committed input, padding included.
		ref := nvfp4GoldenSwizzle(c.Rows, c.ScaleCols, input)
		if !bytes.Equal(ref, packed) {
			t.Fatalf("case %s: committed swizzled bytes disagree with the pinned upstream formula (first diff at offset %d)",
				c.Name, firstNVFP4DiffOffset(ref, packed))
		}
		// The regression witness: production must reproduce the pinned reference bytes
		// exactly, including the zero-filled padding that completes the final tiles.
		gotPacked, err := SwizzleNVFP4Scales(c.Rows, c.ScaleCols, input)
		if err != nil {
			t.Fatalf("case %s: SwizzleNVFP4Scales: %v", c.Name, err)
		}
		if !bytes.Equal(gotPacked, packed) {
			t.Fatalf("case %s: SwizzleNVFP4Scales drifted from the pinned reference bytes (first diff at offset %d) — "+
				"tile/stride mismatch; do not regenerate around it, fix the offset math",
				c.Name, firstNVFP4DiffOffset(gotPacked, packed))
		}

		// Padding must be exactly zero: walk the padded grid through the pinned formula and
		// require every out-of-range slot to read 0 in the production output.
		paddedRows := ((c.Rows + 127) / 128) * 128
		paddedCols := ((c.ScaleCols + 3) / 4) * 4
		for pr := 0; pr < paddedRows; pr++ {
			for pc := 0; pc < paddedCols; pc++ {
				if pr >= c.Rows || pc >= c.ScaleCols {
					if gotPacked[nvfp4ScaleGoldenOffset(pr, pc, paddedCols/4)] != 0 {
						t.Fatalf("case %s: padding at padded[%d,%d] = %d, want 0",
							c.Name, pr, pc, gotPacked[nvfp4ScaleGoldenOffset(pr, pc, paddedCols/4)])
					}
				}
			}
		}

		// Round trip through the production inverse: unswizzling the pinned bytes must strip
		// the padding and restore the input — the acceptance path a real loader takes.
		back, err := UnswizzleNVFP4Scales(c.Rows, c.ScaleCols, gotPacked)
		if err != nil {
			t.Fatalf("case %s: UnswizzleNVFP4Scales: %v", c.Name, err)
		}
		if !bytes.Equal(back, input) {
			t.Fatalf("case %s: unswizzle(swizzle(input)) did not restore the input", c.Name)
		}
	}

	// Coverage guard: keep the no-padding tile and the both-axes-padded case in the fixture,
	// or it silently stops guarding the padding completion logic.
	if !sawNoPadding || !sawBothPadded {
		t.Errorf("fixture lost coverage: sawNoPadding=%v sawBothPadded=%v", sawNoPadding, sawBothPadded)
	}
}

// firstNVFP4DiffOffset returns the first index where a and b differ, or len(a) when they match.
func firstNVFP4DiffOffset(a, b []byte) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}
