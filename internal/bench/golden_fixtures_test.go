package bench

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"os"
	"testing"
)

// golden_fixtures_test.go — layout-guard campaign leaf (flow ticket lg-golden-fixtures):
// pinned input-to-packed byte fixtures for RepackQ4KInterleaved at widths 1/4/8, including
// the short tail group. The fixtures are the regression teeth against tensor layout &
// stride mismatches: any future drift in q4kInterleave's offset math (a lane off by one,
// a tail-group base computed with width instead of lanes, a qs row stride that forgot the
// head) changes the packed bytes and fails here even when round-trip and histogram still
// pass, because those two witnesses cannot see a permutation that is consistently wrong
// in both directions.
//
// PROVENANCE. The pattern source is llama.cpp @571d0d540df04f25298d0e159e520d9fc62ed121,
// ggml/src/ggml-cpu/repack.cpp (sha256 b3a1d6953fc61f2a4088337384166fb5102db617afe37a4b92f7cf07992a9f8a),
// fetched 2026-11-03: repack_q4_K_to_q4_K_8_bl (:3231), make_block_q4_Kx8 (:2836), picker
// ggml_repack_get_optimal_repack_type (~:4520). What this fixture pins is fak's documented
// column-interleaved BYTE-PERMUTATION contract (the LAYOUT block in q4krepack.go): rows
// grouped `width` at a time, per super-block [d x w][dmin x w][scales x w][qs byte j x w],
// tail group truncated to the remaining rows. That is NOT byte-identical to llama.cpp's
// native block_q4_Kx8, which 6-bit-repacks the scale fields across lanes, chunk-interleaves
// qs in 8-byte units, and refuses tensors whose row count is not a multiple of 8; fak keeps
// the GGUF 12-byte scale fields intact per lane so the resident dequant arithmetic is
// unchanged. See testdata/README.md for the full record.
//
// DERIVATION. The expected packed bytes come from q4kGoldenReference below — an
// independently written encoding of the LAYOUT contract (group-first slice copies, not the
// production row-first scatter) — never from RepackQ4KInterleaved, so the fixture cannot be
// laundered by production drift. Inputs come from a deterministic LCG documented in
// testdata/README.md. Regenerate with FAK_UPDATE_GOLDEN=1 (repo convention); verify with a
// plain run.

const q4kGoldenFixturePath = "testdata/q4k_interleave_golden.json"

const q4kGoldenSchema = "fak.q4k-interleave-golden.v1"

type q4kGoldenProvenance struct {
	Reference      string `json:"reference"`
	SourceSHA      string `json:"source_sha"`
	SourceSHA256   string `json:"source_sha256"`
	Fetched        string `json:"fetched"`
	LayoutContract string `json:"layout_contract"`
	Divergence     string `json:"divergence_from_upstream"`
	DerivationCmd  string `json:"derivation_cmd"`
}

type q4kGoldenCase struct {
	Name     string `json:"name"`
	Out      int    `json:"out"`
	In       int    `json:"in"`
	Width    int    `json:"width"`
	InputHex string `json:"input_hex"`
	PackedHex string `json:"packed_hex"`
}

type q4kGoldenFixture struct {
	Schema     string               `json:"schema"`
	Provenance q4kGoldenProvenance  `json:"provenance"`
	Cases      []q4kGoldenCase      `json:"cases"`
}

// q4kGoldenCaseShapes is the pinned coverage set: every width the runtime picker can emit
// (TestQ4KWidthForKernel maps kernels onto exactly 1/4/8), each with a different tail
// structure. out%width != 0 in three of four cases because the short tail group is where
// the offset math differs; in=512 in two cases proves multi-super-block bases.
var q4kGoldenCaseShapes = []struct {
	name         string
	out, in      int
	width        int
}{
	{"width1-identity", 1, 256, 1},
	{"width4-full-group-plus-tail", 5, 512, 4},
	{"width8-pure-tail", 5, 512, 8},
	{"width8-full-group-plus-tail", 9, 256, 8},
}

// q4kGoldenLCGByte is the fixture input source: a 64-bit LCG with Knuth's constants, one
// draw per byte, XOR-folding the two 32-bit halves so low-bit structure does not survive.
// Fixed constants, no math/rand version surface: the committed hex must regenerate on any
// Go release.
func q4kGoldenLCGByte(st *uint64) byte {
	*st = *st*6364136223846793005 + 1442695040888963407
	return byte(*st ^ (*st>>32))
}

// q4kGoldenF16Bits draws a finite half-precision value: sign random, exponent field held
// in [12,16] (never 0 or 31, so never subnormal/Inf/NaN), mantissa random. Same bounded
// range randF16Bits uses, so a fixture record could feed the GEMV without NaN vacuity.
func q4kGoldenF16Bits(st *uint64) uint16 {
	sign := uint16(q4kGoldenLCGByte(st)&1) << 15
	exp := uint16(12+q4kGoldenLCGByte(st)%5) << 10
	mant := uint16(q4kGoldenLCGByte(st)) & 0x3FF
	return sign | exp | mant
}

// q4kGoldenInput builds one case's deterministic row-major Q4_K payload: per record, d and
// dmin as finite f16 (three LCG draws each), then the 12 scale/min bytes and 128 nibble
// bytes straight from the stream (any byte pattern is legal for the layout question).
func q4kGoldenInput(out, in, width int) []byte {
	st := uint64(0x5285)
	st ^= uint64(out) * 0x9E3779B97F4A7C15
	st ^= uint64(in) * 0xC2B2AE3D27D4EB4F
	st ^= uint64(width) * 0x5285

	nblk := in / Q4KSuperBlockWeights
	raw := make([]byte, out*nblk*Q4KBlockBytes)
	for i := 0; i < out*nblk; i++ {
		rec := raw[i*Q4KBlockBytes : i*Q4KBlockBytes+Q4KBlockBytes]
		var d, dmin [2]byte
		little(d[:], q4kGoldenF16Bits(&st))
		little(dmin[:], q4kGoldenF16Bits(&st))
		copy(rec[0:2], d[:])
		copy(rec[2:4], dmin[:])
		for j := 4; j < Q4KBlockBytes; j++ {
			rec[j] = q4kGoldenLCGByte(&st)
		}
	}
	return raw
}

func little(dst []byte, v uint16) {
	dst[0] = byte(v)
	dst[1] = byte(v >> 8)
}

// q4kGoldenReference is the independent encoding of the LAYOUT contract pinned above: for
// each group g of up to `width` rows and each super-block b, copy the w lanes' fields into
// the documented slots. Structurally different from q4kInterleave on purpose — group-first
// contiguous slice copies versus the production row-first per-lane scatter — so an offset
// bug shared by neither survives into the fixture. Offsets follow the LAYOUT block verbatim:
//
//	base := groupStart + b*w*Q4KBlockBytes
//	d[l]     -> base + 2*l
//	dmin[l]  -> base + 2*w + 2*l
//	scales[l]-> base + 4*w + 12*l   (the whole 12-byte field stays contiguous per lane)
//	qs[j][l] -> base + 16*w + j*w + l
func q4kGoldenReference(raw []byte, out, in, width int) []byte {
	nblk := in / Q4KSuperBlockWeights
	rowBytes := nblk * Q4KBlockBytes
	packed := make([]byte, len(raw))
	for g := 0; g*width < out; g++ {
		lanes := width
		if rest := out - g*width; rest < lanes {
			lanes = rest
		}
		groupStart := g * width * rowBytes
		for b := 0; b < nblk; b++ {
			base := groupStart + b*lanes*Q4KBlockBytes
			for l := 0; l < lanes; l++ {
				src := (g*width + l)*rowBytes + b*Q4KBlockBytes
				copy(packed[base+2*l:][:2], raw[src:src+2])
				copy(packed[base+2*lanes+2*l:][:2], raw[src+2:src+4])
				copy(packed[base+4*lanes+q4kScaleBytes*l:][:q4kScaleBytes], raw[src+4:src+q4kHeadBytes])
			}
			qsBase := base + q4kHeadBytes*lanes
			for j := 0; j < q4kQSBytes; j++ {
				for l := 0; l < lanes; l++ {
					packed[qsBase+j*lanes+l] = raw[(g*width+l)*rowBytes+b*Q4KBlockBytes+q4kHeadBytes+j]
				}
			}
		}
	}
	return packed
}

// buildQ4KGoldenFixture derives the whole fixture from the pinned contract alone: LCG
// inputs plus the independent reference. This is the derivation command's body.
func buildQ4KGoldenFixture() q4kGoldenFixture {
	cases := make([]q4kGoldenCase, 0, len(q4kGoldenCaseShapes))
	for _, sh := range q4kGoldenCaseShapes {
		input := q4kGoldenInput(sh.out, sh.in, sh.width)
		packed := q4kGoldenReference(input, sh.out, sh.in, sh.width)
		cases = append(cases, q4kGoldenCase{
			Name:      sh.name,
			Out:       sh.out,
			In:        sh.in,
			Width:     sh.width,
			InputHex:  hex.EncodeToString(input),
			PackedHex: hex.EncodeToString(packed),
		})
	}
	return q4kGoldenFixture{
		Schema: q4kGoldenSchema,
		Provenance: q4kGoldenProvenance{
			Reference: "ggml-org/llama.cpp @ 571d0d540df04f25298d0e159e520d9fc62ed121, ggml/src/ggml-cpu/repack.cpp " +
				"(repack_q4_K_to_q4_K_8_bl :3231, make_block_q4_Kx8 :2836, picker ggml_repack_get_optimal_repack_type ~:4520)",
			SourceSHA:      "571d0d540df04f25298d0e159e520d9fc62ed121",
			SourceSHA256:   "b3a1d6953fc61f2a4088337384166fb5102db617afe37a4b92f7cf07992a9f8a",
			Fetched:        "2026-11-03",
			LayoutContract: "internal/bench/q4krepack.go LAYOUT block: rows grouped width at a time, per super-block " +
				"[d x w][dmin x w][scales x w][qs byte j x w]; tail group truncated to remaining rows; pure byte permutation",
			Divergence: "NOT byte-identical to upstream block_q4_Kx8: upstream 6-bit-repacks the scale fields across lanes, " +
				"chunk-interleaves qs in 8-byte units, and refuses row counts not divisible by 8; fak keeps GGUF scale fields " +
				"intact per lane and byte-interleaves qs so the resident dequant arithmetic is unchanged",
			DerivationCmd:  "FAK_UPDATE_GOLDEN=1 go test ./internal/bench -run TestQ4KInterleaveGoldenFixtures -count=1",
		},
		Cases: cases,
	}
}

// TestQ4KInterleaveGoldenFixtures asserts RepackQ4KInterleaved reproduces the pinned
// reference bytes exactly, at every width including the short tail group. Three-way pin:
// committed fixture vs independent reference (fixture integrity), committed fixture vs
// production repack (the regression witness), and production unrepack of the reference
// output (both directions agree on the same mapping).
func TestQ4KInterleaveGoldenFixtures(t *testing.T) {
	want := buildQ4KGoldenFixture()

	if os.Getenv("FAK_UPDATE_GOLDEN") == "1" {
		data, err := json.MarshalIndent(want, "", "  ")
		if err != nil {
			t.Fatalf("marshal golden fixture: %v", err)
		}
		if err := os.WriteFile(q4kGoldenFixturePath, append(data, '\n'), 0o644); err != nil {
			t.Fatalf("write golden fixture: %v", err)
		}
		t.Logf("updated %s (%d cases)", q4kGoldenFixturePath, len(want.Cases))
		return
	}

	gotData, err := os.ReadFile(q4kGoldenFixturePath)
	if err != nil {
		t.Fatalf("read golden fixture %s: %v (regenerate with FAK_UPDATE_GOLDEN=1)", q4kGoldenFixturePath, err)
	}
	var got q4kGoldenFixture
	if err := json.Unmarshal(gotData, &got); err != nil {
		t.Fatalf("unmarshal golden fixture: %v", err)
	}
	if got.Schema != q4kGoldenSchema {
		t.Fatalf("fixture schema = %q, want %q", got.Schema, q4kGoldenSchema)
	}
	if len(got.Cases) != len(want.Cases) {
		t.Fatalf("fixture has %d cases, derived contract has %d — coverage changed; review before regenerating",
			len(got.Cases), len(want.Cases))
	}

	seenWidths := map[int]bool{}
	tails := 0
	for i, c := range got.Cases {
		w := want.Cases[i]
		if c.Name != w.Name || c.Out != w.Out || c.In != w.In || c.Width != w.Width {
			t.Fatalf("case %d drifted from the pinned shape set: committed %+v, derived %+v", i, c, w)
		}
		seenWidths[c.Width] = true
		if c.Out%c.Width != 0 {
			tails++
		}

		input, err := hex.DecodeString(c.InputHex)
		if err != nil {
			t.Fatalf("case %s: bad input hex: %v", c.Name, err)
		}
		packed, err := hex.DecodeString(c.PackedHex)
		if err != nil {
			t.Fatalf("case %s: bad packed hex: %v", c.Name, err)
		}
		if !bytes.Equal(input, q4kGoldenInput(c.Out, c.In, c.Width)) {
			t.Fatalf("case %s: committed input no longer matches the documented LCG derivation — the fixture was hand-edited", c.Name)
		}

		// Fixture integrity: the committed bytes must equal the independent reference's
		// output for the committed input. Catches a fixture edited without re-derivation.
		ref := q4kGoldenReference(input, c.Out, c.In, c.Width)
		if !bytes.Equal(ref, packed) {
			t.Fatalf("case %s: committed packed bytes disagree with the pinned layout contract (first diff at offset %d)",
				c.Name, firstDiffOffset(ref, packed))
		}

		// The regression witness: production must reproduce the pinned reference bytes
		// exactly, padding/tail included.
		gotPacked, err := RepackQ4KInterleaved(input, c.Out, c.In, c.Width)
		if err != nil {
			t.Fatalf("case %s: RepackQ4KInterleaved: %v", c.Name, err)
		}
		if !bytes.Equal(gotPacked, packed) {
			t.Fatalf("case %s: RepackQ4KInterleaved drifted from the pinned reference bytes (first diff at offset %d) — "+
				"layout/stride mismatch; do not regenerate around it, fix the offset math",
				c.Name, firstDiffOffset(gotPacked, packed))
		}
		if c.Width == 1 && !bytes.Equal(gotPacked, input) {
			t.Fatalf("case %s: width 1 must be the identity repack", c.Name)
		}

		// Both directions agree on the same mapping: unrepacking the reference output must
		// restore the input. A repack that is consistently wrong in one direction but whose
		// inverse shares the bug would still fail the byte comparison above; this pins the
		// inverse against the fixture too.
		back, err := UnrepackQ4KInterleaved(ref, c.Out, c.In, c.Width)
		if err != nil {
			t.Fatalf("case %s: UnrepackQ4KInterleaved(reference): %v", c.Name, err)
		}
		if !bytes.Equal(back, input) {
			t.Fatalf("case %s: unrepack(repack(input)) did not restore the input", c.Name)
		}
	}

	// Coverage guard: the fixture must keep exercising every picker width and at least two
	// short-tail groups, or it silently stops guarding the code path that motivated it.
	for _, w := range []int{1, 4, 8} {
		if !seenWidths[w] {
			t.Errorf("fixture lost width-%d coverage", w)
		}
	}
	if tails < 2 {
		t.Errorf("fixture covers only %d short-tail case(s), want >= 2 — the tail group is where the offset math differs", tails)
	}
}

// firstDiffOffset returns the first index where a and b differ, or len(a) when they match.
func firstDiffOffset(a, b []byte) int {
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
