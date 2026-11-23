package compute

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

func qsaPageIndexProbes(ps int64, rng *rand.Rand) []int32 {
	const top = int64(math.MaxInt32)
	lastPage := top - top%ps
	cands := []int64{0, 1, ps - 1, ps, ps + 1, 2*ps - 1, 2 * ps, top, top - 1, lastPage, lastPage - 1, top - ps, top - ps + 1}
	for range 6 {
		cands = append(cands, rng.Int64N(top+1))
	}
	var out []int32
	for _, c := range cands {
		if c >= 0 && c <= top {
			out = append(out, int32(c))
		}
	}
	return out
}

func checkQSAPageDivisor(t *testing.T, ps uint32, rng *rand.Rand) {
	t.Helper()
	d, err := NewQSAPageDivisor(ps)
	if err != nil {
		t.Fatalf("page size %d: %v", ps, err)
	}
	shift := uint32(0)
	for uint64(1)<<shift < uint64(ps) {
		shift++
	}
	n := (uint64(1) << shift) - uint64(ps)
	if wantMul := uint32((n<<32)/uint64(ps)) + 1; d.Shift != shift || d.Mul != wantMul {
		t.Fatalf("page size %d: shift %d mul %d, want shift %d mul %d", ps, d.Shift, d.Mul, shift, wantMul)
	}
	for _, idx := range qsaPageIndexProbes(int64(ps), rng) {
		page, off := d.Split(idx)
		if page != idx/int32(ps) || off != idx%int32(ps) {
			t.Fatalf("page size %d index %d: got (%d, %d), want (%d, %d)", ps, idx, page, off, idx/int32(ps), idx%int32(ps))
		}
	}
}

func TestQSADecodePageMathIdentity(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	for ps := uint32(1); ps <= 1<<14; ps++ {
		checkQSAPageDivisor(t, ps, rng)
	}
	for k := 0; k <= 31; k++ {
		for _, delta := range []int64{-1, 0, 1} {
			ps := int64(1)<<k + delta
			if ps >= 1 && ps <= 1<<31 {
				checkQSAPageDivisor(t, uint32(ps), rng)
			}
		}
	}
	for range 4000 {
		checkQSAPageDivisor(t, uint32(rng.Int64N(1<<31))+1, rng)
	}

	d, _ := NewQSAPageDivisor(48)
	for _, idx := range []int32{-1, -48, math.MinInt32} {
		if page, off := d.Split(idx); page != 0 || off != 0 {
			t.Fatalf("negative index %d must clamp to slot (0, 0), got (%d, %d)", idx, page, off)
		}
	}
	for _, bad := range []uint32{0, 1<<31 + 1, math.MaxUint32} {
		if _, err := NewQSAPageDivisor(bad); !errors.Is(err, ErrQSADecodeShape) {
			t.Fatalf("page size %d: want ErrQSADecodeShape, got %v", bad, err)
		}
	}
}

func TestQSADecodePageTableOracle(t *testing.T) {
	const ps, blocks = 16, 6
	table := [][]int32{
		{3, 0, 5, -1},
		{2, 7, 1, 4},
	}
	pt, err := NewQSAPageTable(ps, blocks, table)
	if err != nil {
		t.Fatal(err)
	}
	fixed := []struct {
		req, idx int32
		want     QSAPageSlot
	}{
		{0, 0, QSAPageSlot{3, 0, true}},
		{0, 17, QSAPageSlot{0, 1, true}},
		{0, 47, QSAPageSlot{5, 15, true}},
		{0, 48, QSAPageSlot{-1, 0, false}},
		{0, 64, QSAPageSlot{-1, 0, false}},
		{1, 16, QSAPageSlot{7, 0, false}},
		{1, 63, QSAPageSlot{4, 15, true}},
		{-1, 5, QSAPageSlot{-1, 5, false}},
		{2, 5, QSAPageSlot{-1, 5, false}},
		{0, -1, QSAPageSlot{-1, 0, false}},
		{1, math.MaxInt32, QSAPageSlot{-1, 15, false}},
	}
	for _, f := range []QSAKVFormat{QSAKVNVFP4, QSAKVFP8, QSAKVBF16} {
		for _, c := range fixed {
			if got := pt.Locate(f, c.req, c.idx); got != c.want {
				t.Fatalf("%v request %d index %d: got %+v, want %+v", f, c.req, c.idx, got, c.want)
			}
		}
		for req := int32(-1); req <= 2; req++ {
			for idx := int32(-3); idx <= 80; idx++ {
				want := QSAPageSlot{Page: -1}
				if idx >= 0 {
					want.Offset = idx % ps
				}
				if req >= 0 && req < 2 && idx >= 0 && int(idx/ps) < len(table[req]) {
					want.Page = table[req][idx/ps]
				}
				want.Valid = want.Page >= 0 && want.Page < blocks
				if got := pt.Locate(f, req, idx); got != want {
					t.Fatalf("%v request %d index %d: got %+v, want %+v", f, req, idx, got, want)
				}
			}
		}
	}
	if _, err := NewQSAPageTable(ps, blocks, [][]int32{{1, 2}, {3}}); !errors.Is(err, ErrQSADecodeShape) {
		t.Fatalf("ragged table: want ErrQSADecodeShape, got %v", err)
	}
	if !QSAKVNVFP4.ReciprocalPageMath() || QSAKVFP8.ReciprocalPageMath() || QSAKVBF16.ReciprocalPageMath() {
		t.Fatal("only NVFP4 resolves pages with the reciprocal divisor")
	}
}

func TestQSADecodePrologueKeepsNVFP4QInRegisters(t *testing.T) {
	all := []QSADecodeStep{QSAStepLoadIndices, QSAStepLoadQRegisters, QSAStepStoreQShared, QSAStepLookupPages, QSAStepIssueKVCopies, QSAStepBarrier}
	for _, f := range []QSAKVFormat{QSAKVNVFP4, QSAKVFP8, QSAKVBF16} {
		order := QSAPrologueOrder(f)
		sorted := slices.Clone(order)
		slices.Sort(sorted)
		if !slices.Equal(sorted, all) {
			t.Fatalf("%v prologue %v is not a permutation of every step", f, order)
		}
		at := func(s QSADecodeStep) int { return slices.Index(order, s) }
		if at(QSAStepLoadQRegisters) > at(QSAStepStoreQShared) || at(QSAStepStoreQShared) > at(QSAStepBarrier) || at(QSAStepLookupPages) > at(QSAStepIssueKVCopies) {
			t.Fatalf("%v prologue %v breaks load-before-store, store-before-barrier or lookup-before-copy", f, order)
		}
		late := at(QSAStepStoreQShared) > at(QSAStepIssueKVCopies)
		if late != (f == QSAKVNVFP4) {
			t.Fatalf("%v prologue %v: Q stored after the first KV copies = %v, want %v", f, order, late, f == QSAKVNVFP4)
		}
	}
}

func TestQSADecodeSplitCountRule(t *testing.T) {
	h100 := QSAResidency{NumSMs: 132, CTAsPerSM: 2}
	bf16 := QSAResidency{NumSMs: 132, CTAsPerSM: 3}
	cases := []struct {
		f                 QSAKVFormat
		rows, heads, tile int
		r                 QSAResidency
		want              int
	}{
		{QSAKVNVFP4, 1, 1, 32, h100, 32},
		// NVFP4 keeps 8 splits where the shared normalization would shrink to 5.
		{QSAKVNVFP4, 4, 8, 10, h100, 8},
		{QSAKVFP8, 4, 8, 10, h100, 5},
		{QSAKVNVFP4, 300, 1, 9, h100, 1},
		// BF16: one CTA per SM leaves 16 tiles per split, so the wave fill (12) is kept, then shrunk to 11.
		{QSAKVBF16, 4, 8, 64, bf16, 11},
		// BF16: one CTA per SM leaves 5 tiles per split (< 8), so it wins.
		{QSAKVBF16, 4, 8, 20, bf16, 4},
		{QSAKVFP8, 1, 1, 1, h100, 1},
	}
	for _, c := range cases {
		got, err := QSANumSplits(c.f, c.rows, c.heads, c.tile, c.r)
		if err != nil || got != c.want {
			t.Fatalf("%v rows %d heads %d tiles %d: got %d (%v), want %d", c.f, c.rows, c.heads, c.tile, got, err, c.want)
		}
	}

	rng := rand.New(rand.NewPCG(3, 5))
	for range 20000 {
		f := QSAKVFormat(rng.IntN(3))
		rows, heads, tiles := 1+rng.IntN(64), 1+rng.IntN(16), 1+rng.IntN(256)
		r := QSAResidency{NumSMs: 1 + rng.IntN(200), CTAsPerSM: 1 + rng.IntN(4)}
		got, err := QSANumSplits(f, rows, heads, tiles, r)
		if err != nil || got < 1 || got > tiles {
			t.Fatalf("%v: %d splits (%v) outside [1, %d]", f, got, err, tiles)
		}
		work := rows * heads
		if f == QSAKVNVFP4 {
			if want := min(tiles, max(1, 2*r.NumSMs/work)); got != want {
				t.Fatalf("NVFP4 work %d tiles %d sms %d: got %d, want two CTAs per SM %d", work, tiles, r.NumSMs, got, want)
			}
			continue
		}
		per := (tiles + got - 1) / got
		if got > 1 && (tiles+got-2)/(got-1) == per {
			t.Fatalf("%v: %d splits is not the fewest with %d tiles per split", f, got, per)
		}
	}
	if _, err := QSANumSplits(QSAKVFP8, 0, 1, 1, h100); !errors.Is(err, ErrQSADecodeShape) {
		t.Fatalf("zero rows: want ErrQSADecodeShape, got %v", err)
	}
	if _, err := QSANumSplits(QSAKVFormat(9), 1, 1, 1, h100); !errors.Is(err, ErrQSADecodeFormat) {
		t.Fatalf("unknown format: want ErrQSADecodeFormat, got %v", err)
	}
	for _, c := range []struct {
		f           QSAKVFormat
		width, want int
	}{{QSAKVNVFP4, 2048, 32}, {QSAKVBF16, 2048, 64}, {QSAKVFP8, 65, 2}, {QSAKVBF16, 33, 2}, {QSAKVBF16, 32, 1}} {
		if got, err := QSANumTiles(c.f, c.width); err != nil || got != c.want {
			t.Fatalf("%v width %d: %d tiles (%v), want %d", c.f, c.width, got, err, c.want)
		}
	}
}

func TestQSADecodeMergePlanThresholds(t *testing.T) {
	for _, f := range []QSAKVFormat{QSAKVNVFP4, QSAKVFP8, QSAKVBF16} {
		maxWarps := f.MergeWarps()
		for splits := 1; splits <= 4*f.MergeOnePassSplits(); splits++ {
			p, err := PlanQSAMerge(f, splits)
			if err != nil {
				t.Fatal(err)
			}
			wantWarps := maxWarps
			if splits <= maxWarps*QSAMergeDepth {
				wantWarps = (splits + QSAMergeDepth - 1) / QSAMergeDepth
			}
			if p.Warps != wantWarps || p.Needed() != (splits > 1) {
				t.Fatalf("%v %d splits: warps %d needed %v, want %d %v", f, splits, p.Warps, p.Needed(), wantWarps, splits > 1)
			}
			if p.OneWarp != (f == QSAKVNVFP4 && splits <= QSAMergeDepth) {
				t.Fatalf("%v %d splits: one-warp merge %v", f, splits, p.OneWarp)
			}
			covered := make([]int, splits)
			for w := range p.Warps {
				for k := w * p.Chunk; k < min((w+1)*p.Chunk, splits); k++ {
					covered[k]++
				}
			}
			for k, c := range covered {
				if c != 1 {
					t.Fatalf("%v %d splits: split %d merged %d times", f, splits, k, c)
				}
			}
			if onePass := p.Chunk <= QSAMergeDepth; onePass != (splits <= f.MergeOnePassSplits()) {
				t.Fatalf("%v %d splits: single group per warp = %v, one-pass capacity %d", f, splits, onePass, f.MergeOnePassSplits())
			}
		}
	}
	if QSAKVNVFP4.MergeOnePassSplits() != 64 || QSAKVBF16.MergeOnePassSplits() != 128 {
		t.Fatal("one-pass merge capacity must be 64 splits for NVFP4 and 128 for FP8/BF16")
	}
}

func qsaMergeOracle(parts [][]float32, lse []float32) []float64 {
	maxL := math.Inf(-1)
	for _, l := range lse {
		maxL = math.Max(maxL, float64(l))
	}
	out := make([]float64, len(parts[0]))
	if math.IsInf(maxL, -1) {
		return out
	}
	den := 0.0
	for k, part := range parts {
		w := math.Pow(2, float64(lse[k])-maxL)
		den += w
		for j, v := range part {
			out[j] += w * float64(v)
		}
	}
	for j := range out {
		out[j] /= den
	}
	return out
}

func TestQSADecodeMergeMatchesSoftmaxOracle(t *testing.T) {
	const dims = 8
	rng := rand.New(rand.NewPCG(13, 17))
	negInf := float32(math.Inf(-1))
	for _, f := range []QSAKVFormat{QSAKVNVFP4, QSAKVBF16} {
		for _, splits := range []int{1, 2, 3, 7, 8, 9, 16, 63, 64, 65, 100, 129, 200} {
			parts := make([][]float32, splits)
			lse := make([]float32, splits)
			for k := range parts {
				parts[k] = make([]float32, dims)
				for j := range parts[k] {
					parts[k][j] = float32(rng.NormFloat64())
				}
				lse[k] = float32(rng.NormFloat64() * 6)
				if rng.IntN(5) == 0 {
					lse[k] = negInf
				}
			}
			p, _ := PlanQSAMerge(f, splits)
			flat := QSAMergePlan{Format: f, Splits: splits, Warps: splits, Chunk: 1}
			want := qsaMergeOracle(parts, lse)
			for _, plan := range []QSAMergePlan{p, flat} {
				got, err := MergeQSASplits(plan, parts, lse)
				if err != nil {
					t.Fatal(err)
				}
				for j := range got {
					if diff := math.Abs(float64(got[j]) - want[j]); diff > 1e-5*(1+math.Abs(want[j])) {
						t.Fatalf("%v %d splits plan %+v dim %d: got %v, want %v", f, splits, plan, j, got[j], want[j])
					}
				}
			}
		}
	}

	empty := [][]float32{{1, 2}, {3, 4}, {5, 6}}
	allMasked := []float32{negInf, negInf, negInf}
	for _, f := range []QSAKVFormat{QSAKVNVFP4, QSAKVBF16} {
		p, _ := PlanQSAMerge(f, len(empty))
		got, _ := MergeQSASplits(p, empty, allMasked)
		if !slices.Equal(got, []float32{0, 0}) {
			t.Fatalf("%v all splits empty: got %v, want zeros", f, got)
		}
	}
	for _, f := range []QSAKVFormat{QSAKVNVFP4, QSAKVFP8} {
		parts := make([][]float32, 70)
		lse := make([]float32, 70)
		for k := range parts {
			parts[k] = []float32{float32(k), -float32(k) / 3}
			lse[k] = negInf
		}
		lse[41] = 2.5
		p, _ := PlanQSAMerge(f, 70)
		got, _ := MergeQSASplits(p, parts, lse)
		if !slices.Equal(got, parts[41]) {
			t.Fatalf("%v lone live split: got %v, want %v", f, got, parts[41])
		}
		lse[41], lse[3], lse[69] = 300, 0, -300
		got, _ = MergeQSASplits(p, parts, lse)
		if math.Abs(float64(got[0]-41)) > 1e-4 || math.IsNaN(float64(got[1])) {
			t.Fatalf("%v LSE spread 600: got %v, want the dominant split", f, got)
		}
	}
	p, _ := PlanQSAMerge(QSAKVNVFP4, 2)
	if _, err := MergeQSASplits(p, [][]float32{{1}, {1, 2}}, []float32{0, 0}); !errors.Is(err, ErrQSADecodeShape) {
		t.Fatalf("ragged partials: want ErrQSADecodeShape, got %v", err)
	}
	if _, err := MergeQSASplits(p, [][]float32{{1}}, []float32{0}); !errors.Is(err, ErrQSADecodeShape) {
		t.Fatalf("partial count != splits: want ErrQSADecodeShape, got %v", err)
	}
}

// qsaE2M1Oracle decodes an E2M1 code from its sign, 2-bit exponent (bias 1)
// and 1-bit mantissa fields.
func qsaE2M1Oracle(code byte) float64 {
	sign, exp, man := code>>3&1, int(code>>1&3), float64(code&1)
	v := man * 0.5
	if exp != 0 {
		v = math.Ldexp(1+man/2, exp-1)
	}
	if sign == 1 {
		return -v
	}
	return v
}

func qsaExactBF16(t *testing.T, what string, v float64) uint16 {
	t.Helper()
	b := math.Float32bits(float32(v))
	if float64(float32(v)) != v || b&0xFFFF != 0 {
		t.Fatalf("%s: %v is not exactly representable in BF16", what, v)
	}
	return uint16(b >> 16)
}

func qsaWordHalf(words [4]uint32, word, half int) uint16 {
	return uint16(words[word] >> (16 * half))
}

func TestQSADecodeE2M1ToBF16Table(t *testing.T) {
	var want [16]uint16
	for c := range byte(16) {
		v := qsaE2M1Oracle(c)
		if float32(v) != mxE2M1Values[c] || math.Signbit(v) != math.Signbit(float64(mxE2M1Values[c])) {
			t.Fatalf("E2M1 code %#x: field oracle %v disagrees with mxE2M1Values %v", c, v, mxE2M1Values[c])
		}
		want[c] = qsaExactBF16(t, "E2M1 x 2^120", math.Ldexp(v, 120))
	}
	if want[8] != 0x8000 {
		t.Fatalf("code 0x8 must decode to BF16 -0, got %#04x", want[8])
	}

	check := func(packed uint32) {
		x, pairs := QSADecodeE2M1x8(packed), QSADecodeE2M1x8Pairs(packed)
		for k := range 8 {
			wantK := want[packed>>(4*k)&0xF]
			if got := qsaWordHalf(x, k/2, k%2); got != wantK {
				t.Fatalf("packed %#08x element %d: got %#04x, want %#04x", packed, k, got, wantK)
			}
			j := k % 4
			if got := qsaWordHalf(pairs, 2*(k/4)+j%2, j/2); got != wantK {
				t.Fatalf("packed %#08x paired element %d: got %#04x, want %#04x", packed, k, got, wantK)
			}
		}
	}
	rng := rand.New(rand.NewPCG(19, 23))
	fillers := []uint32{0, 0xFFFFFFFF, 0x88888888, 0x77777777, 0x89ABCDEF, 0x01234567}
	for range 8 {
		fillers = append(fillers, rng.Uint32())
	}
	for _, fill := range fillers {
		for k := range 8 {
			for c := range uint32(16) {
				check(fill&^(0xF<<(4*k)) | c<<(4*k))
			}
		}
	}
	for range 20000 {
		check(rng.Uint32())
	}
}

func TestQSADecodeE4M3ScaleAndExactProduct(t *testing.T) {
	var scale [256]uint16
	for b := range 256 {
		v := DecodeE4M3(byte(b))
		if v != v {
			continue
		}
		scale[b] = qsaExactBF16(t, "E4M3 x 2^-120", math.Ldexp(float64(v), -120))
	}
	rng := rand.New(rand.NewPCG(29, 31))
	for b := range uint32(256) {
		if byte(b)&0x7F == 0x7F {
			continue
		}
		for pos := range 4 {
			packed := rng.Uint32()&^(0xFF<<(8*pos)) | b<<(8*pos)
			got := QSADecodeE4M3x4(packed)
			for k := range 4 {
				other := byte(packed >> (8 * k))
				if other&0x7F == 0x7F {
					continue
				}
				if got[k] != scale[other] {
					t.Fatalf("packed %#08x scale %d (%#02x): got %#04x, want %#04x", packed, k, other, got[k], scale[other])
				}
			}
		}
	}
	for c := range byte(16) {
		e2m1 := qsaWordHalf(QSADecodeE2M1x8(uint32(c)), 0, 0)
		for b := range 256 {
			want := float32(qsaE2M1Oracle(c)) * DecodeE4M3(byte(b))
			if want != want {
				continue
			}
			prod := math.Float32frombits(uint32(e2m1)<<16) * math.Float32frombits(uint32(scale[b])<<16)
			if math.Float32bits(prod) != math.Float32bits(want) || math.Float32bits(prod)&0xFFFF != 0 {
				t.Fatalf("E2M1 %#x times E4M3 %#02x: BF16 product %v, want exact %v", c, b, prod, want)
			}
		}
	}
}
