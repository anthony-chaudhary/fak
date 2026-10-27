package model

import (
	"math"
	"slices"
	"testing"
)

// Independent copy of the original serial codec; do not use the new block leaf.
func serialKVQ8BulkReference(src []float32) KVQuantQ8_0 {
	q := KVQuantQ8_0{N: len(src)}
	if len(src) == 0 {
		return q
	}
	blocks := (len(src) + KVQuantQ8_0BlockSize - 1) / KVQuantQ8_0BlockSize
	q.Scale = make([]float32, blocks)
	q.Codes = make([]int8, len(src))
	for b := 0; b < blocks; b++ {
		lo, hi := b*KVQuantQ8_0BlockSize, (b+1)*KVQuantQ8_0BlockSize
		if hi > len(src) {
			hi = len(src)
		}
		var maxAbs float32
		for _, v := range src[lo:hi] {
			abs := v
			if abs < 0 {
				abs = -abs
			}
			if abs > maxAbs {
				maxAbs = abs
			}
		}
		if maxAbs == 0 {
			q.Scale[b] = 0
			continue
		}
		scale := maxAbs / 127.0
		q.Scale[b] = scale
		invScale := 1.0 / float64(scale)
		for i := lo; i < hi; i++ {
			c := int(math.Round(float64(src[i]) * invScale))
			if c < -128 {
				c = -128
			} else if c > 127 {
				c = 127
			}
			q.Codes[i] = int8(c)
		}
	}
	return q
}

func assertKVQ8BulkBits(t *testing.T, got, want KVQuantQ8_0) {
	t.Helper()
	if got.N != want.N || len(got.Scale) != len(want.Scale) || !slices.Equal(got.Codes, want.Codes) {
		t.Fatal("KV code geometry or bytes changed")
	}
	if (got.Codes == nil) != (want.Codes == nil) || (got.Scale == nil) != (want.Scale == nil) {
		t.Fatal("empty storage representation changed")
	}
	for b := range want.Scale {
		if math.Float32bits(got.Scale[b]) != math.Float32bits(want.Scale[b]) {
			t.Fatalf("scale bits changed at block %d", b)
		}
	}
}

func TestKVQ8BulkParallelPreservesSerialCodec(t *testing.T) {
	oldWorkers, oldSource := currentWorkerBudget()
	t.Cleanup(func() {
		workerBudgetMu.Lock()
		numWorkers, workerBudgetSource = oldWorkers, oldSource
		workerBudgetMu.Unlock()
	})
	for _, workers := range []int{1, 4} {
		if err := SetWorkers(workers); err != nil {
			t.Fatal(err)
		}
		for _, n := range []int{0, 1, 31, 32, 33, 97, 65539} {
			for _, exceptional := range []bool{false, true} {
				src := make([]float32, n)
				for i := range src {
					if i%64 >= 32 {
						src[i] = float32(i%257-128) / 31
					}
				}
				if exceptional {
					for i, bits := range []uint32{0x80000000, 0x00000001, 0x7f800000, 0xff800000, 0x7fc12345, 0x7fa12345, 0x7f7fffff} {
						if pos := i * 32; pos < n {
							src[pos] = math.Float32frombits(bits)
						}
					}
				}
				before := slices.Clone(src)
				want := serialKVQ8BulkReference(src)
				assertKVQ8BulkBits(t, QuantizeKVQ8_0(src), want)
				assertKVQ8BulkBits(t, quantizeKVQ8_0Bulk(src), want)
				for i := range src {
					if math.Float32bits(src[i]) != math.Float32bits(before[i]) {
						t.Fatal("source mutated")
					}
				}
			}
		}
	}
}

func TestPackedKVPrefillBulkAppendKeepsPrefixAndOwnership(t *testing.T) {
	oldWorkers, oldSource := currentWorkerBudget()
	t.Cleanup(func() {
		workerBudgetMu.Lock()
		numWorkers, workerBudgetSource = oldWorkers, oldSource
		workerBudgetMu.Unlock()
	})
	if err := SetWorkers(4); err != nil {
		t.Fatal(err)
	}
	const width, rows = 64, 1025 // Whole-panel work exceeds the existing threshold.
	prefix, panel := make([]float32, 2*width), make([]float32, rows*width)
	for i := range prefix {
		prefix[i] = float32(i%13-6) / 7
	}
	for i := range panel {
		panel[i] = float32(i%31-15) / 19
	}
	before := slices.Clone(panel)
	p := kvPackedRow{width: width}
	p.appendRow(prefix[:width])
	p.appendRow(prefix[width:])
	want := serialKVQ8BulkReference(panel)
	want.Codes = append(slices.Clone(p.codes), want.Codes...)
	want.Scale = append(slices.Clone(p.scales), want.Scale...)
	want.N += len(prefix)
	p.appendRowsBulk(panel, rows, width)
	assertKVQ8BulkBits(t, KVQuantQ8_0{N: len(p.codes), Scale: p.scales, Codes: p.codes}, want)
	if p.len() != rows+2 || !slices.Equal(panel, before) {
		t.Fatal("bulk append changed positions or input")
	}
	// Cache append owns its completed packing, not the projection panel.
	clear(panel)
	assertKVQ8BulkBits(t, KVQuantQ8_0{N: len(p.codes), Scale: p.scales, Codes: p.codes}, want)
}

func TestPackedKVReservedBulkReusesOnlyOwnedCapacity(t *testing.T) {
	const width, rows = 64, 1025
	for _, reserve := range []bool{false, true} {
		p := kvPackedRow{width: width}
		prefix := make([]float32, width)
		for i := range prefix {
			prefix[i] = float32(i%13-6) / 7
		}
		p.appendRow(prefix)
		if reserve {
			p.reserve(rows)
			// A previous truncation may leave nonzero bytes in spare capacity.
			for i := len(p.codes); i < cap(p.codes); i++ {
				p.codes[:cap(p.codes)][i] = 91
			}
			for i := len(p.scales); i < cap(p.scales); i++ {
				p.scales[:cap(p.scales)][i] = 23
			}
		}
		codeStart, scaleStart := &p.codes[0], &p.scales[0]
		codeCap, scaleCap := cap(p.codes), cap(p.scales)
		src := make([]float32, rows*width)
		for i := range src {
			if i%64 >= 32 {
				src[i] = float32(i%43-21) / 11
			}
		}
		before := slices.Clone(src)
		want := serialKVQ8BulkReference(src)
		want.Codes = append(slices.Clone(p.codes), want.Codes...)
		want.Scale = append(slices.Clone(p.scales), want.Scale...)
		want.N += width
		if got := p.appendReservedBulk(src); got != reserve {
			t.Fatalf("reserved admission=%t want%t", got, reserve)
		}
		if !reserve {
			// No-capacity refusal must leave the old allocation/growth route.
			if len(p.codes) != width || len(p.scales) != width/32 {
				t.Fatal("capacity decline published a tail")
			}
			p.appendRowsBulk(src, rows, width)
		} else if &p.codes[0] != codeStart || &p.scales[0] != scaleStart || cap(p.codes) != codeCap || cap(p.scales) != scaleCap {
			t.Fatal("reserved encoding replaced or expanded the owner's allocation")
		}
		assertKVQ8BulkBits(t, KVQuantQ8_0{N: len(p.codes), Codes: p.codes, Scale: p.scales}, want)
		if !slices.Equal(src, before) {
			t.Fatal("reserved codec changed source")
		}
	}
}

func TestPackedKVReservedBulkAliasKeepsDefensiveSnapshot(t *testing.T) {
	const width = 64
	p := kvPackedRow{width: width, codes: make([]int8, width, 3*width), scales: make([]float32, width/32, 2*width)}
	for i := range p.codes {
		p.codes[i] = int8(i%17 - 8)
	}
	backing := p.scales[:cap(p.scales)]
	for i := range backing {
		backing[i] = float32(i%29-14) / 13
	}
	// The input overlaps the future scale tail through a legal float32 view.
	src := backing[len(p.scales) : len(p.scales)+width]
	beforeCodes, beforeScales := slices.Clone(p.codes[:cap(p.codes)]), slices.Clone(backing)
	want := serialKVQ8BulkReference(src)
	want.Codes = append(slices.Clone(p.codes), want.Codes...)
	want.Scale = append(slices.Clone(p.scales), want.Scale...)
	want.N += width
	if p.appendReservedBulk(src) {
		t.Fatal("aliased input admitted as owned output")
	}
	if !slices.Equal(p.codes[:cap(p.codes)], beforeCodes) || !slices.Equal(backing, beforeScales) {
		t.Fatal("alias decline changed a prefix or spare tail")
	}
	p.appendRowsBulk(src, 1, width)
	assertKVQ8BulkBits(t, KVQuantQ8_0{N: len(p.codes), Codes: p.codes, Scale: p.scales}, want)
}

func TestPackedKVColdBulkTakesCompletedFreshPacking(t *testing.T) {
	const width = 64
	src := make([]float32, 17*width)
	for i := range src {
		src[i] = float32(i%31-15) / 23
	}
	want := serialKVQ8BulkReference(src)
	p := kvPackedRow{width: width}
	p.appendRowsBulk(src, 17, width)
	assertKVQ8BulkBits(t, KVQuantQ8_0{N: len(p.codes), Codes: p.codes, Scale: p.scales}, want)
	clear(src)
	assertKVQ8BulkBits(t, KVQuantQ8_0{N: len(p.codes), Codes: p.codes, Scale: p.scales}, want)
}
