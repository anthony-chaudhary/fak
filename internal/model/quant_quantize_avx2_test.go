//go:build amd64

package model

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=30ms
func TestQuantizeRowAVX2FiniteAndScalarSuffix(t *testing.T) {
	if qtier < tierAVX2 {
		t.Skip("resolved ISA does not admit AVX2")
	}
	old := qtier
	qtier = tierAVX2
	t.Cleanup(func() { qtier = old })
	rows := [][]float32{mkVec(32, 12345), mkVec(576, 314159), negRow(64, 999), halfBoundaryRow(), mixedRow(), make([]float32, 96), denormRow(32)}
	// These exceptional blocks follow a completed block. The first NaN/Inf or
	// reciprocal-overflow block must remain untouched for scalar suffix replay.
	for _, bits := range []uint32{0x7fc12345, 0x7f812345, 0xffc34567, 0x7f800000, 0xff800000, 0x00800000} {
		x := mkVec(96, 987)
		for i := 32; i < 64; i++ {
			x[i] = math.Float32frombits(bits)
		}
		rows = append(rows, x)
	}
	for _, x := range rows[:7] {
		if len(x) > 1 {
			x[0] = math.Float32frombits(0x80000000)
		}
	}
	for ri, x := range rows {
		before := append([]float32(nil), x...)
		n := len(x) / qBlk
		q, scales := make([]int8, len(x)+2), make([]float32, n+2)
		for i := range q {
			q[i] = 93
		}
		for i := range scales {
			scales[i] = 19
		}
		done := quantizeRowAsm256(&x[0], &q[1], &scales[1], n)
		wantDone := n
		if ri >= 7 {
			wantDone = 1
		}
		if done != wantDone {
			t.Fatalf("row%d done%d want%d", ri, done, wantDone)
		}
		for _, v := range q[1+done*qBlk : len(q)-1] {
			if v != 93 {
				t.Fatal("declined suffix codes were written")
			}
		}
		for _, v := range scales[1+done : len(scales)-1] {
			if v != 19 {
				t.Fatal("declined suffix scales were written")
			}
		}
		quantizeRowQ8scalar(x[done*qBlk:], q[1+done*qBlk:len(q)-1], scales[1+done:len(scales)-1], n-done)
		wantQ, wantD := make([]int8, len(x)), make([]float32, n)
		quantizeRowQ8scalar(x, wantQ, wantD, n)
		compare := func(label string, gotQ []int8, gotD []float32) {
			t.Helper()
			for i := range wantQ {
				if gotQ[i] != wantQ[i] {
					t.Fatalf("%s row%d code%d got%d want%d", label, ri, i, gotQ[i], wantQ[i])
				}
			}
			for i := range wantD {
				if math.Float32bits(gotD[i]) != math.Float32bits(wantD[i]) {
					t.Fatalf("%s row%d scale%d bits differ", label, ri, i)
				}
			}
		}
		compare("direct prefix+suffix", q[1:len(q)-1], scales[1:len(scales)-1])
		quantizeRowQ8(x, q[1:len(q)-1], scales[1:len(scales)-1], n)
		compare("ordinary dispatcher", q[1:len(q)-1], scales[1:len(scales)-1])
		if q[0] != 93 || q[len(q)-1] != 93 || scales[0] != 19 || scales[len(scales)-1] != 19 {
			t.Fatal("output canary overwritten")
		}
		for i := range x {
			if math.Float32bits(x[i]) != math.Float32bits(before[i]) {
				t.Fatal("input mutated")
			}
		}
	}
}

// fak-test:runtime fast est=20ms
func TestQuantizeRowAVX2ShapeFallbackAndOrdinaryPanel(t *testing.T) {
	if qtier < tierAVX2 {
		t.Skip("resolved ISA does not admit AVX2")
	}
	oldTier, oldHook := qtier, q8PanelBuildHook
	qtier = tierAVX2
	t.Cleanup(func() { qtier, q8PanelBuildHook = oldTier, oldHook })
	for _, shape := range [][3]int{{63, 64, 2}, {64, 63, 2}, {64, 64, 1}} {
		run := func(f func([]float32, []int8, []float32, int)) (q []int8, d []float32, panicked bool) {
			q, d = make([]int8, shape[1]), make([]float32, shape[2])
			defer func() { panicked = recover() != nil }()
			f(mkVec(shape[0], 10), q, d, 2)
			return
		}
		q, d, p := run(quantizeRowQ8)
		wantQ, wantD, wantP := run(quantizeRowQ8scalar)
		if p != wantP || !p {
			t.Fatal("short-input panic behavior changed")
		}
		for i := range q {
			if q[i] != wantQ[i] {
				t.Fatal("short-input partial code mutation changed")
			}
		}
		for i := range d {
			if math.Float32bits(d[i]) != math.Float32bits(wantD[i]) {
				t.Fatal("short-input partial scale mutation changed")
			}
		}
	}
	const rows, width = 19, 256
	x := mkVec(rows*width, 123)
	qp := &q8Panel{}
	calls := 0
	q8PanelBuildHook = func(p, w int) {
		calls++
		if p != rows || w != width {
			t.Fatal("panel observation shape changed")
		}
	}
	quantizeBatchPanelInto(qp, x, rows, width)
	wantQ, wantD := make([]int8, len(x)), make([]float32, len(x)/qBlk)
	quantizeRowQ8scalar(x, wantQ, wantD, len(wantD))
	for i := range wantQ {
		if qp.q[i] != wantQ[i] {
			t.Fatal("ordinary panel code differs")
		}
	}
	for i := range wantD {
		if math.Float32bits(qp.d[i]) != math.Float32bits(wantD[i]) {
			t.Fatal("ordinary panel scale differs")
		}
	}
	clear(x)
	quantizeBatchPanelInto(qp, x, rows, width)
	for _, v := range qp.q {
		if v != 0 {
			t.Fatal("reused zero panel leaked codes")
		}
	}
	for _, v := range qp.d {
		if math.Float32bits(v) != 0 {
			t.Fatal("zero panel scale changed")
		}
	}
	if calls != 2 {
		t.Fatal("panel observation count changed")
	}
}
