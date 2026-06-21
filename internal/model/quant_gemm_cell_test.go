package model

import (
	"math"
	"testing"
)

func explicitQGemm8Cell(qw []int8, dw []float32, qx []int8, dx []float32, nblk, lanes int) float32 {
	var acc [16]float32
	for b := 0; b < nblk; b++ {
		wb := qw[b*qBlk : b*qBlk+qBlk]
		xb := qx[b*qBlk : b*qBlk+qBlk]
		var p [16]int32
		switch lanes {
		case 16:
			for lane := 0; lane < 16; lane++ {
				for j := 0; j < 2; j++ {
					i := 2*lane + j
					p[lane] += int32(wb[i]) * int32(xb[i])
				}
			}
		case 8:
			for lane := 0; lane < 8; lane++ {
				for _, base := range []int{0, 16} {
					for j := 0; j < 2; j++ {
						i := base + 2*lane + j
						p[lane] += int32(wb[i]) * int32(xb[i])
					}
				}
			}
		case 4:
			for lane := 0; lane < 4; lane++ {
				for _, base := range []int{0, 16} {
					for j := 0; j < 4; j++ {
						i := base + 4*lane + j
						p[lane] += int32(wb[i]) * int32(xb[i])
					}
				}
			}
		default:
			panic("bad test lane count")
		}
		s := dw[b] * dx[b]
		for lane := 0; lane < lanes; lane++ {
			acc[lane] = float32(math.FMA(float64(p[lane]), float64(s), float64(acc[lane])))
		}
	}
	switch lanes {
	case 16:
		for lane := 0; lane < 8; lane++ {
			acc[lane] += acc[lane+8]
		}
		fallthrough
	case 8:
		for lane := 0; lane < 4; lane++ {
			acc[lane] += acc[lane+4]
		}
	case 4:
	}
	acc[0] += acc[2]
	acc[1] += acc[3]
	return acc[0] + acc[1]
}

func TestQGemm8CellLaneGeometries(t *testing.T) {
	const nblk = 3
	qw := make([]int8, nblk*qBlk)
	qx := make([]int8, nblk*qBlk)
	dw := make([]float32, nblk)
	dx := make([]float32, nblk)
	for i := range qw {
		qw[i] = int8((i*37)%251 - 125)
		qx[i] = int8((i*53+17)%241 - 120)
	}
	for i := 0; i < nblk; i++ {
		dw[i] = 0.003 + float32(i)*0.0017
		dx[i] = 0.004 + float32(i)*0.0013
	}

	for _, lanes := range []int{16, 8, 4} {
		got := qgemm8cell(qw, dw, qx, dx, nblk, lanes)
		want := explicitQGemm8Cell(qw, dw, qx, dx, nblk, lanes)
		if math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("lanes=%d: got %v bits %#x, want %v bits %#x",
				lanes, got, math.Float32bits(got), want, math.Float32bits(want))
		}
	}
}

func TestQGemm8CellRejectsInvalidLaneCount(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("qgemm8cell accepted an invalid lane count")
		}
	}()
	qgemm8cell(make([]int8, qBlk), []float32{1}, make([]int8, qBlk), []float32{1}, 1, 5)
}
