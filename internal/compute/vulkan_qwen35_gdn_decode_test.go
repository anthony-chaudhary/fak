//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"math"
	"testing"
)

func TestVulkanQwen35GDNDecodeMatchesCPUOracleInPlace(t *testing.T) {
	be := vk(t)
	hidden, nK, nV, kHd, vHd, kernel := 7, 1, 2, 3, 65, 3
	keyDim, valueDim := nK*kHd, nV*vHd
	convDim := 2*keyDim + valueDim
	seq := uint64(0x9680)
	vec := func(n int, scale float32) []float32 {
		out := make([]float32, n)
		for i := range out {
			seq = seq*6364136223846793005 + 1442695040888963407
			out[i] = (float32(uint32(seq>>32))/float32(uint64(1)<<32) - .5) * scale
		}
		return out
	}
	upload := func(shape []int, data []float32, class MemoryClass, name string) Tensor {
		return be.UploadClass(NewF32(Default(), shape, data), F32, class, name)
	}
	xData := vec(hidden, .4)
	qkvData, zData := vec(convDim*hidden, .2), vec(valueDim*hidden, .2)
	bData, aData := vec(nV*hidden, .2), vec(nV*hidden, .2)
	convData, aLogData := vec(convDim*kernel, .2), vec(nV, .2)
	for i := range aLogData {
		aLogData[i] -= .5
	}
	dtData, normData, outData := vec(nV, .2), vec(vHd, .1), vec(hidden*valueDim, .2)
	for i := range normData {
		normData[i] += 1
	}
	convStateData := make([]float32, (kernel-1)*convDim)
	recStateData := make([]float32, nV*kHd*vHd)
	x := upload([]int{hidden}, xData, MemoryActivation, "gdn input")
	qkv := upload([]int{convDim, hidden}, qkvData, MemoryWeights, "gdn qkv")
	z := upload([]int{valueDim, hidden}, zData, MemoryWeights, "gdn z")
	b := upload([]int{nV, hidden}, bData, MemoryWeights, "gdn b")
	a := upload([]int{nV, hidden}, aData, MemoryWeights, "gdn a")
	conv := upload([]int{convDim, kernel}, convData, MemoryWeights, "gdn conv")
	aLog := upload([]int{nV}, aLogData, MemoryWeights, "gdn alog")
	dt := upload([]int{nV}, dtData, MemoryWeights, "gdn dt")
	norm := upload([]int{vHd}, normData, MemoryWeights, "gdn norm")
	outW := upload([]int{hidden, valueDim}, outData, MemoryWeights, "gdn out")
	convState := upload([]int{kernel - 1, convDim}, convStateData, MemoryKVCache, "gdn conv state")
	recState := upload([]int{nV, kHd, vHd}, recStateData, MemoryKVCache, "gdn recurrent state")
	oldC, oldR := convState.Buf(), recState.Buf()

	got, nextC, nextR, err := be.Qwen35GDNDecode(x, qkv, z, b, a, conv, aLog, dt, norm, outW, convState, recState, nK, nV, kHd, vHd, kernel, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	if nextC.Buf() != oldC || nextR.Buf() != oldR {
		t.Fatal("persistent state identity changed")
	}
	matvec := func(w []float32, rows int, in []float32) []float32 {
		y := make([]float32, rows)
		for r := 0; r < rows; r++ {
			for c, xv := range in {
				y[r] += w[r*len(in)+c] * xv
			}
		}
		return y
	}
	mixed, zh := matvec(qkvData, convDim, xData), matvec(zData, valueDim, xData)
	bh, ah := matvec(bData, nV, xData), matvec(aData, nV, xData)
	core, wantC, wantR := qwen35GDNPreprojectedOracle(mixed, zh, bh, ah, convData, aLogData, dtData, normData, convStateData, recStateData, 1, nK, nV, kHd, vHd, kernel, 1e-5)
	want := matvec(outData, hidden, core)
	closeVec := func(name string, got, want []float32) {
		if len(got) != len(want) {
			t.Fatalf("%s length=%d want %d", name, len(got), len(want))
		}
		for i := range got {
			if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) || math.IsNaN(float64(want[i])) || math.IsInf(float64(want[i]), 0) {
				t.Fatalf("%s[%d] must be finite: got=%g want=%g", name, i, got[i], want[i])
			}
			if math.Abs(float64(got[i]-want[i])) > 3e-4 {
				t.Fatalf("%s[%d]=%g want %g", name, i, got[i], want[i])
			}
		}
	}
	closeVec("output", be.Read(got), want)
	closeVec("conv state", be.Read(nextC), wantC)
	closeVec("recurrent state", be.Read(nextR), wantR)
}
