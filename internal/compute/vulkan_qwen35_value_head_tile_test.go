//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"math"
	"math/rand"
	"testing"
)

// fak-test:runtime fast est=2s
func TestVulkanQwen35ValueHeadTileMatchesHostOnBothRoutes(t *testing.T) {
	be := vk(t)
	kernel := be.VulkanDebugQwen35ValueHeadTileAvailable()
	t.Logf("device=%s value-head tile kernel=%v", be.Tier(), kernel)
	routes := []bool{false}
	if kernel {
		routes = append(routes, true)
	}
	defer be.VulkanDebugSelectQwen35ValueHeadTile(kernel)
	rng := rand.New(rand.NewSource(0x05))
	for _, useKernel := range routes {
		be.VulkanDebugSelectQwen35ValueHeadTile(useKernel)
		for _, g := range []struct{ tokens, nK, nV, d int }{{1, 16, 48, 128}, {7, 16, 48, 128}, {3, 2, 4, 64}, {300, 4, 12, 8}} {
			host := make([]float32, g.tokens*g.nV*g.d)
			for i := range host {
				host[i] = rng.Float32()*2 - 1
			}
			want, err := TileQwen35ValueHeads(host, g.tokens, g.nK, g.nV, g.d)
			if err != nil {
				t.Fatal(err)
			}
			core := be.UploadClass(NewF32(Default(), []int{g.tokens, g.nV * g.d}, host), F32, MemoryActivation, "tile core")
			kernelBefore, copyBefore := be.VulkanDebugQwen35ValueHeadTileSnapshot()
			tiled, err := be.VulkanDebugQwen35TileValueHeads(core, g.tokens, g.nK, g.nV, g.d)
			if err != nil {
				t.Fatal(err)
			}
			got := be.Read(tiled)
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("kernel=%v geometry %+v element %d = %v, want %v (the tiling is a pure copy)", useKernel, g, i, got[i], want[i])
				}
			}
			kernelAfter, copyAfter := be.VulkanDebugQwen35ValueHeadTileSnapshot()
			if useKernel && (kernelAfter != kernelBefore+1 || copyAfter != copyBefore) ||
				!useKernel && (copyAfter != copyBefore+1 || kernelAfter != kernelBefore) {
				t.Fatalf("kernel=%v route counters kernel %d->%d copy %d->%d", useKernel, kernelBefore, kernelAfter, copyBefore, copyAfter)
			}
			be.Free(core)
		}
	}
}

// fak-test:runtime fast est=2s
func TestVulkanQwen35GDNDecodeTiledOutMatchesGroupedDecode(t *testing.T) {
	be := vk(t)
	hidden, nK, nV, kHd, vHd, kernel := 9, 2, 6, 4, 32, 3
	keyDim, valueDim := nK*kHd, nV*vHd
	convDim := 2*keyDim + valueDim
	rng := rand.New(rand.NewSource(0x5050))
	vec := func(n int, scale float32) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = (rng.Float32() - .5) * scale
		}
		return out
	}
	upload := func(shape []int, data []float32, class MemoryClass) Tensor {
		return be.UploadClass(NewF32(Default(), shape, data), F32, class, "tiled gdn operand")
	}
	xData := vec(hidden, 2)
	qkvData, zData := vec(convDim*hidden, .2), vec(valueDim*hidden, 2)
	bData, aData := vec(nV*hidden, .2), vec(nV*hidden, .2)
	convData, aLogData := vec(convDim*kernel, .2), vec(nV, .2)
	for i := range aLogData {
		aLogData[i] -= .5
	}
	dtData, normData, outData := vec(nV, .2), vec(vHd, .1), vec(hidden*valueDim, 2)
	for i := range normData {
		normData[i] += 1
	}
	src, err := Qwen35TiledValueHeadSource(nK, nV)
	if err != nil {
		t.Fatal(err)
	}
	outTiledData := make([]float32, len(outData))
	for row := 0; row < hidden; row++ {
		base := row * valueDim
		for c, h := range src {
			copy(outTiledData[base+c*vHd:base+(c+1)*vHd], outData[base+h*vHd:base+(h+1)*vHd])
		}
	}
	x := upload([]int{hidden}, xData, MemoryActivation)
	qkv := upload([]int{convDim, hidden}, qkvData, MemoryWeights)
	z := upload([]int{valueDim, hidden}, zData, MemoryWeights)
	b := upload([]int{nV, hidden}, bData, MemoryWeights)
	a := upload([]int{nV, hidden}, aData, MemoryWeights)
	conv := upload([]int{convDim, kernel}, convData, MemoryWeights)
	aLog := upload([]int{nV}, aLogData, MemoryWeights)
	dt := upload([]int{nV}, dtData, MemoryWeights)
	norm := upload([]int{vHd}, normData, MemoryWeights)
	outGrouped := upload([]int{hidden, valueDim}, outData, MemoryWeights)
	outTiled := upload([]int{hidden, valueDim}, outTiledData, MemoryWeights)
	state := func() (Tensor, Tensor) {
		return upload([]int{kernel - 1, convDim}, make([]float32, (kernel-1)*convDim), MemoryKVCache),
			upload([]int{nV, kHd, vHd}, make([]float32, nV*kHd*vHd), MemoryKVCache)
	}
	convA, recA := state()
	convB, recB := state()
	var wantOut, gotOut []float32
	for step := 0; step < 3; step++ {
		want, nextConvA, nextRecA, err := be.Qwen35GDNDecode(x, qkv, z, b, a, conv, aLog, dt, norm, outGrouped, convA, recA, nK, nV, kHd, vHd, kernel, 1e-5)
		if err != nil {
			t.Fatal(err)
		}
		got, nextConvB, nextRecB, err := be.Qwen35GDNDecodeTiledOut(x, qkv, z, b, a, conv, aLog, dt, norm, outTiled, convB, recB, nK, nV, kHd, vHd, kernel, 1e-5)
		if err != nil {
			t.Fatal(err)
		}
		if nextConvB.Buf() != convB.Buf() || nextRecB.Buf() != recB.Buf() || nextConvA.Buf() != convA.Buf() || nextRecA.Buf() != recA.Buf() {
			t.Fatal("tiled decode replaced the persistent in-place state")
		}
		wantOut, gotOut = be.Read(want), be.Read(got)
		// Only the reduction order over the permuted columns differs.
		tol := 1e-6 + 1e-5*maxAbs(wantOut, make([]float32, len(wantOut)))
		if d := maxAbs(gotOut, wantOut); d > tol {
			t.Fatalf("step %d tiled vs grouped max|delta|=%g > %g", step, d, tol)
		}
		wantRec, gotRec := be.Read(recA), be.Read(recB)
		for i := range wantRec {
			if math.Float32bits(wantRec[i]) != math.Float32bits(gotRec[i]) {
				t.Fatalf("step %d recurrent state[%d] differs: the tiling must not touch the recurrence", step, i)
			}
		}
	}
	// Negative control: the tiled weight on the grouped decode must not match.
	wrong, _, _, err := be.Qwen35GDNDecode(x, qkv, z, b, a, conv, aLog, dt, norm, outTiled, convA, recA, nK, nV, kHd, vHd, kernel, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	right, _, _, err := be.Qwen35GDNDecode(x, qkv, z, b, a, conv, aLog, dt, norm, outGrouped, convB, recB, nK, nV, kHd, vHd, kernel, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	rightOut := be.Read(right)
	scale := maxAbs(rightOut, make([]float32, len(rightOut)))
	if d := maxAbs(be.Read(wrong), rightOut); d < 0.05*scale {
		t.Fatalf("tiled weight without core tiling matched (max|delta|=%g of max|out|=%g); the witness is vacuous", d, scale)
	}
	t.Logf("device=%s tiled decode parity over 3 steps, final max|out|=%g", be.Tier(), maxAbs(gotOut, make([]float32, len(gotOut))))
}
