//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

func cpuReferenceSwiGLUQ4K(gateRaw, upRaw []byte, out, in int, x []float32) []float32 {
	yGate := q4kVectorizedReference(gateRaw, out, in, x)
	yUp := q4kVectorizedReference(upRaw, out, in, x)
	inter := make([]float32, out)
	for i := 0; i < out; i++ {
		g := float64(yGate[i])
		u := float64(yUp[i])
		siluG := g / (1.0 + math.Exp(-g))
		inter[i] = float32(siluG * u)
	}
	return inter
}

func cpuReferenceSwiGLUQ8(codesG []int8, scalesG []float32, codesU []int8, scalesU []float32, out, in int, x []float32) []float32 {
	nblk := in / 32
	yGate := make([]float32, out)
	yUp := make([]float32, out)
	for o := 0; o < out; o++ {
		var accG, accU float64
		for b := 0; b < nblk; b++ {
			var sG, sU float64
			for i := 0; i < 32; i++ {
				idx := (o*nblk+b)*32 + i
				xi := float64(x[b*32+i])
				sG += float64(codesG[idx]) * xi
				sU += float64(codesU[idx]) * xi
			}
			accG += sG * float64(scalesG[o*nblk+b])
			accU += sU * float64(scalesU[o*nblk+b])
		}
		yGate[o] = float32(accG)
		yUp[o] = float32(accU)
	}
	inter := make([]float32, out)
	for i := 0; i < out; i++ {
		g := float64(yGate[i])
		u := float64(yUp[i])
		siluG := g / (1.0 + math.Exp(-g))
		inter[i] = float32(siluG * u)
	}
	return inter
}

func generateQ8TestData(out, in int, seed int64) ([]int8, []float32) {
	rng := rand.New(rand.NewSource(seed))
	codes := make([]int8, out*in)
	for i := range codes {
		codes[i] = int8(rng.Intn(255) - 127)
	}
	nblk := in / 32
	scales := make([]float32, out*nblk)
	for i := range scales {
		scales[i] = rng.Float32()*0.02 + 0.001
	}
	return codes, scales
}

func q6kTestRaw(out, in int, seed uint64) []byte {
	nblk := in / 256
	raw := make([]byte, out*nblk*210)
	state := seed
	for i := range raw {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		raw[i] = byte(state >> 56)
	}
	for b := 0; b < out*nblk; b++ {
		base := b * 210
		raw[base+208] = 0x00
		raw[base+209] = 0x2c
	}
	return raw
}

func TestFusedSwiGLUPipelineCache(t *testing.T) {
	if !Available() {
		t.Skip("Metal device not available")
	}

	start := time.Now()
	if err := EnsureFusedSwiGLUPipeline(); err != nil {
		t.Fatalf("first EnsureFusedSwiGLUPipeline failed: %v", err)
	}
	firstDur := time.Since(start)

	start2 := time.Now()
	if err := EnsureFusedSwiGLUPipeline(); err != nil {
		t.Fatalf("second EnsureFusedSwiGLUPipeline failed: %v", err)
	}
	cachedDur := time.Since(start2)

	t.Logf("pipeline init: first=%v, cached=%v", firstDur, cachedDur)
	if cachedDur > firstDur && cachedDur > 100*time.Microsecond {
		t.Logf("note: cached call duration=%v", cachedDur)
	}
}

func TestFusedSwiGLUParity(t *testing.T) {
	if !Available() {
		t.Skip("Metal device not available")
	}
	defer ResetQ4K()
	defer ResetQ8()

	const (
		in  = 512
		out = 512
	)

	// 1. Q4_K Fused Dequantize-GEMV-SwiGLU verification against CPU reference
	gateRaw := q4kTestRaw(out, in, 0x1234)
	upRaw := q4kTestRaw(out, in, 0x5678)
	gate := UploadQ4K(gateRaw, out, in)
	up := UploadQ4K(upRaw, out, in)
	if gate == nil || up == nil {
		t.Fatal("failed to upload Q4_K gate/up weights")
	}

	x := q4kTestVector(in, 42)
	wantQ4K := cpuReferenceSwiGLUQ4K(gateRaw, upRaw, out, in, x)

	gotQ4K := make([]float32, out)
	if !FusedDequantGEMVSwiGLU(gate, up, x, gotQ4K) {
		t.Fatal("FusedDequantGEMVSwiGLU returned false")
	}

	cosineQ4K, maxRelQ4K := q4kTestCosineMaxRel(wantQ4K, gotQ4K)
	t.Logf("Q4_K Parity: cosine=%0.8f, maxRel=%0.8f", cosineQ4K, maxRelQ4K)
	if cosineQ4K < 0.999999 {
		t.Fatalf("Q4_K cosine %0.8f < 0.999999", cosineQ4K)
	}
	if maxRelQ4K > 5e-3 {
		t.Fatalf("Q4_K maxRel %0.8f > 5e-3", maxRelQ4K)
	}

	// 2. Q8_0 Fused Dequantize-GEMV-SwiGLU verification against CPU reference
	codesG, scalesG := generateQ8TestData(out, in, 101)
	codesU, scalesU := generateQ8TestData(out, in, 202)
	gateQ8 := UploadQ8(codesG, scalesG, out, in)
	upQ8 := UploadQ8(codesU, scalesU, out, in)
	if gateQ8 == nil || upQ8 == nil {
		t.Fatal("failed to upload Q8_0 gate/up weights")
	}

	wantQ8 := cpuReferenceSwiGLUQ8(codesG, scalesG, codesU, scalesU, out, in, x)
	gotQ8 := make([]float32, out)
	if !FusedDequantGEMVSwiGLUQ8(gateQ8, upQ8, x, nil, gotQ8) {
		t.Fatal("FusedDequantGEMVSwiGLUQ8 returned false")
	}

	cosineQ8, maxRelQ8 := q4kTestCosineMaxRel(wantQ8, gotQ8)
	t.Logf("Q8_0 Parity: cosine=%0.8f, maxRel=%0.8f", cosineQ8, maxRelQ8)
	if cosineQ8 < 0.999999 {
		t.Fatalf("Q8_0 cosine %0.8f < 0.999999", cosineQ8)
	}
	if maxRelQ8 > 5e-3 {
		t.Fatalf("Q8_0 maxRel %0.8f > 5e-3", maxRelQ8)
	}

	// 3. End-to-end FusedMLPQ6DownFast verification against FusedMLPQ6Down
	downRaw := q6kTestRaw(in, out, 0x9876)
	down := UploadQ6K(downRaw, in, out)
	if down == nil {
		t.Fatal("failed to upload Q6_K down weight")
	}

	yRef := make([]float32, in)
	if !FusedMLPQ6Down(gate, up, down, x, yRef) {
		t.Fatal("unfused FusedMLPQ6Down failed")
	}

	yFast := make([]float32, in)
	if !FusedMLPQ6DownFast(gate, up, down, x, yFast) {
		t.Fatal("fused FusedMLPQ6DownFast failed")
	}

	cosineMLP, maxRelMLP := q4kTestCosineMaxRel(yRef, yFast)
	t.Logf("FusedMLPQ6DownFast vs FusedMLPQ6Down: cosine=%0.8f, maxRel=%0.8f", cosineMLP, maxRelMLP)
	if cosineMLP < 0.999999 {
		t.Fatalf("FusedMLPQ6DownFast cosine %0.8f < 0.999999", cosineMLP)
	}
}

func TestFusedSwiGLUKernel_ParityAndSpeed(t *testing.T) {
	if !Available() {
		t.Skip("Metal device not available")
	}
	defer ResetQ4K()

	const (
		H  = 1024
		Im = 2048
	)

	// Check bandwidth savings formula
	savingsBytes := FusedSwiGLUBandwidthSavingsBytes(Im)
	expectedSavings := 2 * Im * 4
	if savingsBytes != expectedSavings {
		t.Fatalf("FusedSwiGLUBandwidthSavingsBytes(%d) = %d; want %d", Im, savingsBytes, expectedSavings)
	}
	t.Logf("DRAM intermediate memory eliminated per token: %d bytes (%.2f KB)", savingsBytes, float64(savingsBytes)/1024.0)

	gateRaw := q4kTestRaw(Im, H, 0xabcd)
	upRaw := q4kTestRaw(Im, H, 0xdcba)
	gate := UploadQ4K(gateRaw, Im, H)
	up := UploadQ4K(upRaw, Im, H)
	downRaw := q6kTestRaw(H, Im, 0xef01)
	down := UploadQ6K(downRaw, H, Im)
	if gate == nil || up == nil || down == nil {
		t.Fatal("failed to upload weights")
	}

	x := q4kTestVector(H, 123)

	// Verify parity against CPU oracle for Gate+Up SwiGLU
	cpuRef := cpuReferenceSwiGLUQ4K(gateRaw, upRaw, Im, H, x)
	interFused := make([]float32, Im)
	if !FusedDequantGEMVSwiGLU(gate, up, x, interFused) {
		t.Fatal("FusedDequantGEMVSwiGLU failed")
	}
	cosOracle, maxRelOracle := q4kTestCosineMaxRel(cpuRef, interFused)
	t.Logf("SwiGLU Kernel vs CPU Oracle: cosine=%0.8f, maxRel=%0.8f", cosOracle, maxRelOracle)
	if cosOracle < 0.999999 {
		t.Fatalf("SwiGLU Kernel cosine %0.8f < 0.999999", cosOracle)
	}

	// Warmup
	yUnfused := make([]float32, H)
	yFused := make([]float32, H)
	for i := 0; i < 5; i++ {
		FusedMLPQ6Down(gate, up, down, x, yUnfused)
		FusedMLPQ6DownFast(gate, up, down, x, yFused)
	}

	// Numerical parity between fused and unfused MLP outputs
	cosMLP, maxRelMLP := q4kTestCosineMaxRel(yUnfused, yFused)
	t.Logf("FusedMLPQ6DownFast vs FusedMLPQ6Down: cosine=%0.8f, maxRel=%0.8f", cosMLP, maxRelMLP)
	if cosMLP < 0.999999 {
		t.Fatalf("FusedMLP output cosine %0.8f < 0.999999", cosMLP)
	}

	// Benchmark timing
	const iters = 30
	t0 := time.Now()
	for i := 0; i < iters; i++ {
		FusedMLPQ6Down(gate, up, down, x, yUnfused)
	}
	unfusedDur := time.Since(t0) / iters

	t1 := time.Now()
	for i := 0; i < iters; i++ {
		FusedMLPQ6DownFast(gate, up, down, x, yFused)
	}
	fusedDur := time.Since(t1) / iters

	t.Logf("MLP decode latency (H=%d, Im=%d): unfused=%v, fused=%v, speedup=%.2fx",
		H, Im, unfusedDur, fusedDur, float64(unfusedDur)/float64(fusedDur))
}

func TestFusedSwiGLUValidation(t *testing.T) {
	if !Available() {
		t.Skip("Metal device not available")
	}
	defer ResetQ4K()

	const (
		in  = 256
		out = 256
	)
	gateRaw := q4kTestRaw(out, in, 0x1111)
	upRaw := q4kTestRaw(out, in, 0x2222)
	gate := UploadQ4K(gateRaw, out, in)
	up := UploadQ4K(upRaw, out, in)

	x := make([]float32, in)
	inter := make([]float32, out)

	// nil gate
	if FusedDequantGEMVSwiGLU(nil, up, x, inter) {
		t.Fatal("expected false for nil gate")
	}
	// nil up
	if FusedDequantGEMVSwiGLU(gate, nil, x, inter) {
		t.Fatal("expected false for nil up")
	}
	// short x
	if FusedDequantGEMVSwiGLU(gate, up, x[:in-1], inter) {
		t.Fatal("expected false for short x")
	}
	// short inter
	if FusedDequantGEMVSwiGLU(gate, up, x, inter[:out-1]) {
		t.Fatal("expected false for short inter")
	}
}
