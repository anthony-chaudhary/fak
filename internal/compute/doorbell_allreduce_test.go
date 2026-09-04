package compute

import (
	"math"
	"strings"
	"testing"
	"time"
)

// TestDoorbellSizeThresholdRouting verifies that collective dispatch correctly routes
// sub-1MB decode vectors to direct stream-async RDMA with spin-doorbell signaling,
// and bulk prefill chunks (>= 1MB) to pipelined ring collectives.
func TestDoorbellSizeThresholdRouting(t *testing.T) {
	engine := NewDoorbellAllReduceEngine(2, TransportThunderbolt4, StorageModeSharedMetal)

	// RouteForSize checks across sizes
	testCases := []struct {
		name      string
		numFloats int
		wantRoute CollectiveRoute
	}{
		{"scalar_4B", 1, RouteStreamAsyncDoorbell},
		{"decode_16KB", 4096, RouteStreamAsyncDoorbell},
		{"decode_64KB", 16384, RouteStreamAsyncDoorbell},
		{"sub1MB_1000KB", 250000, RouteStreamAsyncDoorbell},
		{"exact1MB", 262144, RoutePipelinedRing},
		{"prefill_2MB", 524288, RoutePipelinedRing},
		{"prefill_8MB", 2097152, RoutePipelinedRing},
	}

	for _, tc := range testCases {
		byteSize := tc.numFloats * 4
		gotRoute := engine.RouteForSize(byteSize)
		if gotRoute != tc.wantRoute {
			t.Errorf("%s (size %d bytes): got route %v, want %v", tc.name, byteSize, gotRoute, tc.wantRoute)
		}
	}

	// Verify dispatch via AllReduceSum on sub-1MB decode vector (4096 floats = 16KB)
	decodeFloats := 4096
	d0 := make([]float32, decodeFloats)
	d1 := make([]float32, decodeFloats)
	for i := 0; i < decodeFloats; i++ {
		d0[i] = float32(i)
		d1[i] = float32(i * 2)
	}
	part0 := NewF32(engine, []int{decodeFloats}, d0)
	part1 := NewF32(engine, []int{decodeFloats}, d1)

	_, err := engine.AllReduceSum([]Tensor{part0, part1})
	if err != nil {
		t.Fatalf("AllReduceSum (decode) failed: %v", err)
	}
	if engine.LastRoute() != RouteStreamAsyncDoorbell {
		t.Errorf("LastRoute for decode vector = %v, want %v", engine.LastRoute(), RouteStreamAsyncDoorbell)
	}
	if engine.DoorbellOps() != 1 {
		t.Errorf("DoorbellOps = %d, want 1", engine.DoorbellOps())
	}
	if engine.RingOps() != 0 {
		t.Errorf("RingOps = %d, want 0", engine.RingOps())
	}

	// Verify dispatch via AllReduceSum on bulk prefill chunk (524,288 floats = 2MB)
	prefillFloats := 524288
	bPart0 := NewF32(engine, []int{prefillFloats}, make([]float32, prefillFloats))
	bPart1 := NewF32(engine, []int{prefillFloats}, make([]float32, prefillFloats))

	_, err = engine.AllReduceSum([]Tensor{bPart0, bPart1})
	if err != nil {
		t.Fatalf("AllReduceSum (prefill) failed: %v", err)
	}
	if engine.LastRoute() != RoutePipelinedRing {
		t.Errorf("LastRoute for prefill chunk = %v, want %v", engine.LastRoute(), RoutePipelinedRing)
	}
	if engine.RingOps() != 1 {
		t.Errorf("RingOps = %d, want 1", engine.RingOps())
	}

	// Verify dynamic threshold modification
	engine.SetThreshold(32 * 1024) // 32KB
	if engine.Threshold() != 32*1024 {
		t.Errorf("Threshold = %d, want %d", engine.Threshold(), 32*1024)
	}
	if engine.RouteForSize(16*1024) != RouteStreamAsyncDoorbell {
		t.Errorf("16KB with 32KB threshold: want doorbell")
	}
	if engine.RouteForSize(64*1024) != RoutePipelinedRing {
		t.Errorf("64KB with 32KB threshold: want ring")
	}
}

// TestDoorbellSharedMemoryExchangeParity verifies shared doorbell exchange and reduction
// between 2 simulated nodes with byte-exact mathematical parity vs the CPU reference.
func TestDoorbellSharedMemoryExchangeParity(t *testing.T) {
	configs := []struct {
		name      string
		transport TransportKind
		mode      StorageMode
	}{
		{"AppleSilicon_Thunderbolt4_MetalShared", TransportThunderbolt4, StorageModeSharedMetal},
		{"CUDA_PCIeP2P_HostAlloc", TransportPCIeP2P, StorageModeCUDAP2P},
		{"Generic_SharedMem_UMA", TransportSharedMem, StorageModeHostAlloc},
	}

	sizes := []int{1, 16, 256, 1024, 4096, 16384}

	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			engine := NewDoorbellAllReduceEngine(2, cfg.transport, cfg.mode)
			cpuRef := Pick("cpu-ref").(CollectiveBackend)

			for _, size := range sizes {
				d0 := make([]float32, size)
				d1 := make([]float32, size)
				for i := 0; i < size; i++ {
					// Varied inputs: negative, fractions, zeroes, integers
					d0[i] = float32(i)*1.25 - 50.0
					d1[i] = float32(i*i)*0.005 + 12.375
				}

				p0 := NewF32(engine, []int{size}, d0)
				p1 := NewF32(engine, []int{size}, d1)

				// Ground truth reference
				refParts := []Tensor{
					NewF32(cpuRef, []int{size}, d0),
					NewF32(cpuRef, []int{size}, d1),
				}
				wantTensor, err := cpuRef.AllReduceSum(refParts)
				if err != nil {
					t.Fatalf("cpuRef.AllReduceSum size=%d: %v", size, err)
				}
				want, _ := cpuRef.Host(wantTensor)

				// Doorbell all-reduce
				gotTensor, err := engine.DoorbellAllReduce([]Tensor{p0, p1})
				if err != nil {
					t.Fatalf("DoorbellAllReduce size=%d: %v", size, err)
				}
				got, _ := engine.Host(gotTensor)

				if len(got) != len(want) {
					t.Fatalf("size=%d: got len %d, want len %d", size, len(got), len(want))
				}

				// Byte-exact bit-for-bit check
				for i := 0; i < size; i++ {
					gBits := math.Float32bits(got[i])
					wBits := math.Float32bits(want[i])
					if gBits != wBits {
						t.Fatalf("size=%d index=%d: got 0x%08x (%f), want 0x%08x (%f) - parity mismatch",
							size, i, gBits, got[i], wBits, want[i])
					}
				}

				// Verify doorbell buffers recorded the transfer
				buf01 := engine.Buffer(0, 1)
				if buf01 == nil {
					t.Fatalf("Buffer 0->1 is nil")
				}
				if buf01.Control.Count != uint32(size) {
					t.Errorf("Buffer 0->1 count = %d, want %d", buf01.Control.Count, size)
				}
				if buf01.Control.ArrivalFlag == 0 {
					t.Errorf("Buffer 0->1 arrival flag not signaled")
				}
			}
		})
	}
}

// TestDoorbellSub100usExchange demonstrates sub-100µs simulated collective exchange
// for typical token decode vectors (4096 float32 elements = 16KB).
func TestDoorbellSub100usExchange(t *testing.T) {
	engine := NewDoorbellAllReduceEngine(2, TransportThunderbolt4, StorageModeSharedMetal)

	decodeDim := 4096 // 16KB activation vector (e.g. Qwen/LLaMA hidden dimension)
	d0 := make([]float32, decodeDim)
	d1 := make([]float32, decodeDim)
	for i := 0; i < decodeDim; i++ {
		d0[i] = float32(i) * 0.1
		d1[i] = float32(i) * 0.2
	}

	p0 := NewF32(engine, []int{decodeDim}, d0)
	p1 := NewF32(engine, []int{decodeDim}, d1)

	// Warmup
	_, err := engine.DoorbellAllReduce([]Tensor{p0, p1})
	if err != nil {
		t.Fatalf("Warmup failed: %v", err)
	}

	// Measure 100 consecutive exchanges
	iterations := 100
	start := time.Now()
	for i := 0; i < iterations; i++ {
		_, err := engine.DoorbellAllReduce([]Tensor{p0, p1})
		if err != nil {
			t.Fatalf("Iteration %d failed: %v", i, err)
		}
	}
	totalDuration := time.Since(start)
	avgDuration := totalDuration / time.Duration(iterations)

	t.Logf("Simulated collective exchange: %v per 16KB decode vector (target: < 100µs)", avgDuration)

	if avgDuration >= 100*time.Microsecond {
		t.Fatalf("Simulated collective exchange average latency %v exceeds sub-100µs threshold", avgDuration)
	}
}

// TestDoorbellShaderEmbedding verifies that the Metal compute shader is embedded and valid.
func TestDoorbellShaderEmbedding(t *testing.T) {
	if len(MetalDoorbellShaderSource) == 0 {
		t.Fatal("MetalDoorbellShaderSource is empty")
	}

	requiredTokens := []string{
		"kernel void tb_doorbell_wait_add",
		"DoorbellControl",
		"atomic_load_explicit",
		"MTLResourceStorageModeShared",
	}

	for _, token := range requiredTokens {
		if !strings.Contains(MetalDoorbellShaderSource, token) {
			t.Errorf("MetalDoorbellShaderSource missing required token %q", token)
		}
	}
}

// TestDoorbellEdgeCases tests single rank, ragged inputs, and empty parts.
func TestDoorbellEdgeCases(t *testing.T) {
	engine := NewDoorbellAllReduceEngine(2, TransportThunderbolt4, StorageModeSharedMetal)

	// Single rank identity
	d := make([]float32, 128)
	d[0] = 42.0
	p0 := NewF32(engine, []int{128}, d)

	res, err := engine.AllReduceSum([]Tensor{p0})
	if err != nil {
		t.Fatalf("Single rank AllReduceSum failed: %v", err)
	}
	hRes, ok := engine.Host(res)
	if !ok || hRes[0] != 42.0 {
		t.Errorf("Single rank identity failed: got %v, want 42.0", hRes[0])
	}

	// Empty parts should fail closed
	_, err = engine.AllReduceSum(nil)
	if err == nil {
		t.Errorf("Empty parts should fail closed, got nil error")
	}

	// Ragged partials should fail closed
	pRagged := NewF32(engine, []int{64}, make([]float32, 64))
	_, err = engine.AllReduceSum([]Tensor{p0, pRagged})
	if err == nil {
		t.Errorf("Ragged partials should fail closed, got nil error")
	}
}

// BenchmarkDoorbellCollectiveExchange benchmarks the simulated collective exchange on decode-sized vectors.
func BenchmarkDoorbellCollectiveExchange(b *testing.B) {
	engine := NewDoorbellAllReduceEngine(2, TransportThunderbolt4, StorageModeSharedMetal)

	decodeDim := 4096 // 16KB
	d0 := make([]float32, decodeDim)
	d1 := make([]float32, decodeDim)
	for i := 0; i < decodeDim; i++ {
		d0[i] = float32(i) * 0.1
		d1[i] = float32(i) * 0.2
	}
	p0 := NewF32(engine, []int{decodeDim}, d0)
	p1 := NewF32(engine, []int{decodeDim}, d1)

	b.ReportAllocs()
	b.SetBytes(int64(decodeDim * 4))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := engine.DoorbellAllReduce([]Tensor{p0, p1})
		if err != nil {
			b.Fatalf("DoorbellAllReduce failed: %v", err)
		}
	}
}
