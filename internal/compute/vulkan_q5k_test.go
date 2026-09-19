//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"unsafe"

	"github.com/anthony-chaudhary/fak/pkg/sysproc"
)

const q5KOptionalBundleChildEnv = "FAK_TEST_VULKAN_Q5K_OPTIONAL_BUNDLE"

func TestVulkanQ5KOptionalShaderBundleCompatibility(t *testing.T) {
	if mode := os.Getenv(q5KOptionalBundleChildEnv); mode != "" {
		q5KOptionalBundleChild(t, mode)
		return
	}

	v := vk(t)
	if !v.SupportsQ5KMatMul() {
		t.Fatal("source Vulkan bundle does not expose q5k_matmul")
	}
	spirvDir := os.Getenv("FAK_VULKAN_SPIRV")
	if spirvDir == "" {
		wd, err := os.Getwd()
		if err != nil {
			t.Fatalf("os.Getwd: %v", err)
		}
		spirvDir = filepath.Join(findRepoRootForTest(t, wd), "internal", "compute", "spirv")
	}
	if _, err := os.Stat(filepath.Join(spirvDir, "q5k_matmul.spv")); err != nil {
		t.Fatalf("source Vulkan bundle lacks q5k_matmul.spv: %v", err)
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}

	for _, mode := range []string{"missing", "invalid"} {
		t.Run(mode, func(t *testing.T) {
			bundleDir := q5KCopyOptionalBundleFixture(t, spirvDir, mode)
			cmd := sysproc.Command(executable, "-test.run=^TestVulkanQ5KOptionalShaderBundleCompatibility$")
			cmd.Env = q5KOptionalBundleChildEnvironment(bundleDir, mode)
			if output, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("optional Q5_K bundle child failed: %v\n%s", err, output)
			}
		})
	}
}

func q5KOptionalBundleChild(t *testing.T, mode string) {
	if mode != "missing" && mode != "invalid" {
		t.Fatalf("unknown optional Q5_K bundle child mode %q", mode)
	}
	backend, ok := Lookup("vulkan")
	if !ok {
		t.Fatal("Vulkan backend was not registered without the optional q5k_matmul pipeline")
	}
	v, ok := backend.(*vulkanBackend)
	if !ok {
		t.Fatalf("registered Vulkan backend has type %T", backend)
	}
	if v.SupportsQ5KMatMul() {
		t.Fatalf("Vulkan reported Q5_K support with %s q5k_matmul.spv", mode)
	}

	window, available, err := BeginBackendExecutionObservation(v)
	if err != nil || !available {
		t.Fatalf("begin refused-upload observation: available=%t err=%v", available, err)
	}
	hostWeight := NewQ5K(Default(), []int{1, q5KBlockValues}, q5KStrixFixture(1, q5KBlockValues))
	q5KExpectPanic(t, "loaded shader bundle lacks q5k_matmul", func() {
		_ = v.Upload(hostWeight, Q5_K)
	})
	observation, err := window.End()
	if err != nil {
		t.Fatalf("end refused-upload observation: %v", err)
	}
	if c := observation.Counters; c.ComputeDispatches != 0 || c.H2DCount != 0 || c.D2HCount != 0 || c.D2DCopies != 0 {
		t.Fatalf("refused Q5_K upload touched the device: %+v", c)
	}
	if !observation.DeviceAllocationObserved || observation.DeviceAllocationPeakBytes != observation.DeviceAllocationLiveBytes {
		t.Fatalf("refused Q5_K upload allocated device memory: observed=%t live=%d peak=%d",
			observation.DeviceAllocationObserved, observation.DeviceAllocationLiveBytes, observation.DeviceAllocationPeakBytes)
	}

	weight := v.Upload(NewF32(Default(), []int{2, 2}, []float32{1, 2, 3, 4}), F32)
	input := v.Upload(NewF32(Default(), []int{2}, []float32{5, 6}), F32)
	output := v.MatMul(weight, input)
	got := v.Read(output)
	v.Free(output)
	v.Free(input)
	v.Free(weight)
	if len(got) != 2 || math.Abs(float64(got[0]-17)) > 1e-5 || math.Abs(float64(got[1]-39)) > 1e-5 {
		t.Fatalf("ordinary F32 Vulkan matmul with optional Q5_K unavailable = %v, want [17 39]", got)
	}
}

func q5KCopyOptionalBundleFixture(t *testing.T, sourceDir, mode string) string {
	t.Helper()
	destinationDir := t.TempDir()
	entries, err := os.ReadDir(sourceDir)
	if err != nil {
		t.Fatalf("read source SPIR-V bundle: %v", err)
	}
	copied := 0
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".spv") {
			continue
		}
		if entry.Name() == "q5k_matmul.spv" && mode == "missing" {
			continue
		}
		data, err := os.ReadFile(filepath.Join(sourceDir, entry.Name()))
		if err != nil {
			t.Fatalf("read source shader %s: %v", entry.Name(), err)
		}
		if entry.Name() == "q5k_matmul.spv" && mode == "invalid" {
			data = []byte{0, 0, 0, 0}
		}
		if err := os.WriteFile(filepath.Join(destinationDir, entry.Name()), data, 0o644); err != nil {
			t.Fatalf("write shader fixture %s: %v", entry.Name(), err)
		}
		copied++
	}
	if copied == 0 {
		t.Fatal("source SPIR-V bundle contained no shaders")
	}
	return destinationDir
}

func q5KOptionalBundleChildEnvironment(spirvDir, mode string) []string {
	filtered := make([]string, 0, len(os.Environ())+2)
	for _, entry := range os.Environ() {
		key := entry
		if index := strings.IndexByte(entry, '='); index >= 0 {
			key = entry[:index]
		}
		if strings.EqualFold(key, "FAK_VULKAN_SPIRV") || strings.EqualFold(key, q5KOptionalBundleChildEnv) {
			continue
		}
		filtered = append(filtered, entry)
	}
	return append(filtered, "FAK_VULKAN_SPIRV="+spirvDir, q5KOptionalBundleChildEnv+"="+mode)
}

func TestVulkanQ5KHostAdmission(t *testing.T) {
	const fullHeadOut, fullHeadIn = 248320, 5120
	payload, resident := vulkanQ5KByteSizes(fullHeadOut, fullHeadIn)
	if payload != fullHeadOut*(fullHeadIn/256)*176 {
		t.Fatalf("full Qwen head [248320,5120] payload=%d, want %d", payload, fullHeadOut*(fullHeadIn/256)*176)
	}
	if resident != ((payload + 3) &^ 3) {
		t.Fatalf("full Qwen head [248320,5120] resident=%d, want padded %d", resident, (payload+3)&^3)
	}

	for _, tc := range []struct {
		name, want string
		out, in    int
	}{
		{name: "zero_output", out: 0, in: 256, want: "positive [out,in]"},
		{name: "zero_reduction", out: 1, in: 0, want: "positive [out,in]"},
		{name: "partial_superblock", out: 1, in: 511, want: "divisible by 256"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			q5KExpectPanic(t, tc.want, func() { _, _ = vulkanQ5KByteSizes(tc.out, tc.in) })
		})
	}
	if strconv.IntSize == 64 {
		q5KExpectPanic(t, "C int ABI", func() { _, _ = vulkanQ5KByteSizes(int(vulkanQ5KMaxCInt)+1, 256) })
	}

	// The host constructor rejects malformed packed bytes before a Vulkan backend
	// or allocation is involved.
	q5KExpectPanic(t, "invalid raw k-quant shape or byte length", func() {
		_ = NewQ5K(Default(), []int{1, 256}, make([]byte, q5KBlockBytes-1))
	})
}

func TestVulkanQ5KFullHeadDescriptorAdmission(t *testing.T) {
	v := vk(t)
	capability, ok := any(v).(interface{ SupportsQ5KMatMul() bool })
	if !ok || !capability.SupportsQ5KMatMul() {
		t.Fatal("registered Vulkan backend does not expose the native q5k_matmul pipeline")
	}
	const fullHeadOut, fullHeadIn = 248320, 5120
	payload, resident := vulkanQ5KByteSizes(fullHeadOut, fullHeadIn)
	dummy := unsafe.Pointer(new(byte))
	weight := makeTensor(v, Q5_K, RowMajor, []int{fullHeadOut, fullHeadIn}, &QuantSpec{Block: 256, Axis: 2, Bits: 5}, &vulkanBuf{ptr: dummy, n: resident, logicalN: payload})
	input := makeTensor(v, F32, RowMajor, []int{fullHeadIn}, nil, &vulkanBuf{ptr: dummy, n: fullHeadIn * F32.Bytes(), logicalN: fullHeadIn * F32.Bytes()})
	if out, in := v.validateQ5KMatMulInputs(weight, input, 1); out != fullHeadOut || in != fullHeadIn {
		t.Fatalf("full Qwen head admission returned [%d,%d], want [%d,%d]", out, in, fullHeadOut, fullHeadIn)
	}

	badPayload := weight
	badPayload.buf = &vulkanBuf{ptr: dummy, n: resident, logicalN: payload - 1}
	q5KExpectPanic(t, "resident payload does not match", func() { v.validateQ5KMatMulInputs(badPayload, input, 1) })

	panel := makeTensor(v, F32, RowMajor, []int{3, fullHeadIn}, nil, &vulkanBuf{ptr: dummy, n: 3 * fullHeadIn * F32.Bytes(), logicalN: 3 * fullHeadIn * F32.Bytes()})
	q5KExpectPanic(t, "input shape does not match P*in", func() { v.validateQ5KMatMulInputs(weight, panel, 4) })
}

func q5KExpectPanic(t *testing.T, want string, fn func()) {
	t.Helper()
	defer func() {
		recovered := recover()
		if recovered == nil || !strings.Contains(fmt.Sprint(recovered), want) {
			t.Fatalf("panic=%v, want text containing %q", recovered, want)
		}
	}()
	fn()
}

const (
	q5KBlockValues = 256
	q5KBlockBytes  = 176
)

// TestVulkanQ5KMatMulStrix verifies that Vulkan consumes GGUF Q5_K bytes in
// place. The one-block case is deliberately 176 bytes (0 mod 4): word-aligned,
// but the host still pads the allocation to four bytes. The wider cases cover
// alternating 176-byte block alignment, the Qwen hidden/FFN reduction sizes,
// output workgroup tails, and P=1/2/4 dispatch.
func TestVulkanQ5KMatMulStrix(t *testing.T) {
	v := vk(t)
	capability, ok := any(v).(interface{ SupportsQ5KMatMul() bool })
	if !ok || !capability.SupportsQ5KMatMul() {
		t.Fatal("registered Vulkan backend does not expose the native q5k_matmul pipeline")
	}
	v.VulkanDebugResetDispatchProfile()
	window, available, err := BeginBackendExecutionObservation(v)
	if err != nil || !available {
		t.Fatalf("begin Vulkan Q5_K execution observation: available=%t err=%v", available, err)
	}
	defer func() {
		observation, endErr := window.End()
		if endErr != nil {
			t.Errorf("end Vulkan Q5_K execution observation: %v", endErr)
			return
		}
		if !observation.TransferCountersObserved || observation.Counters.ComputeDispatches == 0 {
			t.Errorf("Q5_K test did not prove observed Vulkan dispatch: %+v", observation)
		}
	}()

	for _, tc := range []struct {
		name       string
		out, in, p int
	}{
		{name: "single_block", out: 1, in: 256, p: 1},
		{name: "hidden_decode_tail", out: 67, in: 5120, p: 1},
		{name: "hidden_panel_tail", out: 33, in: 5120, p: 2},
		{name: "ffn_panel_tail", out: 17, in: 17408, p: 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := q5KStrixFixture(tc.out, tc.in)

			x := q5KStrixInput(tc.p, tc.in)
			want := q5KDequantF32Oracle(tc.out, tc.in, tc.p, raw, x)

			hostWeight := NewQ5K(Default(), []int{tc.out, tc.in}, raw)
			weight := v.Upload(hostWeight, Q5_K)
			defer v.Free(weight)
			if weight.Dtype != Q5_K {
				t.Fatalf("resident weight dtype=%s, want packed Q5_K", weight.Dtype)
			}
			resident, ok := weight.buf.(*vulkanBuf)
			if !ok || resident.ptr == nil {
				t.Fatal("Q5_K weight did not become a Vulkan resident buffer")
			}
			wantResidentBytes := (len(raw) + 3) &^ 3
			if resident.n != wantResidentBytes {
				t.Fatalf("resident Q5_K bytes=%d, want four-byte padded packed size %d (raw=%d)", resident.n, wantResidentBytes, len(raw))
			}
			if resident.n >= tc.out*tc.in*F32.Bytes() {
				t.Fatalf("resident Q5_K bytes=%d indicate F32 materialization (F32=%d)", resident.n, tc.out*tc.in*F32.Bytes())
			}

			shape := []int{tc.in}
			if tc.p > 1 {
				shape = []int{tc.p, tc.in}
			}
			dx := v.UploadClass(NewF32(Default(), shape, x), F32, MemoryActivation, "Q5_K test input")
			defer v.Free(dx)
			var output Tensor
			if tc.p == 1 {
				output = v.MatMul(weight, dx)
			} else {
				output = v.BatchedMatMul(weight, dx, tc.p)
			}
			defer v.Free(output)
			got := v.Read(output)
			if len(got) != tc.p*tc.out {
				t.Fatalf("output length=%d, want %d", len(got), tc.p*tc.out)
			}

			const minCosine = 0.99999
			if cosine := cosineC(got, want); math.IsNaN(cosine) || cosine < minCosine {
				t.Fatalf("packed Q5_K cosine=%.9f, want >= %.5f", cosine, minCosine)
			}
			maxDelta, maxReference := q5KMaxAbsDelta(got, want)
			const maxRelativeDelta = 2e-5
			if maxDelta > maxRelativeDelta*math.Max(1, maxReference) {
				t.Fatalf("packed Q5_K max-abs delta=%g exceeds relative gate %g (reference max=%g)", maxDelta, maxRelativeDelta, maxReference)
			}
			for token := 0; token < tc.p; token++ {
				start := token * tc.out
				if gotArgmax, wantArgmax := argmaxF32(got[start:start+tc.out]), argmaxF32(want[start:start+tc.out]); gotArgmax != wantArgmax {
					t.Fatalf("token %d argmax=%d, CPU oracle=%d", token, gotArgmax, wantArgmax)
				}
			}
		})
	}
}

func q5KStrixFixture(out, in int) []byte {
	blocksPerRow := in / q5KBlockValues
	raw := make([]byte, out*blocksPerRow*q5KBlockBytes)
	for block := 0; block < out*blocksPerRow; block++ {
		base := block * q5KBlockBytes
		// d and dmin: small exact powers of two; alternate per block so adjacent
		// 176-byte blocks are distinguishable.
		d := uint16(0x2800)    // 2^-5
		dmin := uint16(0x2400) // 2^-6
		if block&1 != 0 {
			d = 0x2c00    // 2^-4
			dmin = 0x2800 // 2^-5
		}
		binary.LittleEndian.PutUint16(raw[base+0:base+2], d)
		binary.LittleEndian.PutUint16(raw[base+2:base+4], dmin)
		// 12 scale/min bytes exercising both ScaleMinK4 branches (j<4 and j>=4).
		for i := 0; i < 12; i++ {
			raw[base+4+i] = byte((block*17 + i*23 + 5) & 0xff)
		}
		// qh: 32 high-bit bytes.
		for i := 0; i < 32; i++ {
			raw[base+16+i] = byte((block*53 + i*71 + 0xa5) & 0xff)
		}
		// ql: 128 low-nibble bytes.
		for i := 0; i < 128; i++ {
			raw[base+48+i] = byte((block*29 + i*37 + 11) & 0xff)
		}
	}
	return raw
}

func q5KStrixInput(tokens, in int) []float32 {
	x := make([]float32, tokens*in)
	for token := 0; token < tokens; token++ {
		for col := 0; col < in; col++ {
			value := float32(((col*17+token*31)%61)-30) / 61
			if value == 0 {
				value = float32(token+1) / 97
			}
			x[token*in+col] = value
		}
	}
	return x
}

func q5KDequantF32Oracle(out, in, tokens int, raw []byte, x []float32) []float32 {
	blocksPerRow := in / q5KBlockValues
	weights := make([]float32, out*in)
	block := make([]float32, q5KBlockValues)
	for row := 0; row < out; row++ {
		for sb := 0; sb < blocksPerRow; sb++ {
			offset := (row*blocksPerRow + sb) * q5KBlockBytes
			dequantQ5K(block, raw[offset:offset+q5KBlockBytes])
			copy(weights[row*in+sb*q5KBlockValues:], block)
		}
	}
	ref := Default()
	weight := NewF32(ref, []int{out, in}, weights)
	inputShape := []int{in}
	if tokens > 1 {
		inputShape = []int{tokens, in}
	}
	input := ref.Upload(NewF32(ref, inputShape, x), F32)
	if tokens == 1 {
		return ref.Read(ref.MatMul(weight, input))
	}
	return ref.Read(ref.BatchedMatMul(weight, input, tokens))
}

func q5KMaxAbsDelta(got, want []float32) (delta, reference float64) {
	for i := range want {
		d := math.Abs(float64(got[i] - want[i]))
		if d > delta {
			delta = d
		}
		w := math.Abs(float64(want[i]))
		if w > reference {
			reference = w
		}
	}
	return delta, reference
}
