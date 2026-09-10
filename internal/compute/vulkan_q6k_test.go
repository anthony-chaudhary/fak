//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/binary"
	"encoding/json"
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

const q6KOptionalBundleChildEnv = "FAK_TEST_VULKAN_Q6K_OPTIONAL_BUNDLE"

func TestVulkanQ6KOptionalShaderBundleCompatibility(t *testing.T) {
	if mode := os.Getenv(q6KOptionalBundleChildEnv); mode != "" {
		q6KOptionalBundleChild(t, mode)
		return
	}

	v := vk(t)
	if !v.SupportsQ6KMatMul() {
		t.Fatal("source Vulkan bundle does not expose q6k_matmul")
	}
	spirvDir := os.Getenv("FAK_VULKAN_SPIRV")
	if spirvDir == "" {
		wd, err := os.Getwd()
		if err != nil {
			t.Fatalf("os.Getwd: %v", err)
		}
		spirvDir = filepath.Join(findRepoRootForTest(t, wd), "internal", "compute", "spirv")
	}
	if _, err := os.Stat(filepath.Join(spirvDir, "q6k_matmul.spv")); err != nil {
		t.Fatalf("source Vulkan bundle lacks q6k_matmul.spv: %v", err)
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}

	for _, mode := range []string{"missing", "invalid"} {
		t.Run(mode, func(t *testing.T) {
			bundleDir := q6KCopyOptionalBundleFixture(t, spirvDir, mode)
			cmd := sysproc.Command(executable, "-test.run=^TestVulkanQ6KOptionalShaderBundleCompatibility$")
			cmd.Env = q6KOptionalBundleChildEnvironment(bundleDir, mode)
			if output, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("optional Q6_K bundle child failed: %v\n%s", err, output)
			}
		})
	}
}

func q6KOptionalBundleChild(t *testing.T, mode string) {
	if mode != "missing" && mode != "invalid" {
		t.Fatalf("unknown optional Q6_K bundle child mode %q", mode)
	}
	backend, ok := Lookup("vulkan")
	if !ok {
		t.Fatal("Vulkan backend was not registered without the optional q6k_matmul pipeline")
	}
	v, ok := backend.(*vulkanBackend)
	if !ok {
		t.Fatalf("registered Vulkan backend has type %T", backend)
	}
	if v.SupportsQ6KMatMul() {
		t.Fatalf("Vulkan reported Q6_K support with %s q6k_matmul.spv", mode)
	}

	window, available, err := BeginBackendExecutionObservation(v)
	if err != nil || !available {
		t.Fatalf("begin refused-upload observation: available=%t err=%v", available, err)
	}
	hostWeight := NewQ6K(Default(), []int{1, q6KBlockValues}, q6KStrixFixture(1, q6KBlockValues))
	q6KExpectPanic(t, "loaded shader bundle lacks q6k_matmul", func() {
		_ = v.Upload(hostWeight, Q6_K)
	})
	observation, err := window.End()
	if err != nil {
		t.Fatalf("end refused-upload observation: %v", err)
	}
	if c := observation.Counters; c.ComputeDispatches != 0 || c.H2DCount != 0 || c.D2HCount != 0 || c.D2DCopies != 0 {
		t.Fatalf("refused Q6_K upload touched the device: %+v", c)
	}
	if !observation.DeviceAllocationObserved || observation.DeviceAllocationPeakBytes != observation.DeviceAllocationLiveBytes {
		t.Fatalf("refused Q6_K upload allocated device memory: observed=%t live=%d peak=%d",
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
		t.Fatalf("ordinary F32 Vulkan matmul with optional Q6_K unavailable = %v, want [17 39]", got)
	}
}

func q6KCopyOptionalBundleFixture(t *testing.T, sourceDir, mode string) string {
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
		if entry.Name() == "q6k_matmul.spv" && mode == "missing" {
			continue
		}
		data, err := os.ReadFile(filepath.Join(sourceDir, entry.Name()))
		if err != nil {
			t.Fatalf("read source shader %s: %v", entry.Name(), err)
		}
		if entry.Name() == "q6k_matmul.spv" && mode == "invalid" {
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

func q6KOptionalBundleChildEnvironment(spirvDir, mode string) []string {
	filtered := make([]string, 0, len(os.Environ())+2)
	for _, entry := range os.Environ() {
		key := entry
		if index := strings.IndexByte(entry, '='); index >= 0 {
			key = entry[:index]
		}
		if strings.EqualFold(key, "FAK_VULKAN_SPIRV") || strings.EqualFold(key, q6KOptionalBundleChildEnv) {
			continue
		}
		filtered = append(filtered, entry)
	}
	return append(filtered, "FAK_VULKAN_SPIRV="+spirvDir, q6KOptionalBundleChildEnv+"="+mode)
}

func TestVulkanQ6KHostAdmission(t *testing.T) {
	const fullHeadOut, fullHeadIn = 248320, 5120
	payload, resident := vulkanQ6KByteSizes(fullHeadOut, fullHeadIn)
	if payload != 1042944000 || resident != payload {
		t.Fatalf("full Qwen head [248320,5120] sizes payload/resident=%d/%d, want 1042944000/1042944000", payload, resident)
	}

	for _, tc := range []struct {
		name, want string
		out, in    int
	}{
		{name: "zero_output", out: 0, in: 256, want: "positive [out,in]"},
		{name: "zero_reduction", out: 1, in: 0, want: "positive [out,in]"},
		{name: "partial_superblock", out: 1, in: 511, want: "divisible by 256"},
		{name: "shader_byte_index_overflow", out: int(vulkanQ6KMaxIndex/210) + 1, in: 256, want: "uint32 shader byte indexing"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			q6KExpectPanic(t, tc.want, func() { _, _ = vulkanQ6KByteSizes(tc.out, tc.in) })
		})
	}
	if strconv.IntSize == 64 {
		q6KExpectPanic(t, "C int ABI", func() { _, _ = vulkanQ6KByteSizes(int(vulkanQ6KMaxCInt)+1, 256) })
	}

	// The host constructor rejects malformed packed bytes before a Vulkan backend
	// or allocation is involved.
	q6KExpectPanic(t, "invalid raw k-quant shape or byte length", func() {
		_ = NewQ6K(Default(), []int{1, 256}, make([]byte, q6KBlockBytes-1))
	})
}

func TestVulkanQ6KFullHeadDescriptorAdmission(t *testing.T) {
	v := vk(t)
	capability, ok := any(v).(interface{ SupportsQ6KMatMul() bool })
	if !ok || !capability.SupportsQ6KMatMul() {
		t.Fatal("registered Vulkan backend does not expose the native q6k_matmul pipeline")
	}
	const fullHeadOut, fullHeadIn = 248320, 5120
	payload, resident := vulkanQ6KByteSizes(fullHeadOut, fullHeadIn)
	dummy := unsafe.Pointer(new(byte))
	weight := makeTensor(v, Q6_K, RowMajor, []int{fullHeadOut, fullHeadIn}, &QuantSpec{Block: 256, Axis: 2, Bits: 6}, &vulkanBuf{ptr: dummy, n: resident, logicalN: payload})
	input := makeTensor(v, F32, RowMajor, []int{fullHeadIn}, nil, &vulkanBuf{ptr: dummy, n: fullHeadIn * F32.Bytes(), logicalN: fullHeadIn * F32.Bytes()})
	if out, in := v.validateQ6KMatMulInputs(weight, input, 1); out != fullHeadOut || in != fullHeadIn {
		t.Fatalf("full Qwen head admission returned [%d,%d], want [%d,%d]", out, in, fullHeadOut, fullHeadIn)
	}

	badPayload := weight
	badPayload.buf = &vulkanBuf{ptr: dummy, n: resident, logicalN: payload - 1}
	q6KExpectPanic(t, "resident payload does not match", func() { v.validateQ6KMatMulInputs(badPayload, input, 1) })

	panel := makeTensor(v, F32, RowMajor, []int{3, fullHeadIn}, nil, &vulkanBuf{ptr: dummy, n: 3 * fullHeadIn * F32.Bytes(), logicalN: 3 * fullHeadIn * F32.Bytes()})
	q6KExpectPanic(t, "input shape does not match P*in", func() { v.validateQ6KMatMulInputs(weight, panel, 4) })

	overflowP := int(vulkanQ6KMaxIndex/uint64(fullHeadOut)) + 1
	overflowInput := makeTensor(v, F32, RowMajor, []int{overflowP, fullHeadIn}, nil, &vulkanBuf{ptr: dummy, n: 1, logicalN: 1})
	q6KExpectPanic(t, "activation or output exceeds uint32 shader indexing", func() {
		v.validateQ6KMatMulInputs(weight, overflowInput, overflowP)
	})
}

func q6KExpectPanic(t *testing.T, want string, fn func()) {
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
	q6KBlockValues = 256
	q6KBlockBytes  = 210
)

// TestVulkanQ6KMatMulStrix verifies that Vulkan consumes GGUF Q6_K bytes in
// place. The one-block case is deliberately 210 bytes (2 mod 4): its trailing
// f16 scale crosses the last uint-addressed word unless the resident allocation
// is padded. The wider cases cover alternating 210-byte block alignment, the
// Qwen hidden/FFN reduction sizes, output workgroup tails, and P=1/2/4 dispatch.
func TestVulkanQ6KMatMulStrix(t *testing.T) {
	v := vk(t)
	capability, ok := any(v).(interface{ SupportsQ6KMatMul() bool })
	if !ok || !capability.SupportsQ6KMatMul() {
		t.Fatal("registered Vulkan backend does not expose the native q6k_matmul pipeline")
	}
	v.VulkanDebugResetDispatchProfile()
	window, available, err := BeginBackendExecutionObservation(v)
	if err != nil || !available {
		t.Fatalf("begin Vulkan Q6_K execution observation: available=%t err=%v", available, err)
	}
	defer func() {
		observation, endErr := window.End()
		if endErr != nil {
			t.Errorf("end Vulkan Q6_K execution observation: %v", endErr)
			return
		}
		if !observation.TransferCountersObserved || observation.Counters.ComputeDispatches == 0 {
			t.Errorf("Q6_K test did not prove observed Vulkan dispatch: %+v", observation)
		}
		profile := v.VulkanDebugDispatchProfileSnapshot()
		if profile.ComputeDispatches != 4 || profile.OtherComputeDispatches != 4 || profile.OtherMatmulDispatches != 4 {
			t.Errorf("Q6_K dispatch classification=%+v, want compute=4 other=4 matmul=4", profile)
		}
		encoded, encodeErr := json.Marshal(observation)
		if encodeErr != nil {
			t.Errorf("encode Vulkan Q6_K execution observation: %v", encodeErr)
			return
		}
		t.Logf("Q6_K Vulkan execution observation: %s", encoded)
	}()

	for _, tc := range []struct {
		name       string
		out, in, p int
	}{
		{name: "single_unaligned_final_word", out: 1, in: 256, p: 1},
		{name: "hidden_decode_tail", out: 67, in: 5120, p: 1},
		{name: "hidden_panel_tail", out: 33, in: 5120, p: 2},
		{name: "ffn_panel_tail", out: 17, in: 17408, p: 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := q6KStrixFixture(tc.out, tc.in)
			if tc.out == 1 && tc.in == q6KBlockValues && len(raw)%4 != 2 {
				t.Fatalf("one-block fixture has %d bytes mod 4, want 2", len(raw)%4)
			}
			if tc.in/q6KBlockValues >= 2 && q6KBlockBytes%4 != 2 {
				t.Fatal("fixture no longer exercises a two-byte-aligned odd block")
			}

			x := q6KStrixInput(tc.p, tc.in)
			want := q6KDequantF32Oracle(tc.out, tc.in, tc.p, raw, x)

			hostWeight := NewQ6K(Default(), []int{tc.out, tc.in}, raw)
			weight := v.Upload(hostWeight, Q6_K)
			if tc.out == 1 && tc.in == q6KBlockValues {
				clone, cloneErr := v.CloneTensor(weight)
				if cloneErr != nil {
					t.Fatalf("clone packed Q6_K weight: %v", cloneErr)
				}
				v.Free(weight)
				weight = clone
			}
			defer v.Free(weight)
			if weight.Dtype != Q6_K {
				t.Fatalf("resident weight dtype=%s, want packed Q6_K", weight.Dtype)
			}
			resident, ok := weight.buf.(*vulkanBuf)
			if !ok || resident.ptr == nil {
				t.Fatal("Q6_K weight did not become a Vulkan resident buffer")
			}
			wantResidentBytes := (len(raw) + 3) &^ 3
			if resident.n != wantResidentBytes {
				t.Fatalf("resident Q6_K bytes=%d, want four-byte padded packed size %d (raw=%d)", resident.n, wantResidentBytes, len(raw))
			}
			if resident.n >= tc.out*tc.in*F32.Bytes() {
				t.Fatalf("resident Q6_K bytes=%d indicate F32 materialization (F32=%d)", resident.n, tc.out*tc.in*F32.Bytes())
			}

			shape := []int{tc.in}
			if tc.p > 1 {
				shape = []int{tc.p, tc.in}
			}
			dx := v.UploadClass(NewF32(Default(), shape, x), F32, MemoryActivation, "Q6_K test input")
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
				t.Fatalf("packed Q6_K cosine=%.9f, want >= %.5f", cosine, minCosine)
			}
			maxDelta, maxReference := q6KMaxAbsDelta(got, want)
			const maxRelativeDelta = 2e-5
			if maxDelta > maxRelativeDelta*math.Max(1, maxReference) {
				t.Fatalf("packed Q6_K max-abs delta=%g exceeds relative gate %g (reference max=%g)", maxDelta, maxRelativeDelta, maxReference)
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

func q6KStrixFixture(out, in int) []byte {
	blocksPerRow := in / q6KBlockValues
	raw := make([]byte, out*blocksPerRow*q6KBlockBytes)
	signedScales := [...]int8{-11, 7, -5, 13, 3, -9, 15, -2, 6, -14, 10, -7, 4, -12, 9, -3}
	for block := 0; block < out*blocksPerRow; block++ {
		base := block * q6KBlockBytes
		for i := 0; i < 128; i++ {
			raw[base+i] = byte((block*29 + i*37 + 11) & 0xff)
		}
		for i := 0; i < 64; i++ {
			raw[base+128+i] = byte((block*53 + i*71 + 0xa5) & 0xff)
		}
		for i, scale := range signedScales {
			if block&1 != 0 {
				scale = -scale
			}
			raw[base+192+i] = byte(scale)
		}
		// Exact powers of two keep the oracle focused on byte layout and the
		// reduction. Alternating values distinguish adjacent 210-byte blocks.
		half := uint16(0x2800) // 2^-5
		if block&1 != 0 {
			half = 0x2c00 // 2^-4
		}
		binary.LittleEndian.PutUint16(raw[base+208:base+210], half)
	}
	return raw
}

func q6KStrixInput(tokens, in int) []float32 {
	x := make([]float32, tokens*in)
	for token := 0; token < tokens; token++ {
		for col := 0; col < in; col++ {
			// Mixed signs, no zeros, and token-dependent phases exercise each
			// panel row without introducing random or near-tie test behavior.
			value := float32(((col*17+token*31)%61)-30) / 61
			if value == 0 {
				value = float32(token+1) / 97
			}
			x[token*in+col] = value
		}
	}
	return x
}

func q6KDequantF32Oracle(out, in, tokens int, raw []byte, x []float32) []float32 {
	blocksPerRow := in / q6KBlockValues
	weights := make([]float32, out*in)
	block := make([]float32, q6KBlockValues)
	for row := 0; row < out; row++ {
		for sb := 0; sb < blocksPerRow; sb++ {
			offset := (row*blocksPerRow + sb) * q6KBlockBytes
			dequantQ6K(block, raw[offset:offset+q6KBlockBytes])
			copy(weights[row*in+sb*q6KBlockValues:], block)
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

func q6KMaxAbsDelta(got, want []float32) (delta, reference float64) {
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
