package compute

import (
	"errors"
	"fmt"
	"go/parser"
	"go/token"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// This source/protocol witness does not compile or execute CUDA. Its intentionally
// narrow structural assertions complement, rather than replace, native witnesses.
// fak-test:runtime fast est=20ms lane=default
func TestCUDAV41TailRoPESourceContract(t *testing.T) {
	read := func(path string) string {
		t.Helper()
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	adapter, native := read("cuda_v41_rope.go"), cudaSourceClosure(t, "cuda_kernels.cu")
	if !strings.HasPrefix(adapter, "//go:build cuda && cgo && fak_cuda_v41_tail_rope\n") {
		t.Fatal("candidate adapter must be absent from ordinary CUDA builds")
	}
	if _, err := parser.ParseFile(token.NewFileSet(), "cuda_v41_rope.go", adapter, parser.AllErrors); err != nil {
		t.Fatal(err)
	}
	function := func(t *testing.T, source, marker string) string {
		t.Helper()
		start := strings.Index(source, marker)
		if start < 0 {
			t.Fatalf("missing function %q", marker)
		}
		open := strings.Index(source[start:], "{") + start
		depth := 0
		for i := open; i < len(source); i++ {
			switch source[i] {
			case '{':
				depth++
			case '}':
				depth--
				if depth == 0 {
					return source[start : i+1]
				}
			}
		}
		t.Fatalf("unterminated function %q", marker)
		return ""
	}
	require := func(t *testing.T, source string, fragments ...string) {
		t.Helper()
		for _, fragment := range fragments {
			if !strings.Contains(source, fragment) {
				t.Errorf("missing contract %q", fragment)
			}
		}
	}
	ordered := func(t *testing.T, source string, fragments ...string) {
		t.Helper()
		previous := -1
		for _, fragment := range fragments {
			position := strings.Index(source, fragment)
			if position <= previous {
				t.Fatalf("missing or reordered contract %q", fragment)
			}
			previous = position
		}
	}
	t.Run("private-ABI-and-gates", func(t *testing.T) {
		canonical := func(s string) string { return strings.Join(strings.Fields(s), " ") }
		for _, name := range []string{"state", "input", "qk", "rows", "readback", "counter", "test_fault", "test_free", "bf16_state"} {
			pattern := regexp.MustCompile(`(?:int|uint64_t)\s+fak_v41_tail_rope_` + name + `\s*\([^)]*\)`)
			decl, def := pattern.FindAllString(adapter, -1), pattern.FindAllString(native, -1)
			if len(decl) != 1 || len(def) != 1 || canonical(decl[0]) != canonical(def[0]) {
				t.Fatalf("private ABI mismatch %s: %q / %q", name, decl, def)
			}
			marker := `extern "C" ` + def[0] + " {"
			index := strings.Index(native, marker)
			if index < 0 {
				t.Fatalf("missing C linkage for %s", name)
			}
			var stack []string
			for _, line := range strings.Split(native[:index], "\n") {
				line = strings.TrimSpace(line)
				if strings.HasPrefix(line, "#if") {
					stack = append(stack, line)
				} else if line == "#endif" {
					if len(stack) == 0 {
						t.Fatal("unbalanced native conditional")
					}
					stack = stack[:len(stack)-1]
				}
			}
			gated := false
			for _, condition := range stack {
				gated = gated || condition == "#ifdef FAK_CUDA_V41_TAIL_ROPE"
			}
			if !gated {
				t.Fatalf("%s escaped candidate gate", name)
			}
		}
		require(t, function(t, adapter, "func (c *cudaBackend) V41TailRoPEExactAvailable("), "return false")
		for _, marker := range []string{"fak_v41_tail_rope_test_fault(", "fak_v41_tail_rope_test_free("} {
			require(t, function(t, native, `extern "C" int `+marker), "#ifdef FAK_CUDA_V41_TAIL_ROPE_TEST", "return -8;")
		}
	})
	t.Run("supplied-adjacent-separate-products", func(t *testing.T) {
		kernel := function(t, native, "__global__ void k_v41_tail_rope(")
		require(t, kernel, "tail = headDim - rotaryDim", "uint64_t i", "i += (uint64_t)blockDim.x * gridDim.x",
			"output[local] = input[local]", "(lane - tail) % 2u == 0u", "input[local + 1u]",
			"table[j]", "table[j + 1u]", "__fmul_rn(a, c)", "__fmul_rn(b, s)",
			"__fmul_rn(b, c)", "__fmul_rn(a, s)", "__fsub_rn(ac, bs)", "__fadd_rn(bc, as)")
		for _, forbidden := range []string{"__fmaf", "fmaf(", "sin(", "cos(", "pow(", "__half", "bfloat", "rotaryDim / 2", "input[local] ="} {
			if strings.Contains(kernel, forbidden) {
				t.Errorf("kernel contains forbidden alternate contract %q", forbidden)
			}
		}
	})
	t.Run("state-ownership-and-publication", func(t *testing.T) {
		state := function(t, native, `extern "C" int fak_v41_tail_rope_state(`)
		require(t, state, "g_capture_open", "v41_candidates_stream_uncertain()", "device != 0", "cudaStreamIsCapturing")
		ordered(t, state, "cudaPeekAtLastError", "cudaGetDevice", "cudaStreamIsCapturing", "cudaStreamQuery")
		require(t, state, "if (error == cudaErrorNotReady) return -3;")
		if strings.Contains(state, "cudaGetLastError") {
			t.Fatal("preflight consumes unknown prior CUDA error")
		}
		require(t, function(t, native, "static int v41_tail_rope_owned("), "g_live.find", "found->second <", "cudaPointerGetAttributes",
			"attributes.type == cudaMemoryTypeDevice", "attributes.device == 0")
		dispatch := function(t, native, "static int v41_tail_rope_dispatch(")
		ordered(t, dispatch, "v41_tail_rope_geometry", "fak_v41_tail_rope_state", "dQOut == dKVOut", "v41_tail_rope_owned",
			"k_v41_tail_rope<<<", "cudaGetLastError", "cudaStreamSynchronize", "++g_v41_rope_operations")
		require(t, dispatch, "dQOut == dQ", "dQOut == dTable", "dQOut == dKV", "dKVOut == dQ", "dKVOut == dKV", "dKVOut == dTable")
		operation := function(t, adapter, "func (c *cudaBackend) cudaV41TailRoPE(")
		ordered(t, operation, "validateV41TailRoPE(", "c.cudaV41RoPEBuffer", "c.cudaV41RoPEAdmit", "C.fak_v41_tail_rope_input",
			"qo, err = c.dallocDeviceOnlyClass", "ko, err = c.dallocDeviceOnlyClass", "C.fak_v41_tail_rope_qk",
			"atomic.AddUint64(&c.fenceGen, 1)", "qOut = makeTensor")
		require(t, operation, "validateV41TailRoPERows(", "MemoryActivation", "qOut, kvOut = Tensor{}, Tensor{}", "C.fcuda_free")
		require(t, function(t, adapter, "func (c *cudaBackend) cudaV41RoPEBuffer("), "t.Backend() != c", "b.device != 0", "b.managed",
			"b.scales != nil", "b.scalesN != 0", "b.be != nil && b.be != c", "b.invalidStateError")
		for _, forbidden := range []string{"c.Upload(", "c.Read(", "Default()", "devTr(", "c.transient =", "dallocManaged", "PartialRoPE", "math.Sin", "math.Cos"} {
			if strings.Contains(adapter, forbidden) {
				t.Errorf("adapter contains fallback, wrong algorithm or ownership route %q", forbidden)
			}
		}
		readback := function(t, native, `extern "C" int fak_v41_tail_rope_readback(`)
		ordered(t, readback, "fak_v41_tail_rope_input", "cudaStreamSynchronize", "cudaMemcpy(hOutput", "g_host_bytes +=")
		if strings.Contains(readback, "cudaMemcpyAsync") || strings.Contains(dispatch, "cudaMemcpy") {
			t.Fatal("operation staged host data or readback retained asynchronous host memory")
		}
	})
	t.Run("cross-gate-quarantine", func(t *testing.T) {
		require(t, native, "g_v41_bf16_quarantine[3]", "g_v41_rope_quarantine[5]")
		shared := function(t, native, "static bool v41_candidates_stream_uncertain(")
		require(t, shared, "#ifdef FAK_CUDA_V41_BF16_PUBLISH", "g_v41_bf16_uncertain", "#ifdef FAK_CUDA_V41_TAIL_ROPE", "g_v41_rope_uncertain")
		require(t, function(t, native, `extern "C" int fak_v41_bf16_publish_state(`), "v41_candidates_stream_uncertain()")
		for _, marker := range []string{`extern "C" void fcuda_free(`, `extern "C" void fcuda_free_on(`} {
			free := function(t, native, marker)
			ordered(t, free, "g_v41_bf16_quarantine", "g_v41_rope_quarantine", "q4_coeff_cache_evict")
			ordered(t, free, "g_v41_rope_quarantine", "++g_v41_rope_test_free_attempts", "cudaFree")
			if strings.Index(free, "g_v41_rope_quarantine") > strings.Index(free, "cudaFree") {
				t.Fatal("quarantine checked after direct/unknown free")
			}
		}
		probe := function(t, native, `extern "C" int fak_v41_tail_rope_test_free(`)
		ordered(t, probe, "const uint64_t attempts = g_v41_rope_test_free_attempts", "fcuda_free_on(0, base)", "g_v41_rope_test_free_attempts != attempts")
		drain := function(t, native, "static int v41_tail_rope_drain(")
		ordered(t, drain, "cudaStreamSynchronize", "g_v41_rope_quarantine[0] = q", "g_v41_rope_quarantine[4] = kvOut", "g_v41_rope_uncertain = true")
		for _, forbidden := range []string{"g_v41_rope_quarantine[5] =", "g_v41_bf16_quarantine[3] = input", "g_v41_rope_uncertain = false;"} {
			if strings.Contains(drain, forbidden) {
				t.Errorf("quarantine overwrite, overflow or reset %q", forbidden)
			}
		}
	})
}

// Geometry is the actual common implementation, not a copied CUDA validator.
// fak-test:runtime fast est=1ms lane=default
func TestCUDAV41TailRoPEGeometryContract(t *testing.T) {
	metadata := func(shape ...int) Tensor { return Tensor{Dtype: F32, Layout: RowMajor, Shape: shape} }
	q, kv, table := metadata(2, 7), metadata(7), metadata(2, 2)
	if qb, kb, tb, err := validateV41TailRoPE(q, kv, table, 2, 7, 4); err != nil || qb != 56 || kb != 28 || tb != 16 {
		t.Fatalf("ordinary geometry failed: %d %d %d %v", qb, kb, tb, err)
	}
	for _, dims := range [][3]int{{0, 7, 4}, {2, 0, 4}, {2, 7, 0}, {2, 7, 3}, {2, 7, 8}, {-1, 7, 4}, {65535, 65536, 2}} {
		if _, _, _, err := validateV41TailRoPE(metadata(dims[0], dims[1]), metadata(dims[1]), metadata(dims[2]/2, 2), dims[0], dims[1], dims[2]); err == nil {
			t.Fatalf("invalid/overflow dimensions accepted: %v", dims)
		}
	}
	if strconv.IntSize == 64 {
		tooLarge := uint64(1) << 31
		if _, _, _, err := validateV41TailRoPE(metadata(int(tooLarge), 2), metadata(2), metadata(1, 2), int(tooLarge), 2, 2); err == nil {
			t.Fatal("dimension narrowing accepted")
		}
	}
	for _, bad := range []Tensor{
		{Dtype: F16, Layout: RowMajor, Shape: []int{2, 7}},
		{Dtype: F32, Layout: ColMajor, Shape: []int{2, 7}},
		{Dtype: F32, Layout: RowMajor, Shape: []int{2, 7}, Quant: &QuantSpec{}},
		metadata(14), metadata(2, 6),
	} {
		if _, _, _, err := validateV41TailRoPE(bad, kv, table, 2, 7, 4); err == nil {
			t.Fatal("malformed Q metadata accepted")
		}
		if _, _, err := validateV41TailRoPERows(bad, table, 2, 7, 4); err == nil {
			t.Fatal("malformed Rows metadata accepted")
		}
	}
	if _, _, _, err := validateV41TailRoPE(q, kv, metadata(4), 2, 7, 4); err == nil {
		t.Fatal("flattened sin/cos table accepted")
	}
}

// Ordinary candidate vectors retained from the specification packet.
// The oracle is the existing v41_rope_test.go helper, not a new production path.
// fak-test:runtime fast est=1ms lane=default
func TestCUDAV41TailRoPEContractVectors(t *testing.T) {
	t.Parallel()
	negzero := math.Float32frombits(0x80000000)
	payload := math.Float32frombits(0x7fc01234)
	// An odd-length untouched prefix, two distinct supplied table rows, and
	// adjacent trailing pairs distinguish this contract from Qwen rotate-half.
	input := []float32{negzero, payload, 9, 2, 3, 5, 7}
	table := []float32{1, 0, 0, 1}
	want := []float32{negzero, payload, 9, -3, 2, 5, 7}
	got := v41TailRoPEReference(input, table, 7, 4)
	cudaV41RoPEBitsEqual(t, "adjacent supplied-table control", got, want)

	// Copied from TestV41TailRoPEContract: contraction must change this result.
	a, c := math.Float32frombits(0x3f800001), math.Float32frombits(0x3f7ffffe)
	separate := v41TailRoPEReference([]float32{a, 1}, []float32{1, c}, 2, 2)
	fused := float32(float64(a)*float64(c) - 1)
	if math.Float32bits(separate[0]) != 0 || separate[0] == fused {
		t.Fatal("separate-product cancellation control lost its discriminator")
	}

	// The strict operand rows are copied from TestVulkanV41TailRoPEExactRows.
	// Their oracle outputs are not evidence that CUDA qualifies as exact.
	for _, angle := range []float64{0, 2, 0.37} {
		x := cudaV41RoPEExactOperandRows()
		table := []float32{float32(math.Sin(angle)), float32(math.Cos(angle))}
		out := v41TailRoPEReference(x, table, 4, 2)
		for row := 0; row < len(x)/4; row++ {
			cudaV41RoPEBitsEqual(t, "strict-vector unchanged prefix", out[row*4:row*4+2], x[row*4:row*4+2])
		}
	}
}

func cudaV41RoPEExactOperandRows() []float32 {
	operands := [][2]uint32{
		{0x80000000, 0}, {0, 0x80000000}, {0x80000000, 0x80000000},
		{0x00000001, 0}, {0x80000001, 0}, {0x00010000, 0}, {0x80010000, 0},
		{0x00400000, 0x80400000}, {0x007fffff, 0x00000003},
		{0x3f808000, 0}, {0xbf818000, 0}, {0x41820000, 0xc0ee0000},
		{0x00800000, 0x80800000}, {0x7f7fffff, 0x00000001},
	}
	x := make([]float32, 0, len(operands)*4)
	for i, op := range operands {
		x = append(x, math.Float32frombits(0x80000000), float32(i)+0.5,
			math.Float32frombits(op[0]), math.Float32frombits(op[1]))
	}
	return x
}

func cudaV41RoPEBitsEqual(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d want %d", label, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s lane %d: %08x want %08x", label, i,
				math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

type cudaV41RoPEWitness interface {
	cudaV41TailRoPEReadback(Tensor) ([]float32, error)
	cudaV41TailRoPESnapshot() [7]uint64
	cudaV41TailRoPEInject(int) error
	cudaV41TailRoPEProbe(Tensor, Tensor, Tensor, int) int
	cudaV41TailRoPEAllocationFault(int) error
	cudaV41TailRoPEBadStorage(Tensor, int) Tensor
}

// Not a launch request. Requires a separately admitted physical CUDA run.
// A skip, source pass, registered name, or CPU oracle is not hardware evidence.
// Ordinary bounds are copied from TestVulkanV41TailRoPEQK: unchanged prefix
// and cancellation lane bit-exact, other finite tails abs/rel 2e-6.
// fak-test:runtime integration est=2s lane=optin
func TestCUDAV41TailRoPEDevice(t *testing.T) {
	if os.Getenv("FAK_TEST_CUDA_V41_TAIL_ROPE") != "1" {
		t.Skip("requires separately admitted CUDA tail-RoPE hardware witness")
	}
	backend, found := Lookup("cuda")
	if !found {
		t.Fatal("required CUDA backend absent; CPU substitution is forbidden")
	}
	qk, ok := backend.(V41TailRoPEBackend)
	if !ok || !qk.SupportsV41TailRoPE() {
		t.Fatal("ordinary CUDA tail-RoPE candidate absent")
	}
	rows, ok := backend.(V41TailRoPERowsBackend)
	if !ok || !rows.V41TailRoPERowsAvailable() {
		t.Fatal("ordinary CUDA row-panel candidate absent")
	}
	if exact, ok := backend.(V41TailRoPEExactBackend); ok && exact.V41TailRoPEExactAvailable() {
		t.Fatal("ordinary candidate must not claim separately qualified exact arithmetic")
	}
	checked, ok := backend.(cudaV41RoPEWitness)
	if !ok {
		t.Fatal("checked candidate readback absent; ordinary CUDA Read logs rather than returns transfer errors")
	}
	read := func(input Tensor) []float32 {
		t.Helper()
		values, err := checked.cudaV41TailRoPEReadback(input)
		if err != nil {
			t.Fatal(err)
		}
		return values
	}
	for _, dims := range [][3]int{{2, 7, 4}, {4, 512, 64}} {
		t.Run(fmt.Sprintf("h%d_d%d_r%d", dims[0], dims[1], dims[2]), func(t *testing.T) {
			heads, width, rotary := dims[0], dims[1], dims[2]
			q, kv, table := make([]float32, heads*width), make([]float32, width), make([]float32, rotary)
			fill := func(x []float32) {
				for i := range x {
					x[i] = float32((i%29)-13)*0.03125 + 0.00390625
				}
				for base := 0; base < len(x); base += width {
					x[base] = math.Float32frombits(0x80000000)
					x[base+1] = math.Float32frombits(0x7fc01234)
					x[base+width-rotary] = math.Float32frombits(0x3f800001)
					x[base+width-rotary+1] = 1
				}
			}
			fill(q)
			fill(kv)
			for j := 0; j < rotary/2; j++ {
				table[2*j], table[2*j+1] = float32(j+1)*0.0625, float32(j%3-1)*0.375+0.125
			}
			table[0], table[1] = 1, math.Float32frombits(0x3f7ffffe)
			upload := func(shape []int, values []float32) Tensor {
				d := backend.Upload(NewF32(Default(), shape, values), F32)
				t.Cleanup(func() { backend.Free(d) })
				return d
			}
			dq, dk, dt := upload([]int{heads, width}, q), upload([]int{width}, kv), upload([]int{rotary / 2, 2}, table)
			before := checked.cudaV41TailRoPESnapshot()
			qo, ko, err := qk.V41TailRoPEQK(dq, dk, dt, heads, width, rotary)
			if err != nil {
				t.Fatal(err)
			}
			after := checked.cudaV41TailRoPESnapshot()
			if after[0] != before[0]+1 || after[1] != before[1]+1 || after[2] != before[2] || after[3] != before[3] || after[6] != before[6]+2 {
				t.Fatalf("QK did not perform one zero-host-transfer dispatch with two owned outputs: %v -> %v", before, after)
			}
			t.Cleanup(func() { backend.Free(qo); backend.Free(ko) })
			for _, output := range []Tensor{qo, ko} {
				if output.Buf() == nil || output.Backend() != backend || output.Dtype != F32 ||
					output.Layout != RowMajor || output.Quant != nil || !output.Ready() {
					t.Fatal("candidate did not publish ready caller-owned row-major F32")
				}
				for _, input := range []Tensor{dq, dk, dt} {
					if output.Buf() == input.Buf() {
						t.Fatal("output aliases an input")
					}
				}
			}
			if qo.Buf() == ko.Buf() {
				t.Fatal("outputs alias one another")
			}
			check := func(label string, got, input, supplied []float32, cancellation bool) {
				want := v41TailRoPEReference(input, supplied, width, rotary)
				if len(got) != len(want) {
					t.Fatalf("%s: output length mismatch", label)
				}
				for i := range want {
					dim := i % width
					if dim < width-rotary || cancellation && dim == width-rotary {
						cudaV41RoPEBitsEqual(t, label, got[i:i+1], want[i:i+1])
						continue
					}
					delta := math.Abs(float64(got[i]) - float64(want[i]))
					if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) ||
						delta > 2e-6+2e-6*math.Abs(float64(want[i])) {
						t.Fatalf("%s tail %d: got=%g want=%g", label, i, got[i], want[i])
					}
				}
			}
			check("Q", read(qo), q, table, true)
			check("KV", read(ko), kv, table, true)
			for repeat := 0; repeat < 3; repeat++ {
				a, b, err := qk.V41TailRoPEQK(dq, dk, dt, heads, width, rotary)
				if err != nil || a.Buf() == qo.Buf() || b.Buf() == ko.Buf() {
					t.Fatalf("repeated call reused a live result: %v", err)
				}
				backend.Free(a)
				backend.Free(b)
			}
			if recycler, ok := backend.(interface{ Recycle() }); ok {
				recycler.Recycle()
			}
			check("Q survives repeat/free/recycle", read(qo), q, table, true)
			check("KV survives repeat/free/recycle", read(ko), kv, table, true)
			// Inverse output rotation passes negated sine. Do not normalize,
			// regenerate frequencies, swap columns, or introduce BF16 rounding.
			conjugate := append([]float32(nil), table...)
			for j := 0; j < rotary; j += 2 {
				conjugate[j] = -conjugate[j]
			}
			dconj := upload([]int{rotary / 2, 2}, conjugate)
			before = checked.cudaV41TailRoPESnapshot()
			out, err := rows.V41TailRoPERows(dq, dconj, heads, width, rotary)
			if err != nil {
				t.Fatal(err)
			}
			after = checked.cudaV41TailRoPESnapshot()
			if after[0] != before[0]+1 || after[1] != before[1]+1 || after[2] != before[2] || after[3] != before[3] || after[6] != before[6]+1 {
				t.Fatalf("Rows used extra allocation/transfer/dispatch: %v -> %v", before, after)
			}
			t.Cleanup(func() { backend.Free(out) })
			if out.Buf() == nil || out.Buf() == dq.Buf() || out.Buf() == dconj.Buf() {
				t.Fatal("row output missing or aliases an input")
			}
			check("inverse rows", read(out), q, conjugate, false)
			for _, input := range []struct {
				t    Tensor
				want []float32
			}{{dq, q}, {dk, kv}, {dt, table}, {dconj, conjugate}} {
				cudaV41RoPEBitsEqual(t, "immutable input", read(input.t), input.want)
			}
			badShape := dq
			badShape.Shape = []int{heads * width}
			malformed := []Tensor{NewF32(Default(), []int{heads, width}, q), badShape}
			for kind := 0; kind < 7; kind++ {
				malformed = append(malformed, checked.cudaV41TailRoPEBadStorage(dq, kind))
			}
			freed := upload([]int{heads, width}, q)
			backend.Free(freed)
			malformed = append(malformed, freed)
			for _, badQ := range malformed {
				before = checked.cudaV41TailRoPESnapshot()
				x, y, err := qk.V41TailRoPEQK(badQ, dk, dt, heads, width, rotary)
				var typed *CUDAOpError
				if x.Buf() != nil || y.Buf() != nil || !errors.As(err, &typed) {
					t.Fatalf("invalid input did not refuse typed with empty outputs: %v", err)
				}
				if after := checked.cudaV41TailRoPESnapshot(); after != before {
					t.Fatalf("preflight refusal changed counters/ownership: %v -> %v", before, after)
				}
			}
			for _, probe := range []int{0, 1, 2, 6} {
				before = checked.cudaV41TailRoPESnapshot()
				if status := checked.cudaV41TailRoPEProbe(dq, dk, dt, probe); status != -2 {
					t.Fatalf("native extent/alias probe %d returned %d, want -2", probe, status)
				}
				if after := checked.cudaV41TailRoPESnapshot(); after != before {
					t.Fatalf("native rejection changed live ownership or enqueued work: %v -> %v", before, after)
				}
			}
		})
	}
}

// One explicitly selected fault per fresh process. No test starts subprocesses
// or resets a fault latch/quarantine. Stage 5 simulates an unproven drain after a
// real fence; it witnesses guard retention, not a physical lost-context event.
// fak-test:runtime integration est=2s lane=optin
func TestCUDAV41TailRoPEFaults(t *testing.T) {
	stageName := os.Getenv("FAK_TEST_CUDA_V41_TAIL_ROPE_FAULT_STAGE")
	if stageName == "" {
		t.Skip("requires separately admitted one-fault-per-process CUDA witness")
	}
	backend, found := Lookup("cuda")
	if !found {
		t.Fatal("CUDA backend absent; no CPU substitution")
	}
	qk, ok := backend.(V41TailRoPEBackend)
	checked, witness := backend.(cudaV41RoPEWitness)
	if !ok || !qk.SupportsV41TailRoPE() || !witness {
		t.Fatal("candidate/witness adapter absent")
	}
	upload := func(shape []int, values []float32) Tensor {
		d := backend.Upload(NewF32(Default(), shape, values), F32)
		t.Cleanup(func() { backend.Free(d) })
		return d
	}
	q := upload([]int{2, 7}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14})
	kv := upload([]int{7}, []float32{1, 2, 3, 4, 5, 6, 7})
	table := upload([]int{2, 2}, []float32{1, 0, 0, 1})
	refused := func(a, b Tensor, err error) {
		t.Helper()
		var typed *CUDAOpError
		if a.Buf() != nil || b.Buf() != nil || !errors.As(err, &typed) {
			t.Fatalf("selected failure did not withhold outputs with typed cause: %v", err)
		}
	}
	if stageName == "alloc" {
		for after := 0; after <= 1; after++ {
			if err := checked.cudaV41TailRoPEAllocationFault(after); err != nil {
				t.Fatal(err)
			}
			before := checked.cudaV41TailRoPESnapshot()
			a, b, err := qk.V41TailRoPEQK(q, kv, table, 2, 7, 4)
			refused(a, b, err)
			var allocation *Qwen35GDNAllocationError
			var deviceAllocation *DeviceAllocError
			if !errors.As(err, &allocation) || !errors.As(err, &deviceAllocation) {
				t.Fatalf("allocator cause lost: %v", err)
			}
			if now := checked.cudaV41TailRoPESnapshot(); now != before {
				t.Fatalf("allocation refusal leaked or dispatched: %v -> %v", before, now)
			}
		}
		if err := checked.cudaV41TailRoPEAllocationFault(-1); err != nil {
			t.Fatal(err)
		}
		return
	}
	rowsOnly := stageName == "5rows"
	stageText := strings.TrimSuffix(stageName, "rows")
	stage, err := strconv.Atoi(stageText)
	if err != nil || stage < 1 || stage > 5 || (stageName != stageText && !rowsOnly) {
		t.Fatalf("unsupported fault stage %q", stageName)
	}
	if err := checked.cudaV41TailRoPEInject(stage); err != nil {
		t.Fatal(err)
	}
	before := checked.cudaV41TailRoPESnapshot()
	if stage == 4 {
		values, err := checked.cudaV41TailRoPEReadback(q)
		var typed *CUDAOpError
		if values != nil || !errors.As(err, &typed) || typed.Code < 50000 {
			t.Fatalf("checked D2H failure became observable data: %v", err)
		}
	} else if rowsOnly {
		rows, ok := backend.(V41TailRoPERowsBackend)
		if !ok {
			t.Fatal("Rows candidate absent")
		}
		a, err := rows.V41TailRoPERows(q, table, 2, 7, 4)
		refused(a, Tensor{}, err)
	} else {
		a, b, err := qk.V41TailRoPEQK(q, kv, table, 2, 7, 4)
		refused(a, b, err)
	}
	now := checked.cudaV41TailRoPESnapshot()
	if now[0] != before[0] || now[2] != before[2] || now[3] != before[3] {
		t.Fatalf("failed call published operation or host transfers: %v -> %v", before, now)
	}
	wantLaunches := before[1]
	if stage == 2 || stage == 3 || stage == 5 {
		wantLaunches++
	}
	if now[1] != wantLaunches {
		t.Fatalf("unexpected dispatch count: %v -> %v", before, now)
	}
	if stage != 5 {
		if now[4] != 0 || now[5] != 0 || now[6] != before[6] {
			t.Fatalf("proven drain/pre-enqueue refusal leaked live allocations: %v -> %v", before, now)
		}
		return
	}
	wantHeld, wantLive := uint64(5), before[6]+2
	if rowsOnly {
		wantHeld, wantLive = 3, before[6]+1
	}
	if now[4] != wantHeld || now[5] != 1 || now[6] != wantLive {
		t.Fatalf("terminal quarantine did not retain exact inputs and outputs: %v", now)
	}
	for _, route := range []int{3, 4} {
		if code := checked.cudaV41TailRoPEProbe(q, kv, table, route); code != 0 {
			t.Fatalf("quarantine free route %d lost ownership: %d", route, code)
		}
	}
	bf16State := checked.cudaV41TailRoPEProbe(q, kv, table, 5)
	if bf16State != -3 && !(bf16State == -8 && os.Getenv("FAK_TEST_CUDA_V41_TAIL_ROPE_COMPOSED") != "1") {
		t.Fatalf("composed BF16 did not reject uncertain shared stream: %d", bf16State)
	}
	a, b, err := qk.V41TailRoPEQK(q, kv, table, 2, 7, 4)
	refused(a, b, err)
	if final := checked.cudaV41TailRoPESnapshot(); final != now || !qk.SupportsV41TailRoPE() {
		t.Fatalf("closed state allocated/dispatched/recycled or became capability decline: %v -> %v", now, final)
	}
}
