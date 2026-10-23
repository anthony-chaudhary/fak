package compute

import (
	"encoding/binary"
	"errors"
	"go/parser"
	"go/token"
	"math"
	"os"
	"regexp"
	"strings"
	"testing"
)

// This is a source/protocol witness, not a CUDA compilation or device receipt.
// The repository's fcuda_* checker does not inspect this private candidate ABI.
// fak-test:runtime fast est=20ms lane=default
func TestCUDAV41BF16PublishContract(t *testing.T) {
	read := func(path string) string {
		t.Helper()
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	goSource, native := read("cuda_v41_bf16_publish.go"), read("cuda_kernels.cu")
	if !strings.HasPrefix(goSource, "//go:build cuda && cgo && fak_cuda_v41_bf16_publish\n") {
		t.Fatal("ordinary CUDA must not expose the candidate interface")
	}
	if _, err := parser.ParseFile(token.NewFileSet(), "cuda_v41_bf16_publish.go", goSource, parser.AllErrors); err != nil {
		t.Fatal(err)
	}
	t.Run("private-ABI", func(t *testing.T) {
		canonical := func(signature string) string { return strings.Join(strings.Fields(signature), " ") }
		for _, name := range []string{"fak_v41_bf16_publish_state", "fak_v41_bf16_publish_input", "fak_v41_bf16_publish_f32"} {
			pattern := regexp.MustCompile(`int\s+` + name + `\s*\([^)]*\)`)
			declarations, definitions := pattern.FindAllString(goSource, -1), pattern.FindAllString(native, -1)
			if len(declarations) != 1 || len(definitions) != 1 || canonical(declarations[0]) != canonical(definitions[0]) {
				t.Fatalf("private declaration/definition mismatch for %s: %q / %q", name, declarations, definitions)
			}
			if !strings.Contains(native, `extern "C" `+definitions[0]+" {") {
				t.Fatalf("%s must retain C linkage", name)
			}
			before := native[:strings.Index(native, definitions[0])]
			if strings.LastIndex(before, "#ifdef FAK_CUDA_V41_BF16_PUBLISH") <= strings.LastIndex(before, "#endif") {
				t.Fatalf("%s escaped the native candidate gate", name)
			}
		}
	})
	t.Run("native-lifetime-and-integer-seam", func(t *testing.T) {
		for _, required := range []string{
			"#ifdef FAK_CUDA_V41_BF16_PUBLISH",
			"site == 1u || site == 2u || (site >= 5u && site <= 27u)",
			"(u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u",
			"(u & 0x7f800000u) == 0x7f800000u",
			"(rounded & 0x7f800000u) == 0x7f800000u",
			"atomicMin((unsigned int *)&status[2], (unsigned int)i)",
			"cudaMemsetAsync(dStatus, 0xff, 16, g_stream)",
			"status[0] = 0x42463143u;", "status[1] = site;", "status[3] = count;",
			"cudaMemcpy(hStatus, dStatus, 16, cudaMemcpyDeviceToHost)",
			"g_v41_bf16_quarantine[0] = input;", "g_v41_bf16_quarantine[1] = output;",
			"g_v41_bf16_quarantine[2] = status;", "g_v41_bf16_uncertain = true;",
			"if (d == held) return;",
		} {
			if !strings.Contains(native, required) {
				t.Errorf("native contract missing %q", required)
			}
		}
		ordered := func(source string, fragments ...string) {
			t.Helper()
			previous := -1
			for _, fragment := range fragments {
				position := strings.Index(source, fragment)
				if position <= previous {
					t.Fatalf("missing or reordered contract: %q", fragment)
				}
				previous = position
			}
		}
		body := native[strings.Index(native, `extern "C" int fak_v41_bf16_publish_f32(`):]
		ordered(body, "v41_bf16_publish_owned(dInput", "fak_v41_bf16_publish_state()",
			"cudaMemsetAsync", "k_v41_bf16_publish_status<<<", "k_v41_bf16_publish<<<",
			"error = cudaStreamSynchronize", "cudaMemcpy(hStatus", "g_host_bytes += 16")
		ordered(goSource, "validateV41BF16Publish(input, site)", "input.Backend() != c",
			"source.invalidStateError", "c.capturing", "c.faultLatch.Admit", "C.fak_v41_bf16_publish_input",
			"output, err = c.dallocDeviceOnlyClass", "statusBuf, err = c.dallocDeviceOnlyClass",
			"status := int(C.fak_v41_bf16_publish_f32", "decodeV41BF16PublishStatus", "return makeTensor")
		kernel := native[strings.Index(native, "__global__ void k_v41_bf16_publish("):strings.Index(native, "static int v41_bf16_publish_drain(")]
		for _, forbidden := range []string{"float", "__half", "__nv_bfloat16", "__float2", "input[i] ="} {
			if strings.Contains(kernel, forbidden) {
				t.Errorf("integer publication contains %q", forbidden)
			}
		}
		for _, forbidden := range []string{"c.Upload(", "c.Read(", "Default()", "dallocManaged", "c.transient =", "return input, nil"} {
			if strings.Contains(goSource, forbidden) {
				t.Errorf("candidate gained a fallback, alias, or recycling route: %q", forbidden)
			}
		}
	})
	t.Run("fresh-status-envelope", func(t *testing.T) {
		const count, site = 257, uint32(21)
		success := [16]byte{0x43, 0x31, 0x46, 0x42, 21, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 1, 1, 0, 0}
		if err := decodeV41BF16PublishStatus(success[:], count, site); err != nil {
			t.Fatal(err)
		}
		for _, lane := range []int{0, 4, 12} {
			wrong := success
			wrong[lane] ^= 1
			var protocol *V41BF16CheckedProtocolError
			if err := decodeV41BF16PublishStatus(wrong[:], count, site); !errors.As(err, &protocol) {
				t.Fatalf("mismatched field %d accepted: %v", lane, err)
			}
		}
		for _, index := range []uint32{0, 256, 257, math.MaxUint32 - 1} {
			raw := success
			binary.LittleEndian.PutUint32(raw[8:12], index)
			err := decodeV41BF16PublishStatus(raw[:], count, site)
			var arithmetic *V41BF16CheckedArithmeticError
			var protocol *V41BF16CheckedProtocolError
			if index < count {
				if !errors.As(err, &arithmetic) || arithmetic.Index != int(index) || arithmetic.Site != site {
					t.Fatalf("lost lowest-index attribution: %v", err)
				}
			} else if !errors.As(err, &protocol) {
				t.Fatalf("out-of-range status accepted: %v", err)
			}
		}
		for _, raw := range [][]byte{nil, make([]byte, 16), success[:15], append(success[:], 0)} {
			var protocol *V41BF16CheckedProtocolError
			if err := decodeV41BF16PublishStatus(raw, count, site); !errors.As(err, &protocol) {
				t.Fatalf("malformed status accepted: %x %v", raw, err)
			}
		}
	})
}

// Deliberately literal golden words, not outputs computed with the production
// RNE expression. Includes both ties, both signs, FTZ-sensitive subnormals and
// the finite/rounded-overflow edge. This optional test is not a launch request.
// fak-test:runtime integration est=2s lane=optin
func TestCUDAV41BF16PublishDevice(t *testing.T) {
	if os.Getenv("FAK_TEST_CUDA_V41_BF16_PUBLISH") != "1" {
		t.Skip("requires an explicitly admitted candidate CUDA hardware witness")
	}
	backend, found := Lookup("cuda")
	if !found {
		t.Fatal("required CUDA backend absent; CPU substitution is forbidden")
	}
	publish, ok := backend.(V41BF16PublishBackend)
	if !ok {
		t.Fatal("candidate build gate absent")
	}
	golden := []struct{ input, output uint32 }{
		{0x00000000, 0x00000000}, {0x80000000, 0x80000000},
		{0x00000001, 0x00000000}, {0x80000001, 0x80000000},
		{0x00007fff, 0x00000000}, {0x00008000, 0x00000000}, {0x00008001, 0x00010000},
		{0x00018000, 0x00020000}, {0x80018000, 0x80020000},
		{0x007fffff, 0x00800000}, {0x807fffff, 0x80800000},
		{0x3f807fff, 0x3f800000}, {0x3f808000, 0x3f800000}, {0x3f808001, 0x3f810000},
		{0x3f817fff, 0x3f810000}, {0x3f818000, 0x3f820000}, {0x3f818001, 0x3f820000},
		{0xbf808000, 0xbf800000}, {0xbf818000, 0xbf820000},
		{0x7f7f7fff, 0x7f7f0000}, {0xff7f7fff, 0xff7f0000},
	}
	values := make([]float32, 513)
	for i := range values {
		values[i] = math.Float32frombits(golden[i%len(golden)].input)
	}
	input := backend.Upload(NewF32(Default(), []int{len(values)}, values), F32)
	defer backend.Free(input)
	for site := uint32(0); site <= 28; site++ {
		out, err := publish.V41BF16Publish(input, site)
		if !V41BF16PublishSite(site) {
			var typed *CUDAOpError
			if out.Buf() != nil || !errors.As(err, &typed) {
				t.Fatalf("invalid site %d not refused: %v", site, err)
			}
			continue
		}
		if err != nil || out.Buf() == nil || out.Buf() == input.Buf() || out.Backend() != backend ||
			out.Dtype != F32 || out.Layout != RowMajor || out.Quant != nil || len(out.Shape) != 1 || out.Shape[0] != len(values) || !out.Ready() {
			t.Fatalf("site %d did not publish fresh ready F32: %v", site, err)
		}
		got := backend.Read(out)
		backend.Free(out)
		if len(got) != len(values) {
			t.Fatal("wrong output size")
		}
		for i, value := range got {
			if bits := math.Float32bits(value); bits != golden[i%len(golden)].output {
				t.Fatalf("site %d lane %d: %08x != %08x", site, i, bits, golden[i%len(golden)].output)
			}
		}
	}
	for _, bits := range []uint32{0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001, 0xffffffff, 0x7f7f8000, 0xff7f8000} {
		bad := append([]float32(nil), values...)
		bad[256], bad[512] = math.Float32frombits(bits), math.Float32frombits(0x7fc00000)
		resident := backend.Upload(NewF32(Default(), []int{len(bad)}, bad), F32)
		out, err := publish.V41BF16Publish(resident, 21)
		backend.Free(resident)
		var arithmetic *V41BF16CheckedArithmeticError
		if out.Buf() != nil || !errors.As(err, &arithmetic) || arithmetic.Index != 256 || arithmetic.Site != 21 {
			t.Fatalf("%08x lost lowest-index failure: %v", bits, err)
		}
		// Same site and count immediately after failure must get a fresh success.
		out, err = publish.V41BF16Publish(input, 21)
		if err != nil || out.Buf() == nil {
			t.Fatalf("stale status poisoned the next publication: %v", err)
		}
		backend.Free(out)
	}
	for i, value := range backend.Read(input) {
		if math.Float32bits(value) != golden[i%len(golden)].input {
			t.Fatalf("input was mutated at %d", i)
		}
	}
	for _, malformed := range []Tensor{
		NewF32(Default(), []int{len(values)}, values),
		{be: backend, buf: input.buf, Dtype: F16, Layout: RowMajor, Shape: []int{len(values)}},
		{be: backend, buf: input.buf, Dtype: F32, Layout: ColMajor, Shape: []int{len(values)}},
		{be: backend, buf: input.buf, Dtype: F32, Layout: RowMajor, Shape: []int{1, len(values)}},
		{be: backend, buf: input.buf, Dtype: F32, Layout: RowMajor, Shape: []int{len(values) + 1}},
	} {
		out, err := publish.V41BF16Publish(malformed, 21)
		var typed *CUDAOpError
		if out.Buf() != nil || !errors.As(err, &typed) {
			t.Fatalf("metadata/residency failure was not typed: %v", err)
		}
	}
}
