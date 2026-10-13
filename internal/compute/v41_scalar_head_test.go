package compute

import (
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
)

// This is a CPU lifecycle, byte-range and frozen-source witness. It does not
// compile/link the native bridge, execute Vulkan or measure throughput.
// fak-test:runtime fast est=20ms lane=default
func TestV41ScalarHeadCompositionContract(t *testing.T) {
	for rows := 1; rows <= 4; rows++ {
		if !v41ScalarHeadShape(Q6_K, 129280, 5120, rows) {
			t.Fatalf("survivor count %d refused", rows)
		}
		var events []int
		err := runV41ScalarHeadWindow(rows, func() error {
			events = append(events, -1)
			return nil
		}, func(row int) error {
			events = append(events, row)
			// Source-bound strides below address each row exactly once, with
			// no shared accumulation and no overlap in panel destinations.
			for _, width := range []int{5120, 129280} {
				start, end := row*width*4, (row+1)*width*4
				if start < 0 || end > rows*width*4 || end-start != width*4 {
					t.Fatalf("rows=%d row=%d width=%d invalid byte range", rows, row, width)
				}
			}
			return nil
		}, func() error {
			events = append(events, -2)
			return nil
		})
		want := []int{-1}
		for row := 0; row < rows; row++ {
			want = append(want, row)
		}
		want = append(want, -2)
		if err != nil || !reflect.DeepEqual(events, want) {
			t.Fatalf("rows=%d events=%v err=%v", rows, events, err)
		}
	}
	for _, shape := range [][3]int{{129280, 5120, 0}, {129280, 5120, 5}, {129279, 5120, 2}, {129280, 4096, 2}} {
		if v41ScalarHeadShape(Q6_K, shape[0], shape[1], shape[2]) {
			t.Fatalf("unsupported shape admitted: %v", shape)
		}
	}
	if v41ScalarHeadShape(Q2_K, 129280, 5120, 2) {
		t.Fatal("non-Q6 head admitted")
	}
	first, last := errors.New("first failure"), errors.New("end failure")
	for _, failure := range []string{"begin", "row", "row-panic", "row-and-end-panic", "end", "end-panic"} {
		t.Run(failure, func(t *testing.T) {
			begins, rows, ends := 0, 0, 0
			var recovered any
			var err error
			func() {
				defer func() { recovered = recover() }()
				err = runV41ScalarHeadWindow(4, func() error {
					begins++
					if failure == "begin" {
						return first
					}
					return nil
				}, func(row int) error {
					rows++
					if row == 1 && strings.HasPrefix(failure, "row") {
						if strings.Contains(failure, "panic") {
							panic(first)
						}
						return first
					}
					return nil
				}, func() error {
					ends++
					if failure == "end-panic" || failure == "row-and-end-panic" {
						panic(last)
					}
					return last
				})
			}()
			wantRows, wantEnds := 4, 1
			wantErr, wantPanic := last, any(nil)
			if failure == "begin" {
				wantRows, wantEnds, wantErr = 0, 0, first
			} else if strings.HasPrefix(failure, "row") {
				wantRows, wantErr = 2, first
				if strings.Contains(failure, "panic") {
					wantErr, wantPanic = nil, first
				}
			} else if failure == "end-panic" {
				wantErr, wantPanic = nil, last
			}
			if begins != 1 || rows != wantRows || ends != wantEnds || err != wantErr || recovered != wantPanic {
				t.Fatalf("begin/row/end=%d/%d/%d err=%v panic=%v", begins, rows, ends, err, recovered)
			}
		})
	}

	read := func(name string) string {
		t.Helper()
		b, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	require := func(source string, fragments ...string) {
		t.Helper()
		for _, fragment := range fragments {
			if !strings.Contains(source, fragment) {
				t.Fatalf("source contract missing %q", fragment)
			}
		}
	}
	head := read("vulkan_v41_shared_head.go")
	require(head,
		"v != vulkanDev || C.fvk_device_type() != 1", "uint32(vendor) == vulkanAMDVendorID && device != 0",
		"C.fvk_batch_active() || C.fvk_v41_step_batch_active() != 0",
		"v.maxBufferBytes, v.maxStorageBufferRange, v.maxMemoryAllocationSize",
		"xs[0], ys[0] = x, y", "if rows > 1 {", "var inputRows, outputRows [4]*vulkanBuf",
		"C.fvk_d2d_range(inputRows[row].ptr, 0, xb.ptr, C.size_t(row*in*4), C.size_t(in*4))",
		"v.q6kMatMulLocked(w, xs[row], ys[row], 1)",
		"C.fvk_d2d_range(output.ptr, C.size_t(row*out*4), outputRows[row].ptr, 0, C.size_t(out*4))",
		"C.fvk_v41_step_batch_begin()", "C.fvk_v41_step_batch_end()", "C.fvk_submission_status()",
		"v.freeDeferredLocked(inputRows[:]...)", "v.freeDeferredLocked(outputRows[:]...)",
		"if !completed {\n\t\t\tv.freeDeferredLocked(output)", "output.v41CheckedRead = true")
	if strings.Count(head, "v.q6kMatMulLocked(") != 1 || strings.Contains(head, "v.Read(") || strings.Contains(head, "v.Upload(") {
		t.Fatal("composition must have exactly one scalar call site and no host computation")
	}
	for _, f := range []string{"vulkan.go", "vulkan_matmul_fused.go"} {
		source := read(f)
		start := strings.Index(source, "func (v *vulkanBackend) "+map[string]string{"vulkan.go": "MatMul(w, x", "vulkan_matmul_fused.go": "BatchedMatMul(w, X"}[f])
		if start < 0 {
			t.Fatalf("missing public matmul in %s", f)
		}
		source = source[start:]
		selected, allocated := strings.Index(source, "v.v41ScalarHeadLocked("), strings.Index(source, "v.devTr(")
		if selected < 0 || allocated < selected {
			t.Fatalf("%s must select composition before generic allocation", f)
		}
		require(source, "v.v41ScalarHeadAvailableLocked(out, in,")
	}
	shim := read("vulkan_shim.cpp")
	require(shim,
		"g_batchCmd = g_stepBatch ? v41BeginCmdChecked() : beginCmd();\n\tif (g_batchCmd == VK_NULL_HANDLE) return;",
		"const bool checkedV41 = forceChecked || g_stepBatch ||",
		"DescriptorSetRecord rec = acquireDescriptorSet(k, checkedV41);",
		"g_submissionStatus = allocated < VK_SUCCESS ? allocated : VK_ERROR_UNKNOWN;",
		"if (g_batchHasV41)", "v41EndSubmitWaitChecked(g_batchCmd, hadWork)",
		"if (g_batching || g_v41SubmissionPendingFailure) { g_batchFreed.push_back(b); return; }",
		"struct PC { int out, in, p; } pc{out, in, P == 1 ? -1 : P};")
}
