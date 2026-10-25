package compute

import "testing"

// fak-test:runtime fast est=10ms lane=default
func TestCUDADeviceOpDeclaration(t *testing.T) {
	t.Parallel()
	table := CUDADeviceOpTable()
	expected := map[[2]string]struct{}{
		{"matmul", "f32"}:                             {},
		{"matmul", "f16"}:                             {},
		{"matmul", "q8_0"}:                            {},
		{"matmul", "q4_k"}:                            {},
		{"matmul", "q5_k"}:                            {},
		{"matmul", "q6_k"}:                            {},
		{"matmul", "q2_0"}:                            {},
		{"matmul", "q2_k"}:                            {},
		{"matmul", "nvfp4"}:                           {},
		{"routed-expert-gemv", "F8_E4M3+F8_E8M0"}:     {},
		{"routed-expert-gemv", "I8+F8_E8M0"}:          {},
		{"routed-expert-resident", "F8_E4M3+F8_E8M0"}: {},
		{"routed-expert-resident", "I8+F8_E8M0"}:      {},
	}
	seen := make(map[[2]string]struct{}, len(expected))
	for _, row := range table {
		key := [2]string{row.Operation, row.Format}
		if _, ok := expected[key]; !ok {
			t.Errorf("unexpected operation/format declaration %q/%q", row.Operation, row.Format)
		}
		if _, duplicate := seen[key]; duplicate {
			t.Errorf("duplicate operation/format declaration %q/%q", row.Operation, row.Format)
		}
		seen[key] = struct{}{}
	}
	for key := range expected {
		if _, ok := seen[key]; !ok {
			t.Errorf("missing operation/format declaration %q/%q", key[0], key[1])
		}
	}
	// Independent oracle from cudaBackend.MatMul, not derived from the table.
	for value := 0; value < 256; value++ {
		dt := Dtype(value)
		want := false
		switch dt {
		case F32, F16, Q8_0, Q4_K, Q5_K, Q6_K, Q2_0, Q2_K, NVFP4:
			want = true
		}
		if got := cudaDeviceWeightDtypeSupported(dt); got != want {
			t.Errorf("dtype %d: shared predicate=%v, want %v", value, got, want)
		}
		for _, card := range []string{"RTX 5070 GB205", "RTX 5090 GB202"} {
			if got := CUDADeviceOpSupported(table, "matmul", dt.String(), "sm_120"); got != want {
				t.Errorf("%s code-side declaration dtype %d=%v, want %v", card, value, got, want)
			}
		}
	}
	for _, operation := range []string{"routed-expert-gemv", "routed-expert-resident"} {
		for _, format := range []string{"F8_E4M3+F8_E8M0", "I8+F8_E8M0"} {
			if !CUDADeviceOpSupported(table, operation, format, "sm_120") {
				t.Errorf("missing expert declaration %s/%s", operation, format)
			}
		}
	}
	for _, row := range table {
		if row.MinArch != "sm_80" {
			t.Errorf("unexpected current source floor: %+v", row)
		}
	}
	for _, arch := range []string{"sm_70", "sm_110", "", "sm_0120", "sm_120a", "sm_100a", "sm_103a", "compute_120", "sm_+120"} {
		if CUDADeviceOpSupported(table, "matmul", "f32", arch) {
			t.Errorf("accepted unsupported or malformed target %q", arch)
		}
	}
	for _, floor := range []string{"", "sm_70", "sm_81", "sm_110", "sm_100a", "sm_103a", "sm_0120"} {
		if CUDADeviceOpSupported([]CUDADeviceOp{{"matmul", "f32", floor}}, "matmul", "f32", "sm_120") {
			t.Errorf("accepted unknown or feature-specific floor %q", floor)
		}
	}
	if CUDADeviceOpSupported([]CUDADeviceOp{{"matmul", "f32", "sm_120"}}, "matmul", "f32", "sm_89") {
		t.Fatal("accepted target below operation floor")
	}
	for _, request := range [][2]string{{"unknown", "f32"}, {"matmul", "F8_E4M3"}, {"matmul", "I8"}, {"matmul-nvfp4-w4a4", "nvfp4"}, {"routed-expert-gemv", "nvfp4"}} {
		if CUDADeviceOpSupported(table, request[0], request[1], "sm_120") {
			t.Errorf("accepted undeclared operation/format %v", request)
		}
	}
	withoutF32 := make([]CUDADeviceOp, 0, len(table))
	for _, row := range table {
		if row.Operation != "matmul" || row.Format != "f32" {
			withoutF32 = append(withoutF32, row)
		}
	}
	if CUDADeviceOpSupported(withoutF32, "matmul", "f32", "sm_120") || CUDADeviceOpSupported(nil, "matmul", "f32", "sm_120") {
		t.Fatal("absence of a supplied row did not fail closed")
	}
	duplicate := []CUDADeviceOp{{"matmul", "f32", "sm_80"}, {"matmul", "f32", "sm_80"}}
	if CUDADeviceOpSupported(duplicate, "matmul", "f32", "sm_120") {
		t.Fatal("ambiguous duplicate rows accepted")
	}
	table[0].Format = "mutated"
	if !CUDADeviceOpSupported(CUDADeviceOpTable(), "matmul", "f32", "sm_120") || !cudaDeviceWeightDtypeSupported(F32) {
		t.Fatal("caller mutated the shared declaration")
	}
}
