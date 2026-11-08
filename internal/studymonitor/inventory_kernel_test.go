package studymonitor

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func TestBuildInventoryMapCountsGPUKernelsAndWalksSubmodules(t *testing.T) {
	root := t.TempDir()
	writeInventoryFixture(t, root, ".gitmodules", "[submodule \"vendor/K\"]\n\tpath = vendor/K\n\turl = https://example/K\n[submodule \"vendor/Missing\"]\n\tpath = vendor/Missing\n")
	writeInventoryFixture(t, root, "vendor/K/kernels/gemm.cu", "__global__ void gemm() {}\n")
	writeInventoryFixture(t, root, "vendor/K/include/impl.hpp", "template <int N>\n__global__ void tile() {}\n")
	writeInventoryFixture(t, root, "vendor/K/include/plain.hpp", "int helper();\n")
	writeInventoryFixture(t, root, "ops/tl.py", "import tilelang\n@tilelang.jit\ndef fused(): pass\n")
	writeInventoryFixture(t, root, "ops/plain.py", "def host(): pass\n")
	writeInventoryFixture(t, root, "tests/test_kernel.py", "@triton.jit\ndef ref(): pass\n")
	writeInventoryFixture(t, root, "node_modules/pkg/k.cu", "__global__ void ignored() {}\n")

	report := buildInventoryMapForTest(t, root, InventoryMapOptions{})
	if report.Totals.KernelFiles != 3 {
		t.Fatalf("kernel files = %d, want 3 (cu, __global__ header, tilelang python)", report.Totals.KernelFiles)
	}
	assertClassStatus(t, report, KernelInventorySourceClass, InventoryClassExternalRequired)
	evidence := classEvidence(t, report, KernelInventorySourceClass)
	for _, want := range []string{"vendor/K/kernels/gemm.cu", "vendor/K/include/impl.hpp", "ops/tl.py"} {
		if !containsString(evidence, want) {
			t.Fatalf("kernel evidence = %v, want %s", evidence, want)
		}
	}
	for _, unwanted := range []string{"ops/plain.py", "tests/test_kernel.py", "vendor/K/include/plain.hpp"} {
		if containsString(evidence, unwanted) {
			t.Fatalf("kernel evidence = %v, must not count %s", evidence, unwanted)
		}
	}
	if containsString(report.SkippedDirs, "vendor") || !containsString(report.SkippedDirs, "node_modules") {
		t.Fatalf("skipped dirs = %v, want submodule parent walked and node_modules skipped", report.SkippedDirs)
	}
	if report.SkippedKernelFiles != 0 {
		t.Fatalf("skipped kernel files = %d, want node_modules never scanned", report.SkippedKernelFiles)
	}
	if !containsString(report.Submodules, "vendor/K") || len(report.UnpopulatedSubmodules) != 1 || report.UnpopulatedSubmodules[0] != "vendor/Missing" {
		t.Fatalf("submodules = %v unpopulated = %v, want vendor/K walked and vendor/Missing unpopulated", report.Submodules, report.UnpopulatedSubmodules)
	}
	if vendor := findSubsystem(t, report, "vendor"); vendor.KernelFiles != 2 {
		t.Fatalf("vendor subsystem = %+v, want 2 kernel files", vendor)
	}
}

func TestBuildInventoryMapSubmoduleParentIsPassThroughOnly(t *testing.T) {
	root := t.TempDir()
	writeInventoryFixture(t, root, ".gitmodules", "[submodule \"K\"]\n\tpath = vendor/K\n")
	writeInventoryFixture(t, root, "vendor/K/kernels/gemm.cu", "__global__ void gemm() {}\n")
	writeInventoryFixture(t, root, "vendor/K/.gitmodules", "[submodule \"nested\"]\n\tpath = build/nested\n")
	writeInventoryFixture(t, root, "vendor/K/build/nested/attn.cu", "__global__ void attn() {}\n")
	writeInventoryFixture(t, root, "vendor/K/build/CMakeFiles/gen.cu", "__global__ void generated() {}\n")
	writeInventoryFixture(t, root, "vendor/K/build/loose.cu", "__global__ void loose() {}\n")
	writeInventoryFixture(t, root, "vendor/github.com/foo/bar.go", "package foo\n")
	writeInventoryFixture(t, root, "vendor/otherlib/dep.cu", "__global__ void dep() {}\n")
	writeInventoryFixture(t, root, "vendor/modules.txt", "# vendored\n")

	report := buildInventoryMapForTest(t, root, InventoryMapOptions{})
	for _, want := range []string{"vendor/github.com", "vendor/otherlib", "vendor/K/build/CMakeFiles"} {
		if !containsString(report.SkippedDirs, want) {
			t.Fatalf("skipped dirs = %v, want %s", report.SkippedDirs, want)
		}
	}
	for _, unwanted := range []string{"vendor", "vendor/K/build"} {
		if containsString(report.SkippedDirs, unwanted) {
			t.Fatalf("skipped dirs = %v, %s leads to a submodule and is pass-through", report.SkippedDirs, unwanted)
		}
	}
	if !reflect.DeepEqual(report.Submodules, []string{"vendor/K", "vendor/K/build/nested"}) || len(report.UnpopulatedSubmodules) != 0 {
		t.Fatalf("submodules = %v unpopulated = %v, want root and nested submodule walked", report.Submodules, report.UnpopulatedSubmodules)
	}
	// .gitmodules, vendor/K/.gitmodules, gemm.cu, attn.cu: sibling and loose files are not walked.
	if report.Totals.Files != 4 || report.Totals.RuntimeFiles != 2 || report.Totals.KernelFiles != 2 {
		t.Fatalf("totals = %+v, want only submodule files walked", report.Totals)
	}
	evidence := classEvidence(t, report, KernelInventorySourceClass)
	for _, unwanted := range []string{"vendor/otherlib/dep.cu", "vendor/github.com/foo/bar.go"} {
		for _, subsystem := range report.Subsystems {
			if containsString(subsystem.ExamplePaths, unwanted) {
				t.Fatalf("subsystem %s walked sibling %s", subsystem.Path, unwanted)
			}
		}
	}
	if !containsString(evidence, "vendor/K/build/nested/attn.cu") {
		t.Fatalf("kernel evidence = %v, want nested submodule kernel", evidence)
	}
	wantSkipped := []string{"vendor/K/build/CMakeFiles/gen.cu", "vendor/K/build/loose.cu", "vendor/otherlib/dep.cu"}
	if report.SkippedKernelFiles != 3 || !reflect.DeepEqual(report.SkippedKernelExamples, wantSkipped) {
		t.Fatalf("skipped kernels = %d %v, want %v", report.SkippedKernelFiles, report.SkippedKernelExamples, wantSkipped)
	}
	assertTotalsMatchSubsystems(t, report)
}

func TestBuildInventoryMapReportsNestedUnpopulatedSubmodules(t *testing.T) {
	root := t.TempDir()
	writeInventoryFixture(t, root, ".gitmodules", "[submodule \"flashinfer\"]\n\tpath = 3rdparty/flashinfer\n")
	writeInventoryFixture(t, root, "3rdparty/flashinfer/.gitmodules", "[submodule \"cutlass\"]\n\tpath = 3rdparty/cutlass\n[submodule \"x\"]\n\tpath = vendor/x\n")
	writeInventoryFixture(t, root, "3rdparty/flashinfer/csrc/decode.cu", "__global__ void decode() {}\n")
	if err := os.MkdirAll(filepath.Join(root, "3rdparty", "flashinfer", "3rdparty", "cutlass"), 0o755); err != nil {
		t.Fatal(err)
	}

	report := buildInventoryMapForTest(t, root, InventoryMapOptions{})
	want := []string{"3rdparty/flashinfer/3rdparty/cutlass", "3rdparty/flashinfer/vendor/x"}
	if !reflect.DeepEqual(report.UnpopulatedSubmodules, want) {
		t.Fatalf("unpopulated = %v, want nested empty submodules %v", report.UnpopulatedSubmodules, want)
	}
	if !containsString(report.Submodules, "3rdparty/flashinfer") || len(report.Submodules) != 3 {
		t.Fatalf("submodules = %v, want root plus two nested", report.Submodules)
	}
}

func TestParseGitmodulesPathsFollowsGitConfigSyntax(t *testing.T) {
	data := strings.Join([]string{
		"# leading comment",
		"[submodule \"a\"]",
		"\tpath = \"vendor/K\"",
		"[submodule \"b\"]",
		"\tpath = vendor/L # pinned",
		"[submodule \"c\"]",
		"\tPath = vendor/M",
		"[Submodule \"d\"]",
		"\tPATH = \"with \\\"quote\\\"\" ; trailing",
		"[core]",
		"\tpath = not/a/submodule",
		"[submodule \"e\"]",
		"\tpath = /abs/path",
		"[submodule \"f\"]",
		"\tpath = C:/drive",
		"[submodule \"g\"]",
		"\tpath = ../escape",
		"[submodule \"h\"]",
		"\tpath = a/../../escape",
		"[submodule \"i\"] path = inline/header",
		"[submodule \"j\"]",
		"\tpath = \"spaced dir\"   \r",
		"[submodule \"k\"]",
		"\tpath = \"unterminated",
		"[submodule.legacy]",
		"\tpath = legacy/form",
	}, "\n")
	got := parseGitmodulesPaths([]byte(data))
	want := []string{"vendor/K", "vendor/L", "vendor/M", "with \"quote\"", "inline/header", "spaced dir", "legacy/form"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("paths = %q, want %q", got, want)
	}
}

func TestParseGitConfigValueEscapes(t *testing.T) {
	tests := []struct {
		raw  string
		want string
		ok   bool
	}{
		{raw: `  plain  `, want: "plain", ok: true},
		{raw: ` "a\\b" `, want: `a\b`, ok: true},
		{raw: ` "tab\there" `, want: "tab\there", ok: true},
		{raw: ` "line\nbreak" `, want: "line\nbreak", ok: true},
		{raw: ` "keep # inside" # drop`, want: "keep # inside", ok: true},
		{raw: ` a b ; drop`, want: "a b", ok: true},
		{raw: ` "open`, ok: false},
		{raw: ` bad\q`, ok: false},
	}
	for _, tt := range tests {
		got, ok := parseGitConfigValue(tt.raw)
		if ok != tt.ok || (ok && got != tt.want) {
			t.Fatalf("parseGitConfigValue(%q) = %q,%v, want %q,%v", tt.raw, got, ok, tt.want, tt.ok)
		}
	}
}

func TestBuildInventoryMapExcludedSubmodulesClearUnpopulatedButKeepKernelsUnproven(t *testing.T) {
	root := t.TempDir()
	writeInventoryFixture(t, root, ".gitmodules", "[submodule \"K\"]\n\tpath = vendor/K\n[submodule \"M\"]\n\tpath = 3rdparty/M\n")
	writeInventoryFixture(t, root, "cmd/app/main.go", "package main\n")
	writeInventoryFixture(t, root, "vendor/K/.git", "gitdir: ../../.git/modules/K\n")

	report := buildInventoryMapForTest(t, root, InventoryMapOptions{ExcludedSubmodules: map[string]string{
		"./vendor/K/": "private SSH-only URL",
	}})
	if !reflect.DeepEqual(report.ExcludedSubmodules, []InventoryExcludedSubmodule{{Path: "vendor/K", Reason: "private SSH-only URL"}}) {
		t.Fatalf("excluded = %+v", report.ExcludedSubmodules)
	}
	if !reflect.DeepEqual(report.UnpopulatedSubmodules, []string{"3rdparty/M"}) {
		t.Fatalf("unpopulated = %v, want only the unexcluded empty submodule", report.UnpopulatedSubmodules)
	}
	if report.Totals.KernelFiles != 0 || report.SkippedKernelFiles != 0 {
		t.Fatalf("totals = %+v skipped kernels = %d, want the excluded submodule neither walked nor scanned", report.Totals, report.SkippedKernelFiles)
	}
	if !strings.Contains(report.CompletenessNote, "vendor/K (private SSH-only URL)") {
		t.Fatalf("completeness note = %q, want the exclusion and its reason", report.CompletenessNote)
	}
	if !report.KernelInventoryRequired() {
		t.Fatalf("report = %+v, want an excluded submodule to keep kernel_inventory required", report)
	}
	assertClassStatus(t, report, KernelInventorySourceClass, InventoryClassExternalRequired)
	if note := classNote(t, report, KernelInventorySourceClass); !strings.Contains(note, "excluded submodules never walked or scanned: vendor/K") {
		t.Fatalf("kernel_inventory note = %q, want the excluded path named", note)
	}

	for name, tt := range map[string]struct {
		excluded map[string]string
		want     error
	}{
		"undeclared":   {excluded: map[string]string{"vendor/Other": "gone"}, want: ErrExcludedSubmoduleUndeclared},
		"empty reason": {excluded: map[string]string{"vendor/K": "  "}, want: ErrExcludedSubmoduleReason},
		"escaping":     {excluded: map[string]string{"../K": "outside"}, want: ErrExcludedSubmodulePath},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := BuildInventoryMap(root, InventoryMapOptions{Repository: "owner/repo", IndexedRevision: "abc", ExcludedSubmodules: tt.excluded})
			if !errors.Is(err, tt.want) {
				t.Fatalf("err = %v, want %v", err, tt.want)
			}
		})
	}
}

func TestBuildInventoryMapScansSkippedVendorForKernels(t *testing.T) {
	root := t.TempDir()
	writeInventoryFixture(t, root, "cmd/app/main.go", "package main\nfunc main() {}\n")
	writeInventoryFixture(t, root, "vendor/dep/dep.cu", "__global__ void notWalked() {}\n")

	report := buildInventoryMapForTest(t, root, InventoryMapOptions{})
	if report.Totals.KernelFiles != 0 || !containsString(report.SkippedDirs, "vendor") {
		t.Fatalf("totals = %+v skipped = %v, want no walked kernels and package vendor/ skipped without .gitmodules", report.Totals, report.SkippedDirs)
	}
	if report.SkippedKernelFiles != 1 || !reflect.DeepEqual(report.SkippedKernelExamples, []string{"vendor/dep/dep.cu"}) {
		t.Fatalf("skipped kernels = %d %v, want vendor/dep/dep.cu", report.SkippedKernelFiles, report.SkippedKernelExamples)
	}
	assertClassStatus(t, report, KernelInventorySourceClass, InventoryClassExternalRequired)
	assertTotalsMatchSubsystems(t, report)
}

func TestBuildInventoryMapWithoutKernelsOmitsKernelInventoryRow(t *testing.T) {
	root := t.TempDir()
	writeInventoryFixture(t, root, "cmd/app/main.go", "package main\nfunc main() {}\n")
	writeInventoryFixture(t, root, "vendor/dep/dep.go", "package dep\n")
	writeInventoryFixture(t, root, "node_modules/pkg/k.cu", "__global__ void neverScanned() {}\n")
	writeInventoryFixture(t, root, "vendor/dep/.venv/lib/k.cu", "__global__ void neverScanned() {}\n")

	report := buildInventoryMapForTest(t, root, InventoryMapOptions{})
	if report.Totals.KernelFiles != 0 || report.SkippedKernelFiles != 0 || report.SkippedKernelScanTruncated || report.KernelInventoryRequired() {
		t.Fatalf("report = %+v, want no kernels anywhere", report)
	}
	// Readers that predate kernel_inventory refuse an unknown class row, so a
	// kernel-free map carries only the base classes.
	for _, row := range report.SourceClasses {
		if !isRequiredInventorySourceClass(row.Class) {
			t.Fatalf("source class %q in a kernel-free map, want only the base classes", row.Class)
		}
	}
	if !strings.Contains(report.CompletenessNote, "kernel_inventory is not required") {
		t.Fatalf("completeness note = %q, want the kernel scan result recorded", report.CompletenessNote)
	}
}

func TestBuildInventoryMapSkippedKernelScanIsBounded(t *testing.T) {
	root := t.TempDir()
	writeInventoryFixture(t, root, "cmd/app/main.go", "package main\n")
	for _, name := range []string{"a.go", "b.go", "c.go", "d.go", "e.go"} {
		writeInventoryFixture(t, root, "build/"+name, "package build\n")
	}

	report := buildInventoryMapForTest(t, root, InventoryMapOptions{skippedKernelScanBudget: 3})
	if !report.SkippedKernelScanTruncated || !report.KernelInventoryRequired() {
		t.Fatalf("report = %+v, want a truncated scan to keep kernel_inventory required", report)
	}
	assertClassStatus(t, report, KernelInventorySourceClass, InventoryClassExternalRequired)
	if note := classNote(t, report, KernelInventorySourceClass); !strings.Contains(note, "truncated") {
		t.Fatalf("kernel_inventory note = %q, want truncation named", note)
	}
}

func TestBuildInventoryMapRefusesExcludingCheckedOutSubmodule(t *testing.T) {
	root := t.TempDir()
	writeInventoryFixture(t, root, ".gitmodules", "[submodule \"kernels\"]\n\tpath = third_party/kernels\n")
	writeInventoryFixture(t, root, "src/main.py", "def main(): pass\n")
	writeInventoryFixture(t, root, "third_party/kernels/csrc/gemm.cu", "__global__ void gemm() {}\n")

	_, err := BuildInventoryMap(root, InventoryMapOptions{Repository: "o/up2", IndexedRevision: "abc", ExcludedSubmodules: map[string]string{
		"third_party/kernels": "too large to fetch",
	}})
	if !errors.Is(err, ErrExcludedSubmodulePopulated) {
		t.Fatalf("err = %v, want %v", err, ErrExcludedSubmodulePopulated)
	}
}

func TestExcludedSubmoduleNeverReadsAsKernelFree(t *testing.T) {
	root := t.TempDir()
	writeInventoryFixture(t, root, ".gitmodules", "[submodule \"kernels\"]\n\tpath = third_party/kernels\n")
	writeInventoryFixture(t, root, "src/main.py", "def main(): pass\n")
	if err := os.MkdirAll(filepath.Join(root, "third_party", "kernels"), 0o755); err != nil {
		t.Fatal(err)
	}
	generated := buildInventoryMapForTest(t, root, InventoryMapOptions{ExcludedSubmodules: map[string]string{
		"third_party/kernels": "too large to fetch",
	}})
	if generated.Totals.KernelFiles != 0 || len(generated.UnpopulatedSubmodules) != 0 {
		t.Fatalf("generated = %+v, want no walked kernels and the exclusion recorded", generated)
	}
	assertClassStatus(t, generated, KernelInventorySourceClass, InventoryClassExternalRequired)

	inventoryMap := validInventoryMapForTest()
	inventoryMap.Submodules = generated.Submodules
	inventoryMap.ExcludedSubmodules = generated.ExcludedSubmodules
	for _, row := range generated.SourceClasses {
		if row.Class == KernelInventorySourceClass {
			inventoryMap.SourceClasses = append(inventoryMap.SourceClasses, row)
		}
	}
	withoutInventory := buildKernelReportForTest(t, inventoryMap, nil)
	if withoutInventory.OK || !containsString(withoutInventory.Repositories[0].MissingSourceClasses, KernelInventorySourceClass) {
		t.Fatalf("report = %+v, want an excluded submodule to need kernel_inventory", withoutInventory.Repositories[0])
	}
	withInventory := buildKernelReportForTest(t, inventoryMap, &InventorySourceEvidence{
		Class: KernelInventorySourceClass, Evidence: []string{"docs/research/o-up2/kernel-inventory.json"},
	})
	if !withInventory.OK {
		t.Fatalf("report = %+v, want recorded kernel inventory evidence to satisfy the row", withInventory.Repositories[0])
	}
}

func TestBuildInventoryMapKernelMarkersIgnoreCommentsAndShims(t *testing.T) {
	root := t.TempDir()
	writeInventoryFixture(t, root, "ops/comment.py", "# dispatch through pallas_call on TPU\ndef host(): pass\n")
	writeInventoryFixture(t, root, "include/shim.h", "#define __global__\n")
	writeInventoryFixture(t, root, "include/guard.h", "#ifndef __global__\n#  define __global__\n#endif\n")
	writeInventoryFixture(t, root, "include/doc.hpp", "// launch a __global__ kernel\n/* __global__ */\n * __global__ in block\nint f();\n")
	writeInventoryFixture(t, root, "kernels/mm.mojo", "from gpu import thread_idx\nfn mm(): pass\n")
	writeInventoryFixture(t, root, "kernels/plain.mojo", "# from gpu import nothing\nfn main(): pass\n")
	writeInventoryFixture(t, root, "src/k.rs", "#[cube(launch)]\nfn k() {}\n")
	writeInventoryFixture(t, root, "src/plain.rs", "// #[cube] is the CubeCL attribute\nfn f() {}\n")
	writeInventoryFixture(t, root, "ops/long.py", strings.Repeat("x", 70000)+"\n@triton.jit\ndef k(): pass\n")

	report := buildInventoryMapForTest(t, root, InventoryMapOptions{})
	got := classEvidence(t, report, KernelInventorySourceClass)
	want := []string{"kernels/mm.mojo", "ops/long.py", "src/k.rs"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("kernel evidence = %v, want %v", got, want)
	}
	if report.Totals.KernelFiles != 3 {
		t.Fatalf("kernel files = %d, want 3", report.Totals.KernelFiles)
	}
}

func TestInventoryReportRequiresKernelInventoryWhenMapCountsKernels(t *testing.T) {
	walked := validInventoryMapForTest()
	walked.Totals.KernelFiles = 1
	walked.Subsystems[0].KernelFiles = 1
	walked.SourceClasses = append(walked.SourceClasses, InventoryClassStatus{
		Class: KernelInventorySourceClass, Status: InventoryClassExternalRequired, Evidence: []string{"cmd/kernel.cu"},
	})
	skipped := validInventoryMapForTest()
	skipped.SkippedKernelFiles = 2
	skipped.SourceClasses = append(skipped.SourceClasses, InventoryClassStatus{
		Class: KernelInventorySourceClass, Status: InventoryClassExternalRequired, Evidence: []string{"vendor/dep/dep.cu"},
	})

	for name, kernelMap := range map[string]InventoryMap{"walked": walked, "skipped": skipped} {
		t.Run(name, func(t *testing.T) {
			withoutInventory := buildKernelReportForTest(t, kernelMap, nil)
			if withoutInventory.OK || !containsString(withoutInventory.Repositories[0].MissingSourceClasses, KernelInventorySourceClass) {
				t.Fatalf("report = %+v, want kernel_inventory missing", withoutInventory.Repositories[0])
			}
			withInventory := buildKernelReportForTest(t, kernelMap, &InventorySourceEvidence{
				Class: KernelInventorySourceClass, Evidence: []string{"docs/research/owner-repo/kernel-inventory.json"},
			})
			if !withInventory.OK {
				t.Fatalf("report = %+v, want kernel inventory evidence to satisfy the row", withInventory.Repositories[0])
			}
		})
	}
}

func TestInventoryReportKernelFreeMapNeedsNoKernelInventory(t *testing.T) {
	report := buildKernelReportForTest(t, validInventoryMapForTest(), nil)
	if !report.OK {
		t.Fatalf("report = %+v, want a kernel-free map without a kernel_inventory row to stay ready", report.Repositories[0])
	}
}

func TestInventoryReportRefusesMapsThatHideKernelSurface(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*InventoryMap)
	}{
		{name: "kernels counted but class checked absent", mutate: func(m *InventoryMap) {
			m.Totals.KernelFiles = 1
			m.Subsystems[0].KernelFiles = 1
			m.SourceClasses = append(m.SourceClasses, InventoryClassStatus{Class: KernelInventorySourceClass, Status: InventoryClassCheckedAbsent})
		}},
		{name: "kernels counted without class status", mutate: func(m *InventoryMap) {
			m.Totals.KernelFiles = 1
			m.Subsystems[0].KernelFiles = 1
		}},
		{name: "skipped kernels but class checked absent", mutate: func(m *InventoryMap) {
			m.SkippedKernelFiles = 1
			m.SourceClasses = append(m.SourceClasses, InventoryClassStatus{Class: KernelInventorySourceClass, Status: InventoryClassCheckedAbsent})
		}},
		{name: "declared submodule not checked out", mutate: func(m *InventoryMap) {
			m.UnpopulatedSubmodules = []string{"vendor/Kernels"}
		}},
		{name: "excluded submodule but class checked absent", mutate: func(m *InventoryMap) {
			m.ExcludedSubmodules = []InventoryExcludedSubmodule{{Path: "vendor/Kernels", Reason: "SSH-only URL"}}
			m.SourceClasses = append(m.SourceClasses, InventoryClassStatus{Class: KernelInventorySourceClass, Status: InventoryClassCheckedAbsent})
		}},
		{name: "excluded submodule without class status", mutate: func(m *InventoryMap) {
			m.ExcludedSubmodules = []InventoryExcludedSubmodule{{Path: "vendor/Kernels", Reason: "SSH-only URL"}}
		}},
		{name: "truncated scan but class checked absent", mutate: func(m *InventoryMap) {
			m.SkippedKernelScanTruncated = true
			m.SourceClasses = append(m.SourceClasses, InventoryClassStatus{Class: KernelInventorySourceClass, Status: InventoryClassCheckedAbsent})
		}},
		{name: "excluded submodule without reason", mutate: func(m *InventoryMap) {
			m.ExcludedSubmodules = []InventoryExcludedSubmodule{{Path: "vendor/Kernels"}}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inventoryMap := validInventoryMapForTest()
			tt.mutate(&inventoryMap)
			report := buildKernelReportForTest(t, inventoryMap, &InventorySourceEvidence{
				Class: KernelInventorySourceClass, Evidence: []string{"docs/research/owner-repo/kernel-inventory.json"},
			})
			if report.OK || report.Repositories[0].Ready {
				t.Fatalf("report = %+v, want the hidden kernel surface to block readiness", report.Repositories[0])
			}
		})
	}
}

func TestKernelInventoryEvidenceCarriesLayoutObligationFields(t *testing.T) {
	// One populated example: a study of an NVFP4 GEMM family recording the
	// layout obligations its kernels expect at load time.
	populated := InventorySourceEvidence{
		Class:               KernelInventorySourceClass,
		Evidence:            []string{"docs/research/o-up2/kernel-inventory.json"},
		ExpectedInputLayout: "tile-swizzled",
		SwizzlePattern:      "nvfp4-scale-128x4",
		ScaleLayout:         "128x4-tiled",
		PackingSource:       "loader-native",
	}
	data, err := json.Marshal(populated)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"expected_input_layout":"tile-swizzled"`, `"swizzle_pattern":"nvfp4-scale-128x4"`, `"scale_layout":"128x4-tiled"`, `"packing_source":"loader-native"`} {
		if !strings.Contains(string(data), want) {
			t.Fatalf("wire form %s missing %s", data, want)
		}
	}
	var decoded InventorySourceEvidence
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded, populated) {
		t.Fatalf("round trip = %+v, want %+v", decoded, populated)
	}
	if got := decoded.UnknownLayoutValues(); len(got) != 0 {
		t.Fatalf("starter values flagged unknown = %v", got)
	}

	// Backward compatibility: an entry written before the fields exist still
	// validates, reads back unknown, and serializes byte-identically.
	legacy := `{"class":"kernel_inventory","evidence":["docs/research/o-up2/kernel-inventory.json"]}`
	var legacyDecoded InventorySourceEvidence
	if err := json.Unmarshal([]byte(legacy), &legacyDecoded); err != nil {
		t.Fatal(err)
	}
	if legacyDecoded.HasLayoutObligation() {
		t.Fatalf("legacy entry %+v reads back with layout obligations", legacyDecoded)
	}
	legacyBytes, err := json.Marshal(legacyDecoded)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(legacyBytes, []byte(legacy)) {
		t.Fatalf("legacy wire form = %s, want unchanged %s", legacyBytes, legacy)
	}

	// Reader support: a novel value is accepted, not refused, and named.
	novel := populated
	novel.SwizzlePattern = "flash-attention-3-paged"
	if got := novel.UnknownLayoutValues(); !reflect.DeepEqual(got, []string{"swizzle_pattern"}) {
		t.Fatalf("novel values = %v, want [swizzle_pattern]", got)
	}

	// The report flow keeps the populated record intact; the registry accepts
	// it on the kernel_inventory class and refuses it anywhere else.
	report := buildKernelReportForTest(t, validInventoryMapForTest(), &populated)
	row := report.Repositories[0]
	if !row.Ready {
		t.Fatalf("report = %+v, want a populated kernel_inventory record to stay ready", row)
	}
	found := false
	for _, evidence := range row.SourceEvidence {
		if evidence.Class == KernelInventorySourceClass && evidence.PackingSource == "loader-native" {
			found = true
		}
	}
	if !found {
		t.Fatalf("row source evidence = %+v, want the populated layout obligations carried through", row.SourceEvidence)
	}

	contract := completeInventoryContractForTest()
	contract.SourceEvidence = append(contract.SourceEvidence, InventorySourceEvidence{
		Class: "readme_docs", Evidence: []string{"README.md"}, PackingSource: "loader-native",
	})
	registry := registryWithInventoryForTest(contract)
	if err := registry.Validate(); err == nil || !strings.Contains(err.Error(), "layout obligation fields belong to the kernel_inventory record") {
		t.Fatalf("Validate() error = %v, want layout fields refused off the kernel_inventory record", err)
	}
}

// TestInventoryCheckCatchesValueHeadTileDeinterleave_LML07 is the layout-ledger
// flip witness for row LML-07: a kernel_inventory record that obligates the
// Qwen3.5-family ssm_out value-head-tiled input columns (lessons PHL-045 and
// PHL-127; fix 36d3e634c3 keeps the GGUF bytes and tiles the GDN core instead
// of deinterleaving into a Q8_0 copy) must be refused when the declared loader
// packing path feeds those columns through the Q8_0 deinterleave, and accepted
// when the packing path keeps the tiled order. Assertions pin the sentinels
// with errors.Is, never error prose.
func TestInventoryCheckCatchesValueHeadTileDeinterleave_LML07(t *testing.T) {
	ssmOutRecord := InventorySourceEvidence{
		Class:               KernelInventorySourceClass,
		Evidence:            []string{"docs/research/o-up2/kernel-inventory.json"},
		ExpectedInputLayout: "value-head-tiled",
		SwizzlePattern:      "none",
		ScaleLayout:         "linear",
		PackingSource:       "loader-native",
	}
	q8Deinterleave := PackingPathFacts{
		ExpectedInputLayout: "grouped", // Q8_0 copy whose columns were deinterleaved into grouped order
		SwizzlePattern:      "none",
		ScaleLayout:         "linear",
		PackingSource:       "runtime-repack",
	}
	nativeTiledRoute := PackingPathFacts{
		ExpectedInputLayout: "value-head-tiled", // GGUF bytes kept; the core is tiled before the projection
		SwizzlePattern:      "none",
		ScaleLayout:         "linear",
		PackingSource:       "loader-native",
	}

	tests := []struct {
		name    string
		rec     InventorySourceEvidence
		packing PackingPathFacts
		want    error
	}{
		{name: "LML07 value-head-tiled ssm_out vs Q8_0 deinterleave packing", rec: ssmOutRecord, packing: q8Deinterleave, want: ErrLayoutMismatch},
		{name: "LML07 value-head-tiled ssm_out vs native tiled packing", rec: ssmOutRecord, packing: nativeTiledRoute, want: nil},
		{name: "novel record value refuses as unknown even when the packing side repeats it", rec: InventorySourceEvidence{
			Class:          KernelInventorySourceClass,
			Evidence:       []string{"docs/research/o-up2/kernel-inventory.json"},
			SwizzlePattern: "flash-attention-3-paged",
		}, packing: PackingPathFacts{SwizzlePattern: "flash-attention-3-paged"}, want: ErrLayoutUnknown},
		{name: "obligated axis undeclared by the packing path refuses as unknown", rec: ssmOutRecord, packing: PackingPathFacts{}, want: ErrLayoutUnknown},
		{name: "whitespace-only obligations carry nothing", rec: InventorySourceEvidence{
			Class:               KernelInventorySourceClass,
			Evidence:            []string{"docs/research/o-up2/kernel-inventory.json"},
			ExpectedInputLayout: "  ",
			PackingSource:       "\t",
		}, packing: PackingPathFacts{}, want: nil},
		{name: "legacy record without obligations passes", rec: InventorySourceEvidence{
			Class:    KernelInventorySourceClass,
			Evidence: []string{"docs/research/o-up2/kernel-inventory.json"},
		}, packing: PackingPathFacts{}, want: nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := CheckLoaderPackingAgainstInventory(tt.rec, tt.packing); !errors.Is(err, tt.want) {
				t.Fatalf("CheckLoaderPackingAgainstInventory() error = %v, want sentinel %v", err, tt.want)
			}
		})
	}
}

// TestInventoryVocabularyNamesTheSSMOutOrderPair verifies the LML-07 class is
// representable inside the closed vocabularies: both the GGUF value-head-tiled
// order and the deinterleaved grouped order are known expected_input_layout
// values, so a record naming either reads back without an unknown flag.
func TestInventoryVocabularyNamesTheSSMOutOrderPair(t *testing.T) {
	for _, value := range []string{"value-head-tiled", "grouped"} {
		if !slices.Contains(KnownExpectedInputLayouts, value) {
			t.Fatalf("KnownExpectedInputLayouts = %v, want %q named", KnownExpectedInputLayouts, value)
		}
	}
	rec := InventorySourceEvidence{
		Class:               KernelInventorySourceClass,
		Evidence:            []string{"docs/research/o-up2/kernel-inventory.json"},
		ExpectedInputLayout: "value-head-tiled",
	}
	if got := rec.UnknownLayoutValues(); len(got) != 0 {
		t.Fatalf("UnknownLayoutValues() = %v, want the value-head-tiled order to read back known", got)
	}
}

func buildInventoryMapForTest(t *testing.T, root string, opts InventoryMapOptions) InventoryMap {
	t.Helper()
	opts.Repository = "owner/repo"
	opts.IndexedRevision = "abc"
	report, err := BuildInventoryMap(root, opts)
	if err != nil {
		t.Fatal(err)
	}
	return report
}

func assertTotalsMatchSubsystems(t *testing.T, report InventoryMap) {
	t.Helper()
	var sum InventoryMapTotals
	for _, subsystem := range report.Subsystems {
		sum.Files += subsystem.Files
		sum.Directories += subsystem.Directories
		sum.Bytes += subsystem.Bytes
		sum.RuntimeFiles += subsystem.RuntimeFiles
		sum.TestFiles += subsystem.TestFiles
		sum.DocsFiles += subsystem.DocsFiles
		sum.TextLines += subsystem.TextLines
		sum.KernelFiles += subsystem.KernelFiles
	}
	if sum != report.Totals {
		t.Fatalf("totals = %+v, subsystem sum = %+v", report.Totals, sum)
	}
}

func buildKernelReportForTest(t *testing.T, inventoryMap InventoryMap, kernelEvidence *InventorySourceEvidence) InventoryReport {
	t.Helper()
	dir := t.TempDir()
	mapPath := filepath.Join(dir, "owner-repo.json")
	var mapBytes bytes.Buffer
	if err := WriteInventoryMapJSON(&mapBytes, inventoryMap); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(mapPath, mapBytes.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	contract := completeInventoryContractForTest()
	contract.MapPath = mapPath
	contract.SourceEvidence = append(contract.SourceEvidence, InventorySourceEvidence{
		Class: "hardware_reproduction", Evidence: []string{"docs/research/HARDWARE-EXPERIMENTS.md#owner-repo"},
	})
	if kernelEvidence != nil {
		contract.SourceEvidence = append(contract.SourceEvidence, *kernelEvidence)
	}
	registry := registryWithInventoryForTest(contract)
	if err := registry.Validate(); err != nil {
		t.Fatalf("registry validate: %v", err)
	}
	return BuildInventoryReportWithMapFiles(registry, ".")
}

func classNote(t *testing.T, report InventoryMap, class string) string {
	t.Helper()
	for _, row := range report.SourceClasses {
		if strings.TrimSpace(row.Class) == class {
			return row.Note
		}
	}
	t.Fatalf("class %s missing from %+v", class, report.SourceClasses)
	return ""
}

func classEvidence(t *testing.T, report InventoryMap, class string) []string {
	t.Helper()
	for _, row := range report.SourceClasses {
		if strings.TrimSpace(row.Class) == class {
			return row.Evidence
		}
	}
	t.Fatalf("class %s missing from %+v", class, report.SourceClasses)
	return nil
}

func completeInventoryContractForTest() *InventoryContract {
	return &InventoryContract{
		IndexedRevision: "abc",
		SourceClasses:   append([]string(nil), RequiredInventorySourceClasses...),
		SourceEvidence: []InventorySourceEvidence{
			{Class: "open_closed_issues_prs_discussions", Evidence: []string{"gh issue list --state all --repo owner/repo", "gh pr list --state all --repo owner/repo", "gh api graphql repository discussions"}},
			{Class: "fak_selfquery_witness", Evidence: []string{"fak capabilities speculator"}},
			{Class: "candidate_matrix", Evidence: []string{"docs/research/STUDY-OWNER-REPO.md#candidate-matrix"}},
			{Class: "issue_tracking", Evidence: []string{"#123"}},
		},
		SubsystemCount:     1,
		CompletenessCritic: "all local and non-tree classes complete",
	}
}

func registryWithInventoryForTest(inventory *InventoryContract) Registry {
	return Registry{Schema: Schema, Methodology: "ranked", Repositories: []Repository{{
		Repository:      "owner/repo",
		URL:             "https://example/repo",
		Status:          "candidate",
		Priority:        1,
		Why:             "fresh",
		LastChecked:     "2026-08-14",
		CheckedRevision: "abc",
		Inventory:       inventory,
	}}}
}
