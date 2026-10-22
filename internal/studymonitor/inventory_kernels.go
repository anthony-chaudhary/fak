package studymonitor

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// KernelInventorySourceClass is required only when the walked tree holds GPU
// kernel sources: a study of such a repository is not exhaustive until every
// kernel is listed with its backend mapping.
const KernelInventorySourceClass = "kernel_inventory"

const (
	defaultSkippedKernelScanBudget = 50000
	maxSkippedKernelExamples       = 12
)

func isKernelLanguage(language string) bool {
	switch language {
	case "CUDA", "HIP", "GLSL", "Metal", "OpenCL", "WGSL", "PTX", "Slang":
		return true
	default:
		return false
	}
}

var kernelFileExtensions = map[string]bool{
	".cu": true, ".cuh": true, ".hip": true, ".comp": true, ".glsl": true, ".metal": true,
	".cl": true, ".wgsl": true, ".ptx": true, ".slang": true, ".mojo": true,
}

func hasKernelFileExtension(rel string) bool {
	return kernelFileExtensions[strings.ToLower(path.Ext(rel))]
}

// kernelMarkerRule names the content markers that make a runtime source file a
// GPU kernel when its extension alone cannot. Comment-only lines never count,
// and for C/C++ neither do preprocessor lines (a `#define __global__` shim is
// not a kernel).
type kernelMarkerRule struct {
	markers          []string
	commentPrefixes  []string
	skipPreprocessor bool
}

var (
	pythonKernelRule = &kernelMarkerRule{
		markers:         []string{"@triton.jit", "T.prim_func", "tilelang.jit", "@cute.kernel", "@cuda.jit", "pallas_call"},
		commentPrefixes: []string{"#"},
	}
	cppKernelRule = &kernelMarkerRule{
		markers:          []string{"__global__"},
		commentPrefixes:  []string{"//", "/*", "*"},
		skipPreprocessor: true,
	}
	mojoKernelRule = &kernelMarkerRule{
		markers:         []string{"from gpu", "import gpu", "enqueue_function"},
		commentPrefixes: []string{"#"},
	}
	rustKernelRule = &kernelMarkerRule{
		markers:         []string{"#[cube", "#[kernel]", "#[spirv("},
		commentPrefixes: []string{"//", "/*", "*"},
	}
)

func kernelMarkerRuleFor(file inventoryFileClass) *kernelMarkerRule {
	if !file.Runtime || file.Kernel {
		return nil
	}
	switch file.Language {
	case "Python":
		return pythonKernelRule
	case "C", "C++":
		return cppKernelRule
	case "Mojo":
		return mojoKernelRule
	case "Rust":
		return rustKernelRule
	default:
		return nil
	}
}

func (rule *kernelMarkerRule) matches(line string) bool {
	trimmed := strings.TrimSpace(line)
	for _, prefix := range rule.commentPrefixes {
		if strings.HasPrefix(trimmed, prefix) {
			return false
		}
	}
	if rule.skipPreprocessor && strings.HasPrefix(trimmed, "#") {
		return false
	}
	for _, marker := range rule.markers {
		if strings.Contains(line, marker) {
			return true
		}
	}
	return false
}

// skippedKernelScan is the bounded, extension-only look into skipped
// vendor/build/dist/target directories: kernels copied there (not as
// submodules) must still open kernel_inventory. Its counts are reported beside
// the totals, never inside them.
type skippedKernelScan struct {
	files     int
	examples  []string
	truncated bool
	limit     int
	budget    int
	excluded  map[string]string
}

func newSkippedKernelScan(limit int, excluded map[string]string) *skippedKernelScan {
	if limit <= 0 {
		limit = defaultSkippedKernelScanBudget
	}
	return &skippedKernelScan{limit: limit, budget: limit, excluded: excluded}
}

func (scan *skippedKernelScan) note(rel string) {
	if !hasKernelFileExtension(rel) {
		return
	}
	scan.files++
	if len(scan.examples) < maxSkippedKernelExamples {
		scan.examples = append(scan.examples, rel)
	}
}

func (scan *skippedKernelScan) scanDir(root, rel string) {
	stack := []string{rel}
	for len(stack) > 0 && !scan.truncated {
		dir := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		entries, err := os.ReadDir(filepath.Join(root, filepath.FromSlash(dir)))
		if err != nil && len(entries) == 0 {
			continue
		}
		var subdirs []string
		for _, entry := range entries {
			if scan.budget <= 0 {
				scan.truncated = true
				return
			}
			scan.budget--
			child := dir + "/" + entry.Name()
			switch {
			case entry.IsDir():
				if _, skip := scan.excluded[child]; !skip && !skippedKernelScanDenied(entry.Name()) {
					subdirs = append(subdirs, child)
				}
			case entry.Type().IsRegular():
				scan.note(child)
			}
		}
		for i := len(subdirs) - 1; i >= 0; i-- {
			stack = append(stack, subdirs[i])
		}
	}
}

// skippedKernelScanEligible admits a skipped location under a vendor, build,
// dist, or target directory, and never one under a package-manager tree,
// version-control dir, virtualenv, or cache.
func skippedKernelScanEligible(rel string) bool {
	eligible := false
	for _, part := range strings.Split(rel, "/") {
		if skippedKernelScanDenied(part) {
			return false
		}
		switch strings.ToLower(part) {
		case "vendor", "build", "dist", "target":
			eligible = true
		}
	}
	return eligible
}

func skippedKernelScanDenied(name string) bool {
	switch strings.ToLower(name) {
	case ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "site-packages", "__pycache__",
		".next", ".cache", ".turbo", ".pytest_cache", "coverage":
		return true
	default:
		return false
	}
}

// KernelInventoryRequired reports whether the map leaves GPU kernels possible:
// kernels were counted (walked, or by the skipped-directory extension scan), a
// declared submodule was excluded or is not checked out, or the scan stopped at
// its budget. Source the walk never saw is unproven, never checked absent.
func (report InventoryMap) KernelInventoryRequired() bool {
	return report.KernelSurfaceFiles() > 0 || len(report.ExcludedSubmodules) > 0 ||
		len(report.UnpopulatedSubmodules) > 0 || report.SkippedKernelScanTruncated
}

// kernelInventoryAbsentNote is the completeness-note clause for a map whose
// walk and skipped-directory scan proved no GPU kernel source. The map then
// carries no kernel_inventory class row, so readers that predate the class
// keep accepting it.
const kernelInventoryAbsentNote = "no GPU kernel source (CUDA, HIP, GLSL, Metal, OpenCL, WGSL, PTX, Slang, Mojo, or Triton/TileLang/CuTe/Pallas/CubeCL kernels) found in the walked tree or by the extension scan of skipped vendor/build/dist/target directories, so kernel_inventory is not required"

func (acc inventoryAccumulator) kernelInventoryClassStatus(report InventoryMap, scanLimit int) InventoryClassStatus {
	kernels := sortedMapKeys(acc.kernelFiles)
	parts := []string{fmt.Sprintf("%d GPU-kernel files in the walked tree", len(kernels))}
	if len(kernels) > 0 {
		parts[0] += fmt.Sprintf(" (%s)", renderLanguages(renderLanguageCounts(acc.kernelLanguages)))
	}
	parts = append(parts, fmt.Sprintf("%d kernel-extension files in skipped directories", report.SkippedKernelFiles))
	if report.SkippedKernelScanTruncated {
		parts = append(parts, fmt.Sprintf("skipped-directory scan truncated at its %d-entry budget, so absence there is unproven", scanLimit))
	}
	if len(report.ExcludedSubmodules) > 0 {
		excluded := make([]string, 0, len(report.ExcludedSubmodules))
		for _, row := range report.ExcludedSubmodules {
			excluded = append(excluded, row.Path)
		}
		parts = append(parts, "excluded submodules never walked or scanned: "+strings.Join(excluded, ", "))
	}
	if len(report.UnpopulatedSubmodules) > 0 {
		parts = append(parts, "submodules not checked out: "+strings.Join(report.UnpopulatedSubmodules, ", "))
	}
	evidence := append(append([]string(nil), kernels...), report.SkippedKernelExamples...)
	return InventoryClassStatus{
		Class:    KernelInventorySourceClass,
		Status:   InventoryClassExternalRequired,
		Evidence: capStrings(evidence, 12),
		Note: strings.Join(parts, ", ") +
			"; add source_evidence pointing at a per-kernel inventory: every kernel with source path:line, arch features, caller, and fak CUDA/Vulkan/ROCm mapping, and for each unseen submodule its kernels from upstream source or why it holds none",
	}
}
