// Package cudaarch validates the repository's declared CUDA architecture matrix.
package cudaarch

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode"
)

// Validate checks that the CUDA build and documentation surfaces agree on the
// declared SASS set and compute_120 PTX floor.
func Validate(root string) ([]string, error) {
	archText, err := os.ReadFile(filepath.Join(root, "internal", "compute", "cuda_arch.txt"))
	if err != nil {
		return nil, err
	}
	arches, parseErrors := ParseArchitectures(string(archText))
	errors := make([]string, 0, len(parseErrors))
	errors = append(errors, parseErrors...)

	files := []struct {
		name    string
		path    []string
		needles []string
	}{
		{"linux", []string{"internal", "compute", "build_cuda.sh"}, []string{"cuda_arch.txt", "code=${arch}", "code=compute_${PTX_CC}"}},
		{"windows", []string{"tools", "build_cuda_windows.ps1"}, []string{"cuda_arch.txt", "code=${item}", "code=compute_${cc}"}},
		{"docker", []string{"Dockerfile.cuda"}, []string{"cuda_arch.txt", "code=${arch}", "code=compute_${cc}"}},
		{"docs", []string{"docs", "cuda-dev.md"}, []string{"internal/compute/cuda_arch.txt", "cuda-build-sm100", "cuda-build-sm120"}},
	}
	for _, file := range files {
		text, readErr := os.ReadFile(filepath.Join(append([]string{root}, file.path...)...))
		if readErr != nil {
			return nil, readErr
		}
		errors = append(errors, ValidateSourceFile(file.name, string(text), file.needles)...)
	}
	if len(arches) > 0 && arches[len(arches)-1] != "sm_120" {
		errors = append(errors, fmt.Sprintf("PTX floor must follow highest declared arch sm_120, got %q", arches[len(arches)-1]))
	}
	return errors, nil
}

// ParseArchitectures parses and validates the architecture declarations from raw text (e.g. cuda_arch.txt).
// It returns the slice of non-empty architecture lines and any syntax or duplication errors encountered.
func ParseArchitectures(text string) ([]string, []string) {
	arches := nonEmptyLines(text)
	errors := make([]string, 0)
	if len(arches) == 0 || hasDuplicate(arches) {
		errors = append(errors, "cuda_arch.txt must contain a non-empty unique architecture list")
	}
	for _, arch := range arches {
		if !validArch(arch) {
			errors = append(errors, "cuda_arch.txt entries must use sm_<digits>")
			break
		}
	}
	return arches, errors
}

// ValidateSourceFile validates that a source file's contents conform to the required architecture contracts.
// It strips UTF-8 BOM if present and checks that all required needle strings are contained.
func ValidateSourceFile(name, contents string, needles []string) []string {
	clean := strings.TrimPrefix(contents, string(rune(0xFEFF)))
	var errors []string
	for _, needle := range needles {
		if !strings.Contains(clean, needle) {
			errors = append(errors, fmt.Sprintf("%s: missing arch-matrix contract %q", name, needle))
		}
	}
	return errors
}

func nonEmptyLines(text string) []string {
	var lines []string
	for _, line := range strings.Split(text, "\n") {
		if line = strings.TrimSpace(line); line != "" {
			lines = append(lines, line)
		}
	}
	return lines
}

func hasDuplicate(values []string) bool {
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if _, ok := seen[value]; ok {
			return true
		}
		seen[value] = struct{}{}
	}
	return false
}

func validArch(value string) bool {
	if !strings.HasPrefix(value, "sm_") || len(value) == 3 {
		return false
	}
	for _, r := range value[3:] {
		if !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}
