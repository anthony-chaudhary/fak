package compute

import (
	"go/build/constraint"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// The published CUDA image is probed (version, runtime-capabilities) on hosts with no GPU
// driver, where libcuda.so.1 does not exist: the container toolkit injects it only when a
// device is attached. A driver-API reference in the default cuda build turns that library
// into a load-time dependency and the binary stops starting there at all.
//
// Source contract only: this does not link or load a native binary.
// fak-test:runtime fast est=20ms lane=default
func TestDefaultCUDABuildDoesNotLinkTheDriverLibrary(t *testing.T) {
	defaultTags := map[string]bool{"cuda": true, "cgo": true, "linux": true, "amd64": true}
	goFiles, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	checkedRoot := false
	for _, name := range goFiles {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		raw, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		source := string(raw)
		if !cgoLinksCUDADriver(source) {
			checkedRoot = checkedRoot || name == "cuda.go"
			continue
		}
		if goFileBuiltWith(t, name, source, defaultTags) {
			t.Errorf("%s links -lcuda in the default cuda build; query through libcudart or gate the file behind an opt-in tag", name)
		}
	}
	if !checkedRoot {
		t.Fatal("cuda.go must stay free of the driver library link")
	}

	native := []string{"cuda_kernels.cu"}
	fragments, err := filepath.Glob(filepath.Join("cuda", "*.cuh"))
	if err != nil {
		t.Fatal(err)
	}
	if len(fragments) == 0 {
		t.Fatal("no default CUDA kernel fragments found")
	}
	native = append(native, fragments...)
	driverCall := regexp.MustCompile(`\bcu[A-Z][a-z][A-Za-z0-9_]*\s*\(`)
	for _, name := range native {
		raw, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		for i, line := range strings.Split(string(raw), "\n") {
			code, _, _ := strings.Cut(line, "//")
			if strings.Contains(code, "#include <cuda.h>") || driverCall.MatchString(code) {
				t.Errorf("%s:%d uses the CUDA driver API in the default kernel archive: %s", name, i+1, strings.TrimSpace(line))
			}
		}
	}
}

func cgoLinksCUDADriver(source string) bool {
	for _, line := range strings.Split(source, "\n") {
		directive, flags, ok := strings.Cut(line, "LDFLAGS:")
		if !ok || !strings.HasPrefix(strings.TrimSpace(directive), "#cgo") {
			continue
		}
		for _, flag := range strings.Fields(flags) {
			if flag == "-lcuda" {
				return true
			}
		}
	}
	return false
}

func goFileBuiltWith(t *testing.T, name, source string, tags map[string]bool) bool {
	t.Helper()
	for _, line := range strings.Split(source, "\n") {
		if strings.HasPrefix(line, "package ") {
			break
		}
		if !constraint.IsGoBuild(line) {
			continue
		}
		expr, err := constraint.Parse(line)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		return expr.Eval(func(tag string) bool { return tags[tag] })
	}
	return true
}
