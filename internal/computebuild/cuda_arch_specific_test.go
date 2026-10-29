package computebuild

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

func TestCUDAArchSpecificDeclarationAndArgs(t *testing.T) {
	p, err := parseCUDAArchSpecific("# lane\nsm_103a 13.0\nsm_100f 13.0\nsm_100a 13.0\n")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(p.arches, []string{"sm_100a", "sm_100f", "sm_103a"}) {
		t.Fatal(p.arches)
	}
	args := cudaArchSpecificCompileArgs(&Toolchain{CXX: "/cxx", CUDAInc: []string{"/cuda/include"}}, p, "/out/probe.o")
	want := []string{"-O3", "-std=c++17", "-gencode", "arch=compute_100a,code=sm_100a", "-gencode", "arch=compute_100f,code=sm_100f", "-gencode", "arch=compute_103a,code=sm_103a", "-ccbin", "/cxx", "-I/cuda/include", "-Xcompiler", "-fPIC", "-c", "cuda_sm10x_probe.cu", "-o", "/out/probe.o"}
	if !reflect.DeepEqual(args, want) {
		t.Fatalf("got %q want %q", args, want)
	}
	if strings.Contains(strings.Join(args, " "), "code=compute_") {
		t.Fatal("arch-specific lane must not emit fallback PTX")
	}
}

func TestCUDAArchSpecificDeclarationRefuses(t *testing.T) {
	valid := "sm_100a 13.0\nsm_103a 13.0\nsm_100f 13.0\n"
	for _, text := range []string{"", "sm_100a 13.0", valid + "sm_100a 13.0", strings.ReplaceAll(valid, "13.0", "12.9"), strings.ReplaceAll(valid, "sm_100f", "sm_120"), valid + "sm_103f 13.0"} {
		if _, err := parseCUDAArchSpecific(text); !errors.Is(err, ErrCUDAArchSpecific) {
			t.Fatalf("%q: %v", text, err)
		}
	}
}
