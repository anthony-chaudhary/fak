package cudaarch

import (
	"fmt"
	"strings"
	"testing"
)

var (
	benchSinkErrors []string
	benchSinkError  error
	benchSinkArches []string
)

// TestBenchmarkSanity verifies that all benchmarked production paths execute cleanly.
func TestBenchmarkSanity(t *testing.T) {
	root := fixture(t)
	errs, err := Validate(root)
	if err != nil {
		t.Fatalf("Validate failed: %v", err)
	}
	if len(errs) != 0 {
		t.Fatalf("unexpected validation errors: %v", errs)
	}

	arches, parseErrs := ParseArchitectures("sm_80\nsm_90\nsm_120\n")
	if len(parseErrs) != 0 || len(arches) != 3 {
		t.Fatalf("ParseArchitectures sanity failed: %v, %v", arches, parseErrs)
	}

	srcErrs := ValidateSourceFile("linux", "cuda_arch.txt code=${arch}", []string{"cuda_arch.txt", "code=${arch}"})
	if len(srcErrs) != 0 {
		t.Fatalf("ValidateSourceFile sanity failed: %v", srcErrs)
	}
}

// BenchmarkCurrentTreeIsValid measures full workspace matrix tree validation throughput.
func BenchmarkCurrentTreeIsValid(b *testing.B) {
	dstRoot := fixture(b)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchSinkErrors, benchSinkError = Validate(dstRoot)
		if benchSinkError != nil {
			b.Fatal(benchSinkError)
		}
		if len(benchSinkErrors) != 0 {
			b.Fatalf("unexpected validation errors: %v", benchSinkErrors)
		}
	}
}

// BenchmarkValidateSourceFile measures source contract validation across Linux, Windows, and Docs surfaces.
func BenchmarkValidateSourceFile(b *testing.B) {
	cases := []struct {
		name     string
		contents string
		needles  []string
	}{
		{
			name: "LinuxScript",
			contents: `#!/usr/bin/env bash
set -euo pipefail
ARCH_LIST="internal/compute/cuda_arch.txt"
while IFS= read -r arch; do
    GENCODE+=(-gencode "arch=${arch},code=${arch}")
done < "$ARCH_LIST"
GENCODE+=(-gencode "arch=compute_${PTX_CC},code=compute_${PTX_CC}")
nvcc "${GENCODE[@]}" -o kernels.o -c kernels.cu
`,
			needles: []string{"cuda_arch.txt", "code=${arch}", "code=compute_${PTX_CC}"},
		},
		{
			name: "WindowsScriptBOM",
			contents: "\xef\xbb\xbf" + `# PowerShell build script
$archList = "internal/compute/cuda_arch.txt"
$items = Get-Content $archList
foreach ($item in $items) {
    $gencode += "code=${item}"
}
$gencode += "code=compute_${cc}"
`,
			needles: []string{"cuda_arch.txt", "code=${item}", "code=compute_${cc}"},
		},
		{
			name: "DocsMarkdown",
			contents: `# CUDA Architecture Guide
The build sources internal/compute/cuda_arch.txt for all supported architectures.
Targets:
- cuda-build-sm100
- cuda-build-sm120
`,
			needles: []string{"internal/compute/cuda_arch.txt", "cuda-build-sm100", "cuda-build-sm120"},
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.SetBytes(int64(len(tc.contents)))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkErrors = ValidateSourceFile(tc.name, tc.contents, tc.needles)
				if len(benchSinkErrors) != 0 {
					b.Fatalf("unexpected errors: %v", benchSinkErrors)
				}
			}
		})
	}
}

// BenchmarkParseArchitectures measures parsing and validation of CUDA architecture text files.
func BenchmarkParseArchitectures(b *testing.B) {
	b.Run("StandardMatrix", func(b *testing.B) {
		archText := "sm_80\nsm_89\nsm_90\nsm_100\nsm_120\n"
		b.SetBytes(int64(len(archText)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkArches, benchSinkErrors = ParseArchitectures(archText)
			if len(benchSinkErrors) != 0 || len(benchSinkArches) != 5 {
				b.Fatalf("unexpected parse result: %v, %v", benchSinkArches, benchSinkErrors)
			}
		}
	})

	b.Run("ScaledMatrix", func(b *testing.B) {
		var buf strings.Builder
		for i := 10; i <= 150; i++ {
			fmt.Fprintf(&buf, "sm_%d\n", i)
		}
		archText := buf.String()
		b.SetBytes(int64(len(archText)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkArches, benchSinkErrors = ParseArchitectures(archText)
			if len(benchSinkErrors) != 0 || len(benchSinkArches) == 0 {
				b.Fatalf("unexpected parse result: %v, %v", benchSinkArches, benchSinkErrors)
			}
		}
	})
}
