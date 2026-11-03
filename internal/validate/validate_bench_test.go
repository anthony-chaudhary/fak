package validate

import (
	"io"
	"testing"
)

// These benchmarks protect the validate lane's hot pure-CPU paths (flag
// parsing, private test-event decoding, GPU-path classification) so perf
// regressions show up in the debt-lanes reader's evidence instead of in
// operator latency. Pure CPU only: no subprocess exec, no network, no GPU.

// BenchmarkParseValidateFlags exercises the full flag-parse hot path with a
// realistic argument set (root, ref, json, timeout, two owned paths).
func BenchmarkParseValidateFlags(b *testing.B) {
	argv := []string{
		"--root", ".",
		"--ref", "main",
		"--json",
		"--timeout", "30m",
		"--mine", "internal/validate/validate.go",
		"--mine", "internal/validate/test_events.go",
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = parseValidateFlags(io.Discard, argv)
	}
}

// validateBenchEventStream mirrors the go test -json shape decodeValidateTestEvents
// consumes: run/output/pass/skip events across three packages, each package
// closed by its terminal event. ~50 lines.
const validateBenchEventStream = `{"Action":"run","Package":"p","Test":"TestAlpha"}
{"Action":"output","Package":"p","Test":"TestAlpha","Output":"alpha started\n"}
{"Action":"pass","Package":"p","Test":"TestAlpha","Elapsed":0.1}
{"Action":"run","Package":"p","Test":"TestBeta"}
{"Action":"output","Package":"p","Test":"TestBeta","Output":"beta detail\n"}
{"Action":"skip","Package":"p","Test":"TestBeta"}
{"Action":"run","Package":"p","Test":"TestGamma"}
{"Action":"run","Package":"p","Test":"TestGamma/Sub"}
{"Action":"pass","Package":"p","Test":"TestGamma/Sub","Elapsed":0.05}
{"Action":"pass","Package":"p","Test":"TestGamma","Elapsed":0.06}
{"Action":"run","Package":"p","Test":"TestDelta"}
{"Action":"pass","Package":"p","Test":"TestDelta","Elapsed":0.02}
{"Action":"run","Package":"p","Test":"TestEpsilon"}
{"Action":"pass","Package":"p","Test":"TestEpsilon","Elapsed":0.03}
{"Action":"run","Package":"p","Test":"TestZeta"}
{"Action":"pass","Package":"p","Test":"TestZeta","Elapsed":0.01}
{"Action":"pass","Package":"p"}
{"Action":"run","Package":"q","Test":"TestOne"}
{"Action":"output","Package":"q","Test":"TestOne","Output":"one log\n"}
{"Action":"pass","Package":"q","Test":"TestOne","Elapsed":0.11}
{"Action":"run","Package":"q","Test":"TestTwo"}
{"Action":"pass","Package":"q","Test":"TestTwo","Elapsed":0.04}
{"Action":"run","Package":"q","Test":"TestThree"}
{"Action":"output","Package":"q","Test":"TestThree","Output":"three log\n"}
{"Action":"skip","Package":"q","Test":"TestThree"}
{"Action":"run","Package":"q","Test":"TestFour"}
{"Action":"pass","Package":"q","Test":"TestFour","Elapsed":0.07}
{"Action":"run","Package":"q","Test":"TestFive"}
{"Action":"pass","Package":"q","Test":"TestFive","Elapsed":0.08}
{"Action":"run","Package":"q","Test":"TestSix"}
{"Action":"pass","Package":"q","Test":"TestSix","Elapsed":0.09}
{"Action":"pass","Package":"q"}
{"Action":"run","Package":"r","Test":"TestA"}
{"Action":"pass","Package":"r","Test":"TestA","Elapsed":0.01}
{"Action":"run","Package":"r","Test":"TestB"}
{"Action":"output","Package":"r","Test":"TestB","Output":"b log\n"}
{"Action":"pass","Package":"r","Test":"TestB","Elapsed":0.02}
{"Action":"run","Package":"r","Test":"TestC"}
{"Action":"pass","Package":"r","Test":"TestC","Elapsed":0.03}
{"Action":"run","Package":"r","Test":"TestD"}
{"Action":"skip","Package":"r","Test":"TestD"}
{"Action":"run","Package":"r","Test":"TestE"}
{"Action":"pass","Package":"r","Test":"TestE","Elapsed":0.04}
{"Action":"run","Package":"r","Test":"TestF"}
{"Action":"pass","Package":"r","Test":"TestF","Elapsed":0.05}
{"Action":"run","Package":"r","Test":"TestG"}
{"Action":"pass","Package":"r","Test":"TestG","Elapsed":0.06}
{"Action":"run","Package":"r","Test":"TestH"}
{"Action":"pass","Package":"r","Test":"TestH","Elapsed":0.07}
{"Action":"pass","Package":"r"}
`

// BenchmarkDecodeValidateTestEvents decodes a complete ~50-line private test
// event stream (built once outside the loop) on a fresh copy per iteration.
func BenchmarkDecodeValidateTestEvents(b *testing.B) {
	stream := []byte(validateBenchEventStream)
	packages := []string{"p", "q", "r"}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		snapshot := append([]byte(nil), stream...)
		_, _, _ = decodeValidateTestEvents(snapshot, packages)
	}
}

// BenchmarkIsGPURelatedValidation classifies fixed realistic repo-relative
// path slices: GPU-root hits, model-marker hits, docs misses, and CUDA-only
// misses, covering both the early-exit and full-walk arms.
func BenchmarkIsGPURelatedValidation(b *testing.B) {
	cases := []struct {
		name  string
		paths []string
	}{
		{"gpuRootFirst", []string{
			"internal/compute/gemm.cu",
			"cmd/fak/validate_acceptance.go",
		}},
		{"mixedFullWalk", []string{
			"docs/sota/README.md",
			"internal/policy/adjudicator.go",
			"internal/compute/cuda/cuda_device_context.cpp",
			"internal/model/llm.go",
			"platform/strix/halo.go",
			"internal/model/metal_backend.metal",
		}},
		{"allMiss", []string{
			"docs/sota/README.md",
			"internal/policy/adjudicator.go",
			"internal/compute/cuda/cuda_device_context.cpp",
			"internal/model/llm.go",
			"platform/strix/halo.go",
		}},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = isGPURelatedValidation(tc.paths)
			}
		})
	}
}
