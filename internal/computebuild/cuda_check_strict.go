package computebuild

import (
	"context"
	"errors"
	"os"
	"os/exec"
)

type cudaStrictCheckKey struct{}

// WithStrictCUDACheck makes the GPU-free check verb fail closed on a missing Python
// interpreter or parity tool. Plain build-cuda check never sets it and stays fail-open.
func WithStrictCUDACheck(ctx context.Context) context.Context {
	return context.WithValue(ctx, cudaStrictCheckKey{}, true)
}

// IsStrictCUDACheck reports whether ctx was marked by WithStrictCUDACheck.
func IsStrictCUDACheck(ctx context.Context) bool {
	strict, _ := ctx.Value(cudaStrictCheckKey{}).(bool)
	return strict
}

// cudaCheckPython probes PATH for python3 then python. The strict route also honors PYTHON first,
// so a pinned interpreter wins over ambient python3; plain check keeps its PATH-only probe.
func cudaCheckPython(strict bool) (string, error) {
	if py := os.Getenv("PYTHON"); strict && py != "" {
		return py, nil
	}
	for _, name := range []string{"python3", "python"} {
		if path, err := exec.LookPath(name); err == nil {
			return path, nil
		}
	}
	return "", errors.New("no python3/python on PATH — cannot run tools/cuda_abi_parity.py")
}
