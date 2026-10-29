package computebuild

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
)

// ErrCUDAArchSpecific refuses an unsupported opt-in build before kernel compilation.
var ErrCUDAArchSpecific = errors.New("computebuild: arch-specific CUDA build unavailable")

type cudaArchSpecificKey struct{}
type cudaArchSpecificOptions struct {
	arches  []string
	minimum string
}

var cudaNVCCRelease = regexp.MustCompile(`\brelease ([0-9]+\.[0-9]+)\b`)

// The initial lane deliberately supports exactly the targets in #13981. The
// CUDA 13.0 floor is this opt-in lane's policy, not the generic toolkit minimum.
func parseCUDAArchSpecific(text string) (*cudaArchSpecificOptions, error) {
	p := &cudaArchSpecificOptions{}
	seen := map[string]bool{}
	for line, raw := range strings.Split(text, "\n") {
		raw, _, _ = strings.Cut(raw, "#")
		fields := strings.Fields(raw)
		if len(fields) == 0 {
			continue
		}
		if len(fields) != 2 || (fields[0] != "sm_100a" && fields[0] != "sm_103a" && fields[0] != "sm_100f") || fields[1] != "13.0" || seen[fields[0]] {
			return nil, fmt.Errorf("%w: invalid or duplicate declaration at line %d", ErrCUDAArchSpecific, line+1)
		}
		seen[fields[0]] = true
		p.arches = append(p.arches, fields[0])
		p.minimum = fields[1]
	}
	if len(p.arches) != 3 {
		return nil, fmt.Errorf("%w: all three declared targets are required", ErrCUDAArchSpecific)
	}
	sortCUDAImageArchitectures(p.arches)
	return p, nil
}

func prepareCUDAArchSpecific(ctx context.Context, tc *Toolchain, root string) (*cudaArchSpecificOptions, error) {
	if tc == nil || tc.NVCC == "" || tc.IsWindows || runtime.GOOS == "windows" {
		return nil, fmt.Errorf("%w: requires Linux nvcc", ErrCUDAArchSpecific)
	}
	text, err := os.ReadFile(filepath.Join(root, "internal", "compute", "cuda_arch_specific.txt"))
	if err != nil {
		return nil, fmt.Errorf("%w: declaration: %v", ErrCUDAArchSpecific, err)
	}
	p, err := parseCUDAArchSpecific(string(text))
	if err != nil {
		return nil, err
	}
	cmd := exec.CommandContext(ctx, tc.NVCC, "--version")
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("%w: nvcc version: %v", ErrCUDAArchSpecific, err)
	}
	match := cudaNVCCRelease.FindStringSubmatch(string(out))
	if len(match) != 2 || compareVersionStrings(match[1], p.minimum) < 0 {
		return nil, fmt.Errorf("%w: nvcc must report release >= %s", ErrCUDAArchSpecific, p.minimum)
	}
	return p, nil
}

func cudaArchSpecificCompileArgs(tc *Toolchain, p *cudaArchSpecificOptions, obj string) []string {
	args := []string{"-O3", "-std=c++17"}
	for _, arch := range p.arches {
		args = append(args, "-gencode", "arch=compute_"+strings.TrimPrefix(arch, "sm_")+",code="+arch)
	}
	if tc.CXX != "" {
		args = append(args, "-ccbin", tc.CXX)
	}
	for _, inc := range tc.CUDAInc {
		args = append(args, "-I"+inc)
	}
	// No PTX fallback: a device outside the declared native objects cannot run it.
	return append(args, "-Xcompiler", "-fPIC", "-c", "cuda_sm10x_probe.cu", "-o", obj)
}
