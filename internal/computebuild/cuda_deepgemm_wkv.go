package computebuild

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
)

var ErrCUDADeepGEMMWKV = errors.New("computebuild: DeepGEMM WKV candidate unavailable")

const cudaDeepGEMMManifestSHA256 = "41cad574726b9d3ad961ea560ff1f6d4e2dc1c49536237a84d7516af1c9d32b3"

type cudaDeepGEMMKey struct{}
type cudaDeepGEMMOptions struct{ receipt CUDADeepGEMMWKVReceipt }

// CUDADeepGEMMWKVReceipt identifies a build candidate, never physical acceptance.
type CUDADeepGEMMWKVReceipt struct {
	ManifestSHA256    string                 `json:"manifest_sha256"`
	DeepGEMMCommit    string                 `json:"deepgemm_commit"`
	CUTLASSCommit     string                 `json:"cutlass_commit"`
	MinimumNVCC       string                 `json:"minimum_nvcc"`
	SMCount           int                    `json:"sm_count"`
	HardwareQualified bool                   `json:"hardware_qualified"`
	ObjectImages      *CUDAArchiveImageAudit `json:"object_images,omitempty"`
}

func requireCUDADeepGEMMVersion(text string) error {
	match := cudaNVCCRelease.FindStringSubmatch(text)
	if len(match) != 2 || compareVersionStrings(match[1], "13.1") < 0 {
		return fmt.Errorf("%w: nvcc release >= 13.1 required", ErrCUDADeepGEMMWKV)
	}
	return nil
}

func prepareCUDADeepGEMM(ctx context.Context, cfg *CUDAConfig) (*cudaDeepGEMMOptions, error) {
	if !cfg.EnableArchSpecific || cfg.DeepGEMMWKVSMs <= 0 || cfg.Toolchain == nil || cfg.Toolchain.NVCC == "" || cfg.Toolchain.IsWindows || runtime.GOOS != "linux" {
		return nil, fmt.Errorf("%w: explicit arch-specific Linux nvcc build and positive physical SM count required", ErrCUDADeepGEMMWKV)
	}
	out, err := exec.CommandContext(ctx, cfg.Toolchain.NVCC, "--version").Output()
	if err != nil {
		return nil, fmt.Errorf("%w: nvcc version: %v", ErrCUDADeepGEMMWKV, err)
	}
	if err := requireCUDADeepGEMMVersion(string(out)); err != nil {
		return nil, err
	}
	manifest, err := cudaVendorRegularPath(cfg.RepoRoot, "third_party/deepgemm-wkv-source.json")
	if err != nil {
		return nil, err
	}
	info, err := os.Stat(manifest)
	if err != nil || info.Size() > 1<<20 {
		return nil, fmt.Errorf("%w: oversized manifest", ErrCUDADeepGEMMWKV)
	}
	raw, err := os.ReadFile(manifest)
	if err != nil {
		return nil, fmt.Errorf("%w: manifest read: %v", ErrCUDADeepGEMMWKV, err)
	}
	sum := sha256.Sum256(raw)
	if len(raw) > 1<<20 || hex.EncodeToString(sum[:]) != cudaDeepGEMMManifestSHA256 {
		return nil, fmt.Errorf("%w: manifest identity", ErrCUDADeepGEMMWKV)
	}
	var source struct {
		Schema   string `json:"schema"`
		DeepGEMM string `json:"deepgemm"`
		CUTLASS  string `json:"cutlass"`
		Files    []struct {
			Path   string `json:"path"`
			Size   int64  `json:"size"`
			SHA256 string `json:"sha256"`
		} `json:"files"`
	}
	if json.Unmarshal(raw, &source) != nil || source.Schema != "fak.deepgemm-wkv-source/v1" || len(source.Files) != 177 {
		return nil, fmt.Errorf("%w: manifest shape", ErrCUDADeepGEMMWKV)
	}
	seen := map[string]bool{}
	for _, file := range source.Files {
		if seen[file.Path] || file.Size < 0 || len(file.SHA256) != 64 {
			return nil, fmt.Errorf("%w: duplicate or invalid source", ErrCUDADeepGEMMWKV)
		}
		seen[file.Path] = true
		path, err := cudaVendorRegularPath(cfg.RepoRoot, file.Path)
		if err != nil {
			return nil, err
		}
		info, err := os.Stat(path)
		if err != nil || info.Size() != file.Size {
			return nil, fmt.Errorf("%w: source size: %s", ErrCUDADeepGEMMWKV, file.Path)
		}
		digest, err := cudaFileDigest(path)
		if err != nil || digest != file.SHA256 {
			return nil, fmt.Errorf("%w: source digest: %s", ErrCUDADeepGEMMWKV, file.Path)
		}
	}
	return &cudaDeepGEMMOptions{receipt: CUDADeepGEMMWKVReceipt{ManifestSHA256: cudaDeepGEMMManifestSHA256, DeepGEMMCommit: source.DeepGEMM, CUTLASSCommit: source.CUTLASS, MinimumNVCC: "13.1", SMCount: cfg.DeepGEMMWKVSMs}}, nil
}

func cudaDeepGEMMCompileArgs(tc *Toolchain, root, object string, sms int) []string {
	args := []string{"-gencode", "arch=compute_100f,code=sm_100f", "-gencode", "arch=compute_103f,code=sm_103f", "-O3", "--compiler-options=-O3", "--ptxas-options=--register-usage-level=10", "-std=c++20", "--compiler-options=-fPIC", "--compiler-options=-fconcepts", "--expt-relaxed-constexpr", "--expt-extended-lambda", "--diag-suppress=39,161,174,177,186,940", "--compiler-options=-Wno-deprecated-declarations,-Wno-abi", "-DFAK_DEEPGEMM_WKV_SMS=" + strconv.Itoa(sms)}
	if tc.CXX != "" {
		args = append(args, "-ccbin", tc.CXX)
	}
	for _, include := range tc.CUDAInc {
		args = append(args, "-I"+include, "-I"+filepath.Join(include, "cccl"))
	}
	if tc.CUDAHome != "" {
		for _, target := range []string{"x86_64-linux", "sbsa-linux"} {
			args = append(args, "-I"+filepath.Join(tc.CUDAHome, "targets", target, "include", "cccl"))
		}
	}
	for _, include := range []string{"third_party/deepgemm/deep_gemm/include", "third_party/deepgemm_cutlass/include"} {
		args = append(args, "-I"+filepath.Join(root, filepath.FromSlash(include)))
	}
	return append(args, "-c", "cuda_deepgemm_wkv.cu", "-o", object)
}

func compileCUDADeepGEMM(ctx context.Context, tc *Toolchain, root string, env []string, receipt *CUDANativeBuildReceipt, stdout io.Writer) (string, error) {
	plan, ok := ctx.Value(cudaDeepGEMMKey{}).(*cudaDeepGEMMOptions)
	if !ok {
		return "", nil
	}
	pkg := filepath.Join(root, "internal", "compute")
	object := filepath.Join(pkg, "cuda_deepgemm_wkv.o")
	if err := compileCUDAObject(ctx, tc, pkg, "cuda_deepgemm_wkv.cu", object, cudaDeepGEMMCompileArgs(tc, root, object, plan.receipt.SMCount), env, receipt, stdout); err != nil {
		return "", fmt.Errorf("%w: dedicated translation unit: %v", ErrCUDADeepGEMMWKV, err)
	}
	images, err := auditCUDAImages(ctx, tc, object, []string{"sm_100f", "sm_103f"}, nil)
	if err != nil {
		return "", err
	}
	plan.receipt.ObjectImages = images
	if receipt.DeepGEMMWKV != nil {
		receipt.DeepGEMMWKV.ObjectImages = images
	}
	return object, nil
}

func cudaDeepGEMMGenericFlags(ctx context.Context, args []string) []string {
	if _, ok := ctx.Value(cudaDeepGEMMKey{}).(*cudaDeepGEMMOptions); ok {
		return append(args, "-DFAK_CUDA_DEEPGEMM_WKV")
	}
	return args
}
