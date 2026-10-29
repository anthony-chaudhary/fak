package computebuild

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

var ErrCUDAVendorSource = errors.New("computebuild: pinned CUDA vendor source unavailable")

type cudaVendorSourceFile struct {
	Path   string `json:"path"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}

type cudaVendorSourceManifest struct {
	Schema         string                 `json:"schema"`
	FlashMLACommit string                 `json:"flashmla_commit"`
	CUTLASSCommit  string                 `json:"cutlass_commit"`
	Files          []cudaVendorSourceFile `json:"files"`
}

func cudaVendorRegularPath(root, relative string) (string, error) {
	if filepath.IsAbs(relative) || strings.Contains(relative, "\\") || !strings.HasPrefix(relative, "third_party/") || filepath.ToSlash(filepath.Clean(relative)) != relative {
		return "", fmt.Errorf("%w: invalid source path", ErrCUDAVendorSource)
	}
	path := root
	for _, part := range strings.Split(relative, "/") {
		path = filepath.Join(path, part)
		info, err := os.Lstat(path)
		if err != nil || info.Mode()&os.ModeSymlink != 0 {
			return "", fmt.Errorf("%w: missing or symbolic source %s", ErrCUDAVendorSource, relative)
		}
	}
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() {
		return "", fmt.Errorf("%w: nonregular source %s", ErrCUDAVendorSource, relative)
	}
	return path, nil
}

func verifyCUDAVendorSources(root, relative, pin string) (*cudaVendorSourceManifest, error) {
	path, err := cudaVendorRegularPath(root, relative)
	if err != nil {
		return nil, err
	}
	info, err := os.Stat(path)
	if err != nil || info.Size() > 1<<20 {
		return nil, fmt.Errorf("%w: oversized manifest", ErrCUDAVendorSource)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("%w: manifest read: %v", ErrCUDAVendorSource, err)
	}
	sum := sha256.Sum256(raw)
	if len(raw) > 1<<20 || hex.EncodeToString(sum[:]) != pin {
		return nil, fmt.Errorf("%w: manifest digest mismatch", ErrCUDAVendorSource)
	}
	var manifest cudaVendorSourceManifest
	if err := json.Unmarshal(raw, &manifest); err != nil || manifest.Schema != "fak.flashmla.vendor.reconstruction.v1" || len(manifest.Files) == 0 {
		return nil, fmt.Errorf("%w: invalid manifest", ErrCUDAVendorSource)
	}
	seen := map[string]bool{}
	for _, file := range manifest.Files {
		if seen[file.Path] || file.Bytes < 0 || len(file.SHA256) != 64 {
			return nil, fmt.Errorf("%w: invalid or duplicate source", ErrCUDAVendorSource)
		}
		seen[file.Path] = true
		path, err := cudaVendorRegularPath(root, file.Path)
		if err != nil {
			return nil, err
		}
		info, err := os.Stat(path)
		if err != nil || info.Size() != file.Bytes {
			return nil, fmt.Errorf("%w: source size mismatch: %s", ErrCUDAVendorSource, file.Path)
		}
		digest, err := cudaFileDigest(path)
		if err != nil || digest != file.SHA256 {
			return nil, fmt.Errorf("%w: source digest mismatch: %s", ErrCUDAVendorSource, file.Path)
		}
	}
	return &manifest, nil
}
