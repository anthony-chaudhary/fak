package computebuild

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

func readStableRegularFile(path, label string) ([]byte, error) {
	clean, before, err := strictAbsoluteRegularFile(path, label)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(clean)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", label, err)
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil || !os.SameFile(before, opened) {
		return nil, fmt.Errorf("%s changed before read", label)
	}
	first, err := io.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", label, err)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, fmt.Errorf("rewind %s: %w", label, err)
	}
	second, err := io.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("reread %s: %w", label, err)
	}
	afterHandle, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("restat %s: %w", label, err)
	}
	afterPath, err := os.Lstat(clean)
	if err != nil || afterPath.Mode()&os.ModeSymlink != 0 || !os.SameFile(opened, afterPath) || !sameFileMetadata(opened, afterHandle) || !bytes.Equal(first, second) {
		return nil, fmt.Errorf("%s changed during read", label)
	}
	return first, nil
}

func stableRegularFileSHA256(path, label string) (string, int64, error) {
	clean, before, err := strictAbsoluteRegularFile(path, label)
	if err != nil {
		return "", 0, err
	}
	f, err := os.Open(clean)
	if err != nil {
		return "", 0, fmt.Errorf("open %s: %w", label, err)
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil || !os.SameFile(before, opened) {
		return "", 0, fmt.Errorf("%s changed before hashing", label)
	}
	h1 := sha256.New()
	if _, err := io.Copy(h1, f); err != nil {
		return "", 0, fmt.Errorf("hash %s: %w", label, err)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return "", 0, fmt.Errorf("rewind %s: %w", label, err)
	}
	h2 := sha256.New()
	if _, err := io.Copy(h2, f); err != nil {
		return "", 0, fmt.Errorf("rehash %s: %w", label, err)
	}
	afterHandle, err := f.Stat()
	if err != nil {
		return "", 0, fmt.Errorf("restat %s: %w", label, err)
	}
	afterPath, err := os.Lstat(clean)
	if err != nil || afterPath.Mode()&os.ModeSymlink != 0 || !os.SameFile(opened, afterPath) || !sameFileMetadata(opened, afterHandle) || !bytes.Equal(h1.Sum(nil), h2.Sum(nil)) {
		return "", 0, fmt.Errorf("%s changed during hashing", label)
	}
	return hex.EncodeToString(h1.Sum(nil)), opened.Size(), nil
}

func strictAbsoluteRegularFile(path, label string) (string, os.FileInfo, error) {
	if !filepath.IsAbs(path) {
		return "", nil, fmt.Errorf("%s path must be absolute", label)
	}
	clean := filepath.Clean(path)
	info, err := os.Lstat(clean)
	if err != nil {
		return "", nil, fmt.Errorf("stat %s: %w", label, err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
		return "", nil, fmt.Errorf("%s must be a non-symlink regular file", label)
	}
	resolved, err := filepath.EvalSymlinks(clean)
	if err != nil || !samePath(clean, resolved) {
		return "", nil, fmt.Errorf("%s path must not traverse symlinks", label)
	}
	return clean, info, nil
}

func strictAbsoluteDirectory(path, label string) (string, error) {
	if !filepath.IsAbs(path) {
		return "", fmt.Errorf("%s path must be absolute", label)
	}
	clean := filepath.Clean(path)
	info, err := os.Lstat(clean)
	if err != nil {
		return "", fmt.Errorf("stat %s: %w", label, err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return "", fmt.Errorf("%s must be a non-symlink directory", label)
	}
	resolved, err := filepath.EvalSymlinks(clean)
	if err != nil || !samePath(clean, resolved) {
		return "", fmt.Errorf("%s path must not traverse symlinks", label)
	}
	return clean, nil
}

func samePath(a, b string) bool {
	a = filepath.Clean(a)
	b = filepath.Clean(b)
	if os.PathSeparator == '\\' {
		return strings.EqualFold(a, b)
	}
	return a == b
}

func sameFileMetadata(a, b os.FileInfo) bool {
	return os.SameFile(a, b) && a.Size() == b.Size() && a.Mode() == b.Mode() && a.ModTime().Equal(b.ModTime())
}

func stableOpenFileSHA256(f *os.File, pinned os.FileInfo, label string) (string, error) {
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return "", fmt.Errorf("rewind %s: %w", label, err)
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", fmt.Errorf("hash %s: %w", label, err)
	}
	after, err := f.Stat()
	if err != nil || !sameFileMetadata(pinned, after) {
		return "", fmt.Errorf("%s changed while pinned", label)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
