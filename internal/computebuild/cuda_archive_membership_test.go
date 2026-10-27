package computebuild

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"
)

// fak-test:justify why=regression when=changed:internal/computebuild/**
// fak-test:runtime fast est=2s
func TestCUDAArchiveResetsMembershipAndPreservesFailure(t *testing.T) {
	root, tc := cudaIncrementalFixture(t)
	dir := filepath.Join(root, "internal", "compute")
	kernel := filepath.Join(dir, "cuda_kernels.o")
	nccl := filepath.Join(dir, "cuda_nccl.o")
	for _, path := range []string{kernel, nccl} {
		if err := os.WriteFile(path, []byte("object"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	archive := func(objs []string) error {
		_, err := ArchiveCUDALib(context.Background(), tc, dir, objs, io.Discard)
		return err
	}
	if err := archive([]string{kernel, nccl}); err != nil {
		t.Fatal(err)
	}
	if err := archive([]string{kernel}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "libfakcuda.a")
	old, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var members map[string]bool
	if err := json.Unmarshal(old, &members); err != nil {
		t.Fatal(err)
	}
	if len(members) != 1 || !members["cuda_kernels.o"] {
		t.Fatalf("archive retained stale NCCL member: %v", members)
	}
	t.Setenv("FAK_CUDA_INCREMENTAL_ARCHIVE_FAIL", "1")
	if err := archive([]string{kernel, nccl}); err == nil {
		t.Fatal("failed archive accepted")
	}
	after, err := os.ReadFile(path)
	if err != nil || string(after) != string(old) {
		t.Fatalf("failed archiver replaced old output: err=%v", err)
	}
	temps, err := filepath.Glob(filepath.Join(dir, ".cuda-archive-*.a"))
	if err != nil || len(temps) != 0 {
		t.Fatalf("archive temps retained: %v err=%v", temps, err)
	}
}
