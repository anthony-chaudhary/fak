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
// fak-test:runtime fast est=10ms
func TestCUDANativeReceiptSeparatesBuildWorkFromHardwareEvidence(t *testing.T) {
	root := t.TempDir()
	main := filepath.Join(root, "receipt.json")
	ctx := context.WithValue(context.Background(), cudaReceiptPathContext{}, CUDANativeReceiptPath(root, main))
	r := newCUDANativeReceipt(ctx, root, "sm_120", true)
	r.Objects = []CUDAObjectBuildReceipt{
		{Source: "cuda_kernels.cu", Action: "reused", InputSHA256: "input-a", SourceClosureSHA256: "source-a", OutputSHA256: "object-a"},
		{Source: "cuda_nccl.cu", Action: "executed", MissReason: "inputs_changed", InputSHA256: "input-b", SourceClosureSHA256: "source-b", OutputSHA256: "object-b"},
	}
	if err := r.finish(root, nil); err != nil {
		t.Fatal(err)
	}
	read := func() CUDANativeBuildReceipt {
		t.Helper()
		raw, err := os.ReadFile(CUDANativeReceiptPath(root, main))
		if err != nil {
			t.Fatal(err)
		}
		var got CUDANativeBuildReceipt
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatal(err)
		}
		return got
	}
	got := read()
	if got.Schema != CUDANativeBuildReceiptSchema || got.Mode != "mixed" || got.Executed != 1 || got.Reused != 1 || got.SourceClosureSHA256 == "" || got.HostOS == "" || got.StartedAt.IsZero() || got.FinishedAt.Before(got.StartedAt) {
		t.Fatalf("incomplete build evidence: %+v", got)
	}
	if got.SourceKnown {
		t.Fatal("non-Git fixture claimed known source revision")
	}
	archive := filepath.Join(root, "internal", "compute", "libfakcuda.a")
	if err := os.MkdirAll(filepath.Dir(archive), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(archive, []byte("archive"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(main, []byte(`{"phases":[{"name":"compile_kernels","elapsed_ms":2},{"name":"archive","elapsed_ms":3},{"name":"build_or_test","elapsed_ms":4}]}`), 0600); err != nil {
		t.Fatal(err)
	}
	if err := finishCUDANativeArchiveReceipt(root, main, nil); err != nil {
		t.Fatal(err)
	}
	got = read()
	if len(got.ArchiveSHA256) != 64 || len(got.ComputeReceiptSHA256) != 64 || len(got.ComputeReceiptCanonicalSHA256) != 64 || got.NativeElapsedMS != 5 {
		t.Fatalf("unbound archive/main receipt: %+v", got)
	}
	if err := os.Remove(main); err != nil {
		t.Fatal(err)
	}
	if err := finishCUDANativeArchiveReceipt(root, main, nil); err == nil {
		t.Fatal("missing main receipt digest accepted")
	}
}

// fak-test:justify why=regression when=changed:internal/computebuild/**
// fak-test:runtime fast est=2s
func TestCUDAEarlyFailureReplacesPriorNativeReceipt(t *testing.T) {
	root, tc := cudaIncrementalFixture(t)
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"nvcc", "arch", "audit"} {
		t.Run(mode, func(t *testing.T) {
			old := newCUDANativeReceipt(context.Background(), root, "old-architecture", true)
			old.SourceCommit = "previous-source"
			old.Objects = []CUDAObjectBuildReceipt{{Source: "old.cu", Action: "reused", SourceClosureSHA256: "old-source"}}
			if err := old.finish(root, nil); err != nil {
				t.Fatal(err)
			}
			selected, arch := *tc, "sm_120"
			switch mode {
			case "nvcc":
				selected.NVCC = filepath.Join(root, "missing-nvcc")
			case "arch":
				arch = "sm_9999"
			case "audit":
				selected.NVCC = self
			}
			if _, err := BuildCUDALib(context.Background(), &selected, root, arch, false, io.Discard); err == nil {
				t.Fatal("invalid native preflight succeeded")
			}
			got := cudaReadNativeReceipt(t, root)
			if got.Outcome != "failed" || got.Executed != 0 || got.Reused != 0 || len(got.Objects) != 0 || got.SourceCommit == "previous-source" || got.SourceClosureSHA256 != "" || got.Architecture != arch || got.NCCL {
				t.Fatalf("stale native evidence after %s preflight: %+v", mode, got)
			}
		})
	}
}
