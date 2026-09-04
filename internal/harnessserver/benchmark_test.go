package harnessserver_test

import (
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessserver"
)

var (
	benchBindingSink  harnessserver.Binding
	benchVerifiedSink harnessserver.Verified
)

// BenchmarkHarnessServer benchmarks the complete import, write, read, and revalidation loop.
func BenchmarkHarnessServer(b *testing.B) {
	root := b.TempDir()
	serverDir := filepath.Join(root, "server-product")
	harnessDir := filepath.Join(root, "harness-product")
	receipt := validReceipt(root, "local-code", 4, []string{"chat.completions", "models.list"})
	receiptPath := writeReceipt(b, serverDir, receipt)
	req := requirements("local-code", "2026-02", 4)
	bindingPath := filepath.Join(harnessDir, harnessserver.BindingFileName)

	binding, err := harnessserver.Import(harnessDir, receiptPath, req)
	if err != nil {
		b.Fatalf("setup import: %v", err)
	}
	if _, err := harnessserver.WriteBinding(bindingPath, binding); err != nil {
		b.Fatalf("setup write binding: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bnd, err := harnessserver.Import(harnessDir, receiptPath, req)
		if err != nil {
			b.Fatalf("import: %v", err)
		}
		if _, err := harnessserver.WriteBinding(bindingPath, bnd); err != nil {
			b.Fatalf("write binding: %v", err)
		}
		read, err := harnessserver.ReadBinding(bindingPath)
		if err != nil {
			b.Fatalf("read binding: %v", err)
		}
		benchBindingSink = read
		verified, err := harnessserver.VerifyFile(bindingPath)
		if err != nil {
			b.Fatalf("verify file: %v", err)
		}
		benchVerifiedSink = verified
	}
}

// BenchmarkImport measures parsing, compatibility verification, and binding derivation.
func BenchmarkImport(b *testing.B) {
	root := b.TempDir()
	receiptPath := writeReceipt(b, filepath.Join(root, "server"), validReceipt(root, "local-code", 2, []string{"chat.completions", "models.list"}))
	req := requirements("local-code", "2026-02", 2)
	harnessDir := filepath.Join(root, "harness")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		binding, err := harnessserver.Import(harnessDir, receiptPath, req)
		if err != nil {
			b.Fatalf("import: %v", err)
		}
		benchBindingSink = binding
	}
}

// BenchmarkVerifyFile measures reading a binding and cryptographically revalidating its external receipt.
func BenchmarkVerifyFile(b *testing.B) {
	root := b.TempDir()
	serverDir := filepath.Join(root, "server")
	harnessDir := filepath.Join(root, "harness")
	receiptPath := writeReceipt(b, serverDir, validReceipt(root, "local-code", 2, []string{"chat.completions", "models.list"}))
	req := requirements("local-code", "2026-02", 2)
	bindingPath := filepath.Join(harnessDir, harnessserver.BindingFileName)

	binding, err := harnessserver.Import(harnessDir, receiptPath, req)
	if err != nil {
		b.Fatalf("setup import: %v", err)
	}
	if _, err := harnessserver.WriteBinding(bindingPath, binding); err != nil {
		b.Fatalf("setup write: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		verified, err := harnessserver.VerifyFile(bindingPath)
		if err != nil {
			b.Fatalf("verify: %v", err)
		}
		benchVerifiedSink = verified
	}
}

// BenchmarkReadBinding measures strict decoding and validation of binding files.
func BenchmarkReadBinding(b *testing.B) {
	root := b.TempDir()
	harnessDir := filepath.Join(root, "harness")
	receiptPath := writeReceipt(b, filepath.Join(root, "server"), validReceipt(root, "local-code", 1, []string{"chat.completions", "models.list"}))
	req := requirements("local-code", "2026-02", 1)
	bindingPath := filepath.Join(harnessDir, harnessserver.BindingFileName)

	binding, err := harnessserver.Import(harnessDir, receiptPath, req)
	if err != nil {
		b.Fatalf("setup import: %v", err)
	}
	if _, err := harnessserver.WriteBinding(bindingPath, binding); err != nil {
		b.Fatalf("setup write: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bnd, err := harnessserver.ReadBinding(bindingPath)
		if err != nil {
			b.Fatalf("read binding: %v", err)
		}
		benchBindingSink = bnd
	}
}

// BenchmarkWriteBinding measures idempotent write preservation for unchanged bindings.
func BenchmarkWriteBinding(b *testing.B) {
	root := b.TempDir()
	harnessDir := filepath.Join(root, "harness")
	receiptPath := writeReceipt(b, filepath.Join(root, "server"), validReceipt(root, "local-code", 1, []string{"chat.completions", "models.list"}))
	req := requirements("local-code", "2026-02", 1)
	bindingPath := filepath.Join(harnessDir, harnessserver.BindingFileName)

	binding, err := harnessserver.Import(harnessDir, receiptPath, req)
	if err != nil {
		b.Fatalf("setup import: %v", err)
	}
	if _, err := harnessserver.WriteBinding(bindingPath, binding); err != nil {
		b.Fatalf("setup write: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := harnessserver.WriteBinding(bindingPath, binding)
		if err != nil {
			b.Fatalf("write binding: %v", err)
		}
		if !res.Preserved {
			b.Fatalf("expected preserved binding, got: %+v", res)
		}
	}
}

func TestBenchmarkHarnessServerExecution(t *testing.T) {
	res := testing.Benchmark(BenchmarkHarnessServer)
	if res.N <= 0 {
		t.Fatalf("expected BenchmarkHarnessServer iterations > 0, got %d", res.N)
	}
}
