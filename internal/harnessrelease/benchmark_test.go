package harnessrelease

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func createArchive(tb testing.TB, path, format, binaryName string, content []byte) {
	tb.Helper()
	f, err := os.Create(path)
	if err != nil {
		tb.Fatalf("create archive %s: %v", path, err)
	}
	defer f.Close()

	switch format {
	case "zip":
		z := zip.NewWriter(f)
		h := &zip.FileHeader{Name: binaryName, Method: zip.Deflate}
		h.SetMode(0o755)
		w, err := z.CreateHeader(h)
		if err != nil {
			tb.Fatalf("create zip header: %v", err)
		}
		if _, err := w.Write(content); err != nil {
			tb.Fatalf("write zip entry: %v", err)
		}
		if err := z.Close(); err != nil {
			tb.Fatalf("close zip: %v", err)
		}
	case "tar.gz":
		gw := gzip.NewWriter(f)
		tw := tar.NewWriter(gw)
		h := &tar.Header{
			Name:     binaryName,
			Mode:     0o755,
			Size:     int64(len(content)),
			Typeflag: tar.TypeReg,
		}
		if err := tw.WriteHeader(h); err != nil {
			tb.Fatalf("write tar header: %v", err)
		}
		if _, err := tw.Write(content); err != nil {
			tb.Fatalf("write tar entry: %v", err)
		}
		if err := tw.Close(); err != nil {
			tb.Fatalf("close tar: %v", err)
		}
		if err := gw.Close(); err != nil {
			tb.Fatalf("close gzip: %v", err)
		}
	default:
		tb.Fatalf("unsupported format: %s", format)
	}
}

func BenchmarkHarnessRelease(b *testing.B) {
	dir := b.TempDir()
	binName := "fak"
	if runtime.GOOS == "windows" {
		binName += ".exe"
	}
	archivePath := filepath.Join(dir, "release.zip")
	content := []byte("#!/bin/sh\necho mock-fak-binary\n")
	createArchive(b, archivePath, "zip", binName, content)

	sum, err := fileSHA256(archivePath)
	if err != nil {
		b.Fatalf("fileSHA256: %v", err)
	}
	sidecarPath := archivePath + ".sha256"
	if err := os.WriteFile(sidecarPath, []byte(fmt.Sprintf("%s  release.zip\n", sum)), 0o644); err != nil {
		b.Fatalf("write sidecar: %v", err)
	}

	extractDir := filepath.Join(dir, "extract")
	receiptPath := filepath.Join(dir, "receipt.json")

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		hash, err := VerifyChecksum(archivePath, sidecarPath)
		if err != nil {
			b.Fatalf("VerifyChecksum: %v", err)
		}

		if err := Extract(archivePath, extractDir); err != nil {
			b.Fatalf("Extract: %v", err)
		}

		binPath, err := findBinary(extractDir)
		if err != nil {
			b.Fatalf("findBinary: %v", err)
		}

		binHash, err := fileSHA256(binPath)
		if err != nil {
			b.Fatalf("fileSHA256: %v", err)
		}

		r := Receipt{
			Schema:        Schema,
			Outcome:       "success",
			Target:        runtime.GOOS + "_" + runtime.GOARCH,
			Archive:       archivePath,
			ArchiveSHA256: hash,
			BinarySHA256:  binHash,
			ProductDir:    extractDir,
			Module:        "example.test/harness",
		}
		if err := writeReceipt(receiptPath, r); err != nil {
			b.Fatalf("writeReceipt: %v", err)
		}
	}
}

func BenchmarkVerifyChecksum(b *testing.B) {
	dir := b.TempDir()
	archivePath := filepath.Join(dir, "asset.zip")
	payload := make([]byte, 64*1024)
	for i := range payload {
		payload[i] = byte(i % 251)
	}
	if err := os.WriteFile(archivePath, payload, 0o644); err != nil {
		b.Fatalf("write payload: %v", err)
	}
	sum := sha256.Sum256(payload)
	sidecarPath := archivePath + ".sha256"
	if err := os.WriteFile(sidecarPath, []byte(fmt.Sprintf("%x  asset.zip\n", sum)), 0o644); err != nil {
		b.Fatalf("write sidecar: %v", err)
	}

	b.SetBytes(int64(len(payload)))
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := VerifyChecksum(archivePath, sidecarPath)
		if err != nil {
			b.Fatalf("VerifyChecksum: %v", err)
		}
	}
}

func BenchmarkExtractZip(b *testing.B) {
	dir := b.TempDir()
	archivePath := filepath.Join(dir, "release.zip")
	createArchive(b, archivePath, "zip", "fak", []byte("sample payload data"))
	outDir := filepath.Join(dir, "zip_out")

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if err := Extract(archivePath, outDir); err != nil {
			b.Fatalf("Extract: %v", err)
		}
	}
}

func BenchmarkExtractTarGZ(b *testing.B) {
	dir := b.TempDir()
	archivePath := filepath.Join(dir, "release.tar.gz")
	createArchive(b, archivePath, "tar.gz", "fak", []byte("sample payload data"))
	outDir := filepath.Join(dir, "targz_out")

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if err := Extract(archivePath, outDir); err != nil {
			b.Fatalf("Extract: %v", err)
		}
	}
}

func BenchmarkReceiptSerialization(b *testing.B) {
	dir := b.TempDir()
	receiptPath := filepath.Join(dir, "bench_receipt.json")
	r := Receipt{
		Schema:              Schema,
		Outcome:             "success",
		Target:              "linux_amd64",
		Archive:             "/releases/fak_v1.0.0.tar.gz",
		ArchiveSHA256:       "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
		BinarySHA256:        "cb8379ac2098aa165029e3938a51da0bcecfc008fd6795f401178647f96c5b34",
		ProductDir:          "/srv/harness/product",
		Module:              "example.test/product",
		Generator:           "fak-harness/v1",
		ContractVersion:     "v1alpha1",
		FAKModule:           "example.test/fak",
		FAKVersion:          "v1.0.0",
		UpgradeCommand:      "fak harness init --dir .",
		RollbackCommand:     "install-v0.9.0",
		UserConfigSHA256:    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
		UserConfigPreserved: true,
		ElapsedSeconds:      0.452,
		Commands: []CommandReceipt{
			{Command: []string{"fak", "harness", "init"}, ExitCode: 0, ElapsedSeconds: 0.12},
			{Command: []string{"go", "build", "./cmd/product"}, ExitCode: 0, ElapsedSeconds: 0.25},
			{Command: []string{"./product-bin", "--selfcheck"}, ExitCode: 0, ElapsedSeconds: 0.08},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if err := writeReceipt(receiptPath, r); err != nil {
			b.Fatalf("writeReceipt: %v", err)
		}
	}
}

func TestBenchmarkHarnessReleaseRuns(t *testing.T) {
	dir := t.TempDir()
	binName := "fak"
	if runtime.GOOS == "windows" {
		binName += ".exe"
	}
	archivePath := filepath.Join(dir, "release.zip")
	content := []byte("#!/bin/sh\necho mock-fak-binary\n")
	createArchive(t, archivePath, "zip", binName, content)

	sum, err := fileSHA256(archivePath)
	if err != nil {
		t.Fatalf("fileSHA256: %v", err)
	}
	sidecarPath := archivePath + ".sha256"
	if err := os.WriteFile(sidecarPath, []byte(fmt.Sprintf("%s  release.zip\n", sum)), 0o644); err != nil {
		t.Fatalf("write sidecar: %v", err)
	}

	digest, err := VerifyChecksum(archivePath, sidecarPath)
	if err != nil || digest != sum {
		t.Fatalf("checksum mismatch or err: digest=%q sum=%q err=%v", digest, sum, err)
	}

	extractDir := filepath.Join(dir, "extract")
	if err := Extract(archivePath, extractDir); err != nil {
		t.Fatalf("extract err: %v", err)
	}

	binPath, err := findBinary(extractDir)
	if err != nil {
		t.Fatalf("findBinary err: %v", err)
	}
	binHash, err := fileSHA256(binPath)
	if err != nil {
		t.Fatalf("fileSHA256 err: %v", err)
	}

	receiptPath := filepath.Join(dir, "receipt.json")
	receipt := Receipt{
		Schema:        Schema,
		Outcome:       "success",
		Target:        runtime.GOOS + "_" + runtime.GOARCH,
		Archive:       archivePath,
		ArchiveSHA256: digest,
		BinarySHA256:  binHash,
		ProductDir:    extractDir,
		Module:        "example.test/bench-harness",
	}
	if err := writeReceipt(receiptPath, receipt); err != nil {
		t.Fatalf("writeReceipt err: %v", err)
	}
}
