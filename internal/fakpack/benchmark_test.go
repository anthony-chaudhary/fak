package fakpack

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

var (
	benchInspectSink *InspectResult
	benchVerifySink  *VerifyResult
	benchCreateSink  *CreateResult
	benchBytesSink   []byte
)

// TestBenchmarkSanity ensures that all benchmarked code paths execute cleanly.
func TestBenchmarkSanity(t *testing.T) {
	dir, lockPath, policyPath, assetsDir, binDir, modelPath := createTestFixtures(t)
	bundlePath := filepath.Join(dir, "sanity.fakpack")

	cRes, err := Create(CreateOptions{
		LockPath:   lockPath,
		PolicyPath: policyPath,
		AssetsDir:  assetsDir,
		BinDir:     binDir,
		ModelPath:  modelPath,
		OutPath:    bundlePath,
	})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}
	if cRes == nil || cRes.ManifestDigest == "" {
		t.Fatal("empty create result")
	}

	iRes, err := Inspect(bundlePath)
	if err != nil {
		t.Fatalf("Inspect failed: %v", err)
	}
	if iRes == nil || iRes.LockSummary.ID == "" {
		t.Fatal("empty inspect result")
	}

	vRes, err := Verify(VerifyOptions{
		BundlePath:       bundlePath,
		ExpectedLockPath: lockPath,
	})
	if err != nil {
		t.Fatalf("Verify failed: %v", err)
	}
	if !vRes.AirGapVerified {
		t.Fatal("expected airgap verified")
	}

	lockBytes, err := ExtractLock(bundlePath)
	if err != nil || len(lockBytes) == 0 {
		t.Fatalf("ExtractLock failed: %v", err)
	}

	destDir := filepath.Join(dir, "sanity_unpacked")
	if err := Unpack(bundlePath, destDir); err != nil {
		t.Fatalf("Unpack failed: %v", err)
	}
}

// BenchmarkFakPackInspect measures inspecting bundle metadata and layer mappings.
func BenchmarkFakPackInspect(b *testing.B) {
	dir, lockPath, policyPath, assetsDir, binDir, modelPath := createTestFixtures(b)
	bundlePath := filepath.Join(dir, "inspect.fakpack")

	_, err := Create(CreateOptions{
		LockPath:   lockPath,
		PolicyPath: policyPath,
		AssetsDir:  assetsDir,
		BinDir:     binDir,
		ModelPath:  modelPath,
		OutPath:    bundlePath,
	})
	if err != nil {
		b.Fatalf("Create failed: %v", err)
	}

	fi, err := os.Stat(bundlePath)
	if err != nil {
		b.Fatalf("stat bundle: %v", err)
	}

	b.SetBytes(fi.Size())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Inspect(bundlePath)
		if err != nil {
			b.Fatalf("Inspect failed: %v", err)
		}
		benchInspectSink = res
	}
}

// BenchmarkFakPackRoundtrip measures hermetic bundle creation and verification end-to-end.
func BenchmarkFakPackRoundtrip(b *testing.B) {
	dir, lockPath, policyPath, assetsDir, binDir, modelPath := createTestFixtures(b)
	outBundle := filepath.Join(dir, "roundtrip.fakpack")

	createOpts := CreateOptions{
		LockPath:   lockPath,
		PolicyPath: policyPath,
		AssetsDir:  assetsDir,
		BinDir:     binDir,
		ModelPath:  modelPath,
		OutPath:    outBundle,
	}
	verifyOpts := VerifyOptions{
		BundlePath:       outBundle,
		ExpectedLockPath: lockPath,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cRes, err := Create(createOpts)
		if err != nil {
			b.Fatalf("Create failed: %v", err)
		}
		vRes, err := Verify(verifyOpts)
		if err != nil {
			b.Fatalf("Verify failed: %v", err)
		}
		benchCreateSink = cRes
		benchVerifySink = vRes
	}
}

// BenchmarkValidateAirgapArchive measures full offline integrity and air-gap verification of an archive.
func BenchmarkValidateAirgapArchive(b *testing.B) {
	dir, lockPath, policyPath, assetsDir, binDir, modelPath := createTestFixtures(b)
	bundlePath := filepath.Join(dir, "airgap.fakpack")

	_, err := Create(CreateOptions{
		LockPath:   lockPath,
		PolicyPath: policyPath,
		AssetsDir:  assetsDir,
		BinDir:     binDir,
		ModelPath:  modelPath,
		OutPath:    bundlePath,
	})
	if err != nil {
		b.Fatalf("Create failed: %v", err)
	}

	fi, err := os.Stat(bundlePath)
	if err != nil {
		b.Fatalf("stat bundle: %v", err)
	}

	verifyOpts := VerifyOptions{
		BundlePath:       bundlePath,
		ExpectedLockPath: lockPath,
	}

	b.SetBytes(fi.Size())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Verify(verifyOpts)
		if err != nil {
			b.Fatalf("Verify failed: %v", err)
		}
		if !res.AirGapVerified {
			b.Fatal("expected airgap verified")
		}
		benchVerifySink = res
	}
}

// BenchmarkFakPackCreate measures archive assembly and layer generation.
func BenchmarkFakPackCreate(b *testing.B) {
	dir, lockPath, policyPath, assetsDir, binDir, modelPath := createTestFixtures(b)
	outBundle := filepath.Join(dir, "bench_create.fakpack")

	createOpts := CreateOptions{
		LockPath:   lockPath,
		PolicyPath: policyPath,
		AssetsDir:  assetsDir,
		BinDir:     binDir,
		ModelPath:  modelPath,
		OutPath:    outBundle,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Create(createOpts)
		if err != nil {
			b.Fatalf("Create failed: %v", err)
		}
		benchCreateSink = res
	}
}

// BenchmarkFakPackVerify measures archive layer verification without lock comparison.
func BenchmarkFakPackVerify(b *testing.B) {
	dir, lockPath, policyPath, assetsDir, binDir, modelPath := createTestFixtures(b)
	bundlePath := filepath.Join(dir, "bench_verify.fakpack")

	_, err := Create(CreateOptions{
		LockPath:   lockPath,
		PolicyPath: policyPath,
		AssetsDir:  assetsDir,
		BinDir:     binDir,
		ModelPath:  modelPath,
		OutPath:    bundlePath,
	})
	if err != nil {
		b.Fatalf("Create failed: %v", err)
	}

	fi, err := os.Stat(bundlePath)
	if err != nil {
		b.Fatalf("stat bundle: %v", err)
	}

	verifyOpts := VerifyOptions{
		BundlePath: bundlePath,
	}

	b.SetBytes(fi.Size())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Verify(verifyOpts)
		if err != nil {
			b.Fatalf("Verify failed: %v", err)
		}
		benchVerifySink = res
	}
}

// BenchmarkFakPackExtractLock measures extracting the lock layer from a bundle archive.
func BenchmarkFakPackExtractLock(b *testing.B) {
	dir, lockPath, policyPath, assetsDir, binDir, modelPath := createTestFixtures(b)
	bundlePath := filepath.Join(dir, "bench_extract.fakpack")

	_, err := Create(CreateOptions{
		LockPath:   lockPath,
		PolicyPath: policyPath,
		AssetsDir:  assetsDir,
		BinDir:     binDir,
		ModelPath:  modelPath,
		OutPath:    bundlePath,
	})
	if err != nil {
		b.Fatalf("Create failed: %v", err)
	}

	fi, err := os.Stat(bundlePath)
	if err != nil {
		b.Fatalf("stat bundle: %v", err)
	}

	b.SetBytes(fi.Size())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, err := ExtractLock(bundlePath)
		if err != nil {
			b.Fatalf("ExtractLock failed: %v", err)
		}
		benchBytesSink = data
	}
}

// BenchmarkFakPackUnpack measures unpacking bundle layers into destination directory.
func BenchmarkFakPackUnpack(b *testing.B) {
	dir, lockPath, policyPath, assetsDir, binDir, modelPath := createTestFixtures(b)
	bundlePath := filepath.Join(dir, "bench_unpack.fakpack")

	_, err := Create(CreateOptions{
		LockPath:   lockPath,
		PolicyPath: policyPath,
		AssetsDir:  assetsDir,
		BinDir:     binDir,
		ModelPath:  modelPath,
		OutPath:    bundlePath,
	})
	if err != nil {
		b.Fatalf("Create failed: %v", err)
	}
	destDir := filepath.Join(dir, "unpacked")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Unpack(bundlePath, destDir); err != nil {
			b.Fatalf("Unpack failed: %v", err)
		}
	}
}

// BenchmarkCheckAirgap measures in-memory evaluation of air-gap constraints over lock graphs.
func BenchmarkCheckAirgap(b *testing.B) {
	for _, count := range []int{10, 50, 200} {
		b.Run(fmt.Sprintf("%d_elements", count), func(b *testing.B) {
			lock := harnesskit.ProductLock{
				Schema: harnesskit.ProductLockSchemaV2,
				ID:     "sha256:benchmark-lock",
			}
			for i := 0; i < count; i++ {
				lock.Components = append(lock.Components, harnesskit.LockedComponent{
					ID:      fmt.Sprintf("component-%04d", i),
					Version: "1.0.0",
					Digest:  "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
					Source:  fmt.Sprintf("bin/component-%04d", i),
				})
				lock.Assets = append(lock.Assets, harnesskit.LockedAsset{
					Kind:   "asset",
					ID:     fmt.Sprintf("asset-%04d", i),
					Source: fmt.Sprintf("assets/asset-%04d.txt", i),
				})
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := checkAirgap(lock); err != nil {
					b.Fatalf("checkAirgap failed: %v", err)
				}
			}
		})
	}
}
