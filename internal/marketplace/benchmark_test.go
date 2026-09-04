package marketplace

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

var (
	benchCatalogSink    Catalog
	benchReportSink     Report
	benchBytesSink      []byte
	benchDescriptorSink Descriptor
	benchStringSink     string
)

func setupBenchmarkCatalog(root string, count int) (Catalog, []byte, map[string]int, error) {
	adapters := []Adapter{
		ComputeAdapter,
		ABIAdapter,
		TUIPaneAdapter,
		QualityAdapter,
		TrajectoryScorerAdapter,
	}
	trustClasses := []TrustClass{TrustCompiled, TrustData, TrustUntrusted}
	errorBehaviors := []ErrorBehavior{ErrorClosed, ErrorIsolate, ErrorOpen}

	descriptors := make([]Descriptor, 0, count)
	for i := 0; i < count; i++ {
		adapter := adapters[i%len(adapters)]
		fileName := fmt.Sprintf("ext_%03d.bin", i)
		filePath := filepath.Join(root, fileName)
		content := []byte(fmt.Sprintf("extension payload data for entry %d with seam %s\n", i, adapter.Seam))
		if err := os.WriteFile(filePath, content, 0o600); err != nil {
			return Catalog{}, nil, nil, err
		}

		rawDesc := Descriptor{
			ID:             fmt.Sprintf("vendor.org/ext-%03d", i),
			Module:         Module(fmt.Sprintf("internal/%s", adapter.Seam), i+1, fmt.Sprintf("%07x", i+0x1000000)),
			Compatibility:  Compatibility{Min: 1, Max: 3},
			Artifact:       fileName,
			ArtifactSHA256: SHA256(content),
			Trust:          trustClasses[i%len(trustClasses)],
			OnError:        errorBehaviors[i%len(errorBehaviors)],
			Capabilities:   []string{fmt.Sprintf("cap.%s", adapter.Seam), "runtime.safe"},
		}
		descriptors = append(descriptors, adapter.Descriptor(rawDesc))
	}

	cat := Catalog{
		Schema:     Schema,
		Extensions: descriptors,
	}
	rawJSON, err := Marshal(cat)
	if err != nil {
		return Catalog{}, nil, nil, err
	}

	versions := map[string]int{
		"fak-abi":               1,
		"fak-compute":           2,
		"fak-tui-pane":          1,
		"fak-quality":           2,
		"fak-trajectory-scorer": 1,
	}

	return cat, rawJSON, versions, nil
}

func BenchmarkMarketplace(b *testing.B) {
	root := b.TempDir()
	_, rawJSON, versions, err := setupBenchmarkCatalog(root, 5)
	if err != nil {
		b.Fatalf("setup failed: %v", err)
	}
	ctx := context.Background()
	opts := VerifyOptions{Root: root, ABIVersions: versions}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		catalog, err := Parse(rawJSON, versions)
		if err != nil {
			b.Fatalf("parse failed: %v", err)
		}
		report, err := Verify(ctx, catalog, opts)
		if err != nil || !report.Valid {
			b.Fatalf("verify failed: err=%v, report=%+v", err, report)
		}
		marshaled, err := Marshal(catalog)
		if err != nil {
			b.Fatalf("marshal failed: %v", err)
		}
		benchCatalogSink = catalog
		benchReportSink = report
		benchBytesSink = marshaled
	}
}

func BenchmarkParse(b *testing.B) {
	for _, size := range []int{1, 5, 25} {
		b.Run(fmt.Sprintf("%d_descriptors", size), func(b *testing.B) {
			root := b.TempDir()
			_, rawJSON, versions, err := setupBenchmarkCatalog(root, size)
			if err != nil {
				b.Fatalf("setup failed: %v", err)
			}

			b.SetBytes(int64(len(rawJSON)))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				catalog, err := Parse(rawJSON, versions)
				if err != nil {
					b.Fatalf("parse failed: %v", err)
				}
				benchCatalogSink = catalog
			}
		})
	}
}

func BenchmarkMarshal(b *testing.B) {
	for _, size := range []int{1, 5, 25} {
		b.Run(fmt.Sprintf("%d_descriptors", size), func(b *testing.B) {
			root := b.TempDir()
			cat, _, _, err := setupBenchmarkCatalog(root, size)
			if err != nil {
				b.Fatalf("setup failed: %v", err)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				data, err := Marshal(cat)
				if err != nil {
					b.Fatalf("marshal failed: %v", err)
				}
				benchBytesSink = data
			}
		})
	}
}

func BenchmarkVerify(b *testing.B) {
	root := b.TempDir()
	cat, _, versions, err := setupBenchmarkCatalog(root, 5)
	if err != nil {
		b.Fatalf("setup failed: %v", err)
	}
	ctx := context.Background()
	opts := VerifyOptions{Root: root, ABIVersions: versions}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report, err := Verify(ctx, cat, opts)
		if err != nil || !report.Valid {
			b.Fatalf("verify failed: err=%v report=%+v", err, report)
		}
		benchReportSink = report
	}
}

func BenchmarkValidateDescriptor(b *testing.B) {
	root := b.TempDir()
	cat, _, versions, err := setupBenchmarkCatalog(root, 5)
	if err != nil {
		b.Fatalf("setup failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		d := cat.Extensions[i%len(cat.Extensions)]
		if err := validateDescriptor(d, versions); err != nil {
			b.Fatalf("validation failed: %v", err)
		}
	}
}

func BenchmarkAdapterDescriptor(b *testing.B) {
	adapters := []Adapter{ComputeAdapter, ABIAdapter, TUIPaneAdapter, QualityAdapter, TrajectoryScorerAdapter}
	base := Descriptor{
		ID:             "vendor.org/base",
		Module:         "internal/ext@r1+g0123456",
		Compatibility:  Compatibility{Min: 1, Max: 2},
		Artifact:       "base.bin",
		ArtifactSHA256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		Trust:          TrustData,
		OnError:        ErrorClosed,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchDescriptorSink = adapters[i%len(adapters)].Descriptor(base)
	}
}

func BenchmarkSHA256(b *testing.B) {
	payloads := [][]byte{
		[]byte("small payload"),
		make([]byte, 1024),
		make([]byte, 65536),
	}
	for _, p := range payloads {
		b.Run(fmt.Sprintf("%d_bytes", len(p)), func(b *testing.B) {
			b.SetBytes(int64(len(p)))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchStringSink = SHA256(p)
			}
		})
	}
}

func TestBenchmarkMarketplaceExecution(t *testing.T) {
	root := t.TempDir()
	cat, rawJSON, versions, err := setupBenchmarkCatalog(root, 5)
	if err != nil {
		t.Fatalf("setup failed: %v", err)
	}
	parsed, err := Parse(rawJSON, versions)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if len(parsed.Extensions) != len(cat.Extensions) {
		t.Fatalf("expected %d extensions, got %d", len(cat.Extensions), len(parsed.Extensions))
	}
	ctx := context.Background()
	report, err := Verify(ctx, parsed, VerifyOptions{Root: root, ABIVersions: versions})
	if err != nil || !report.Valid || len(report.Verified) != len(cat.Extensions) {
		t.Fatalf("verify failed: err=%v, report=%+v", err, report)
	}
	out, err := Marshal(parsed)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	if len(out) == 0 {
		t.Fatal("empty marshal output")
	}
}
