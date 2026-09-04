package managedinventory

import (
	"os"
	"path/filepath"
	"testing"
)

func loadBenchCatalog(b *testing.B) Catalog {
	b.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		b.Fatal(err)
	}
	c, err := Load(filepath.Join(root, filepath.FromSlash(DefaultSourceRel)))
	if err != nil {
		b.Fatal(err)
	}
	return c
}

// BenchmarkManagedInventory measures end-to-end catalog validation and Markdown report projection.
func BenchmarkManagedInventory(b *testing.B) {
	c := loadBenchCatalog(b)
	regs := Registrations()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ds := Validate(c, regs)
		if len(ds) != 0 {
			b.Fatalf("validation produced %d unexpected diagnostics", len(ds))
		}
		md := RenderMarkdown(c)
		if len(md) == 0 {
			b.Fatal("unexpected empty markdown render")
		}
	}
}

// BenchmarkValidate measures catalog validation against registered object types.
func BenchmarkValidate(b *testing.B) {
	c := loadBenchCatalog(b)
	regs := Registrations()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ds := Validate(c, regs)
		if len(ds) != 0 {
			b.Fatalf("validation produced %d unexpected diagnostics", len(ds))
		}
	}
}

// BenchmarkRenderMarkdown measures Markdown report rendering from an in-memory catalog.
func BenchmarkRenderMarkdown(b *testing.B) {
	c := loadBenchCatalog(b)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		md := RenderMarkdown(c)
		if len(md) == 0 {
			b.Fatal("unexpected empty markdown render")
		}
	}
}

// BenchmarkCountGrepOutput measures line and file cardinality extraction from git grep output.
func BenchmarkCountGrepOutput(b *testing.B) {
	sample := []byte(
		"deadbeef:internal/adjudicator/kernel.go:10:type Policy struct\n" +
			"deadbeef:internal/adjudicator/kernel.go:25:func Admit\n" +
			"deadbeef:internal/gateway/server.go:42:type Gateway struct\n" +
			"deadbeef:internal/vdso/vdso.go:15:func Lookup\n" +
			"deadbeef:internal/ctxmmu/mmu.go:88:func Evict\n",
	)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		lines, files := CountGrepOutput(sample)
		if lines != 5 || files != 4 {
			b.Fatalf("unexpected grep count: %d lines, %d files", lines, files)
		}
	}
}

// BenchmarkLoad measures JSON parsing and schema validation of the inventory file.
func BenchmarkLoad(b *testing.B) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		b.Fatal(err)
	}
	src := filepath.Join(root, filepath.FromSlash(DefaultSourceRel))
	if _, err := os.Stat(src); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c, err := Load(src)
		if err != nil {
			b.Fatalf("load failed: %v", err)
		}
		if c.Schema != Schema {
			b.Fatalf("unexpected schema: %q", c.Schema)
		}
	}
}

// TestBenchmarkManagedInventorySanity verifies that BenchmarkManagedInventory executes cleanly.
func TestBenchmarkManagedInventorySanity(t *testing.T) {
	res := testing.Benchmark(BenchmarkManagedInventory)
	if res.N <= 0 {
		t.Fatalf("expected benchmark iterations > 0, got %d", res.N)
	}
}
