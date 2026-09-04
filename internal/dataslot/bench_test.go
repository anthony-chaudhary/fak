package dataslot

import (
	"os"
	"path/filepath"
	"testing"
)

// BenchmarkDataSlot exercises data slot descriptor validation in a loop.
func BenchmarkDataSlot(b *testing.B) {
	desc := DataSlotDescriptor{
		ID:             "sqlite:app.db",
		Family:         FamilySQLite,
		Status:         StatusReady,
		SourceArtifact: "app.db",
		LocalPath:      "app.db",
		ReadOnly:       true,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := desc.Validate(); err != nil {
			b.Fatalf("validate failed: %v", err)
		}
	}
}

// BenchmarkDataSlotDetect exercises data slot detection in a loop.
func BenchmarkDataSlotDetect(b *testing.B) {
	dir := b.TempDir()
	dbPath := filepath.Join(dir, "dev.sqlite")
	content := append([]byte(sqliteMagic), make([]byte, 100)...)
	if err := os.WriteFile(dbPath, content, 0644); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		slots, err := Detect(dir)
		if err != nil || len(slots) != 1 {
			b.Fatalf("detect failed: %v", err)
		}
	}
}
