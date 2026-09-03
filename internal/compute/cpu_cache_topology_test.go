package compute

import (
	"os"
	"path/filepath"
	"testing"
)

func TestHostCPUTopologyWitness(t *testing.T) {
	// First witness requirements (#9955):
	// 1. Fixture parser for Linux sysfs produces typed L1d/L1i/L2/L3 rows
	// 2. Missing data reports "unknown" and -1 bytes (never zero!)
	// 3. Live adapter discovery
	// 4. Native Qwen3.8 receipt names the topology without claiming a gain

	t.Run("linux_sysfs_complete_fixture", func(t *testing.T) {
		tmpDir := t.TempDir()

		createIndex := func(idx string, level, typ, size, line string) {
			dir := filepath.Join(tmpDir, idx)
			if err := os.MkdirAll(dir, 0o755); err != nil {
				t.Fatal(err)
			}
			if level != "" {
				_ = os.WriteFile(filepath.Join(dir, "level"), []byte(level), 0o644)
			}
			if typ != "" {
				_ = os.WriteFile(filepath.Join(dir, "type"), []byte(typ), 0o644)
			}
			if size != "" {
				_ = os.WriteFile(filepath.Join(dir, "size"), []byte(size), 0o644)
			}
			if line != "" {
				_ = os.WriteFile(filepath.Join(dir, "coherency_line_size"), []byte(line), 0o644)
			}
		}

		createIndex("index0", "1", "Data", "64K", "64")
		createIndex("index1", "1", "Instruction", "128K", "64")
		createIndex("index2", "2", "Unified", "1M", "64")
		createIndex("index3", "3", "Unified", "32M", "64")

		env, err := ParseLinuxSysfsCPUTopology(tmpDir, "qwen3.8-native")
		if err != nil {
			t.Fatalf("ParseLinuxSysfsCPUTopology failed: %v", err)
		}

		if env.ModelEnvelope != "qwen3.8-native" {
			t.Fatalf("expected model envelope qwen3.8-native, got %s", env.ModelEnvelope)
		}

		wantSizes := map[HostCPUTopologyLevel]int64{
			LevelL1d: 64 * 1024,
			LevelL1i: 128 * 1024,
			LevelL2:  1024 * 1024,
			LevelL3:  32 * 1024 * 1024,
		}

		for _, row := range env.Rows {
			want, ok := wantSizes[row.Level]
			if !ok {
				t.Fatalf("unexpected level: %s", row.Level)
			}
			if row.Status != "known" {
				t.Fatalf("level %s expected known, got %s", row.Level, row.Status)
			}
			if row.SizeBytes != want {
				t.Fatalf("level %s size %d != want %d", row.Level, row.SizeBytes, want)
			}
			if row.LineSize != 64 {
				t.Fatalf("level %s line size %d != want 64", row.Level, row.LineSize)
			}
		}
	})

	t.Run("linux_sysfs_missing_data_is_unknown_never_zero", func(t *testing.T) {
		tmpDir := t.TempDir()

		dir0 := filepath.Join(tmpDir, "index0")
		_ = os.MkdirAll(dir0, 0o755)
		_ = os.WriteFile(filepath.Join(dir0, "level"), []byte("1"), 0o644)
		_ = os.WriteFile(filepath.Join(dir0, "type"), []byte("Data"), 0o644)

		env, err := ParseLinuxSysfsCPUTopology(tmpDir, "qwen3.8-native")
		if err != nil {
			t.Fatalf("ParseLinuxSysfsCPUTopology failed: %v", err)
		}

		for _, row := range env.Rows {
			if row.SizeBytes == 0 {
				t.Fatalf("row %s reported 0 bytes; missing data must be -1", row.Level)
			}
			if row.Status != "unknown" || row.SizeBytes != -1 {
				t.Fatalf("row %s expected unknown/-1, got status=%s, bytes=%d", row.Level, row.Status, row.SizeBytes)
			}
		}
	})

	t.Run("live_host_adapter_discovery", func(t *testing.T) {
		env := DiscoverHostCPUTopology("qwen3.8-native")
		if env.ModelEnvelope != "qwen3.8-native" {
			t.Fatalf("model envelope = %s, want qwen3.8-native", env.ModelEnvelope)
		}
		if len(env.Rows) != 4 {
			t.Fatalf("expected 4 rows, got %d", len(env.Rows))
		}

		for _, row := range env.Rows {
			if row.Status == "known" {
				if row.SizeBytes <= 0 {
					t.Fatalf("known row %s has non-positive size %d", row.Level, row.SizeBytes)
				}
			} else {
				if row.SizeBytes != -1 {
					t.Fatalf("unknown row %s has size %d, want -1", row.Level, row.SizeBytes)
				}
			}
		}
	})
}
