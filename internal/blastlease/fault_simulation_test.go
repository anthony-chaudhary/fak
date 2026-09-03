package blastlease_test

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/blastlease"
	"github.com/anthony-chaudhary/fak/internal/blastradius"
)

func TestFaultSimulation(t *testing.T) {
	// 1. Injected mid-line truncation (e.g. {"lane": "foo", "tree_ without closing brace).
	// Verify clean error with line number, no panic.
	t.Run("mid-line truncation", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "truncated.jsonl")
		content := `{"lane": "foo", "tree_`
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}

		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("Read panicked on truncated JSON: %v", r)
			}
		}()

		leases, err := blastlease.Read(path)
		if err == nil {
			t.Fatalf("Read(%q) expected error on truncated JSON, got nil (leases: %+v)", path, leases)
		}
		expectedLineRef := fmt.Sprintf("%s line 1:", path)
		if !strings.Contains(err.Error(), expectedLineRef) {
			t.Fatalf("Read error %q does not contain expected line reference %q", err.Error(), expectedLineRef)
		}
	})

	// 2. Injected invalid/corrupted JSON types (e.g. {"lane": 12345, "tree_globs": "not-an-array"}).
	// Verify clean error with line number, no panic.
	t.Run("invalid corrupted JSON types", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "invalid_types.jsonl")
		content := `{"lane": 12345, "tree_globs": "not-an-array"}` + "\n"
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}

		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("Read panicked on invalid JSON types: %v", r)
			}
		}()

		leases, err := blastlease.Read(path)
		if err == nil {
			t.Fatalf("Read(%q) expected error on invalid JSON types, got nil (leases: %+v)", path, leases)
		}
		expectedLineRef := fmt.Sprintf("%s line 1:", path)
		if !strings.Contains(err.Error(), expectedLineRef) {
			t.Fatalf("Read error %q does not contain expected line reference %q", err.Error(), expectedLineRef)
		}
	})

	// 3. Injected malformed UTF-8 / binary garbage. Verify clean rejection with line number.
	t.Run("malformed UTF-8 and binary garbage", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "binary_garbage.jsonl")
		garbage := []byte{0x80, 0x81, 0xFF, 0xFE, 0x00, 0x01, 0x02, 0x7F, '\n'}
		if err := os.WriteFile(path, garbage, 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}

		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("Read panicked on binary garbage: %v", r)
			}
		}()

		leases, err := blastlease.Read(path)
		if err == nil {
			t.Fatalf("Read(%q) expected error on binary garbage, got nil (leases: %+v)", path, leases)
		}
		expectedLineRef := fmt.Sprintf("%s line 1:", path)
		if !strings.Contains(err.Error(), expectedLineRef) {
			t.Fatalf("Read error %q does not contain expected line reference %q", err.Error(), expectedLineRef)
		}
	})

	// 4. Injected oversized lines (e.g. 512KB line). Verify it parses or returns clean error without memory exhaustion.
	t.Run("oversized lines", func(t *testing.T) {
		dir := t.TempDir()

		t.Run("valid 512KB line parses cleanly without memory exhaustion", func(t *testing.T) {
			validPath := filepath.Join(dir, "oversized_valid.jsonl")
			globs := make([]string, 0, 15000)
			for i := 0; i < 15000; i++ {
				globs = append(globs, fmt.Sprintf("internal/subsystem/module_%06d/**", i))
			}
			leaseObj := blastradius.Lease{
				Lane:      "oversized-lane",
				TreeGlobs: globs,
			}
			data, err := json.Marshal(leaseObj)
			if err != nil {
				t.Fatalf("Marshal: %v", err)
			}
			if len(data) < 512*1024 {
				t.Fatalf("serialized line size %d is less than 512KB", len(data))
			}
			if err := os.WriteFile(validPath, append(data, '\n'), 0o644); err != nil {
				t.Fatalf("WriteFile: %v", err)
			}

			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("Read panicked on valid oversized line: %v", r)
				}
			}()

			got, err := blastlease.Read(validPath)
			if err != nil {
				t.Fatalf("Read(%q) failed on valid oversized line: %v", validPath, err)
			}
			if len(got) != 1 {
				t.Fatalf("Read(%q) returned %d leases, want 1", validPath, len(got))
			}
			if got[0].Lane != "oversized-lane" {
				t.Errorf("got[0].Lane = %q, want %q", got[0].Lane, "oversized-lane")
			}
			if len(got[0].TreeGlobs) != len(globs) {
				t.Errorf("got[0].TreeGlobs len = %d, want %d", len(got[0].TreeGlobs), len(globs))
			}
		})

		t.Run("corrupted 512KB line returns clean line error without memory exhaustion", func(t *testing.T) {
			corruptPath := filepath.Join(dir, "oversized_corrupt.jsonl")
			corruptContent := `{"lane": "huge", "tree_globs": [` + strings.Repeat(`"internal/foo/**", `, 30000)
			if len(corruptContent) < 512*1024 {
				t.Fatalf("corrupt content size %d is less than 512KB", len(corruptContent))
			}
			if err := os.WriteFile(corruptPath, []byte(corruptContent), 0o644); err != nil {
				t.Fatalf("WriteFile: %v", err)
			}

			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("Read panicked on corrupt oversized line: %v", r)
				}
			}()

			leases, err := blastlease.Read(corruptPath)
			if err == nil {
				t.Fatalf("Read(%q) expected error on corrupt oversized line, got nil (leases: %+v)", corruptPath, leases)
			}
			expectedLineRef := fmt.Sprintf("%s line 1:", corruptPath)
			if !strings.Contains(err.Error(), expectedLineRef) {
				t.Fatalf("Read error %q does not contain expected line reference %q", err.Error(), expectedLineRef)
			}
		})
	})

	// 5. Multi-line fixture with corrupt line interspersed between valid lines
	// (e.g. Line 1 valid, Line 2 corrupt, Line 3 valid) - verify it fails on Line 2 with error referencing line 2.
	t.Run("corrupt line interspersed between valid lines fails on line 2", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "interspersed_corrupt.jsonl")
		lines := []string{
			`{"lane": "line-1-valid", "tree_globs": ["internal/auth/**"]}`,
			`{"lane": "line-2-corrupt", "tree_globs": NOT_VALID_JSON`,
			`{"lane": "line-3-valid", "tree_globs": ["internal/gateway/**"]}`,
		}
		content := strings.Join(lines, "\n") + "\n"
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}

		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("Read panicked on interspersed corrupt line: %v", r)
			}
		}()

		leases, err := blastlease.Read(path)
		if err == nil {
			t.Fatalf("Read(%q) expected error on line 2, got nil (leases: %+v)", path, leases)
		}
		expectedLineRef := fmt.Sprintf("%s line 2:", path)
		if !strings.Contains(err.Error(), expectedLineRef) {
			t.Fatalf("Read error %q does not reference line 2 (expected line ref %q)", err.Error(), expectedLineRef)
		}
	})
}
