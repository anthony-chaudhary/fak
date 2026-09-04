package toolbound

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

// TestSpine drives the generated leaf's real surface end to end. Keep this
// representative path working while the proof envelope expands around it.
func TestSpine(t *testing.T) {
	if !Ready() {
		t.Fatal("generated leaf spine did not reach Ready")
	}
}

func TestWithinBounds(t *testing.T) {
	t.Run("short single line", func(t *testing.T) {
		b := New(Options{MaxLines: 10, MaxBytes: 100})
		raw := "hello world"
		out, err := b.Bound(raw)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if out.Truncated {
			t.Fatalf("expected Truncated false, got true")
		}
		if out.CompletePath != "" {
			t.Fatalf("expected empty CompletePath, got %q", out.CompletePath)
		}
		if out.Preview != raw {
			t.Fatalf("expected Preview %q, got %q", raw, out.Preview)
		}
		if out.OriginalBytes != len(raw) {
			t.Fatalf("expected OriginalBytes %d, got %d", len(raw), out.OriginalBytes)
		}
		if out.OriginalLines != 1 {
			t.Fatalf("expected OriginalLines 1, got %d", out.OriginalLines)
		}
	})

	t.Run("multiple lines within limits", func(t *testing.T) {
		b := New(Options{MaxLines: 5, MaxBytes: 100})
		raw := "line 1\nline 2\nline 3\n"
		out, err := b.Bound(raw)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if out.Truncated {
			t.Fatalf("expected Truncated false, got true")
		}
		if out.CompletePath != "" {
			t.Fatalf("expected empty CompletePath, got %q", out.CompletePath)
		}
		if out.Preview != raw {
			t.Fatalf("expected Preview %q, got %q", raw, out.Preview)
		}
		if out.OriginalLines != 3 {
			t.Fatalf("expected OriginalLines 3, got %d", out.OriginalLines)
		}
	})

	t.Run("multiple lines no trailing newline", func(t *testing.T) {
		b := New(Options{MaxLines: 3, MaxBytes: 50})
		raw := "alpha\nbeta\ngamma"
		out, err := b.Bound(raw)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if out.Truncated {
			t.Fatalf("expected Truncated false, got true")
		}
		if out.CompletePath != "" {
			t.Fatalf("expected empty CompletePath, got %q", out.CompletePath)
		}
		if out.Preview != raw {
			t.Fatalf("expected Preview %q, got %q", raw, out.Preview)
		}
		if out.OriginalLines != 3 {
			t.Fatalf("expected OriginalLines 3, got %d", out.OriginalLines)
		}
	})

	t.Run("exact limits", func(t *testing.T) {
		raw := "12345"
		b := New(Options{MaxLines: 1, MaxBytes: 5})
		out, err := b.Bound(raw)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if out.Truncated {
			t.Fatalf("expected Truncated false at exact bounds")
		}
		if out.Preview != raw {
			t.Fatalf("expected preview %q, got %q", raw, out.Preview)
		}
	})
}

func TestExceedingMaxBytes(t *testing.T) {
	dir := t.TempDir()
	b := New(Options{
		MaxBytes:    20,
		SpillDir:    dir,
		SpillPrefix: "fak-tool-output-",
	})

	raw := "0123456789abcdefghijklmnopqrstuvwxyz"
	out, err := b.Bound(raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !out.Truncated {
		t.Fatal("expected Truncated true")
	}
	if out.OriginalBytes != len(raw) {
		t.Fatalf("expected OriginalBytes %d, got %d", len(raw), out.OriginalBytes)
	}
	if out.CompletePath == "" {
		t.Fatal("expected non-empty CompletePath")
	}

	// Verify spill file contents match the exact original bytes.
	diskBytes, err := os.ReadFile(out.CompletePath)
	if err != nil {
		t.Fatalf("failed reading spill file: %v", err)
	}
	if string(diskBytes) != raw {
		t.Fatalf("spill file content mismatch: got %q, want %q", string(diskBytes), raw)
	}

	// Verify preview contains truncation notice and spill path.
	if !strings.Contains(out.Preview, "output truncated") {
		t.Fatalf("preview missing truncation notice: %q", out.Preview)
	}
	if !strings.Contains(out.Preview, out.CompletePath) {
		t.Fatalf("preview missing spill path: %q", out.Preview)
	}
	if !strings.Contains(out.Preview, "spilled") {
		t.Fatalf("preview missing spilled description: %q", out.Preview)
	}

	// Check that head and tail are preserved.
	headExpected := "0123456789"
	tailExpected := "qrstuvwxyz"
	if !strings.HasPrefix(out.Preview, headExpected) {
		t.Fatalf("preview missing expected head %q: %q", headExpected, out.Preview)
	}
	if !strings.HasSuffix(out.Preview, tailExpected) {
		t.Fatalf("preview missing expected tail %q: %q", tailExpected, out.Preview)
	}
}

func TestExceedingMaxLines(t *testing.T) {
	dir := t.TempDir()
	b := New(Options{
		MaxLines: 4,
		SpillDir: dir,
	})

	raw := "line 0\nline 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8\nline 9\n"
	out, err := b.Bound(raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !out.Truncated {
		t.Fatal("expected Truncated true")
	}
	if out.OriginalLines != 10 {
		t.Fatalf("expected OriginalLines 10, got %d", out.OriginalLines)
	}
	if out.CompletePath == "" {
		t.Fatal("expected non-empty CompletePath")
	}

	// Verify spill file contains exact lines.
	diskBytes, err := os.ReadFile(out.CompletePath)
	if err != nil {
		t.Fatalf("failed reading spill file: %v", err)
	}
	if string(diskBytes) != raw {
		t.Fatalf("spill file content mismatch: got %q, want %q", string(diskBytes), raw)
	}

	// Preview preserves head lines (line 0, line 1) and tail lines (line 8, line 9).
	headLines := "line 0\nline 1\n"
	tailLines := "line 8\nline 9\n"
	if !strings.HasPrefix(out.Preview, headLines) {
		t.Fatalf("preview missing expected head lines: %q", out.Preview)
	}
	if !strings.HasSuffix(out.Preview, tailLines) {
		t.Fatalf("preview missing expected tail lines: %q", out.Preview)
	}

	if !strings.Contains(out.Preview, "output truncated") {
		t.Fatalf("preview missing truncation notice: %q", out.Preview)
	}
	if !strings.Contains(out.Preview, out.CompletePath) {
		t.Fatalf("preview missing spill path: %q", out.Preview)
	}
}

func TestExceedingBothLimits(t *testing.T) {
	dir := t.TempDir()
	b := New(Options{
		MaxLines: 4,
		MaxBytes: 30,
		SpillDir: dir,
	})

	raw := "long line number 0\nlong line number 1\nlong line number 2\nlong line number 3\nlong line number 4\nlong line number 5\n"
	out, err := b.Bound(raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !out.Truncated {
		t.Fatal("expected Truncated true")
	}
	diskBytes, err := os.ReadFile(out.CompletePath)
	if err != nil {
		t.Fatalf("failed reading spill file: %v", err)
	}
	if string(diskBytes) != raw {
		t.Fatalf("spill file content mismatch: got %q, want %q", string(diskBytes), raw)
	}
	if !strings.Contains(out.Preview, "output truncated") {
		t.Fatalf("preview missing truncation notice: %q", out.Preview)
	}
}

func TestCleanupSpillFiles(t *testing.T) {
	dir := t.TempDir()
	prefix := "test-tool-output-"
	b := New(Options{
		MaxBytes:    10,
		SpillDir:    dir,
		SpillPrefix: prefix,
	})

	out1, err := b.Bound("large text number one to spill")
	if err != nil {
		t.Fatalf("failed bounding out1: %v", err)
	}
	out2, err := b.Bound("large text number two to spill")
	if err != nil {
		t.Fatalf("failed bounding out2: %v", err)
	}
	out3, err := b.Bound("large text number three to spill")
	if err != nil {
		t.Fatalf("failed bounding out3: %v", err)
	}

	// Create an unrelated file that should not be touched.
	unrelatedPath := filepath.Join(dir, "other-file.txt")
	if err := os.WriteFile(unrelatedPath, []byte("preserve me"), 0644); err != nil {
		t.Fatalf("failed creating unrelated file: %v", err)
	}

	// Simulate past timestamps for out1, out2, and unrelatedPath.
	past := time.Now().Add(-2 * time.Hour)
	if err := os.Chtimes(out1.CompletePath, past, past); err != nil {
		t.Fatalf("failed setting out1 time: %v", err)
	}
	if err := os.Chtimes(out2.CompletePath, past, past); err != nil {
		t.Fatalf("failed setting out2 time: %v", err)
	}
	if err := os.Chtimes(unrelatedPath, past, past); err != nil {
		t.Fatalf("failed setting unrelated time: %v", err)
	}

	// Cleanup files older than 1 hour.
	deletedCount, err := b.CleanupSpillFiles(1 * time.Hour)
	if err != nil {
		t.Fatalf("cleanup failed: %v", err)
	}
	if deletedCount != 2 {
		t.Fatalf("expected 2 deleted files, got %d", deletedCount)
	}

	// Verify out1 and out2 are gone.
	if _, err := os.Stat(out1.CompletePath); !os.IsNotExist(err) {
		t.Fatalf("expected out1 to be removed, err was: %v", err)
	}
	if _, err := os.Stat(out2.CompletePath); !os.IsNotExist(err) {
		t.Fatalf("expected out2 to be removed, err was: %v", err)
	}

	// Verify out3 is preserved.
	if _, err := os.Stat(out3.CompletePath); err != nil {
		t.Fatalf("expected out3 to remain, got err: %v", err)
	}

	// Verify unrelated file is preserved even though modtime was in the past.
	if _, err := os.Stat(unrelatedPath); err != nil {
		t.Fatalf("expected unrelated file to remain, got err: %v", err)
	}
}

func TestEdgeCases(t *testing.T) {
	t.Run("empty string", func(t *testing.T) {
		b := New(Options{MaxLines: 10, MaxBytes: 50})
		out, err := b.Bound("")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if out.Truncated {
			t.Fatal("expected Truncated false for empty string")
		}
		if out.Preview != "" {
			t.Fatalf("expected empty preview, got %q", out.Preview)
		}
		if out.OriginalBytes != 0 {
			t.Fatalf("expected 0 bytes, got %d", out.OriginalBytes)
		}
		if out.OriginalLines != 0 {
			t.Fatalf("expected 0 lines, got %d", out.OriginalLines)
		}
		if out.CompletePath != "" {
			t.Fatalf("expected empty CompletePath, got %q", out.CompletePath)
		}
	})

	t.Run("zero limits no-op", func(t *testing.T) {
		raw := "line 1\nline 2\nline 3\n"
		b := New(Options{MaxLines: 0, MaxBytes: 0})
		out, err := b.Bound(raw)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if out.Truncated {
			t.Fatal("expected Truncated false when limits are zero")
		}
		if out.Preview != raw {
			t.Fatalf("expected Preview == raw")
		}
		if out.CompletePath != "" {
			t.Fatalf("expected empty CompletePath, got %q", out.CompletePath)
		}
		if out.OriginalBytes != len(raw) {
			t.Fatalf("expected %d bytes, got %d", len(raw), out.OriginalBytes)
		}
		if out.OriginalLines != 3 {
			t.Fatalf("expected 3 lines, got %d", out.OriginalLines)
		}
	})

	t.Run("utf8 multibyte bounding", func(t *testing.T) {
		raw := "こんにちは世界！Hello World! 1234567890"
		b := New(Options{MaxBytes: 25, SpillDir: t.TempDir()})
		out, err := b.Bound(raw)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !out.Truncated {
			t.Fatal("expected Truncated true")
		}
		if !utf8.ValidString(out.Preview) {
			t.Fatal("preview is not valid UTF-8")
		}
		diskBytes, err := os.ReadFile(out.CompletePath)
		if err != nil {
			t.Fatalf("failed reading spill file: %v", err)
		}
		if string(diskBytes) != raw {
			t.Fatal("disk bytes did not match original")
		}
	})

	t.Run("single line limit 1", func(t *testing.T) {
		raw := "first\nsecond\nthird\n"
		b := New(Options{MaxLines: 1, SpillDir: t.TempDir()})
		out, err := b.Bound(raw)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !out.Truncated {
			t.Fatal("expected Truncated true")
		}
		if !strings.HasPrefix(out.Preview, "first\n") {
			t.Fatalf("expected first line in preview, got %q", out.Preview)
		}
	})
}

func TestInvalidSpillDir(t *testing.T) {
	// Create a regular file to use as an invalid directory path.
	filePath := filepath.Join(t.TempDir(), "not_a_directory")
	if err := os.WriteFile(filePath, []byte("file content"), 0644); err != nil {
		t.Fatalf("failed creating file: %v", err)
	}

	b := New(Options{
		MaxBytes: 5,
		SpillDir: filePath,
	})

	_, err := b.Bound("this text exceeds the five byte limit")
	if err == nil {
		t.Fatal("expected error when SpillDir is a file, got nil")
	}

	_, err = b.CleanupSpillFiles(time.Hour)
	if err == nil {
		t.Fatal("expected error when CleanupSpillFiles runs on a file path, got nil")
	}
}

func TestSpillInlineImages(t *testing.T) {
	tempDir := t.TempDir()
	// 1x1 red PNG pixel in base64
	pngB64 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
	raw := "Result contains chart: data:image/png;base64," + pngB64 + " and conclusion text."

	transformed, images, err := SpillInlineImages(raw, tempDir, "test-spill-")
	if err != nil {
		t.Fatalf("SpillInlineImages failed: %v", err)
	}
	if len(images) != 1 {
		t.Fatalf("expected 1 spilled image, got %d", len(images))
	}
	if strings.Contains(transformed, pngB64) {
		t.Fatalf("transformed text should not contain raw base64 data")
	}
	if !strings.Contains(transformed, "[image: ") || !strings.Contains(transformed, "png") {
		t.Fatalf("transformed text missing reference tag: %q", transformed)
	}

	// Verify image file on disk
	content, err := os.ReadFile(images[0])
	if err != nil {
		t.Fatalf("failed reading spilled image file: %v", err)
	}
	if len(content) == 0 {
		t.Fatalf("spilled image file is empty")
	}

	// Test Bound with SpillImages option
	b := New(Options{
		MaxLines:    100,
		MaxBytes:    10000,
		SpillDir:    tempDir,
		SpillImages: true,
	})
	out, err := b.Bound(raw)
	if err != nil {
		t.Fatalf("Bound with SpillImages failed: %v", err)
	}
	if len(out.SpilledImages) != 1 {
		t.Fatalf("expected 1 SpilledImages in BoundedOutput, got %d", len(out.SpilledImages))
	}
	if !out.Truncated {
		t.Fatalf("expected Truncated=true when image was spilled")
	}
}
