package codetools

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
)

// TestGrepFileExceedingMaxReadBytesTruncation verifies that grep on a file
// exceeding MaxReadBytes truncates properly without error and matches lines up to the limit.
func TestGrepFileExceedingMaxReadBytesTruncation(t *testing.T) {
	dir := t.TempDir()
	const maxRead = 512
	ts, err := New(Config{
		Root: dir,
		Limits: Limits{
			MaxReadBytes: maxRead,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// Construct file exceeding MaxReadBytes:
	// Line 1: match within limit (< 512 bytes)
	// Lines 2-100: filler padding exceeding 512 bytes
	// Last line: match beyond limit (> 512 bytes)
	line1 := "line1_needle: early match\n"
	padding := strings.Repeat("padding line exceeding read buffer\n", 40) // ~1400 bytes
	lineBeyond := "line_beyond_needle: late match beyond max read bytes\n"
	content := line1 + padding + lineBeyond

	fileName := "large_stream.txt"
	mustWrite(t, filepath.Join(dir, fileName), content)

	ctx := context.Background()

	// 1. Search for match within MaxReadBytes
	out, isErr := ts.grep(ctx, argsOf(t, GrepArgs{Pattern: "early match"}))
	if isErr {
		t.Fatalf("grep for early match failed: %s", string(out))
	}
	res := decodeResult(t, out)
	if res["truncated"] != true {
		t.Fatalf("truncated = %v, want true", res["truncated"])
	}
	if res["truncation_reason"] != "file_size" {
		t.Fatalf("truncation_reason = %v, want file_size", res["truncation_reason"])
	}
	if res["match_count"] != float64(1) {
		t.Fatalf("match_count = %v, want 1", res["match_count"])
	}
	matches := res["matches"].([]any)
	if len(matches) != 1 {
		t.Fatalf("len(matches) = %d, want 1", len(matches))
	}
	m0 := matches[0].(map[string]any)
	if m0["line"] != float64(1) {
		t.Fatalf("match line = %v, want 1", m0["line"])
	}
	if m0["file"] != fileName {
		t.Fatalf("match file = %v, want %s", m0["file"], fileName)
	}

	// 2. Search for match beyond MaxReadBytes
	out, isErr = ts.grep(ctx, argsOf(t, GrepArgs{Pattern: "late match"}))
	if isErr {
		t.Fatalf("grep for late match failed: %s", string(out))
	}
	res = decodeResult(t, out)
	if res["truncated"] != true {
		t.Fatalf("truncated = %v, want true", res["truncated"])
	}
	if res["truncation_reason"] != "file_size" {
		t.Fatalf("truncation_reason = %v, want file_size", res["truncation_reason"])
	}
	if res["match_count"] != float64(0) {
		t.Fatalf("match_count = %v, want 0 (pattern is beyond limit)", res["match_count"])
	}

	// 3. File within MaxReadBytes should not be truncated
	smallFile := "small.txt"
	mustWrite(t, filepath.Join(dir, smallFile), "small_needle: all fit\n")
	out, isErr = ts.grep(ctx, argsOf(t, GrepArgs{Path: smallFile, Pattern: "small_needle"}))
	if isErr {
		t.Fatalf("grep on small file failed: %s", string(out))
	}
	res = decodeResult(t, out)
	if res["truncated"] != false {
		t.Fatalf("truncated = %v, want false", res["truncated"])
	}
	if res["match_count"] != float64(1) {
		t.Fatalf("match_count = %v, want 1", res["match_count"])
	}
}
