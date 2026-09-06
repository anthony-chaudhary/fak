package codetools

import (
	"context"
	"path/filepath"
	"reflect"
	"sort"
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

// TestGlobMidPathRecursiveGlobbing verifies that ** wildcards anywhere in the pattern
// (e.g. src/**/*.go or internal/**/test_*.go) match deeply nested files.
func TestGlobMidPathRecursiveGlobbing(t *testing.T) {
	dir := t.TempDir()
	ts, err := New(Config{Root: dir})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// Set up nested directory structure:
	// src/root.go
	// src/pkg/lib.go
	// src/pkg/deep/nested.go
	// src/pkg/deep/doc.txt
	// internal/test_root.go
	// internal/foo.go
	// internal/codetools/test_search.go
	// internal/codetools/search.go
	// internal/codetools/sub/deep/test_nested.go
	// internal/codetools/sub/deep/nested.go
	// cmd/test_cmd.go
	// cmd/main.go
	mustWrite(t, filepath.Join(dir, "src/root.go"), "package src\n")
	mustWrite(t, filepath.Join(dir, "src/pkg/lib.go"), "package pkg\n")
	mustWrite(t, filepath.Join(dir, "src/pkg/deep/nested.go"), "package deep\n")
	mustWrite(t, filepath.Join(dir, "src/pkg/deep/doc.txt"), "documentation\n")
	mustWrite(t, filepath.Join(dir, "internal/test_root.go"), "package internal\n")
	mustWrite(t, filepath.Join(dir, "internal/foo.go"), "package internal\n")
	mustWrite(t, filepath.Join(dir, "internal/codetools/test_search.go"), "package codetools\n")
	mustWrite(t, filepath.Join(dir, "internal/codetools/search.go"), "package codetools\n")
	mustWrite(t, filepath.Join(dir, "internal/codetools/sub/deep/test_nested.go"), "package deep\n")
	mustWrite(t, filepath.Join(dir, "internal/codetools/sub/deep/nested.go"), "package deep\n")
	mustWrite(t, filepath.Join(dir, "cmd/test_cmd.go"), "package cmd\n")
	mustWrite(t, filepath.Join(dir, "cmd/main.go"), "package main\n")

	ctx := context.Background()

	// 1. src/**/*.go matches root, shallow, and deeply nested .go files under src
	t.Run("src/**/*.go", func(t *testing.T) {
		out, isErr := ts.glob(ctx, argsOf(t, GlobArgs{Pattern: "src/**/*.go"}))
		if isErr {
			t.Fatalf("Glob failed: %s", string(out))
		}
		res := decodeResult(t, out)
		if res["count"] != float64(3) {
			t.Fatalf("count = %v, want 3: %s", res["count"], string(out))
		}
		files := decodeResultFiles(res)
		want := []string{"src/pkg/deep/nested.go", "src/pkg/lib.go", "src/root.go"}
		assertFileSetsEqual(t, files, want)
	})

	// 2. internal/**/test_*.go matches test files across various nesting depths under internal
	t.Run("internal/**/test_*.go", func(t *testing.T) {
		out, isErr := ts.glob(ctx, argsOf(t, GlobArgs{Pattern: "internal/**/test_*.go"}))
		if isErr {
			t.Fatalf("Glob failed: %s", string(out))
		}
		res := decodeResult(t, out)
		if res["count"] != float64(3) {
			t.Fatalf("count = %v, want 3: %s", res["count"], string(out))
		}
		files := decodeResultFiles(res)
		want := []string{
			"internal/codetools/sub/deep/test_nested.go",
			"internal/codetools/test_search.go",
			"internal/test_root.go",
		}
		assertFileSetsEqual(t, files, want)
	})

	// 3. Multi-segment mid-path glob: internal/**/deep/*.go
	t.Run("internal/**/deep/*.go", func(t *testing.T) {
		out, isErr := ts.glob(ctx, argsOf(t, GlobArgs{Pattern: "internal/**/deep/*.go"}))
		if isErr {
			t.Fatalf("Glob failed: %s", string(out))
		}
		res := decodeResult(t, out)
		if res["count"] != float64(2) {
			t.Fatalf("count = %v, want 2: %s", res["count"], string(out))
		}
		files := decodeResultFiles(res)
		want := []string{
			"internal/codetools/sub/deep/nested.go",
			"internal/codetools/sub/deep/test_nested.go",
		}
		assertFileSetsEqual(t, files, want)
	})

	// 4. Scoped search with Path: Glob with Path="src" and Pattern="**/*.go"
	t.Run("scoped Path with **/*.go", func(t *testing.T) {
		out, isErr := ts.glob(ctx, argsOf(t, GlobArgs{Path: "src", Pattern: "**/*.go"}))
		if isErr {
			t.Fatalf("Glob failed: %s", string(out))
		}
		res := decodeResult(t, out)
		if res["count"] != float64(3) {
			t.Fatalf("count = %v, want 3: %s", res["count"], string(out))
		}
	})

	// 5. Direct unit test of matchGlob helper with various patterns
	t.Run("matchGlob helper edge cases", func(t *testing.T) {
		cases := []struct {
			pattern string
			path    string
			want    bool
		}{
			{"src/**/*.go", "src/a.go", true},
			{"src/**/*.go", "src/a/b.go", true},
			{"src/**/*.go", "src/a/b/c/d.go", true},
			{"src/**/*.go", "other/a.go", false},
			{"src/**/*.go", "src/a.txt", false},
			{"internal/**/test_*.go", "internal/test_foo.go", true},
			{"internal/**/test_*.go", "internal/sub/test_foo.go", true},
			{"internal/**/test_*.go", "internal/a/b/c/test_foo.go", true},
			{"internal/**/test_*.go", "internal/foo.go", false},
			{"internal/**/test_*.go", "cmd/test_foo.go", false},
			{"**", "any/file.go", true},
			{"src/**", "src/a/b/c.go", true},
			{"*.go", "a.go", true},
			{"*.go", "sub/a.go", false},
		}
		for _, tc := range cases {
			got, err := matchGlob(tc.pattern, tc.path)
			if err != nil {
				t.Errorf("matchGlob(%q, %q) unexpected error: %v", tc.pattern, tc.path, err)
				continue
			}
			if got != tc.want {
				t.Errorf("matchGlob(%q, %q) = %v, want %v", tc.pattern, tc.path, got, tc.want)
			}
		}
	})
}

func decodeResultFiles(res map[string]any) []string {
	raw, ok := res["files"].([]any)
	if !ok {
		return nil
	}
	files := make([]string, len(raw))
	for i, v := range raw {
		files[i] = v.(string)
	}
	sort.Strings(files)
	return files
}

func assertFileSetsEqual(t *testing.T, got, want []string) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("files = %v, want %v", got, want)
	}
}
