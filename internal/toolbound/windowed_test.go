package toolbound

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"
)

func createTestFile(t *testing.T, dir, name, content string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatalf("mkdir failed: %v", err)
	}
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("write file failed: %v", err)
	}
	return path
}

// TestWindowedFile1000Lines verifies opening a 1000-line file displays lines 1-100.
func TestWindowedFile1000Lines(t *testing.T) {
	dir := t.TempDir()
	var sb strings.Builder
	for i := 1; i <= 1000; i++ {
		fmt.Fprintf(&sb, "line %d\n", i)
	}
	filePath := createTestFile(t, dir, "thousand.txt", sb.String())

	w := NewWindowedFileReader(100)

	lines, err := w.Open(filePath, 1, 100)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	if len(lines) != 100 {
		t.Fatalf("expected 100 lines, got %d", len(lines))
	}

	if lines[0] != "1: line 1" {
		t.Errorf("expected first line '1: line 1', got %q", lines[0])
	}
	if lines[99] != "100: line 100" {
		t.Errorf("expected 100th line '100: line 100', got %q", lines[99])
	}

	// Verify struct state
	if w.FilePath != filePath {
		t.Errorf("expected FilePath %q, got %q", filePath, w.FilePath)
	}
	if w.CurrentLine != 1 {
		t.Errorf("expected CurrentLine 1, got %d", w.CurrentLine)
	}
	if w.WindowSize != 100 {
		t.Errorf("expected WindowSize 100, got %d", w.WindowSize)
	}
	if len(w.Lines) != 1000 {
		t.Errorf("expected 1000 Lines, got %d", len(w.Lines))
	}
	if w.TotalLines != 1000 {
		t.Errorf("expected TotalLines 1000, got %d", w.TotalLines)
	}

	// Verify Status
	status := w.Status()
	if !strings.Contains(status, filePath) || !strings.Contains(status, "lines 1-100 of 1000") {
		t.Errorf("unexpected Status string: %q", status)
	}

	// Verify GetWindow matches without cursor mutation
	activeWindow := w.GetWindow()
	if len(activeWindow) != 100 {
		t.Fatalf("expected GetWindow to return 100 lines, got %d", len(activeWindow))
	}
	if activeWindow[0] != lines[0] || activeWindow[99] != lines[99] {
		t.Errorf("GetWindow output does not match Open output")
	}
	if w.CurrentLine != 1 {
		t.Errorf("GetWindow must not move cursor, CurrentLine is %d", w.CurrentLine)
	}
}

// TestWindowedFileScrollDown verifies ScrollDown advances window deterministically.
func TestWindowedFileScrollDown(t *testing.T) {
	dir := t.TempDir()
	var sb strings.Builder
	for i := 1; i <= 1000; i++ {
		fmt.Fprintf(&sb, "line %d\n", i)
	}
	filePath := createTestFile(t, dir, "scroll.txt", sb.String())

	w := NewWindowedFileReader(100)
	lines, err := w.Open(filePath, 1, 100)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	if len(lines) != 100 || w.CurrentLine != 1 {
		t.Fatalf("initial state invalid")
	}

	// ScrollDown(50) -> advances by 50 lines (starts at 51, lines 51-150)
	lines, err = w.ScrollDown(50)
	if err != nil {
		t.Fatalf("ScrollDown(50) failed: %v", err)
	}
	if w.CurrentLine != 51 {
		t.Errorf("expected CurrentLine 51, got %d", w.CurrentLine)
	}
	if len(lines) != 100 {
		t.Fatalf("expected 100 lines, got %d", len(lines))
	}
	if lines[0] != "51: line 51" || lines[99] != "150: line 150" {
		t.Errorf("unexpected lines: first=%q, last=%q", lines[0], lines[99])
	}

	// ScrollDown(0) -> defaults to WindowSize (100 lines), advances from 51 to 151
	lines, err = w.ScrollDown(0)
	if err != nil {
		t.Fatalf("ScrollDown(0) failed: %v", err)
	}
	if w.CurrentLine != 151 {
		t.Errorf("expected CurrentLine 151 after ScrollDown(0), got %d", w.CurrentLine)
	}
	if lines[0] != "151: line 151" || lines[99] != "250: line 250" {
		t.Errorf("unexpected lines after default ScrollDown: first=%q, last=%q", lines[0], lines[99])
	}

	// ScrollDown(-10) -> negative n defaults to WindowSize (100 lines), advances from 151 to 251
	lines, err = w.ScrollDown(-10)
	if err != nil {
		t.Fatalf("ScrollDown(-10) failed: %v", err)
	}
	if w.CurrentLine != 251 {
		t.Errorf("expected CurrentLine 251 after ScrollDown(-10), got %d", w.CurrentLine)
	}
	if lines[0] != "251: line 251" || lines[99] != "350: line 350" {
		t.Errorf("unexpected lines after negative ScrollDown: first=%q, last=%q", lines[0], lines[99])
	}

	// ScrollDown past the end clamps to line 1000
	lines, err = w.ScrollDown(800)
	if err != nil {
		t.Fatalf("ScrollDown(800) failed: %v", err)
	}
	if w.CurrentLine != 1000 {
		t.Errorf("expected CurrentLine clamped to 1000, got %d", w.CurrentLine)
	}
	if len(lines) != 1 || lines[0] != "1000: line 1000" {
		t.Errorf("expected single clamped EOF line '1000: line 1000', got %v", lines)
	}

	// Further ScrollDown stays clamped at line 1000
	lines, err = w.ScrollDown(10)
	if err != nil {
		t.Fatalf("ScrollDown(10) at EOF failed: %v", err)
	}
	if w.CurrentLine != 1000 {
		t.Errorf("expected CurrentLine 1000, got %d", w.CurrentLine)
	}
	if len(lines) != 1 || lines[0] != "1000: line 1000" {
		t.Errorf("expected single clamped EOF line '1000: line 1000', got %v", lines)
	}
}

// TestWindowedFileScrollUp verifies ScrollUp moves back deterministically and clamps at line 1.
func TestWindowedFileScrollUp(t *testing.T) {
	dir := t.TempDir()
	var sb strings.Builder
	for i := 1; i <= 1000; i++ {
		fmt.Fprintf(&sb, "line %d\n", i)
	}
	filePath := createTestFile(t, dir, "scrollup.txt", sb.String())

	w := NewWindowedFileReader(100)
	_, err := w.Open(filePath, 1000, 100)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	if w.CurrentLine != 1000 {
		t.Fatalf("expected initial CurrentLine 1000, got %d", w.CurrentLine)
	}

	// ScrollUp(100) -> moves back to line 900
	lines, err := w.ScrollUp(100)
	if err != nil {
		t.Fatalf("ScrollUp(100) failed: %v", err)
	}
	if w.CurrentLine != 900 {
		t.Errorf("expected CurrentLine 900, got %d", w.CurrentLine)
	}
	if len(lines) != 100 {
		t.Fatalf("expected 100 lines, got %d", len(lines))
	}
	if lines[0] != "900: line 900" || lines[99] != "999: line 999" {
		t.Errorf("unexpected lines: first=%q, last=%q", lines[0], lines[99])
	}

	// ScrollUp(0) -> defaults to WindowSize (100 lines), moves from 900 to 800
	lines, err = w.ScrollUp(0)
	if err != nil {
		t.Fatalf("ScrollUp(0) failed: %v", err)
	}
	if w.CurrentLine != 800 {
		t.Errorf("expected CurrentLine 800 after ScrollUp(0), got %d", w.CurrentLine)
	}
	if lines[0] != "800: line 800" || lines[99] != "899: line 899" {
		t.Errorf("unexpected lines after ScrollUp(0): first=%q, last=%q", lines[0], lines[99])
	}

	// ScrollUp(-5) -> negative n defaults to WindowSize (100 lines), moves from 800 to 700
	lines, err = w.ScrollUp(-5)
	if err != nil {
		t.Fatalf("ScrollUp(-5) failed: %v", err)
	}
	if w.CurrentLine != 700 {
		t.Errorf("expected CurrentLine 700 after ScrollUp(-5), got %d", w.CurrentLine)
	}
	if lines[0] != "700: line 700" || lines[99] != "799: line 799" {
		t.Errorf("unexpected lines after ScrollUp(-5): first=%q, last=%q", lines[0], lines[99])
	}

	// ScrollUp(1000) -> clamps to line 1
	lines, err = w.ScrollUp(1000)
	if err != nil {
		t.Fatalf("ScrollUp(1000) failed: %v", err)
	}
	if w.CurrentLine != 1 {
		t.Errorf("expected CurrentLine clamped to 1, got %d", w.CurrentLine)
	}
	if len(lines) != 100 {
		t.Fatalf("expected 100 lines, got %d", len(lines))
	}
	if lines[0] != "1: line 1" || lines[99] != "100: line 100" {
		t.Errorf("unexpected lines clamped at top: first=%q, last=%q", lines[0], lines[99])
	}

	// Further ScrollUp stays clamped at line 1
	lines, err = w.ScrollUp(50)
	if err != nil {
		t.Fatalf("ScrollUp(50) at top failed: %v", err)
	}
	if w.CurrentLine != 1 {
		t.Errorf("expected CurrentLine 1, got %d", w.CurrentLine)
	}
	if lines[0] != "1: line 1" || lines[99] != "100: line 100" {
		t.Errorf("unexpected lines: first=%q, last=%q", lines[0], lines[99])
	}
}

// TestWindowedFileGoto verifies Goto jumps to specific lines accurately.
func TestWindowedFileGoto(t *testing.T) {
	dir := t.TempDir()
	var sb strings.Builder
	for i := 1; i <= 1000; i++ {
		fmt.Fprintf(&sb, "line %d\n", i)
	}
	filePath := createTestFile(t, dir, "goto.txt", sb.String())

	w := NewWindowedFileReader(100)
	_, err := w.Open(filePath, 1, 100)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	// Goto(500) -> jumps to line 500
	lines, err := w.Goto(500)
	if err != nil {
		t.Fatalf("Goto(500) failed: %v", err)
	}
	if w.CurrentLine != 500 {
		t.Errorf("expected CurrentLine 500, got %d", w.CurrentLine)
	}
	if len(lines) != 100 {
		t.Fatalf("expected 100 lines, got %d", len(lines))
	}
	if lines[0] != "500: line 500" || lines[99] != "599: line 599" {
		t.Errorf("unexpected lines at 500: first=%q, last=%q", lines[0], lines[99])
	}

	// Goto(-50) -> clamps to line 1
	lines, err = w.Goto(-50)
	if err != nil {
		t.Fatalf("Goto(-50) failed: %v", err)
	}
	if w.CurrentLine != 1 {
		t.Errorf("expected CurrentLine clamped to 1, got %d", w.CurrentLine)
	}
	if lines[0] != "1: line 1" || lines[99] != "100: line 100" {
		t.Errorf("unexpected lines at clamped min: first=%q, last=%q", lines[0], lines[99])
	}

	// Goto(2000) -> clamps to total lines (1000)
	lines, err = w.Goto(2000)
	if err != nil {
		t.Fatalf("Goto(2000) failed: %v", err)
	}
	if w.CurrentLine != 1000 {
		t.Errorf("expected CurrentLine clamped to 1000, got %d", w.CurrentLine)
	}
	if len(lines) != 1 || lines[0] != "1000: line 1000" {
		t.Errorf("unexpected lines at clamped max: %v", lines)
	}

	// Goto(1) -> back to start
	lines, err = w.Goto(1)
	if err != nil {
		t.Fatalf("Goto(1) failed: %v", err)
	}
	if w.CurrentLine != 1 {
		t.Errorf("expected CurrentLine 1, got %d", w.CurrentLine)
	}
	if lines[0] != "1: line 1" {
		t.Errorf("expected '1: line 1', got %q", lines[0])
	}
}

// TestWindowedFileEdgeCases tests short files, empty files, boundary clamping, and line number formatting.
func TestWindowedFileEdgeCases(t *testing.T) {
	dir := t.TempDir()

	// 1. Short file with fewer lines than WindowSize (displays all lines from line 1)
	shortPath := createTestFile(t, dir, "short.txt", "alpha\nbeta\ngamma\n")
	w := NewWindowedFileReader(10)

	lines, err := w.Open(shortPath, 1, 10)
	if err != nil {
		t.Fatalf("Open short file failed: %v", err)
	}
	if len(lines) != 3 {
		t.Fatalf("expected 3 lines for short file, got %d", len(lines))
	}
	if lines[0] != "1: alpha" || lines[1] != "2: beta" || lines[2] != "3: gamma" {
		t.Errorf("unexpected short file lines: %v", lines)
	}
	if w.CurrentLine != 1 {
		t.Errorf("expected CurrentLine 1, got %d", w.CurrentLine)
	}

	// ScrollDown on short file clamps at line 3
	lines, err = w.ScrollDown(5)
	if err != nil {
		t.Fatalf("ScrollDown failed: %v", err)
	}
	if w.CurrentLine != 3 {
		t.Errorf("expected CurrentLine 3, got %d", w.CurrentLine)
	}
	if len(lines) != 1 || lines[0] != "3: gamma" {
		t.Errorf("unexpected clamped lines: %v", lines)
	}

	// 2. Empty file: returns empty list without error
	emptyPath := createTestFile(t, dir, "empty.txt", "")
	lines, err = w.Open(emptyPath, 1, 10)
	if err != nil {
		t.Fatalf("Open empty file failed: %v", err)
	}
	if len(lines) != 0 {
		t.Errorf("expected empty list for empty file, got %d items", len(lines))
	}
	if w.CurrentLine != 1 {
		t.Errorf("expected CurrentLine 1 for empty file, got %d", w.CurrentLine)
	}
	if len(w.Lines) != 0 {
		t.Errorf("expected 0 Lines, got %d", len(w.Lines))
	}
	if getLines := w.GetWindow(); len(getLines) != 0 {
		t.Errorf("expected GetWindow to be empty for empty file, got %d items", len(getLines))
	}
	if status := w.Status(); !strings.Contains(status, "lines 0-0 of 0") {
		t.Errorf("unexpected empty file status: %q", status)
	}

	// Navigation on empty file remains empty and error-free
	lines, err = w.ScrollDown(5)
	if err != nil || len(lines) != 0 {
		t.Errorf("ScrollDown on empty file failed: err=%v, lines=%v", err, lines)
	}
	lines, err = w.ScrollUp(5)
	if err != nil || len(lines) != 0 {
		t.Errorf("ScrollUp on empty file failed: err=%v, lines=%v", err, lines)
	}
	lines, err = w.Goto(5)
	if err != nil || len(lines) != 0 {
		t.Errorf("Goto on empty file failed: err=%v, lines=%v", err, lines)
	}

	// 3. Line number formatting: empty lines, leading whitespace, exact format
	formatContent := "first line\n\n  indented line\n\ttab indented\n"
	formatPath := createTestFile(t, dir, "format.txt", formatContent)

	lines, err = w.Open(formatPath, 1, 10)
	if err != nil {
		t.Fatalf("Open format file failed: %v", err)
	}
	if len(lines) != 4 {
		t.Fatalf("expected 4 lines, got %d", len(lines))
	}
	if lines[0] != "1: first line" {
		t.Errorf("expected %q, got %q", "1: first line", lines[0])
	}
	if lines[1] != "2: " {
		t.Errorf("expected %q, got %q", "2: ", lines[1])
	}
	if lines[2] != "3:   indented line" {
		t.Errorf("expected %q, got %q", "3:   indented line", lines[2])
	}
	if lines[3] != "4: \ttab indented" {
		t.Errorf("expected %q, got %q", "4: \ttab indented", lines[3])
	}

	// 4. Boundary clamping on Open:
	// startLine <= 0 clamps to 1
	lines, err = w.Open(formatPath, -10, 2)
	if err != nil {
		t.Fatalf("Open with negative startLine failed: %v", err)
	}
	if w.CurrentLine != 1 || len(lines) != 2 || lines[0] != "1: first line" {
		t.Errorf("clamping to line 1 failed: CurrentLine=%d, lines=%v", w.CurrentLine, lines)
	}

	// startLine > totalLines clamps to totalLines (4)
	lines, err = w.Open(formatPath, 999, 2)
	if err != nil {
		t.Fatalf("Open past totalLines failed: %v", err)
	}
	if w.CurrentLine != 4 || len(lines) != 1 || lines[0] != "4: \ttab indented" {
		t.Errorf("clamping to totalLines failed: CurrentLine=%d, lines=%v", w.CurrentLine, lines)
	}

	// windowSize <= 0 defaults to defaultWindowSize
	lines, err = w.Open(formatPath, 1, 0)
	if err != nil {
		t.Fatalf("Open with zero windowSize failed: %v", err)
	}
	if w.WindowSize != 10 {
		t.Errorf("expected WindowSize 10, got %d", w.WindowSize)
	}
	if len(lines) != 4 {
		t.Errorf("expected all 4 lines, got %d", len(lines))
	}
}

// TestWindowedFileErrors tests non-existent files, directories, empty paths, and un-opened navigation.
func TestWindowedFileErrors(t *testing.T) {
	w := NewWindowedFileReader(100)

	// Open non-existent file
	if _, err := w.Open("this/file/does/not/exist.txt", 1, 10); err == nil {
		t.Error("expected error for non-existent file, got nil")
	}

	// Open directory
	dir := t.TempDir()
	if _, err := w.Open(dir, 1, 10); err == nil {
		t.Error("expected error for directory, got nil")
	}

	// Open empty path
	if _, err := w.Open("", 1, 10); err == nil {
		t.Error("expected error for empty path, got nil")
	}
	if _, err := w.Open("   ", 1, 10); err == nil {
		t.Error("expected error for whitespace path, got nil")
	}

	// Path with NUL byte
	if _, err := w.Open("foo\x00bar", 1, 10); err == nil {
		t.Error("expected error for NUL byte in path, got nil")
	}

	// Navigation before opening any file returns ErrNoFileOpen
	if _, err := w.ScrollDown(1); !errors.Is(err, ErrNoFileOpen) {
		t.Errorf("expected ErrNoFileOpen from ScrollDown, got %v", err)
	}
	if _, err := w.ScrollUp(1); !errors.Is(err, ErrNoFileOpen) {
		t.Errorf("expected ErrNoFileOpen from ScrollUp, got %v", err)
	}
	if _, err := w.Goto(1); !errors.Is(err, ErrNoFileOpen) {
		t.Errorf("expected ErrNoFileOpen from Goto, got %v", err)
	}

	if lines := w.GetWindow(); len(lines) != 0 {
		t.Errorf("expected empty window when no file open, got %d", len(lines))
	}
	if status := w.Status(); status != "no file open" {
		t.Errorf("expected 'no file open' status, got %q", status)
	}

	// Open and then Close
	filePath := createTestFile(t, dir, "close_test.txt", "one\ntwo\n")
	if _, err := w.Open(filePath, 1, 10); err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	if !w.IsOpen() {
		t.Error("expected IsOpen to be true")
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}
	if w.IsOpen() {
		t.Error("expected IsOpen to be false after Close")
	}
	if _, err := w.ScrollDown(1); !errors.Is(err, ErrNoFileOpen) {
		t.Errorf("expected ErrNoFileOpen after Close, got %v", err)
	}
}

// TestWindowedFileLineEndings tests CRLF, CR, and mixed line endings.
func TestWindowedFileLineEndings(t *testing.T) {
	dir := t.TempDir()

	crlfPath := createTestFile(t, dir, "crlf.txt", "line1\r\nline2\r\nline3\r\n")
	crPath := createTestFile(t, dir, "cr.txt", "lineA\rlineB\rlineC\r")
	mixedPath := createTestFile(t, dir, "mixed.txt", "alpha\r\nbeta\rgamma\ndelta")

	w := NewWindowedFileReader(100)

	// CRLF
	lines, err := w.Open(crlfPath, 1, 10)
	if err != nil {
		t.Fatalf("Open CRLF failed: %v", err)
	}
	if len(lines) != 3 || lines[0] != "1: line1" || lines[1] != "2: line2" || lines[2] != "3: line3" {
		t.Errorf("unexpected CRLF output: %v", lines)
	}

	// CR
	lines, err = w.Open(crPath, 1, 10)
	if err != nil {
		t.Fatalf("Open CR failed: %v", err)
	}
	if len(lines) != 3 || lines[0] != "1: lineA" || lines[1] != "2: lineB" || lines[2] != "3: lineC" {
		t.Errorf("unexpected CR output: %v", lines)
	}

	// Mixed without trailing newline
	lines, err = w.Open(mixedPath, 1, 10)
	if err != nil {
		t.Fatalf("Open mixed failed: %v", err)
	}
	if len(lines) != 4 || lines[3] != "4: delta" {
		t.Errorf("unexpected mixed output: %v", lines)
	}
}

// TestWindowedFileUTF8Safety tests UTF-8 multibyte handling, invalid bytes, and BOM stripping.
func TestWindowedFileUTF8Safety(t *testing.T) {
	dir := t.TempDir()

	// 1. Multibyte UTF-8
	utf8Content := "Line 1: こんにちは世界\nLine 2: 🚀 Rocket Launch\nLine 3: Café & résumé\n"
	utf8Path := createTestFile(t, dir, "multibyte.txt", utf8Content)

	w := NewWindowedFileReader(100)
	lines, err := w.Open(utf8Path, 1, 10)
	if err != nil {
		t.Fatalf("Open multibyte UTF-8 file failed: %v", err)
	}
	for _, l := range lines {
		if !utf8.ValidString(l) {
			t.Errorf("expected valid UTF-8 string, got: %q", l)
		}
	}
	if lines[0] != "1: Line 1: こんにちは世界" || lines[1] != "2: Line 2: 🚀 Rocket Launch" {
		t.Errorf("multibyte characters corrupted: %v", lines)
	}

	// 2. Invalid UTF-8 bytes sanitized
	invalidBytes := []byte("header\ncorrupt: \xff\xfe data\nfooter\n")
	invalidPath := filepath.Join(dir, "invalid.txt")
	if err := os.WriteFile(invalidPath, invalidBytes, 0644); err != nil {
		t.Fatalf("write invalid UTF-8 failed: %v", err)
	}

	lines, err = w.Open(invalidPath, 1, 10)
	if err != nil {
		t.Fatalf("Open invalid UTF-8 file failed: %v", err)
	}
	for _, l := range lines {
		if !utf8.ValidString(l) {
			t.Errorf("expected sanitized valid UTF-8, got: %q", l)
		}
	}
	if !strings.Contains(lines[1], "\uFFFD") {
		t.Errorf("expected replacement character \\uFFFD in line 2, got: %q", lines[1])
	}

	// 3. UTF-8 BOM stripped cleanly
	bomBytes := append([]byte("\xef\xbb\xbf"), []byte("bom line 1\nbom line 2\n")...)
	bomPath := filepath.Join(dir, "bom.txt")
	if err := os.WriteFile(bomPath, bomBytes, 0644); err != nil {
		t.Fatalf("write BOM file failed: %v", err)
	}

	lines, err = w.Open(bomPath, 1, 10)
	if err != nil {
		t.Fatalf("Open BOM file failed: %v", err)
	}
	if lines[0] != "1: bom line 1" {
		t.Errorf("expected BOM to be stripped, got: %q", lines[0])
	}
}

// TestWindowedFileConcurrency tests concurrent access to navigation and state methods.
func TestWindowedFileConcurrency(t *testing.T) {
	dir := t.TempDir()
	var sb strings.Builder
	for i := 1; i <= 200; i++ {
		fmt.Fprintf(&sb, "line %d\n", i)
	}
	filePath := createTestFile(t, dir, "concurrency.txt", sb.String())

	w := NewWindowedFileReader(10)
	if _, err := w.Open(filePath, 1, 10); err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	var wg sync.WaitGroup
	workers := 8
	iterations := 50

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				switch (id + j) % 6 {
				case 0:
					_, _ = w.ScrollDown(2)
				case 1:
					_, _ = w.ScrollUp(2)
				case 2:
					_, _ = w.Goto((id*7 + j) % 200)
				case 3:
					_ = w.GetWindow()
				case 4:
					_ = w.Status()
				case 5:
					_, _, _, _ = w.CurrentPosition()
				}
			}
		}(i)
	}

	wg.Wait()
}

// TestWindowedFileReader_BinaryAndSizeGuard asserts size and binary file checks.
func TestWindowedFileReader_BinaryAndSizeGuard(t *testing.T) {
	dir := t.TempDir()

	// (a) File exceeding MaxFileSize returns ErrFileTooLarge
	t.Run("ExceedsMaxFileSize", func(t *testing.T) {
		largeContent := strings.Repeat("a", 200)
		largePath := createTestFile(t, dir, "large.txt", largeContent)

		w := NewWindowedFileReader()
		w.SetMaxFileSize(100)
		if w.MaxFileSize() != 100 {
			t.Fatalf("expected MaxFileSize 100, got %d", w.MaxFileSize())
		}

		lines, err := w.Open(largePath, 1, 10)
		if err == nil {
			t.Fatal("expected error for file exceeding MaxFileSize, got nil")
		}
		if !errors.Is(err, ErrFileTooLarge) {
			t.Fatalf("expected ErrFileTooLarge, got %v", err)
		}
		if lines != nil {
			t.Fatalf("expected nil lines on ErrFileTooLarge, got %v", lines)
		}
		if w.IsOpen() {
			t.Fatal("expected IsOpen to be false after failed Open")
		}

		// Also test with int64 constructor arg
		w2 := NewWindowedFileReader(int64(50))
		if w2.MaxFileSize() != 50 {
			t.Fatalf("expected MaxFileSize 50, got %d", w2.MaxFileSize())
		}
		_, err = w2.Open(largePath, 1, 10)
		if !errors.Is(err, ErrFileTooLarge) {
			t.Fatalf("expected ErrFileTooLarge with constructor limit, got %v", err)
		}

		// Also test with WindowedOptions / WindowedConfig
		wOpts := NewWindowedFileReader(WindowedOptions{MaxFileSize: 100})
		if wOpts.MaxFileSize() != 100 {
			t.Fatalf("expected MaxFileSize 100 from WindowedOptions, got %d", wOpts.MaxFileSize())
		}
		_, err = wOpts.Open(largePath, 1, 10)
		if !errors.Is(err, ErrFileTooLarge) {
			t.Fatalf("expected ErrFileTooLarge with WindowedOptions limit, got %v", err)
		}

		// 100MB+ file check against default limit (50MB = DefaultMaxFileSize)
		large100MBPath := filepath.Join(dir, "large_100mb.txt")
		f, err := os.Create(large100MBPath)
		if err != nil {
			t.Fatalf("failed to create 100MB file: %v", err)
		}
		const size105MB = int64(105 * 1024 * 1024)
		if err := f.Truncate(size105MB); err != nil {
			_ = f.Close()
			t.Fatalf("failed to truncate 100MB+ file: %v", err)
		}
		_ = f.Close()

		defaultReader := NewWindowedFileReader()
		if defaultReader.MaxFileSize() != DefaultMaxFileSize {
			t.Fatalf("expected default MaxFileSize %d, got %d", DefaultMaxFileSize, defaultReader.MaxFileSize())
		}
		lines100, err := defaultReader.Open(large100MBPath, 1, 10)
		if err == nil {
			t.Fatal("expected error for 100MB+ file exceeding DefaultMaxFileSize, got nil")
		}
		if !errors.Is(err, ErrFileTooLarge) {
			t.Fatalf("expected ErrFileTooLarge, got %v", err)
		}
		if lines100 != nil {
			t.Fatalf("expected nil lines on ErrFileTooLarge, got %v", lines100)
		}
		if defaultReader.IsOpen() {
			t.Fatal("expected IsOpen to be false after 100MB+ file rejection")
		}
	})

	// (b) File containing NUL bytes returns ErrBinaryFile
	t.Run("BinaryFileDetected", func(t *testing.T) {
		headerBinaryContent := "hello\x00world\nline 2\n"
		headerBinaryPath := filepath.Join(dir, "binary_header.bin")
		if err := os.WriteFile(headerBinaryPath, []byte(headerBinaryContent), 0644); err != nil {
			t.Fatalf("failed to write binary file: %v", err)
		}

		w := NewWindowedFileReader()
		lines, err := w.Open(headerBinaryPath, 1, 10)
		if err == nil {
			t.Fatal("expected error for binary file, got nil")
		}
		if !errors.Is(err, ErrBinaryFile) {
			t.Fatalf("expected ErrBinaryFile, got %v", err)
		}
		if lines != nil {
			t.Fatalf("expected nil lines on ErrBinaryFile, got %v", lines)
		}
		if w.IsOpen() {
			t.Fatal("expected IsOpen to be false after binary file rejection")
		}

		// Binary file with NUL in first 512 bytes rejected early without reading full 5MB payload
		largeBinaryHeaderPath := filepath.Join(dir, "binary_large_header.bin")
		largeBinFile, err := os.Create(largeBinaryHeaderPath)
		if err != nil {
			t.Fatalf("failed to create large binary file: %v", err)
		}
		if _, err := largeBinFile.Write([]byte("header\x00payload")); err != nil {
			_ = largeBinFile.Close()
			t.Fatalf("failed to write binary header: %v", err)
		}
		if err := largeBinFile.Truncate(5 * 1024 * 1024); err != nil {
			_ = largeBinFile.Close()
			t.Fatalf("failed to truncate large binary file: %v", err)
		}
		_ = largeBinFile.Close()

		wLargeBin := NewWindowedFileReader()
		linesBin, err := wLargeBin.Open(largeBinaryHeaderPath, 1, 10)
		if err == nil {
			t.Fatal("expected error for large binary file, got nil")
		}
		if !errors.Is(err, ErrBinaryFile) {
			t.Fatalf("expected ErrBinaryFile, got %v", err)
		}
		if linesBin != nil {
			t.Fatalf("expected nil lines on ErrBinaryFile, got %v", linesBin)
		}
		if wLargeBin.IsOpen() {
			t.Fatal("expected IsOpen to be false after binary rejection")
		}

		largeBinary := append(bytes.Repeat([]byte("x"), 9000), 0x00, 'y', '\n')
		deepBinaryPath := filepath.Join(dir, "binary_deep.bin")
		if err := os.WriteFile(deepBinaryPath, largeBinary, 0644); err != nil {
			t.Fatalf("failed to write deep binary file: %v", err)
		}
		_, err = w.Open(deepBinaryPath, 1, 10)
		if !errors.Is(err, ErrBinaryFile) {
			t.Fatalf("expected ErrBinaryFile for deep NUL, got %v", err)
		}
	})

	// (c) Normal text file paginates cleanly
	t.Run("NormalTextFilePaginatesCleanly", func(t *testing.T) {
		textContent := "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8\n"
		textPath := createTestFile(t, dir, "normal.txt", textContent)

		w := NewWindowedFileReader(3)
		lines, err := w.Open(textPath, 1, 3)
		if err != nil {
			t.Fatalf("unexpected error opening normal text file: %v", err)
		}
		if len(lines) != 3 {
			t.Fatalf("expected 3 lines, got %d", len(lines))
		}
		if w.TotalLines != 8 {
			t.Fatalf("expected 8 TotalLines, got %d", w.TotalLines)
		}
		if w.CurrentLine != 1 {
			t.Fatalf("expected CurrentLine 1, got %d", w.CurrentLine)
		}

		// Scroll down
		lines, err = w.ScrollDown(2)
		if err != nil {
			t.Fatalf("ScrollDown failed: %v", err)
		}
		if w.CurrentLine != 3 || len(lines) != 3 {
			t.Fatalf("expected CurrentLine 3, got %d", w.CurrentLine)
		}

		// Scroll up
		lines, err = w.ScrollUp(1)
		if err != nil {
			t.Fatalf("ScrollUp failed: %v", err)
		}
		if w.CurrentLine != 2 || len(lines) != 3 {
			t.Fatalf("expected CurrentLine 2, got %d", w.CurrentLine)
		}

		// Goto
		lines, err = w.Goto(6)
		if err != nil {
			t.Fatalf("Goto failed: %v", err)
		}
		if w.CurrentLine != 6 || len(lines) != 3 {
			t.Fatalf("expected CurrentLine 6, got %d", w.CurrentLine)
		}
	})
}

// TestWindowedFileBinaryAndSizeGuard is an alias for TestWindowedFileReader_BinaryAndSizeGuard.
func TestWindowedFileBinaryAndSizeGuard(t *testing.T) {
	TestWindowedFileReader_BinaryAndSizeGuard(t)
}

// TestWindowedFileWindowView verifies companion WindowView methods.
func TestWindowedFileWindowView(t *testing.T) {
	dir := t.TempDir()
	content := "line 1\nline 2\nline 3\nline 4\nline 5\n"
	filePath := createTestFile(t, dir, "view.txt", content)

	w := NewWindowedFileReader(3)
	view, err := w.OpenView(filePath, 1, 3)
	if err != nil {
		t.Fatalf("OpenView failed: %v", err)
	}
	if view.StartLine != 1 || view.EndLine != 3 || view.TotalLines != 5 {
		t.Errorf("unexpected view: %+v", view)
	}
	if !strings.Contains(view.Content, "1: line 1") {
		t.Errorf("unexpected view content: %q", view.Content)
	}

	view, err = w.ScrollDownView(1)
	if err != nil {
		t.Fatalf("ScrollDownView failed: %v", err)
	}
	if view.StartLine != 2 {
		t.Errorf("expected StartLine 2, got %d", view.StartLine)
	}

	view, err = w.ScrollUpView(1)
	if err != nil {
		t.Fatalf("ScrollUpView failed: %v", err)
	}
	if view.StartLine != 1 {
		t.Errorf("expected StartLine 1, got %d", view.StartLine)
	}

	view, err = w.GotoView(4)
	if err != nil {
		t.Fatalf("GotoView failed: %v", err)
	}
	if view.StartLine != 4 {
		t.Errorf("expected StartLine 4, got %d", view.StartLine)
	}
}

// TestWindowedFileConfig verifies defaultWindowSize fallback and SetWindowSize.
func TestWindowedFileConfig(t *testing.T) {
	w0 := NewWindowedFileReader(0)
	if w0.DefaultWindowSize() != DefaultWindowSize {
		t.Errorf("expected default 100, got %d", w0.DefaultWindowSize())
	}

	w := NewWindowedFileReader(20)
	if w.DefaultWindowSize() != 20 {
		t.Errorf("expected default 20, got %d", w.DefaultWindowSize())
	}

	w.SetDefaultWindowSize(50)
	if w.DefaultWindowSize() != 50 {
		t.Errorf("expected default 50, got %d", w.DefaultWindowSize())
	}

	w.SetWindowSize(30)
	if w.WindowSize != 30 {
		t.Errorf("expected WindowSize 30, got %d", w.WindowSize)
	}

	w.SetWindowSize(0) // resets to defaultWindowSize
	if w.WindowSize != 50 {
		t.Errorf("expected WindowSize 50, got %d", w.WindowSize)
	}
}

// TestWindowedFileRejectNonRegularFile verifies that non-regular files (devices, FIFOs, pipes, directories)
// are rejected with ErrInvalidPath and a "not a regular file" error to prevent hangs.
func TestWindowedFileRejectNonRegularFile(t *testing.T) {
	// 1. Directory is a non-regular file and returns ErrInvalidPath
	dir := t.TempDir()
	wDir := NewWindowedFileReader(10)
	linesDir, err := wDir.Open(dir, 1, 10)
	if err == nil {
		t.Fatal("expected error opening directory, got nil")
	}
	if !errors.Is(err, ErrInvalidPath) {
		t.Errorf("expected ErrInvalidPath for directory, got %v", err)
	}
	if linesDir != nil {
		t.Errorf("expected nil lines for directory, got %v", linesDir)
	}
	if wDir.IsOpen() {
		t.Error("expected IsOpen to be false after failed open on directory")
	}

	// 2. Character device / non-regular file (os.DevNull)
	fi, err := os.Stat(os.DevNull)
	if err != nil {
		t.Skipf("cannot stat os.DevNull on this platform: %v", err)
	}
	if fi.Mode().IsRegular() {
		t.Fatalf("expected os.DevNull to be a non-regular file, got Mode=%v", fi.Mode())
	}

	w := NewWindowedFileReader(10)
	lines, err := w.Open(os.DevNull, 1, 10)
	if err == nil {
		t.Fatal("expected error opening non-regular file (os.DevNull), got nil")
	}
	if !errors.Is(err, ErrInvalidPath) {
		t.Errorf("expected ErrInvalidPath, got %v", err)
	}
	if !strings.Contains(err.Error(), "not a regular file") {
		t.Errorf("expected 'not a regular file' in error message, got %q", err.Error())
	}
	if lines != nil {
		t.Errorf("expected nil lines on error, got %v", lines)
	}
	if w.IsOpen() {
		t.Error("expected IsOpen to be false after failed open on non-regular file")
	}
	if w.Lines != nil {
		t.Errorf("expected Lines to be nil, got %v", w.Lines)
	}
	if w.FilePath != "" {
		t.Errorf("expected FilePath to be empty, got %q", w.FilePath)
	}
	if w.CurrentLine != 0 {
		t.Errorf("expected CurrentLine 0, got %d", w.CurrentLine)
	}
	if _, err := w.View(); !errors.Is(err, ErrNoFileOpen) {
		t.Errorf("expected ErrNoFileOpen from View, got %v", err)
	}
}

// TestWindowedFileResetStateOnFailedOpen verifies that when Open fails after a file was already open,
// all internal state (Lines, FilePath, CurrentLine) is cleanly reset and IsOpen becomes false.
func TestWindowedFileResetStateOnFailedOpen(t *testing.T) {
	dir := t.TempDir()
	validPath := createTestFile(t, dir, "valid.txt", "line 1\nline 2\nline 3\n")

	w := NewWindowedFileReader(10)

	// Step 1: Open a valid file and verify it is active
	lines, err := w.Open(validPath, 1, 10)
	if err != nil {
		t.Fatalf("initial Open failed: %v", err)
	}
	if len(lines) != 3 || !w.IsOpen() {
		t.Fatalf("expected file to be open with 3 lines, got IsOpen=%v, len=%d", w.IsOpen(), len(lines))
	}
	if w.FilePath != validPath {
		t.Fatalf("expected FilePath %q, got %q", validPath, w.FilePath)
	}
	if w.CurrentLine != 1 {
		t.Fatalf("expected CurrentLine 1, got %d", w.CurrentLine)
	}

	// Step 2: Attempt to open missing_file_xyz.txt
	failedLines, err := w.Open("missing_file_xyz.txt", 100, 10)
	if err == nil {
		t.Fatal("expected error opening nonexistent file, got nil")
	}
	if failedLines != nil {
		t.Errorf("expected nil lines on failed Open, got %v", failedLines)
	}
	if w.IsOpen() {
		t.Error("expected IsOpen == false after failed Open")
	}
	if w.Lines != nil {
		t.Errorf("expected Lines == nil, got %v", w.Lines)
	}
	if w.FilePath != "" {
		t.Errorf("expected FilePath to be empty, got %q", w.FilePath)
	}
	if w.CurrentLine != 0 {
		t.Errorf("expected CurrentLine 0, got %d", w.CurrentLine)
	}

	// Subsequent navigation methods must fail with ErrNoFileOpen
	if _, err := w.View(); !errors.Is(err, ErrNoFileOpen) {
		t.Errorf("expected ErrNoFileOpen from View, got %v", err)
	}
	if _, err := w.ScrollDown(1); !errors.Is(err, ErrNoFileOpen) {
		t.Errorf("expected ErrNoFileOpen from ScrollDown, got %v", err)
	}
	if _, err := w.ScrollUp(1); !errors.Is(err, ErrNoFileOpen) {
		t.Errorf("expected ErrNoFileOpen from ScrollUp, got %v", err)
	}
	if _, err := w.Goto(1); !errors.Is(err, ErrNoFileOpen) {
		t.Errorf("expected ErrNoFileOpen from Goto, got %v", err)
	}
	if status := w.Status(); status != "no file open" {
		t.Errorf("expected 'no file open' status, got %q", status)
	}
	path, curLine, winSize, totalLines := w.CurrentPosition()
	if path != "" || curLine != 0 || winSize != 0 || totalLines != 0 {
		t.Errorf("expected zeroed CurrentPosition, got path=%q, curLine=%d, winSize=%d, totalLines=%d",
			path, curLine, winSize, totalLines)
	}

	// Step 3: Re-open valid file, then fail with non-regular file
	if _, err := w.Open(validPath, 1, 10); err != nil {
		t.Fatalf("re-opening valid file failed: %v", err)
	}
	if !w.IsOpen() {
		t.Fatal("expected file to be open after re-opening")
	}
	if _, err := w.Open(os.DevNull, 1, 10); err == nil {
		t.Fatal("expected error opening os.DevNull, got nil")
	}
	if w.IsOpen() {
		t.Error("expected IsOpen to be false after failed Open on non-regular file")
	}
	if w.Lines != nil {
		t.Errorf("expected Lines to be nil after failed Open on non-regular file")
	}
	if _, err := w.View(); !errors.Is(err, ErrNoFileOpen) {
		t.Errorf("expected ErrNoFileOpen from View, got %v", err)
	}

	// Step 4: Re-open valid file, then fail with binary file
	binaryPath := createTestFile(t, dir, "binary.bin", "text\x00binary\n")
	if _, err := w.Open(validPath, 1, 10); err != nil {
		t.Fatalf("re-opening valid file failed: %v", err)
	}
	if !w.IsOpen() {
		t.Fatal("expected file to be open after re-opening")
	}
	if _, err := w.Open(binaryPath, 1, 10); err == nil {
		t.Fatal("expected error opening binary file, got nil")
	}
	if w.IsOpen() {
		t.Error("expected IsOpen to be false after failed Open on binary file")
	}
	if w.Lines != nil {
		t.Errorf("expected Lines to be nil after failed Open on binary file")
	}
	if _, err := w.View(); !errors.Is(err, ErrNoFileOpen) {
		t.Errorf("expected ErrNoFileOpen from View, got %v", err)
	}
}
