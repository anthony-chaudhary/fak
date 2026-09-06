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

func TestWindowedFileOpen(t *testing.T) {
	dir := t.TempDir()
	content := "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8\nline 9\nline 10\n"
	filePath := createTestFile(t, dir, "sample.txt", content)

	w := NewWindowedFileReader(5)

	// Open with line 1 and window size 3
	view, err := w.Open(filePath, 1, 3)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	if view.Path != filePath {
		t.Errorf("expected Path %q, got %q", filePath, view.Path)
	}
	if view.StartLine != 1 {
		t.Errorf("expected StartLine 1, got %d", view.StartLine)
	}
	if view.EndLine != 3 {
		t.Errorf("expected EndLine 3, got %d", view.EndLine)
	}
	if view.TotalLines != 10 {
		t.Errorf("expected TotalLines 10, got %d", view.TotalLines)
	}

	// Verify Header
	expectedHeader := fmt.Sprintf("=== [%s] (lines 1-3 of 10) ===\n", filePath)
	if !strings.HasPrefix(view.Content, expectedHeader) {
		t.Errorf("expected header %q, got content:\n%s", expectedHeader, view.Content)
	}

	// Verify Body line numbering
	expectedBody := "1: line 1\n2: line 2\n3: line 3\n"
	if !strings.Contains(view.Content, expectedBody) {
		t.Errorf("expected body %q, got content:\n%s", expectedBody, view.Content)
	}

	// Verify Footer (not EOF, so End of Window)
	expectedFooter := "\n=== Navigation: ScrollDown(n), ScrollUp(n), Goto(line) | End of Window ==="
	if !strings.HasSuffix(view.Content, expectedFooter) {
		t.Errorf("expected footer %q, got content:\n%s", expectedFooter, view.Content)
	}

	// Open with defaults (startLine <= 0, windowSize <= 0)
	view, err = w.Open(filePath, 0, 0)
	if err != nil {
		t.Fatalf("Open with defaults failed: %v", err)
	}
	if view.StartLine != 1 {
		t.Errorf("expected StartLine 1 with default, got %d", view.StartLine)
	}
	if view.EndLine != 5 { // defaultWindowSize = 5
		t.Errorf("expected EndLine 5 with default, got %d", view.EndLine)
	}

	// Open with startLine > TotalLines clamps to TotalLines
	view, err = w.Open(filePath, 50, 3)
	if err != nil {
		t.Fatalf("Open with startLine > total failed: %v", err)
	}
	if view.StartLine != 10 {
		t.Errorf("expected StartLine 10 when clamped, got %d", view.StartLine)
	}
	if view.EndLine != 10 {
		t.Errorf("expected EndLine 10 when clamped, got %d", view.EndLine)
	}
	if !strings.HasSuffix(view.Content, "| [EOF] ===") {
		t.Errorf("expected [EOF] in footer, got:\n%s", view.Content)
	}
}

func TestWindowedFileScrollDown(t *testing.T) {
	dir := t.TempDir()
	content := "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\n"
	filePath := createTestFile(t, dir, "scroll.txt", content)

	w := NewWindowedFileReader(100)
	view, err := w.Open(filePath, 1, 3)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	if view.StartLine != 1 || view.EndLine != 3 {
		t.Fatalf("initial view unexpected: %d-%d", view.StartLine, view.EndLine)
	}

	// ScrollDown(2) -> advances to line 3 (lines 3-5)
	view, err = w.ScrollDown(2)
	if err != nil {
		t.Fatalf("ScrollDown(2) failed: %v", err)
	}
	if view.StartLine != 3 || view.EndLine != 5 {
		t.Errorf("expected lines 3-5, got %d-%d", view.StartLine, view.EndLine)
	}
	if !strings.Contains(view.Content, "3: l3\n4: l4\n5: l5\n") {
		t.Errorf("unexpected content:\n%s", view.Content)
	}
	if strings.Contains(view.Content, "[EOF]") {
		t.Errorf("expected not EOF at line 5 of 10")
	}

	// ScrollDown(3) -> advances to line 6 (lines 6-8)
	view, err = w.ScrollDown(3)
	if err != nil {
		t.Fatalf("ScrollDown(3) failed: %v", err)
	}
	if view.StartLine != 6 || view.EndLine != 8 {
		t.Errorf("expected lines 6-8, got %d-%d", view.StartLine, view.EndLine)
	}

	// ScrollDown(5) -> clamps gracefully at EOF (maxStart = 10 - 3 + 1 = 8, lines 8-10)
	view, err = w.ScrollDown(5)
	if err != nil {
		t.Fatalf("ScrollDown(5) failed: %v", err)
	}
	if view.StartLine != 8 || view.EndLine != 10 {
		t.Errorf("expected lines 8-10 at EOF, got %d-%d", view.StartLine, view.EndLine)
	}
	if !strings.HasSuffix(view.Content, "| [EOF] ===") {
		t.Errorf("expected [EOF] footer when clamped at EOF, got:\n%s", view.Content)
	}

	// Further ScrollDown stays clamped at EOF
	view, err = w.ScrollDown(10)
	if err != nil {
		t.Fatalf("ScrollDown(10) failed: %v", err)
	}
	if view.StartLine != 8 || view.EndLine != 10 {
		t.Errorf("expected lines 8-10, got %d-%d", view.StartLine, view.EndLine)
	}

	// ScrollDown(0) is a no-op
	view, err = w.ScrollDown(0)
	if err != nil {
		t.Fatalf("ScrollDown(0) failed: %v", err)
	}
	if view.StartLine != 8 || view.EndLine != 10 {
		t.Errorf("expected lines 8-10, got %d-%d", view.StartLine, view.EndLine)
	}

	// Negative ScrollDown scrolls up
	view, err = w.ScrollDown(-2)
	if err != nil {
		t.Fatalf("ScrollDown(-2) failed: %v", err)
	}
	if view.StartLine != 6 || view.EndLine != 8 {
		t.Errorf("expected lines 6-8 after ScrollDown(-2), got %d-%d", view.StartLine, view.EndLine)
	}
}

func TestWindowedFileScrollUp(t *testing.T) {
	dir := t.TempDir()
	content := "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\n"
	filePath := createTestFile(t, dir, "scrollup.txt", content)

	w := NewWindowedFileReader(100)
	view, err := w.Open(filePath, 8, 3)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	if view.StartLine != 8 || view.EndLine != 10 {
		t.Fatalf("initial view unexpected: %d-%d", view.StartLine, view.EndLine)
	}

	// ScrollUp(3) -> moves back to line 5 (lines 5-7)
	view, err = w.ScrollUp(3)
	if err != nil {
		t.Fatalf("ScrollUp(3) failed: %v", err)
	}
	if view.StartLine != 5 || view.EndLine != 7 {
		t.Errorf("expected lines 5-7, got %d-%d", view.StartLine, view.EndLine)
	}

	// ScrollUp(10) -> clamps to line 1 (lines 1-3)
	view, err = w.ScrollUp(10)
	if err != nil {
		t.Fatalf("ScrollUp(10) failed: %v", err)
	}
	if view.StartLine != 1 || view.EndLine != 3 {
		t.Errorf("expected lines 1-3 when clamped at top, got %d-%d", view.StartLine, view.EndLine)
	}

	// Further ScrollUp stays clamped at line 1
	view, err = w.ScrollUp(5)
	if err != nil {
		t.Fatalf("ScrollUp(5) failed: %v", err)
	}
	if view.StartLine != 1 || view.EndLine != 3 {
		t.Errorf("expected lines 1-3, got %d-%d", view.StartLine, view.EndLine)
	}

	// ScrollUp(0) is a no-op
	view, err = w.ScrollUp(0)
	if err != nil {
		t.Fatalf("ScrollUp(0) failed: %v", err)
	}
	if view.StartLine != 1 || view.EndLine != 3 {
		t.Errorf("expected lines 1-3, got %d-%d", view.StartLine, view.EndLine)
	}

	// Negative ScrollUp scrolls down
	view, err = w.ScrollUp(-2)
	if err != nil {
		t.Fatalf("ScrollUp(-2) failed: %v", err)
	}
	if view.StartLine != 3 || view.EndLine != 5 {
		t.Errorf("expected lines 3-5 after ScrollUp(-2), got %d-%d", view.StartLine, view.EndLine)
	}
}

func TestWindowedFileGoto(t *testing.T) {
	dir := t.TempDir()
	content := "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\n"
	filePath := createTestFile(t, dir, "goto.txt", content)

	w := NewWindowedFileReader(100)
	_, err := w.Open(filePath, 1, 3)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	// Goto(5) -> target line 5 is at top (lines 5-7)
	view, err := w.Goto(5)
	if err != nil {
		t.Fatalf("Goto(5) failed: %v", err)
	}
	if view.StartLine != 5 || view.EndLine != 7 {
		t.Errorf("expected lines 5-7, got %d-%d", view.StartLine, view.EndLine)
	}
	if !strings.Contains(view.Content, "5: l5\n6: l6\n7: l7\n") {
		t.Errorf("unexpected content:\n%s", view.Content)
	}

	// Goto(-10) -> clamps to line 1
	view, err = w.Goto(-10)
	if err != nil {
		t.Fatalf("Goto(-10) failed: %v", err)
	}
	if view.StartLine != 1 || view.EndLine != 3 {
		t.Errorf("expected lines 1-3 after clamping to min, got %d-%d", view.StartLine, view.EndLine)
	}

	// Goto(50) -> clamps to TotalLines (10)
	view, err = w.Goto(50)
	if err != nil {
		t.Fatalf("Goto(50) failed: %v", err)
	}
	if view.StartLine != 10 || view.EndLine != 10 {
		t.Errorf("expected lines 10-10 after clamping to EOF, got %d-%d", view.StartLine, view.EndLine)
	}
	if !strings.HasSuffix(view.Content, "| [EOF] ===") {
		t.Errorf("expected [EOF] footer, got:\n%s", view.Content)
	}

	// Goto(1) returns to line 1
	view, err = w.Goto(1)
	if err != nil {
		t.Fatalf("Goto(1) failed: %v", err)
	}
	if view.StartLine != 1 || view.EndLine != 3 {
		t.Errorf("expected lines 1-3, got %d-%d", view.StartLine, view.EndLine)
	}
}

func TestWindowedFileEmptyAndSmall(t *testing.T) {
	dir := t.TempDir()

	// 1. Empty file
	emptyPath := createTestFile(t, dir, "empty.txt", "")
	w := NewWindowedFileReader(100)

	view, err := w.Open(emptyPath, 1, 10)
	if err != nil {
		t.Fatalf("Open empty file failed: %v", err)
	}
	if view.TotalLines != 0 {
		t.Errorf("expected 0 TotalLines, got %d", view.TotalLines)
	}
	if view.StartLine != 0 || view.EndLine != 0 {
		t.Errorf("expected start=0, end=0 for empty file, got start=%d, end=%d", view.StartLine, view.EndLine)
	}
	expectedHeader := fmt.Sprintf("=== [%s] (lines 0-0 of 0) ===\n", emptyPath)
	if !strings.HasPrefix(view.Content, expectedHeader) {
		t.Errorf("expected header %q, got content:\n%s", expectedHeader, view.Content)
	}
	if !strings.HasSuffix(view.Content, "| [EOF] ===") {
		t.Errorf("expected [EOF] footer for empty file, got:\n%s", view.Content)
	}

	// Navigation on empty file remains safe
	view, err = w.ScrollDown(5)
	if err != nil || view.TotalLines != 0 {
		t.Errorf("ScrollDown on empty file failed: err=%v, view=%v", err, view)
	}
	view, err = w.ScrollUp(5)
	if err != nil || view.TotalLines != 0 {
		t.Errorf("ScrollUp on empty file failed: err=%v, view=%v", err, view)
	}
	view, err = w.Goto(5)
	if err != nil || view.TotalLines != 0 {
		t.Errorf("Goto on empty file failed: err=%v, view=%v", err, view)
	}

	// 2. Small file (smaller than window size)
	smallPath := createTestFile(t, dir, "small.txt", "alpha\nbeta\ngamma\n")
	view, err = w.Open(smallPath, 1, 10)
	if err != nil {
		t.Fatalf("Open small file failed: %v", err)
	}
	if view.TotalLines != 3 || view.StartLine != 1 || view.EndLine != 3 {
		t.Errorf("expected lines 1-3 of 3, got start=%d, end=%d, total=%d", view.StartLine, view.EndLine, view.TotalLines)
	}
	if !strings.HasSuffix(view.Content, "| [EOF] ===") {
		t.Errorf("expected [EOF] footer for small file fitting in window, got:\n%s", view.Content)
	}

	// ScrollDown on small file stays clamped at line 1
	view, err = w.ScrollDown(5)
	if err != nil {
		t.Fatalf("ScrollDown on small file failed: %v", err)
	}
	if view.StartLine != 1 || view.EndLine != 3 {
		t.Errorf("expected lines 1-3 after ScrollDown, got %d-%d", view.StartLine, view.EndLine)
	}
}

func TestWindowedFileNonExistent(t *testing.T) {
	w := NewWindowedFileReader(100)

	// Open missing file
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

	// Navigation before open
	if _, err := w.ScrollDown(1); !errors.Is(err, ErrNoFileOpen) {
		t.Errorf("expected ErrNoFileOpen from ScrollDown, got %v", err)
	}
	if _, err := w.ScrollUp(1); !errors.Is(err, ErrNoFileOpen) {
		t.Errorf("expected ErrNoFileOpen from ScrollUp, got %v", err)
	}
	if _, err := w.Goto(1); !errors.Is(err, ErrNoFileOpen) {
		t.Errorf("expected ErrNoFileOpen from Goto, got %v", err)
	}
	if _, err := w.View(); !errors.Is(err, ErrNoFileOpen) {
		t.Errorf("expected ErrNoFileOpen from View, got %v", err)
	}

	// Close resets open state
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

func TestWindowedFileConcurrency(t *testing.T) {
	dir := t.TempDir()
	var content strings.Builder
	for i := 1; i <= 100; i++ {
		fmt.Fprintf(&content, "line %d\n", i)
	}
	filePath := createTestFile(t, dir, "concurrency.txt", content.String())

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
				switch (id + j) % 5 {
				case 0:
					_, _ = w.ScrollDown(2)
				case 1:
					_, _ = w.ScrollUp(2)
				case 2:
					_, _ = w.Goto((id*7 + j) % 100)
				case 3:
					_, _, _, _ = w.CurrentPosition()
				case 4:
					_, _ = w.View()
				}
			}
		}(i)
	}

	wg.Wait()
}

func TestWindowedFileLineEndings(t *testing.T) {
	dir := t.TempDir()

	crlfPath := createTestFile(t, dir, "crlf.txt", "line1\r\nline2\r\nline3\r\n")
	crPath := createTestFile(t, dir, "cr.txt", "lineA\rlineB\rlineC\r")
	mixedPath := createTestFile(t, dir, "mixed.txt", "alpha\r\nbeta\rgamma\ndelta")

	w := NewWindowedFileReader(100)

	// CRLF
	view, err := w.Open(crlfPath, 1, 10)
	if err != nil {
		t.Fatalf("Open CRLF failed: %v", err)
	}
	if view.TotalLines != 3 || !strings.Contains(view.Content, "1: line1\n2: line2\n3: line3\n") {
		t.Errorf("unexpected CRLF view: %v", view)
	}

	// CR
	view, err = w.Open(crPath, 1, 10)
	if err != nil {
		t.Fatalf("Open CR failed: %v", err)
	}
	if view.TotalLines != 3 || !strings.Contains(view.Content, "1: lineA\n2: lineB\n3: lineC\n") {
		t.Errorf("unexpected CR view: %v", view)
	}

	// Mixed without trailing newline
	view, err = w.Open(mixedPath, 1, 10)
	if err != nil {
		t.Fatalf("Open mixed failed: %v", err)
	}
	if view.TotalLines != 4 || !strings.Contains(view.Content, "1: alpha\n2: beta\n3: gamma\n4: delta\n") {
		t.Errorf("unexpected mixed view: %v", view)
	}
}

func TestWindowedFilePathConfinement(t *testing.T) {
	root := t.TempDir()
	createTestFile(t, root, "inside.txt", "inside content\nline 2\n")
	createTestFile(t, root, "sub/deep.txt", "deep content\n")

	outsideDir := t.TempDir()
	createTestFile(t, outsideDir, "secret.txt", "secret content\n")

	w := NewWindowedFileReader(root)
	if w.RootDir() != root {
		t.Errorf("expected RootDir %q, got %q", root, w.RootDir())
	}

	// Valid relative paths
	view, err := w.Open("inside.txt", 1, 10)
	if err != nil {
		t.Fatalf("failed to open valid relative path: %v", err)
	}
	if view.TotalLines != 2 {
		t.Errorf("expected 2 lines, got %d", view.TotalLines)
	}

	if _, err := w.Open("sub/deep.txt", 1, 10); err != nil {
		t.Errorf("failed to open valid nested relative path: %v", err)
	}

	// Valid absolute path within root
	absInside := filepath.Join(root, "inside.txt")
	if _, err := w.Open(absInside, 1, 10); err != nil {
		t.Errorf("failed to open valid absolute path inside root: %v", err)
	}

	// Traversal escapes
	traversalCases := []string{
		"../secret.txt",
		"sub/../../secret.txt",
		filepath.Join(outsideDir, "secret.txt"),
		"..",
	}
	for _, tc := range traversalCases {
		_, err := w.Open(tc, 1, 10)
		if err == nil {
			t.Errorf("expected error for escaping path %q, got nil", tc)
		} else if !errors.Is(err, ErrPathEscape) && !errors.Is(err, ErrInvalidPath) {
			t.Errorf("expected ErrPathEscape or ErrInvalidPath for %q, got: %v", tc, err)
		}
	}

	// Invalid paths
	invalidCases := []string{
		"",
		"   ",
		"file\x00withnul.txt",
	}
	for _, ic := range invalidCases {
		_, err := w.Open(ic, 1, 10)
		if err == nil {
			t.Errorf("expected error for invalid path %q, got nil", ic)
		} else if !errors.Is(err, ErrInvalidPath) {
			t.Errorf("expected ErrInvalidPath for %q, got: %v", ic, err)
		}
	}

	// Symlink escape test
	symlinkTarget := filepath.Join(outsideDir, "secret.txt")
	symlinkPath := filepath.Join(root, "escapelink.txt")
	if err := os.Symlink(symlinkTarget, symlinkPath); err == nil {
		_, err := w.Open("escapelink.txt", 1, 10)
		if err == nil {
			t.Error("expected symlink escape to be refused, got nil error")
		} else if !errors.Is(err, ErrPathEscape) {
			t.Errorf("expected ErrPathEscape for symlink escape, got: %v", err)
		}
	}

	// Symlink within root
	validSymlink := filepath.Join(root, "validlink.txt")
	if err := os.Symlink(filepath.Join(root, "inside.txt"), validSymlink); err == nil {
		if _, err := w.Open("validlink.txt", 1, 10); err != nil {
			t.Errorf("expected valid internal symlink to succeed, got: %v", err)
		}
	}

	// Dynamic SetRootDir
	newRoot := t.TempDir()
	createTestFile(t, newRoot, "new.txt", "new content\n")
	w.SetRootDir(newRoot)
	if w.RootDir() != newRoot {
		t.Errorf("expected new root %q, got %q", newRoot, w.RootDir())
	}
	if _, err := w.Open("new.txt", 1, 10); err != nil {
		t.Errorf("expected Open under new root to succeed, got: %v", err)
	}
	// Old path now escapes new root
	if _, err := w.Open(absInside, 1, 10); !errors.Is(err, ErrPathEscape) {
		t.Errorf("expected old path to escape new root with ErrPathEscape, got: %v", err)
	}
}

func TestWindowedFileUTF8Safety(t *testing.T) {
	dir := t.TempDir()

	// 1. Multibyte UTF-8 characters (emojis, CJK, accents)
	utf8Content := "Line 1: こんにちは世界\nLine 2: 🚀 Rocket Launch\nLine 3: Café & résumé\n"
	utf8Path := createTestFile(t, dir, "multibyte.txt", utf8Content)

	w := NewWindowedFileReader(100)
	view, err := w.Open(utf8Path, 1, 10)
	if err != nil {
		t.Fatalf("Open multibyte UTF-8 file failed: %v", err)
	}
	if !utf8.ValidString(view.Content) {
		t.Errorf("expected valid UTF-8 in view content")
	}
	if !strings.Contains(view.Content, "こんにちは世界") || !strings.Contains(view.Content, "🚀 Rocket Launch") {
		t.Errorf("expected multibyte runes preserved, got:\n%s", view.Content)
	}

	// 2. Invalid UTF-8 bytes sanitized
	invalidBytes := []byte("header\ncorrupt: \xff\xfe data\nfooter\n")
	invalidPath := filepath.Join(dir, "invalid.txt")
	if err := os.WriteFile(invalidPath, invalidBytes, 0644); err != nil {
		t.Fatalf("write invalid UTF-8 file failed: %v", err)
	}

	view, err = w.Open(invalidPath, 1, 10)
	if err != nil {
		t.Fatalf("Open invalid UTF-8 file failed: %v", err)
	}
	if !utf8.ValidString(view.Content) {
		t.Errorf("expected output to be sanitized to valid UTF-8, but got invalid UTF-8 string")
	}
	if !strings.Contains(view.Content, "\uFFFD") {
		t.Errorf("expected replacement character \\uFFFD for invalid bytes, got:\n%s", view.Content)
	}

	// 3. UTF-8 BOM stripped cleanly
	bomBytes := append([]byte("\xef\xbb\xbf"), []byte("bom line 1\nbom line 2\n")...)
	bomPath := filepath.Join(dir, "bom.txt")
	if err := os.WriteFile(bomPath, bomBytes, 0644); err != nil {
		t.Fatalf("write BOM file failed: %v", err)
	}

	view, err = w.Open(bomPath, 1, 10)
	if err != nil {
		t.Fatalf("Open BOM file failed: %v", err)
	}
	if !utf8.ValidString(view.Content) {
		t.Errorf("expected valid UTF-8 with BOM file")
	}
	if !strings.Contains(view.Content, "1: bom line 1") {
		t.Errorf("expected clean BOM stripping on line 1, got:\n%s", view.Content)
	}
	if strings.Contains(view.Content, "\ufeff") {
		t.Errorf("expected BOM character to be stripped from content")
	}
}

func TestWindowedFileSetWindowSize(t *testing.T) {
	dir := t.TempDir()
	filePath := createTestFile(t, dir, "sample.txt", "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n")
	w := NewWindowedFileReader(3)
	view, err := w.Open(filePath, 1, 3)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	if view.EndLine != 3 {
		t.Errorf("expected EndLine 3, got %d", view.EndLine)
	}
	w.SetWindowSize(5)
	view, err = w.View()
	if err != nil {
		t.Fatalf("View failed: %v", err)
	}
	if view.EndLine != 5 {
		t.Errorf("expected EndLine 5 after SetWindowSize(5), got %d", view.EndLine)
	}
}

func TestWindowedFileReader_BinaryAndSizeGuard(t *testing.T) {
	dir := t.TempDir()

	// (a) A file exceeding MaxFileSize returns ErrFileTooLarge.
	t.Run("ExceedsMaxFileSize", func(t *testing.T) {
		largeContent := strings.Repeat("a", 200)
		largePath := createTestFile(t, dir, "large.txt", largeContent)

		w := NewWindowedFileReader()
		w.SetMaxFileSize(100)
		if w.MaxFileSize() != 100 {
			t.Fatalf("expected MaxFileSize 100, got %d", w.MaxFileSize())
		}

		view, err := w.Open(largePath, 1, 10)
		if err == nil {
			t.Fatal("expected error for file exceeding MaxFileSize, got nil")
		}
		if !errors.Is(err, ErrFileTooLarge) {
			t.Fatalf("expected ErrFileTooLarge, got %v", err)
		}
		if view != nil {
			t.Fatalf("expected nil view on ErrFileTooLarge, got %v", view)
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
	})

	// (b) A file containing NUL bytes returns ErrBinaryFile.
	t.Run("BinaryFileDetected", func(t *testing.T) {
		// NUL in header
		headerBinaryContent := "hello\x00world\nline 2\n"
		headerBinaryPath := filepath.Join(dir, "binary_header.bin")
		if err := os.WriteFile(headerBinaryPath, []byte(headerBinaryContent), 0644); err != nil {
			t.Fatalf("failed to write binary file: %v", err)
		}

		w := NewWindowedFileReader()
		view, err := w.Open(headerBinaryPath, 1, 10)
		if err == nil {
			t.Fatal("expected error for binary file, got nil")
		}
		if !errors.Is(err, ErrBinaryFile) {
			t.Fatalf("expected ErrBinaryFile, got %v", err)
		}
		if view != nil {
			t.Fatalf("expected nil view on ErrBinaryFile, got %v", view)
		}
		if w.IsOpen() {
			t.Fatal("expected IsOpen to be false after binary file rejection")
		}

		// NUL past 8KB buffer
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

	// (c) Normal text files continue to open and paginate cleanly.
	t.Run("NormalTextFilePaginatesCleanly", func(t *testing.T) {
		textContent := "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8\n"
		textPath := createTestFile(t, dir, "normal.txt", textContent)

		w := NewWindowedFileReader(3)
		view, err := w.Open(textPath, 1, 3)
		if err != nil {
			t.Fatalf("unexpected error opening normal text file: %v", err)
		}
		if view == nil {
			t.Fatal("expected non-nil view")
		}
		if view.TotalLines != 8 {
			t.Fatalf("expected 8 TotalLines, got %d", view.TotalLines)
		}
		if view.StartLine != 1 || view.EndLine != 3 {
			t.Fatalf("expected lines 1-3, got %d-%d", view.StartLine, view.EndLine)
		}

		// Scroll down
		view, err = w.ScrollDown(2)
		if err != nil {
			t.Fatalf("ScrollDown failed: %v", err)
		}
		if view.StartLine != 3 || view.EndLine != 5 {
			t.Fatalf("expected lines 3-5 after ScrollDown(2), got %d-%d", view.StartLine, view.EndLine)
		}

		// Scroll up
		view, err = w.ScrollUp(1)
		if err != nil {
			t.Fatalf("ScrollUp failed: %v", err)
		}
		if view.StartLine != 2 || view.EndLine != 4 {
			t.Fatalf("expected lines 2-4 after ScrollUp(1), got %d-%d", view.StartLine, view.EndLine)
		}

		// Goto
		view, err = w.Goto(6)
		if err != nil {
			t.Fatalf("Goto failed: %v", err)
		}
		if view.StartLine != 6 || view.EndLine != 8 {
			t.Fatalf("expected lines 6-8 after Goto(6), got %d-%d", view.StartLine, view.EndLine)
		}
	})
}
