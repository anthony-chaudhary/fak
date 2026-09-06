package toolbound

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
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

func TestWindowedFileReader_Open(t *testing.T) {
	dir := t.TempDir()
	content := "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8\nline 9\nline 10\n"
	createTestFile(t, dir, "sample.txt", content)

	w := NewWindowedFileReader(dir)

	// Basic open with line 1 and window size 3
	lines, err := w.Open("sample.txt", 1, 3)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	if len(lines) != 3 {
		t.Fatalf("expected 3 lines, got %d", len(lines))
	}
	expected := []string{"1: line 1", "2: line 2", "3: line 3"}
	for i, exp := range expected {
		if lines[i] != exp {
			t.Errorf("line %d: expected %q, got %q", i, exp, lines[i])
		}
	}

	path, line, size, total := w.CurrentPosition()
	if path != "sample.txt" || line != 1 || size != 3 || total != 10 {
		t.Errorf("CurrentPosition unexpected: path=%q, line=%d, size=%d, total=%d", path, line, size, total)
	}

	// Open with defaults (windowSize <= 0, line <= 0)
	lines, err = w.Open("sample.txt", 0, 0)
	if err != nil {
		t.Fatalf("Open with defaults failed: %v", err)
	}
	if len(lines) != 10 {
		t.Fatalf("expected 10 lines with default windowSize, got %d", len(lines))
	}
	path, line, size, total = w.CurrentPosition()
	if line != 1 || size != DefaultWindowSize || total != 10 {
		t.Errorf("CurrentPosition defaults unexpected: line=%d, size=%d, total=%d", line, size, total)
	}

	// Open non-existent file
	if _, err := w.Open("missing.txt", 1, 10); err == nil {
		t.Error("expected error opening missing file, got nil")
	}

	// Open directory
	if _, err := w.Open(".", 1, 10); err == nil {
		t.Error("expected error opening directory, got nil")
	}
}

func TestWindowedFileReader_LineNumberFormatting(t *testing.T) {
	dir := t.TempDir()
	content := "first line\n\n  indented line\n"
	createTestFile(t, dir, "format.txt", content)

	w := NewWindowedFileReader(dir)
	lines, err := w.Open("format.txt", 1, 10)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	if len(lines) != 3 {
		t.Fatalf("expected 3 lines, got %d", len(lines))
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
}

func TestWindowedFileReader_ScrollDown(t *testing.T) {
	dir := t.TempDir()
	content := "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\n"
	createTestFile(t, dir, "scroll.txt", content)

	w := NewWindowedFileReader(dir)
	_, err := w.Open("scroll.txt", 1, 3)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	// Scroll down 2 lines -> line 3
	lines, err := w.ScrollDown(2)
	if err != nil {
		t.Fatalf("ScrollDown failed: %v", err)
	}
	if len(lines) != 3 || lines[0] != "3: l3" || lines[2] != "5: l5" {
		t.Errorf("unexpected lines after ScrollDown(2): %v", lines)
	}
	_, line, _, _ := w.CurrentPosition()
	if line != 3 {
		t.Errorf("expected line 3, got %d", line)
	}

	// Scroll down 3 lines -> line 6
	lines, err = w.ScrollDown(3)
	if err != nil {
		t.Fatalf("ScrollDown failed: %v", err)
	}
	if len(lines) != 3 || lines[0] != "6: l6" || lines[2] != "8: l8" {
		t.Errorf("unexpected lines after ScrollDown(3): %v", lines)
	}

	// Scroll down 0 lines -> no-op
	lines, err = w.ScrollDown(0)
	if err != nil {
		t.Fatalf("ScrollDown(0) failed: %v", err)
	}
	if len(lines) != 3 || lines[0] != "6: l6" {
		t.Errorf("unexpected lines after ScrollDown(0): %v", lines)
	}

	// Scroll down negative lines -> retreats backward
	lines, err = w.ScrollDown(-2)
	if err != nil {
		t.Fatalf("ScrollDown(-2) failed: %v", err)
	}
	if len(lines) != 3 || lines[0] != "4: l4" {
		t.Errorf("unexpected lines after ScrollDown(-2): %v", lines)
	}
}

func TestWindowedFileReader_ScrollUp(t *testing.T) {
	dir := t.TempDir()
	content := "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\n"
	createTestFile(t, dir, "scrollup.txt", content)

	w := NewWindowedFileReader(dir)
	_, err := w.Open("scrollup.txt", 8, 3)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	// Scroll up 3 lines -> line 5
	lines, err := w.ScrollUp(3)
	if err != nil {
		t.Fatalf("ScrollUp failed: %v", err)
	}
	if len(lines) != 3 || lines[0] != "5: l5" || lines[2] != "7: l7" {
		t.Errorf("unexpected lines after ScrollUp(3): %v", lines)
	}
	_, line, _, _ := w.CurrentPosition()
	if line != 5 {
		t.Errorf("expected line 5, got %d", line)
	}

	// Scroll up past start (10 lines from 5) -> clamped to line 1
	lines, err = w.ScrollUp(10)
	if err != nil {
		t.Fatalf("ScrollUp(10) failed: %v", err)
	}
	if len(lines) != 3 || lines[0] != "1: l1" || lines[2] != "3: l3" {
		t.Errorf("unexpected lines after ScrollUp(10): %v", lines)
	}
	_, line, _, _ = w.CurrentPosition()
	if line != 1 {
		t.Errorf("expected line 1, got %d", line)
	}

	// Scroll up 0 lines -> no-op
	lines, err = w.ScrollUp(0)
	if err != nil {
		t.Fatalf("ScrollUp(0) failed: %v", err)
	}
	if len(lines) != 3 || lines[0] != "1: l1" {
		t.Errorf("unexpected lines after ScrollUp(0): %v", lines)
	}

	// Scroll up negative lines -> advances forward
	lines, err = w.ScrollUp(-2)
	if err != nil {
		t.Fatalf("ScrollUp(-2) failed: %v", err)
	}
	if len(lines) != 3 || lines[0] != "3: l3" {
		t.Errorf("unexpected lines after ScrollUp(-2): %v", lines)
	}
}

func TestWindowedFileReader_Goto(t *testing.T) {
	dir := t.TempDir()
	content := "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\n"
	createTestFile(t, dir, "goto.txt", content)

	w := NewWindowedFileReader(dir)
	_, err := w.Open("goto.txt", 1, 3)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	// Goto line 5
	lines, err := w.Goto(5)
	if err != nil {
		t.Fatalf("Goto(5) failed: %v", err)
	}
	if len(lines) != 3 || lines[0] != "5: l5" || lines[2] != "7: l7" {
		t.Errorf("unexpected lines after Goto(5): %v", lines)
	}
	_, line, _, _ := w.CurrentPosition()
	if line != 5 {
		t.Errorf("expected line 5, got %d", line)
	}

	// Goto line <= 0 -> clamped to 1
	lines, err = w.Goto(-5)
	if err != nil {
		t.Fatalf("Goto(-5) failed: %v", err)
	}
	if len(lines) != 3 || lines[0] != "1: l1" {
		t.Errorf("unexpected lines after Goto(-5): %v", lines)
	}
	_, line, _, _ = w.CurrentPosition()
	if line != 1 {
		t.Errorf("expected line 1, got %d", line)
	}

	// Goto past EOF -> clamped to totalLines (10)
	lines, err = w.Goto(50)
	if err != nil {
		t.Fatalf("Goto(50) failed: %v", err)
	}
	if len(lines) != 1 || lines[0] != "10: l10" {
		t.Errorf("unexpected lines after Goto(50): %v", lines)
	}
	_, line, _, _ = w.CurrentPosition()
	if line != 10 {
		t.Errorf("expected line 10, got %d", line)
	}
}

func TestWindowedFileReader_EOFClamping(t *testing.T) {
	dir := t.TempDir()
	content := "a\nb\nc\nd\ne\n"
	createTestFile(t, dir, "five.txt", content)

	w := NewWindowedFileReader(dir)

	// Open near EOF with window size 3
	lines, err := w.Open("five.txt", 4, 3)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	if len(lines) != 2 {
		t.Fatalf("expected 2 lines clamped at EOF, got %d", len(lines))
	}
	if lines[0] != "4: d" || lines[1] != "5: e" {
		t.Errorf("unexpected lines: %v", lines)
	}

	// Scroll down past EOF -> clamped to line 5
	lines, err = w.ScrollDown(10)
	if err != nil {
		t.Fatalf("ScrollDown failed: %v", err)
	}
	if len(lines) != 1 || lines[0] != "5: e" {
		t.Errorf("expected single EOF line [5: e], got %v", lines)
	}
	_, line, _, _ := w.CurrentPosition()
	if line != 5 {
		t.Errorf("expected line 5, got %d", line)
	}

	// Scroll down again -> stays at EOF
	lines, err = w.ScrollDown(5)
	if err != nil {
		t.Fatalf("ScrollDown failed: %v", err)
	}
	if len(lines) != 1 || lines[0] != "5: e" {
		t.Errorf("expected single EOF line [5: e], got %v", lines)
	}

	// Open past EOF directly -> clamped to line 5
	lines, err = w.Open("five.txt", 999, 3)
	if err != nil {
		t.Fatalf("Open past EOF failed: %v", err)
	}
	if len(lines) != 1 || lines[0] != "5: e" {
		t.Errorf("expected single EOF line [5: e], got %v", lines)
	}

	// Window size larger than file -> returns all lines, clamped at EOF
	lines, err = w.Open("five.txt", 1, 100)
	if err != nil {
		t.Fatalf("Open large window failed: %v", err)
	}
	if len(lines) != 5 {
		t.Fatalf("expected 5 lines, got %d", len(lines))
	}
	if lines[4] != "5: e" {
		t.Errorf("expected last line 5: e, got %s", lines[4])
	}
}

func TestWindowedFileReader_EmptyFile(t *testing.T) {
	dir := t.TempDir()
	createTestFile(t, dir, "empty.txt", "")

	w := NewWindowedFileReader(dir)
	lines, err := w.Open("empty.txt", 1, 10)
	if err != nil {
		t.Fatalf("Open empty file failed: %v", err)
	}
	if len(lines) != 0 {
		t.Fatalf("expected 0 lines from empty file, got %d", len(lines))
	}

	path, line, size, total := w.CurrentPosition()
	if path != "empty.txt" || line != 1 || size != 10 || total != 0 {
		t.Errorf("CurrentPosition unexpected for empty file: path=%q, line=%d, size=%d, total=%d", path, line, size, total)
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
}

func TestWindowedFileReader_PathConfinement(t *testing.T) {
	root := t.TempDir()
	createTestFile(t, root, "inside.txt", "inside content")
	createTestFile(t, root, "sub/deep.txt", "deep content")

	outsideDir := t.TempDir()
	createTestFile(t, outsideDir, "secret.txt", "secret content")

	w := NewWindowedFileReader(root)

	// Valid relative paths
	if _, err := w.Open("inside.txt", 1, 10); err != nil {
		t.Errorf("failed to open valid relative path: %v", err)
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
}

func TestWindowedFileReader_NotOpenedAndClose(t *testing.T) {
	w := NewWindowedFileReader()

	if w.IsOpen() {
		t.Error("expected IsOpen to be false initially")
	}

	path, line, size, total := w.CurrentPosition()
	if path != "" || line != 0 || size != 0 || total != 0 {
		t.Errorf("expected empty position, got (%q, %d, %d, %d)", path, line, size, total)
	}

	if _, err := w.ScrollDown(1); !errors.Is(err, ErrNoFileOpen) {
		t.Errorf("expected ErrNoFileOpen, got %v", err)
	}
	if _, err := w.ScrollUp(1); !errors.Is(err, ErrNoFileOpen) {
		t.Errorf("expected ErrNoFileOpen, got %v", err)
	}
	if _, err := w.Goto(1); !errors.Is(err, ErrNoFileOpen) {
		t.Errorf("expected ErrNoFileOpen, got %v", err)
	}

	// Open and then close
	dir := t.TempDir()
	createTestFile(t, dir, "test.txt", "one\ntwo\n")
	w.SetRootDir(dir)

	if _, err := w.Open("test.txt", 1, 5); err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	if !w.IsOpen() {
		t.Error("expected IsOpen to be true after Open")
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

func TestWindowedFileReader_Concurrency(t *testing.T) {
	dir := t.TempDir()
	content := ""
	for i := 1; i <= 100; i++ {
		content += fmt.Sprintf("line %d\n", i)
	}
	createTestFile(t, dir, "concurrency.txt", content)

	w := NewWindowedFileReader(dir)
	if _, err := w.Open("concurrency.txt", 1, 10); err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	var wg sync.WaitGroup
	workers := 8
	iterations := 50

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				switch (workerID + j) % 4 {
				case 0:
					_, _ = w.ScrollDown(2)
				case 1:
					_, _ = w.ScrollUp(2)
				case 2:
					_, _ = w.Goto(workerID*5 + j)
				case 3:
					_, _, _, _ = w.CurrentPosition()
				}
			}
		}(i)
	}

	wg.Wait()
}
