package toolbound

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// DefaultWindowSize is the default number of lines displayed in a window when unspecified.
const DefaultWindowSize = 100

var (
	// ErrNoFileOpen indicates navigation was attempted when no file is open.
	ErrNoFileOpen = errors.New("no file open")
	// ErrPathEscape indicates a path attempted to traverse outside the allowed root directory.
	ErrPathEscape = errors.New("path escapes root directory")
	// ErrInvalidPath indicates a malformed or invalid path argument.
	ErrInvalidPath = errors.New("invalid path")
)

// WindowedFileReader provides stateful, windowed file reading with relative
// line navigation to bound tool observation bloat.
type WindowedFileReader struct {
	mu                sync.RWMutex
	rootDir           string
	defaultWindowSize int

	path         string
	resolvedPath string
	currentLine  int
	windowSize   int
	totalLines   int
	lines        []string
}

// NewWindowedFileReader creates a WindowedFileReader confined to rootDir.
// If rootDir is omitted or empty, it defaults to the current working directory.
func NewWindowedFileReader(rootDir ...string) *WindowedFileReader {
	root := ""
	if len(rootDir) > 0 {
		root = rootDir[0]
	}
	return &WindowedFileReader{
		rootDir:           root,
		defaultWindowSize: DefaultWindowSize,
		currentLine:       1,
	}
}

// RootDir returns the configured root confinement directory.
func (w *WindowedFileReader) RootDir() string {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.rootDir
}

// SetRootDir sets the root confinement directory.
func (w *WindowedFileReader) SetRootDir(root string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.rootDir = root
}

// DefaultWindowSize returns the configured default window size.
func (w *WindowedFileReader) DefaultWindowSize() int {
	w.mu.RLock()
	defer w.mu.RUnlock()
	if w.defaultWindowSize <= 0 {
		return DefaultWindowSize
	}
	return w.defaultWindowSize
}

// SetDefaultWindowSize updates the default window size.
func (w *WindowedFileReader) SetDefaultWindowSize(size int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if size <= 0 {
		size = DefaultWindowSize
	}
	w.defaultWindowSize = size
}

// Open opens the file at path, sets the initial line and window size, and
// returns the window slice formatted with 1-based line numbers.
//
// If windowSize <= 0, it defaults to 100.
// If line <= 0, it defaults to 1.
// If line exceeds total lines, it is clamped to EOF (totalLines, or 1 if empty).
func (w *WindowedFileReader) Open(path string, line int, windowSize int) ([]string, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	resolved, err := w.resolvePathLocked(path)
	if err != nil {
		return nil, err
	}

	data, err := os.ReadFile(resolved)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	parsedLines := splitLines(data)

	if windowSize <= 0 {
		if w.defaultWindowSize > 0 {
			windowSize = w.defaultWindowSize
		} else {
			windowSize = DefaultWindowSize
		}
	}

	total := len(parsedLines)
	if line <= 0 {
		line = 1
	}
	if total > 0 && line > total {
		line = total
	} else if total == 0 {
		line = 1
	}

	w.path = path
	w.resolvedPath = resolved
	w.lines = parsedLines
	w.totalLines = total
	w.currentLine = line
	w.windowSize = windowSize

	return w.windowSliceLocked(), nil
}

// ScrollDown advances the window forward by n lines and returns the new window slice.
// If n is negative, it retreats backwards by -n lines.
// Window start line is clamped to totalLines (EOF clamping).
func (w *WindowedFileReader) ScrollDown(n int) ([]string, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.isOpenLocked() {
		return nil, ErrNoFileOpen
	}
	return w.scrollDownLocked(n), nil
}

// ScrollUp retreats the window backward by n lines and returns the new window slice.
// If n is negative, it advances forward by -n lines.
// Window start line is clamped to line 1.
func (w *WindowedFileReader) ScrollUp(n int) ([]string, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.isOpenLocked() {
		return nil, ErrNoFileOpen
	}
	return w.scrollUpLocked(n), nil
}

// Goto jumps the window start to the specified line number and returns the new window slice.
// Line numbers are clamped between 1 and totalLines (or 1 for an empty file).
func (w *WindowedFileReader) Goto(line int) ([]string, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.isOpenLocked() {
		return nil, ErrNoFileOpen
	}

	if line < 1 {
		line = 1
	}
	if w.totalLines > 0 && line > w.totalLines {
		line = w.totalLines
	} else if w.totalLines == 0 {
		line = 1
	}

	w.currentLine = line
	return w.windowSliceLocked(), nil
}

// CurrentPosition returns the state of the currently open file:
// path, 1-based start line, windowSize, and totalLines.
// If no file is open, it returns ("", 0, 0, 0).
func (w *WindowedFileReader) CurrentPosition() (path string, line int, windowSize int, totalLines int) {
	w.mu.RLock()
	defer w.mu.RUnlock()

	if !w.isOpenLocked() {
		return "", 0, 0, 0
	}
	return w.path, w.currentLine, w.windowSize, w.totalLines
}

// Close unloads the currently opened file and resets position.
func (w *WindowedFileReader) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.path = ""
	w.resolvedPath = ""
	w.lines = nil
	w.totalLines = 0
	w.currentLine = 1
	return nil
}

// IsOpen reports whether a file is currently open.
func (w *WindowedFileReader) IsOpen() bool {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.isOpenLocked()
}

func (w *WindowedFileReader) isOpenLocked() bool {
	return w.lines != nil
}

func (w *WindowedFileReader) scrollDownLocked(n int) []string {
	if n < 0 {
		return w.scrollUpLocked(-n)
	}
	if n == 0 {
		return w.windowSliceLocked()
	}

	newLine := w.currentLine + n
	if w.totalLines > 0 && newLine > w.totalLines {
		newLine = w.totalLines
	} else if w.totalLines == 0 {
		newLine = 1
	}
	w.currentLine = newLine
	return w.windowSliceLocked()
}

func (w *WindowedFileReader) scrollUpLocked(n int) []string {
	if n < 0 {
		return w.scrollDownLocked(-n)
	}
	if n == 0 {
		return w.windowSliceLocked()
	}

	newLine := w.currentLine - n
	if newLine < 1 {
		newLine = 1
	}
	w.currentLine = newLine
	return w.windowSliceLocked()
}

func (w *WindowedFileReader) windowSliceLocked() []string {
	if len(w.lines) == 0 {
		return []string{}
	}

	start := w.currentLine
	if start < 1 {
		start = 1
	}
	if start > len(w.lines) {
		start = len(w.lines)
	}

	end := start + w.windowSize - 1
	if end > len(w.lines) {
		end = len(w.lines)
	}

	count := end - start + 1
	if count <= 0 {
		return []string{}
	}

	res := make([]string, count)
	for i := 0; i < count; i++ {
		lineNum := start + i
		res[i] = fmt.Sprintf("%d: %s", lineNum, w.lines[lineNum-1])
	}
	return res
}

func (w *WindowedFileReader) resolvePathLocked(path string) (string, error) {
	trimmed := strings.TrimSpace(path)
	if trimmed == "" {
		return "", fmt.Errorf("%w: empty path", ErrInvalidPath)
	}
	if strings.ContainsRune(trimmed, 0) {
		return "", fmt.Errorf("%w: path contains NUL byte", ErrInvalidPath)
	}

	root := w.rootDir
	if root == "" {
		root = "."
	}
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return "", fmt.Errorf("resolve root directory: %w", err)
	}
	absRoot = filepath.Clean(absRoot)

	// Normalize separators for cross-platform traversal checks
	normalized := filepath.FromSlash(strings.ReplaceAll(trimmed, "\\", "/"))

	var target string
	if filepath.IsAbs(normalized) {
		target = filepath.Clean(normalized)
	} else {
		target = filepath.Clean(filepath.Join(absRoot, normalized))
	}

	// 1. Lexical confinement check
	if !isPathWithin(absRoot, target) {
		return "", fmt.Errorf("%w: path %q escapes root %q", ErrPathEscape, path, absRoot)
	}

	// 2. Symlink evaluation check
	evalRoot, err := filepath.EvalSymlinks(absRoot)
	if err == nil {
		absRoot = evalRoot
	}

	evalTarget, err := filepath.EvalSymlinks(target)
	if err == nil {
		if !isPathWithin(absRoot, evalTarget) {
			return "", fmt.Errorf("%w: path %q escapes root via symlink to %q", ErrPathEscape, path, evalTarget)
		}
		target = evalTarget
	}

	info, err := os.Stat(target)
	if err != nil {
		return "", err
	}
	if info.IsDir() {
		return "", fmt.Errorf("%w: path %q is a directory", ErrInvalidPath, path)
	}

	return target, nil
}

func isPathWithin(base, target string) bool {
	rel, err := filepath.Rel(base, target)
	if err != nil {
		return false
	}
	if rel == "." {
		return true
	}
	if rel == ".." || hasPathDotDotPrefix(rel) {
		return false
	}
	return true
}

func hasPathDotDotPrefix(rel string) bool {
	return len(rel) >= 3 && rel[0] == '.' && rel[1] == '.' && (rel[2] == '/' || rel[2] == '\\')
}

func splitLines(data []byte) []string {
	if len(data) == 0 {
		return []string{}
	}
	s := string(data)
	s = strings.ReplaceAll(s, "\r\n", "\n")
	hasTrailingNewline := strings.HasSuffix(s, "\n")
	if hasTrailingNewline {
		s = s[:len(s)-1]
	}
	if s == "" && hasTrailingNewline {
		return []string{""}
	}
	return strings.Split(s, "\n")
}
