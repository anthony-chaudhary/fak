package toolbound

import (
	"bytes"
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

// WindowView represents the formatted output of a file window.
type WindowView struct {
	Path       string
	StartLine  int
	EndLine    int
	TotalLines int
	Content    string
}

// WindowedFileReader represents an active windowed file reader state with relative line navigation.
type WindowedFileReader struct {
	mu sync.RWMutex

	rootDir           string
	defaultWindowSize int

	Path         string
	Lines        []string
	TotalLines   int
	CurrentStart int
	WindowSize   int
}

// NewWindowedFileReader creates a WindowedFileReader.
// Optional arguments may provide a root confinement directory (string)
// and/or a default window size (int).
// If unspecified, defaultWindowSize defaults to DefaultWindowSize (100) and rootDir is unset.
func NewWindowedFileReader(args ...any) *WindowedFileReader {
	size := DefaultWindowSize
	root := ""
	for _, arg := range args {
		switch v := arg.(type) {
		case int:
			if v > 0 {
				size = v
			}
		case string:
			root = v
		}
	}
	return &WindowedFileReader{
		rootDir:           root,
		defaultWindowSize: size,
		CurrentStart:      1,
		WindowSize:        size,
	}
}

// RootDir returns the configured root confinement directory.
func (w *WindowedFileReader) RootDir() string {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.rootDir
}

// SetRootDir sets the root confinement directory for path confinement.
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

// SetWindowSize updates the active window size.
func (w *WindowedFileReader) SetWindowSize(size int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if size <= 0 {
		if w.defaultWindowSize > 0 {
			size = w.defaultWindowSize
		} else {
			size = DefaultWindowSize
		}
	}
	w.WindowSize = size
}

// Open reads the file at path (using os.ReadFile), splits it into lines with UTF-8 safety,
// clamps startLine to [1, TotalLines] (or line 1 if empty),
// sets WindowSize (defaulting to defaultWindowSize if <= 0), and returns the WindowView.
// If rootDir is configured, path confinement is enforced against rootDir.
func (w *WindowedFileReader) Open(path string, startLine int, windowSize int) (*WindowView, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	resolved, err := w.resolvePathLocked(path)
	if err != nil {
		return nil, err
	}

	data, err := os.ReadFile(resolved)
	if err != nil {
		return nil, err
	}

	parsedLines := splitLines(data)
	total := len(parsedLines)

	if windowSize <= 0 {
		if w.defaultWindowSize > 0 {
			windowSize = w.defaultWindowSize
		} else {
			windowSize = DefaultWindowSize
		}
	}

	if total == 0 {
		startLine = 1
	} else {
		if startLine < 1 {
			startLine = 1
		} else if startLine > total {
			startLine = total
		}
	}

	w.Path = path
	w.Lines = parsedLines
	w.TotalLines = total
	w.CurrentStart = startLine
	w.WindowSize = windowSize

	return w.viewLocked(), nil
}

// ScrollDown advances CurrentStart by n lines. If CurrentStart + WindowSize > TotalLines, it clamps gracefully.
// If n < 0, it calls ScrollUp(-n).
// Returns the updated WindowView.
func (w *WindowedFileReader) ScrollDown(n int) (*WindowView, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.isOpenLocked() {
		return nil, ErrNoFileOpen
	}

	if n < 0 {
		return w.scrollUpLocked(-n), nil
	}
	if n == 0 {
		return w.viewLocked(), nil
	}

	return w.scrollDownLocked(n), nil
}

// ScrollUp decrements CurrentStart by n lines. Clamps to line 1.
// If n < 0, it calls ScrollDown(-n).
// Returns the updated WindowView.
func (w *WindowedFileReader) ScrollUp(n int) (*WindowView, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.isOpenLocked() {
		return nil, ErrNoFileOpen
	}

	if n < 0 {
		return w.scrollDownLocked(-n), nil
	}
	if n == 0 {
		return w.viewLocked(), nil
	}

	return w.scrollUpLocked(n), nil
}

// Goto jumps CurrentStart so that target line is at the top.
// Clamps to valid range [1, TotalLines] (or line 1 if empty).
// Returns the updated WindowView.
func (w *WindowedFileReader) Goto(line int) (*WindowView, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.isOpenLocked() {
		return nil, ErrNoFileOpen
	}

	if w.TotalLines == 0 {
		w.CurrentStart = 1
		return w.viewLocked(), nil
	}

	if line < 1 {
		line = 1
	} else if line > w.TotalLines {
		line = w.TotalLines
	}

	w.CurrentStart = line
	return w.viewLocked(), nil
}

// CurrentPosition returns the current path, start line, window size, and total lines.
// If no file is open, it returns ("", 0, 0, 0).
func (w *WindowedFileReader) CurrentPosition() (path string, currentStart int, windowSize int, totalLines int) {
	w.mu.RLock()
	defer w.mu.RUnlock()

	if !w.isOpenLocked() {
		return "", 0, 0, 0
	}
	return w.Path, w.CurrentStart, w.WindowSize, w.TotalLines
}

// View returns the current WindowView without moving the current line position.
func (w *WindowedFileReader) View() (*WindowView, error) {
	w.mu.RLock()
	defer w.mu.RUnlock()

	if !w.isOpenLocked() {
		return nil, ErrNoFileOpen
	}
	return w.viewLocked(), nil
}

// Close unloads the currently opened file and resets position.
func (w *WindowedFileReader) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.Path = ""
	w.Lines = nil
	w.TotalLines = 0
	w.CurrentStart = 1
	return nil
}

// IsOpen reports whether a file is currently open in the reader.
func (w *WindowedFileReader) IsOpen() bool {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.isOpenLocked()
}

func (w *WindowedFileReader) isOpenLocked() bool {
	return w.Lines != nil
}

func (w *WindowedFileReader) scrollDownLocked(n int) *WindowView {
	if w.TotalLines == 0 {
		w.CurrentStart = 1
		return w.viewLocked()
	}

	target := w.CurrentStart + n
	maxStart := w.TotalLines - w.WindowSize + 1
	if maxStart < 1 {
		maxStart = 1
	}

	if target > maxStart {
		if w.CurrentStart > maxStart {
			// Positioned past maxStart (e.g. from Goto); clamp to TotalLines
			if target > w.TotalLines {
				target = w.TotalLines
			}
		} else {
			target = maxStart
		}
	}
	w.CurrentStart = target
	return w.viewLocked()
}

func (w *WindowedFileReader) scrollUpLocked(n int) *WindowView {
	if w.TotalLines == 0 {
		w.CurrentStart = 1
		return w.viewLocked()
	}

	target := w.CurrentStart - n
	if target < 1 {
		target = 1
	}
	w.CurrentStart = target
	return w.viewLocked()
}

func (w *WindowedFileReader) viewLocked() *WindowView {
	total := w.TotalLines
	path := w.Path
	windowSize := w.WindowSize

	if total == 0 {
		content := fmt.Sprintf("=== [%s] (lines 0-0 of 0) ===\n\n=== Navigation: ScrollDown(n), ScrollUp(n), Goto(line) | [EOF] ===", path)
		return &WindowView{
			Path:       path,
			StartLine:  0,
			EndLine:    0,
			TotalLines: 0,
			Content:    content,
		}
	}

	startLine := w.CurrentStart
	if startLine < 1 {
		startLine = 1
	} else if startLine > total {
		startLine = total
	}

	endLine := startLine + windowSize - 1
	if endLine > total {
		endLine = total
	}

	var sb strings.Builder
	// Header: === [path] (lines <start>-<end> of <total>) ===\n
	fmt.Fprintf(&sb, "=== [%s] (lines %d-%d of %d) ===\n", path, startLine, endLine, total)

	// Body: <line>: <text>\n for each line in window
	for i := startLine; i <= endLine; i++ {
		fmt.Fprintf(&sb, "%d: %s\n", i, w.Lines[i-1])
	}

	// Footer: \n=== Navigation: ScrollDown(n), ScrollUp(n), Goto(line) | End of Window === (or [EOF] if end of file reached).
	navStatus := "End of Window"
	if endLine >= total {
		navStatus = "[EOF]"
	}
	fmt.Fprintf(&sb, "\n=== Navigation: ScrollDown(n), ScrollUp(n), Goto(line) | %s ===", navStatus)

	return &WindowView{
		Path:       path,
		StartLine:  startLine,
		EndLine:    endLine,
		TotalLines: total,
		Content:    sb.String(),
	}
}

func (w *WindowedFileReader) resolvePathLocked(path string) (string, error) {
	trimmed := strings.TrimSpace(path)
	if trimmed == "" {
		return "", fmt.Errorf("%w: empty path", ErrInvalidPath)
	}
	if strings.ContainsRune(trimmed, 0) {
		return "", fmt.Errorf("%w: path contains NUL byte", ErrInvalidPath)
	}

	if w.rootDir != "" {
		absRoot, err := filepath.Abs(w.rootDir)
		if err != nil {
			return "", fmt.Errorf("resolve root directory: %w", err)
		}
		absRoot = filepath.Clean(absRoot)

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

	info, err := os.Stat(trimmed)
	if err != nil {
		return "", err
	}
	if info.IsDir() {
		return "", fmt.Errorf("%w: path %q is a directory", ErrInvalidPath, path)
	}

	return trimmed, nil
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
	// Strip UTF-8 BOM if present
	data = bytes.TrimPrefix(data, []byte("\xef\xbb\xbf"))
	if len(data) == 0 {
		return []string{}
	}
	s := strings.ToValidUTF8(string(data), "\uFFFD")
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	hasTrailingNewline := strings.HasSuffix(s, "\n")
	if hasTrailingNewline {
		s = s[:len(s)-1]
	}
	if s == "" && hasTrailingNewline {
		return []string{""}
	}
	return strings.Split(s, "\n")
}
