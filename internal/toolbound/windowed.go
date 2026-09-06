package toolbound

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// DefaultWindowSize is the default number of lines displayed in a window when unspecified.
const DefaultWindowSize = 100

// DefaultMaxFileSize is the default maximum file size in bytes (50MB) allowed for windowed reading.
const DefaultMaxFileSize int64 = 50 * 1024 * 1024

var (
	// ErrNoFileOpen indicates navigation was attempted when no file is open.
	ErrNoFileOpen = errors.New("no file open")
	// ErrPathEscape indicates a path attempted to traverse outside the allowed root directory.
	ErrPathEscape = errors.New("path escapes root directory")
	// ErrInvalidPath indicates a malformed or invalid path argument.
	ErrInvalidPath = errors.New("invalid path")
	// ErrFileTooLarge indicates the requested file exceeds the configured maximum file size.
	ErrFileTooLarge = errors.New("file too large")
	// ErrBinaryFile indicates binary data (e.g. NUL byte) was detected in the file.
	ErrBinaryFile = errors.New("binary file detected")
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
	maxFileSize       int64

	FilePath     string
	Path         string
	Lines        []string
	CurrentLine  int
	CurrentStart int
	WindowSize   int
	TotalLines   int
}

// NewWindowedFileReader creates a WindowedFileReader.
// Optional arguments may provide a root confinement directory (string),
// a default window size (int), and/or a max file size in bytes (int64).
// If unspecified, defaultWindowSize defaults to DefaultWindowSize (100),
// maxFileSize defaults to DefaultMaxFileSize (50MB), and rootDir is unset.
func NewWindowedFileReader(args ...any) *WindowedFileReader {
	size := DefaultWindowSize
	root := ""
	var maxFileSize int64 = DefaultMaxFileSize
	for _, arg := range args {
		switch v := arg.(type) {
		case int:
			if v > 0 {
				size = v
			}
		case int64:
			if v > 0 {
				maxFileSize = v
			}
		case string:
			root = v
		}
	}
	return &WindowedFileReader{
		rootDir:           root,
		defaultWindowSize: size,
		maxFileSize:       maxFileSize,
		CurrentLine:       1,
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

// MaxFileSize returns the configured maximum file size in bytes.
func (w *WindowedFileReader) MaxFileSize() int64 {
	w.mu.RLock()
	defer w.mu.RUnlock()
	if w.maxFileSize <= 0 {
		return DefaultMaxFileSize
	}
	return w.maxFileSize
}

// SetMaxFileSize updates the maximum file size in bytes.
func (w *WindowedFileReader) SetMaxFileSize(size int64) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if size <= 0 {
		size = DefaultMaxFileSize
	}
	w.maxFileSize = size
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

// Open opens the file at path, reads lines, sets current window starting at startLine
// (1-based, clamped to 1..totalLines), and returns lines formatted with "<line>: <content>" prefix.
func (w *WindowedFileReader) Open(path string, startLine int, windowSize int) ([]string, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	data, err := w.readFileLocked(path)
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

	w.FilePath = path
	w.Path = path
	w.Lines = parsedLines
	w.TotalLines = total
	w.CurrentLine = startLine
	w.CurrentStart = startLine
	w.WindowSize = windowSize

	return w.formatWindowLocked(), nil
}

// ScrollDown advances window by n lines (if n <= 0, defaults to WindowSize),
// clamped so window doesn't go past the last line. Returns formatted lines.
func (w *WindowedFileReader) ScrollDown(n int) ([]string, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.isOpenLocked() {
		return nil, ErrNoFileOpen
	}

	if n <= 0 {
		n = w.WindowSize
		if n <= 0 {
			n = DefaultWindowSize
		}
	}

	total := len(w.Lines)
	if total == 0 {
		w.CurrentLine = 1
		w.CurrentStart = 1
		return []string{}, nil
	}

	target := w.CurrentLine + n
	if target > total {
		target = total
	}
	if target < 1 {
		target = 1
	}
	w.CurrentLine = target
	w.CurrentStart = target

	return w.formatWindowLocked(), nil
}

// ScrollUp moves window up by n lines (if n <= 0, defaults to WindowSize),
// clamped at line 1. Returns formatted lines.
func (w *WindowedFileReader) ScrollUp(n int) ([]string, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.isOpenLocked() {
		return nil, ErrNoFileOpen
	}

	if n <= 0 {
		n = w.WindowSize
		if n <= 0 {
			n = DefaultWindowSize
		}
	}

	total := len(w.Lines)
	if total == 0 {
		w.CurrentLine = 1
		w.CurrentStart = 1
		return []string{}, nil
	}

	target := w.CurrentLine - n
	if target < 1 {
		target = 1
	}
	if target > total {
		target = total
	}
	w.CurrentLine = target
	w.CurrentStart = target

	return w.formatWindowLocked(), nil
}

// Goto jumps window to start at line (clamped to 1..totalLines). Returns formatted lines.
func (w *WindowedFileReader) Goto(line int) ([]string, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.isOpenLocked() {
		return nil, ErrNoFileOpen
	}

	total := len(w.Lines)
	if total == 0 {
		w.CurrentLine = 1
		w.CurrentStart = 1
		return []string{}, nil
	}

	if line < 1 {
		line = 1
	} else if line > total {
		line = total
	}

	w.CurrentLine = line
	w.CurrentStart = line
	return w.formatWindowLocked(), nil
}

// GetWindow returns the currently active window formatted lines without moving the cursor.
func (w *WindowedFileReader) GetWindow() []string {
	w.mu.RLock()
	defer w.mu.RUnlock()

	if !w.isOpenLocked() {
		return []string{}
	}
	return w.formatWindowLocked()
}

// Status returns path, total lines, and current line range.
func (w *WindowedFileReader) Status() string {
	w.mu.RLock()
	defer w.mu.RUnlock()

	if !w.isOpenLocked() {
		return "no file open"
	}

	total := len(w.Lines)
	if total == 0 {
		return fmt.Sprintf("%s: lines 0-0 of 0", w.FilePath)
	}

	end := w.CurrentLine + w.WindowSize - 1
	if end > total {
		end = total
	}

	return fmt.Sprintf("%s: lines %d-%d of %d", w.FilePath, w.CurrentLine, end, total)
}

// OpenView opens the file and returns a structured WindowView.
func (w *WindowedFileReader) OpenView(path string, startLine int, windowSize int) (*WindowView, error) {
	lines, err := w.Open(path, startLine, windowSize)
	if err != nil {
		return nil, err
	}
	_ = lines
	return w.View()
}

// ScrollDownView advances window by n lines and returns the updated WindowView.
func (w *WindowedFileReader) ScrollDownView(n int) (*WindowView, error) {
	lines, err := w.ScrollDown(n)
	if err != nil {
		return nil, err
	}
	_ = lines
	return w.View()
}

// ScrollUpView moves window up by n lines and returns the updated WindowView.
func (w *WindowedFileReader) ScrollUpView(n int) (*WindowView, error) {
	lines, err := w.ScrollUp(n)
	if err != nil {
		return nil, err
	}
	_ = lines
	return w.View()
}

// GotoView jumps window to start at line and returns the updated WindowView.
func (w *WindowedFileReader) GotoView(line int) (*WindowView, error) {
	lines, err := w.Goto(line)
	if err != nil {
		return nil, err
	}
	_ = lines
	return w.View()
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

// CurrentPosition returns the path, current start line, window size, and total lines.
func (w *WindowedFileReader) CurrentPosition() (path string, currentLine int, windowSize int, totalLines int) {
	w.mu.RLock()
	defer w.mu.RUnlock()

	if !w.isOpenLocked() {
		return "", 0, 0, 0
	}
	return w.FilePath, w.CurrentLine, w.WindowSize, len(w.Lines)
}

// Close unloads the currently opened file and resets position.
func (w *WindowedFileReader) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.FilePath = ""
	w.Path = ""
	w.Lines = nil
	w.TotalLines = 0
	w.CurrentLine = 1
	w.CurrentStart = 1
	w.WindowSize = w.defaultWindowSize
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

func (w *WindowedFileReader) formatWindowLocked() []string {
	total := len(w.Lines)
	if total == 0 {
		return []string{}
	}

	start := w.CurrentLine
	if start < 1 {
		start = 1
	} else if start > total {
		start = total
	}

	end := start + w.WindowSize - 1
	if end > total {
		end = total
	}

	count := end - start + 1
	if count <= 0 {
		return []string{}
	}

	res := make([]string, count)
	for i := 0; i < count; i++ {
		lineNum := start + i
		res[i] = fmt.Sprintf("%d: %s", lineNum, w.Lines[lineNum-1])
	}
	return res
}

func (w *WindowedFileReader) viewLocked() *WindowView {
	total := len(w.Lines)
	path := w.FilePath
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

	startLine := w.CurrentLine
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
	fmt.Fprintf(&sb, "=== [%s] (lines %d-%d of %d) ===\n", path, startLine, endLine, total)
	for i := startLine; i <= endLine; i++ {
		fmt.Fprintf(&sb, "%d: %s\n", i, w.Lines[i-1])
	}

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

func (w *WindowedFileReader) readFileLocked(path string) ([]byte, error) {
	resolved, err := w.resolvePathLocked(path)
	if err != nil {
		return nil, err
	}

	fi, err := os.Stat(resolved)
	if err != nil {
		return nil, err
	}

	maxSize := w.maxFileSize
	if maxSize <= 0 {
		maxSize = DefaultMaxFileSize
	}
	if fi.Size() > maxSize {
		return nil, ErrFileTooLarge
	}

	f, err := os.Open(resolved)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	header := make([]byte, 8192)
	n, err := f.Read(header)
	if err != nil && err != io.EOF {
		return nil, err
	}
	if bytes.IndexByte(header[:n], 0) != -1 {
		return nil, ErrBinaryFile
	}
	if int64(n) > maxSize {
		return nil, ErrFileTooLarge
	}

	var data []byte
	if err == io.EOF || n == 0 {
		data = header[:n]
	} else {
		rest, readErr := io.ReadAll(io.LimitReader(f, maxSize-int64(n)+1))
		if readErr != nil {
			return nil, readErr
		}
		if int64(n)+int64(len(rest)) > maxSize {
			return nil, ErrFileTooLarge
		}
		if bytes.IndexByte(rest, 0) != -1 {
			return nil, ErrBinaryFile
		}
		if len(rest) == 0 {
			data = header[:n]
		} else {
			data = append(header[:n], rest...)
		}
	}
	return data, nil
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

		if !isPathWithin(absRoot, target) {
			return "", fmt.Errorf("%w: path %q escapes root %q", ErrPathEscape, path, absRoot)
		}

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
