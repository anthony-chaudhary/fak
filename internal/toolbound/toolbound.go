package toolbound

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"
)

// Ready reports that the leaf is wired.
func Ready() bool { return true }

// BoundedOutput represents the result of bounding tool output.
type BoundedOutput struct {
	Preview       string // bounded preview string
	CompletePath  string // path to managed spill file if spilled, or empty
	OriginalBytes int    // original byte length
	OriginalLines int    // original line count
	Truncated     bool   // whether bounding / truncation was applied
}

// Options configures output bounding limits and spill locations.
type Options struct {
	MaxLines    int    // maximum allowed lines before truncation (0 = unconstrained)
	MaxBytes    int    // maximum allowed UTF-8 bytes before truncation (0 = unconstrained)
	SpillDir    string // directory for managed spill files (if empty, defaults to os.TempDir())
	SpillPrefix string // file prefix (defaults to "fak-tool-output-")
}

// Bounder bounds tool output strings and manages spill files.
type Bounder struct {
	opts Options
}

// New constructs a Bounder with the provided options.
func New(opts Options) *Bounder {
	if opts.SpillDir == "" {
		opts.SpillDir = os.TempDir()
	}
	if opts.SpillPrefix == "" {
		opts.SpillPrefix = "fak-tool-output-"
	}
	return &Bounder{opts: opts}
}

// Bound bounds raw output according to configured limits, spilling to disk if exceeded.
func (b *Bounder) Bound(raw string) (*BoundedOutput, error) {
	origBytes := len(raw)
	origLines := measureLines(raw)

	exceedsLines := b.opts.MaxLines > 0 && origLines > b.opts.MaxLines
	exceedsBytes := b.opts.MaxBytes > 0 && origBytes > b.opts.MaxBytes

	if !exceedsLines && !exceedsBytes {
		return &BoundedOutput{
			Preview:       raw,
			CompletePath:  "",
			OriginalBytes: origBytes,
			OriginalLines: origLines,
			Truncated:     false,
		}, nil
	}

	spillDir := b.opts.SpillDir
	if spillDir == "" {
		spillDir = os.TempDir()
	}
	spillPrefix := b.opts.SpillPrefix
	if spillPrefix == "" {
		spillPrefix = "fak-tool-output-"
	}

	f, err := os.CreateTemp(spillDir, spillPrefix+"*")
	if err != nil {
		return nil, fmt.Errorf("create spill file: %w", err)
	}
	completePath := filepath.Clean(f.Name())

	if _, err := f.WriteString(raw); err != nil {
		_ = f.Close()
		_ = os.Remove(completePath)
		return nil, fmt.Errorf("write spill file: %w", err)
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(completePath)
		return nil, fmt.Errorf("close spill file: %w", err)
	}

	preview := b.buildPreview(raw, origLines, origBytes, completePath)

	return &BoundedOutput{
		Preview:       preview,
		CompletePath:  completePath,
		OriginalBytes: origBytes,
		OriginalLines: origLines,
		Truncated:     true,
	}, nil
}

func (b *Bounder) buildPreview(raw string, origLines, origBytes int, completePath string) string {
	headEnd := len(raw)
	tailStart := 0

	if b.opts.MaxLines > 0 && origLines > b.opts.MaxLines {
		headCount := b.opts.MaxLines / 2
		tailCount := b.opts.MaxLines - headCount
		if headCount == 0 && b.opts.MaxLines > 0 {
			headCount = 1
			tailCount = 0
		}
		headEnd = headLineEnd(raw, headCount)
		tailStart = tailLineStart(raw, tailCount)
	}

	if b.opts.MaxBytes > 0 {
		headByteLimit := b.opts.MaxBytes / 2
		tailByteLimit := b.opts.MaxBytes - headByteLimit
		if headByteLimit == 0 && b.opts.MaxBytes > 0 {
			headByteLimit = 1
			tailByteLimit = 0
		}

		if headEnd > headByteLimit {
			headEnd = adjustUTF8Head(raw, headByteLimit)
		}
		if len(raw)-tailStart > tailByteLimit {
			tailStart = adjustUTF8Tail(raw, len(raw)-tailByteLimit)
		}
	}

	if headEnd > tailStart {
		headEnd = tailStart
	}

	head := raw[:headEnd]
	tail := raw[tailStart:]

	headLines := measureLines(head)
	tailLines := measureLines(tail)

	spilledLines := origLines - (headLines + tailLines)
	if spilledLines < 0 {
		spilledLines = 0
	}
	spilledBytes := origBytes - (len(head) + len(tail))
	if spilledBytes < 0 {
		spilledBytes = 0
	}

	notice := fmt.Sprintf("[... output truncated: spilled %d lines, %d bytes to %s ...]", spilledLines, spilledBytes, completePath)

	var sb strings.Builder
	if len(head) > 0 {
		sb.WriteString(head)
		if !strings.HasSuffix(head, "\n") {
			sb.WriteString("\n")
		}
	}
	sb.WriteString(notice)
	sb.WriteString("\n")
	if len(tail) > 0 {
		sb.WriteString(tail)
	}

	return sb.String()
}

// CleanupSpillFiles scans SpillDir for files matching SpillPrefix,
// removes files with mod time older than retention, and returns the
// count of deleted files.
func (b *Bounder) CleanupSpillFiles(retention time.Duration) (int, error) {
	spillDir := b.opts.SpillDir
	if spillDir == "" {
		spillDir = os.TempDir()
	}
	spillPrefix := b.opts.SpillPrefix
	if spillPrefix == "" {
		spillPrefix = "fak-tool-output-"
	}

	entries, err := os.ReadDir(spillDir)
	if err != nil {
		return 0, fmt.Errorf("read spill directory: %w", err)
	}

	cutoff := time.Now().Add(-retention)
	deleted := 0
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if !strings.HasPrefix(entry.Name(), spillPrefix) {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		if info.ModTime().Before(cutoff) {
			target := filepath.Join(spillDir, entry.Name())
			if err := os.Remove(target); err == nil {
				deleted++
			}
		}
	}
	return deleted, nil
}

func measureLines(s string) int {
	if s == "" {
		return 0
	}
	n := strings.Count(s, "\n")
	if !strings.HasSuffix(s, "\n") {
		n++
	}
	return n
}

func headLineEnd(s string, headCount int) int {
	if headCount <= 0 {
		return 0
	}
	linesSeen := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			linesSeen++
			if linesSeen == headCount {
				return i + 1
			}
		}
	}
	return len(s)
}

func tailLineStart(s string, tailCount int) int {
	if tailCount <= 0 {
		return len(s)
	}
	end := len(s)
	if end > 0 && s[end-1] == '\n' {
		end--
		if end > 0 && s[end-1] == '\r' {
			end--
		}
	}
	linesSeen := 0
	for i := end - 1; i >= 0; i-- {
		if s[i] == '\n' {
			linesSeen++
			if linesSeen == tailCount {
				return i + 1
			}
		}
	}
	return 0
}

func adjustUTF8Head(s string, pos int) int {
	if pos >= len(s) {
		return len(s)
	}
	for pos > 0 && !utf8.RuneStart(s[pos]) {
		pos--
	}
	return pos
}

func adjustUTF8Tail(s string, pos int) int {
	if pos <= 0 {
		return 0
	}
	for pos < len(s) && !utf8.RuneStart(s[pos]) {
		pos++
	}
	return pos
}
