package taskmgr

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
)

const defaultArtifactOutputLimit = 64 << 10

// ArtifactCommand describes a command whose bounded, redacted output should be
// retained as issue evidence. ArtifactPath is required so the caller chooses a
// durable, linkable location rather than silently writing session scratch.
type ArtifactCommand struct {
	Argv         []string
	Dir          string
	ArtifactPath string
	MaxBytes     int
}

// ArtifactResult records the command outcome and the independently readable
// path EvidenceRef. A non-zero command exit is evidence, not a helper error;
// Err is reserved for setup, execution, redaction, or artifact-write failures.
type ArtifactResult struct {
	Evidence  EvidenceRef
	ExitCode  int
	Truncated bool
}

// CaptureCommandArtifact runs a command, retains only bounded output, redacts
// common credential forms, and atomically stores the result with owner-only
// permissions. It returns a path EvidenceRef suitable for a dogfood issue.
func CaptureCommandArtifact(ctx context.Context, spec ArtifactCommand) (ArtifactResult, error) {
	if len(spec.Argv) == 0 || strings.TrimSpace(spec.Argv[0]) == "" {
		return ArtifactResult{}, errors.New("artifact command argv is required")
	}
	if strings.TrimSpace(spec.ArtifactPath) == "" {
		return ArtifactResult{}, errors.New("artifact path is required")
	}
	limit := spec.MaxBytes
	if limit <= 0 {
		limit = defaultArtifactOutputLimit
	}

	capture := &boundedCommandOutput{limit: limit}
	cmd := exec.CommandContext(ctx, spec.Argv[0], spec.Argv[1:]...)
	cmd.Dir = spec.Dir
	cmd.Stdout = capture
	cmd.Stderr = capture
	runErr := cmd.Run()
	exitCode := 0
	if runErr != nil {
		var exitErr *exec.ExitError
		if errors.As(runErr, &exitErr) {
			exitCode = exitErr.ExitCode()
		} else {
			return ArtifactResult{}, fmt.Errorf("run artifact command: %w", runErr)
		}
	}

	body := RedactCommandOutput(string(capture.Bytes()))
	if capture.Truncated() {
		body += fmt.Sprintf("\n[fak artifact output truncated at %d bytes]\n", limit)
	}
	if err := writeArtifactAtomic(spec.ArtifactPath, []byte(body)); err != nil {
		return ArtifactResult{}, err
	}
	return ArtifactResult{
		Evidence: EvidenceRef{Kind: "path", Ref: filepath.Clean(spec.ArtifactPath), Note: fmt.Sprintf("command exit=%d", exitCode)},
		ExitCode: exitCode, Truncated: capture.Truncated(),
	}, nil
}

type boundedCommandOutput struct {
	mu        sync.Mutex
	buf       []byte
	limit     int
	truncated bool
}

func (w *boundedCommandOutput) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	remaining := w.limit - len(w.buf)
	if remaining > 0 {
		keep := len(p)
		if keep > remaining {
			keep = remaining
		}
		w.buf = append(w.buf, p[:keep]...)
	}
	if len(p) > remaining {
		w.truncated = true
	}
	return len(p), nil
}

func (w *boundedCommandOutput) Bytes() []byte {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]byte(nil), w.buf...)
}
func (w *boundedCommandOutput) Truncated() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.truncated
}

var commandSecretPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)(authorization\s*:\s*(?:bearer|basic)\s+)[^\s]+`),
	regexp.MustCompile(`(?i)((?:api[_-]?key|token|secret|password)\s*[=:]\s*)[^\s]+`),
}

// RedactCommandOutput removes common credential forms before evidence leaves
// the local command boundary. It intentionally preserves the field name so the
// resulting artifact remains diagnostically useful.
func RedactCommandOutput(s string) string {
	for _, pattern := range commandSecretPatterns {
		s = pattern.ReplaceAllString(s, `${1}[REDACTED]`)
	}
	return s
}

func writeArtifactAtomic(path string, data []byte) error {
	clean := filepath.Clean(path)
	if err := os.MkdirAll(filepath.Dir(clean), 0o700); err != nil {
		return fmt.Errorf("create artifact directory: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(clean), ".fak-artifact-*")
	if err != nil {
		return fmt.Errorf("create artifact temp file: %w", err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return fmt.Errorf("protect artifact temp file: %w", err)
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return fmt.Errorf("write artifact: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close artifact: %w", err)
	}
	if err := os.Rename(tmpName, clean); err != nil {
		return fmt.Errorf("publish artifact: %w", err)
	}
	return nil
}
