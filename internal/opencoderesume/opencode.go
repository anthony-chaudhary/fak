package opencoderesume

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// Closed verdict vocabulary for an OpenCode session identity check. Only
// VerdictResumable may proceed to process launch; every other value blocks.
const (
	// VerdictResumable: the id appears in the project-scoped listing.
	VerdictResumable = "resumable"
	// VerdictNotFound: the listing parsed cleanly and the id is absent from
	// it. Covers both never-existed and belongs-to-another-project; both are
	// fail-closed for resume-in-cwd.
	VerdictNotFound = "not_found_in_project"
	// VerdictInvalidIdentity: the id does not match the opencode session-id
	// shape (or is empty). No CLI is executed.
	VerdictInvalidIdentity = "invalid_identity"
	// VerdictStoreUnreadable: the opencode CLI could not be executed or
	// returned unusable output. The only verdict that carries a non-nil error.
	VerdictStoreUnreadable = "store_unreadable"
)

// ErrInvalidSessionID reports that a session id failed the ses_ shape check.
var ErrInvalidSessionID = errors.New("opencoderesume: invalid OpenCode session id")

const (
	defaultBin      = "opencode"
	defaultMaxCount = 200
	listTimeout     = 10 * time.Second
)

// sessionIDPattern pins the conservative observed shape: the "ses_" prefix
// followed by 16..64 mixed-case alphanumerics (observed live: 26 chars, e.g.
// ses_e3861b794ffeKK5UOg25IFE1mR). OpenCode ids are NOT UUIDs; do not reuse
// codexresume's canonicalUUIDPattern here.
var sessionIDPattern = regexp.MustCompile(`^ses_[A-Za-z0-9]{16,64}$`)

// IsSessionID reports whether id has the OpenCode session-id shape. The id
// flows into an exec argv, so this shape check is also the injection defense:
// anything outside [A-Za-z0-9] after the prefix is refused.
func IsSessionID(id string) bool {
	return sessionIDPattern.MatchString(id)
}

// Options selects the identity oracle for one Verify call.
type Options struct {
	// SessionID is the candidate OpenCode session id to verify. Required:
	// an empty or ill-shaped id yields VerdictInvalidIdentity without
	// executing the CLI.
	SessionID string
	// Bin is the opencode executable; default "opencode", resolved via
	// exec.LookPath at call time (never at init).
	Bin string
	// Dir is the project directory the listing is scoped to (cmd.Dir).
	// Empty means the current process directory.
	Dir string
	// MaxCount bounds the listing (-n flag); default 200. Non-positive
	// values fall back to the default.
	MaxCount int
	// Now is reserved for future freshness gating; unused in v1.
	Now time.Time
}

// runListCommand executes `opencode session list --format json -n <max>` with
// cmd.Dir = dir and returns its stdout. It is a var so tests can inject canned
// output; the production implementation is bounded by ctx and fails closed on
// any resolution, start, or exit failure.
var runListCommand = func(ctx context.Context, bin, dir string, maxCount int) ([]byte, error) {
	resolved, err := exec.LookPath(bin)
	if err != nil {
		return nil, fmt.Errorf("resolve %q on PATH: %w", bin, err)
	}
	cmd := exec.CommandContext(ctx, resolved, "session", "list", "--format", "json", "-n", strconv.Itoa(maxCount))
	cmd.Dir = dir
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return stdout.Bytes(), fmt.Errorf("run %q session list: %w%s", bin, err, stderrTail(stderr.String()))
	}
	return stdout.Bytes(), nil
}

// stderrTail keeps a short tail of stderr inside wrapped diagnostics.
func stderrTail(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	const max = 256
	if len(s) > max {
		s = s[:max] + "..."
	}
	return "; stderr: " + s
}

// sessionRow is the minimal projection of one `opencode session list` entry.
// Unknown fields (title, projectId, ...) are ignored on purpose: the listing
// is trusted only for identity membership, never for display or policy.
type sessionRow struct {
	ID        string `json:"id"`
	Directory string `json:"directory"`
	Updated   int64  `json:"updated"`
	Created   int64  `json:"created"`
}

// Verify checks that opts.SessionID names a session of the project rooted at
// opts.Dir, using the opencode CLI as the identity oracle.
//
// It returns exactly one of the closed verdicts:
//   - VerdictInvalidIdentity: the id is empty or fails the ses_ shape check;
//     no CLI is executed.
//   - VerdictStoreUnreadable: the CLI could not be executed or stdout is not
//     a JSON array; err is non-nil (wrapped cause) ONLY for this verdict.
//   - VerdictResumable: the id appears in the project-scoped listing.
//   - VerdictNotFound: the listing parsed cleanly and the id is absent.
//
// Every non-resumable verdict is fail-closed: callers must block launch.
func Verify(ctx context.Context, opts Options) (string, error) {
	if !IsSessionID(opts.SessionID) {
		return VerdictInvalidIdentity, nil
	}
	bin := strings.TrimSpace(opts.Bin)
	if bin == "" {
		bin = defaultBin
	}
	maxCount := opts.MaxCount
	if maxCount <= 0 {
		maxCount = defaultMaxCount
	}
	ctx, cancel := context.WithTimeout(ctx, listTimeout)
	defer cancel()

	out, err := runListCommand(ctx, bin, opts.Dir, maxCount)
	if err != nil {
		return VerdictStoreUnreadable, fmt.Errorf("verify %s: %w", opts.SessionID, err)
	}
	trimmed := bytes.TrimLeft(out, " \t\r\n")
	if len(trimmed) == 0 || trimmed[0] != '[' {
		return VerdictStoreUnreadable, fmt.Errorf("verify %s: session list stdout is not a JSON array", opts.SessionID)
	}
	var sessions []sessionRow
	if err := json.Unmarshal(out, &sessions); err != nil {
		return VerdictStoreUnreadable, fmt.Errorf("verify %s: parse session list: %w", opts.SessionID, err)
	}
	for i := range sessions {
		if sessions[i].ID == opts.SessionID {
			return VerdictResumable, nil
		}
	}
	return VerdictNotFound, nil
}

// ResumeArgv builds the exact-resume argv for a verified OpenCode session id.
// interactive selects the TUI entry point (`opencode --session <id>`);
// headless mode ships exactly the shape internal/resumeactuator already
// launches (`opencode run --session <id> [prompt]`, prompt appended as a
// single trailing arg when non-empty). Callers must pass an id that already
// earned VerdictResumable; this function re-checks the shape but cannot prove
// existence.
func ResumeArgv(bin, sessionID, prompt string, interactive bool) ([]string, error) {
	if !IsSessionID(sessionID) {
		return nil, fmt.Errorf("%w: %q", ErrInvalidSessionID, sessionID)
	}
	exe := strings.TrimSpace(bin)
	if exe == "" {
		exe = defaultBin
	}
	if interactive {
		return []string{exe, "--session", sessionID}, nil
	}
	argv := []string{exe, "run", "--session", sessionID}
	if strings.TrimSpace(prompt) != "" {
		argv = append(argv, prompt)
	}
	return argv, nil
}
