package safesync

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/branchrole"
)

// Schema constants for shadow landing audit telemetry and canary discrepancy logger.
const (
	ShadowLandingReceiptSchema        = "fak-shadow-landing/v1"
	CanaryDiscrepancySchema           = "fak-canary-discrepancy/v1"
	DefaultShadowDiscrepanciesRelPath = ".fak/shadow_discrepancies.jsonl"
	EnvSafesyncShadow                 = "FAK_SAFESYNC_SHADOW"
)

// Decision constants for shadow landing audit.
const (
	DecisionCleared             = "CLEARED"
	DecisionDiscrepancyDetected = "DISCREPANCY_DETECTED"
)

// Discrepancy kind constants.
const (
	DiscrepancyKindRouteMismatch = "route_mismatch"
	DiscrepancyKindStateMismatch = "state_mismatch"
)

// CanaryDiscrepancy records a discrepancy detected during shadow landing assessment.
type CanaryDiscrepancy struct {
	Schema    string    `json:"schema"`
	Timestamp time.Time `json:"timestamp"`
	Repo      string    `json:"repo"`
	Branch    string    `json:"branch"`
	Kind      string    `json:"kind"`
	Expected  string    `json:"expected"`
	Actual    string    `json:"actual"`
	Paths     []string  `json:"paths,omitempty"`
	HeadSHA   string    `json:"head_sha,omitempty"`
	TargetSHA string    `json:"target_sha,omitempty"`
	Detail    string    `json:"detail,omitempty"`
}

// CanaryDiscrepancyLogger provides thread-safe, append-only JSONL logging
// of canary discrepancies.
type CanaryDiscrepancyLogger struct {
	mu      sync.Mutex
	repo    string
	logPath string
}

// NewCanaryDiscrepancyLogger constructs a CanaryDiscrepancyLogger for repo.
// If customLogPath is provided and non-empty, it overrides the default path.
func NewCanaryDiscrepancyLogger(repo string, customLogPath ...string) *CanaryDiscrepancyLogger {
	var logPath string
	if len(customLogPath) > 0 && strings.TrimSpace(customLogPath[0]) != "" {
		lp := strings.TrimSpace(customLogPath[0])
		if filepath.IsAbs(lp) || strings.TrimSpace(repo) == "" {
			logPath = lp
		} else {
			logPath = filepath.Join(repo, lp)
		}
	} else {
		r := strings.TrimSpace(repo)
		if r == "" {
			r = "."
		}
		logPath = filepath.Join(r, DefaultShadowDiscrepanciesRelPath)
	}
	return &CanaryDiscrepancyLogger{
		repo:    repo,
		logPath: logPath,
	}
}

// LogPath returns the resolved filesystem path for discrepancy logs.
func (l *CanaryDiscrepancyLogger) LogPath() string {
	if l == nil {
		return ""
	}
	return l.logPath
}

// LogDiscrepancy appends one discrepancy record to the JSONL log file.
// It fails open gracefully by returning the underlying error without panicking.
func (l *CanaryDiscrepancyLogger) LogDiscrepancy(d CanaryDiscrepancy) error {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()

	if d.Schema == "" {
		d.Schema = CanaryDiscrepancySchema
	}
	if d.Timestamp.IsZero() {
		d.Timestamp = time.Now().UTC()
	}

	dir := filepath.Dir(l.logPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	f, err := os.OpenFile(l.logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	data, err := json.Marshal(d)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if _, err := f.Write(data); err != nil {
		return err
	}
	return nil
}

// ReadDiscrepancies reads and parses all discrepancy records from the JSONL log.
// Returns an empty slice if the file does not exist.
func (l *CanaryDiscrepancyLogger) ReadDiscrepancies() ([]CanaryDiscrepancy, error) {
	if l == nil {
		return []CanaryDiscrepancy{}, nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()

	f, err := os.Open(l.logPath)
	if err != nil {
		if os.IsNotExist(err) {
			return []CanaryDiscrepancy{}, nil
		}
		return nil, err
	}
	defer f.Close()

	var out []CanaryDiscrepancy
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var d CanaryDiscrepancy
		if err := json.Unmarshal(line, &d); err != nil {
			return nil, fmt.Errorf("read discrepancy: %w", err)
		}
		out = append(out, d)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if out == nil {
		out = []CanaryDiscrepancy{}
	}
	return out, nil
}

// ShadowOptions configures non-destructive shadow landing assessment.
type ShadowOptions struct {
	Repo          string           `json:"repo,omitempty"`
	Remote        string           `json:"remote,omitempty"`
	Branch        string           `json:"branch,omitempty"`
	TargetRef     string           `json:"target_ref,omitempty"`
	Shadow        bool             `json:"shadow,omitempty"`
	LogPath       string           `json:"log_path,omitempty"`
	ExpectedRoute string           `json:"expected_route,omitempty"`
	ExpectedState string           `json:"expected_state,omitempty"`
	Runner        Runner           `json:"-"`
	Now           func() time.Time `json:"-"`
	Session       string           `json:"session,omitempty"`
}

// ShadowLandingReceipt records the audited outcome of a shadow landing assessment.
type ShadowLandingReceipt struct {
	Schema           string              `json:"schema"`
	ReceiptID        string              `json:"receipt_id"`
	Timestamp        time.Time           `json:"timestamp"`
	Repo             string              `json:"repo"`
	Branch           string              `json:"branch"`
	Remote           string              `json:"remote"`
	HeadSHA          string              `json:"head_sha"`
	TargetSHA        string              `json:"target_sha"`
	TargetRef        string              `json:"target_ref"`
	SyncState        string              `json:"sync_state"`
	ShadowMode       bool                `json:"shadow_mode"`
	CanaryRoute      string              `json:"canary_route"`
	ActualRoute      string              `json:"actual_route"`
	DiscrepancyCount int                 `json:"discrepancy_count"`
	Discrepancies    []CanaryDiscrepancy `json:"discrepancies,omitempty"`
	Decision         string              `json:"decision"`
	TelemetryLogged  bool                `json:"telemetry_logged"`
	TelemetryError   string              `json:"telemetry_error,omitempty"`
}

// IsShadowModeEnabled reports whether shadow landing mode is enabled via options or environment.
func IsShadowModeEnabled(opts ShadowOptions) bool {
	return opts.Shadow || os.Getenv(EnvSafesyncShadow) == "1"
}

// AuditShadowLanding executes a non-destructive read-only assessment of the repository,
// compares actual routing and state against any expected route/state, logs discrepancies
// via CanaryDiscrepancyLogger (failing open on telemetry failure), and returns the receipt.
func AuditShadowLanding(ctx context.Context, repo string, opts ShadowOptions) (*ShadowLandingReceipt, error) {
	if repo == "" {
		repo = opts.Repo
	}
	if repo == "" {
		repo = "."
	}
	opts.Repo = repo
	if opts.Remote == "" {
		opts.Remote = "origin"
	}
	if opts.Runner == nil {
		opts.Runner = RealRunner
	}
	nowFn := opts.Now
	if nowFn == nil {
		nowFn = time.Now
	}
	now := nowFn().UTC()

	branch := strings.TrimSpace(opts.Branch)
	if branch == "" && opts.TargetRef != "" {
		remotePrefix := opts.Remote + "/"
		if strings.HasPrefix(opts.TargetRef, remotePrefix) {
			branch = strings.TrimPrefix(opts.TargetRef, remotePrefix)
		}
	}
	if branch == "" {
		b, err := currentBranch(ctx, opts.Runner, opts.Repo)
		if err == nil {
			branch = b
		}
	}
	if branch == "" {
		roles, _ := branchrole.Load(opts.Repo)
		branch = strings.TrimSpace(roles.DevelopmentBranch)
		if branch == "" {
			branch = "main"
		}
	}

	targetRef := opts.TargetRef
	if targetRef == "" {
		targetRef = opts.Remote + "/" + branch
	}

	reconcileOpts := ReconcileOptions{
		Repo:    opts.Repo,
		Remote:  opts.Remote,
		Branch:  branch,
		Goal:    "publish",
		Apply:   false,
		Fetch:   false,
		Runner:  opts.Runner,
		Now:     nowFn,
		Session: opts.Session,
	}

	assessment, err := RouteReconciliation(ctx, reconcileOpts)
	if err != nil {
		return nil, err
	}

	actualRoute := assessment.Route
	canaryRoute := assessment.Route
	if opts.ExpectedRoute != "" {
		canaryRoute = opts.ExpectedRoute
	}

	var discrepancies []CanaryDiscrepancy

	if opts.ExpectedRoute != "" && opts.ExpectedRoute != actualRoute {
		discrepancies = append(discrepancies, CanaryDiscrepancy{
			Schema:    CanaryDiscrepancySchema,
			Timestamp: now,
			Repo:      opts.Repo,
			Branch:    branch,
			Kind:      DiscrepancyKindRouteMismatch,
			Expected:  opts.ExpectedRoute,
			Actual:    actualRoute,
			Paths:     collidingOrDirtyPaths(assessment),
			HeadSHA:   assessment.Head,
			TargetSHA: assessment.Target,
			Detail:    fmt.Sprintf("route mismatch: expected %s, got actual %s", opts.ExpectedRoute, actualRoute),
		})
	}

	if opts.ExpectedState != "" && opts.ExpectedState != assessment.State {
		discrepancies = append(discrepancies, CanaryDiscrepancy{
			Schema:    CanaryDiscrepancySchema,
			Timestamp: now,
			Repo:      opts.Repo,
			Branch:    branch,
			Kind:      DiscrepancyKindStateMismatch,
			Expected:  opts.ExpectedState,
			Actual:    assessment.State,
			Paths:     collidingOrDirtyPaths(assessment),
			HeadSHA:   assessment.Head,
			TargetSHA: assessment.Target,
			Detail:    fmt.Sprintf("state mismatch: expected %s, got actual %s", opts.ExpectedState, assessment.State),
		})
	}

	decision := DecisionCleared
	if len(discrepancies) > 0 {
		decision = DecisionDiscrepancyDetected
	}

	logger := NewCanaryDiscrepancyLogger(opts.Repo, opts.LogPath)
	telemetryLogged := true
	var telemetryError string

	for _, d := range discrepancies {
		if logErr := logger.LogDiscrepancy(d); logErr != nil {
			telemetryLogged = false
			telemetryError = logErr.Error()
			break
		}
	}

	h := sha256.New()
	fmt.Fprintf(h, "%s:%s:%s:%s:%d", opts.Repo, branch, assessment.Head, assessment.Target, now.UnixNano())
	receiptID := fmt.Sprintf("slr-%s", hex.EncodeToString(h.Sum(nil))[:16])

	targetRefOut := assessment.TargetRef
	if targetRefOut == "" {
		targetRefOut = targetRef
	}
	remoteOut := assessment.Remote
	if remoteOut == "" {
		remoteOut = opts.Remote
	}

	receipt := &ShadowLandingReceipt{
		Schema:           ShadowLandingReceiptSchema,
		ReceiptID:        receiptID,
		Timestamp:        now,
		Repo:             opts.Repo,
		Branch:           branch,
		Remote:           remoteOut,
		HeadSHA:          assessment.Head,
		TargetSHA:        assessment.Target,
		TargetRef:        targetRefOut,
		SyncState:        assessment.State,
		ShadowMode:       IsShadowModeEnabled(opts),
		CanaryRoute:      canaryRoute,
		ActualRoute:      actualRoute,
		DiscrepancyCount: len(discrepancies),
		Discrepancies:    discrepancies,
		Decision:         decision,
		TelemetryLogged:  telemetryLogged,
		TelemetryError:   telemetryError,
	}

	return receipt, nil
}

func collidingOrDirtyPaths(a ReconcileAssessment) []string {
	if len(a.CollidingPaths) > 0 {
		return append([]string(nil), a.CollidingPaths...)
	}
	if len(a.DirtyPaths) > 0 {
		return append([]string(nil), a.DirtyPaths...)
	}
	return nil
}
