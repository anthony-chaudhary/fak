// Package gym coordinates agent evaluation environments, sub-10ms CoW
// snapshot lifecycles, and isolated execution trajectories.
package gym

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sandbox"
)

// Config defines the configuration parameters for initializing an isolated gym arena.
type Config struct {
	// BaseDir is the read-only host trunk / base workspace directory path.
	BaseDir string
	// WorkspaceName is a human-readable identifier for the gym instance.
	WorkspaceName string
	// SanitizedEnv specifies an explicit sanitized environment; if nil/empty, host env is filtered.
	SanitizedEnv []string
	// PinnedPTY forces fixed terminal dimensions (PTYRows x PTYCols).
	PinnedPTY bool
	// PTYRows defines the terminal row count (defaults to 24).
	PTYRows int
	// PTYCols defines the terminal column count (defaults to 80).
	PTYCols int
	// DeterministicTime sets the deterministic time stamp.
	DeterministicTime time.Time
	// Locale specifies the forced locale (defaults to C.UTF-8).
	Locale string
}

// Arena represents an active, isolated gym execution environment backed by a sub-10ms CoW overlay.
type Arena struct {
	mu         sync.Mutex
	cfg        Config
	overlay    CoWOverlay
	parent     *Arena
	branchName string
	destroyed  bool
}

// Create initializes a new isolated gym arena over cfg.BaseDir.
func Create(ctx context.Context, cfg Config) (*Arena, error) {
	if strings.TrimSpace(cfg.BaseDir) == "" {
		return nil, errors.New("base directory is required")
	}
	absBase, err := filepath.Abs(cfg.BaseDir)
	if err != nil {
		return nil, fmt.Errorf("invalid base directory: %w", err)
	}
	if info, err := os.Stat(absBase); err != nil || !info.IsDir() {
		return nil, fmt.Errorf("base directory does not exist or is not a directory: %s", absBase)
	}
	cfg.BaseDir = absBase

	if cfg.WorkspaceName == "" {
		cfg.WorkspaceName = "gym-arena"
	}
	if cfg.PTYRows <= 0 {
		cfg.PTYRows = 24
	}
	if cfg.PTYCols <= 0 {
		cfg.PTYCols = 80
	}
	if cfg.Locale == "" {
		cfg.Locale = "C.UTF-8"
	}

	overlay, err := NewOverlay(cfg.BaseDir, "")
	if err != nil {
		return nil, fmt.Errorf("failed to initialize CoW overlay: %w", err)
	}

	return &Arena{
		cfg:     cfg,
		overlay: overlay,
	}, nil
}

// Path returns the unified workspace path of the arena (overlay MergedDir).
func (a *Arena) Path() string {
	return a.overlay.MergedDir()
}

// Overlay returns the underlying CoWOverlay implementation.
func (a *Arena) Overlay() CoWOverlay {
	return a.overlay
}

// ModifiedArtifacts returns a list of files modified, created, or deleted relative to base.
func (a *Arena) ModifiedArtifacts() []string {
	if u, ok := a.overlay.(*UserspaceOverlay); ok {
		return u.ModifiedArtifacts()
	}
	// Generic fallback by reading upper directory
	var list []string
	_ = filepath.Walk(a.overlay.UpperDir(), func(p string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(a.overlay.UpperDir(), p)
		if err == nil && rel != "." {
			list = append(list, filepath.ToSlash(rel))
		}
		return nil
	})
	return list
}

// Execute dispatches a command inside the gym arena under default-deny isolation,
// stabilized deterministic environment variables, and pinned PTY geometry.
func (a *Arena) Execute(ctx context.Context, req sandbox.ExecutionRequest) (sandbox.ExecutionResult, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.destroyed {
		return sandbox.ExecutionResult{}, errors.New("arena has been destroyed")
	}

	if strings.TrimSpace(req.WorkingDir) == "" {
		req.WorkingDir = a.Path()
	}

	// 1. Stabilize environment variables
	req.Env = a.stabilizeEnv(req.Env)

	// 2. Build sandbox spec and dispatch to default registry
	spec := sandbox.Spec{
		Tier:         sandbox.TierL1NativeOS,
		WorkspaceDir: a.Path(),
		Env:          req.Env,
		EgressPolicy: sandbox.EgressBlocked,
	}

	reg := sandbox.DefaultRegistry()
	res, err := reg.Execute(ctx, spec, req)

	// 3. Reconcile modifications into UpperDir if userspace overlay
	if u, ok := a.overlay.(*UserspaceOverlay); ok {
		_ = u.Reconcile()
	}

	return res, err
}

// Reset wipes all mutations in <10ms, restoring pristine workspace state.
func (a *Arena) Reset(ctx context.Context) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.destroyed {
		return errors.New("arena has been destroyed")
	}

	return a.overlay.Reset()
}

// Fork spawns a child arena branching from the current state of this arena.
func (a *Arena) Fork(ctx context.Context, branchName string) (*Arena, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.destroyed {
		return nil, errors.New("cannot fork destroyed arena")
	}

	childCfg := a.cfg
	childCfg.WorkspaceName = fmt.Sprintf("%s-%s", a.cfg.WorkspaceName, branchName)

	// Child forks over the current merged view of this arena
	childOverlay, err := NewOverlay(a.Path(), "")
	if err != nil {
		return nil, fmt.Errorf("failed to create forked overlay: %w", err)
	}

	return &Arena{
		cfg:        childCfg,
		overlay:    childOverlay,
		parent:     a,
		branchName: branchName,
	}, nil
}

// Promote applies modifications made in this arena back to the pristine base workspace.
func (a *Arena) Promote(ctx context.Context) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.destroyed {
		return errors.New("arena has been destroyed")
	}

	// Promote to root base directory
	if err := a.overlay.Promote(a.cfg.BaseDir); err != nil {
		return fmt.Errorf("failed to promote arena changes to base directory: %w", err)
	}

	// If this is a forked child, also sync changes into the parent arena's path
	if a.parent != nil {
		_ = a.overlay.Promote(a.parent.Path())
	}

	return nil
}

// Destroy cleans up all ephemeral overlay resources and unmounts/deletes temporary trees.
func (a *Arena) Destroy() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.destroyed {
		return nil
	}
	a.destroyed = true
	return a.overlay.Destroy()
}

// stabilizeEnv constructs a deterministic, sanitized environment variable slice:
// - Strips secrets, access keys, tokens, auth passwords.
// - Enforces LC_ALL, LANG, TZ.
// - Enforces pinned PTY geometry (LINES, COLUMNS, TERM).
// - Preserves essential host lookup paths (PATH, SYSTEMROOT).
func (a *Arena) stabilizeEnv(userEnv []string) []string {
	var baseEnv []string
	if len(a.cfg.SanitizedEnv) > 0 {
		baseEnv = a.cfg.SanitizedEnv
	} else {
		baseEnv = os.Environ()
	}

	envMap := make(map[string]string)

	// Ingest base environment, filtering secrets
	for _, kv := range baseEnv {
		k, v, ok := strings.Cut(kv, "=")
		if !ok {
			continue
		}
		if isSecretKey(k) {
			continue
		}
		envMap[k] = v
	}

	// Ingest user-provided overrides, filtering secrets
	for _, kv := range userEnv {
		k, v, ok := strings.Cut(kv, "=")
		if !ok {
			continue
		}
		if isSecretKey(k) {
			continue
		}
		envMap[k] = v
	}

	// Enforce deterministic locale and timezone
	envMap["LC_ALL"] = a.cfg.Locale
	envMap["LANG"] = a.cfg.Locale
	envMap["TZ"] = "UTC"

	// Enforce pinned PTY geometry
	rows := 24
	cols := 80
	if a.cfg.PTYRows > 0 {
		rows = a.cfg.PTYRows
	}
	if a.cfg.PTYCols > 0 {
		cols = a.cfg.PTYCols
	}

	if a.cfg.PinnedPTY || true {
		envMap["LINES"] = strconv.Itoa(rows)
		envMap["COLUMNS"] = strconv.Itoa(cols)
		if _, ok := envMap["TERM"]; !ok {
			envMap["TERM"] = "xterm-256color"
		}
	}

	if !a.cfg.DeterministicTime.IsZero() {
		envMap["FAK_DETERMINISTIC_TIME"] = a.cfg.DeterministicTime.UTC().Format(time.RFC3339)
	}

	// Ensure essential platform system lookup paths are present
	if runtime.GOOS == "windows" {
		if _, ok := envMap["SYSTEMROOT"]; !ok {
			if sr := os.Getenv("SYSTEMROOT"); sr != "" {
				envMap["SYSTEMROOT"] = sr
			}
		}
		if _, ok := envMap["COMSPEC"]; !ok {
			if cs := os.Getenv("COMSPEC"); cs != "" {
				envMap["COMSPEC"] = cs
			}
		}
		if _, ok := envMap["PATHEXT"]; !ok {
			if pe := os.Getenv("PATHEXT"); pe != "" {
				envMap["PATHEXT"] = pe
			}
		}
	}

	result := make([]string, 0, len(envMap))
	for k, v := range envMap {
		result = append(result, k+"="+v)
	}
	return result
}

// isSecretKey returns true if the environment variable name matches common secret/credential patterns.
func isSecretKey(key string) bool {
	upper := strings.ToUpper(key)
	patterns := []string{
		"SECRET",
		"TOKEN",
		"PASSWORD",
		"PASSWD",
		"AUTH",
		"KEY",
		"AWS_",
		"GITHUB_",
		"OPENAI_",
		"ANTHROPIC_",
		"PRIVATE",
		"CREDENTIAL",
		"BEARER",
		"ACCESS_KEY",
		"SESSION_TOKEN",
	}
	for _, p := range patterns {
		if strings.Contains(upper, p) {
			return true
		}
	}
	return false
}
