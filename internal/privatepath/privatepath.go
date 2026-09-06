// Package privatepath resolves private operator artifacts outside the public checkout.
package privatepath

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const EnvRoot = "FAK_PRIVATE_ROOT"

type Options struct {
	RepoRoot string
	Root     string
	Now      time.Time
	Create   bool
}

type Result struct {
	Path    string `json:"path"`
	Root    string `json:"root"`
	RunID   string `json:"run_id"`
	Created bool   `json:"created"`
}

// ResolveRun returns an opaque run directory below the paired private repository.
// It intentionally accepts no machine, channel, model, customer, or campaign label:
// those identifiers must not become path names in the public checkout or process argv.
func ResolveRun(opts Options) (Result, error) {
	root, err := resolveRoot(opts)
	if err != nil {
		return Result{}, err
	}
	if opts.Now.IsZero() {
		opts.Now = time.Now().UTC()
	}
	suffix := make([]byte, 4)
	if _, err := rand.Read(suffix); err != nil {
		return Result{}, fmt.Errorf("generate opaque run id: %w", err)
	}
	runID := opts.Now.UTC().Format("20060102T150405Z") + "-" + hex.EncodeToString(suffix)
	path := filepath.Join(root, "fleet-runs", "codex", runID)
	created := false
	if opts.Create {
		if err := os.MkdirAll(path, 0o700); err != nil {
			return Result{}, fmt.Errorf("create private run directory: %w", err)
		}
		created = true
	}
	return Result{Path: path, Root: root, RunID: runID, Created: created}, nil
}

// ResolveRoot resolves the private repository root from options, environment, or relative sibling directory.
func ResolveRoot(opts Options) (string, error) {
	return resolveRoot(opts)
}

func resolveRoot(opts Options) (string, error) {
	root := strings.TrimSpace(opts.Root)
	if root == "" {
		root = strings.TrimSpace(os.Getenv(EnvRoot))
	}
	if root == "" && opts.RepoRoot != "" {
		root = filepath.Join(filepath.Dir(opts.RepoRoot), "fak-private")
	}
	if root == "" {
		return "", errors.New("private root unavailable: set FAK_PRIVATE_ROOT or run inside a checkout paired with ../fak-private")
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return "", fmt.Errorf("resolve private root: %w", err)
	}
	info, err := os.Stat(abs)
	if err != nil {
		return "", fmt.Errorf("private root unavailable at %s: %w", abs, err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("private root is not a directory: %s", abs)
	}
	if opts.RepoRoot != "" {
		publicAbs, err := filepath.Abs(opts.RepoRoot)
		if err != nil {
			return "", fmt.Errorf("resolve public root: %w", err)
		}
		rel, err := filepath.Rel(publicAbs, abs)
		if err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && rel != "." {
			return "", fmt.Errorf("private root must resolve outside the public checkout: %s", abs)
		}
		if rel == "." {
			return "", fmt.Errorf("private root must not be the public checkout: %s", abs)
		}
	}
	return abs, nil
}
