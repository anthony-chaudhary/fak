package scratchjanitor

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const DefaultMaxAge = 168 * time.Hour

type Config struct {
	Root       string
	MaxAge     time.Duration
	Now        func() time.Time
	Referenced map[string]bool
	Apply      bool
}

type Candidate struct {
	Project        string `json:"project"`
	Session        string `json:"session"`
	Path           string `json:"path"`
	ScratchpadPath string `json:"scratchpad_path"`
	AgeSeconds     int64  `json:"age_seconds"`
}

type Action struct {
	Action string `json:"action"`
	Path   string `json:"path"`
}

type Result struct {
	Mode       string      `json:"mode"`
	Candidates []Candidate `json:"candidates"`
	Actions    []Action    `json:"actions"`
}

func Scan(cfg Config) (Result, error) {
	result := Result{
		Mode:       "dry-run",
		Candidates: []Candidate{},
		Actions:    []Action{},
	}
	if cfg.Apply {
		result.Mode = "apply"
	}
	if strings.TrimSpace(cfg.Root) == "" {
		return result, fmt.Errorf("root is required")
	}
	if cfg.MaxAge < 0 {
		return result, fmt.Errorf("max age must not be negative")
	}

	root, err := canonicalPath(cfg.Root)
	if err != nil {
		return result, fmt.Errorf("canonicalize root: %w", err)
	}
	referenced, err := canonicalReferences(cfg.Referenced)
	if err != nil {
		return result, err
	}
	now := time.Now
	if cfg.Now != nil {
		now = cfg.Now
	}
	currentTime := now()
	cutoff := currentTime.Add(-cfg.MaxAge)

	projects, err := os.ReadDir(root)
	if err != nil {
		return result, fmt.Errorf("read root %q: %w", root, err)
	}
	for _, projectEntry := range projects {
		if !projectEntry.IsDir() {
			continue
		}
		project := projectEntry.Name()
		projectPath := filepath.Join(root, project)
		sessions, err := os.ReadDir(projectPath)
		if err != nil {
			return result, fmt.Errorf("read project %q: %w", projectPath, err)
		}
		for _, sessionDir := range sessions {
			if !sessionDir.IsDir() {
				continue
			}
			session := sessionDir.Name()
			sessionPath, err := canonicalPath(filepath.Join(projectPath, session))
			if err != nil {
				return result, fmt.Errorf("canonicalize session %q: %w", session, err)
			}
			info, err := sessionDir.Info()
			if err != nil {
				return result, fmt.Errorf("stat session %q: %w", sessionPath, err)
			}
			if !info.ModTime().Before(cutoff) || referenced[sessionPath] {
				continue
			}
			age := currentTime.Sub(info.ModTime())
			candidate := Candidate{
				Project:        project,
				Session:        session,
				Path:           sessionPath,
				ScratchpadPath: sessionPath,
				AgeSeconds:     int64(age / time.Second),
			}
			result.Candidates = append(result.Candidates, candidate)
		}
	}

	if !cfg.Apply {
		return result, nil
	}
	for _, candidate := range result.Candidates {
		if err := verifyTwoLevelsBelow(root, candidate.Path); err != nil {
			return result, err
		}
		if err := os.RemoveAll(candidate.Path); err != nil {
			return result, fmt.Errorf("remove %q: %w", candidate.Path, err)
		}
		result.Actions = append(result.Actions, Action{
			Action: "removed",
			Path:   candidate.Path,
		})
	}
	return result, nil
}

func canonicalReferences(paths map[string]bool) (map[string]bool, error) {
	result := make(map[string]bool, len(paths))
	for path, isReferenced := range paths {
		if !isReferenced {
			continue
		}
		canonical, err := canonicalPath(path)
		if err != nil {
			return nil, fmt.Errorf("canonicalize reference %q: %w", path, err)
		}
		result[canonical] = true
	}
	return result, nil
}

func canonicalPath(path string) (string, error) {
	absolute, err := filepath.Abs(filepath.Clean(path))
	if err != nil {
		return "", err
	}
	return filepath.Clean(absolute), nil
}

func verifyTwoLevelsBelow(root, path string) error {
	relative, err := filepath.Rel(root, path)
	if err != nil {
		return fmt.Errorf("verify removal path %q: %w", path, err)
	}
	if relative == "." || filepath.IsAbs(relative) {
		return fmt.Errorf("refuse removal outside root/project/session: %q", path)
	}
	parts := strings.FieldsFunc(relative, func(r rune) bool {
		return r == '/' || r == '\\'
	})
	if len(parts) != 2 || parts[0] == ".." || parts[1] == ".." {
		return fmt.Errorf("refuse removal outside root/project/session: %q", path)
	}
	return nil
}
