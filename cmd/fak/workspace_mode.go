package main

import (
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/branchrole"
)

// workspaceModeInfo describes the execution context detected from cwd without
// mutating anything or spawning subprocesses.
type workspaceModeInfo struct {
	Mode             string `json:"mode"`
	InFakRepo        bool   `json:"in_fak_repo"`
	CompanionPresent bool   `json:"companion_present"`
	RepoRoot         string `json:"repo_root,omitempty"`
	CompanionRoot    string `json:"companion_root,omitempty"`
	HasDevPolicy     bool   `json:"has_dev_policy"`
	Note             string `json:"note"`
}

// detectWorkspaceMode classifies cwd into one of:
//   - "companion_repo": inside a fak checkout with sibling fak-private present
//   - "public_repo":    inside a public-only fak checkout (no fak-private)
//   - "standalone":     outside any fak checkout (external project / user dir)
func detectWorkspaceMode() workspaceModeInfo {
	root, err := branchrole.FindRoot("")
	if err != nil || root == "" {
		return workspaceModeInfo{
			Mode: "standalone",
			Note: "outside a fak checkout; runtime verbs, MCP stdio, and local project config work standalone",
		}
	}

	hasDevPolicy := false
	if _, statErr := os.Stat(filepath.Join(root, "examples", "dev-agent-policy.json")); statErr == nil {
		hasDevPolicy = true
	}

	companionRoot := findCompanionPrivateRoot(root)
	if companionRoot != "" {
		return workspaceModeInfo{
			Mode:             "companion_repo",
			InFakRepo:        true,
			CompanionPresent: true,
			RepoRoot:         root,
			CompanionRoot:    companionRoot,
			HasDevPolicy:     hasDevPolicy,
			Note:             "inside fak checkout with companion fak-private detected",
		}
	}

	return workspaceModeInfo{
		Mode:         "public_repo",
		InFakRepo:    true,
		RepoRoot:     root,
		HasDevPolicy: hasDevPolicy,
		Note:         "inside public-only fak checkout (no fak-private companion); runtime and public OSS dev verbs work standalone",
	}
}

func findCompanionPrivateRoot(fakRoot string) string {
	if env := os.Getenv("FAK_PRIVATE_ROOT"); env != "" {
		if info, err := os.Stat(env); err == nil && info.IsDir() {
			return filepath.Clean(env)
		}
	}
	if fakRoot != "" {
		cand := filepath.Clean(filepath.Join(fakRoot, "..", "fak-private"))
		if info, err := os.Stat(filepath.Join(cand, "go.mod")); err == nil && !info.IsDir() {
			return cand
		}
	}
	return ""
}
