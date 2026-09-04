package workerworktree

import (
	"os"
	"path/filepath"
	"strings"
)

const (
	// DefaultsSchema is the schema identifier for managed-worker worktree defaults.
	DefaultsSchema = "fak.worktree.defaults.v1"

	// DefaultLeaseIdentityBasis describes the identity basis used to distinguish
	// and register worker worktrees.
	DefaultLeaseIdentityBasis = "lane_key_timestamp"
)

// DefaultsReport captures the resolved default configuration and environment
// overrides for managed worker worktrees.
type DefaultsReport struct {
	Schema                    string   `json:"schema"`
	RepoRoot                  string   `json:"repo_root"`
	WorkerWorktreeRoot        string   `json:"worker_worktree_root"`
	RootSource                string   `json:"root_source"`
	DefaultLeaseIdentityBasis string   `json:"default_lease_identity_basis"`
	SupportedEnvOverrides     []string `json:"supported_env_overrides"`
}

func scrubPath(p string) string {
	p = strings.TrimSpace(p)
	if p == "" {
		return ""
	}
	clean := filepath.Clean(filepath.FromSlash(p))
	if filepath.IsAbs(clean) {
		if abs, err := filepath.Abs(clean); err == nil {
			return filepath.Clean(abs)
		}
	}
	return clean
}

func resolveRootSource() string {
	if strings.TrimSpace(os.Getenv(WorktreeRootEnv)) != "" {
		return "environment: " + WorktreeRootEnv
	}
	if os.Getenv("LOCALAPPDATA") != "" {
		return "os_fallback: LOCALAPPDATA"
	}
	return "os_fallback: temp_dir"
}

// ResolveDefaults inspects the environment and OS defaults to produce a DefaultsReport.
// It reuses DefaultRoot() and WorktreeRootEnv, scrubbing path fields.
func ResolveDefaults(repoRoot string) DefaultsReport {
	return DefaultsReport{
		Schema:                    DefaultsSchema,
		RepoRoot:                  scrubPath(repoRoot),
		WorkerWorktreeRoot:        scrubPath(DefaultRoot()),
		RootSource:                resolveRootSource(),
		DefaultLeaseIdentityBasis: DefaultLeaseIdentityBasis,
		SupportedEnvOverrides:     []string{WorktreeRootEnv},
	}
}
