package gitgate

import (
	"github.com/anthony-chaudhary/fak/internal/gitbroker"
)

// ShadowCheckpointer manages per-turn shadow checkpoints preserving tracked and untracked files (#10980).
type ShadowCheckpointer = gitbroker.ShadowCheckpointer

// GitShadowCheckpointer is the default git-backed shadow checkpointer.
type GitShadowCheckpointer = gitbroker.GitShadowCheckpointer

// CreateShadowCheckpoint captures working tree modifications and untracked files
// into a shadow stash commit with a 3rd-parent commit for untracked files (#10980).
func CreateShadowCheckpoint(repoDir string) (string, error) {
	return gitbroker.CreateShadowCheckpoint(repoDir)
}

// RestoreShadowCheckpoint atomically restores working tree modifications and untracked files
// from a shadow checkpoint commit without altering trunk HEAD history (#10980).
func RestoreShadowCheckpoint(repoDir, ref string) error {
	return gitbroker.RestoreShadowCheckpoint(repoDir, ref)
}
