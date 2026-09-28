package naivecontrol

import (
	"bytes"
	"context"
	"errors"
	"os"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// ExecGit is the production GitRunner. It clears the git environment a hook or a
// worktree exports (GIT_DIR, GIT_WORK_TREE, GIT_INDEX_FILE, ...), so a verdict is
// always about dir's repository and never about whichever repo invoked fak.
// assumecheck's default runner does not clear them, which is why this package
// injects its own.
func ExecGit(ctx context.Context, dir string, args ...string) (string, int, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = dir
	cmd.Env = cleanGitEnv(os.Environ())
	var out bytes.Buffer
	cmd.Stdout = &out
	err := cmd.Run()
	if err == nil {
		return out.String(), 0, nil
	}
	var exit *exec.ExitError
	if errors.As(err, &exit) {
		return out.String(), exit.ExitCode(), nil
	}
	return "", -1, err
}

var repoScopedGitEnv = []string{
	"GIT_DIR=", "GIT_WORK_TREE=", "GIT_INDEX_FILE=", "GIT_OBJECT_DIRECTORY=",
	"GIT_ALTERNATE_OBJECT_DIRECTORIES=", "GIT_COMMON_DIR=", "GIT_NAMESPACE=",
	"GIT_PREFIX=", "GIT_CEILING_DIRECTORIES=",
}

func cleanGitEnv(env []string) []string {
	out := make([]string, 0, len(env))
	for _, kv := range env {
		drop := false
		for _, p := range repoScopedGitEnv {
			if strings.HasPrefix(kv, p) {
				drop = true
				break
			}
		}
		if !drop {
			out = append(out, kv)
		}
	}
	return out
}
