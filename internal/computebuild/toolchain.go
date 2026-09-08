package computebuild

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// DiscoverToolchain discovers the host's available compute build toolchain.
func DiscoverToolchain() (*Toolchain, error) {
	return discoverToolchainOS()
}

// LookPath searches for an executable in the system PATH.
func LookPath(file string) (string, error) {
	return exec.LookPath(file)
}

// LookPathInDirs checks if the given executable file exists in any of the specified directories.
func LookPathInDirs(file string, dirs ...string) string {
	for _, dir := range dirs {
		if dir == "" {
			continue
		}
		cand := filepath.Join(dir, file)
		if FileExists(cand) {
			return cand
		}
	}
	return ""
}

// FileExists returns true if path exists and is not a directory.
func FileExists(path string) bool {
	if path == "" {
		return false
	}
	fi, err := os.Stat(path)
	if err != nil {
		return false
	}
	return !fi.IsDir()
}

// DirExists returns true if path exists and is a directory.
func DirExists(path string) bool {
	if path == "" {
		return false
	}
	fi, err := os.Stat(path)
	if err != nil {
		return false
	}
	return fi.IsDir()
}

// ResolveRepoRoot finds the module root directory (containing go.mod) by walking upwards.
func ResolveRepoRoot(startDir string) (string, error) {
	if startDir == "" {
		var err error
		startDir, err = os.Getwd()
		if err != nil {
			return "", fmt.Errorf("resolving working directory: %w", err)
		}
	}
	cur, err := filepath.Abs(startDir)
	if err != nil {
		return "", err
	}
	for {
		if FileExists(filepath.Join(cur, "go.mod")) {
			return cur, nil
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			return "", fmt.Errorf("go.mod not found starting from %s", startDir)
		}
		cur = parent
	}
}

// RunCmd executes a command with specified directory, environment, and streams.
func RunCmd(ctx context.Context, name string, args []string, dir string, env []string, stdout, stderr io.Writer) error {
	cmd := exec.CommandContext(ctx, name, args...)
	if dir != "" {
		cmd.Dir = dir
	}
	if len(env) > 0 {
		cmd.Env = env
	}
	if stdout != nil {
		cmd.Stdout = stdout
	}
	if stderr != nil {
		cmd.Stderr = stderr
	}
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("%s %s failed: %w", name, strings.Join(args, " "), err)
	}
	return nil
}
