package computebuild

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
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

// ParseEnvBlock parses KEY=VALUE environment blocks from command output.
func ParseEnvBlock(raw []byte) map[string]string {
	scanner := bufio.NewScanner(bytes.NewReader(raw))
	env := make(map[string]string)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		idx := strings.Index(line, "=")
		if idx > 0 {
			key := line[:idx]
			val := line[idx+1:]
			env[key] = val
		}
	}
	return env
}

// compareVersionStrings compares two dotted version strings (e.g. "1.4.350.0" vs "1.3.290.0").
// Returns 1 if v1 > v2, -1 if v1 < v2, and 0 if equal.
func compareVersionStrings(v1, v2 string) int {
	p1 := strings.Split(v1, ".")
	p2 := strings.Split(v2, ".")
	maxLen := len(p1)
	if len(p2) > maxLen {
		maxLen = len(p2)
	}
	for i := 0; i < maxLen; i++ {
		var n1, n2 int
		if i < len(p1) {
			n1, _ = strconv.Atoi(p1[i])
		}
		if i < len(p2) {
			n2, _ = strconv.Atoi(p2[i])
		}
		if n1 > n2 {
			return 1
		}
		if n1 < n2 {
			return -1
		}
	}
	return 0
}
