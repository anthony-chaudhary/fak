//go:build linux

package taskrun

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"
)

var bubblewrapLookPath = exec.LookPath
var bubblewrapCommandContext = exec.CommandContext

func platformSandboxAvailable() error {
	bwrap, err := bubblewrapLookPath("bwrap")
	if err != nil {
		return &SandboxUnavailableError{Reason: "bubblewrap executable not found"}
	}
	goRoot, err := filepath.EvalSymlinks(filepath.Clean(runtime.GOROOT()))
	if err != nil {
		return &SandboxUnavailableError{Reason: "resolve pinned GOROOT: " + err.Error()}
	}
	probeCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	probe := bubblewrapCommandContext(probeCtx, bwrap,
		"--unshare-all", "--die-with-parent", "--new-session",
		"--ro-bind", goRoot, "/goroot", "--proc", "/proc", "--dev", "/dev",
		"--tmpfs", "/tmp", "--clearenv", "--setenv", "HOME", "/tmp",
		"/goroot/bin/go", "version",
	)
	probe.Env = []string{}
	if output, err := probe.CombinedOutput(); err != nil {
		return &SandboxUnavailableError{Reason: fmt.Sprintf("bubblewrap functional probe failed: %v: %s", err, boundedText(output, 512))}
	}
	return nil
}

func platformSandboxName() string { return "linux-bwrap-userns" }

func platformSandboxCommand(ctx context.Context, cfg sandboxCommandConfig) (*exec.Cmd, error) {
	bwrap, err := bubblewrapLookPath("bwrap")
	if err != nil {
		return nil, &SandboxUnavailableError{Reason: "bubblewrap executable not found"}
	}
	// The namespace exposes only these explicit binds. In particular it never
	// binds /, /etc, the oracle, or a parent of the oracle.
	for label, protected := range map[string]string{"trial": cfg.Trial, "oracle": cfg.Oracle, "cache": cfg.Cache, "temp": cfg.Temp} {
		if pathsOverlap(protected, cfg.GoRoot) {
			return nil, fmt.Errorf("agentbench sandbox: protected %s root overlaps GOROOT bind", label)
		}
	}
	trialBind := "--bind"
	if !cfg.TrialWritable {
		trialBind = "--ro-bind"
	}
	args := []string{
		"--unshare-all", "--die-with-parent", "--new-session",
		"--ro-bind", cfg.GoRoot, "/goroot",
		trialBind, cfg.Trial, "/workspace",
		"--bind", cfg.Cache, "/gocache",
		"--bind", cfg.Temp, "/tmp",
		"--proc", "/proc", "--dev", "/dev", "--chdir", "/workspace",
		"--clearenv",
		"--setenv", "GOTOOLCHAIN", "local",
		"--setenv", "GOENV", "off",
		"--setenv", "GOPROXY", "off",
		"--setenv", "GOSUMDB", "off",
		"--setenv", "CGO_ENABLED", "0",
		"--setenv", "GOFLAGS", "-p=2",
		"--setenv", "GOMAXPROCS", "2",
		"--setenv", "GOCACHE", "/gocache",
		"--setenv", "GOTMPDIR", "/tmp",
		"--setenv", "TMPDIR", "/tmp",
		"--setenv", "HOME", "/tmp",
		"--setenv", "PATH", "/goroot/bin",
		"/goroot/bin/go", "test", "./...", "-count=1",
	}
	cmd := bubblewrapCommandContext(ctx, bwrap, args...)
	cmd.Dir = string(os.PathSeparator)
	cmd.Env = []string{}
	return cmd, nil
}

func boundedText(data []byte, limit int) string {
	if len(data) > limit {
		data = data[:limit]
	}
	return string(data)
}
