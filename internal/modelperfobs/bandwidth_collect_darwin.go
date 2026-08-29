//go:build darwin

package modelperfobs

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"time"
)

const (
	darwinCollectionTimeout  = 2 * time.Second
	darwinCommandOutputLimit = 1 << 20
)

func collectHostSnapshot() (hostSnapshot, error) {
	ctx, cancel := context.WithTimeout(context.Background(), darwinCollectionTimeout)
	defer cancel()
	return collectDarwinHostSnapshot(ctx, os.Getpid(), runDarwinNativeCommand, time.Now), nil
}

// runDarwinNativeCommand bounds the whole collection with ctx and caps each
// native command's stdout. Command failures are handled by the caller as
// unavailable fields rather than synthesized zero-valued observations.
func runDarwinNativeCommand(ctx context.Context, name string, args ...string) (string, error) {
	var cmd *exec.Cmd
	switch name {
	case "/usr/bin/vm_stat":
		cmd = exec.CommandContext(ctx, "/usr/bin/vm_stat", args...)
	case "/usr/sbin/sysctl":
		cmd = exec.CommandContext(ctx, "/usr/sbin/sysctl", args...)
	case "/bin/ps":
		cmd = exec.CommandContext(ctx, "/bin/ps", args...)
	default:
		return "", fmt.Errorf("unsupported Darwin native command %q", name)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", err
	}
	cmd.Stderr = io.Discard
	if err := cmd.Start(); err != nil {
		return "", err
	}
	out, readErr := io.ReadAll(io.LimitReader(stdout, darwinCommandOutputLimit+1))
	if readErr != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return "", readErr
	}
	if len(out) > darwinCommandOutputLimit {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return "", fmt.Errorf("darwin command output exceeds %d bytes", darwinCommandOutputLimit)
	}
	if err := cmd.Wait(); err != nil {
		return "", err
	}
	return string(out), nil
}
