package dockerprocess

import (
	"context"
	"os/exec"
)

// Available reports whether the Docker CLI can be resolved without starting it.
func Available() bool {
	_, err := exec.LookPath("docker")
	return err == nil
}

// ComposeCombinedOutput runs one Docker Compose operation and captures its output.
// The executable is intentionally a static literal; callers control only Compose args.
func ComposeCombinedOutput(ctx context.Context, dir string, env []string, args ...string) ([]byte, error) {
	return composeCommand(ctx, dir, env, args...).CombinedOutput()
}

// ComposeRun runs one Docker Compose operation without retaining its output.
func ComposeRun(ctx context.Context, dir string, env []string, args ...string) error {
	return composeCommand(ctx, dir, env, args...).Run()
}

func composeCommand(ctx context.Context, dir string, env []string, args ...string) *exec.Cmd {
	argv := make([]string, 1, len(args)+1)
	argv[0] = "compose"
	argv = append(argv, args...)
	cmd := exec.CommandContext(ctx, "docker", argv...)
	cmd.Dir = dir
	cmd.Env = append([]string(nil), env...)
	return cmd
}
