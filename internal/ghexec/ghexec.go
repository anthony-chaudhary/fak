// Package ghexec builds deadlined gh invocations.
package ghexec

import (
	"context"
	"os"
	"os/exec"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// DefaultTimeout bounds a single gh network round-trip.
const DefaultTimeout = 60 * time.Second

// Command builds a deadlined gh exec.Cmd configured for non-interactive execution.
func Command(ctx context.Context, args ...string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, "gh", hooks.ScrubGitHubTextArgs(args)...)
	cmd.Env = append(os.Environ(), "GH_PROMPT_DISABLED=1", "GH_NO_UPDATE_NOTIFIER=1")
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd
}

// CommandTimeout derives a deadlined context from parent and returns the gh command.
func CommandTimeout(parent context.Context, d time.Duration, args ...string) (*exec.Cmd, context.CancelFunc) {
	if parent == nil {
		parent = context.Background()
	}
	ctx, cancel := context.WithTimeout(parent, d)
	return Command(ctx, args...), cancel
}
