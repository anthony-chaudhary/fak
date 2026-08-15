// Package ghexec builds deadlined `gh` invocations (issue #3473).
//
// `gh` is a network CLI: on a dead connection or a missing auth session it
// can block forever, and a bare exec.Command("gh", ...) call site inherits
// that hang. Every construction here therefore carries a context deadline
// (the process is killed when the context expires), disables gh's
// interactive prompting (GH_PROMPT_DISABLED=1) and update notifier, and
// suppresses the console window on Windows via windowgate — so a wedged
// `gh` can never wedge its caller.
package ghexec

import (
	"context"
	"os"
	"os/exec"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// DefaultTimeout bounds one `gh` network round-trip generously; callers on
// hotter paths should pass a tighter budget to CommandTimeout.
const DefaultTimeout = 60 * time.Second

// Command returns a `gh` command bound to ctx: when ctx expires the process
// is killed, so the call can never outlive its caller's budget. Prompting
// is disabled so a missing auth session fails fast with an error instead of
// blocking on TTY input.
func Command(ctx context.Context, args ...string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, "gh", hooks.ScrubGitHubTextArgs(args)...)
	cmd.Env = append(os.Environ(), "GH_PROMPT_DISABLED=1", "GH_NO_UPDATE_NOTIFIER=1")
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd
}

// CommandTimeout is Command with a fresh deadline of d derived from parent
// (context.Background() when parent is nil). The caller must defer cancel().
func CommandTimeout(parent context.Context, d time.Duration, args ...string) (*exec.Cmd, context.CancelFunc) {
	if parent == nil {
		parent = context.Background()
	}
	ctx, cancel := context.WithTimeout(parent, d)
	return Command(ctx, args...), cancel
}
