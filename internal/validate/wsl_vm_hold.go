package validate

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// The WSL2 VM shuts down once no wsl.exe session has been attached for its idle
// timeout, and the reboot wipes /tmp. The validator builds and uses its WSL
// workspace across separate wsl.exe launches, so on a loaded host the gap
// between two launches can outlast that timeout and the next step finds
// /tmp/fak-validate-* gone. A held session keeps the VM up for the run; the
// boot id recorded at extraction names a restart that happened anyway.

// errValidateWSLVMRestarted marks a workspace step that failed because the WSL
// VM rebooted after the workspace was extracted.
var errValidateWSLVMRestarted = errors.New("WSL_VM_RESTARTED")

const (
	validateWSLBootIDPath   = "/proc/sys/kernel/random/boot_id"
	validateWSLBootIDPrefix = "__FAK_VALIDATE_BOOT_ID__="
	// validateWSLHoldMargin covers the deferred workspace cleanup after the run
	// deadline; it is also the most extra VM uptime a killed validator leaves.
	validateWSLHoldMargin      = time.Minute
	validateWSLHoldNoDeadline  = time.Hour
	validateWSLHoldReleaseWait = 5 * time.Second
)

// holdValidateWSLVM attaches one `wsl.exe -e sleep` session until release. It
// deliberately ignores cancellation of ctx so the VM stays up through the
// deferred workspace cleanup after a timeout. A hold that fails to launch only
// loses the mitigation; the boot-id check still types the resulting failure.
func holdValidateWSLVM(ctx context.Context) (release func()) {
	seconds := strconv.FormatInt(validateWSLHoldSeconds(ctx), 10)
	holdCtx, cancel := context.WithCancel(context.Background())
	launch := validateWSLCommand
	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _ = launch(holdCtx, "-e", "sleep", seconds)
	}()
	return func() {
		cancel()
		timer := time.NewTimer(validateWSLHoldReleaseWait)
		defer timer.Stop()
		select {
		case <-done:
		case <-timer.C:
		}
	}
}

func validateWSLHoldSeconds(ctx context.Context) int64 {
	bound := validateWSLHoldNoDeadline
	if deadline, ok := ctx.Deadline(); ok {
		bound = max(deadline.Sub(validateNow()), 0)
	}
	bound += validateWSLHoldMargin
	return int64((bound + time.Second - 1) / time.Second)
}

func parseValidateWSLBootID(out []byte) string {
	for _, line := range strings.Split(string(out), "\n") {
		if id, ok := strings.CutPrefix(strings.TrimSpace(line), validateWSLBootIDPrefix); ok {
			return strings.TrimSpace(id)
		}
	}
	return ""
}

// typeValidateWSLWorkspaceLoss wraps cause with errValidateWSLVMRestarted when
// the VM now runs a different boot than the one the workspace was built in. An
// unknown anchor or a failed probe returns cause unchanged.
func typeValidateWSLWorkspaceLoss(ctx context.Context, bootID, dir string, cause error) error {
	if cause == nil || bootID == "" || ctx.Err() != nil {
		return cause
	}
	probeCtx, cancel := context.WithTimeout(ctx, validateWSLPreflightBounds()[0])
	defer cancel()
	out, err := validateWSLCommand(probeCtx, "-e", "cat", validateWSLBootIDPath)
	now := strings.TrimSpace(string(out))
	if err != nil || now == "" || now == bootID {
		return cause
	}
	return fmt.Errorf("%w: workspace %s was built in WSL boot %s, the VM now runs boot %s: %w", errValidateWSLVMRestarted, dir, bootID, now, cause)
}
