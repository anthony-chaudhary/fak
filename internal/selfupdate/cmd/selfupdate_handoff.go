package selfupdatecmd

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/selfinstall"
)

const (
	selfUpdateSessionEnv  = "FAK_HANDOFF_SESSION_ID"
	selfUpdateRevisionEnv = "FAK_HANDOFF_REVISION"
)

type selfUpdateHandoffReceipt struct {
	State             selfinstall.HandoffState `json:"state"`
	SessionID         string                   `json:"session_id"`
	SuccessorRevision string                   `json:"successor_revision"`
	Detail            string                   `json:"detail,omitempty"`
}

func runSelfUpdateHandoff(ctx context.Context, target, sessionID, revision string, args []string) selfUpdateHandoffReceipt {
	var handoff selfinstall.Handoff
	result := handoff.Drain(ctx, sessionID, revision, func(ctx context.Context, session, rev string) error {
		cmd := selfUpdateSuccessorCommand(ctx, target, args, session, rev)
		cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
		return cmd.Start()
	})
	receipt := selfUpdateHandoffReceipt{State: result.State, SessionID: result.SessionID, SuccessorRevision: result.Revision}
	if result.Err != nil {
		receipt.Detail = result.Err.Error()
	}
	return receipt
}

func selfUpdateSuccessorCommand(ctx context.Context, target string, args []string, sessionID, revision string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, target, args...)
	cmd.Env = withSelfUpdateHandoffEnv(os.Environ(), sessionID, revision)
	configureSelfUpdateSuccessor(cmd)
	return cmd
}

func withSelfUpdateHandoffEnv(env []string, sessionID, revision string) []string {
	out := make([]string, 0, len(env)+2)
	for _, item := range env {
		name, _, _ := strings.Cut(item, "=")
		if strings.EqualFold(name, selfUpdateSessionEnv) || strings.EqualFold(name, selfUpdateRevisionEnv) {
			continue
		}
		out = append(out, item)
	}
	return append(out,
		fmt.Sprintf("%s=%s", selfUpdateSessionEnv, sessionID),
		fmt.Sprintf("%s=%s", selfUpdateRevisionEnv, revision),
	)
}
