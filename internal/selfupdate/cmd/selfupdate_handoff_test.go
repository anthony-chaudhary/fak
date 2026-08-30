package selfupdatecmd

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/selfinstall"
)

func TestSelfUpdateSuccessorCommandPreservesIdentityAndRevision(t *testing.T) {
	cmd := selfUpdateSuccessorCommand(context.Background(), os.Args[0], []string{"-test.run=none"}, "session-42", "revision-9")
	if cmd.Path != os.Args[0] || len(cmd.Args) != 2 || cmd.Args[1] != "-test.run=none" {
		t.Fatalf("command = %#v", cmd.Args)
	}
	joined := strings.Join(cmd.Env, "\n")
	if !strings.Contains(joined, selfUpdateSessionEnv+"=session-42") || !strings.Contains(joined, selfUpdateRevisionEnv+"=revision-9") {
		t.Fatalf("environment missing handoff identity:\n%s", joined)
	}
	if cmd.SysProcAttr == nil {
		t.Fatal("successor has no platform process attributes")
	}
}

func TestWithSelfUpdateHandoffEnvReplacesInheritedValues(t *testing.T) {
	got := withSelfUpdateHandoffEnv([]string{"KEEP=yes", selfUpdateSessionEnv + "=old", selfUpdateRevisionEnv + "=old"}, "new-session", "new-rev")
	joined := strings.Join(got, "\n")
	for _, want := range []string{"KEEP=yes", selfUpdateSessionEnv + "=new-session", selfUpdateRevisionEnv + "=new-rev"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("%q missing from %v", want, got)
		}
	}
	if strings.Contains(joined, "=old") {
		t.Fatalf("stale identity retained: %v", got)
	}
}

func TestRunSelfUpdateHandoffRefusesCanceledLaunch(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	got := runSelfUpdateHandoff(ctx, os.Args[0], "session", "revision", nil)
	if got.State != selfinstall.HandoffRefused || got.SessionID != "session" || got.SuccessorRevision != "revision" || !errors.Is(ctx.Err(), context.Canceled) {
		t.Fatalf("receipt = %+v", got)
	}
}
