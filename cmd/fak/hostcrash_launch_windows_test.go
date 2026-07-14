//go:build windows

package main

import (
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/hostresurrect"
)

func TestLaunchHostSessionPlatformUsesFreshWindowCWDAndResumeContext(t *testing.T) {
	if os.Getenv("FAK_WT_LAUNCH_HELPER") == "1" {
		if os.Getenv("FAK_RESUME_HANDLE") != "g123" || os.Getenv("FAK_HOST_CRASH_EVENT") != "evt" {
			os.Exit(3)
		}
		os.Exit(0)
	}
	old := hostSessionExecCommand
	var gotName string
	var gotArgs []string
	hostSessionExecCommand = func(name string, args ...string) *exec.Cmd {
		gotName, gotArgs = name, append([]string(nil), args...)
		cmd := exec.Command(os.Args[0], "-test.run=TestLaunchHostSessionPlatformUsesFreshWindowCWDAndResumeContext")
		cmd.Env = append(os.Environ(), "FAK_WT_LAUNCH_HELPER=1")
		return cmd
	}
	t.Cleanup(func() { hostSessionExecCommand = old })
	pid, err := launchHostSessionPlatform(hostresurrect.Request{EventID: "evt", CWD: `C:\work\repo`, Command: []string{"claude", "--continue"}, ResumeHandle: "g123"})
	if err != nil || pid <= 0 {
		t.Fatalf("launch pid=%d err=%v", pid, err)
	}
	if gotName != "wt.exe" {
		t.Fatalf("launcher=%q", gotName)
	}
	joined := strings.Join(gotArgs, "|")
	if joined != `-w|new|new-tab|-d|C:\work\repo|claude|--continue` {
		t.Fatalf("args=%q", joined)
	}
}
