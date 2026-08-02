//go:build windows

package main

import (
	"bytes"
	"strings"
	"syscall"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/hostresurrect"
)

// TestHostRelaunchBrokerDryRunMatchesSyscallEscapeArg pins the --dry-run
// preview to the authority rather than to a hand-written expectation: on
// Windows the broker's launch reaches syscall.StartProcess with no
// SysProcAttr.CmdLine set, so the kernel receives makeCmdLine(argv) — every
// element through syscall.EscapeArg, joined by one space. Comparing the preview
// against that same construction proves the previewed line is the command line
// that would really run, including for a CWD holding a space and for arguments
// holding quotes and trailing backslashes.
func TestHostRelaunchBrokerDryRunMatchesSyscallEscapeArg(t *testing.T) {
	for _, tc := range []struct {
		name   string
		cwd    string
		window string
		handle string
	}{
		{name: "plain path", cwd: `C:\lane\proj`, handle: "g1"},
		{name: "program files path", cwd: `C:\Program Files\fak\lane`, handle: "g1"},
		{name: "path with trailing backslash", cwd: `C:\Program Files\fak\`, handle: "g1"},
		{name: "window id with a space", cwd: `C:\lane\proj`, window: "tab 2", handle: "g1"},
		{name: "handle with a quote", cwd: `C:\lane\proj`, handle: `g"1`},
		{name: "handle with a space and a quote", cwd: `C:\Program Files\p`, handle: `g 1"x`},
		{name: "handle with a trailing backslash", cwd: `C:\lane\proj`, handle: `g 1\`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			command := []string{"claude", "--resume", tc.handle}
			req := hostresurrect.Request{
				Schema:       hostresurrect.Schema,
				EventID:      "evt",
				Session:      "s1",
				CWD:          tc.cwd,
				WindowID:     tc.window,
				Command:      command,
				ResumeHandle: tc.handle,
			}
			if _, err := hostresurrect.Enqueue(dir, req); err != nil {
				t.Fatal(err)
			}
			var out, stderr bytes.Buffer
			if rc := runHostRelaunchBroker(&out, &stderr, []string{"--dir", dir, "--dry-run"}); rc != 0 {
				t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
			}
			window := tc.window
			if strings.TrimSpace(window) == "" {
				window = "new"
			}
			argv := append([]string{"wt.exe", "-w", window, "new-tab", "-d", tc.cwd}, command...)
			escaped := make([]string, 0, len(argv))
			for _, a := range argv {
				escaped = append(escaped, syscall.EscapeArg(a))
			}
			want := strings.Join(escaped, " ")
			if got := strings.TrimSuffix(out.String(), "\n"); got != want {
				t.Fatalf("preview is not the real Windows command line\n got: %s\nwant: %s", got, want)
			}
		})
	}
}
