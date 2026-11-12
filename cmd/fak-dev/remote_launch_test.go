package main

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/servelaunch"
)

// fakeExecAnswer is one canned answer for a remote command substring.
type fakeExecAnswer struct {
	out  string
	code int
}

// installFakeExecRunner swaps the verb's process seam for a fake that answers
// by substring over the remote command (the last argv element). It returns a
// pointer to the recorded call list and restores the real runner on cleanup.
func installFakeExecRunner(t *testing.T, answers map[string]fakeExecAnswer) *[][]string {
	t.Helper()
	calls := &[][]string{}
	prev := remoteLaunchExecRunner
	remoteLaunchExecRunner = func(_ context.Context, argv []string, _ string) (string, int, error) {
		*calls = append(*calls, argv)
		remote := ""
		if len(argv) > 0 {
			remote = argv[len(argv)-1]
		}
		for match, a := range answers {
			if strings.Contains(remote, match) {
				return a.out, a.code, nil
			}
		}
		return "", 0, nil
	}
	t.Cleanup(func() { remoteLaunchExecRunner = prev })
	return calls
}

// TestRemoteLaunchRequiresHost proves the usage guard: no --host is exit 2 with
// the usage line, never a probe of an empty target.
// fak-test:runtime fast est=1ms lane=default
func TestRemoteLaunchRequiresHost(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runRemoteLaunch(&out, &errb, []string{"inspect"}); code != 2 {
		t.Errorf("code = %d, want 2", code)
	}
	if !strings.Contains(errb.String(), "--host is required") {
		t.Errorf("stderr = %q, want the host-required line", errb.String())
	}
}

// TestRemoteLaunchStartRequiresCommand proves the start-mode guard: a launch
// with nothing to launch is a usage error.
// fak-test:runtime fast est=1ms lane=default
func TestRemoteLaunchStartRequiresCommand(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runRemoteLaunch(&out, &errb, []string{"start", "--host", "gpu-box-a"}); code != 2 {
		t.Errorf("code = %d, want 2", code)
	}
	if !strings.Contains(errb.String(), "--command is required") {
		t.Errorf("stderr = %q, want the command-required line", errb.String())
	}
}

// TestRemoteLaunchRejectsUnknownMode proves an unknown mode is a usage error.
// fak-test:runtime fast est=1ms lane=default
func TestRemoteLaunchRejectsUnknownMode(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runRemoteLaunch(&out, &errb, []string{"bogus", "--host", "h"}); code != 2 {
		t.Errorf("code = %d, want 2", code)
	}
}

// TestRemoteLaunchInspectDrivesServelaunch proves the verb drives the real ssh
// argv build and receipt path end to end against a fake exec seam: the issued
// argv is the hardened ssh form and the JSON receipt carries the resolved
// session and the log tail.
// fak-test:runtime fast est=1ms lane=default
func TestRemoteLaunchInspectDrivesServelaunch(t *testing.T) {
	calls := installFakeExecRunner(t, map[string]fakeExecAnswer{
		"which screen": {out: "/usr/bin/screen", code: 0},
		"screen -ls":   {out: "\t999.bench-s1-sglang-1109\t(Detached)\n", code: 0},
		"tail -n":      {out: "server ready\n", code: 0},
	})

	var out, errb bytes.Buffer
	savePath := t.TempDir() + "/r.json"
	code := runRemoteLaunch(&out, &errb, []string{
		"inspect", "--host", "gpu-box-a", "--user", "bench",
		"--server", "s1", "--session", "bench-s1-sglang-1109",
		"--save", savePath, "--json",
	})
	if code != 0 {
		t.Fatalf("code = %d, want 0; stderr=%s", code, errb.String())
	}
	if len(*calls) == 0 || (*calls)[0][0] != "ssh" {
		t.Fatalf("calls = %v, want ssh argv", *calls)
	}
	if !hasPair((*calls)[0], "-o", "BatchMode=yes") {
		t.Errorf("argv = %v, want BatchMode hardening", (*calls)[0])
	}
	stdout := out.String()
	if !strings.Contains(stdout, "999.bench-s1-sglang-1109") {
		t.Errorf("stdout =\n%s\nwant the resolved session", stdout)
	}
	if !strings.Contains(stdout, "server ready") {
		t.Errorf("stdout =\n%s\nwant the log tail", stdout)
	}
	// The receipt landed on disk and reads back.
	if back, err := servelaunch.LoadReceipt(savePath); err != nil {
		t.Fatalf("LoadReceipt: %v", err)
	} else if back.ScreenSession != "999.bench-s1-sglang-1109" {
		t.Errorf("saved session = %q, want the resolved name", back.ScreenSession)
	}
}

// TestRemoteLaunchInspectRecordsFailuresExitNonZero proves the operator gets a
// non-zero exit when a probe records a failure, so an unanswered host is not
// mistaken for a clean inspection.
// fak-test:runtime fast est=1ms lane=default
func TestRemoteLaunchInspectRecordsFailuresExitNonZero(t *testing.T) {
	installFakeExecRunner(t, map[string]fakeExecAnswer{
		"which screen": {out: "/usr/bin/screen", code: 0},
		"screen -ls":   {out: "\t1.other\t(Detached)\n", code: 0},
	})
	var out, errb bytes.Buffer
	code := runRemoteLaunch(&out, &errb, []string{"inspect", "--host", "gpu-box-a", "--session", "bench-absent"})
	if code != 1 {
		t.Errorf("code = %d, want 1 on a recorded probe failure", code)
	}
	if !strings.Contains(out.String(), "not found") {
		t.Errorf("stdout =\n%s\nwant the not-found record in the banner", out.String())
	}
}

// TestRemoteLaunchSaveToDir proves --save DIR writes a named receipt file.
// fak-test:runtime fast est=1ms lane=default
func TestRemoteLaunchSaveToDir(t *testing.T) {
	installFakeExecRunner(t, map[string]fakeExecAnswer{
		"which screen": {out: "/usr/bin/screen", code: 0},
		"screen -ls":   {out: "\t1.a\t(Detached)\n", code: 0},
	})
	dir := t.TempDir()
	var out, errb bytes.Buffer
	if code := runRemoteLaunch(&out, &errb, []string{"inspect", "--host", "gpu-box-a", "--save", dir}); code != 0 {
		t.Fatalf("code = %d, want 0; stderr=%s", code, errb.String())
	}
	if !strings.Contains(errb.String(), "receipt saved to") {
		t.Errorf("stderr = %q, want the saved-to line", errb.String())
	}
}

func hasPair(argv []string, key, val string) bool {
	for i := 0; i+1 < len(argv); i++ {
		if argv[i] == key && argv[i+1] == val {
			return true
		}
	}
	return false
}
