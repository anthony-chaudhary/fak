package main

import (
	"bytes"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/hostresurrect"
)

func TestHostRelaunchBrokerExpandsTildeSpool(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	dir := filepath.Join(home, "spool")
	cwd := t.TempDir()
	req := hostresurrect.Request{Schema: hostresurrect.Schema, EventID: "evt", Session: "g1", CWD: cwd, Command: []string{"claude", "--resume", "g1"}, ResumeHandle: "g1"}
	if _, err := hostresurrect.Enqueue(dir, req); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	if rc := runHostRelaunchBroker(&out, &stderr, []string{"--dir", "~/spool", "--dry-run"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(out.String(), "wt.exe") {
		t.Fatalf("expanded spool was not drained: %q", out.String())
	}
}

func TestHostRelaunchBrokerDryRunDrainsTypedSpool(t *testing.T) {
	dir := t.TempDir()
	cwd := t.TempDir()
	req := hostresurrect.Request{Schema: hostresurrect.Schema, EventID: "evt", Session: "g1", CWD: cwd, Command: []string{"claude", "--resume", "g1"}, ResumeHandle: "g1"}
	if _, err := hostresurrect.Enqueue(dir, req); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	if rc := runHostRelaunchBroker(&out, &stderr, []string{"--dir", dir, "--dry-run"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	got := out.String()
	for _, w := range []string{"wt.exe", "new-tab", cwd, "claude", "--resume", "g1"} {
		if !strings.Contains(got, w) {
			t.Fatalf("%q missing %q", got, w)
		}
	}
	pending, _ := hostresurrect.Pending(dir)
	if len(pending) != 1 {
		t.Fatal("dry run consumed request")
	}
}

// TestHostRelaunchBrokerDryRunPreviewsRealArgv pins --dry-run to the command
// line the launch actually builds. The preview exists so an operator can decide
// whether the real invocation is safe, so a rendering that only looks like the
// argv is a defect: %q Go-escapes, which doubles every backslash in a Windows
// CWD (C:\Users\… previewed as C:\\Users\…) while exec.Command passes the CWD
// through untouched. The wants below are the CommandLineToArgvW escaping that
// syscall.EscapeArg applies for real — quotes only around an argument holding a
// space, backslashes doubled only where they precede a quote.
func TestHostRelaunchBrokerDryRunPreviewsRealArgv(t *testing.T) {
	base, sep := "/srv", "/"
	if runtime.GOOS == "windows" {
		base, sep = `C:\srv`, `\`
	}
	plain := base + sep + "lane"
	spaced := base + sep + "Program Files" + sep + "lane"
	for _, tc := range []struct {
		name   string
		cwd    string
		handle string
		want   string
	}{
		{
			name:   "plain path keeps single backslashes and stays unquoted",
			cwd:    plain,
			handle: "g1",
			want:   "wt.exe -w new new-tab -d " + plain + " claude --resume g1",
		},
		{
			name:   "path containing a space gets exactly one pair of quotes",
			cwd:    spaced,
			handle: "g1",
			want:   `wt.exe -w new new-tab -d "` + spaced + `" claude --resume g1`,
		},
		{
			name:   "backslash not preceding a quote is never doubled",
			cwd:    plain,
			handle: `a\b`,
			want:   `wt.exe -w new new-tab -d ` + plain + ` claude --resume a\b`,
		},
		{
			name:   "embedded quote is backslash escaped",
			cwd:    plain,
			handle: `g"1`,
			want:   `wt.exe -w new new-tab -d ` + plain + ` claude --resume g\"1`,
		},
		{
			name:   "space and quote together quote the argument and escape the quote",
			cwd:    plain,
			handle: `g 1"x`,
			want:   `wt.exe -w new new-tab -d ` + plain + ` claude --resume "g 1\"x"`,
		},
		{
			name:   "trailing backslash is doubled only against the closing quote",
			cwd:    plain,
			handle: `g 1\`,
			want:   `wt.exe -w new new-tab -d ` + plain + ` claude --resume "g 1\\"`,
		},
		// The two cases below feed POSIX-shaped strings through the same
		// rendering as arguments (only CWD must satisfy filepath.IsAbs), so the
		// exact byte sequence linux CI produces for its own temp-dir CWD is
		// asserted on every platform, not only where the paths are legal.
		{
			name:   "posix path argument renders identically on every platform",
			cwd:    plain,
			handle: "/srv/lane",
			want:   "wt.exe -w new new-tab -d " + plain + " claude --resume /srv/lane",
		},
		{
			name:   "posix path argument with a space is quoted without escaping",
			cwd:    plain,
			handle: "/srv/Program Files/lane",
			want:   `wt.exe -w new new-tab -d ` + plain + ` claude --resume "/srv/Program Files/lane"`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := hostRelaunchBrokerPreview(t, tc.cwd, tc.handle)
			if got != tc.want {
				t.Fatalf("dry-run preview\n got: %s\nwant: %s", got, tc.want)
			}
		})
	}
}

// hostRelaunchBrokerPreview drains a one-request spool in --dry-run and returns
// the single previewed command line without its trailing newline.
func hostRelaunchBrokerPreview(t *testing.T, cwd, handle string) string {
	t.Helper()
	dir := t.TempDir()
	req := hostresurrect.Request{
		Schema:       hostresurrect.Schema,
		EventID:      "evt",
		Session:      "s1",
		CWD:          cwd,
		Command:      []string{"claude", "--resume", handle},
		ResumeHandle: handle,
	}
	if _, err := hostresurrect.Enqueue(dir, req); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	if rc := runHostRelaunchBroker(&out, &stderr, []string{"--dir", dir, "--dry-run"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	return strings.TrimSuffix(out.String(), "\n")
}
