package systemservice

import (
	"strings"
	"testing"
)

func TestRenderLaunchDaemonIsSystemOwnedWithoutTerminal(t *testing.T) {
	p, e := RenderLaunchDaemon(LaunchdConfig{Executable: "/usr/local/libexec/fak", StateDir: "/var/db/fak", StdoutPath: "/var/log/fak/out", StderrPath: "/var/log/fak/err", UserName: "_fakguard"})
	if e != nil {
		t.Fatal(e)
	}
	for _, w := range []string{
		"<string>com.fak.guard-control</string>", "<key>KeepAlive</key><true/>", "<key>RunAtLoad</key><true/>",
		"<key>ProcessType</key><string>Background</string>", "<key>UserName</key><string>_fakguard</string>",
		"<string>launchd-system</string>", "service</string><string>run",
	} {
		if !strings.Contains(p, w) {
			t.Fatalf("missing %q: %s", w, p)
		}
	}
	for _, forbidden := range []string{"Terminal.app", "Aqua", "LimitLoadToSessionType"} {
		if strings.Contains(p, forbidden) {
			t.Fatalf("system daemon has GUI dependency %q", forbidden)
		}
	}
}

func TestRenderLaunchDaemonEscapesAndRejectsUnsafeInput(t *testing.T) {
	p, e := RenderLaunchDaemon(LaunchdConfig{Executable: "/opt/a & b/fak", StateDir: "/var/db/fak", StdoutPath: "/var/log/fak/out", StderrPath: "/var/log/fak/err", UserName: "_fakguard"})
	if e != nil || !strings.Contains(p, "/opt/a &amp; b/fak") {
		t.Fatalf("XML escaping failed: %v %s", e, p)
	}
	if _, e := RenderLaunchDaemon(LaunchdConfig{Executable: "/x\ny", StateDir: "/s", StdoutPath: "/o", StderrPath: "/e", UserName: "u"}); e == nil {
		t.Fatal("accepted injection")
	}
	if _, e := RenderLaunchDaemon(LaunchdConfig{Executable: "/x", StateDir: "/s", StdoutPath: "/o", StderrPath: "/e"}); e == nil {
		t.Fatal("accepted missing principal")
	}
}
