package systemservice

import (
	"strings"
	"testing"
)

func TestRenderLaunchAgentOwnsControlPlaneWithoutTerminal(t *testing.T) {
	p, e := RenderLaunchAgent(LaunchdConfig{Executable: "/Users/a & b/bin/fak", StateDir: "/Users/a/state", StdoutPath: "/Users/a/log/out", StderrPath: "/Users/a/log/err"})
	if e != nil {
		t.Fatal(e)
	}
	for _, w := range []string{"<string>com.fak.guard-control</string>", "<key>KeepAlive</key><true/>", "<key>RunAtLoad</key><true/>", "<key>ProcessType</key><string>Background</string>", "service</string><string>run", "/Users/a &amp; b/bin/fak"} {
		if !strings.Contains(p, w) {
			t.Fatalf("missing %q: %s", w, p)
		}
	}
	if strings.Contains(p, "Terminal.app") {
		t.Fatal("UI dependency")
	}
}
func TestRenderLaunchAgentRejectsNewline(t *testing.T) {
	if _, e := RenderLaunchAgent(LaunchdConfig{Executable: "/x\ny", StateDir: "/s", StdoutPath: "/o", StderrPath: "/e"}); e == nil {
		t.Fatal("accepted injection")
	}
}
