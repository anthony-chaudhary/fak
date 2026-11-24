package main

import (
	"bytes"
	"strings"
	"testing"
)

// With no router and no local gateway, every platform must get a runnable
// public next step, not only macOS.
func TestClaudeUnreachableNextStepNamesFakServeOnEveryOS(t *testing.T) {
	for _, goos := range []string{"linux", "windows", "darwin"} {
		var buf bytes.Buffer
		printClaudeUnreachableNextStep(&buf, "http://127.0.0.1:8080", goos)
		out := buf.String()
		for _, want := range []string{
			"fak serve --addr 127.0.0.1:8080 --base-url <upstream>/v1 --model <id>",
			"fak claude --gateway-url",
		} {
			if !strings.Contains(out, want) {
				t.Errorf("%s: next step missing %q:\n%s", goos, want, out)
			}
		}
		if hasMac := strings.Contains(out, "--gguf"); hasMac != (goos == "darwin") {
			t.Errorf("%s: Mac Metal hint present=%v:\n%s", goos, hasMac, out)
		}
	}
}

func TestClaudeUnreachableNextStepSilentForRemoteGateway(t *testing.T) {
	var buf bytes.Buffer
	printClaudeUnreachableNextStep(&buf, "http://gpu-box.example:8080", "linux")
	if buf.Len() != 0 {
		t.Fatalf("remote gateway must not suggest a local serve command:\n%s", buf.String())
	}
}
