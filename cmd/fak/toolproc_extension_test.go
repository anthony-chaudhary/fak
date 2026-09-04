package main

import (
	"bytes"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestToolprocExtensionSubcommand(t *testing.T) {
	pluginName := "fault-plugin"
	if runtime.GOOS == "windows" {
		pluginName += ".exe"
	}
	binary := filepath.Join(t.TempDir(), pluginName)
	cmd := buildToolprocExtensionPlugin(t, binary)
	if cmd == "" {
		t.Skip("skipping toolproc extension test: cannot build fault-plugin")
	}

	// 1. Unknown / usage error exits 2
	var errOut bytes.Buffer
	rc := runToolprocExtension(&bytes.Buffer{}, &errOut, []string{})
	if rc != 2 {
		t.Fatalf("expected exit code 2 on missing --cmd, got %d", rc)
	}

	// 2. Successful call via healthy mode
	var stdout, stderr bytes.Buffer
	rc = runToolprocExtension(&stdout, &stderr, []string{
		"--cmd", binary + " --mode healthy",
		"--name", "healthy-ext",
		"--call", "ping",
		"--json",
	})
	if rc != 0 {
		t.Fatalf("expected exit code 0 for healthy extension, got %d, stderr: %s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), `"result": "ok:ping"`) {
		t.Fatalf("unexpected output: %s", stdout.String())
	}
	if !strings.Contains(stdout.String(), `"running": true`) {
		t.Fatalf("expected running: true, got: %s", stdout.String())
	}
}

func buildToolprocExtensionPlugin(t *testing.T, target string) string {
	t.Helper()
	root := filepath.Clean(filepath.Join("..", ".."))
	buildCmd := execCommand("go", "build", "-o", target, "./examples/mcp/fault-plugin")
	buildCmd.Dir = root
	out, err := buildCmd.CombinedOutput()
	if err != nil {
		t.Logf("build fault plugin failed: %v\n%s", err, string(out))
		return ""
	}
	return target
}
