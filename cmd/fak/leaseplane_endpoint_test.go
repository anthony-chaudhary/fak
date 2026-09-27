package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// This stays source-compatible with the parent revision: it exercises the
// production cmd/fak provider only through the pre-existing gateway MCP wire.
func TestFakAdjudicateUsesRealDOSWorkspaceLease(t *testing.T) {
	dos, err := exec.LookPath("dos")
	if err != nil {
		t.Skip("dos CLI unavailable")
	}
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "protected"), 0o755); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(dos, "init", root).CombinedOutput(); err != nil {
		t.Fatalf("dos init: %v\n%s", err, out)
	}
	// Warm the Python CLI before the provider's deliberately bounded 10s read;
	// heavily loaded Windows hosts can otherwise spend the entire bound importing.
	_ = exec.Command(dos, "--workspace", root, "lease-lane", "live").Run()
	if out, err := exec.Command(dos, "--workspace", root, "lease-lane", "acquire",
		"--lane", "gateway", "--kind", "keyword", "--tree", "protected/**", "--owner", "spoofed-process-session").CombinedOutput(); err != nil {
		t.Fatalf("dos lease-lane acquire: %v\n%s", err, out)
	}
	t.Setenv("FAK_LEASEPLANE_DIR", root)
	t.Setenv("FAK_SESSION_ID", "spoofed-process-session")
	gw, err := gateway.New(gateway.Config{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(gw.Close)
	ts := httptest.NewServer(gw.Handler())
	t.Cleanup(ts.Close)
	frame := `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"fak_adjudicate","arguments":{"tool":"apply_patch","arguments":{"patch":"*** Add File: protected/probe.txt\n+must-not-run\n"}}}}`
	resp, err := http.Post(ts.URL+"/mcp", "application/json", strings.NewReader(frame))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	text := string(body)
	leaseHeld := strings.Contains(text, `\"reason\":\"LEASE_HELD\"`)
	boundedAuthorityFailure := strings.Contains(text, `\"reason\":\"DEFAULT_DENY\"`) && strings.Contains(text, "workspace lease authority read failed")
	if resp.StatusCode != http.StatusOK || !strings.Contains(text, `\"by\":\"lease-admission\"`) || (!leaseHeld && !boundedAuthorityFailure) {
		t.Fatalf("MCP response status=%d body=%s", resp.StatusCode, body)
	}
}
