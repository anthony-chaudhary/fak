package main

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestGuardMCPLaunchCredentialsAreDistinctAndRemovedOnExit(t *testing.T) {
	root := t.TempDir()
	helperSource := filepath.Join(root, "helper.go")
	helperBinary := filepath.Join(root, "fak-agent")
	if runtime.GOOS == "windows" {
		helperBinary += ".exe"
	}
	const helper = `package main
import ("encoding/json"; "os")
func main() {
	p := os.Getenv("FAK_MCP_CONFIG")
	b, err := os.ReadFile(p); if err != nil { panic(err) }
	out, _ := json.Marshal(map[string]string{"path": p, "body": string(b)})
	if err := os.WriteFile(os.Getenv("FAK_MCP_CAPTURE"), out, 0600); err != nil { panic(err) }
}`
	if err := os.WriteFile(helperSource, []byte(helper), 0o600); err != nil {
		t.Fatal(err)
	}
	build := exec.Command("go", "build", "-o", helperBinary, helperSource)
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build fak-agent witness: %v\n%s", err, out)
	}

	capture := filepath.Join(root, "capture.json")
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	guard := exec.CommandContext(ctx, os.Args[0], "guard")
	guard.Env = append(os.Environ(),
		guardE2EHelperEnv+"=--quiet --provider anthropic --api-key-env FAK_TEST_MCP_UPSTREAM --require-key-env FAK_TEST_MCP_GATEWAY --audit off -- "+helperBinary,
		"FAK_TEST_MCP_UPSTREAM=upstream-test-only",
		"FAK_TEST_MCP_GATEWAY=gateway-test-only",
		"FAK_MCP_CAPTURE="+capture,
		"FAK_FLEET_BUS="+filepath.Join(root, "fleet-bus"),
		"FAK_SESSION_REGISTRY="+filepath.Join(root, "sessions.jsonl"),
		"FAK_HEADLESS=1", "TMP="+root, "TEMP="+root,
	)
	if out, err := guard.CombinedOutput(); err != nil {
		t.Fatalf("guard process: %v\n%s", err, out)
	}
	if ctx.Err() != nil {
		t.Fatal(ctx.Err())
	}

	data, err := os.ReadFile(capture)
	if err != nil {
		t.Fatal(err)
	}
	var observed map[string]string
	if err := json.Unmarshal(data, &observed); err != nil {
		t.Fatal(err)
	}
	var config map[string]any
	if err := json.Unmarshal([]byte(observed["body"]), &config); err != nil {
		t.Fatalf("child captured invalid MCP JSON: %v", err)
	}
	servers, _ := config["mcpServers"].(map[string]any)
	fak, _ := servers["fak"].(map[string]any)
	headers, _ := fak["headers"].(map[string]any)
	launch, _ := headers["X-Fak-MCP-Session"].(string)
	gateway, _ := headers["Authorization"].(string)
	if gateway != "Bearer gateway-test-only" {
		t.Errorf("gateway Authorization = %q, want configured bearer", gateway)
	}
	if !strings.HasPrefix(launch, "Bearer ") || strings.TrimPrefix(launch, "Bearer ") == "" {
		t.Errorf("launch provenance header = %q, want non-empty bearer", launch)
	}
	if launch == gateway {
		t.Errorf("launch provenance bearer reused gateway Authorization")
	}
	if _, err := os.Stat(observed["path"]); !os.IsNotExist(err) {
		t.Errorf("plaintext MCP credential config survived normal exit: path=%q err=%v", observed["path"], err)
	}
}
