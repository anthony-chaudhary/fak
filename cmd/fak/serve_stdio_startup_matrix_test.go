package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// TestServeStdioStartupMatrix is the process-level witness for #4656. It builds
// and launches the real fak binary for every pre-handshake registry state and
// captures both sides of the MCP stdio boundary. Package tests around FileStore
// and gateway.dispatchRPC cannot catch startup exits, stdout contamination, or a
// bootstrap hang before ServeStdio reads initialize.
func TestServeStdioStartupMatrix(t *testing.T) {
	if testing.Short() {
		t.Skip("builds and launches the real fak binary")
	}

	root := repoRootForDoctorTest(t)
	suffix := ""
	if runtime.GOOS == "windows" {
		suffix = ".exe"
	}
	exe := filepath.Join(t.TempDir(), "fak-stdio-startup"+suffix)
	build := exec.Command("go", "build", "-o", exe, "./cmd/fak")
	build.Dir = root
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build real fak binary: %v\n%s", err, out)
	}
	policyPath := filepath.Join(root, "examples", "dev-agent-policy.json")

	valid := []byte("{\n  \"version\": \"fak.session-descriptors.v1\",\n  \"descriptors\": []\n}\n")
	cases := []struct {
		name       string
		registry   []byte
		create     bool
		quarantine bool
		strictOld  bool
	}{
		{name: "absent"},
		{name: "empty", registry: []byte{}, create: true},
		{name: "valid", registry: valid, create: true},
		{name: "malformed-json", registry: []byte(`{"version":`), create: true, quarantine: true, strictOld: true},
		{name: "nul-filled", registry: bytes.Repeat([]byte{0}, 128), create: true, quarantine: true, strictOld: true},
		{name: "unsupported-version", registry: []byte(`{"version":"fak.session-descriptors.v999","descriptors":[]}`), create: true, quarantine: true, strictOld: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			state := t.TempDir()
			registry := filepath.Join(state, "session-registry.json")
			if tc.create {
				if err := os.WriteFile(registry, tc.registry, 0o600); err != nil {
					t.Fatal(err)
				}
			}

			// Failing control: this is the pre-#4647 startup behavior -- treating a
			// corrupt FileStore.List result as fatal instead of quarantining it.
			if tc.strictOld {
				if _, err := session.NewFileStore(registry).List(); err == nil {
					t.Fatal("pre-#4647 strict registry control unexpectedly accepted corrupt fixture")
				}
			}

			fakeHome := filepath.Join(state, "home")
			if err := os.MkdirAll(fakeHome, 0o755); err != nil {
				t.Fatal(err)
			}
			// A sentinel at the default user-state location proves the explicit
			// FAK_SESSION_REGISTRY isolation is honored by the real child.
			sentinel := filepath.Join(fakeHome, ".config", "fak", "session-registry.json")
			if runtime.GOOS == "windows" {
				sentinel = filepath.Join(fakeHome, "AppData", "Roaming", "fak", "session-registry.json")
			}
			if err := os.MkdirAll(filepath.Dir(sentinel), 0o755); err != nil {
				t.Fatal(err)
			}
			sentinelBytes := []byte("operator-state-must-remain-untouched\n")
			if err := os.WriteFile(sentinel, sentinelBytes, 0o600); err != nil {
				t.Fatal(err)
			}
			before := sha256.Sum256(sentinelBytes)

			result := runStdioStartupProcess(t, exe, root, policyPath, registry, fakeHome, 12*time.Second)
			if result.timedOut {
				t.Fatalf("initialize exceeded bounded timeout\nstderr:\n%s", result.stderr)
			}
			if result.err != nil {
				t.Fatalf("serve process failed: %v\nstdout:\n%s\nstderr:\n%s", result.err, result.stdout, result.stderr)
			}
			assertSingleInitializeFrame(t, result.stdout)

			afterBytes, err := os.ReadFile(sentinel)
			if err != nil {
				t.Fatalf("read user-state sentinel: %v", err)
			}
			after := sha256.Sum256(afterBytes)
			if after != before {
				t.Fatalf("serve touched default user state %s", sentinel)
			}

			matches, err := filepath.Glob(registry + ".corrupt-*")
			if err != nil {
				t.Fatal(err)
			}
			if tc.quarantine {
				if len(matches) != 1 {
					t.Fatalf("quarantine artifacts=%v, want exactly one; stderr=%s", matches, result.stderr)
				}
				got, err := os.ReadFile(matches[0])
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(got, tc.registry) {
					t.Fatalf("quarantine did not preserve corrupt evidence: got %q want %q", got, tc.registry)
				}
				if !strings.Contains(result.stderr, "corrupt session registry quarantined") {
					t.Fatalf("missing stderr-only quarantine diagnostic:\n%s", result.stderr)
				}
			} else if len(matches) != 0 {
				t.Fatalf("healthy registry unexpectedly quarantined: %v", matches)
			}
		})
	}

	t.Run("missing-policy-is-fatal-with-clean-protocol", func(t *testing.T) {
		state := t.TempDir()
		missing := filepath.Join(state, "missing-policy.json")
		result := runStdioStartupProcess(t, exe, root, missing, filepath.Join(state, "registry.json"), filepath.Join(state, "home"), 12*time.Second)
		if result.timedOut {
			t.Fatal("missing policy hung instead of failing startup")
		}
		if result.err == nil {
			t.Fatalf("missing policy unexpectedly succeeded: stdout=%q stderr=%q", result.stdout, result.stderr)
		}
		if strings.TrimSpace(result.stdout) != "" {
			t.Fatalf("fatal config contaminated MCP stdout: %q", result.stdout)
		}
		// "fak:" is the CLI's stable fatal-diagnostic class; retain the path and
		// underlying cause on stderr without emitting a JSON-RPC fragment.
		if !strings.HasPrefix(strings.TrimSpace(result.stderr), "fak:") || !strings.Contains(result.stderr, missing) {
			t.Fatalf("missing stable policy diagnostic: %q", result.stderr)
		}
	})
}

type stdioStartupResult struct {
	stdout   string
	stderr   string
	err      error
	timedOut bool
}

func runStdioStartupProcess(t *testing.T, exe, root, policyPath, registry, home string, timeout time.Duration) stdioStartupResult {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe, "serve", "--stdio", "--policy", policyPath)
	cmd.Dir = root
	cmd.Env = append(os.Environ(),
		"FAK_SESSION_REGISTRY="+registry,
		"HOME="+home,
		"USERPROFILE="+home,
		"XDG_CONFIG_HOME="+filepath.Join(home, ".config"),
	)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	initialize := `{"jsonrpc":"2.0","id":4656,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"startup-matrix","version":"1"}}}` + "\n"
	_, _ = fmt.Fprint(stdin, initialize)
	_ = stdin.Close()
	err = cmd.Wait()
	return stdioStartupResult{stdout: stdout.String(), stderr: stderr.String(), err: err, timedOut: ctx.Err() == context.DeadlineExceeded}
}

func assertSingleInitializeFrame(t *testing.T, stdout string) {
	t.Helper()
	lines := strings.Split(strings.TrimSpace(stdout), "\n")
	if len(lines) != 1 {
		t.Fatalf("MCP stdout contained %d frames/banners, want one initialize frame:\n%s", len(lines), stdout)
	}
	var response struct {
		JSONRPC string `json:"jsonrpc"`
		ID      int    `json:"id"`
		Result  struct {
			ProtocolVersion string `json:"protocolVersion"`
			ServerInfo      struct {
				Name    string `json:"name"`
				Version string `json:"version"`
			} `json:"serverInfo"`
		} `json:"result"`
		Error json.RawMessage `json:"error"`
	}
	if err := json.Unmarshal([]byte(lines[0]), &response); err != nil {
		t.Fatalf("stdout is not one valid JSON-RPC document: %v\n%s", err, stdout)
	}
	if len(response.Error) != 0 && string(response.Error) != "null" {
		t.Fatalf("initialize returned error: %s", response.Error)
	}
	if response.JSONRPC != "2.0" || response.ID != 4656 || response.Result.ProtocolVersion != "2025-03-26" {
		t.Fatalf("wrong initialize envelope: %+v", response)
	}
	if response.Result.ServerInfo.Name != "fak-gateway" || response.Result.ServerInfo.Version == "" {
		t.Fatalf("wrong server identity: %+v", response.Result.ServerInfo)
	}
}
