//go:build linux

package taskrun

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestLinuxSandboxIsolationAndAccess(t *testing.T) {
	requireLinuxSandbox(t)
	base := t.TempDir()
	trial := filepath.Join(base, "trial")
	oracle := filepath.Join(base, "oracle")
	outside := filepath.Join(base, "outside.txt")
	mustMkdir(t, trial)
	mustMkdir(t, oracle)
	secret := filepath.Join(oracle, "secret.txt")
	mustWrite(t, secret, "ORACLE_SENTINEL_linux_agentbench")
	mustWrite(t, filepath.Join(trial, "go.mod"), "module example.com/linux-sandbox\n\ngo 1.26\n")

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	probe := fmt.Sprintf(`package sandboxprobe
import (
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"
)
func TestIsolation(t *testing.T) {
	inside := filepath.Join(%q, "agent-write.txt")
	if err := os.WriteFile(inside, []byte("ok"), 0600); err != nil { t.Fatalf("trial write: %%v", err) }
	if got, err := os.ReadFile(inside); err != nil || string(got) != "ok" { t.Fatalf("trial read: %%q %%v", got, err) }
	if got, err := os.ReadFile(%q); err == nil { t.Fatalf("oracle read escaped: %%q", got) }
	if err := os.WriteFile(%q, []byte("changed"), 0600); err == nil { t.Fatal("oracle write escaped") }
	if err := os.WriteFile(%q, []byte("escape"), 0600); err == nil { t.Fatal("outside write escaped") }
	if conn, err := net.DialTimeout("tcp", %q, 250*time.Millisecond); err == nil { conn.Close(); t.Fatal("host network escaped") }
}
`, "/workspace", secret, secret, outside, listener.Addr().String())
	mustWrite(t, filepath.Join(trial, "sandbox_test.go"), probe)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	result, err := RunSandboxedTests(ctx, trial, oracle)
	if err != nil {
		t.Fatalf("RunSandboxedTests: %v\n%s", err, result.Output)
	}
	if result.ExitCode != 0 {
		t.Fatalf("sandboxed go test exit=%d\n%s", result.ExitCode, result.Output)
	}
	if result.Sandbox != "linux-bwrap-userns" {
		t.Fatalf("sandbox receipt = %q, want linux-bwrap-userns", result.Sandbox)
	}
	if got, err := os.ReadFile(secret); err != nil || string(got) != "ORACLE_SENTINEL_linux_agentbench" {
		t.Fatalf("oracle changed: %q, %v", got, err)
	}
	if _, err := os.Stat(outside); !os.IsNotExist(err) {
		t.Fatalf("outside write exists after sandbox run: %v", err)
	}

	mustWrite(t, filepath.Join(trial, "sandbox_test.go"), fmt.Sprintf(`package sandboxprobe
import (
	"os"
	"testing"
)
func TestReadOnlyTrial(t *testing.T) {
	if err := os.WriteFile(%q, []byte("escape"), 0600); err == nil { t.Fatal("read-only trial write escaped") }
}
`, "/workspace/readonly-escape.txt"))
	result, err = runSandboxedTests(ctx, trial, oracle, false)
	if err != nil {
		t.Fatalf("read-only RunSandboxedTests: %v\n%s", err, result.Output)
	}
	if result.ExitCode != 0 {
		t.Fatalf("read-only sandboxed go test exit=%d\n%s", result.ExitCode, result.Output)
	}
	if _, err := os.Stat(filepath.Join(trial, "readonly-escape.txt")); !os.IsNotExist(err) {
		t.Fatalf("read-only trial was mutated: %v", err)
	}
}

func TestLinuxSandboxRedThenGreenFixture(t *testing.T) {
	requireLinuxSandbox(t)
	base := t.TempDir()
	trial := filepath.Join(base, "trial")
	oracle := filepath.Join(base, "oracle")
	mustMkdir(t, trial)
	mustMkdir(t, oracle)
	mustWrite(t, filepath.Join(trial, "go.mod"), "module example.com/linux-red-green\n\ngo 1.26\n")
	testPath := filepath.Join(trial, "fixture_test.go")
	mustWrite(t, testPath, "package fixture\nimport \"testing\"\nfunc TestFixture(t *testing.T) { t.Fatal(\"red witness\") }\n")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	red, err := RunSandboxedTests(ctx, trial, oracle)
	if err != nil {
		t.Fatalf("red runner error: %v", err)
	}
	if red.ExitCode == 0 || !strings.Contains(red.Output, "red witness") {
		t.Fatalf("red fixture was not observed: exit=%d\n%s", red.ExitCode, red.Output)
	}

	mustWrite(t, testPath, "package fixture\nimport \"testing\"\nfunc TestFixture(t *testing.T) {}\n")
	green, err := RunSandboxedTests(ctx, trial, oracle)
	if err != nil {
		t.Fatalf("green runner error: %v\n%s", err, green.Output)
	}
	if green.ExitCode != 0 {
		t.Fatalf("green fixture exit=%d\n%s", green.ExitCode, green.Output)
	}
}

func TestLinuxSandboxCancellationKillsDescendant(t *testing.T) {
	requireLinuxSandbox(t)
	base := t.TempDir()
	trial := filepath.Join(base, "trial")
	oracle := filepath.Join(base, "oracle")
	mustMkdir(t, trial)
	mustMkdir(t, oracle)
	mustWrite(t, filepath.Join(trial, "go.mod"), "module example.com/linux-cancel\n\ngo 1.26\n")
	ready := filepath.Join(trial, "descendant-ready")
	marker := filepath.Join(trial, "descendant-survived")
	mustWrite(t, filepath.Join(trial, "cancel_test.go"), fmt.Sprintf(`package cancelprobe
import (
	"os"
	"os/exec"
	"testing"
	"time"
)
func TestDescendantHelper(t *testing.T) {
	if os.Getenv("AGENTBENCH_DESCENDANT") != "1" { return }
	time.Sleep(1200 * time.Millisecond)
	if err := os.WriteFile(%q, []byte("escaped"), 0600); err != nil { t.Fatal(err) }
}
func TestParent(t *testing.T) {
	cmd := exec.Command(os.Args[0], "-test.run=^TestDescendantHelper$")
	cmd.Env = append(os.Environ(), "AGENTBENCH_DESCENDANT=1")
	if err := cmd.Start(); err != nil { t.Fatal(err) }
	if err := os.WriteFile(%q, []byte("ready"), 0600); err != nil { t.Fatal(err) }
	select {}
}
`, "/workspace/descendant-survived", "/workspace/descendant-ready"))

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := RunSandboxedTests(ctx, trial, oracle)
		done <- err
	}()
	deadline := time.Now().Add(15 * time.Second)
	for {
		if _, err := os.Stat(ready); err == nil {
			break
		}
		if time.Now().After(deadline) {
			cancel()
			t.Fatal("sandboxed fixture did not start its descendant")
		}
		time.Sleep(20 * time.Millisecond)
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("RunSandboxedTests cancellation error = %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("RunSandboxedTests did not return after cancellation")
	}
	time.Sleep(1500 * time.Millisecond)
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("descendant survived sandbox cancellation: %v", err)
	}
}

func TestLinuxSandboxUnavailableBackendFailsClosed(t *testing.T) {
	original := bubblewrapLookPath
	bubblewrapLookPath = func(string) (string, error) { return "", exec.ErrNotFound }
	t.Cleanup(func() { bubblewrapLookPath = original })

	err := SandboxAvailable()
	var unavailable *SandboxUnavailableError
	if !errors.As(err, &unavailable) {
		t.Fatalf("SandboxAvailable error = %T %v, want *SandboxUnavailableError", err, err)
	}
}

func requireLinuxSandbox(t *testing.T) {
	t.Helper()
	if err := SandboxAvailable(); err != nil {
		t.Skipf("Linux sandbox physically unavailable: %v", err)
	}
}
