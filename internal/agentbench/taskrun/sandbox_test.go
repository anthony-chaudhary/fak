package taskrun

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestAgentBenchOracleIsolation(t *testing.T) {
	if err := SandboxAvailable(); err != nil {
		t.Skipf("OS sandbox physically unavailable: %v", err)
	}
	if runtime.GOOS != "darwin" {
		t.Fatalf("SandboxAvailable succeeded on %s without a witnessed implementation", runtime.GOOS)
	}

	t.Run("go test permits trial work and denies oracle outside writes and network", func(t *testing.T) {
		base := t.TempDir()
		trial := filepath.Join(base, "trial")
		oracle := filepath.Join(base, "private-oracle")
		outside := filepath.Join(base, "outside-write.txt")
		mustMkdir(t, trial)
		mustMkdir(t, oracle)
		secret := filepath.Join(oracle, "secret.txt")
		mustWrite(t, secret, "ORACLE_SENTINEL_agentbench_12739")
		mustWrite(t, filepath.Join(trial, "go.mod"), "module example.com/agentbench-sandbox\n\ngo 1.26\n")
		probe := fmt.Sprintf(`package sandboxprobe
import (
	"net"
	"os"
	"path/filepath"
	"testing"
)
func TestSandboxProbe(t *testing.T) {
	inside := filepath.Join(%q, "agent-write.txt")
	if err := os.WriteFile(inside, []byte("ok"), 0600); err != nil { t.Fatalf("trial write: %%v", err) }
	if got, err := os.ReadFile(inside); err != nil || string(got) != "ok" { t.Fatalf("trial read: %%q %%v", got, err) }
	if got, err := os.ReadFile(%q); err == nil { t.Fatalf("oracle read escaped: %%q", got) }
	if err := os.WriteFile(%q, []byte("escape"), 0600); err == nil { t.Fatal("outside write escaped") }
	if listener, err := net.Listen("tcp", "127.0.0.1:0"); err == nil { listener.Close(); t.Fatal("network listen escaped") }
	if conn, err := net.Dial("tcp", "127.0.0.1:9"); err == nil { conn.Close(); t.Fatal("network dial escaped") }
}
`, trial, secret, outside)
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
		if !strings.Contains(strings.ToLower(result.Sandbox), "sandbox-exec") {
			t.Errorf("sandbox receipt %q does not identify sandbox-exec", result.Sandbox)
		}
		if result.Duration <= 0 {
			t.Errorf("duration = %v, want observed positive duration", result.Duration)
		}
		if _, err := os.Stat(outside); !os.IsNotExist(err) {
			t.Fatalf("outside write exists after sandbox run: %v", err)
		}
	})

	t.Run("output is bounded", func(t *testing.T) {
		base := t.TempDir()
		trial := filepath.Join(base, "trial")
		oracle := filepath.Join(base, "oracle")
		mustMkdir(t, trial)
		mustMkdir(t, oracle)
		mustWrite(t, filepath.Join(trial, "go.mod"), "module example.com/agentbench-output\n\ngo 1.26\n")
		mustWrite(t, filepath.Join(trial, "output_test.go"), `package outputprobe
import (
	"strings"
	"testing"
)
func TestBoundedOutput(t *testing.T) { t.Fatal(strings.Repeat("x", 2<<20)) }
`)
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		result, err := RunSandboxedTests(ctx, trial, oracle)
		if err != nil {
			t.Fatalf("runner error for ordinary test failure: %v", err)
		}
		if result.ExitCode == 0 {
			t.Fatal("deliberately failing test returned exit 0")
		}
		if len(result.Output) > 1<<20 {
			t.Fatalf("captured output is unbounded: %d bytes", len(result.Output))
		}
	})

	t.Run("rejects overlapping and unresolved roots", func(t *testing.T) {
		base := t.TempDir()
		trial := filepath.Join(base, "trial")
		oracle := filepath.Join(base, "oracle")
		mustMkdir(t, trial)
		mustMkdir(t, oracle)
		insideTrial := filepath.Join(trial, "oracle")
		mustMkdir(t, insideTrial)
		insideOracle := filepath.Join(oracle, "trial")
		mustMkdir(t, insideOracle)
		oracleLink := filepath.Join(base, "oracle-link")
		if err := os.Symlink(oracle, oracleLink); err != nil {
			t.Fatal(err)
		}
		trialLink := filepath.Join(base, "trial-link")
		if err := os.Symlink(trial, trialLink); err != nil {
			t.Fatal(err)
		}

		for _, roots := range []struct{ trial, oracle string }{
			{trial, trial},
			{trial, insideTrial},
			{insideOracle, oracle},
			{trial, oracleLink},
			{trialLink, oracle},
			{filepath.Join(base, "missing-trial"), oracle},
			{trial, filepath.Join(base, "missing-oracle")},
		} {
			if result, err := RunSandboxedTests(context.Background(), roots.trial, roots.oracle); err == nil {
				t.Errorf("roots trial=%q oracle=%q unexpectedly admitted: %+v", roots.trial, roots.oracle, result)
			}
		}
	})

	t.Run("honors cancellation", func(t *testing.T) {
		base := t.TempDir()
		trial := filepath.Join(base, "trial")
		oracle := filepath.Join(base, "oracle")
		mustMkdir(t, trial)
		mustMkdir(t, oracle)
		mustWrite(t, filepath.Join(trial, "go.mod"), "module example.com/agentbench-cancel\n\ngo 1.26\n")
		mustWrite(t, filepath.Join(trial, "cancel_test.go"), "package cancelprobe\nimport \"testing\"\nfunc TestFine(t *testing.T) {}\n")
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		start := time.Now()
		if result, err := RunSandboxedTests(ctx, trial, oracle); err == nil {
			t.Fatalf("cancelled context unexpectedly ran: %+v", result)
		}
		if elapsed := time.Since(start); elapsed > 2*time.Second {
			t.Fatalf("cancelled call returned after %v", elapsed)
		}
	})
}

func mustMkdir(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(path, 0o700); err != nil {
		t.Fatal(err)
	}
}

func mustWrite(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}
