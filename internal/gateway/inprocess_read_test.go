package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/vdso"
)

func TestGateway_PromoteShellReadToInProcess(t *testing.T) {
	srv := newTestServer(t)
	defer srv.Close()

	dir := t.TempDir()
	p := filepath.Join(dir, "gateway_read_test.txt")
	content := "line 1\nline 2\nline 3\nline 4\nline 5\n"
	if err := os.WriteFile(p, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	posixPath := filepath.ToSlash(p)

	t.Run("cat promotion via PromoteShellReadToInProcess", func(t *testing.T) {
		args := fmt.Sprintf(`{"command":"cat %s"}`, posixPath)
		env, wv, ok := srv.PromoteShellReadToInProcess(context.Background(), "Bash", args, "trace-read-1")
		if !ok || env == nil {
			t.Fatalf("expected promotion to succeed, ok=%v", ok)
		}
		if wv.Kind != "ALLOW" {
			t.Errorf("expected ALLOW verdict, got %s", wv.Kind)
		}
		if env.Meta["served_by"] != "in_process_read" {
			t.Errorf("expected meta served_by=in_process_read, got %q", env.Meta["served_by"])
		}

		var bashRes vdso.ShellReadResult
		if err := json.Unmarshal([]byte(env.Content), &bashRes); err != nil {
			t.Fatalf("failed to unmarshal content: %v", err)
		}
		if bashRes.ExitCode != 0 || bashRes.Stdout != content {
			t.Fatalf("unexpected content: %+v", bashRes)
		}
	})

	t.Run("cat promotion via s.syscall", func(t *testing.T) {
		args := fmt.Sprintf(`{"command":"cat %s"}`, posixPath)
		wv, env, err := srv.syscall(context.Background(), "Bash", args, true, "", "trace-read-2")
		if err != nil {
			t.Fatalf("syscall failed: %v", err)
		}
		if wv.Kind != "ALLOW" {
			t.Errorf("expected ALLOW verdict, got %s", wv.Kind)
		}
		if env.Meta["served_by"] != "in_process_read" {
			t.Errorf("expected meta served_by=in_process_read, got %q", env.Meta["served_by"])
		}

		var bashRes vdso.ShellReadResult
		if err := json.Unmarshal([]byte(env.Content), &bashRes); err != nil {
			t.Fatalf("failed to unmarshal content: %v", err)
		}
		if bashRes.ExitCode != 0 || bashRes.Stdout != content {
			t.Fatalf("unexpected content: %+v", bashRes)
		}
	})

	t.Run("head promotion via s.syscall", func(t *testing.T) {
		args := fmt.Sprintf(`{"command":"head -n 2 %s"}`, posixPath)
		wv, env, err := srv.syscall(context.Background(), "Bash", args, true, "", "trace-read-3")
		if err != nil {
			t.Fatalf("syscall failed: %v", err)
		}
		if wv.Kind != "ALLOW" {
			t.Errorf("expected ALLOW verdict, got %s", wv.Kind)
		}

		var bashRes vdso.ShellReadResult
		if err := json.Unmarshal([]byte(env.Content), &bashRes); err != nil {
			t.Fatalf("failed to unmarshal content: %v", err)
		}
		want := "line 1\nline 2\n"
		if bashRes.ExitCode != 0 || bashRes.Stdout != want {
			t.Fatalf("head content mismatch: got %q, want %q", bashRes.Stdout, want)
		}
	})

	t.Run("tail promotion via s.syscall", func(t *testing.T) {
		args := fmt.Sprintf(`{"command":"tail -n 2 %s"}`, posixPath)
		wv, env, err := srv.syscall(context.Background(), "Bash", args, true, "", "trace-read-4")
		if err != nil {
			t.Fatalf("syscall failed: %v", err)
		}
		if wv.Kind != "ALLOW" {
			t.Errorf("expected ALLOW verdict, got %s", wv.Kind)
		}

		var bashRes vdso.ShellReadResult
		if err := json.Unmarshal([]byte(env.Content), &bashRes); err != nil {
			t.Fatalf("failed to unmarshal content: %v", err)
		}
		want := "line 4\nline 5\n"
		if bashRes.ExitCode != 0 || bashRes.Stdout != want {
			t.Fatalf("tail content mismatch: got %q, want %q", bashRes.Stdout, want)
		}
	})

	t.Run("mutating command does not promote", func(t *testing.T) {
		args := `{"command":"rm -rf /tmp/foo"}`
		_, _, ok := srv.PromoteShellReadToInProcess(context.Background(), "Bash", args, "trace-rm")
		if ok {
			t.Fatalf("mutating command must not be promoted")
		}
	})

	t.Run("exec_command with Get-Content", func(t *testing.T) {
		args := fmt.Sprintf(`{"command":"Get-Content -Path %s"}`, posixPath)
		env, wv, ok := srv.PromoteShellReadToInProcess(context.Background(), "exec_command", args, "trace-exec-1")
		if !ok || env == nil {
			t.Fatalf("expected promotion to succeed, ok=%v", ok)
		}
		if wv.Kind != "ALLOW" {
			t.Errorf("expected ALLOW verdict, got %s", wv.Kind)
		}
		if env.Meta["served_by"] != "in_process_read" {
			t.Errorf("expected meta served_by=in_process_read, got %q", env.Meta["served_by"])
		}

		var res vdso.ShellReadResult
		if err := json.Unmarshal([]byte(env.Content), &res); err != nil {
			t.Fatalf("failed to unmarshal content: %v", err)
		}
		if res.ExitCode != 0 || res.Stdout != content {
			t.Fatalf("unexpected content: %+v", res)
		}
	})

	t.Run("exec_command with cat via syscall", func(t *testing.T) {
		args := fmt.Sprintf(`{"command":"cat %s"}`, posixPath)
		wv, env, err := srv.syscall(context.Background(), "exec_command", args, true, "", "trace-exec-2")
		if err != nil {
			t.Fatalf("syscall failed: %v", err)
		}
		if wv.Kind != "ALLOW" {
			t.Errorf("expected ALLOW verdict, got %s", wv.Kind)
		}
		if env.Meta["served_by"] != "in_process_read" {
			t.Errorf("expected meta served_by=in_process_read, got %q", env.Meta["served_by"])
		}

		var res vdso.ShellReadResult
		if err := json.Unmarshal([]byte(env.Content), &res); err != nil {
			t.Fatalf("failed to unmarshal content: %v", err)
		}
		if res.ExitCode != 0 || res.Stdout != content {
			t.Fatalf("unexpected content: %+v", res)
		}
	})

	t.Run("functions.exec_command with Get-Content via syscall", func(t *testing.T) {
		args := fmt.Sprintf(`{"cmd":"Get-Content -Head 2 %s"}`, posixPath)
		wv, env, err := srv.syscall(context.Background(), "functions.exec_command", args, true, "", "trace-exec-3")
		if err != nil {
			t.Fatalf("syscall failed: %v", err)
		}
		if wv.Kind != "ALLOW" {
			t.Errorf("expected ALLOW verdict, got %s", wv.Kind)
		}
		if env.Meta["served_by"] != "in_process_read" {
			t.Errorf("expected meta served_by=in_process_read, got %q", env.Meta["served_by"])
		}

		var res vdso.ShellReadResult
		if err := json.Unmarshal([]byte(env.Content), &res); err != nil {
			t.Fatalf("failed to unmarshal content: %v", err)
		}
		want := "line 1\nline 2\n"
		if res.ExitCode != 0 || res.Stdout != want {
			t.Fatalf("unexpected content: %+v, want %q", res, want)
		}
	})
}

func TestGateway_PromoteShellRead_CompoundPipeline(t *testing.T) {
	srv := newTestServer(t)
	defer srv.Close()

	dir := t.TempDir()
	p1 := filepath.Join(dir, "file1.txt")
	p2 := filepath.Join(dir, "file2.txt")
	if err := os.WriteFile(p1, []byte("content from file1\n"), 0644); err != nil {
		t.Fatalf("failed to write file1: %v", err)
	}
	if err := os.WriteFile(p2, []byte("content from file2\n"), 0644); err != nil {
		t.Fatalf("failed to write file2: %v", err)
	}

	cmd := fmt.Sprintf("Get-Content %s; Get-Content %s", filepath.ToSlash(p1), filepath.ToSlash(p2))
	args := fmt.Sprintf(`{"command":%q}`, cmd)

	env, wv, ok := srv.PromoteShellReadToInProcess(context.Background(), "exec_command", args, "trace-compound-pipeline")
	if !ok || env == nil {
		t.Fatalf("expected compound pipeline promotion to succeed, ok=%v", ok)
	}
	if wv.Kind != "ALLOW" {
		t.Errorf("expected ALLOW verdict, got %s", wv.Kind)
	}
	if env.Meta["served_by"] != "in_process_read" {
		t.Errorf("expected meta served_by=in_process_read, got %q", env.Meta["served_by"])
	}
	if env.Meta["in_process_op"] != "compound" {
		t.Errorf("expected meta in_process_op=compound, got %q", env.Meta["in_process_op"])
	}

	var res vdso.ShellReadResult
	if err := json.Unmarshal([]byte(env.Content), &res); err != nil {
		t.Fatalf("failed to unmarshal content: %v", err)
	}
	if res.ExitCode != 0 {
		t.Fatalf("expected ExitCode 0, got %d, stderr: %s", res.ExitCode, res.Stderr)
	}
	want := "content from file1\ncontent from file2\n"
	if res.Stdout != want {
		t.Fatalf("stdout mismatch: got %q, want %q", res.Stdout, want)
	}

	t.Run("syscall promotion with workdir", func(t *testing.T) {
		cmdWorkdir := "Get-Content file1.txt; Get-Content file2.txt"
		argsWorkdir := fmt.Sprintf(`{"cmd":%q,"workdir":%q}`, cmdWorkdir, filepath.ToSlash(dir))
		wvCall, envCall, err := srv.syscall(context.Background(), "exec_command", argsWorkdir, true, "", "trace-compound-workdir")
		if err != nil {
			t.Fatalf("syscall failed: %v", err)
		}
		if wvCall.Kind != "ALLOW" {
			t.Errorf("expected ALLOW verdict, got %s", wvCall.Kind)
		}
		if envCall.Meta["served_by"] != "in_process_read" {
			t.Errorf("expected meta served_by=in_process_read, got %q", envCall.Meta["served_by"])
		}
		var resWorkdir vdso.ShellReadResult
		if err := json.Unmarshal([]byte(envCall.Content), &resWorkdir); err != nil {
			t.Fatalf("failed to unmarshal content: %v", err)
		}
		if resWorkdir.ExitCode != 0 || resWorkdir.Stdout != want {
			t.Fatalf("unexpected result: exitCode=%d, stdout=%q, stderr=%s", resWorkdir.ExitCode, resWorkdir.Stdout, resWorkdir.Stderr)
		}
	})
}

func BenchmarkGateway_PromoteShellRead_ExecCommand(b *testing.B) {
	srv := newAdjudicateBenchmarkServer(b)
	dir := b.TempDir()
	p := filepath.Join(dir, "bench_exec.txt")
	_ = os.WriteFile(p, []byte("line 1\nline 2\nline 3\n"), 0644)
	args := fmt.Sprintf(`{"command":"Get-Content %s"}`, filepath.ToSlash(p))
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		env, _, ok := srv.PromoteShellReadToInProcess(ctx, "exec_command", args, "bench-trace")
		if !ok || env == nil {
			b.Fatalf("PromoteShellReadToInProcess failed")
		}
	}
}
