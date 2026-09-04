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
}
