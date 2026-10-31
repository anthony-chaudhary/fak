package leaserefcli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/testgit"
)

func TestMain(m *testing.M) { os.Exit(testgit.RunGitIsolated(m)) }

func fixtureGit(t *testing.T, dir string, args ...string) string {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = dir
	b, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v: %s", args, err, b)
	}
	return strings.TrimSpace(string(b))
}

// fak-test:runtime integration est=1s lane=default
func TestSessionPublishLifecycle(t *testing.T) {
	t.Setenv("GIT_CONFIG_GLOBAL", os.DevNull)
	t.Setenv("GIT_CONFIG_NOSYSTEM", "1")
	dir := t.TempDir()
	fixtureGit(t, dir, "init", "-q")
	fixtureGit(t, dir, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", "commit", "--allow-empty", "-qm", "fixture")
	if err := os.WriteFile(filepath.Join(dir, "sentinel"), []byte("not a session file\n"), 0600); err != nil {
		t.Fatal(err)
	}
	head := fixtureGit(t, dir, "rev-parse", "HEAD")
	status := fixtureGit(t, dir, "status", "--porcelain")
	store := leaseref.NewInDir(dir)
	lease, verdict, err := store.AcquireFenced(context.Background(), leaseref.Record{
		ID: "lease-fixture", Holder: "holder", SessionID: "fixture", TTLSeconds: 600,
	}, time.Now())
	if err != nil || !verdict.OK {
		t.Fatalf("fixture lease: %+v %v", verdict, err)
	}
	leaseOID := fixtureGit(t, dir, "rev-parse", lease.Ref())

	for _, tc := range []struct {
		name, state, wantState string
		ttl                    int64
	}{
		{"running", "RUNNING", "RUNNING", 600},
		{"stopped", "STOPPED", "STOPPED", 0},
		{"blank-state", "  ", "RUNNING", 0},
		{"trimmed-state", " PAUSED ", "PAUSED", 300},
		{"legacy-negative-ttl", "RUNNING", "RUNNING", -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			args := []string{"--dir", dir, "--session", "fixture", "--host", "fixture-host", "--state", tc.state, "--ttl", strconv.FormatInt(tc.ttl, 10)}
			before := time.Now().Unix()
			var out, stderr bytes.Buffer
			if code := RunSessionPublish(&out, &stderr, args); code != 0 {
				t.Fatalf("exit=%d stderr=%s", code, &stderr)
			}
			after := time.Now().Unix()
			if stderr.Len() != 0 {
				t.Fatalf("unexpected stderr: %s", &stderr)
			}
			var got leaseref.SessionDescriptor
			if err := json.Unmarshal(out.Bytes(), &got); err != nil {
				t.Fatal(err)
			}
			if got.ID != "fixture" || got.Host != "fixture-host" || got.PCBState != tc.wantState || got.TTLSecs != tc.ttl || got.UpdatedAt < before || got.UpdatedAt > after {
				t.Fatalf("descriptor: %+v", got)
			}
			saved, ok, err := store.GetSession(context.Background(), "fixture")
			if err != nil || !ok || saved != got {
				t.Fatalf("stored=%+v present=%v err=%v; emitted=%+v", saved, ok, err, got)
			}
			var expected bytes.Buffer
			enc := json.NewEncoder(&expected)
			enc.SetIndent("", "  ")
			if err := enc.Encode(saved); err != nil {
				t.Fatal(err)
			}
			if out.String() != expected.String() {
				t.Fatalf("wire formatting changed: %q != %q", out.String(), expected.String())
			}
			if fixtureGit(t, dir, "rev-parse", "HEAD") != head || fixtureGit(t, dir, "status", "--porcelain") != status {
				t.Fatal("session descriptor operation changed branch/index/worktree")
			}
			if fixtureGit(t, dir, "rev-parse", lease.Ref()) != leaseOID {
				t.Fatal("session descriptor changed the owned lease")
			}
		})
	}
}

// fak-test:runtime unit est=20ms lane=default
func TestSessionPublishFlagsAndRefusals(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		code int
		want string
	}{
		{"help", []string{"--help"}, 0, "Usage of fak leaseref session-publish"},
		{"missing-session", nil, 2, "--session is required"},
		{"unknown-flag", []string{"--unknown"}, 2, "flag provided but not defined"},
		{"positional", []string{"unexpected"}, 2, "unexpected argument"},
		{"bad-ttl", []string{"--ttl", "bad"}, 2, "invalid value"},
		{"unsafe-session", []string{"--session", "../unsafe"}, 1, "invalid session id"},
		{"namespace-prefix", []string{"--session", "session-fixture-1818"}, 1, "invalid session id"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out, stderr bytes.Buffer
			if got := RunSessionPublish(&out, &stderr, tc.args); got != tc.code {
				t.Fatalf("exit=%d want=%d stderr=%s", got, tc.code, &stderr)
			}
			if out.Len() != 0 || !strings.Contains(stderr.String(), tc.want) {
				t.Fatalf("stdout=%q stderr=%q", out.String(), stderr.String())
			}
		})
	}
}

type failedWriter struct{}

func (failedWriter) Write([]byte) (int, error) { return 0, errors.New("fixture output failure") }

// fak-test:runtime unit est=1ms lane=default
func TestSessionPublishJSONFailure(t *testing.T) {
	var stderr bytes.Buffer
	if code := emitLeaserefJSON(failedWriter{}, &stderr, leaseref.SessionDescriptor{}, "session-publish"); code != 1 {
		t.Fatalf("exit=%d", code)
	}
	if !strings.Contains(stderr.String(), "fak leaseref session-publish: encode json: fixture output failure") {
		t.Fatal(stderr.String())
	}
	if code := emitLeaserefJSON(io.Discard, io.Discard, leaseref.SessionDescriptor{}, "session-publish"); code != 0 {
		t.Fatalf("exit=%d", code)
	}
}
