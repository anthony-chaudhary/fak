package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

func enableAmbientAnnounce(t *testing.T) string {
	t.Helper()
	t.Setenv("FAK_LEASEREF_ANNOUNCE", "on")
	t.Setenv("FAK_LEASEREF_ANNOUNCE_ISSUE", "8947")
	t.Setenv("FAK_LEASEREF_ANNOUNCE_REPO", "owner/repo")
	key := filepath.Join(t.TempDir(), "private-key")
	if err := os.WriteFile(key, []byte("private shared key\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_LEASEREF_ANNOUNCE_KEY_FILE", key)
	return key
}

func unsetAmbientEnv(t *testing.T, name string) {
	t.Helper()
	old, present := os.LookupEnv(name)
	if err := os.Unsetenv(name); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if present {
			_ = os.Setenv(name, old)
		} else {
			_ = os.Unsetenv(name)
		}
	})
}

func requireLeaserefJSON(t *testing.T, code int, stdout, stderr *bytes.Buffer) {
	t.Helper()
	if code != 0 {
		t.Fatalf("exit=%d, want 0; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !json.Valid(stdout.Bytes()) {
		t.Fatalf("stdout is not JSON: %q (stderr=%q)", stdout.String(), stderr.String())
	}
}

func TestAmbientLeaserefAnnounceCommandsOnlyPostSuccessfulTransitions(t *testing.T) {
	enableAmbientAnnounce(t)
	dir := leaserefTempRepo(t)
	old := ambientLeaserefAnnouncePost
	t.Cleanup(func() { ambientLeaserefAnnouncePost = old })
	var bodies []string
	ambientLeaserefAnnouncePost = func(_ string, repo string, issue int, body string) error {
		if repo != "owner/repo" || issue != 8947 {
			t.Fatalf("edge target = %q#%d", repo, issue)
		}
		bodies = append(bodies, body)
		return nil
	}

	run := func(wantCode int, fn func(*bytes.Buffer, *bytes.Buffer) int) {
		t.Helper()
		var out, errb bytes.Buffer
		if code := fn(&out, &errb); code != wantCode {
			t.Fatalf("exit=%d, want %d; stdout=%q stderr=%q", code, wantCode, out.String(), errb.String())
		}
		if !json.Valid(out.Bytes()) {
			t.Fatalf("stdout is not JSON: %q", out.String())
		}
	}

	run(0, func(out, errb *bytes.Buffer) int {
		return runLeaserefAcquire(out, errb, []string{"--dir", dir, "--id", "secret-raw-lease-8947", "--holder", "node-a/session", "--tree", "private/tree/**", "--ttl", "900"})
	})
	if len(bodies) != 1 {
		t.Fatalf("posts after acquire=%d, want 1", len(bodies))
	}
	run(leaserefRefused, func(out, errb *bytes.Buffer) int {
		return runLeaserefAcquire(out, errb, []string{"--dir", dir, "--id", "secret-raw-lease-8947", "--holder", "node-b/session"})
	})
	run(leaserefRefused, func(out, errb *bytes.Buffer) int {
		return runLeaserefRenew(out, errb, []string{"--dir", dir, "--id", "secret-raw-lease-8947", "--holder", "node-b/session", "--ttl", "800"})
	})
	if len(bodies) != 1 {
		t.Fatalf("refusals posted: %d bodies, want 1", len(bodies))
	}
	run(0, func(out, errb *bytes.Buffer) int {
		return runLeaserefRenew(out, errb, []string{"--dir", dir, "--id", "secret-raw-lease-8947", "--holder", "node-a/session", "--ttl", "800"})
	})
	if len(bodies) != 2 {
		t.Fatalf("posts after renew=%d, want 2", len(bodies))
	}
	run(leaserefRefused, func(out, errb *bytes.Buffer) int {
		return runLeaserefRelease(out, errb, []string{"--dir", dir, "--id", "secret-raw-lease-8947", "--holder", "node-b/session"})
	})
	if len(bodies) != 2 {
		t.Fatalf("release refusal posted: %d bodies, want 2", len(bodies))
	}
	run(0, func(out, errb *bytes.Buffer) int {
		return runLeaserefRelease(out, errb, []string{"--dir", dir, "--id", "secret-raw-lease-8947", "--holder", "node-a/session"})
	})
	if len(bodies) != 3 {
		t.Fatalf("posts after release=%d, want 3", len(bodies))
	}

	joined := strings.Join(bodies, "\n")
	for _, secret := range []string{"secret-raw-lease-8947", "node-a/session", "node-b/session", "private/tree/**", "private shared key", dir, "owner/repo"} {
		if strings.Contains(joined, secret) {
			t.Fatalf("public-safe comments leaked %q: %q", secret, joined)
		}
	}
}

func TestAmbientLeaserefAnnounceCommandErrorsDoNotPost(t *testing.T) {
	enableAmbientAnnounce(t)
	old := ambientLeaserefAnnouncePost
	t.Cleanup(func() { ambientLeaserefAnnouncePost = old })
	posts := 0
	ambientLeaserefAnnouncePost = func(string, string, int, string) error { posts++; return nil }
	missing := filepath.Join(t.TempDir(), "missing", "repo")

	cases := []struct {
		name string
		run  func(*bytes.Buffer, *bytes.Buffer) int
	}{
		{"acquire", func(out, errb *bytes.Buffer) int {
			return runLeaserefAcquire(out, errb, []string{"--dir", missing, "--id", "secret-raw-lease-8947", "--holder", "node/session"})
		}},
		{"renew", func(out, errb *bytes.Buffer) int {
			return runLeaserefRenew(out, errb, []string{"--dir", missing, "--id", "secret-raw-lease-8947", "--holder", "node/session"})
		}},
		{"release", func(out, errb *bytes.Buffer) int {
			return runLeaserefRelease(out, errb, []string{"--dir", missing, "--id", "secret-raw-lease-8947", "--holder", "node/session"})
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var out, errb bytes.Buffer
			if code := tc.run(&out, &errb); code != 1 {
				t.Fatalf("exit=%d, want store error 1; stdout=%q stderr=%q", code, out.String(), errb.String())
			}
		})
	}
	if posts != 0 {
		t.Fatalf("failed transitions made %d posts", posts)
	}
}

func TestAmbientLeaserefAnnouncePostFailuresPreserveCommandResults(t *testing.T) {
	enableAmbientAnnounce(t)
	dir := leaserefTempRepo(t)
	old := ambientLeaserefAnnouncePost
	t.Cleanup(func() { ambientLeaserefAnnouncePost = old })
	edgeSecret := "injected-edge-secret"
	posts := 0
	ambientLeaserefAnnouncePost = func(string, string, int, string) error {
		posts++
		return errors.New(edgeSecret)
	}

	commands := []struct {
		name string
		run  func(*bytes.Buffer, *bytes.Buffer) int
	}{
		{"acquire", func(out, errb *bytes.Buffer) int {
			return runLeaserefAcquire(out, errb, []string{"--dir", dir, "--id", "private-lane", "--holder", "private-holder", "--tree", "private/**"})
		}},
		{"renew", func(out, errb *bytes.Buffer) int {
			return runLeaserefRenew(out, errb, []string{"--dir", dir, "--id", "private-lane", "--holder", "private-holder", "--ttl", "700"})
		}},
		{"release", func(out, errb *bytes.Buffer) int {
			return runLeaserefRelease(out, errb, []string{"--dir", dir, "--id", "private-lane", "--holder", "private-holder"})
		}},
	}
	for _, tc := range commands {
		t.Run(tc.name, func(t *testing.T) {
			var out, errb bytes.Buffer
			requireLeaserefJSON(t, tc.run(&out, &errb), &out, &errb)
			warning := errb.String()
			if !strings.Contains(warning, "WARNING") || !strings.Contains(warning, "local lease operation succeeded") {
				t.Fatalf("stderr=%q", warning)
			}
			for _, secret := range []string{edgeSecret, dir, "owner/repo", "private-lane", "private-holder", "private/**", "700"} {
				if strings.Contains(warning, secret) {
					t.Fatalf("warning leaked %q: %q", secret, warning)
				}
			}
		})
	}
	if posts != 3 {
		t.Fatalf("posts=%d, want 3", posts)
	}
}

func TestAmbientLeaserefAnnounceTwoNodesRealCommandsFoldEmpty(t *testing.T) {
	enableAmbientAnnounce(t)
	dir := leaserefTempRepo(t)
	old := ambientLeaserefAnnouncePost
	t.Cleanup(func() { ambientLeaserefAnnouncePost = old })
	var bodies []string
	ambientLeaserefAnnouncePost = func(_ string, _ string, _ int, body string) error {
		bodies = append(bodies, body)
		return nil
	}

	call := func(fn func(*bytes.Buffer, *bytes.Buffer) int) {
		t.Helper()
		var out, errb bytes.Buffer
		requireLeaserefJSON(t, fn(&out, &errb), &out, &errb)
	}
	call(func(out, errb *bytes.Buffer) int {
		return runLeaserefAcquire(out, errb, []string{"--dir", dir, "--id", "node-a-lane", "--holder", "node-a/session"})
	})
	call(func(out, errb *bytes.Buffer) int {
		return runLeaserefAcquire(out, errb, []string{"--dir", dir, "--id", "node-b-lane", "--holder", "node-b/session"})
	})
	call(func(out, errb *bytes.Buffer) int {
		return runLeaserefRenew(out, errb, []string{"--dir", dir, "--id", "node-a-lane", "--holder", "node-a/session", "--ttl", "600"})
	})
	call(func(out, errb *bytes.Buffer) int {
		return runLeaserefRelease(out, errb, []string{"--dir", dir, "--id", "node-b-lane", "--holder", "node-b/session"})
	})
	call(func(out, errb *bytes.Buffer) int {
		return runLeaserefRelease(out, errb, []string{"--dir", dir, "--id", "node-a-lane", "--holder", "node-a/session"})
	})
	if len(bodies) != 5 {
		t.Fatalf("posts=%d, want 5", len(bodies))
	}
	if held := leaseref.FoldAnnouncements(bodies); len(held) != 0 {
		t.Fatalf("two-node lifecycle did not fold empty: %+v", held)
	}
}

func TestAmbientLeaserefAnnounceConfigurationIsFailOpenAndPrivate(t *testing.T) {
	dir := leaserefTempRepo(t)
	keyPath := filepath.Join(t.TempDir(), "secret-key-path")
	if err := os.WriteFile(keyPath, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	unreadablePath := filepath.Join(t.TempDir(), "secret-unreadable-path")
	if err := os.Mkdir(unreadablePath, 0o700); err != nil {
		t.Fatal(err)
	}

	old := ambientLeaserefAnnouncePost
	t.Cleanup(func() { ambientLeaserefAnnouncePost = old })
	ambientLeaserefAnnouncePost = func(string, string, int, string) error {
		t.Fatal("configuration failure unexpectedly posted")
		return nil
	}

	cases := []struct {
		name, mode, issue, repo, key, want string
		unsetMode                          bool
	}{
		{name: "default-unset-disabled", issue: "8947", repo: "secret-owner/secret-repo", key: keyPath, want: "disabled", unsetMode: true},
		{name: "unreadable-key", mode: "on", issue: "8947", repo: "secret-owner/secret-repo", key: unreadablePath, want: "unavailable or empty"},
		{name: "empty-key", mode: "on", issue: "8947", repo: "secret-owner/secret-repo", key: keyPath, want: "unavailable or empty"},
		{name: "invalid-issue", mode: "on", issue: "secret-invalid-issue", repo: "secret-owner/secret-repo", key: keyPath, want: "ISSUE is missing or invalid"},
		{name: "missing-repo", mode: "on", issue: "8947", repo: "", key: keyPath, want: "REPO is missing"},
		{name: "unrecognized-mode", mode: "secret-unrecognized-mode", issue: "8947", repo: "secret-owner/secret-repo", key: keyPath, want: "unrecognized"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.unsetMode {
				unsetAmbientEnv(t, "FAK_LEASEREF_ANNOUNCE")
			} else {
				t.Setenv("FAK_LEASEREF_ANNOUNCE", tc.mode)
			}
			t.Setenv("FAK_LEASEREF_ANNOUNCE_ISSUE", tc.issue)
			t.Setenv("FAK_LEASEREF_ANNOUNCE_REPO", tc.repo)
			t.Setenv("FAK_LEASEREF_ANNOUNCE_KEY_FILE", tc.key)
			var out, errb bytes.Buffer
			code := runLeaserefAcquire(&out, &errb, []string{"--dir", dir, "--id", "private-" + tc.name, "--holder", "private-holder", "--tree", "private/tree/**"})
			requireLeaserefJSON(t, code, &out, &errb)
			warning := errb.String()
			if !strings.Contains(warning, tc.want) {
				t.Fatalf("stderr=%q, want %q", warning, tc.want)
			}
			for _, secret := range []string{keyPath, unreadablePath, tc.key, tc.repo, tc.issue, "private-" + tc.name, "private-holder", "private/tree"} {
				if secret != "" && strings.Contains(warning, secret) {
					t.Fatalf("diagnostic leaked %q: %q", secret, warning)
				}
			}
		})
	}
}
