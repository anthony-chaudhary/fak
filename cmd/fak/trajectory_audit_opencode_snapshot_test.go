package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

// fak-test:runtime fast est=1s lane=default
func TestTrajectoryAuditOpenCodeSnapshotFlags(t *testing.T) {
	pin := strings.Repeat("a", 64)
	base := []string{"--opencode-snapshot", "unused.sqlite", "--opencode-snapshot-sha256", pin, "--opencode-snapshot-profile", trajectory.OpenCodeSnapshotProfile}
	for _, tc := range []struct {
		name string
		args []string
		code string
	}{
		{"missing-pin", []string{"--opencode-snapshot", "unused.sqlite"}, "OPENCODE_SNAPSHOT_INPUT_REFUSED"},
		{"empty-explicit", []string{"--opencode-snapshot="}, "OPENCODE_SNAPSHOT_INPUT_REFUSED"},
		{"unknown-profile", []string{"--opencode-snapshot", "unused.sqlite", "--opencode-snapshot-sha256", pin, "--opencode-snapshot-profile", "legacy"}, "OPENCODE_SNAPSHOT_INPUT_REFUSED"},
		{"live-root", append(append([]string(nil), base...), "--codex-root", "not-read"), "OPENCODE_SNAPSHOT_FLAGS_INCOMPATIBLE"},
		{"filter", append(append([]string(nil), base...), "--user-contains", "not-read"), "OPENCODE_SNAPSHOT_FLAGS_INCOMPATIBLE"},
		{"capture", append(append([]string(nil), base...), "--snapshot-out", "not-created"), "OPENCODE_SNAPSHOT_FLAGS_INCOMPATIBLE"},
		{"replay", append(append([]string(nil), base...), "--snapshot", "not-read"), "OPENCODE_SNAPSHOT_FLAGS_INCOMPATIBLE"},
		{"baseline", append(append([]string(nil), base...), "--baseline", "not-read"), "OPENCODE_SNAPSHOT_FLAGS_INCOMPATIBLE"},
		{"usage-ledger", append(append([]string(nil), base...), "--snapshot-usage-ledger", "not-created"), "OPENCODE_SNAPSHOT_FLAGS_INCOMPATIBLE"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out, diagnostic bytes.Buffer
			code := runTrajectoryAudit(&out, &diagnostic, tc.args)
			if code != 2 || !strings.Contains(diagnostic.String(), tc.code) || out.Len() != 0 {
				t.Fatalf("flag refusal missing: code=%d diagnostic=%q", code, diagnostic.String())
			}
		})
	}
	var out, diagnostic bytes.Buffer
	if code := runTrajectoryAudit(&out, &diagnostic, []string{"--help"}); code != 0 || !strings.Contains(out.String()+diagnostic.String(), "opencode-snapshot-sha256") {
		t.Fatal("explicit snapshot flags absent from help")
	}
}

// fak-test:runtime fast est=1s lane=default
func TestTrajectoryAuditOpenCodeSnapshotOnlyAndOutputGuard(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "explicit.sqlite")
	image := make([]byte, 512)
	// A deliberately unsupported image reaches the normal RunAudit dispatch but
	// refuses before Python. Every other source home contains bait and is ignored.
	if err := os.WriteFile(path, image, 0600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(image)
	args := []string{"--opencode-snapshot", path, "--opencode-snapshot-sha256", hex.EncodeToString(sum[:]), "--opencode-snapshot-profile", trajectory.OpenCodeSnapshotProfile}
	for _, key := range []string{"CLAUDE_CONFIG_DIR", "CODEX_HOME", "OPENCODE_HOME", "XDG_DATA_HOME"} {
		t.Setenv(key, filepath.Join(root, "not-selected"))
	}
	output := filepath.Join(root, "audit.jsonl")
	var out, diagnostic bytes.Buffer
	code := runTrajectoryAudit(&out, &diagnostic, append(append([]string(nil), args...), "--jsonl", output))
	if code != 1 || !strings.Contains(diagnostic.String(), "opencode_snapshot_unsupported_image") {
		t.Fatalf("normal snapshot refusal missing: %d %s", code, diagnostic.String())
	}
	data, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(data, []byte(path)) || bytes.Contains(data, []byte("not-selected")) || bytes.Contains(data, []byte(`"source":"codex"`)) || bytes.Contains(data, []byte(`"source":"claude"`)) {
		t.Fatal("image-only mode exposed a path or selected an ambient source")
	}
	if !bytes.Contains(data, []byte("opencode_snapshot_profile")) {
		t.Fatal("bound profile missing from normal report")
	}
	for _, target := range []string{path, filepath.Join(root, "alias.sqlite")} {
		if target != path {
			if err := os.Link(path, target); err != nil {
				continue
			}
		}
		out.Reset()
		diagnostic.Reset()
		code = runTrajectoryAudit(&out, &diagnostic, append(append([]string(nil), args...), "--jsonl", target))
		if code != 2 || !strings.Contains(diagnostic.String(), "OPENCODE_SNAPSHOT_OUTPUT_CONFLICT") {
			t.Fatal("input/output alias accepted")
		}
		after, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(after, image) {
			t.Fatal("output guard failed to preserve input")
		}
	}
}
