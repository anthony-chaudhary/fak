package blastlease_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/blastlease"
	"github.com/anthony-chaudhary/fak/internal/blastradius"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

func initTestRepo(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir := t.TempDir()
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "test@example.com"},
		{"config", "user.name", "test"},
	} {
		c := exec.Command("git", args...)
		c.Dir = dir
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	return dir
}

func TestRead(t *testing.T) {
	t.Run("valid JSONL multiple entries", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "leases.jsonl")
		content := strings.Join([]string{
			`{"lane": "lane-auth", "tree_globs": ["internal/auth/**", "cmd/auth/**"]}`,
			`{"lane": "lane-gate", "tree_globs": ["internal/gateway/**"]}`,
			`{"lane": "lane-single", "tree_globs": ["README.md"]}`,
		}, "\n") + "\n"

		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}

		got, err := blastlease.Read(path)
		if err != nil {
			t.Fatalf("Read(%q) unexpected error: %v", path, err)
		}

		want := []blastradius.Lease{
			{Lane: "lane-auth", TreeGlobs: []string{"internal/auth/**", "cmd/auth/**"}},
			{Lane: "lane-gate", TreeGlobs: []string{"internal/gateway/**"}},
			{Lane: "lane-single", TreeGlobs: []string{"README.md"}},
		}

		if !reflect.DeepEqual(got, want) {
			t.Fatalf("Read(%q) =\n%+v\nwant:\n%+v", path, got, want)
		}
	})

	t.Run("empty file and whitespace lines", func(t *testing.T) {
		dir := t.TempDir()

		// Completely empty file
		emptyPath := filepath.Join(dir, "empty.jsonl")
		if err := os.WriteFile(emptyPath, []byte(""), 0o644); err != nil {
			t.Fatalf("WriteFile empty: %v", err)
		}
		gotEmpty, err := blastlease.Read(emptyPath)
		if err != nil {
			t.Fatalf("Read(empty) unexpected error: %v", err)
		}
		if len(gotEmpty) != 0 {
			t.Fatalf("Read(empty) len = %d, want 0", len(gotEmpty))
		}

		// Whitespace-only file
		wsPath := filepath.Join(dir, "whitespace.jsonl")
		if err := os.WriteFile(wsPath, []byte("   \n\n\t  \r\n\n"), 0o644); err != nil {
			t.Fatalf("WriteFile whitespace: %v", err)
		}
		gotWS, err := blastlease.Read(wsPath)
		if err != nil {
			t.Fatalf("Read(whitespace) unexpected error: %v", err)
		}
		if len(gotWS) != 0 {
			t.Fatalf("Read(whitespace) len = %d, want 0", len(gotWS))
		}

		// Entries with blank lines and surrounding whitespace
		sparsePath := filepath.Join(dir, "sparse.jsonl")
		sparseContent := "\n\n   \n" +
			`  {"lane": "lane-1", "tree_globs": ["pkg1/**"]}  ` + "\n" +
			"\n\t\n" +
			`{"lane": "lane-2", "tree_globs": ["pkg2/**"]}` + "\n" +
			"\n   \n"
		if err := os.WriteFile(sparsePath, []byte(sparseContent), 0o644); err != nil {
			t.Fatalf("WriteFile sparse: %v", err)
		}
		gotSparse, err := blastlease.Read(sparsePath)
		if err != nil {
			t.Fatalf("Read(sparse) unexpected error: %v", err)
		}
		wantSparse := []blastradius.Lease{
			{Lane: "lane-1", TreeGlobs: []string{"pkg1/**"}},
			{Lane: "lane-2", TreeGlobs: []string{"pkg2/**"}},
		}
		if !reflect.DeepEqual(gotSparse, wantSparse) {
			t.Fatalf("Read(sparse) =\n%+v\nwant:\n%+v", gotSparse, wantSparse)
		}
	})

	t.Run("non-existent file", func(t *testing.T) {
		missingPath := filepath.Join(t.TempDir(), "does_not_exist.jsonl")
		_, err := blastlease.Read(missingPath)
		if err == nil {
			t.Fatalf("Read(%q) expected error, got nil", missingPath)
		}
		if !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("Read(%q) error = %v, want errors.Is(os.ErrNotExist)", missingPath, err)
		}
	})

	t.Run("malformed JSONL line formats error with line number", func(t *testing.T) {
		dir := t.TempDir()

		// Malformed JSON at line 1
		line1Path := filepath.Join(dir, "bad_line1.jsonl")
		if err := os.WriteFile(line1Path, []byte("not valid json\n"), 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		_, err1 := blastlease.Read(line1Path)
		if err1 == nil {
			t.Fatalf("Read(bad_line1) expected error, got nil")
		}
		expectedPrefix1 := fmt.Sprintf("%s line 1: ", line1Path)
		if !strings.HasPrefix(err1.Error(), expectedPrefix1) {
			t.Fatalf("error %q does not have prefix %q", err1.Error(), expectedPrefix1)
		}

		// Malformed JSON at line 3 with preceding valid line and empty line
		line3Path := filepath.Join(dir, "bad_line3.jsonl")
		contentLine3 := strings.Join([]string{
			`{"lane": "valid-lane", "tree_globs": ["pkg/**"]}`,
			``,
			`{malformed json`,
		}, "\n")
		if err := os.WriteFile(line3Path, []byte(contentLine3), 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		_, err3 := blastlease.Read(line3Path)
		if err3 == nil {
			t.Fatalf("Read(bad_line3) expected error, got nil")
		}
		expectedPrefix3 := fmt.Sprintf("%s line 3: ", line3Path)
		if !strings.HasPrefix(err3.Error(), expectedPrefix3) {
			t.Fatalf("error %q does not have prefix %q", err3.Error(), expectedPrefix3)
		}
	})
}

func TestLive(t *testing.T) {
	t.Run("fresh empty directory without repo", func(t *testing.T) {
		dir := t.TempDir()
		got, err := blastlease.Live(dir, time.Now())
		if err != nil {
			t.Fatalf("Live on empty non-repo returned error: %v", err)
		}
		if len(got) != 0 {
			t.Fatalf("Live on empty non-repo returned %d leases, want 0", len(got))
		}
	})

	t.Run("fresh initialized git repo without leases", func(t *testing.T) {
		repoDir := initTestRepo(t)
		got, err := blastlease.Live(repoDir, time.Now())
		if err != nil {
			t.Fatalf("Live on fresh repo returned error: %v", err)
		}
		if len(got) != 0 {
			t.Fatalf("Live on fresh repo returned %d leases, want 0", len(got))
		}
	})

	t.Run("real leases projection and expiry filtering", func(t *testing.T) {
		repoDir := initTestRepo(t)
		store := leaseref.NewInDir(repoDir)
		ctx := context.Background()

		t0 := time.Unix(1_700_000_000, 0)

		// 1. Lease with TTL=600s (active at t0+100s, expired at t0+700s)
		rec1 := leaseref.Record{
			ID:          "lane-alpha",
			TreeGlobs:   []string{"internal/alpha/**", "cmd/alpha/**"},
			Holder:      "agent-1",
			AcquiredAt:  t0.Unix(),
			TTLSeconds:  600,
			Description: "alpha lease",
		}
		if _, err := store.Acquire(ctx, rec1); err != nil {
			t.Fatalf("Acquire(rec1): %v", err)
		}

		// 2. Lease with TTL=0 (never expires)
		rec2 := leaseref.Record{
			ID:          "lane-beta",
			TreeGlobs:   []string{"internal/beta/**"},
			Holder:      "agent-2",
			AcquiredAt:  t0.Unix(),
			TTLSeconds:  0,
			Description: "beta permanent lease",
		}
		if _, err := store.Acquire(ctx, rec2); err != nil {
			t.Fatalf("Acquire(rec2): %v", err)
		}

		// 3. Short-lived lease with TTL=60s (expired at t0+100s)
		rec3 := leaseref.Record{
			ID:          "lane-gamma",
			TreeGlobs:   []string{"internal/gamma/**"},
			Holder:      "agent-3",
			AcquiredAt:  t0.Unix(),
			TTLSeconds:  60,
			Description: "gamma short lease",
		}
		if _, err := store.Acquire(ctx, rec3); err != nil {
			t.Fatalf("Acquire(rec3): %v", err)
		}

		// 4. Session descriptor (refs/fak/locks/session-*) should NOT be projected as a lock lease
		sess := leaseref.SessionDescriptor{
			ID:        "sess-test",
			Host:      "host-1",
			PCBState:  "RUNNING",
			UpdatedAt: t0.Unix(),
			TTLSecs:   300,
		}
		if _, err := store.PublishSession(ctx, sess); err != nil {
			t.Fatalf("PublishSession: %v", err)
		}

		// Query at t0 + 100s: rec1 and rec2 should be active; rec3 is expired; session descriptor ignored.
		now1 := t0.Add(100 * time.Second)
		got1, err := blastlease.Live(repoDir, now1)
		if err != nil {
			t.Fatalf("Live(now1): %v", err)
		}

		want1 := []blastradius.Lease{
			{Lane: "lane-alpha", TreeGlobs: []string{"internal/alpha/**", "cmd/alpha/**"}},
			{Lane: "lane-beta", TreeGlobs: []string{"internal/beta/**"}},
		}

		if !reflect.DeepEqual(got1, want1) {
			t.Fatalf("Live(now1) =\n%+v\nwant:\n%+v", got1, want1)
		}

		// Query at t0 + 700s: rec1 and rec3 expired; only rec2 (TTL=0) remains active.
		now2 := t0.Add(700 * time.Second)
		got2, err := blastlease.Live(repoDir, now2)
		if err != nil {
			t.Fatalf("Live(now2): %v", err)
		}

		want2 := []blastradius.Lease{
			{Lane: "lane-beta", TreeGlobs: []string{"internal/beta/**"}},
		}

		if !reflect.DeepEqual(got2, want2) {
			t.Fatalf("Live(now2) =\n%+v\nwant:\n%+v", got2, want2)
		}

		// Query at t0 - 10s: all 3 should be active (within TTL window relative to t0).
		now0 := t0.Add(-10 * time.Second)
		got0, err := blastlease.Live(repoDir, now0)
		if err != nil {
			t.Fatalf("Live(now0): %v", err)
		}
		if len(got0) != 3 {
			t.Fatalf("Live(now0) len = %d, want 3", len(got0))
		}
	})
}
