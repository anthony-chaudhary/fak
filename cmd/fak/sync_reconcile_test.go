package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safesync"
)

func TestRunSyncReconcileCLI(t *testing.T) {
	t.Run("in-sync emits ROUTE_NOOP and JSON schema", func(t *testing.T) {
		clone := syncCLIFixture(t)
		syncGit(t, clone, "merge", "--ff-only", "origin/work")

		var out, errb bytes.Buffer
		code := runSync(&out, &errb, []string{"reconcile", "--repo", clone, "--remote", "origin", "--branch", "work", "--json"})
		if code != syncExitOK {
			t.Fatalf("exit = %d, want 0; stderr=%s stdout=%s", code, errb.String(), out.String())
		}

		var got safesync.ReconcileAssessment
		if err := json.Unmarshal(out.Bytes(), &got); err != nil {
			t.Fatalf("JSON decode: %v\n%s", err, out.String())
		}
		if got.Schema != safesync.ReconcileSchema {
			t.Errorf("schema = %q, want %q", got.Schema, safesync.ReconcileSchema)
		}
		if got.Route != safesync.RouteNoop || !got.OK {
			t.Errorf("route = %q, ok = %v, want ROUTE_NOOP and ok=true", got.Route, got.OK)
		}
	})

	t.Run("ahead emits ROUTE_PUSH and executes with apply", func(t *testing.T) {
		clone := syncCLIFixture(t)
		origin := filepath.Join(filepath.Dir(clone), "origin")
		syncGit(t, origin, "config", "receive.denyCurrentBranch", "updateInstead")
		syncGit(t, clone, "merge", "--ff-only", "origin/work")
		syncWriteFile(t, filepath.Join(clone, "feature.txt"), "feat\n")
		syncGit(t, clone, "add", ".")
		syncGit(t, clone, "commit", "-m", "feature commit")

		var out, errb bytes.Buffer
		code := runSync(&out, &errb, []string{"reconcile", "--repo", clone, "--remote", "origin", "--branch", "work", "--apply"})
		if code != syncExitOK {
			t.Fatalf("exit = %d, want 0; stderr=%s stdout=%s", code, errb.String(), out.String())
		}
		if !strings.Contains(out.String(), "ROUTE_PUSH") {
			t.Errorf("output missing ROUTE_PUSH:\n%s", out.String())
		}
		if !strings.Contains(out.String(), "primitive") {
			t.Errorf("output missing primitive:\n%s", out.String())
		}
	})

	t.Run("behind safe emits ROUTE_APPLY and executes with apply", func(t *testing.T) {
		clone := syncCLIFixture(t) // behind by origin/work's c2 commit

		var out, errb bytes.Buffer
		code := runSync(&out, &errb, []string{"reconcile", "--repo", clone, "--remote", "origin", "--branch", "work", "--apply"})
		if code != syncExitOK {
			t.Fatalf("exit = %d, want 0; stderr=%s stdout=%s", code, errb.String(), out.String())
		}
		if !strings.Contains(out.String(), "ROUTE_APPLY") {
			t.Errorf("output missing ROUTE_APPLY:\n%s", out.String())
		}
	})

	t.Run("behind dirty collision returns refused", func(t *testing.T) {
		clone := syncCLIFixture(t)
		syncWriteFile(t, filepath.Join(clone, "a.txt"), "conflicting working tree\n")

		var out, errb bytes.Buffer
		code := runSync(&out, &errb, []string{"reconcile", "--repo", clone, "--remote", "origin", "--branch", "work"})
		if code != syncExitRefused {
			t.Fatalf("exit = %d, want %d; stderr=%s stdout=%s", code, syncExitRefused, errb.String(), out.String())
		}
		if !strings.Contains(out.String(), "ROUTE_HOLD_DIRTY_COLLISION") {
			t.Errorf("output missing ROUTE_HOLD_DIRTY_COLLISION:\n%s", out.String())
		}
		if !strings.Contains(out.String(), "DIRTY_WRITE_OVERLAP") {
			t.Errorf("output missing DIRTY_WRITE_OVERLAP:\n%s", out.String())
		}
	})

	t.Run("goal integrate already containing target emits ROUTE_NOOP", func(t *testing.T) {
		clone := syncCLIFixture(t)
		syncGit(t, clone, "merge", "--ff-only", "origin/work")
		syncWriteFile(t, filepath.Join(clone, "ahead.txt"), "ahead\n")
		syncGit(t, clone, "add", ".")
		syncGit(t, clone, "commit", "-m", "ahead commit")

		var out, errb bytes.Buffer
		code := runSync(&out, &errb, []string{"reconcile", "--repo", clone, "--remote", "origin", "--branch", "work", "--goal", "integrate origin/work", "--json"})
		if code != syncExitOK {
			t.Fatalf("exit = %d, want 0; stderr=%s stdout=%s", code, errb.String(), out.String())
		}
		var got safesync.ReconcileAssessment
		if err := json.Unmarshal(out.Bytes(), &got); err != nil {
			t.Fatalf("JSON decode: %v\n%s", err, out.String())
		}
		if got.Route != safesync.RouteNoop || !got.OK {
			t.Errorf("route = %q, ok = %v, want ROUTE_NOOP and ok=true", got.Route, got.OK)
		}
	})
}
