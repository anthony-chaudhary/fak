package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// A real Git two-clone witness for the read-only readiness boundary. Local
// convergence never supplies the missing authoritative freshness evidence.
// fak-test:runtime integration est=2s lane=default
func TestLeaserefReadiness(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	git := func(dir, input string, args ...string) string {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		cmd.Stdin = strings.NewReader(input)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	first, second := t.TempDir(), t.TempDir()
	git(first, "", "init", "-q")
	git(second, "", "init", "-q")
	now := time.Unix(1800000000, 0).UTC()
	put := func(name string, value any) {
		t.Helper()
		data, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		oid := git(first, string(data), "hash-object", "-w", "--stdin")
		git(first, "", "update-ref", "refs/fak/locks/"+name, oid)
	}
	lease := func(id, session string, ttl, active int64) leaseref.Record {
		return leaseref.Record{ID: id, Holder: "fixture", SessionID: session, AcquiredAt: active,
			TTLSeconds: ttl, Generation: 1, TreeGlobs: []string{"fixture/" + id}}
	}
	put("session-live", leaseref.SessionDescriptor{ID: "live", UpdatedAt: now.Unix(), TTLSecs: 3600, PCBState: "RUNNING"})
	put("session-dead", leaseref.SessionDescriptor{ID: "dead", UpdatedAt: now.Unix() - 7200, TTLSecs: 60, PCBState: "RUNNING"})
	put("epoch-backup", lease("epoch-backup", "live", 3600, now.Unix()))
	put("epoch-42", 7) // Incompatible high-water scalar, not permission to hide epoch-* leases.
	put("stale-live", lease("stale-live", "live", 0, now.Unix()-86401))
	put("expired-live", lease("expired-live", "live", 60, now.Unix()-7200))
	put("expired-dead", lease("expired-dead", "dead", 60, now.Unix()-7200))
	put("unknown", lease("unknown", "missing", 3600, now.Unix()))
	put("invalid-time", lease("invalid-time", "live", 3600, now.Unix()+100))
	put("session-future", leaseref.SessionDescriptor{ID: "future", UpdatedAt: now.Unix() + 100, TTLSecs: 3600, PCBState: "RUNNING"})
	put("invalid-session", lease("invalid-session", "future", 3600, now.Unix()))
	git(second, "", "fetch", "--no-tags", first, "+refs/fak/locks/*:refs/fak/locks/*")
	before := git(second, "", "for-each-ref", "--format=%(refname) %(objectname)")
	report := collectLeaserefReadiness(context.Background(), second, now)
	if report.Ready || !report.OK || !report.Snapshot.Stable || report.Snapshot.Authoritative {
		t.Fatalf("unsafe snapshot verdict: %+v", report)
	}
	want := map[string]string{
		"epoch-42": "INCOMPATIBLE_EPOCH_RECORD", "stale-live": "BLOCKED_CONTRADICTION",
		"expired-live": "BLOCKED_CONTRADICTION", "expired-dead": "ACTION_REAP",
		"unknown": "UNKNOWN_LIVENESS", "dead": "ACTION_DRAIN", "": "AUTHORITATIVE_SNAPSHOT_UNAVAILABLE",
		"invalid-time": "UNKNOWN_LEASE_EVIDENCE", "future": "UNKNOWN_SESSION_EVIDENCE", "invalid-session": "UNKNOWN_LIVENESS",
	}
	for _, b := range report.Blockers {
		if b.ID == "epoch-backup" {
			t.Fatalf("legitimate epoch-* lease excluded or blocked: %+v", b)
		}
		if code, ok := want[b.ID]; ok && code == b.Code {
			delete(want, b.ID)
		}
	}
	if len(want) != 0 {
		t.Fatalf("missing blockers: %v; got %+v", want, report.Blockers)
	}
	if report.Counts["leases"] != 7 || report.Counts["incompatible_records"] != 1 || report.Snapshot.Complete {
		t.Fatalf("lost records or claimed complete decode: %+v", report)
	}
	if after := git(second, "", "for-each-ref", "--format=%(refname) %(objectname)"); after != before {
		t.Fatalf("readiness mutated refs: before=%s after=%s", before, after)
	}
	// A replacement blob must not redefine the observed immutable lease object.
	original := git(second, "", "rev-parse", "refs/fak/locks/epoch-backup")
	scalar := git(second, "9", "hash-object", "-w", "--stdin")
	git(second, "", "replace", original, scalar)
	replaced := collectLeaserefReadiness(context.Background(), second, now)
	if replaced.Counts["leases"] != 7 || replaced.Counts["incompatible_records"] != 1 {
		t.Fatalf("replacement object changed the observed lease: %+v", replaced)
	}
	git(second, "", "replace", "-d", original)
	// Move the ref between enumeration and blob reads. Decode the pinned original,
	// then explicitly report that the observation changed rather than mixing heads.
	moved := collectLeaserefReadinessWithReader(context.Background(), second, now, func(ctx context.Context, dir, input string, args ...string) (string, error) {
		if args[0] == "cat-file" {
			git(second, "", "update-ref", "refs/fak/locks/epoch-backup", scalar, original)
		}
		return leaserefReadinessGit(ctx, dir, input, args...)
	})
	if moved.Snapshot.Stable || moved.Counts["leases"] != 7 || moved.Counts["incompatible_records"] != 1 {
		t.Fatalf("moved ref escaped pinned/change detection: %+v", moved)
	}
	foundChanged := false
	for _, b := range moved.Blockers {
		foundChanged = foundChanged || b.Code == "SNAPSHOT_CHANGED"
	}
	if !foundChanged || moved.Ready || moved.Snapshot.Authoritative {
		t.Fatalf("changed view escaped readiness block: %+v", moved)
	}
	git(second, "", "update-ref", "refs/fak/locks/epoch-backup", original, scalar)
	bounded := leaserefReadinessOutput{limit: 4}
	_, _ = bounded.Write([]byte("too large"))
	_, _ = bounded.Write([]byte("more"))
	if !bounded.exceeded || bounded.Len() != 4 {
		t.Fatalf("unbounded capture: %+v", bounded)
	}
	// Publishing a healthy update elsewhere cannot make this clone authoritative.
	put("stale-live", lease("stale-live", "live", 3600, now.Unix()))
	stale := collectLeaserefReadiness(context.Background(), second, now)
	if stale.Ready || stale.Snapshot.Freshness != "unverified" {
		t.Fatalf("stale clone promoted: %+v", stale)
	}
	git(second, "", "fetch", "--no-tags", first, "+refs/fak/locks/*:refs/fak/locks/*")
	converged := collectLeaserefReadiness(context.Background(), second, now)
	if converged.Ready || converged.Snapshot.Authoritative {
		t.Fatalf("Git convergence became authority: %+v", converged)
	}
	var out, errOut bytes.Buffer
	if code := runLeaseref(&out, &errOut, []string{"audit", "--require-ready", "--dir", second}); code != 1 {
		t.Fatalf("not-ready exit=%d stdout=%s stderr=%s", code, &out, &errOut)
	}
	var cli leaserefReadinessReport
	if err := json.Unmarshal(out.Bytes(), &cli); err != nil || cli.Ready || cli.Schema != "fak.leaseref-readiness.v1" {
		t.Fatalf("CLI report: %v %+v", err, cli)
	}
	empty := t.TempDir()
	git(empty, "", "init", "-q")
	clean := collectLeaserefReadiness(context.Background(), empty, now)
	if clean.Ready || !clean.OK || !clean.Snapshot.Complete || len(clean.Blockers) != 1 || clean.Blockers[0].Code != "AUTHORITATIVE_SNAPSHOT_UNAVAILABLE" {
		t.Fatalf("empty visibility must not be READY: %+v", clean)
	}
	// An existing temp directory can be inside a Git worktree when TMPDIR is
	// workspace-local. Use a missing directory so discovery cannot find an ancestor.
	bad := collectLeaserefReadiness(context.Background(), filepath.Join(t.TempDir(), "missing"), now)
	if bad.OK || bad.Ready || bad.Snapshot.Complete {
		t.Fatalf("failed enumeration became healthy: ok=%v ready=%v complete=%v", bad.OK, bad.Ready, bad.Snapshot.Complete)
	}
}
