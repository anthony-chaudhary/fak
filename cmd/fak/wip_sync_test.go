package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/wipref"
)

// wipSyncTestRemote gives the throwaway repo a LOCAL bare remote named "origin" — a
// directory, never a network host, so the replication path is exercised end to end
// without any outward-facing publication. It also pins core.hooksPath at an empty
// directory so a developer's global hooks cannot fire on the test's push.
func wipSyncTestRemote(t *testing.T, repo string) string {
	t.Helper()
	ctx := context.Background()
	bare := filepath.Join(t.TempDir(), "remote.git")
	if _, err := gitWipOut(ctx, "", nil, "init", "--bare", "-q", bare); err != nil {
		t.Fatalf("init bare: %v", err)
	}
	hooks := filepath.Join(t.TempDir(), "nohooks")
	if err := os.MkdirAll(hooks, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{
		{"config", "core.hooksPath", filepath.ToSlash(hooks)},
		{"remote", "add", "origin", filepath.ToSlash(bare)},
	} {
		if _, err := gitWipOut(ctx, repo, nil, args...); err != nil {
			t.Fatalf("git %v: %v", args, err)
		}
	}
	return bare
}

// wipSyncRefsIn lists one namespace's refs in dir as "ref object" lines.
func wipSyncRefsIn(t *testing.T, dir, pattern string) []string {
	t.Helper()
	out, err := gitWipOut(context.Background(), dir, nil,
		"for-each-ref", "--format=%(refname) %(objectname)", pattern)
	if err != nil {
		t.Fatalf("for-each-ref %s in %s: %v", pattern, dir, err)
	}
	if strings.TrimSpace(out) == "" {
		return nil
	}
	return strings.Split(strings.TrimSpace(out), "\n")
}

// wipSyncStatusRow returns the status row for a session, graded against origin.
func wipSyncStatusRow(t *testing.T, repo, session string) wipref.SessionStatus {
	t.Helper()
	rep, err := wipStatusFor(context.Background(), repo, "origin", 1000)
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	for _, s := range rep.Sessions {
		if s.Session == session {
			return s
		}
	}
	t.Fatalf("no status row for session %q (report: %+v)", session, rep)
	return wipref.SessionStatus{}
}

// TestWipSyncPushOnlyReportsReplicated is the ticket's second done-condition end to end:
// before a sync a checkpoint reads LOCAL_ONLY, and after `fak wip sync --push-only` the
// SAME checkpoint reads REPLICATED — with the bare remote actually holding the object,
// so the column is evidence rather than bookkeeping. --push-only is the mode that
// matters most here: an operator who wants durability must not have to download every
// peer host's captured working tree to get an honest status column.
func TestWipSyncPushOnlyReportsReplicated(t *testing.T) {
	ctx := context.Background()
	dir, file := wipTestRepo(t)
	bare := wipSyncTestRemote(t, dir)

	if err := os.WriteFile(file, []byte("base line\nedit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cp, err := wipCheckpoint(ctx, dir, "sessA", true, 500)
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}

	if got := wipSyncStatusRow(t, dir, "sessA").Replication; got != string(wipref.ReplicationLocalOnly) {
		t.Fatalf("before sync: replication=%q, want LOCAL_ONLY", got)
	}

	res, err := wipSync(ctx, dir, "origin", true, false)
	if err != nil {
		t.Fatalf("sync --push-only: %v", err)
	}
	if !res.Pushed || res.Fetched {
		t.Fatalf("push-only sync: pushed=%v fetched=%v, want true/false", res.Pushed, res.Fetched)
	}

	// The remote really holds it — the claim is not self-reported.
	remoteRefs := wipSyncRefsIn(t, bare, "refs/fak/wip")
	wantLine := "refs/fak/wip/sessA " + cp.Object
	if len(remoteRefs) != 1 || remoteRefs[0] != wantLine {
		t.Fatalf("remote refs = %v, want exactly [%q]", remoteRefs, wantLine)
	}

	row := wipSyncStatusRow(t, dir, "sessA")
	if row.Replication != string(wipref.ReplicationReplicated) {
		t.Fatalf("after push-only sync: replication=%q, want REPLICATED", row.Replication)
	}
	if row.RemoteObject != cp.Object {
		t.Errorf("remote_object=%q, want %q", row.RemoteObject, cp.Object)
	}
	if res.Replicated != 1 || res.LocalOnly != 0 || res.StaleRemote != 0 {
		t.Errorf("sync census = repl %d / stale %d / local %d, want 1/0/0", res.Replicated, res.StaleRemote, res.LocalOnly)
	}
}

// TestWipSyncStaleRemoteAfterNewCheckpoint covers the state that actually bites: an
// operator synced once, kept working, and checkpointed again. An OLDER delta is on the
// remote and the current one is not — which must not read as REPLICATED.
func TestWipSyncStaleRemoteAfterNewCheckpoint(t *testing.T) {
	ctx := context.Background()
	dir, file := wipTestRepo(t)
	wipSyncTestRemote(t, dir)

	if err := os.WriteFile(file, []byte("base line\nedit one\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	first, err := wipCheckpoint(ctx, dir, "sessA", true, 500)
	if err != nil {
		t.Fatalf("checkpoint one: %v", err)
	}
	if _, err := wipSync(ctx, dir, "origin", true, true); err != nil {
		t.Fatalf("sync: %v", err)
	}

	if err := os.WriteFile(file, []byte("base line\nedit two\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	second, err := wipCheckpoint(ctx, dir, "sessA", true, 600)
	if err != nil {
		t.Fatalf("checkpoint two: %v", err)
	}
	if second.Object == first.Object {
		t.Fatalf("second checkpoint did not mint a new object (%s)", second.Object)
	}

	row := wipSyncStatusRow(t, dir, "sessA")
	if row.Replication != string(wipref.ReplicationStaleRemote) {
		t.Fatalf("after re-checkpoint: replication=%q, want STALE_REMOTE", row.Replication)
	}
	if row.RemoteObject != first.Object {
		t.Errorf("remote_object=%q, want the FIRST checkpoint %q", row.RemoteObject, first.Object)
	}

	// Re-syncing publishes the new delta and the row clears.
	if _, err := wipSync(ctx, dir, "origin", true, true); err != nil {
		t.Fatalf("resync: %v", err)
	}
	if got := wipSyncStatusRow(t, dir, "sessA").Replication; got != string(wipref.ReplicationReplicated) {
		t.Fatalf("after resync: replication=%q, want REPLICATED", got)
	}
}

// TestWipSyncFetchNeverRepopulatesLiveNamespace is the divergence from leaseref.Sync
// asserted as behaviour. leaseref force-fetches peers' refs into its own live
// namespace; here the fetch must land in the mirror ONLY, because refs/fak/wip/* is the
// population `wip reap` deletes from and `wip land`/`reconcile` apply to THIS working
// tree. Deleting the local ref and fetching must not bring it back into the live set.
func TestWipSyncFetchNeverRepopulatesLiveNamespace(t *testing.T) {
	ctx := context.Background()
	dir, file := wipTestRepo(t)
	wipSyncTestRemote(t, dir)

	if err := os.WriteFile(file, []byte("base line\nedit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cp, err := wipCheckpoint(ctx, dir, "sessA", true, 500)
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if _, err := wipSync(ctx, dir, "origin", true, true); err != nil {
		t.Fatalf("sync: %v", err)
	}
	if _, err := gitWipOut(ctx, dir, nil, "update-ref", "-d", wipref.SessionRef("sessA")); err != nil {
		t.Fatalf("delete local ref: %v", err)
	}

	if _, err := wipSync(ctx, dir, "origin", false, true); err != nil {
		t.Fatalf("fetch-only sync: %v", err)
	}

	if live := wipSyncRefsIn(t, dir, "refs/fak/wip"); len(live) != 0 {
		t.Fatalf("fetch repopulated the LIVE namespace: %v", live)
	}
	mirror := wipSyncRefsIn(t, dir, strings.TrimSuffix(wipref.RemoteMirrorNamespace, "/"))
	wantLine := wipref.MirrorSessionRef("origin", "sessA") + " " + cp.Object
	if len(mirror) != 1 || mirror[0] != wantLine {
		t.Fatalf("mirror = %v, want exactly [%q]", mirror, wantLine)
	}
}

// TestWipSyncEmptyNamespaceIsACleanNoOp: syncing a clone that has never checkpointed
// must exit 0 rather than erroring on an empty namespace.
//
// This one is a REGRESSION FENCE around a claim that turns out to be half true.
// internal/leaseref/sync.go:53-55 documents "a wildcard refspec matching ZERO refs is a
// successful no-op in git, so syncing an empty namespace is clean on both sides". It is
// clean on the FETCH side. On the PUSH side git answers "No refs in common and none
// specified; doing nothing." with exit 1, which this test caught the first time it ran
// (git version recorded in the ticket). wipSync therefore short-circuits an empty push
// instead of relying on the claim.
func TestWipSyncEmptyNamespaceIsACleanNoOp(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipTestRepo(t)
	wipSyncTestRemote(t, dir)

	res, err := wipSync(ctx, dir, "origin", true, true)
	if err != nil {
		t.Fatalf("sync of an empty namespace: %v", err)
	}
	if !res.Pushed || !res.Fetched {
		t.Fatalf("pushed=%v fetched=%v, want both true", res.Pushed, res.Fetched)
	}
	if res.Published != 0 || res.Mirrored != 0 {
		t.Errorf("published=%d mirrored=%d, want 0/0", res.Published, res.Mirrored)
	}
}

// TestWipSyncRefusesUnusableArguments: an argv-unsafe remote and a both-directions-off
// call are usage errors, not silent no-ops — the same contract as leaseref.Sync.
func TestWipSyncRefusesUnusableArguments(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipTestRepo(t)

	if _, err := wipSync(ctx, dir, "--upload-pack=evil", true, true); err == nil {
		t.Error("expected a refusal for an argv-unsafe remote")
	}
	if _, err := wipSync(ctx, dir, "origin", false, false); err == nil {
		t.Error("expected a refusal when neither direction is enabled")
	}

	var out, errb bytes.Buffer
	if rc := runWipSync(&out, &errb, []string{"--push-only", "--fetch-only"}); rc != 2 {
		t.Errorf("--push-only --fetch-only rc=%d, want 2 (usage)", rc)
	}
}

// TestWipSyncFailedPushStopsBeforeFetch: with no such remote the push fails, the sync
// reports neither direction done, and nothing local was written — so the operator sees
// one unambiguous failure instead of a quietly reclassified status column.
func TestWipSyncFailedPushStopsBeforeFetch(t *testing.T) {
	ctx := context.Background()
	dir, file := wipTestRepo(t)
	if err := os.WriteFile(file, []byte("base line\nedit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, dir, "sessA", true, 500); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}

	res, err := wipSync(ctx, dir, "no-such-remote-here", true, true)
	if err == nil {
		t.Fatal("expected a push failure against a nonexistent remote")
	}
	if res.Pushed || res.Fetched {
		t.Errorf("pushed=%v fetched=%v, want both false", res.Pushed, res.Fetched)
	}
	if !strings.Contains(err.Error(), "sync stopped before fetch") {
		t.Errorf("error %q does not carry the stop-before-fetch rationale", err)
	}
	if mirror := wipSyncRefsIn(t, dir, strings.TrimSuffix(wipref.RemoteMirrorNamespace, "/")); len(mirror) != 0 {
		t.Errorf("a failed sync wrote mirror refs: %v", mirror)
	}
	if got := wipSyncStatusRow(t, dir, "sessA").Replication; got != string(wipref.ReplicationLocalOnly) {
		t.Errorf("after a failed sync: replication=%q, want LOCAL_ONLY", got)
	}
}

// TestWipStatusPlainOutputNamesTheDistinction: the plain (non-JSON) status must SAY
// which durability the checkpoint has. The whole point of the ticket is that
// "checkpointed" and "safe" were the same word, so a JSON-only field would not fix it.
func TestWipStatusPlainOutputNamesTheDistinction(t *testing.T) {
	ctx := context.Background()
	dir, file := wipTestRepo(t)
	if err := os.WriteFile(file, []byte("base line\nedit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, dir, "sessA", true, 500); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}

	var out, errb bytes.Buffer
	if rc := runWipStatus(&out, &errb, []string{"-C", dir}); rc != 0 {
		t.Fatalf("status rc=%d: %s", rc, errb.String())
	}
	got := out.String()
	for _, want := range []string{"sessA", "LOCAL_ONLY", "not this machine"} {
		if !strings.Contains(got, want) {
			t.Errorf("status output missing %q:\n%s", want, got)
		}
	}
}
