package main

import (
	"bytes"
	"context"
	"encoding/json"
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
	// The other half of #5567: a clone that DID publish must not borrow the empty-skip
	// excuse. Pushed=true is only ever earned by a push subprocess that exited 0.
	if res.PushSkippedEmpty {
		t.Fatalf("a clone holding a checkpoint must PUSH it, not report an empty skip: %+v", res)
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
// must exit 0 rather than erroring on an empty namespace — and must say so HONESTLY.
//
// This one is a REGRESSION FENCE around an asymmetry both substrates now document:
// internal/leaseref/sync.go's syncRefspec comment ("ZERO MATCHES ARE NOT SYMMETRIC",
// #5550) for the lease namespace, and internal/wipref.PushRefspec for this one. A
// zero-match wildcard refspec is a clean no-op on the FETCH side; on the PUSH side git
// answers "No refs in common and none specified; doing nothing." with exit 1, which this
// test caught the first time it ran (git version recorded in the ticket). wipSync
// therefore short-circuits an empty push instead of handing git the refspec.
//
// What the skip may NOT do is call itself a push (#5567). Pushed stays false — no
// subprocess started — and PushSkippedEmpty carries the no-op, matching
// leaseref.SyncResult so a ledger reading either substrate can tell "published my
// checkpoints" from "had none to publish". The fetch still runs: that is the whole point
// of not erroring.
func TestWipSyncEmptyNamespaceIsACleanNoOp(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipTestRepo(t)
	wipSyncTestRemote(t, dir)

	res, err := wipSync(ctx, dir, "origin", true, true)
	if err != nil {
		t.Fatalf("sync of an empty namespace: %v", err)
	}
	if res.Pushed {
		t.Errorf("Pushed must stay honest — no push subprocess ran: %+v", res)
	}
	if !res.PushSkippedEmpty {
		t.Errorf("want PushSkippedEmpty to record the no-op, got %+v", res)
	}
	if !res.Fetched {
		t.Errorf("Fetched=false: a clone with nothing to send must still import its peers': %+v", res)
	}
	if res.Published != 0 || res.Mirrored != 0 {
		t.Errorf("published=%d mirrored=%d, want 0/0", res.Published, res.Mirrored)
	}
}

// TestWipSyncEmptyNamespacePushOnlyReportsTheSkip pins the same honesty on the
// direction where nothing else can stand in for it. With --push-only there is no fetch
// to have happened either, so (Pushed=false, PushSkippedEmpty=true) is the ONLY thing
// separating "this clone had no checkpoints" from "the push failed" — and the two must
// not look alike, because one is exit 0 and the other is exit 1.
//
// It also pins the surface a ledger actually reads: `--json` must carry pushed:false
// alongside push_skipped_empty:true, and the plain line must name the no-op rather than
// leave a bare pushed=false reading like the failure it replaced a lie with.
func TestWipSyncEmptyNamespacePushOnlyReportsTheSkip(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipTestRepo(t)
	bare := wipSyncTestRemote(t, dir)

	res, err := wipSync(ctx, dir, "origin", true, false)
	if err != nil {
		t.Fatalf("push-only sync of an empty namespace: %v", err)
	}
	if res.Pushed || res.Fetched || !res.PushSkippedEmpty {
		t.Fatalf("pushed=%v fetched=%v skipped=%v, want false/false/true", res.Pushed, res.Fetched, res.PushSkippedEmpty)
	}
	// Nothing was published, and the evidence is the remote itself rather than the field.
	if remote := wipSyncRefsIn(t, bare, "refs/fak/wip"); len(remote) != 0 {
		t.Errorf("a skipped push put refs on the remote: %v", remote)
	}

	var out, errb bytes.Buffer
	if rc := runWipSync(&out, &errb, []string{"-C", dir, "--push-only", "--json"}); rc != 0 {
		t.Fatalf("wip sync --json rc=%d: %s", rc, errb.String())
	}
	var got wipref.SyncResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("decode --json: %v (raw: %s)", err, out.String())
	}
	if got.Pushed || !got.PushSkippedEmpty {
		t.Errorf("--json reported pushed=%v push_skipped_empty=%v, want false/true", got.Pushed, got.PushSkippedEmpty)
	}
	if !strings.Contains(out.String(), `"push_skipped_empty": true`) {
		t.Errorf("--json omitted the push_skipped_empty key:\n%s", out.String())
	}

	out.Reset()
	errb.Reset()
	if rc := runWipSync(&out, &errb, []string{"-C", dir, "--push-only"}); rc != 0 {
		t.Fatalf("wip sync (plain) rc=%d: %s", rc, errb.String())
	}
	for _, want := range []string{"pushed=false", "nothing to publish", "not a failure"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("plain output missing %q:\n%s", want, out.String())
		}
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
	// The empty-namespace skip must never become a blanket forgiveness of a push exit 1:
	// this namespace is NOT empty, so the failure is real and must not be excused (#5567).
	if res.PushSkippedEmpty {
		t.Errorf("a real push failure was recorded as an empty skip: %+v", res)
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
