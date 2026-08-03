package main

// wip_mirrorstamp_test.go — the end-to-end arm of the mirror stamp (#5556, deferred from
// #5479). Every remote here is a LOCAL bare directory built inside t.TempDir(); nothing
// in this file contacts a real remote, exactly like wip_sync_test.go's fixture.
//
// What these tests are for: `fak wip status --remote R` reads a LOCAL mirror, so a mirror
// that holds nothing for a session is either "R does not have it" or "this clone has not
// looked". The unit tests in internal/wipref pin the grading; these prove the stamp is
// actually written by a real sync, read back by a real status, and that the SAME mirror
// content changes meaning when the picture goes stale.

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/wipref"
)

// wipStampPeerClone gives a SECOND working clone pointed at the same bare remote, so a
// test can have a peer publish a checkpoint this clone has not fetched — the exact
// situation in which staleness and absence look alike.
func wipStampPeerClone(t *testing.T, bare string) (string, string) {
	t.Helper()
	ctx := context.Background()
	dir, file := wipTestRepo(t)
	hooks := filepath.Join(t.TempDir(), "nohooks")
	if err := os.MkdirAll(hooks, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{
		{"config", "core.hooksPath", filepath.ToSlash(hooks)},
		{"remote", "add", "origin", filepath.ToSlash(bare)},
	} {
		if _, err := gitWipOut(ctx, dir, nil, args...); err != nil {
			t.Fatalf("git %v: %v", args, err)
		}
	}
	return dir, file
}

// TestWipMirrorStampBeforeAnySyncIsNeverSynced: a clone that has never synced must say so
// rather than present its empty mirror as a survey. This is the state every clone starts
// in, so it is the one a fleet reader will meet most often.
func TestWipMirrorStampBeforeAnySyncIsNeverSynced(t *testing.T) {
	ctx := context.Background()
	dir, file := wipTestRepo(t)
	wipSyncTestRemote(t, dir)

	if err := os.WriteFile(file, []byte("base line\nedit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, dir, "sessA", true, 500); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}

	rep, err := wipStatusFor(ctx, dir, "origin", 1000)
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	if rep.Mirror == nil {
		t.Fatal("status graded a remote but attached no mirror provenance")
	}
	if rep.Mirror.Freshness != wipref.MirrorNeverSynced {
		t.Errorf("freshness = %q, want NEVER_SYNCED", rep.Mirror.Freshness)
	}
	if rep.Mirror.EmptyIsAbsence {
		t.Error("an unsynced mirror licensed reading its emptiness as absence")
	}
	if rep.Mirror.AgeSeconds >= 0 {
		t.Errorf("age_seconds = %d, want a negative sentinel (a 0 renders as \"just now\")", rep.Mirror.AgeSeconds)
	}

	// And PLAIN output must carry it — a JSON-only flag would leave the human column free
	// to print the zero unqualified.
	var out, errb bytes.Buffer
	if rc := runWipStatus(&out, &errb, []string{"-C", dir}); rc != 0 {
		t.Fatalf("status rc=%d: %s", rc, errb.String())
	}
	for _, want := range []string{"mirror of origin", "NEVER_SYNCED", "IGNORANCE, not absence"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("plain status missing %q:\n%s", want, out.String())
		}
	}
}

// TestWipMirrorStampPushOnlyIsNotASurvey is the distinction a bare timestamp would lose.
// `--push-only` leaves a perfectly FRESH mirror that was never told anything by the
// remote: it describes this clone's own sessions. Reading a peer's absence out of it is
// unlicensed no matter how recent it is.
func TestWipMirrorStampPushOnlyIsNotASurvey(t *testing.T) {
	ctx := context.Background()
	dir, file := wipTestRepo(t)
	wipSyncTestRemote(t, dir)

	if err := os.WriteFile(file, []byte("base line\nedit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, dir, "sessA", true, 500); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	res, err := wipSync(ctx, dir, "origin", true, false)
	if err != nil {
		t.Fatalf("sync --push-only: %v", err)
	}
	if res.Source != string(wipref.MirrorFromPush) || res.SyncedAt <= 0 {
		t.Fatalf("sync result stamp = source %q at %d, want PUSH at a real time", res.Source, res.SyncedAt)
	}

	rep, err := wipStatusFor(ctx, dir, "origin", res.SyncedAt+10)
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	if rep.Mirror.Freshness != wipref.MirrorFresh {
		t.Errorf("freshness = %q, want FRESH", rep.Mirror.Freshness)
	}
	if rep.Mirror.Source != wipref.MirrorFromPush {
		t.Errorf("source = %q, want PUSH", rep.Mirror.Source)
	}
	if rep.Mirror.EmptyIsAbsence {
		t.Error("a push-only mirror licensed reading its emptiness as absence — a publication is not a census")
	}
	if c := wipref.MirrorCaveat(*rep.Mirror); !strings.Contains(c, "PUBLISHED") {
		t.Errorf("caveat %q does not say the mirror was only published to", c)
	}

	// A FETCH over the same remote is what upgrades the same mirror to a survey.
	res, err = wipSync(ctx, dir, "origin", false, true)
	if err != nil {
		t.Fatalf("sync --fetch-only: %v", err)
	}
	if res.Source != string(wipref.MirrorFromFetch) {
		t.Fatalf("sync result source = %q, want FETCH", res.Source)
	}
	rep, err = wipStatusFor(ctx, dir, "origin", res.SyncedAt+10)
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	if !rep.Mirror.EmptyIsAbsence {
		t.Error("a fresh fetch did not license reading the mirror as a survey")
	}
	if c := wipref.MirrorCaveat(*rep.Mirror); c != "" {
		t.Errorf("licensed view still carries a caveat: %q", c)
	}
}

// TestWipMirrorStampTellsStaleApartFromEmpty is the ticket's property with real refs. A
// peer publishes a checkpoint AFTER this clone's last fetch. The mirror holds nothing for
// that peer either way — what changes is whether this clone is entitled to report it.
func TestWipMirrorStampTellsStaleApartFromEmpty(t *testing.T) {
	ctx := context.Background()
	dir, file := wipTestRepo(t)
	bare := wipSyncTestRemote(t, dir)

	if err := os.WriteFile(file, []byte("base line\nedit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, dir, "sessA", true, 500); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	first, err := wipSync(ctx, dir, "origin", true, true)
	if err != nil {
		t.Fatalf("sync: %v", err)
	}

	// A peer host publishes its own checkpoint to the same remote. This clone does not
	// fetch again, so its mirror is now behind reality by exactly one session.
	peer, peerFile := wipStampPeerClone(t, bare)
	if err := os.WriteFile(peerFile, []byte("base line\npeer edit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, peer, "sessPeer", true, 500); err != nil {
		t.Fatalf("peer checkpoint: %v", err)
	}
	if _, err := wipSync(ctx, peer, "origin", true, false); err != nil {
		t.Fatalf("peer sync: %v", err)
	}

	// The remote really holds both now; this clone's mirror holds only its own.
	if got := len(wipSyncRefsIn(t, bare, "refs/fak/wip")); got != 2 {
		t.Fatalf("remote holds %d checkpoint refs, want 2", got)
	}
	mine := wipSyncRefsIn(t, dir, strings.TrimSuffix(wipref.RemoteMirrorNamespace, "/"))
	if len(mine) != 1 {
		t.Fatalf("local mirror = %v, want exactly this clone's own session", mine)
	}

	// READ ONE — inside the tolerance. The reader is told the mirror is a recent survey.
	fresh, err := wipStatusFor(ctx, dir, "origin", first.SyncedAt+10)
	if err != nil {
		t.Fatalf("status (fresh): %v", err)
	}
	if fresh.Mirror.Freshness != wipref.MirrorFresh || !fresh.Mirror.EmptyIsAbsence {
		t.Errorf("fresh read: %q / licensed=%v, want FRESH / true", fresh.Mirror.Freshness, fresh.Mirror.EmptyIsAbsence)
	}

	// READ TWO — the SAME mirror bytes, read past the tolerance. Nothing on disk changed;
	// what changed is that this clone's picture aged out, and the report must stop
	// presenting sessPeer's absence from the mirror as sessPeer's absence from the remote.
	stale, err := wipStatusFor(ctx, dir, "origin", first.SyncedAt+wipref.DefaultMirrorMaxAgeSeconds+1)
	if err != nil {
		t.Fatalf("status (stale): %v", err)
	}
	if stale.Mirror.Freshness != wipref.MirrorStale {
		t.Errorf("stale read: freshness = %q, want STALE", stale.Mirror.Freshness)
	}
	if stale.Mirror.EmptyIsAbsence {
		t.Error("a stale mirror licensed reading its emptiness as absence — this is the overstatement the stamp exists to stop")
	}
	if stale.Mirror.Mirrored != fresh.Mirror.Mirrored {
		t.Fatalf("mirror content changed between reads (%d vs %d) — the test is no longer about the stamp",
			stale.Mirror.Mirrored, fresh.Mirror.Mirrored)
	}
	if c := wipref.MirrorCaveat(*stale.Mirror); !strings.Contains(c, "staleness, not absence") {
		t.Errorf("stale caveat %q does not name the failure mode", c)
	}

	// READ THREE — re-sync. What looked like "the peer has nothing" was this clone not
	// having looked: the peer's checkpoint was on the remote the whole time.
	again, err := wipSync(ctx, dir, "origin", true, true)
	if err != nil {
		t.Fatalf("resync: %v", err)
	}
	after, err := wipStatusFor(ctx, dir, "origin", again.SyncedAt+10)
	if err != nil {
		t.Fatalf("status (after resync): %v", err)
	}
	if after.Mirror.Mirrored != 2 || !after.Mirror.EmptyIsAbsence {
		t.Fatalf("after resync: mirrored=%d licensed=%v, want 2 / true", after.Mirror.Mirrored, after.Mirror.EmptyIsAbsence)
	}
	if !strings.Contains(strings.Join(wipSyncRefsIn(t, dir, strings.TrimSuffix(wipref.RemoteMirrorNamespace, "/")), "\n"), "sessPeer") {
		t.Error("the peer's checkpoint did not arrive in the mirror after a re-fetch")
	}
}

// TestWipMirrorStampIsLocalOnlyAndOutsideBothCheckpointSweeps: the stamp must never reach
// the remote (a peer's picture of a remote is not this clone's), must not appear in the
// live namespace `wip reap` deletes from, and must not appear in the mirror namespace
// keyed by session — where it would read as a phantom peer session.
func TestWipMirrorStampIsLocalOnlyAndOutsideBothCheckpointSweeps(t *testing.T) {
	ctx := context.Background()
	dir, file := wipTestRepo(t)
	bare := wipSyncTestRemote(t, dir)

	if err := os.WriteFile(file, []byte("base line\nedit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, dir, "sessA", true, 500); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if _, err := wipSync(ctx, dir, "origin", true, true); err != nil {
		t.Fatalf("sync: %v", err)
	}

	stampNS := strings.TrimSuffix(wipref.MirrorStampNamespace, "/")
	if got := wipSyncRefsIn(t, dir, stampNS); len(got) != 1 {
		t.Fatalf("local stamp refs = %v, want exactly one", got)
	}
	if got := wipSyncRefsIn(t, bare, stampNS); len(got) != 0 {
		t.Errorf("the stamp reached the remote: %v", got)
	}
	for _, ns := range []string{"refs/fak/wip", strings.TrimSuffix(wipref.RemoteMirrorNamespace, "/")} {
		for _, line := range wipSyncRefsIn(t, dir, ns) {
			if strings.Contains(line, wipref.MirrorStampNamespace) {
				t.Errorf("stamp ref enumerated by the %s sweep: %q", ns, line)
			}
		}
	}
	// The session-keyed mirror index must carry the checkpoint and nothing else — a stamp
	// leaking in here would become a peer session that does not exist.
	idx, err := wipMirrorIndex(ctx, dir, "origin")
	if err != nil {
		t.Fatalf("mirror index: %v", err)
	}
	if len(idx) != 1 {
		t.Fatalf("mirror index = %v, want exactly sessA", idx)
	}
	if _, ok := idx["sessA"]; !ok {
		t.Errorf("mirror index = %v, want the key sessA", idx)
	}
}

// TestWipMirrorStampFailedSyncLeavesNoStamp: a sync that errors must not date a picture it
// never took. The push here fails against a nonexistent remote, so the clone stays
// NEVER_SYNCED — the honest and safe direction.
func TestWipMirrorStampFailedSyncLeavesNoStamp(t *testing.T) {
	ctx := context.Background()
	dir, file := wipTestRepo(t)
	if err := os.WriteFile(file, []byte("base line\nedit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, dir, "sessA", true, 500); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}

	if _, err := wipSync(ctx, dir, "no-such-remote-here", true, true); err == nil {
		t.Fatal("expected a push failure against a nonexistent remote")
	}
	st, ok, err := wipReadMirrorStamp(ctx, dir, "no-such-remote-here")
	if err != nil {
		t.Fatalf("read stamp: %v", err)
	}
	if ok {
		t.Errorf("a failed sync stamped the mirror: %+v", st)
	}
	rep, err := wipStatusFor(ctx, dir, "no-such-remote-here", 1000)
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	if rep.Mirror.Freshness != wipref.MirrorNeverSynced || rep.Mirror.EmptyIsAbsence {
		t.Errorf("after a failed sync: %q / licensed=%v, want NEVER_SYNCED / false",
			rep.Mirror.Freshness, rep.Mirror.EmptyIsAbsence)
	}
}
