package main

// wip_fleet_test.go — the TWO-CLONE witness for #3880. Everything here runs real git
// against real repos: a checkpoint minted on clone A, published to a bare "coordinator",
// and then enumerated from clone B, which has never met clone A and shares nothing with
// it but that remote. The pure folds are covered in internal/wipref/fleet_test.go; what
// only an end-to-end test can prove is that the ref, the stamp, and the objects actually
// survive the round trip — and, for the size gate, that they deliberately do not.

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

// wipFleetPeerClone builds the SECOND clone: its own repo with its own base commit,
// wired to the same bare remote as the first. Deliberately not a `git clone` of the
// coordinator — a fleet host is a working checkout with its own history that happens to
// share a remote, and cloning would give clone B clone A's objects for free, which is
// exactly the thing the fetch is supposed to have to do.
func wipFleetPeerClone(t *testing.T, bare string) (string, string) {
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

// wipFleetRow returns the fleet row for a session, or fails naming what was there.
func wipFleetRow(t *testing.T, rep wipref.FleetReport, session string) wipref.FleetRow {
	t.Helper()
	for _, r := range rep.Rows {
		if r.Session == session {
			return r
		}
	}
	t.Fatalf("no fleet row for session %q (report: %+v)", session, rep)
	return wipref.FleetRow{}
}

// TestWipFleetEnumeratesAPeerClonesCheckpoint is the ticket's FIRST done condition end to
// end: a WIP ref minted on one clone is enumerable via `fak wip status --fleet` from a
// second clone after a sync. Before the sync clone B can say nothing about clone A; after
// it, B names A's session, A's owner host, and the disposition that says the delta is
// recoverable from here.
func TestWipFleetEnumeratesAPeerClonesCheckpoint(t *testing.T) {
	ctx := context.Background()
	dirA, fileA := wipTestRepo(t)
	bare := wipSyncTestRemote(t, dirA)
	dirB, _ := wipFleetPeerClone(t, bare)

	// Clone B, before it has ever fetched, must not claim the fleet is clean.
	before, err := wipFleetStatus(ctx, dirB, "origin", 1000)
	if err != nil {
		t.Fatalf("fleet status before sync: %v", err)
	}
	if before.Count != 0 {
		t.Fatalf("clone B saw %d rows before any sync: %+v", before.Count, before.Rows)
	}
	if before.Mirror == nil || before.Mirror.EmptyIsAbsence {
		t.Fatalf("a never-synced clone was licensed to read its empty fleet listing as absence: %+v", before.Mirror)
	}

	if err := os.WriteFile(fileA, []byte("base line\ncrashed session work\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cp, err := wipCheckpoint(ctx, dirA, "sessa", true, 500)
	if err != nil {
		t.Fatalf("checkpoint on clone A: %v", err)
	}
	if _, err := wipSync(ctx, dirA, "origin", true, false); err != nil {
		t.Fatalf("clone A publish: %v", err)
	}

	// Clone B fetches. Nothing else about B changes — it never learns of A directly.
	if _, err := wipSync(ctx, dirB, "origin", false, true); err != nil {
		t.Fatalf("clone B fetch: %v", err)
	}
	rep, err := wipFleetStatus(ctx, dirB, "origin", 900)
	if err != nil {
		t.Fatalf("fleet status after sync: %v", err)
	}
	if rep.Count != 1 {
		t.Fatalf("clone B enumerated %d checkpoints, want 1: %+v", rep.Count, rep.Rows)
	}
	row := wipFleetRow(t, rep, "sessa")
	if row.Disposition != wipref.FleetReclaimable {
		t.Fatalf("peer checkpoint graded %s, want %s (reason: %s)", row.Disposition, wipref.FleetReclaimable, row.Reason)
	}
	if row.Object != cp.Object {
		t.Fatalf("fleet row object %q, want clone A's checkpoint %q", row.Object, cp.Object)
	}
	// The OWNER HOST is the field the ref name cannot carry, and the reason the whole
	// stamp had to grow for this ticket.
	if row.Host == "" || row.Host == wipref.HostUnknown {
		t.Fatalf("owner host = %q, want clone A's node id", row.Host)
	}
	if want := wipLocalHost(dirA); row.Host != want {
		t.Fatalf("owner host = %q, want %q", row.Host, want)
	}
	if row.StartSHA == "" || row.AgeSeconds != 400 {
		t.Fatalf("peer row lost its capture facts: %+v", row)
	}
	if rep.Reclaimable != 1 || rep.OwnLocal != 0 {
		t.Fatalf("census = %+v, want exactly one reclaimable peer row", rep)
	}

	// RECLAIMABLE is a claim that the BYTES are here, not just the ref. Prove it: clone B
	// can read the peer delta's blob out of the object it just enumerated.
	blob, err := gitWipOut(ctx, dirB, nil, "cat-file", "blob", row.Object+":note.txt")
	if err != nil {
		t.Fatalf("clone B cannot read the delta it called reclaimable: %v", err)
	}
	if !strings.Contains(blob, "crashed session work") {
		t.Fatalf("peer delta content = %q, want clone A's uncommitted edit", blob)
	}
}

// TestWipFleetSizeGateWithholdsTheDeltaButPublishesTheRef is the ticket's SECOND done
// condition: the object-size gate suppresses an over-bound delta's object push while still
// publishing its metadata ref. The two halves are asserted separately and against the
// COORDINATOR, not against the sync's own report — a self-reported "withheld" is worth
// nothing if the objects went out anyway.
func TestWipFleetSizeGateWithholdsTheDeltaButPublishesTheRef(t *testing.T) {
	ctx := context.Background()
	dirA, fileA := wipTestRepo(t)
	bare := wipSyncTestRemote(t, dirA)
	dirB, _ := wipFleetPeerClone(t, bare)

	fat := "base line\n" + strings.Repeat("oversize delta line\n", 64)
	if err := os.WriteFile(fileA, []byte(fat), 0o644); err != nil {
		t.Fatal(err)
	}
	cp, err := wipCheckpoint(ctx, dirA, "sessfat", true, 500)
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	// The gate reads a number the CAPTURE measured; if that measurement is silently zero
	// the gate can never fire and this whole test would pass vacuously.
	recs, err := wipListRecords(ctx, dirA)
	if err != nil {
		t.Fatalf("list records: %v", err)
	}
	if len(recs) != 1 || recs[0].Stamp.DeltaBytes < int64(len(fat)) {
		t.Fatalf("capture did not measure the delta: %+v (want >= %d bytes)", recs, len(fat))
	}

	const bound = 64
	res, err := wipSyncBounded(ctx, dirA, "origin", true, false, bound)
	if err != nil {
		t.Fatalf("gated publish: %v", err)
	}
	if res.MetadataOnly != 1 || res.Published != 1 || res.MaxDeltaBytes != bound {
		t.Fatalf("sync result = %+v, want 1 published, 1 metadata-only at bound %d", res, bound)
	}

	// HALF ONE: the ref IS published.
	remoteRefs := wipSyncRefsIn(t, bare, "refs/fak/wip")
	if len(remoteRefs) != 1 || !strings.HasPrefix(remoteRefs[0], "refs/fak/wip/sessfat ") {
		t.Fatalf("coordinator refs = %v, want the gated session's ref published", remoteRefs)
	}
	stub := strings.Fields(remoteRefs[0])[1]
	if stub == cp.Object {
		t.Fatalf("the gated ref points at the real checkpoint %s — nothing was withheld", cp.Object)
	}

	// HALF TWO: the delta's objects are NOT on the coordinator. This is the assertion the
	// ticket actually buys, and it has to be made against the bare repo itself.
	if _, _, code, err := gitWip(ctx, bare, nil, "cat-file", "-e", cp.Object); err == nil && code == 0 {
		t.Fatalf("the coordinator holds the withheld checkpoint object %s — the size gate did not suppress the object push", cp.Object)
	}

	// The stub is DETERMINISTIC: re-syncing an unchanged checkpoint must not churn the ref.
	if _, err := wipSyncBounded(ctx, dirA, "origin", true, false, bound); err != nil {
		t.Fatalf("re-sync: %v", err)
	}
	if again := wipSyncRefsIn(t, bare, "refs/fak/wip"); len(again) != 1 || again[0] != remoteRefs[0] {
		t.Fatalf("re-syncing an unchanged checkpoint churned the coordinator ref: %v -> %v", remoteRefs, again)
	}

	// A withheld delta must NEVER read as durable on the publishing clone.
	if got := wipSyncStatusRow(t, dirA, "sessfat").Replication; got != string(wipref.ReplicationStaleRemote) {
		t.Fatalf("gated session reads %q on the owner clone; a delta that never left the machine must not read REPLICATED", got)
	}

	// And the peer sees it as VISIBLE but not recoverable from here.
	if _, err := wipSync(ctx, dirB, "origin", false, true); err != nil {
		t.Fatalf("clone B fetch: %v", err)
	}
	rep, err := wipFleetStatus(ctx, dirB, "origin", 600)
	if err != nil {
		t.Fatalf("fleet status: %v", err)
	}
	row := wipFleetRow(t, rep, "sessfat")
	if row.Disposition != wipref.FleetMetadataOnly {
		t.Fatalf("gated peer row graded %s, want %s", row.Disposition, wipref.FleetMetadataOnly)
	}
	if row.DeltaObject != cp.Object {
		t.Fatalf("stub row names withheld object %q, want %q — a peer cannot ask for what it cannot name", row.DeltaObject, cp.Object)
	}
	if row.Host != wipLocalHost(dirA) || row.DeltaBytes < int64(len(fat)) {
		t.Fatalf("stub row lost the metadata that makes it actionable: %+v", row)
	}
	if _, _, code, _ := gitWip(ctx, dirB, nil, "cat-file", "-e", cp.Object); code == 0 {
		t.Fatalf("clone B holds the withheld delta %s; the gate leaked it through the fetch", cp.Object)
	}
}

// TestWipFleetUngatedSyncStillUsesTheGlobRefspec pins the no-regression property: when
// nothing is over the bound the publish path is the single glob refspec it always was —
// no stub mint, no per-ref argv, no extra object on the coordinator.
func TestWipFleetUngatedSyncStillUsesTheGlobRefspec(t *testing.T) {
	ctx := context.Background()
	dir, file := wipTestRepo(t)
	bare := wipSyncTestRemote(t, dir)

	if err := os.WriteFile(file, []byte("base line\nsmall\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cp, err := wipCheckpoint(ctx, dir, "sessthin", true, 500)
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	res, err := wipSyncBounded(ctx, dir, "origin", true, false, wipref.DefaultMaxDeltaBytes)
	if err != nil {
		t.Fatalf("sync: %v", err)
	}
	if res.MetadataOnly != 0 {
		t.Fatalf("a small delta was gated: %+v", res)
	}
	if res.PushRefspec != wipref.PushRefspec {
		t.Fatalf("ungated sync reported refspec %q, want the glob %q", res.PushRefspec, wipref.PushRefspec)
	}
	refs := wipSyncRefsIn(t, bare, "refs/fak/wip")
	if len(refs) != 1 || refs[0] != "refs/fak/wip/sessthin "+cp.Object {
		t.Fatalf("coordinator refs = %v, want the real checkpoint object %s", refs, cp.Object)
	}
	if got := wipSyncStatusRow(t, dir, "sessthin").Replication; got != string(wipref.ReplicationReplicated) {
		t.Fatalf("an ungated publish reads %q, want REPLICATED", got)
	}
}

// TestWipStatusFleetFlagRendersThePeerRow is the CAPTURED-OUTPUT witness the ticket asks
// for: the operator-facing `fak wip status --fleet` on clone B, folding a remote WIP ref.
// It asserts the rendered text and the JSON shape, because the fleet listing is read by
// humans at a recovery boundary and by scripts at a dispatch one.
func TestWipStatusFleetFlagRendersThePeerRow(t *testing.T) {
	ctx := context.Background()
	dirA, fileA := wipTestRepo(t)
	bare := wipSyncTestRemote(t, dirA)
	dirB, _ := wipFleetPeerClone(t, bare)

	if err := os.WriteFile(fileA, []byte("base line\nstranded\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, dirA, "sessa", true, 500); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if _, err := wipSync(ctx, dirA, "origin", true, false); err != nil {
		t.Fatalf("publish: %v", err)
	}
	if _, err := wipSync(ctx, dirB, "origin", false, true); err != nil {
		t.Fatalf("fetch: %v", err)
	}

	var out, errb bytes.Buffer
	if rc := runWipStatus(&out, &errb, []string{"-C", dirB, "--fleet"}); rc != 0 {
		t.Fatalf("wip status --fleet rc=%d\nstdout: %s\nstderr: %s", rc, out.String(), errb.String())
	}
	text := out.String()
	t.Logf("fak wip status --fleet (clone B):\n%s", text)
	for _, want := range []string{
		"no working-tree checkpoints", // clone B has none of its own...
		"sessa",                       // ...and still names the peer's stranded session,
		wipLocalHost(dirA),            // whose host it can now report,
		"RECLAIMABLE",                 // with the disposition that says it is recoverable.
		"fleet: 1 checkpoint(s) across 1 host(s) on origin",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("`wip status --fleet` output missing %q:\n%s", want, text)
		}
	}

	// Without --fleet the same call must say nothing about peers — the enumeration is
	// opt-in, exactly like the sync that populates it.
	var plain bytes.Buffer
	if rc := runWipStatus(&plain, &errb, []string{"-C", dirB}); rc != 0 {
		t.Fatalf("wip status rc=%d: %s", rc, errb.String())
	}
	if strings.Contains(plain.String(), "sessa") || strings.Contains(plain.String(), "fleet:") {
		t.Fatalf("plain `wip status` leaked the fleet listing:\n%s", plain.String())
	}

	var js bytes.Buffer
	if rc := runWipStatus(&js, &errb, []string{"-C", dirB, "--fleet", "--json"}); rc != 0 {
		t.Fatalf("wip status --fleet --json rc=%d: %s", rc, errb.String())
	}
	var report wipref.StatusReport
	if err := json.Unmarshal(js.Bytes(), &report); err != nil {
		t.Fatalf("decode --fleet --json: %v\n%s", err, js.String())
	}
	if report.Fleet == nil || report.Fleet.Count != 1 {
		t.Fatalf("JSON fleet section = %+v, want one peer row", report.Fleet)
	}
	if report.Fleet.Rows[0].Session != "sessa" || report.Fleet.Rows[0].Disposition != wipref.FleetReclaimable {
		t.Fatalf("JSON fleet row = %+v", report.Fleet.Rows[0])
	}
}
