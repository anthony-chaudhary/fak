package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wiprecon"
)

// wipReclaimFixture builds a repo carrying exactly the three reconcile verdicts #5480 is
// about, with NO live lease published so every owner reads CRASHED:
//
//	alpha   RECLAIM    — unlanded delta, still applies, captured on a base HEAD has moved past
//	bravo   RECLAIM    — unlanded delta, still applies, captured on the CURRENT HEAD
//	charlie QUARANTINE — an untracked-file delta whose file is still on disk, so
//	                     `git apply --check` refuses it ("already exists")
//
// The RECLAIM pair is produced the only way it can be: checkpoint the delta, then take
// it back OUT of the working tree, so the recorded patch applies to the tree again. That
// is the narrow window §1 of the ticket says the fleet loses by default.
//
// It returns the repo dir and the `now` the ages are measured against.
func wipReclaimFixture(t *testing.T) (string, time.Time) {
	t.Helper()
	ctx := context.Background()
	dir, file := wipTestRepo(t)
	now := time.Now()
	base := now.Add(-8 * time.Hour).Unix()

	// alpha: a tracked delta captured on HEAD0, then withdrawn from the tree.
	if err := os.WriteFile(file, []byte("base line\nedit A\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if r, err := wipCheckpoint(ctx, dir, "alpha", true, base); err != nil || r.Object == "" {
		t.Fatalf("checkpoint alpha: %v (%+v)", err, r)
	}
	if err := os.WriteFile(file, []byte("base line\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Advance HEAD by one unrelated commit so alpha's base drifts and bravo's does not.
	wipReclaimAdvanceHEAD(t, ctx, dir, "other.txt", "unrelated\n")

	// bravo: a tracked delta captured on HEAD1 (zero drift), then withdrawn.
	if err := os.WriteFile(file, []byte("base line\nedit B\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if r, err := wipCheckpoint(ctx, dir, "bravo", true, now.Add(-1*time.Hour).Unix()); err != nil || r.Object == "" {
		t.Fatalf("checkpoint bravo: %v (%+v)", err, r)
	}
	if err := os.WriteFile(file, []byte("base line\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// charlie: an untracked file, checkpointed and LEFT on disk. Its patch creates a
	// file that already exists, so it can never apply -> QUARANTINE, not RECLAIM.
	if err := os.WriteFile(filepath.Join(dir, "orphan.txt"), []byte("untracked charlie\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if r, err := wipCheckpoint(ctx, dir, "charlie", true, now.Add(-30*time.Minute).Unix()); err != nil || r.Object == "" {
		t.Fatalf("checkpoint charlie: %v (%+v)", err, r)
	}
	return dir, now
}

// wipReclaimAdvanceHEAD commits one new file on top of HEAD with plumbing only (no
// checkout, no branch), moving HEAD forward inside the throwaway fixture repo.
func wipReclaimAdvanceHEAD(t *testing.T, ctx context.Context, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := gitWipOut(ctx, dir, nil, "add", name); err != nil {
		t.Fatalf("stage %s: %v", name, err)
	}
	tree, err := gitWipOut(ctx, dir, nil, "write-tree")
	if err != nil {
		t.Fatalf("write-tree: %v", err)
	}
	head, err := gitWipOut(ctx, dir, nil, "rev-parse", "HEAD")
	if err != nil {
		t.Fatalf("rev-parse HEAD: %v", err)
	}
	commit, err := gitWipOut(ctx, dir, nil, "commit-tree", tree, "-p", head, "-m", "advance trunk")
	if err != nil {
		t.Fatalf("commit-tree: %v", err)
	}
	if _, err := gitWipOut(ctx, dir, nil, "update-ref", "HEAD", commit); err != nil {
		t.Fatalf("update-ref HEAD: %v", err)
	}
}

// TestWipReconcileBuildsRankedReclaimWorklist is the #5480 witness at the fold: the
// recovery half of reconcile now PRODUCES something. Before this, reconcile emitted only
// per-checkpoint verdicts and nothing ever consumed the RECLAIM slice, so the window in
// which a crashed session's delta was still recoverable expired unattended.
//
// It pins all three properties the queue is worth having for: it contains exactly the
// RECLAIM rows (charlie's QUARANTINE never appears as recoverable), it is ranked
// most-decayed-first by base drift, and it carries the drift/age facts that make the
// verdict's remaining life visible BEFORE it runs out.
func TestWipReconcileBuildsRankedReclaimWorklist(t *testing.T) {
	dir, now := wipReclaimFixture(t)

	res, err := wipReconcileAt(context.Background(), dir, now)
	if err != nil {
		t.Fatalf("wipReconcileAt: %v", err)
	}

	// Precondition: the fixture really does produce the verdicts the queue sorts over.
	verdict := map[string]wiprecon.Action{}
	for _, d := range res.Decisions {
		verdict[d.Session] = d.Action
	}
	for sess, want := range map[string]wiprecon.Action{
		"alpha":   wiprecon.ActReclaim,
		"bravo":   wiprecon.ActReclaim,
		"charlie": wiprecon.ActQuarantine,
	} {
		if verdict[sess] != want {
			t.Fatalf("fixture broken: %s classified %q, want %q; decisions=%+v", sess, verdict[sess], want, res.Decisions)
		}
	}

	if len(res.Reclaim) != 2 {
		t.Fatalf("recovery worklist must hold exactly the 2 RECLAIM rows, got %d: %+v", len(res.Reclaim), res.Reclaim)
	}
	// Ranked most-decayed-first: alpha's base is one commit behind HEAD, bravo's is HEAD.
	if res.Reclaim[0].Session != "alpha" || res.Reclaim[1].Session != "bravo" {
		t.Fatalf("worklist not ranked most-decayed-first: %s then %s", res.Reclaim[0].Session, res.Reclaim[1].Session)
	}
	if res.Reclaim[0].TrunkDistance != 1 {
		t.Errorf("alpha base drift = %d, want 1 (HEAD advanced one commit past its base)", res.Reclaim[0].TrunkDistance)
	}
	if res.Reclaim[1].TrunkDistance != 0 {
		t.Errorf("bravo base drift = %d, want 0 (captured on the current HEAD)", res.Reclaim[1].TrunkDistance)
	}
	// The age column must be real, not a zero placeholder: alpha was stamped 8h ago.
	if got := res.Reclaim[0].AgeHours; got < 7.5 || got > 8.5 {
		t.Errorf("alpha age = %.2fh, want ~8h from its stamp", got)
	}
	if res.Reclaim[0].StartSHA == "" || res.Reclaim[0].Object == "" {
		t.Errorf("worklist row must carry the checkpoint object and its base: %+v", res.Reclaim[0])
	}
	// A QUARANTINE session must never be offered as recoverable.
	for _, r := range res.Reclaim {
		if r.Session == "charlie" {
			t.Fatalf("QUARANTINE row leaked into the recovery worklist: %+v", r)
		}
	}
}

// TestWipReconcileReclaimFlagExitsThreeWithQueue is the CLI contract #5480 asks for
// verbatim: "surfaces them as a ranked worklist with exit 3, the shape `wip blocked
// --landable` already uses". Exit 3 is what lets the garden tick / a hook branch on
// "there is recoverable work here" without parsing output — the driver the epic was
// missing between the classifier (C3) and the lander (C4).
func TestWipReconcileReclaimFlagExitsThreeWithQueue(t *testing.T) {
	dir, _ := wipReclaimFixture(t)

	var out, errb bytes.Buffer
	rc := runWipReconcile(&out, &errb, []string{"-C", dir, "--reclaim"})
	if rc != 3 {
		t.Fatalf("`wip reconcile --reclaim` with a non-empty queue: rc=%d, want 3\nstdout:\n%s\nstderr:\n%s", rc, out.String(), errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "DRIFT") {
		t.Errorf("worklist must surface the base-drift column; stdout:\n%s", got)
	}
	ai, bi := strings.Index(got, "alpha"), strings.Index(got, "bravo")
	if ai < 0 || bi < 0 || ai > bi {
		t.Errorf("worklist must list alpha (drift 1) before bravo (drift 0); stdout:\n%s", got)
	}
	if strings.Contains(got, "charlie") {
		t.Errorf("--reclaim must print ONLY the RECLAIM rows; charlie (QUARANTINE) leaked:\n%s", got)
	}
	if !strings.Contains(got, "fak wip reconcile adopt alpha") {
		t.Errorf("each row must name its recovery command; stdout:\n%s", got)
	}
	// #5998: the queue is read by a fleet, so it must say who holds each row. An
	// unclaimed row's OWNER cell is "-", and the footer counts what is actually free.
	if !strings.Contains(got, "OWNER") || !strings.Contains(got, "2 unclaimed") {
		t.Errorf("worklist must surface adoption ownership; stdout:\n%s", got)
	}

	// The default (unfiltered) listing is unchanged: every verdict, exit 0.
	var out2, errb2 bytes.Buffer
	if rc2 := runWipReconcile(&out2, &errb2, []string{"-C", dir}); rc2 != 0 {
		t.Fatalf("plain `wip reconcile` must stay exit 0: rc=%d\nstderr:\n%s", rc2, errb2.String())
	}
	if !strings.Contains(out2.String(), "QUARANTINE") || !strings.Contains(out2.String(), "charlie") {
		t.Errorf("plain listing must still show every verdict; stdout:\n%s", out2.String())
	}
}

// TestWipReconcileReclaimJSONCarriesRankedQueue: the scripted view and the human view
// must agree. In --reclaim mode the JSON decisions narrow to the same rows the listing
// printed, and the ranked worklist rides along with its drift facts.
func TestWipReconcileReclaimJSONCarriesRankedQueue(t *testing.T) {
	dir, _ := wipReclaimFixture(t)

	var out, errb bytes.Buffer
	if rc := runWipReconcile(&out, &errb, []string{"-C", dir, "--reclaim", "--json"}); rc != 3 {
		t.Fatalf("rc=%d, want 3\nstderr:\n%s", rc, errb.String())
	}
	var res wipReconcileResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("decode JSON: %v\nstdout:\n%s", err, out.String())
	}
	if len(res.Decisions) != 2 {
		t.Errorf("--reclaim --json must narrow decisions to the printed queue, got %d: %+v", len(res.Decisions), res.Decisions)
	}
	for _, d := range res.Decisions {
		if d.Action != wiprecon.ActReclaim {
			t.Errorf("narrowed decision list holds a non-RECLAIM row: %+v", d)
		}
	}
	if len(res.Reclaim) != 2 || res.Reclaim[0].Session != "alpha" {
		t.Fatalf("ranked worklist missing or misordered in JSON: %+v", res.Reclaim)
	}
	if res.Reclaim[0].TrunkDistance <= res.Reclaim[1].TrunkDistance {
		t.Errorf("JSON worklist must be ranked by descending drift: %+v", res.Reclaim)
	}
}

// TestWipReconcileReclaimEmptyQueueExitsZero: exit 3 means "there is a lever here", so
// an empty queue must NOT report one. A repo whose only checkpoint is quarantined has no
// recoverable work, and a scheduler polling this must not spin on it.
func TestWipReconcileReclaimEmptyQueueExitsZero(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipTestRepo(t)
	if err := os.WriteFile(filepath.Join(dir, "orphan.txt"), []byte("still on disk\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if r, err := wipCheckpoint(ctx, dir, "solo", true, time.Now().Unix()); err != nil || r.Object == "" {
		t.Fatalf("checkpoint solo: %v (%+v)", err, r)
	}

	var out, errb bytes.Buffer
	rc := runWipReconcile(&out, &errb, []string{"-C", dir, "--reclaim"})
	if rc != 0 {
		t.Fatalf("empty recovery queue must exit 0, got %d\nstdout:\n%s\nstderr:\n%s", rc, out.String(), errb.String())
	}
	if !strings.Contains(out.String(), "no reclaimable checkpoints") {
		t.Errorf("empty queue must say so plainly; stdout:\n%s", out.String())
	}
}

// TestWipBaseDistanceReportsUnknownRatherThanZero pins the honesty rule the ranking
// depends on: an unresolvable base is DriftUnknown, never 0. Reporting 0 would read as
// "captured on today's HEAD, maximum remaining life" — the one misreading that buries a
// genuinely unmeasurable checkpoint at the bottom of the recovery queue.
func TestWipBaseDistanceReportsUnknownRatherThanZero(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipTestRepo(t)

	head, err := gitWipOut(ctx, dir, nil, "rev-parse", "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	if got := wipBaseDistance(ctx, dir, head); got != 0 {
		t.Errorf("distance from HEAD to itself = %d, want 0", got)
	}
	if got := wipBaseDistance(ctx, dir, ""); got != wiprecon.DriftUnknown {
		t.Errorf("empty base = %d, want DriftUnknown (%d)", got, wiprecon.DriftUnknown)
	}
	if got := wipBaseDistance(ctx, dir, "0000000000000000000000000000000000000000"); got != wiprecon.DriftUnknown {
		t.Errorf("absent base object = %d, want DriftUnknown (%d)", got, wiprecon.DriftUnknown)
	}
}
