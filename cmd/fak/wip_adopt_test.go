package main

// wip_adopt_test.go — the CRASH MATRIX for witnessed adoption (#5998).
//
// The ticket's done-condition is not "adoption works"; it is that adoption survives the
// specific ways this substrate actually breaks. Each test below is one row of that matrix
// against a real git repo, because every guarantee here is a guarantee ABOUT git:
//
//	original owner crashed            -> the checkpoint is adoptable at all
//	two successors bid at once        -> exactly one wins, the loser mutates nothing
//	successor crashed mid-materialize -> the receipt resumes into its OWN target
//	holder gone and its claim lapsed  -> takeover, audited, naming who it displaced
//	holder alive or claim still fresh -> HELD, no bytes written
//	the recovery lands                -> and re-running it is a benign no-op
//	the checkpoint is QUARANTINE      -> refused, and no claim is left behind
//
// The clock is injected (wipAdoptOptions.Now) so the TTL rows are exact rather than slept
// through, and the two-successor race is driven by wipAdoptOptions.raceHook rather than by
// goroutine timing — a race whose whole point is that git decides it must be witnessed
// deterministically or it is not witnessed at all.

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wiprecon"
	"github.com/anthony-chaudhary/fak/internal/wipref"
)

// wipAdoptCrashed builds the state every row below starts from: a session that
// checkpointed real work and then died, leaving the delta ONLY in refs/fak/wip/<session>.
// Wiping the tree is what makes the checkpoint reconcile RECLAIM — the delta applies
// forward again — and it is exactly the state #5998 says the fleet currently abandons.
func wipAdoptCrashed(t *testing.T, sess, content string) (string, string) {
	t.Helper()
	ctx := context.Background()
	dir, file := landTestRepo(t)
	if err := os.WriteFile(file, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, dir, sess, true, 1000); err != nil {
		t.Fatalf("checkpoint %s: %v", sess, err)
	}
	// The crash: the owner never committed, and the tree went back to baseline.
	if _, err := gitWipOut(ctx, dir, nil, "checkout", "--", "."); err != nil {
		t.Fatalf("wipe tree: %v", err)
	}
	return dir, file
}

// wipAdoptNoReceipt asserts a refusal left NO claim behind. A refusal that still wrote a
// receipt would pin a recoverable checkpoint under a successor that never touched it.
func wipAdoptNoReceipt(t *testing.T, dir, sess string) {
	t.Helper()
	_, has, err := wipCurrentOID(context.Background(), dir, wipAdoptRef(sess))
	if err != nil {
		t.Fatalf("read receipt ref: %v", err)
	}
	if has {
		t.Fatalf("a refused adoption left a claim on %s", sess)
	}
}

// wipAdoptRefPresent asserts the standing guarantee: adoption NEVER drops a checkpoint, on
// any path. Every row re-checks it, because the one bug that would make this whole verb a
// net loss is a recovery that destroys the thing it was recovering.
func wipAdoptRefPresent(t *testing.T, dir, sess string) {
	t.Helper()
	_, has, err := wipCurrentOID(context.Background(), dir, wipref.SessionRef(sess))
	if err != nil {
		t.Fatalf("read checkpoint ref: %v", err)
	}
	if !has {
		t.Fatalf("checkpoint %s was dropped; adoption must preserve it until a witnessed landing", sess)
	}
}

// TestWipAdoptClaimsACrashedCheckpointAndMaterializesOutsideTheTree is the base row: an
// abandoned checkpoint becomes a claimed, materialized, verified recovery — and the SHARED
// working tree is untouched, which is the property that let this verb exist at all on a
// trunk running concurrent agents.
func TestWipAdoptClaimsACrashedCheckpointAndMaterializesOutsideTheTree(t *testing.T) {
	ctx := context.Background()
	dir, file := wipAdoptCrashed(t, "crashed-1", "base line\nrescued work\n")

	res, code, err := wipAdoptRun(ctx, dir, wipAdoptOptions{
		Session: "crashed-1", Successor: "rescuer-a", Now: time.Unix(1_700_000_000, 0),
	})
	if err != nil || code != 0 {
		t.Fatalf("adopt rc=%d err=%v res=%+v", code, err, res)
	}
	if res.Verdict != string(wiprecon.AdoptGrant) {
		t.Fatalf("verdict = %q, want GRANT for an unclaimed RECLAIM checkpoint (%s)", res.Verdict, res.Reason)
	}
	if res.TargetKind != "patch" || !strings.HasSuffix(res.Target, "crashed-1.patch") {
		t.Fatalf("default materialization must be an explicit patch target, got %s %q", res.TargetKind, res.Target)
	}
	if res.Verified != 1 || len(res.Materialized) != 1 {
		t.Fatalf("materialization must verify its bytes: %+v", res)
	}

	// The claim is DURABLE and readable by another process: it is in git, not in memory.
	rec, _, has, err := wipReadReceipt(ctx, dir, "crashed-1")
	if err != nil || !has {
		t.Fatalf("receipt not durable: has=%v err=%v", has, err)
	}
	if rec.Successor != "rescuer-a" || rec.Phase != wiprecon.PhaseMaterialized {
		t.Fatalf("receipt = %+v, want rescuer-a at MATERIALIZED", rec)
	}
	if rec.CheckpointSHA != res.CheckpointSHA || rec.DeltaDigest == "" {
		t.Fatalf("receipt must bind ref+SHA+bytes: %+v", rec)
	}

	// The bytes went to the patch target, and the shared tree is exactly as it was found.
	patch, err := os.ReadFile(res.Target)
	if err != nil {
		t.Fatalf("read materialized patch: %v", err)
	}
	if !strings.Contains(string(patch), "rescued work") {
		t.Fatalf("patch target does not carry the checkpoint delta:\n%s", patch)
	}
	tree, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	if string(tree) != "base line\n" {
		t.Fatalf("adoption wrote into the shared working tree (%q); a stranger's delta must never land on peers' edits", tree)
	}
	wipAdoptRefPresent(t, dir, "crashed-1")
	if !res.Preserved {
		t.Fatalf("the preservation witness must be OBSERVED, not narrated: %+v", res)
	}
}

// TestWipAdoptLetsExactlyOneOfTwoSuccessorsWin is the two-reclaimer row. Both bids read
// "unclaimed"; the compare-and-swap decides. The loser must refuse having materialized
// NOTHING — a loser that had already written bytes would be the double-recovery this
// verb exists to prevent, merely reported after the fact.
func TestWipAdoptLetsExactlyOneOfTwoSuccessorsWin(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipAdoptCrashed(t, "contested", "base line\ncontested work\n")

	var winner wipAdoptResult
	var wcode int
	// The hook fires between this bid's receipt READ and its swap — the exact window in
	// which a concurrent successor can claim the checkpoint.
	loser, lcode, err := wipAdoptRun(ctx, dir, wipAdoptOptions{
		Session: "contested", Successor: "rescuer-slow", Now: time.Unix(1_700_000_000, 0),
		raceHook: func() {
			var rerr error
			winner, wcode, rerr = wipAdoptRun(ctx, dir, wipAdoptOptions{
				Session: "contested", Successor: "rescuer-fast", Now: time.Unix(1_700_000_000, 0),
			})
			if rerr != nil {
				t.Errorf("concurrent successor: %v", rerr)
			}
		},
	})
	if err != nil {
		t.Fatalf("losing bid errored instead of refusing: %v", err)
	}
	if wcode != 0 || winner.Verdict != string(wiprecon.AdoptGrant) {
		t.Fatalf("the first swap must win: rc=%d %+v", wcode, winner)
	}
	if lcode != 3 || loser.Verdict != wipReasonAdoptLostRace {
		t.Fatalf("the losing bid must refuse with %s (rc 3), got rc=%d %q: %s",
			wipReasonAdoptLostRace, lcode, loser.Verdict, loser.Reason)
	}
	if loser.Receipt != nil || len(loser.Materialized) != 0 || loser.Verified != 0 {
		t.Fatalf("the loser must mutate nothing: %+v", loser)
	}

	// Exactly ONE claim exists, and it names the winner.
	rec, _, has, err := wipReadReceipt(ctx, dir, "contested")
	if err != nil || !has {
		t.Fatalf("receipt: has=%v err=%v", has, err)
	}
	if rec.Successor != "rescuer-fast" || rec.Attempt != 1 {
		t.Fatalf("receipt = %+v, want a single attempt held by rescuer-fast", rec)
	}
	wipAdoptRefPresent(t, dir, "contested")
}

// TestWipAdoptResumesFromTheReceiptAfterACrashMidMaterialization is the row the receipt's
// journal-before-mutation ordering was bought for. The predecessor self is simulated
// exactly: a receipt at ADOPTED naming a target, plus a TORN half-written file at that
// target — the state a process leaves when it dies between the claim and the last byte.
//
// Resume must find its OWN target rather than picking a fresh one (picking a fresh one
// orphans the torn bytes and leaks a directory nobody will ever attribute), overwrite it,
// and verify the result against the checkpoint object.
func TestWipAdoptResumesFromTheReceiptAfterACrashMidMaterialization(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipAdoptCrashed(t, "torn", "base line\nhalf written\n")

	obj, has, err := wipCurrentOID(ctx, dir, wipref.SessionRef("torn"))
	if err != nil || !has {
		t.Fatalf("checkpoint object: has=%v err=%v", has, err)
	}
	want, digest, err := wipAdoptDelta(ctx, dir, obj)
	if err != nil {
		t.Fatal(err)
	}

	target := filepath.Join(t.TempDir(), "torn.patch")
	if err := os.WriteFile(target, []byte(want[:len(want)/2]), 0o644); err != nil {
		t.Fatal(err)
	}
	journaled, ok := wiprecon.ApplyAdopt(nil, wiprecon.AdoptRequest{
		Session: "torn", Action: wiprecon.ActReclaim, CheckpointRef: wipref.SessionRef("torn"),
		CheckpointSHA: obj, DeltaDigest: digest, Successor: "rescuer-a", Now: 1_700_000_000,
	}, wiprecon.AdoptGrant)
	if !ok {
		t.Fatal("ApplyAdopt refused a granted verdict")
	}
	journaled.Target = target
	if _, won, err := wipWriteReceipt(ctx, dir, journaled, "", false); err != nil || !won {
		t.Fatalf("journal the pre-crash claim: won=%v err=%v", won, err)
	}

	res, code, err := wipAdoptRun(ctx, dir, wipAdoptOptions{
		Session: "torn", Successor: "rescuer-a", Now: time.Unix(1_700_000_060, 0),
	})
	if err != nil || code != 0 {
		t.Fatalf("resume rc=%d err=%v res=%+v", code, err, res)
	}
	if res.Verdict != string(wiprecon.AdoptResume) {
		t.Fatalf("verdict = %q, want RESUME for the successor's own claim (%s)", res.Verdict, res.Reason)
	}
	if res.Target != target {
		t.Fatalf("resume materialized to %q, want its own recorded target %q", res.Target, target)
	}
	got, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != want {
		t.Fatalf("the torn write was inherited rather than repaired:\ngot:\n%s\nwant:\n%s", got, want)
	}

	rec, _, _, err := wipReadReceipt(ctx, dir, "torn")
	if err != nil {
		t.Fatal(err)
	}
	if rec.Phase != wiprecon.PhaseMaterialized || rec.Attempt != 2 {
		t.Fatalf("receipt = %+v, want MATERIALIZED on attempt 2", rec)
	}
	var resumed bool
	for _, ev := range rec.Audit {
		if ev.Event == wiprecon.EventResumed && ev.Actor == "rescuer-a" {
			resumed = true
		}
	}
	if !resumed {
		t.Fatalf("the resume must be auditable: %+v", rec.Audit)
	}
	wipAdoptRefPresent(t, dir, "torn")
}

// TestWipAdoptTakeoverNeedsBothALapsedClaimAndAGoneHolder pins the stale-adoption rule.
// Inside the TTL the second successor is HELD and writes nothing; past it — with the
// holder provably leaseless — it takes over, and the takeover names who it displaced.
// A takeover that is not auditable is indistinguishable from a lost claim.
func TestWipAdoptTakeoverNeedsBothALapsedClaimAndAGoneHolder(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipAdoptCrashed(t, "lapsing", "base line\nstale claim\n")
	t0 := time.Unix(1_700_000_000, 0)

	first, code, err := wipAdoptRun(ctx, dir, wipAdoptOptions{
		Session: "lapsing", Successor: "peer-a", TTL: 60, Now: t0,
	})
	if err != nil || code != 0 {
		t.Fatalf("first adoption rc=%d err=%v res=%+v", code, err, first)
	}

	held, hcode, err := wipAdoptRun(ctx, dir, wipAdoptOptions{
		Session: "lapsing", Successor: "rescuer-b", TTL: 60, Now: t0.Add(30 * time.Second),
	})
	if err != nil {
		t.Fatalf("held bid errored instead of refusing: %v", err)
	}
	if hcode != 3 || held.Verdict != string(wiprecon.AdoptHeld) {
		t.Fatalf("a claim inside its TTL must HOLD (rc 3), got rc=%d %q: %s", hcode, held.Verdict, held.Reason)
	}
	if mid, _, _, rerr := wipReadReceipt(ctx, dir, "lapsing"); rerr != nil || mid.Successor != "peer-a" {
		t.Fatalf("a held bid must not move the claim: %+v (%v)", mid, rerr)
	}

	took, tcode, err := wipAdoptRun(ctx, dir, wipAdoptOptions{
		Session: "lapsing", Successor: "rescuer-b", TTL: 60, Now: t0.Add(120 * time.Second),
	})
	if err != nil || tcode != 0 {
		t.Fatalf("takeover rc=%d err=%v res=%+v", tcode, err, took)
	}
	if took.Verdict != string(wiprecon.AdoptTakeover) {
		t.Fatalf("verdict = %q, want TAKEOVER once the claim lapsed with no live holder (%s)", took.Verdict, took.Reason)
	}
	rec, _, _, err := wipReadReceipt(ctx, dir, "lapsing")
	if err != nil {
		t.Fatal(err)
	}
	if rec.Successor != "rescuer-b" || rec.Attempt != 2 {
		t.Fatalf("receipt = %+v, want rescuer-b on attempt 2", rec)
	}
	var displaced string
	for _, ev := range rec.Audit {
		if ev.Event == wiprecon.EventTakeover {
			displaced = ev.From
		}
	}
	if displaced != "peer-a" {
		t.Fatalf("the takeover must name whom it displaced, audit=%+v", rec.Audit)
	}
	wipAdoptRefPresent(t, dir, "lapsing")
}

// TestWipAdoptLandsThroughTheExistingLanderAndStaysIdempotent closes the loop: an adopted
// checkpoint reaches a real commit through wip land's existing scope and divergence
// refusals (adoption adds no second way to write to the trunk), the checkpoint ref SURVIVES
// the landing, and re-running the whole recovery is a benign no-op rather than a second
// landing — which is what makes a crash after the commit but before the receipt update safe.
func TestWipAdoptLandsThroughTheExistingLanderAndStaysIdempotent(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipAdoptCrashed(t, "landme", "base line\nlanded by a successor\n")

	res, code, err := wipAdoptRun(ctx, dir, wipAdoptOptions{
		Session: "landme", Successor: "rescuer-a", Land: true, Now: time.Unix(1_700_000_000, 0),
	})
	if err != nil || code != 0 {
		t.Fatalf("adopt --land rc=%d err=%v res=%+v", code, err, res)
	}
	if !res.Landed || res.LandedSHA == "" {
		t.Fatalf("expected a witnessed landing: %+v", res)
	}
	head, err := gitWipOut(ctx, dir, nil, "show", "HEAD:note.txt")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(head, "landed by a successor") {
		t.Fatalf("HEAD:note.txt = %q, want the adopted delta", head)
	}
	rec, _, _, err := wipReadReceipt(ctx, dir, "landme")
	if err != nil {
		t.Fatal(err)
	}
	if rec.Phase != wiprecon.PhaseLanded || rec.LandedSHA != res.LandedSHA {
		t.Fatalf("receipt = %+v, want LANDED bound to %s", rec, res.LandedSHA)
	}
	// The checkpoint outlives its own landing: `fak wip reap` clears it only once the
	// delta is provably in HEAD, so a landing that is later found wanting is still backed.
	wipAdoptRefPresent(t, dir, "landme")

	again, acode, err := wipAdoptRun(ctx, dir, wipAdoptOptions{
		Session: "landme", Successor: "rescuer-a", Land: true, Now: time.Unix(1_700_000_300, 0),
	})
	if err != nil || acode != 0 {
		t.Fatalf("re-running a finished recovery must be a no-op, rc=%d err=%v", acode, err)
	}
	if again.Verdict != string(wiprecon.AdoptSettled) || again.Landed {
		t.Fatalf("verdict = %q (landed=%v), want ALREADY_LANDED and no second commit", again.Verdict, again.Landed)
	}
}

// TestWipAdoptRefusesAQuarantinedCheckpoint is the fail-safe row. QUARANTINE is an
// operator's call, and it must stay one whether it is reached through the dispatcher, the
// queue, or a hand-typed adopt — so the refusal lives in the classifier adoption routes
// through, not in a caller-side check that a future caller could forget.
func TestWipAdoptRefusesAQuarantinedCheckpoint(t *testing.T) {
	ctx := context.Background()
	dir, _ := landTestRepo(t)
	// An untracked file, checkpointed and LEFT on disk: its delta creates a path that
	// already exists, so `git apply --check` refuses it and the verdict is QUARANTINE.
	if err := os.WriteFile(filepath.Join(dir, "orphan.txt"), []byte("untracked\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, dir, "quarantined", true, 1000); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}

	res, code, err := wipAdoptRun(ctx, dir, wipAdoptOptions{
		Session: "quarantined", Successor: "rescuer-a", Now: time.Unix(1_700_000_000, 0),
	})
	if err != nil {
		t.Fatalf("a refusal must not be an error: %v", err)
	}
	if code != 3 || res.Verdict != string(wiprecon.AdoptRefused) {
		t.Fatalf("rc=%d verdict=%q, want NOT_RECLAIMABLE (rc 3): %s", code, res.Verdict, res.Reason)
	}
	if res.Target != "" || len(res.Materialized) != 0 {
		t.Fatalf("a refused adoption must materialize nothing: %+v", res)
	}
	wipAdoptNoReceipt(t, dir, "quarantined")
	wipAdoptRefPresent(t, dir, "quarantined")
}

// TestWipAdoptRefusesATargetInsideTheSharedWorkingTree: the whole reason `wip reconcile`
// stayed advisory is that materializing a crashed stranger's delta into a shared tree lands
// on live peers' edits. Adoption may only write somewhere isolated, so a target that
// resolves inside the tree is a refusal — before the claim is journaled, so a mistyped
// --into does not leave a checkpoint pinned under a successor that wrote nothing.
func TestWipAdoptRefusesATargetInsideTheSharedWorkingTree(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipAdoptCrashed(t, "intree", "base line\nkeep out\n")

	for _, tc := range []struct {
		name string
		opts wipAdoptOptions
	}{
		{"worker dir", wipAdoptOptions{Into: filepath.Join(dir, "recovered")}},
		{"patch file", wipAdoptOptions{PatchOut: filepath.Join(dir, "recovered.patch")}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts := tc.opts
			opts.Session, opts.Successor, opts.Now = "intree", "rescuer-a", time.Unix(1_700_000_000, 0)
			res, code, err := wipAdoptRun(ctx, dir, opts)
			if err != nil {
				t.Fatalf("a refusal must not be an error: %v", err)
			}
			if code != 3 || res.Verdict != wipReasonAdoptTargetInTree {
				t.Fatalf("rc=%d verdict=%q, want %s (rc 3): %s", code, res.Verdict, wipReasonAdoptTargetInTree, res.Reason)
			}
			wipAdoptNoReceipt(t, dir, "intree")
		})
	}
}

// TestWipAdoptMaterializesIntoAnIsolatedWorkerPathWithVerifiedBytes covers the other
// sanctioned target: an isolated worker directory whose contents are READ BACK and hashed
// against what git holds. "os.WriteFile returned nil" is not the same claim as "the
// successor's copy is the checkpoint", and only the second one survives a torn resume.
func TestWipAdoptMaterializesIntoAnIsolatedWorkerPathWithVerifiedBytes(t *testing.T) {
	ctx := context.Background()
	dir, file := wipAdoptCrashed(t, "worker1", "base line\nworker copy\n")
	into := filepath.Join(t.TempDir(), "adopted")

	res, code, err := wipAdoptRun(ctx, dir, wipAdoptOptions{
		Session: "worker1", Successor: "rescuer-a", Into: into, Now: time.Unix(1_700_000_000, 0),
	})
	if err != nil || code != 0 {
		t.Fatalf("adopt --into rc=%d err=%v res=%+v", code, err, res)
	}
	if res.TargetKind != "worker" || res.Verified != 1 {
		t.Fatalf("expected one verified file in an isolated worker path: %+v", res)
	}
	got, err := os.ReadFile(filepath.Join(into, "note.txt"))
	if err != nil {
		t.Fatalf("read materialized worker copy: %v", err)
	}
	if string(got) != "base line\nworker copy\n" {
		t.Fatalf("worker copy = %q, want the checkpointed post-image", got)
	}
	tree, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	if string(tree) != "base line\n" {
		t.Fatalf("the shared working tree was mutated: %q", tree)
	}
	wipAdoptRefPresent(t, dir, "worker1")
}

// TestWipAdoptRefusesAnAbsentCheckpointAndReportsAnAbsentReceipt: "there is nothing here"
// is an ANSWER, and both halves report it as a checkable token with exit 3 rather than as a
// runtime error a scheduler would have to parse prose to distinguish from a broken repo.
func TestWipAdoptRefusesAnAbsentCheckpointAndReportsAnAbsentReceipt(t *testing.T) {
	ctx := context.Background()
	dir, _ := landTestRepo(t)

	res, code, err := wipAdoptRun(ctx, dir, wipAdoptOptions{
		Session: "ghost", Successor: "rescuer-a", Now: time.Unix(1_700_000_000, 0),
	})
	if err != nil {
		t.Fatalf("a missing checkpoint must not be an error: %v", err)
	}
	if code != 3 || res.Verdict != wipReasonAdoptNoCheckpoint {
		t.Fatalf("rc=%d verdict=%q, want %s (rc 3)", code, res.Verdict, wipReasonAdoptNoCheckpoint)
	}
	// The preservation guarantee is a READ, never a constant: a session with no checkpoint
	// must not be reported as one whose checkpoint was preserved.
	if res.Preserved {
		t.Fatalf("a session with no checkpoint reported checkpoint_preserved=true: %+v", res)
	}

	var out, errb bytes.Buffer
	if rc := runWipAdoptReceipt(ctx, &out, &errb, dir, "ghost", false); rc != 3 {
		t.Fatalf("`receipt` on an unclaimed session: rc=%d, want 3\nstdout:%s\nstderr:%s", rc, out.String(), errb.String())
	}
	if !strings.Contains(out.String(), "no adoption receipt") {
		t.Fatalf("receipt output must say so plainly: %q", out.String())
	}
}

// TestWipReconcileAdoptSubVerbIsReachableAsTheQueuePrintsIt is the anti-drift row. The
// recovery queue prints `fak wip reconcile adopt <session>`; this runs exactly that string
// through the real CLI entrypoint. The bug it exists to prevent already happened once — the
// lifecycle queue printed `fak wip land <id> --apply`, a flag that never existed.
func TestWipReconcileAdoptSubVerbIsReachableAsTheQueuePrintsIt(t *testing.T) {
	dir, _ := wipAdoptCrashed(t, "viacli", "base line\ncli recovery\n")

	argv := wiprecon.AdoptArgv(wiprecon.ReclaimRow{Session: "viacli"})
	if len(argv) < 3 || argv[0] != "wip" || argv[1] != "reconcile" {
		t.Fatalf("queue argv = %v, want a `wip reconcile <verb> <session>` command", argv)
	}
	var out, errb bytes.Buffer
	// argv[2:] is the printed sub-verb and session; -C and --successor are the only
	// additions the test needs to run it against a throwaway repo.
	rc := runWipReconcile(&out, &errb, append([]string{argv[2], "-C", dir, "--successor", "cli-rescuer"}, argv[3:]...))
	if rc != 0 {
		t.Fatalf("the command the queue prints must run: rc=%d\nstdout:%s\nstderr:%s", rc, out.String(), errb.String())
	}
	if !strings.Contains(out.String(), string(wiprecon.AdoptGrant)) {
		t.Fatalf("stdout = %q, want a GRANT", out.String())
	}

	out.Reset()
	if rc := runWipReconcile(&out, &errb, []string{"receipt", "-C", dir, "viacli"}); rc != 0 {
		t.Fatalf("`receipt` rc=%d\nstderr:%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "cli-rescuer") {
		t.Fatalf("receipt must name the holder: %q", out.String())
	}
}

// TestWipReconcileDispatchAdoptsOnlyTheHeadUnclaimedRow is the lifecycle-queue integration
// row: the opt-in dispatcher claims ONE row, never lands it, never quarantines anything,
// never drops a ref — and the queue then reads back its own claim, so the next tick offers
// `resume` to its owner instead of `adopt` to the whole fleet.
func TestWipReconcileDispatchAdoptsOnlyTheHeadUnclaimedRow(t *testing.T) {
	ctx := context.Background()
	t.Setenv("CLAUDE_CODE_SESSION_ID", "dispatcher-1")
	dir, _ := wipReclaimFixture(t)

	var out, errb bytes.Buffer
	if rc := runWipReconcile(&out, &errb, []string{"-C", dir, "--reclaim", "--dispatch"}); rc != 3 {
		t.Fatalf("rc=%d, want 3 (the queue is non-empty)\nstdout:%s\nstderr:%s", rc, out.String(), errb.String())
	}
	if !strings.Contains(out.String(), "dispatch: alpha") {
		t.Fatalf("the dispatcher must act on the head row: %q", out.String())
	}

	rec, _, has, err := wipReadReceipt(ctx, dir, "alpha")
	if err != nil || !has || rec.Successor != "dispatcher-1" {
		t.Fatalf("head row not claimed: has=%v rec=%+v err=%v", has, rec, err)
	}
	// One row per tick, and never a QUARANTINE row on any tick.
	for _, sess := range []string{"bravo", "charlie"} {
		if _, has, err := wipCurrentOID(ctx, dir, wipAdoptRef(sess)); err != nil || has {
			t.Fatalf("dispatch claimed more than the head row (%s): has=%v err=%v", sess, has, err)
		}
	}
	for _, sess := range []string{"alpha", "bravo", "charlie"} {
		wipAdoptRefPresent(t, dir, sess)
	}

	// The queue reads its own claim back: the owner is named, and the printed command
	// becomes `resume` — the difference between continuing a recovery and starting a race.
	out.Reset()
	if rc := runWipReconcile(&out, &errb, []string{"-C", dir, "--reclaim"}); rc != 3 {
		t.Fatalf("rc=%d, want 3\nstderr:%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "self:dispatcher-1") {
		t.Fatalf("queue must show the caller's own claim: %q", out.String())
	}
	if !strings.Contains(out.String(), "fak wip reconcile resume alpha") {
		t.Fatalf("a row this session holds must be offered as a resume: %q", out.String())
	}
}
