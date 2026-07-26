package main

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipref"
)

// Fixture identity for the stamped half of the scale corpus. The session id is
// deliberately NOT derivable from any ref name, so a decoded stamp is distinguishable
// from the ref-name fallback at a glance.
const (
	wipScaleStampSession = "scale-stamped-owner"
	wipScaleStampAt      = 1750000000
)

// TestWipListRecordsSpawnsAreConstantInRefCount is the scaling gate of #5336: listing
// the checkpoint namespace must cost a CONSTANT number of git subprocesses no matter
// how many refs it holds.
//
// The regression it fences is the one that made the reconciliation spine unrunnable:
// wipListRecords used to spawn `git log -1 --format=%B` per ref, so at the 4051 local
// refs a week of fleet work accumulated, `fak wip status | reconcile | attribute`
// fanned out to thousands of children and timed out past 120s — orphaned WIP was
// never reconciled. The fold to a single `git for-each-ref ...%(contents)` fixed it;
// this test is what keeps it fixed, because the cost is invisible in a 4-ref test and
// a reviewer reading the code cannot see a spawn count.
//
// It asserts O(1) rather than "fast": a wall-clock bound would be flaky on a loaded
// CI box, while the spawn count is exact and machine-checkable. Sweeping N over three
// orders of magnitude is what makes the claim CONSTANT-in-N rather than a single fast
// data point — one spawn at N=1 and one at N=5000 cannot both hold under any per-ref
// fan-out. The setup builds its N refs in ONE `git update-ref --stdin` batch, so the
// fixture itself is O(1) in spawns too and N=5000 stays quick.
//
// The test is deliberately NOT t.Parallel: gitWipSpawns is a process-global counter,
// and Go resumes parallel tests only after the sequential ones finish, so the
// before/after delta measured here is this call's alone.
func TestWipListRecordsSpawnsAreConstantInRefCount(t *testing.T) {
	for _, n := range []int{1, 8, 512, 5000} {
		t.Run(fmt.Sprintf("refs=%d", n), func(t *testing.T) {
			ctx := context.Background()
			dir, _ := wipTestRepo(t)
			stamped, plain := wipScaleFixtureCommits(ctx, t, dir)
			wipScaleCreateRefs(ctx, t, dir, n, stamped, plain)

			before := gitWipSpawns.Load()
			start := time.Now()
			recs, err := wipListRecords(ctx, dir)
			elapsed := time.Since(start)
			spawns := gitWipSpawns.Load() - before
			if err != nil {
				t.Fatalf("wipListRecords over %d refs: %v", n, err)
			}
			if spawns != 1 {
				t.Errorf("listing %d refs cost %d git spawns, want exactly 1 — a per-ref fan-out is back", n, spawns)
			}
			if len(recs) != n {
				t.Fatalf("listed %d records over %d refs, want %d", len(recs), n, n)
			}
			t.Logf("%d refs: %d git spawn(s) in %s", n, spawns, elapsed.Round(time.Millisecond))

			// Every record must survive the batch intact: the stamped half decoded from
			// the commit contents, the unstamped half labelled from its ref name. A
			// listing that scales but drops or garbles records at volume is not a fix.
			var decoded, fallback int
			for _, r := range recs {
				switch r.Object {
				case stamped:
					if r.Stamp.SessionID != wipScaleStampSession || r.Stamp.CheckpointedAt != wipScaleStampAt {
						t.Fatalf("%s: stamp not decoded at N=%d: %+v", r.Ref, n, r.Stamp)
					}
					decoded++
				case plain:
					// No parseable stamp: the record is labelled from its ref name and
					// carries no checkpoint metadata — listed, never silently dropped.
					if r.Stamp.SessionID != wipref.SessionFromRef(r.Ref) || r.Stamp.StartSHA != "" || r.Stamp.CheckpointedAt != 0 {
						t.Fatalf("%s: want ref-name fallback at N=%d, got %+v", r.Ref, n, r.Stamp)
					}
					fallback++
				default:
					t.Fatalf("%s: unexpected object %s at N=%d", r.Ref, r.Object, n)
				}
			}
			if decoded == 0 || (n > 1 && fallback == 0) {
				t.Fatalf("corpus not exercised at N=%d: %d decoded, %d fallback", n, decoded, fallback)
			}
		})
	}
}

// wipScaleFixtureCommits builds the two checkpoint objects the scale corpus points at:
// one carrying a stamp on a NON-subject line (the case only full-contents parsing
// decodes) and one carrying no stamp at all (the ref-name fallback). Two objects are
// enough — the invariant under test is the listing's spawn count in the number of
// REFS, and a commit-tree per ref would itself be an O(N) fan-out in the fixture.
func wipScaleFixtureCommits(ctx context.Context, t *testing.T, dir string) (stamped, plain string) {
	t.Helper()
	head, err := gitWipOut(ctx, dir, nil, "rev-parse", "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	tree, err := gitWipOut(ctx, dir, nil, "rev-parse", "HEAD^{tree}")
	if err != nil {
		t.Fatal(err)
	}
	line, err := wipref.EncodeStamp(wipref.Stamp{
		SessionID: wipScaleStampSession, StartSHA: head, Buildable: true, CheckpointedAt: wipScaleStampAt,
	})
	if err != nil {
		t.Fatal(err)
	}
	if stamped, err = gitWipOut(ctx, dir, nil, "commit-tree", tree, "-p", head,
		"-m", "wip checkpoint\n\n"+line+"\n"); err != nil {
		t.Fatalf("commit-tree (stamped): %v", err)
	}
	if plain, err = gitWipOut(ctx, dir, nil, "commit-tree", tree, "-p", head,
		"-m", "no stamp here"); err != nil {
		t.Fatalf("commit-tree (unstamped): %v", err)
	}
	return stamped, plain
}

// wipScaleCreateRefs creates n checkpoint refs — alternating the stamped and unstamped
// objects — in a SINGLE `git update-ref --stdin` batch.
func wipScaleCreateRefs(ctx context.Context, t *testing.T, dir string, n int, stamped, plain string) {
	t.Helper()
	var batch strings.Builder
	for i := 0; i < n; i++ {
		obj := plain
		if i%2 == 0 {
			obj = stamped
		}
		fmt.Fprintf(&batch, "create %s %s\n", wipref.SessionRef(fmt.Sprintf("scale-%05d", i)), obj)
	}
	_, errStr, code, err := gitWipStdin(ctx, dir, batch.String(), "update-ref", "--stdin")
	if err != nil {
		t.Fatalf("git update-ref --stdin (%d refs): %v", n, err)
	}
	if code != 0 {
		t.Fatalf("git update-ref --stdin (%d refs) exited %d: %s", n, code, strings.TrimSpace(errStr))
	}
}
