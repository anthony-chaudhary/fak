package workerworktree

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The warm-pool contract (#3572), stated as the four acceptance witnesses:
//
//  1. a RETURNED worktree is LEASED by the next Prepare — Reused=true, pooled path,
//     and NO `worktree add`;
//  2. Reap returns-to-pool up to the cap and force-removes past it, counted in
//     `worktree add` / `worktree remove` calls across a lease/return cycle;
//  3. the pool disabled (=0, and unset) reproduces today's create/reap exactly;
//  4. every miss path still falls back to create/reap.
//
// All of it runs against fakeGit, so no real repo, checkout or `git worktree` is
// involved — the counts ARE the proof, since the whole point of the pool is which git
// commands it does and does not run.

// poolMember materializes a worktree directory the fake git will "own", and returns
// its path. The pool indexes directories on disk, so a member has to actually exist.
func poolMember(t *testing.T, wtRoot, lane, key string) string {
	t.Helper()
	wt := Path(lane, key, wtRoot)
	if err := os.MkdirAll(wt, 0o755); err != nil {
		t.Fatalf("could not materialize pool member: %v", err)
	}
	return wt
}

func countPrefix(g *fakeGit, prefix ...string) int {
	return len(g.callsWithPrefix(prefix...))
}

// ---- 1. Prepare leases a returned worktree ------------------------------------ //

func TestPrepareLeasesReturnedWorktreeWithoutWorktreeAdd(t *testing.T) {
	t.Setenv(PoolCapEnv, "2")
	wtRoot := t.TempDir()

	// Worker A gets a fresh worktree (cold pool: one add), then finishes and reaps.
	g := newFakeGit().reply("worktree", 0, "").reply("reset", 0, "").reply("clean", 0, "")
	a := Prepare("/r", "tools", "111", "base1", wtRoot, g.run)
	if !a.OK || a.Reused {
		t.Fatalf("cold pool must create, not reuse: %+v", a)
	}
	// fakeGit does not create the dir the real `worktree add` would; materialize it so
	// the returned member is indexable.
	poolMember(t, wtRoot, "tools", "111")
	ret := Reap("/r", a.Path, g.run)
	if !ret.OK || ret.Removed {
		t.Fatalf("reap under cap must RETURN (ok, not removed): %+v", ret)
	}
	if !strings.Contains(ret.Reason, "warm worktree pool") {
		t.Fatalf("return should name the pool: %q", ret.Reason)
	}

	// Worker B is a DIFFERENT key — the pre-#3572 same-key reuse could never serve it.
	g2 := newFakeGit().reply("worktree", 0, "").reply("reset", 0, "").reply("clean", 0, "")
	b := Prepare("/r", "tools", "222", "base2", wtRoot, g2.run)
	if !b.OK {
		t.Fatalf("pooled prepare failed: %+v", b)
	}
	if !b.Reused {
		t.Fatalf("a pool hit must report Reused=true: %+v", b)
	}
	if !samePath(b.Path, a.Path) {
		t.Fatalf("lease must hand back the pooled path %q, got %q", a.Path, b.Path)
	}
	if n := countPrefix(g2, "worktree", "add"); n != 0 {
		t.Fatalf("a pool hit must not `worktree add`, got %d: %v", n, g2.calls)
	}
	// The lease re-points the warm tree at the NEW base rather than checking out afresh.
	resets := g2.callsWithPrefix("reset", "--hard")
	if len(resets) != 1 || resets[0][len(resets[0])-1] != "base2" {
		t.Fatalf("lease must reset --hard to the new base, got %v", g2.calls)
	}
	if len(g2.callsWithPrefix("clean", "-fd")) != 1 {
		t.Fatalf("lease must clean the leased tree, got %v", g2.calls)
	}
	if b.BaseSHA != "base2" {
		t.Fatalf("leased result must carry the new base, got %q", b.BaseSHA)
	}
}

func TestPrepareLeaseIsConsumedOnce(t *testing.T) {
	t.Setenv(PoolCapEnv, "2")
	wtRoot := t.TempDir()
	wt := poolMember(t, wtRoot, "tools", "111")
	g := newFakeGit().reply("worktree", 0, "").reply("reset", 0, "").reply("clean", 0, "")
	if res, ok := returnPooled("/r", wt, 2, g.run); !ok || !res.OK {
		t.Fatalf("seed return failed: %+v", res)
	}

	g2 := newFakeGit().reply("worktree", 0, "").reply("reset", 0, "").reply("clean", 0, "")
	first := Prepare("/r", "tools", "222", "b", wtRoot, g2.run)
	second := Prepare("/r", "tools", "333", "b", wtRoot, g2.run)
	if !first.Reused {
		t.Fatalf("first prepare should hit the pool: %+v", first)
	}
	if second.Reused {
		t.Fatalf("one idle member must serve exactly one lease, second: %+v", second)
	}
	if n := countPrefix(g2, "worktree", "add"); n != 1 {
		t.Fatalf("second prepare must fall back to one add, got %d: %v", n, g2.calls)
	}
}

func TestPrepareDoesNotLeaseAnotherLanesMember(t *testing.T) {
	t.Setenv(PoolCapEnv, "2")
	wtRoot := t.TempDir()
	wt := poolMember(t, wtRoot, "gateway", "111")
	g := newFakeGit().reply("worktree", 0, "").reply("reset", 0, "").reply("clean", 0, "")
	if _, ok := returnPooled("/r", wt, 2, g.run); !ok {
		t.Fatal("seed return failed")
	}

	g2 := newFakeGit().reply("worktree", 0, "").reply("reset", 0, "").reply("clean", 0, "")
	res := Prepare("/r", "tools", "222", "b", wtRoot, g2.run)
	if res.Reused {
		t.Fatalf("the pool is keyed by LANE: a gateway member must not serve tools: %+v", res)
	}
	if countPrefix(g2, "worktree", "add") != 1 {
		t.Fatalf("cross-lane miss must fall back to add: %v", g2.calls)
	}
}

// ---- 2. Reap returns up to the cap, force-removes past it --------------------- //

func TestReapReturnsToPoolUpToCapThenForceRemoves(t *testing.T) {
	t.Setenv(PoolCapEnv, "2")
	wtRoot := t.TempDir()
	g := newFakeGit().reply("worktree", 0, "").reply("reset", 0, "").reply("clean", 0, "")

	var members []string
	for _, key := range []string{"1", "2", "3"} {
		members = append(members, poolMember(t, wtRoot, "tools", key))
	}
	for i, wt := range members {
		res := Reap("/r", wt, g.run)
		if !res.OK {
			t.Fatalf("reap %d failed: %+v", i, res)
		}
		if i < 2 && res.Removed {
			t.Fatalf("reap %d is under the cap of 2 and must be RETURNED: %+v", i, res)
		}
		if i == 2 && !res.Removed {
			t.Fatalf("reap %d overflows the cap of 2 and must be force-removed: %+v", i, res)
		}
	}
	removes := g.callsWithPrefix("worktree", "remove")
	if len(removes) != 1 {
		t.Fatalf("cap 2 over 3 reaps must force-remove exactly the 1 overflow, got %d: %v", len(removes), g.calls)
	}
	if !contains(removes[0], "--force") || removes[0][len(removes[0])-1] != members[2] {
		t.Fatalf("the overflow member must be the removed one, got %v", removes[0])
	}

	// And the two retained members are exactly what the next two prepares lease — the
	// add/remove ledger across the full lease/return cycle: 0 adds, 1 remove.
	g2 := newFakeGit().reply("worktree", 0, "").reply("reset", 0, "").reply("clean", 0, "")
	for i, key := range []string{"90", "91"} {
		res := Prepare("/r", "tools", key, "b", wtRoot, g2.run)
		if !res.Reused {
			t.Fatalf("prepare %d should have leased a retained member: %+v", i, res)
		}
	}
	if n := countPrefix(g2, "worktree", "add"); n != 0 {
		t.Fatalf("both retained members must be leased with no add, got %d: %v", n, g2.calls)
	}
}

func TestReapReturnIsIdempotent(t *testing.T) {
	t.Setenv(PoolCapEnv, "1")
	wtRoot := t.TempDir()
	wt := poolMember(t, wtRoot, "tools", "1")
	g := newFakeGit().reply("worktree", 0, "").reply("reset", 0, "").reply("clean", 0, "")
	if res := Reap("/r", wt, g.run); res.Removed {
		t.Fatalf("first reap must return, not remove: %+v", res)
	}
	// The witness sweep is best-effort and can fire twice on one worktree. The second
	// reap must not count the member against its OWN cap and destroy what it parked.
	if res := Reap("/r", wt, g.run); res.Removed {
		t.Fatalf("re-reaping an already-idle member must stay a return: %+v", res)
	}
	if n := countPrefix(g, "worktree", "remove"); n != 0 {
		t.Fatalf("idempotent return must never force-remove, got %d: %v", n, g.calls)
	}
}

func TestReapStillRefusesNonWorkerPathWithPoolOn(t *testing.T) {
	t.Setenv(PoolCapEnv, "4")
	g := newFakeGit()
	if res := Reap("/r", filepath.FromSlash("/work/fak"), g.run); res.OK {
		t.Fatalf("the marker guardrail must outrank the pool: %+v", res)
	}
	if len(g.calls) != 0 {
		t.Fatalf("must never touch git for a non-worker path, got %v", g.calls)
	}
}

// ---- 3. Disabled reproduces today's create/reap -------------------------------- //

func TestPoolDisabledReproducesCreateAndReap(t *testing.T) {
	for _, val := range []string{"0", "", "nonsense", "-3"} {
		t.Run("cap="+val, func(t *testing.T) {
			// Register the restore first, THEN unset: t.Setenv's cleanup puts the
			// original value back either way, so the "unset" case cannot leak.
			t.Setenv(PoolCapEnv, val)
			if val == "" {
				os.Unsetenv(PoolCapEnv)
			}
			wtRoot := t.TempDir()
			g := newFakeGit().reply("rev-parse", 0, "feedface\n").reply("worktree", 0, "")

			res := Prepare("/r", "tools", "1", "", wtRoot, g.run)
			if !res.OK || res.Reused {
				t.Fatalf("disabled pool must create fresh: %+v", res)
			}
			adds := g.callsWithPrefix("worktree", "add")
			if len(adds) != 1 || !contains(adds[0], "--detach") || adds[0][len(adds[0])-1] != "feedface" {
				t.Fatalf("disabled pool must `worktree add --detach <sha>`, got %v", g.calls)
			}

			wt := poolMember(t, wtRoot, "tools", "1")
			rr := Reap("/r", wt, g.run)
			if !rr.OK || !rr.Removed {
				t.Fatalf("disabled pool must force-remove: %+v", rr)
			}
			removes := g.callsWithPrefix("worktree", "remove")
			if len(removes) != 1 || !contains(removes[0], "--force") {
				t.Fatalf("want exactly one forced remove, got %v", g.calls)
			}
			// No reset/clean, and no sidecar state: byte-for-byte the pre-#3572 path.
			if len(g.callsWithPrefix("reset")) != 0 || len(g.callsWithPrefix("clean")) != 0 {
				t.Fatalf("disabled pool must not reset/clean anything, got %v", g.calls)
			}
			if _, err := os.Stat(poolStatePath(wtRoot)); !os.IsNotExist(err) {
				t.Fatalf("disabled pool must not write pool state under %s", wtRoot)
			}
		})
	}
}

func TestPoolCapParsing(t *testing.T) {
	cases := map[string]int{"": 0, "  ": 0, "0": 0, "1": 1, "12": 12, "-1": 0, "x": 0, " 3 ": 3}
	for in, want := range cases {
		t.Setenv(PoolCapEnv, in)
		if in == "" {
			os.Unsetenv(PoolCapEnv)
		}
		if got := PoolCap(); got != want {
			t.Fatalf("PoolCap(%q) = %d, want %d", in, got, want)
		}
	}
}

// ---- 4. Every miss path falls back ------------------------------------------- //

func TestLeaseFallsBackWhenMemberWillNotReset(t *testing.T) {
	t.Setenv(PoolCapEnv, "2")
	wtRoot := t.TempDir()
	wt := poolMember(t, wtRoot, "tools", "1")
	seed := newFakeGit().reply("reset", 0, "").reply("clean", 0, "")
	if _, ok := returnPooled("/r", wt, 2, seed.run); !ok {
		t.Fatal("seed return failed")
	}

	// A member whose reset fails is unusable: destroyed, and the prepare creates fresh.
	g := newFakeGit().reply("reset", 128, "fatal: not a git repository").reply("worktree", 0, "")
	res := Prepare("/r", "tools", "2", "b", wtRoot, g.run)
	if !res.OK || res.Reused {
		t.Fatalf("an unusable member must fall back to a fresh create: %+v", res)
	}
	if countPrefix(g, "worktree", "add") != 1 {
		t.Fatalf("want one fallback add, got %v", g.calls)
	}
	if n := countPrefix(g, "worktree", "remove"); n != 1 {
		t.Fatalf("the unusable member must be force-removed, got %d: %v", n, g.calls)
	}
	if _, err := os.Stat(poolMarker(wtRoot, filepath.Base(wt))); !os.IsNotExist(err) {
		t.Fatal("a failed lease must not leave the member marked idle")
	}
}

func TestReturnFallsBackToRemoveWhenResetFails(t *testing.T) {
	t.Setenv(PoolCapEnv, "4")
	wtRoot := t.TempDir()
	wt := poolMember(t, wtRoot, "tools", "1")
	g := newFakeGit().reply("reset", 1, "boom").reply("worktree", 0, "")
	res := Reap("/r", wt, g.run)
	if !res.OK || !res.Removed {
		t.Fatalf("a worktree that will not park clean must be force-removed: %+v", res)
	}
	if _, err := os.Stat(poolMarker(wtRoot, filepath.Base(wt))); !os.IsNotExist(err) {
		t.Fatal("a failed return must not mark the member idle")
	}
}

// A member whose directory vanished (the cold sweep reclaimed it) must release its pool
// slot, or the cap starves after the first sweep and the pool degrades to zero.
func TestVanishedMemberReleasesItsSlot(t *testing.T) {
	t.Setenv(PoolCapEnv, "1")
	wtRoot := t.TempDir()
	gone := Path("tools", "1", wtRoot)
	if err := os.MkdirAll(poolStatePath(wtRoot), 0o755); err != nil {
		t.Fatal(err)
	}
	marker := poolMarker(wtRoot, filepath.Base(gone))
	if err := os.WriteFile(marker, []byte("idle\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	live := poolMember(t, wtRoot, "tools", "2")
	g := newFakeGit().reply("reset", 0, "").reply("clean", 0, "").reply("worktree", 0, "")
	if res := Reap("/r", live, g.run); res.Removed {
		t.Fatalf("the vanished member holds no slot, so this return fits the cap of 1: %+v", res)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatal("a stale marker should be cleared when its worktree is gone")
	}
}

// Same-lane+key reuse (the pre-#3572 behavior) must still win before any pool lease:
// it is the cheaper hit and it hands the worker back its OWN tree, WIP included.
func TestSameKeyReuseStillPrecedesPoolLease(t *testing.T) {
	t.Setenv(PoolCapEnv, "2")
	wtRoot := t.TempDir()
	own := poolMember(t, wtRoot, "tools", "1")
	other := poolMember(t, wtRoot, "tools", "2")
	seed := newFakeGit().reply("reset", 0, "").reply("clean", 0, "")
	if _, ok := returnPooled("/r", other, 2, seed.run); !ok {
		t.Fatal("seed return failed")
	}

	g := newFakeGit().reply("worktree", 0, "worktree "+own+"\n")
	res := Prepare("/r", "tools", "1", "b", wtRoot, g.run)
	if !res.OK || !res.Reused || !samePath(res.Path, own) {
		t.Fatalf("same-key reuse must win and return the worker's own tree: %+v", res)
	}
	if len(g.callsWithPrefix("reset")) != 0 {
		t.Fatalf("same-key reuse must not reset the worker's own tree: %v", g.calls)
	}
}
