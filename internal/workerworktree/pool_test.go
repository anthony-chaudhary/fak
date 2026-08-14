package workerworktree

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func materializePoolPath(t *testing.T, wt string) {
	t.Helper()
	if err := os.MkdirAll(wt, 0o755); err != nil {
		t.Fatalf("materialize worktree %s: %v", wt, err)
	}
}

func stampedPoolMember(t *testing.T, wtRoot, lane, key string) string {
	t.Helper()
	wt := Path(lane, key, wtRoot)
	materializePoolPath(t, wt)
	if err := writeOwnerStamp(wt, OwnerStamp{
		PID:       1001,
		LeaseID:   "resolve-" + canonicalPoolLane(lane),
		CreatedAt: time.Date(2026, time.August, 14, 12, 0, 0, 0, time.UTC),
	}); err != nil {
		t.Fatalf("owner stamp %s: %v", wt, err)
	}
	return wt
}

func cleanPoolGit() *fakeGit {
	return newFakeGit().
		reply("worktree", 0, "").
		reply("status", 0, "").
		reply("rev-parse", 0, "pooled-head\n").
		reply("merge-base", 0, "").
		reply("reset", 0, "").
		reply("clean", 0, "")
}

func countPrefix(g *fakeGit, prefix ...string) int {
	return len(g.callsWithPrefix(prefix...))
}

func mustPoolState(t *testing.T, wt, state string) poolMemberMetadata {
	t.Helper()
	meta, err := readPoolMember(wt)
	if err != nil {
		t.Fatalf("read pool member %s: %v", wt, err)
	}
	if meta.State != state {
		t.Fatalf("pool member %s state=%q, want %q: %+v", wt, meta.State, state, meta)
	}
	return meta
}

// The end-to-end acceptance witness: one cold add, one safe return, and one new
// worker lease. The warm lease resets/cleans the returned path and never adds.
func TestPoolLeaseReturnLeaseAvoidsSecondWorktreeAdd(t *testing.T) {
	t.Setenv(PoolCapEnv, "2")
	wtRoot := t.TempDir()

	firstGit := cleanPoolGit()
	first := Prepare("/repo", "tools", "worker-a", "base-a", wtRoot, firstGit.run)
	if !first.OK || first.Reused {
		t.Fatalf("first prepare must create: %+v", first)
	}
	materializePoolPath(t, first.Path) // fake `worktree add` does not create it.
	if countPrefix(firstGit, "worktree", "add") != 1 {
		t.Fatalf("cold prepare add count=%d, calls=%v", countPrefix(firstGit, "worktree", "add"), firstGit.calls)
	}
	mustPoolState(t, first.Path, poolStateLeased)

	returned := Reap("/repo", first.Path, firstGit.run)
	if !returned.OK || returned.Removed {
		t.Fatalf("clean worktree should return to pool: %+v", returned)
	}
	mustPoolState(t, first.Path, poolStateIdle)
	if countPrefix(firstGit, "worktree", "remove") != 0 {
		t.Fatalf("under-cap return removed a worktree: %v", firstGit.calls)
	}
	if countPrefix(firstGit, "reset", "--hard") != 1 || countPrefix(firstGit, "clean", "-fd") != 1 {
		t.Fatalf("return must normalize once, calls=%v", firstGit.calls)
	}

	secondGit := cleanPoolGit()
	owner := OwnerStamp{
		PID:       2002,
		LeaseID:   "resolve-tools-worker-b",
		CreatedAt: time.Date(2026, time.August, 14, 12, 5, 0, 0, time.UTC),
	}
	second := PrepareOwned("/repo", "tools", "worker-b", "base-b", wtRoot, secondGit.run, owner)
	if !second.OK || !second.Reused {
		t.Fatalf("warm prepare must reuse: %+v", second)
	}
	if !samePath(second.Path, first.Path) {
		t.Fatalf("warm prepare path=%q, want returned %q", second.Path, first.Path)
	}
	if countPrefix(secondGit, "worktree", "add") != 0 {
		t.Fatalf("warm prepare called worktree add: %v", secondGit.calls)
	}
	if countPrefix(secondGit, "reset", "--hard") != 1 || countPrefix(secondGit, "clean", "-fd") != 1 {
		t.Fatalf("warm lease must reset+clean once: %v", secondGit.calls)
	}
	resets := secondGit.callsWithPrefix("reset", "--hard")
	if resets[0][len(resets[0])-1] != "base-b" {
		t.Fatalf("warm lease reset=%v, want base-b", resets[0])
	}
	meta := mustPoolState(t, second.Path, poolStateLeased)
	if meta.Owner.PID != owner.PID || meta.Owner.LeaseID != owner.LeaseID {
		t.Fatalf("leased metadata owner=%+v, want %+v", meta.Owner, owner)
	}
}

func TestPoolCapOverflowForceRemovesOnlyOverflow(t *testing.T) {
	t.Setenv(PoolCapEnv, "2")
	wtRoot := t.TempDir()
	g := cleanPoolGit()

	members := []string{
		stampedPoolMember(t, wtRoot, "tools", "one"),
		stampedPoolMember(t, wtRoot, "tools", "two"),
		stampedPoolMember(t, wtRoot, "tools", "three"),
	}
	for i, wt := range members {
		res := Reap("/repo", wt, g.run)
		if !res.OK {
			t.Fatalf("reap %d: %+v", i, res)
		}
		if i < 2 && res.Removed {
			t.Fatalf("under-cap member %d removed: %+v", i, res)
		}
		if i == 2 && !res.Removed {
			t.Fatalf("overflow member was retained: %+v", res)
		}
	}
	if got := countPrefix(g, "worktree", "remove"); got != 1 {
		t.Fatalf("remove count=%d, want 1: %v", got, g.calls)
	}
	if got := countPrefix(g, "reset", "--hard"); got != 2 {
		t.Fatalf("reset count=%d, want only two retained members: %v", got, g.calls)
	}
	if got := countPrefix(g, "clean", "-fd"); got != 2 {
		t.Fatalf("clean count=%d, want only two retained members: %v", got, g.calls)
	}
}

func TestPoolDisabledReproducesCreateAndReapExactly(t *testing.T) {
	t.Setenv(PoolCapEnv, "0")
	wtRoot := t.TempDir()
	g := cleanPoolGit()

	res := Prepare("/repo", "tools", "disabled", "base", wtRoot, g.run)
	if !res.OK || res.Reused {
		t.Fatalf("disabled prepare: %+v", res)
	}
	materializePoolPath(t, res.Path)
	reaped := Reap("/repo", res.Path, g.run)
	if !reaped.OK || !reaped.Removed {
		t.Fatalf("disabled reap: %+v", reaped)
	}
	if countPrefix(g, "worktree", "add") != 1 || countPrefix(g, "worktree", "remove") != 1 {
		t.Fatalf("disabled add/remove ledger=%v", g.calls)
	}
	if countPrefix(g, "status") != 0 ||
		countPrefix(g, "merge-base") != 0 ||
		countPrefix(g, "reset") != 0 ||
		countPrefix(g, "clean") != 0 {
		t.Fatalf("disabled mode must not execute pool probes/transitions: %v", g.calls)
	}
	if _, err := os.Stat(poolMemberPath(res.Path)); !os.IsNotExist(err) {
		t.Fatalf("disabled mode wrote pool metadata: %v", err)
	}
}

func TestPoolCapDefaultSmallAndExplicitZeroDisables(t *testing.T) {
	cases := []struct {
		value string
		unset bool
		want  int
	}{
		{value: "", unset: true, want: defaultPoolCap},
		{value: " ", want: defaultPoolCap},
		{value: "garbage", want: defaultPoolCap},
		{value: "-1", want: defaultPoolCap},
		{value: "0", want: 0},
		{value: "1", want: 1},
		{value: " 4 ", want: 4},
	}
	for _, tc := range cases {
		t.Run("value="+tc.value, func(t *testing.T) {
			t.Setenv(PoolCapEnv, tc.value)
			if tc.unset {
				_ = os.Unsetenv(PoolCapEnv)
			}
			if got := PoolCap(); got != tc.want {
				t.Fatalf("PoolCap(%q)=%d, want %d", tc.value, got, tc.want)
			}
		})
	}
	if defaultPoolCap <= 0 || defaultPoolCap > 4 {
		t.Fatalf("defaultPoolCap=%d, want a small enabled bound", defaultPoolCap)
	}
}

func TestDirtyReturnFallsBackWithoutResetOrClean(t *testing.T) {
	t.Setenv(PoolCapEnv, "2")
	wt := stampedPoolMember(t, t.TempDir(), "tools", "dirty")
	g := cleanPoolGit().reply("status", 0, " M internal/workerworktree/pool.go\n")

	res := Reap("/repo", wt, g.run)
	if !res.OK || !res.Removed {
		t.Fatalf("dirty return must use forced-removal fallback: %+v", res)
	}
	if !strings.Contains(res.Reason, "working_tree_dirty") {
		t.Fatalf("dirty fallback reason=%q", res.Reason)
	}
	if countPrefix(g, "reset") != 0 || countPrefix(g, "clean") != 0 {
		t.Fatalf("dirty member was silently reset/cleaned: %v", g.calls)
	}
	if countPrefix(g, "worktree", "remove") != 1 {
		t.Fatalf("dirty fallback remove ledger=%v", g.calls)
	}
}

func TestUnpushedReturnFallsBackWithoutResetOrClean(t *testing.T) {
	t.Setenv(PoolCapEnv, "2")
	wt := stampedPoolMember(t, t.TempDir(), "tools", "unpushed")
	g := cleanPoolGit().reply("merge-base", 1, "")

	res := Reap("/repo", wt, g.run)
	if !res.OK || !res.Removed {
		t.Fatalf("unpushed return must use forced-removal fallback: %+v", res)
	}
	if !strings.Contains(res.Reason, "unpushed_commit") {
		t.Fatalf("unpushed fallback reason=%q", res.Reason)
	}
	if countPrefix(g, "reset") != 0 || countPrefix(g, "clean") != 0 {
		t.Fatalf("unpushed member was silently reset/cleaned: %v", g.calls)
	}
}

func TestPoolIsKeyedByLane(t *testing.T) {
	t.Setenv(PoolCapEnv, "2")
	wtRoot := t.TempDir()
	g := cleanPoolGit()
	gateway := stampedPoolMember(t, wtRoot, "gateway", "one")
	if res := Reap("/repo", gateway, g.run); !res.OK || res.Removed {
		t.Fatalf("seed gateway idle: %+v", res)
	}

	prepareGit := cleanPoolGit()
	res := Prepare("/repo", "tools", "two", "base", wtRoot, prepareGit.run)
	if !res.OK || res.Reused {
		t.Fatalf("tools prepare must not consume gateway member: %+v", res)
	}
	if countPrefix(prepareGit, "worktree", "add") != 1 {
		t.Fatalf("cross-lane miss must add: %v", prepareGit.calls)
	}
}

func TestPoolLaneLockMakesConcurrentCapAccountingExact(t *testing.T) {
	t.Setenv(PoolCapEnv, "1")
	wtRoot := t.TempDir()
	one := stampedPoolMember(t, wtRoot, "tools", "one")
	two := stampedPoolMember(t, wtRoot, "tools", "two")

	var mu sync.Mutex
	var calls [][]string
	runner := func(_ string, args []string) (int, string) {
		mu.Lock()
		calls = append(calls, append([]string{}, args...))
		mu.Unlock()
		switch args[0] {
		case "status":
			return 0, ""
		case "rev-parse":
			return 0, "head\n"
		case "merge-base":
			return 0, ""
		default:
			return 0, ""
		}
	}

	var wg sync.WaitGroup
	results := make(chan Result, 2)
	for _, wt := range []string{one, two} {
		wg.Add(1)
		go func(path string) {
			defer wg.Done()
			results <- Reap("/repo", path, runner)
		}(wt)
	}
	wg.Wait()
	close(results)

	retained, removed := 0, 0
	for res := range results {
		if !res.OK {
			t.Fatalf("concurrent reap failed: %+v", res)
		}
		if res.Removed {
			removed++
		} else {
			retained++
		}
	}
	if retained != 1 || removed != 1 {
		t.Fatalf("cap=1 concurrent results retained=%d removed=%d", retained, removed)
	}
	idle := poolIdleMembers(wtRoot, "tools")
	if len(idle) != 1 {
		t.Fatalf("idle members=%v, want exactly one", idle)
	}
	mu.Lock()
	defer mu.Unlock()
	removeCalls, resetCalls, cleanCalls := 0, 0, 0
	for _, call := range calls {
		if len(call) >= 2 && call[0] == "worktree" && call[1] == "remove" {
			removeCalls++
		}
		if len(call) >= 1 && call[0] == "reset" {
			resetCalls++
		}
		if len(call) >= 1 && call[0] == "clean" {
			cleanCalls++
		}
	}
	if removeCalls != 1 || resetCalls != 1 || cleanCalls != 1 {
		t.Fatalf("concurrent call ledger remove=%d reset=%d clean=%d calls=%v",
			removeCalls, resetCalls, cleanCalls, calls)
	}
}

func TestPoolMetadataReplacementLeavesNoTemporaryRecord(t *testing.T) {
	t.Setenv(PoolCapEnv, "2")
	wt := stampedPoolMember(t, t.TempDir(), "tools", "atomic")
	owner, err := readOwnerStamp(wt)
	if err != nil {
		t.Fatal(err)
	}
	if err := recordPoolLease(wt, "tools", owner); err != nil {
		t.Fatal(err)
	}
	if _, ok, why := returnPooled("/repo", wt, 2, cleanPoolGit().run); !ok {
		t.Fatalf("return: %s", why)
	}
	matches, err := filepath.Glob(poolMemberPath(wt) + ".tmp-*")
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 0 {
		t.Fatalf("atomic replacement left temp files: %v", matches)
	}
	mustPoolState(t, wt, poolStateIdle)
}

func TestLeaseFailureRemovesBrokenMemberThenFallsBackToAdd(t *testing.T) {
	t.Setenv(PoolCapEnv, "2")
	wtRoot := t.TempDir()
	wt := stampedPoolMember(t, wtRoot, "tools", "broken")
	if _, ok, why := returnPooled("/repo", wt, 2, cleanPoolGit().run); !ok {
		t.Fatalf("seed return: %s", why)
	}

	g := cleanPoolGit().reply("reset", 128, "not a worktree")
	res := Prepare("/repo", "tools", "new-worker", "base", wtRoot, g.run)
	if !res.OK || res.Reused {
		t.Fatalf("broken pooled member must fall back to create: %+v", res)
	}
	if countPrefix(g, "worktree", "remove") != 1 || countPrefix(g, "worktree", "add") != 1 {
		t.Fatalf("broken-member fallback ledger=%v", g.calls)
	}
	if _, err := os.Stat(poolMemberPath(wt)); !os.IsNotExist(err) {
		t.Fatalf("broken member metadata survived forced removal: %v", err)
	}
}
