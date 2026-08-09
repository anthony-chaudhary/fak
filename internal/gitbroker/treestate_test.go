package gitbroker

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// THE TESTS THAT DECIDE WHETHER THE CLASS B CACHE IS SAFE (#5623).
//
// Two mechanisms land in this rung and they are proven — and measured —
// SEPARATELY on purpose:
//
//   - SINGLE-FLIGHT collapses concurrent identical queries into one execution.
//     It stores nothing, so it needs no invalidation argument at all. It is
//     exercised here on keys that CANNOT be cached, so every saved execution is
//     attributable to coalescing and nothing else.
//   - THE CLASS B CACHE reuses a working-tree answer across calls, which is only
//     safe if a peer's write provably busts it. That is the negative test below,
//     against real git, with a real commit and a real bare index write.
//
// Bundled into one measurement neither contribution would be attributable and a
// later regression could not be localized to the mechanism that caused it.

// testRaceWindow is the filesystem-granularity window these tests run the broker
// with. It is far below DefaultTreeRaceWindow because the guard it feeds is about
// mtime resolution, and the filesystems this suite runs on resolve in
// nanoseconds; a 2s production default would just make every test sleep 2s.
const testRaceWindow = 25 * time.Millisecond

// settle waits until a just-finished git write is outside testRaceWindow, so the
// next sample is eligible for the cache. Without it a test would be measuring the
// unsettled path and would pass for the wrong reason.
func settle() { time.Sleep(8 * testRaceWindow) }

// joinWindow is how long a gated leader is held open so its fellow callers can
// arrive and join the flight. It is generous by design: too short and the test
// reports MORE executions than expected, which fails loudly rather than passing
// quietly, so the failure mode of this constant is honest.
const joinWindow = 500 * time.Millisecond

// treeBackend is a TreeRunner whose answers are fixed, whose calls are counted,
// and which can be held open — the three things needed to prove that a second
// read did NOT reach git, and that N concurrent reads reached it exactly once.
type treeBackend struct {
	mu      sync.Mutex
	calls   int
	status  string
	entered chan struct{} // one token per entry; nil to disable
	hold    chan struct{} // if non-nil, every call blocks until closed
}

func (b *treeBackend) TreeState(ctx context.Context) (TreeState, error) {
	b.mu.Lock()
	b.calls++
	status := b.status
	entered, hold := b.entered, b.hold
	b.mu.Unlock()

	if entered != nil {
		select {
		case entered <- struct{}{}:
		default:
		}
	}
	if hold != nil {
		select {
		case <-hold:
		case <-ctx.Done():
			return TreeState{}, ctx.Err()
		}
	}
	return TreeState{Dirty: strings.TrimSpace(status) != "", Status: status}, nil
}

func (b *treeBackend) count() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.calls
}

func (b *treeBackend) setStatus(s string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.status = s
}

// objectBackend is the Runner counterpart: counted, and holdable so concurrent
// callers are provably concurrent rather than hopefully concurrent.
type objectBackend struct {
	mu      sync.Mutex
	calls   int
	obj     Object
	delay   time.Duration
	entered chan struct{}
	hold    chan struct{}
}

func (b *objectBackend) Object(ctx context.Context, rev string) (Object, error) {
	b.mu.Lock()
	b.calls++
	obj, delay := b.obj, b.delay
	entered, hold := b.entered, b.hold
	b.mu.Unlock()

	if entered != nil {
		select {
		case entered <- struct{}{}:
		default:
		}
	}
	if hold != nil {
		select {
		case <-hold:
		case <-ctx.Done():
			return Object{}, ctx.Err()
		}
	}
	if delay > 0 {
		time.Sleep(delay)
	}
	return obj, nil
}

func (b *objectBackend) count() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.calls
}

// newTestRepo builds a REAL git repository with one commit, and returns a runner
// for further git commands against it.
//
// The invalidation claim is about what git actually does to `.git/index` and the
// refs when a peer commits. A fake cannot witness that, so the negative tests use
// the real binary.
func newTestRepo(tb testing.TB) (string, func(args ...string) string) {
	tb.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		tb.Skip("git not on PATH")
	}
	repo, err := os.MkdirTemp("", "gbtree")
	if err != nil {
		tb.Fatalf("temp repo: %v", err)
	}
	tb.Cleanup(func() { _ = os.RemoveAll(repo) })

	git := func(args ...string) string {
		// Identity and signing are pinned inline: this repo has no user config of
		// its own, and a host with commit.gpgsign=true globally would otherwise
		// fail the commit for reasons unrelated to the broker.
		full := append([]string{
			"-C", repo,
			"-c", "user.name=gitbroker test",
			"-c", "user.email=gitbroker@example.invalid",
			"-c", "commit.gpgsign=false",
		}, args...)
		out, err := exec.Command("git", full...).CombinedOutput()
		if err != nil {
			tb.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git("init", "-q")
	writeFile(tb, filepath.Join(repo, "seed.txt"), "seed\n")
	git("add", "seed.txt")
	git("commit", "-q", "-m", "seed")
	return repo, git
}

func writeFile(tb testing.TB, path, body string) {
	tb.Helper()
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		tb.Fatalf("write %s: %v", path, err)
	}
}

// synthGitDir builds the minimum `.git` a StateKey needs — an index, a resolvable
// (detached) HEAD, and a refs/ directory — and back-dates it so the tree is
// already settled.
//
// It exists for the benchmarks and for the pure key-shape tests, where spawning
// real git would measure git rather than the broker. Nothing that makes a claim
// about INVALIDATION uses it; those use newTestRepo.
func synthGitDir(tb testing.TB, root, oid string) {
	tb.Helper()
	gitDir := filepath.Join(root, ".git")
	if err := os.MkdirAll(filepath.Join(gitDir, "refs", "heads"), 0o755); err != nil {
		tb.Fatalf("mkdir .git: %v", err)
	}
	writeFile(tb, filepath.Join(gitDir, "index"), "not a real index, but it has an mtime and a size\n")
	writeFile(tb, filepath.Join(gitDir, "HEAD"), oid+"\n")
	past := time.Now().Add(-time.Hour)
	for _, p := range []string{
		filepath.Join(gitDir, "index"),
		filepath.Join(gitDir, "refs"),
	} {
		if err := os.Chtimes(p, past, past); err != nil {
			tb.Fatalf("back-date %s: %v", p, err)
		}
	}
}

// treeBroker starts a broker over repo with a counted backend and returns a
// client wired to a SEPARATE fallback backend, so a silent fail-open can never be
// mistaken for a broker answer.
func treeBroker(tb testing.TB, repo string, backend *treeBackend) (*Server, *Client, *treeBackend) {
	tb.Helper()
	dir := rendezvousDir(tb)
	srv, err := Serve(Config{RepoRoot: repo, Dir: dir, Tree: backend, TreeRaceWindow: testRaceWindow})
	if err != nil {
		tb.Fatalf("Serve: %v", err)
	}
	tb.Cleanup(func() { _ = srv.Close() })
	fallback := &treeBackend{status: "SPAWNED-NOT-BROKER\n"}
	c := &Client{RepoRoot: repo, Dir: dir, TreeRunner: fallback, Timeout: 5 * time.Second}
	return srv, c, fallback
}

// TestAPeerCommitBustsTheWorkingTreeCache is THE negative test of this rung, and
// the one the acceptance gate of #5623 says decides whether the cache is safe at
// all.
//
// It is negative in the sense that matters: it establishes a real cached window
// (a second read provably served from memory, git untouched), then has a PEER
// commit into the same checkout mid-window — which is what happens continuously
// in this shared tree — and requires the very next read to MISS. A positive test
// showing that caching works would prove nothing about correctness here.
func TestAPeerCommitBustsTheWorkingTreeCache(t *testing.T) {
	repo, git := newTestRepo(t)
	backend := &treeBackend{status: " M seed.txt\n"}
	srv, c, fallback := treeBroker(t, repo, backend)
	ctx := context.Background()
	settle()

	first, err := c.Tree(ctx, ClassB)
	if err != nil {
		t.Fatalf("first read: %v", err)
	}
	if first.Provenance != Broker {
		t.Fatalf("first read provenance = %q, want %q (a cold cache must reach the backend)", first.Provenance, Broker)
	}
	second, err := c.Tree(ctx, ClassB)
	if err != nil {
		t.Fatalf("second read: %v", err)
	}
	if second.Provenance != Cache {
		t.Fatalf("second read provenance = %q, want %q — without a real cached window this test cannot say anything about invalidation", second.Provenance, Cache)
	}
	if got := backend.count(); got != 1 {
		t.Fatalf("backend ran %d times for two reads, want 1", got)
	}

	// A PEER COMMITS into the same checkout. This rewrites `.git/index` and moves
	// the branch ref — the two halves of the key most likely to matter.
	writeFile(t, filepath.Join(repo, "peer.txt"), "a peer's work\n")
	git("add", "peer.txt")
	git("commit", "-q", "-m", "peer commit")
	// A DIFFERENT answer, so a stale hit is visible in the bytes and not only in
	// the provenance tag.
	backend.setStatus("?? after-the-peer-commit.txt\n")
	settle()

	third, err := c.Tree(ctx, ClassB)
	if err != nil {
		t.Fatalf("post-commit read: %v", err)
	}
	if third.Provenance != Cache && third.Provenance != Broker {
		t.Fatalf("post-commit provenance = %q", third.Provenance)
	}
	if third.Provenance == Cache {
		t.Fatalf("a peer's commit did NOT bust the cache: the read after it was still served from memory. Every consumer of working-tree state would now be looking at a tree that no longer exists")
	}
	if got := backend.count(); got != 2 {
		t.Fatalf("backend ran %d times, want 2 — the post-commit read must recompute", got)
	}
	if third.Status != "?? after-the-peer-commit.txt\n" {
		t.Fatalf("post-commit status = %q, want the NEW answer — the reader was served the pre-commit state", third.Status)
	}
	if third.Key == first.Key {
		t.Fatalf("the invalidation key did not move across a peer commit: %+v. The key is the entire cache-validity test, so a key that survives a commit is a cache that serves stale answers", third.Key)
	}
	if got := fallback.count(); got != 0 {
		t.Fatalf("the fallback runner ran %d times; a live broker must not be bypassed, or this test proves nothing about the broker", got)
	}
	if st := srv.Stats(); st.TreeHits != 1 || st.TreeMisses != 2 {
		t.Fatalf("stats = %+v, want 1 tree hit and 2 tree misses", st)
	}
}

// TestABareIndexWriteBustsTheCacheWithHEADUnmoved is the sharper half of the
// invalidation claim. A peer that runs `git add` and has not committed yet has
// changed what `git status` answers while HEAD sits exactly where it was — so a
// key that leaned on the commit OID would happily serve the pre-add answer.
func TestABareIndexWriteBustsTheCacheWithHEADUnmoved(t *testing.T) {
	repo, git := newTestRepo(t)
	backend := &treeBackend{status: "?? staged.txt\n"}
	_, c, _ := treeBroker(t, repo, backend)
	ctx := context.Background()
	settle()

	headBefore := git("rev-parse", "HEAD")
	warm, err := c.Tree(ctx, ClassB)
	if err != nil {
		t.Fatalf("warm read: %v", err)
	}
	hit, err := c.Tree(ctx, ClassB)
	if err != nil {
		t.Fatalf("cached read: %v", err)
	}
	if warm.Provenance != Broker || hit.Provenance != Cache {
		t.Fatalf("provenance sequence = %q then %q, want %q then %q", warm.Provenance, hit.Provenance, Broker, Cache)
	}

	// The peer stages, and does NOT commit.
	writeFile(t, filepath.Join(repo, "staged.txt"), "staged by a peer\n")
	git("add", "staged.txt")
	backend.setStatus("A  staged.txt\n")
	settle()

	after, err := c.Tree(ctx, ClassB)
	if err != nil {
		t.Fatalf("post-add read: %v", err)
	}
	if headAfter := git("rev-parse", "HEAD"); headAfter != headBefore {
		t.Fatalf("HEAD moved (%s -> %s); this test only says something if it does NOT", headBefore, headAfter)
	}
	if after.Key.HeadOID != warm.Key.HeadOID {
		t.Fatalf("the key's HEAD OID moved (%s -> %s); this test only says something if it does NOT", warm.Key.HeadOID, after.Key.HeadOID)
	}
	if after.Provenance == Cache {
		t.Fatalf("a bare `git add` did NOT bust the cache. The index is what `git status` reads, so an index write that the key cannot see is exactly the stale read this cache promised not to serve")
	}
	if got := backend.count(); got != 2 {
		t.Fatalf("backend ran %d times, want 2", got)
	}
	if after.Status != "A  staged.txt\n" {
		t.Fatalf("post-add status = %q, want the NEW answer", after.Status)
	}
}

// TestDecisionalQueriesAreNeverServedFromTheCache is the correctness line stated
// behaviourally: even with a warm, valid entry sitting right there, a Class C
// caller gets a fresh execution. classc_scan_test.go enforces the same rule
// structurally, so a future path that never gets a test still cannot break it.
func TestDecisionalQueriesAreNeverServedFromTheCache(t *testing.T) {
	repo, _ := newTestRepo(t)
	backend := &treeBackend{status: " M seed.txt\n"}
	srv, c, _ := treeBroker(t, repo, backend)
	ctx := context.Background()
	settle()

	if _, err := c.Tree(ctx, ClassB); err != nil {
		t.Fatalf("warm: %v", err)
	}
	hit, err := c.Tree(ctx, ClassB)
	if err != nil {
		t.Fatalf("hit: %v", err)
	}
	if hit.Provenance != Cache {
		t.Fatalf("could not establish a warm entry: provenance = %q", hit.Provenance)
	}
	warmCalls := backend.count()

	// Every spelling a decisional caller can arrive with, including the one that
	// forgot to say — an older client, or a caller that never thought about it.
	for i, class := range []Class{ClassC, "", "unknown-to-this-broker"} {
		res, err := c.Tree(ctx, class)
		if err != nil {
			t.Fatalf("decisional read %d (class %q): %v", i, class, err)
		}
		if res.Provenance != Broker {
			t.Fatalf("decisional read %d (class %q) provenance = %q, want %q — a query that feeds a commit gate, a mutation, or a refusal is computed fresh, permanently", i, class, res.Provenance, Broker)
		}
		if got, want := backend.count(), warmCalls+i+1; got != want {
			t.Fatalf("after decisional read %d (class %q) the backend had run %d times, want %d — it was answered from something instead of git", i, class, got, want)
		}
	}
	st := srv.Stats()
	if st.TreeHits != 1 {
		t.Fatalf("tree cache hits = %d, want 1 — a decisional read was served from the cache", st.TreeHits)
	}
	if st.TreeFresh != 3 {
		t.Fatalf("tree fresh reads = %d, want 3", st.TreeFresh)
	}
	if !st.TreeEntry {
		t.Fatalf("the warm entry is gone: a decisional read must not disturb the Class B entry, only decline to use it")
	}
}

// TestDecisionalQueriesAreNeverCoalesced closes the other half of the same rule.
// Single-flight is not a cache, but a joiner still observes a snapshot taken when
// the LEADER started — which is exactly the staleness a decisional caller is not
// allowed to inherit. All N must reach git.
func TestDecisionalQueriesAreNeverCoalesced(t *testing.T) {
	const callers = 6
	repo, _ := newTestRepo(t)
	backend := &treeBackend{
		status:  " M seed.txt\n",
		entered: make(chan struct{}, callers),
		hold:    make(chan struct{}),
	}
	srv, c, _ := treeBroker(t, repo, backend)
	ctx := context.Background()
	settle()

	var wg sync.WaitGroup
	provenances := make([]Provenance, callers)
	errs := make([]error, callers)
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			res, err := c.Tree(ctx, ClassC)
			provenances[i], errs[i] = res.Provenance, err
		}(i)
	}
	// Every caller must land in git ON ITS OWN. If any had coalesced, fewer than
	// `callers` tokens would ever arrive and this receive would block until the
	// test's own deadline — the failure is a timeout naming this line.
	for i := 0; i < callers; i++ {
		select {
		case <-backend.entered:
		case <-time.After(10 * time.Second):
			t.Fatalf("only %d of %d decisional callers reached the backend; the rest joined someone else's execution, which is the staleness Class C is defined to refuse", i, callers)
		}
	}
	close(backend.hold)
	wg.Wait()

	for i := range errs {
		if errs[i] != nil {
			t.Fatalf("decisional caller %d: %v", i, errs[i])
		}
		if provenances[i] != Broker {
			t.Fatalf("decisional caller %d provenance = %q, want %q", i, provenances[i], Broker)
		}
	}
	if got := backend.count(); got != callers {
		t.Fatalf("backend ran %d times for %d decisional callers, want %d", got, callers, callers)
	}
	if st := srv.Stats(); st.Coalesced != 0 {
		t.Fatalf("coalesced = %d, want 0 — a decisional query joined an in-flight execution", st.Coalesced)
	}
}

// TestSingleFlightCollapsesConcurrentIdenticalQueries measures COALESCING ALONE.
//
// The key is `HEAD`, which IsOID rejects, so no cache can hold it and no cache
// can serve it: every execution avoided here is attributable to single-flight and
// to nothing else. That separation is what #5623's acceptance gate asks for, and
// it is asserted at the end rather than merely intended.
func TestSingleFlightCollapsesConcurrentIdenticalQueries(t *testing.T) {
	const callers = 8
	dir := rendezvousDir(t)
	root := filepath.Join(dir, "repo")
	backend := &objectBackend{
		obj:     blob(oidA, "computed once, handed to everyone"),
		entered: make(chan struct{}, callers),
		hold:    make(chan struct{}),
	}
	srv, err := Serve(Config{RepoRoot: root, Dir: dir, Runner: backend})
	if err != nil {
		t.Fatalf("Serve: %v", err)
	}
	defer srv.Close()
	fallback := newFakeRunner(map[string]Object{"HEAD": blob(oidA, "SPAWNED-NOT-BROKER")})
	c := &Client{RepoRoot: root, Dir: dir, Runner: fallback, Timeout: 10 * time.Second}

	var wg sync.WaitGroup
	results := make([]Result, callers)
	errs := make([]error, callers)
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i], errs[i] = c.Object(context.Background(), "HEAD")
		}(i)
	}
	select {
	case <-backend.entered:
	case <-time.After(10 * time.Second):
		t.Fatal("no caller reached the backend")
	}
	// Hold the leader open long enough for the others to arrive and join. If one
	// misses the window it starts its own execution, and the count below reports
	// MORE work than expected — this constant can only cause a loud failure, never
	// a quiet pass.
	time.Sleep(joinWindow)
	close(backend.hold)
	wg.Wait()

	for i := range errs {
		if errs[i] != nil {
			t.Fatalf("caller %d: %v", i, errs[i])
		}
		if results[i].Provenance != Broker {
			t.Fatalf("caller %d provenance = %q, want %q", i, results[i].Provenance, Broker)
		}
		if string(results[i].Data) != string(results[0].Data) {
			t.Fatalf("caller %d saw %q, caller 0 saw %q — coalesced callers must share one answer", i, results[i].Data, results[0].Data)
		}
	}
	if got := backend.count(); got != 1 {
		t.Fatalf("backend ran %d times for %d concurrent identical queries, want 1 — single-flight did not collapse them", got, callers)
	}
	if got := fallback.count("HEAD"); got != 0 {
		t.Fatalf("the fallback runner ran %d times; a live broker must not be bypassed", got)
	}
	st := srv.Stats()
	if st.Coalesced != callers-1 {
		t.Fatalf("coalesced = %d, want %d — the single-flight saving must be countable on its own", st.Coalesced, callers-1)
	}
	if st.Hits != 0 || st.Entries != 0 {
		t.Fatalf("stats = %+v, want a cold empty cache: this measurement is single-flight's alone, and a cache contribution would make it unattributable", st)
	}
}

// TestTheWorkingTreeCacheIsMeasuredApartFromSingleFlight is the mirror image, and
// together with the test above it is what makes each mechanism's contribution
// attributable. Here the reads are SERIAL, so no two are ever in flight at once
// and single-flight cannot save a thing: every avoided execution is the cache's.
func TestTheWorkingTreeCacheIsMeasuredApartFromSingleFlight(t *testing.T) {
	const reads = 5
	repo, _ := newTestRepo(t)
	backend := &treeBackend{status: " M seed.txt\n"}
	srv, c, _ := treeBroker(t, repo, backend)
	ctx := context.Background()
	settle()

	for i := 0; i < reads; i++ {
		res, err := c.Tree(ctx, ClassB)
		if err != nil {
			t.Fatalf("read %d: %v", i, err)
		}
		want := Cache
		if i == 0 {
			want = Broker
		}
		if res.Provenance != want {
			t.Fatalf("read %d provenance = %q, want %q", i, res.Provenance, want)
		}
	}
	if got := backend.count(); got != 1 {
		t.Fatalf("backend ran %d times for %d serial reads, want 1", got, reads)
	}
	st := srv.Stats()
	if st.TreeHits != reads-1 || st.TreeMisses != 1 {
		t.Fatalf("stats = %+v, want %d tree hits and 1 miss", st, reads-1)
	}
	if st.Coalesced != 0 {
		t.Fatalf("coalesced = %d, want 0: nothing was concurrent here, so a non-zero count would mean the cache's saving is being credited to single-flight", st.Coalesced)
	}
}

// TestAJustWrittenTreeIsNeverCached pins the one residual hazard the stated
// stale-read budget names: filesystem mtime granularity. If a peer rewrote the
// index to the same size within one tick of our sample, the key would not move
// and the cache would serve a tree that had already changed. settledFor closes
// that by refusing to STORE a sample taken too soon after a write, so a tree
// being actively written degrades to always-fresh — the conservative direction.
func TestAJustWrittenTreeIsNeverCached(t *testing.T) {
	repo, _ := newTestRepo(t)
	backend := &treeBackend{status: " M seed.txt\n"}
	dir := rendezvousDir(t)
	// A window far wider than this test's own runtime, so the repository that was
	// just built by newTestRepo is unambiguously inside it.
	srv, err := Serve(Config{RepoRoot: repo, Dir: dir, Tree: backend, TreeRaceWindow: time.Hour})
	if err != nil {
		t.Fatalf("Serve: %v", err)
	}
	defer srv.Close()
	c := &Client{RepoRoot: repo, Dir: dir, TreeRunner: &treeBackend{}, Timeout: 5 * time.Second}

	for i := 0; i < 3; i++ {
		res, err := c.Tree(context.Background(), ClassB)
		if err != nil {
			t.Fatalf("read %d: %v", i, err)
		}
		if res.Provenance != Broker {
			t.Fatalf("read %d provenance = %q, want %q — a tree written within the granularity window must be recomputed, because a peer's write could hide behind our own mtime sample", i, res.Provenance, Broker)
		}
	}
	if got := backend.count(); got != 3 {
		t.Fatalf("backend ran %d times, want 3", got)
	}
	st := srv.Stats()
	if st.TreeEntry {
		t.Fatalf("an unsettled sample was STORED; the entry it left behind is exactly the one whose key cannot be trusted")
	}
	if st.TreeFresh != 3 || st.TreeHits != 0 {
		t.Fatalf("stats = %+v, want 3 fresh reads and 0 hits", st)
	}
}

// TestAServedEntryDescribesATreeThatHasNotChanged pins the other half of the
// stated budget: the cache itself contributes ZERO staleness. An entry is served
// only when a key sampled at read time equals the key it was stored under, so a
// served answer describes a tree that has not changed in any way the key can see.
func TestAServedEntryDescribesATreeThatHasNotChanged(t *testing.T) {
	repo, _ := newTestRepo(t)
	backend := &treeBackend{status: " M seed.txt\n"}
	_, c, _ := treeBroker(t, repo, backend)
	ctx := context.Background()
	settle()

	if _, err := c.Tree(ctx, ClassB); err != nil {
		t.Fatalf("warm: %v", err)
	}
	hit, err := c.Tree(ctx, ClassB)
	if err != nil {
		t.Fatalf("hit: %v", err)
	}
	if hit.Provenance != Cache {
		t.Fatalf("provenance = %q, want %q", hit.Provenance, Cache)
	}
	now, settledNow := sampleStateKey(repo, testRaceWindow)
	if !settledNow {
		t.Fatalf("the tree stopped being settled mid-test; the sample cannot be compared")
	}
	if hit.Key != now {
		t.Fatalf("a served entry's key %+v differs from the tree's key right now %+v — the cache is serving an answer about a tree that has moved", hit.Key, now)
	}
}

// TestAnUndeclaredClassIsDecisional pins the default, which is the whole safety
// argument for putting Class on the wire at all: forgetting to classify must cost
// a spawn, and must never buy a stale refusal.
func TestAnUndeclaredClassIsDecisional(t *testing.T) {
	fresh := []Class{"", ClassC, "c", "C", "A", "B", " a", "b ", "class-b", "unknown"}
	for _, c := range fresh {
		if !c.Decisional() {
			t.Fatalf("class %q is reusable, want decisional — anything but an exact ClassA/ClassB must fail safe", c)
		}
	}
	for _, c := range []Class{ClassA, ClassB} {
		if c.Decisional() {
			t.Fatalf("class %q is decisional, want reusable", c)
		}
	}
}

// TestAnUnkeyableRepositoryIsNeverCached covers the other refusal to store: a
// directory with no index and no resolvable HEAD cannot be keyed at all, and a
// weak key is not a key — the answer is computed and returned, only its
// cacheability is lost.
func TestAnUnkeyableRepositoryIsNeverCached(t *testing.T) {
	dir := rendezvousDir(t)
	root := filepath.Join(dir, "no-repo-here")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	backend := &treeBackend{status: " M nothing.txt\n"}
	srv, err := Serve(Config{RepoRoot: root, Dir: dir, Tree: backend, TreeRaceWindow: testRaceWindow})
	if err != nil {
		t.Fatalf("Serve: %v", err)
	}
	defer srv.Close()
	c := &Client{RepoRoot: root, Dir: dir, TreeRunner: &treeBackend{}, Timeout: 5 * time.Second}

	for i := 0; i < 2; i++ {
		res, err := c.Tree(context.Background(), ClassB)
		if err != nil {
			t.Fatalf("read %d: %v", i, err)
		}
		if res.Provenance != Broker {
			t.Fatalf("read %d provenance = %q, want %q", i, res.Provenance, Broker)
		}
		if res.Status != " M nothing.txt\n" {
			t.Fatalf("read %d status = %q; an unkeyable repo must still get a real answer", i, res.Status)
		}
	}
	if got := backend.count(); got != 2 {
		t.Fatalf("backend ran %d times, want 2 — an answer under no key must never be stored", got)
	}
	if st := srv.Stats(); st.TreeEntry {
		t.Fatalf("an unkeyable answer was stored; there is no key that could ever invalidate it")
	}
}

// BenchmarkSingleFlightAlone measures COALESCING WITH NO CACHE IN THE PICTURE.
//
// The key is `HEAD`, which IsOID rejects, so nothing can be stored or served from
// the Class A cache: every execution this avoids is single-flight's. Read it
// against BenchmarkWorkingTreeCacheAlone, never as one bundled number — two
// mechanisms with two different correctness arguments need two measurements, or a
// later regression cannot be localized to the one that caused it (#5623). The
// stats assertion below is what keeps that separation honest rather than merely
// claimed.
func BenchmarkSingleFlightAlone(b *testing.B) {
	dir := rendezvousDir(b)
	root := filepath.Join(dir, "repo")
	backend := &objectBackend{obj: blob(oidA, "one execution, many callers"), delay: 200 * time.Microsecond}
	srv, err := Serve(Config{RepoRoot: root, Dir: dir, Runner: backend})
	if err != nil {
		b.Fatalf("Serve: %v", err)
	}
	defer srv.Close()
	ctx := context.Background()

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if _, _, err := srv.object(ctx, "HEAD"); err != nil {
				b.Error(err)
				return
			}
		}
	})
	b.StopTimer()

	st := srv.Stats()
	if st.Hits != 0 || st.Entries != 0 || st.TreeHits != 0 {
		b.Fatalf("a cache contributed to a single-flight measurement (%+v); the number would not be attributable", st)
	}
	b.ReportMetric(float64(st.Coalesced)/float64(b.N), "coalesced/op")
	b.ReportMetric(float64(backend.count())/float64(b.N), "git-execs/op")
}

// BenchmarkWorkingTreeCacheAlone measures the Class B cache WITH NO COALESCING.
//
// The reads are serial, so nothing is ever in flight beside anything else and
// single-flight cannot save one execution. The counterpart to
// BenchmarkSingleFlightAlone; the Coalesced assertion is the separation.
func BenchmarkWorkingTreeCacheAlone(b *testing.B) {
	dir := rendezvousDir(b)
	root := filepath.Join(dir, "repo")
	synthGitDir(b, root, oidA)
	backend := &treeBackend{status: " M seed.txt\n"}
	srv, err := Serve(Config{RepoRoot: root, Dir: dir, Tree: backend, TreeRaceWindow: testRaceWindow})
	if err != nil {
		b.Fatalf("Serve: %v", err)
	}
	defer srv.Close()
	ctx := context.Background()
	if _, _, err := srv.treeState(ctx, ClassB); err != nil {
		b.Fatalf("warm: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, prov, err := srv.treeState(ctx, ClassB)
		if err != nil {
			b.Fatalf("read: %v", err)
		}
		if prov != Cache {
			b.Fatalf("read provenance = %q, want %q; this benchmark is measuring the cache, so a miss makes it measure git instead", prov, Cache)
		}
	}
	b.StopTimer()

	if st := srv.Stats(); st.Coalesced != 0 {
		b.Fatalf("single-flight contributed to a cache measurement (%+v); the number would not be attributable", st)
	}
	if got := backend.count(); got != 1 {
		b.Fatalf("the backend ran %d times during a pure cache benchmark, want 1", got)
	}
}
