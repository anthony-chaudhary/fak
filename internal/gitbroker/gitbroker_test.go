package gitbroker

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// rendezvousDir returns a SHORT temp dir for a test's socket.
//
// t.TempDir() embeds the test's name, and an AF_UNIX path is bounded by
// sockaddr_un.sun_path (108 bytes on Linux, 104 on macOS). A descriptive test
// name plus the default temp layout can push a socket path past that bound and
// fail the bind for reasons that have nothing to do with the code under test, so
// every socket in this file lives under a name-free directory instead.
func rendezvousDir(t *testing.T) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "gb")
	if err != nil {
		t.Fatalf("temp dir: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	return dir
}

// fakeRunner is a Runner whose answers are fixed and whose calls are counted —
// the only way to prove a second read did NOT reach the backend.
type fakeRunner struct {
	mu    sync.Mutex
	objs  map[string]Object
	calls map[string]int
	delay time.Duration
}

func newFakeRunner(objs map[string]Object) *fakeRunner {
	return &fakeRunner{objs: objs, calls: map[string]int{}}
}

func (f *fakeRunner) Object(ctx context.Context, rev string) (Object, error) {
	f.mu.Lock()
	f.calls[rev]++
	delay := f.delay
	obj, ok := f.objs[rev]
	f.mu.Unlock()
	if delay > 0 {
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return Object{}, ctx.Err()
		}
	}
	if !ok {
		return Object{}, fmt.Errorf("%w: %s", ErrMissingObject, rev)
	}
	return obj, nil
}

func (f *fakeRunner) count(rev string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls[rev]
}

const (
	oidA = "0123456789abcdef0123456789abcdef01234567"
	oidB = "fedcba9876543210fedcba9876543210fedcba98"
)

func blob(oid, body string) Object {
	return Object{OID: oid, Type: "blob", Size: int64(len(body)), Data: []byte(body)}
}

// TestIsOIDIsTheClassAGate pins the one predicate that decides what may be
// cached. Everything this rung promises about needing no invalidation rests on
// it: a full OID names immutable content, and nothing else here does.
func TestIsOIDIsTheClassAGate(t *testing.T) {
	cacheable := []string{
		oidA,
		strings.Repeat("a", 64), // SHA-256 spelling
	}
	for _, k := range cacheable {
		if !IsOID(k) {
			t.Errorf("IsOID(%q) = false, want true — a full OID is content-addressed", k)
		}
	}
	// Every one of these names something MUTABLE or AMBIGUOUS. Caching any of
	// them without an invalidation path is the Class B/C mistake this rung is
	// defined by not making.
	uncacheable := []string{
		"", "HEAD", "main", "@{u}", "main:go.mod", "HEAD~3",
		"0123456",             // abbreviated: can become ambiguous as the repo grows
		strings.ToUpper(oidA), // uppercase would double-key the same object
		oidA + "0",            // over-long
		"0123456789abcdef0123456789abcdef0123456g", // not hex
	}
	for _, k := range uncacheable {
		if IsOID(k) {
			t.Errorf("IsOID(%q) = true, want false — it is not content-addressed", k)
		}
	}
}

// TestRendezvousIsDerivedFromTheRepoAlone is the "no env hand-wiring" witness:
// a client that knows which repo it is asking about computes the same socket
// path the broker bound, with nothing passed between them.
func TestRendezvousIsDerivedFromTheRepoAlone(t *testing.T) {
	dir := rendezvousDir(t)
	a := RendezvousIn(dir, filepath.Join(dir, "repo"))
	b := RendezvousIn(dir, filepath.Join(dir, "repo"))
	if a != b {
		t.Fatalf("same repo produced two rendezvous:\n %+v\n %+v", a, b)
	}
	other := RendezvousIn(dir, filepath.Join(dir, "other"))
	if other.Socket == a.Socket {
		t.Fatalf("two repos share a socket (%s) — one broker would answer for both", a.Socket)
	}
	if a.Token == a.Socket {
		t.Fatal("token and socket must be distinct paths")
	}
	// A trailing separator or an unclean path is the same repo.
	if got := RendezvousIn(dir, filepath.Join(dir, "repo")+string(filepath.Separator)); got != a {
		t.Fatalf("unclean repo path produced a different rendezvous: %+v vs %+v", got, a)
	}
}

// TestCachedObjectIsSharedAcrossClients is the cross-client Class A witness at
// the package level: two independent Clients, one broker, one backend call. The
// separate-OS-process form of this same claim is TestGitdCachesAcrossSeparate
// ClientProcesses in cmd/fak.
func TestCachedObjectIsSharedAcrossClients(t *testing.T) {
	dir := rendezvousDir(t)
	root := filepath.Join(dir, "repo")
	backend := newFakeRunner(map[string]Object{oidA: blob(oidA, "hello broker")})
	srv, err := Serve(Config{RepoRoot: root, Dir: dir, Runner: backend})
	if err != nil {
		t.Fatalf("Serve: %v", err)
	}
	defer srv.Close()

	// Each client carries a fallback runner that would be VISIBLE if used, so a
	// silent fallback cannot masquerade as a broker hit.
	fallback := newFakeRunner(map[string]Object{oidA: blob(oidA, "SPAWNED-NOT-BROKER")})
	first := &Client{RepoRoot: root, Dir: dir, Runner: fallback}
	second := &Client{RepoRoot: root, Dir: dir, Runner: fallback}

	r1, err := first.Object(context.Background(), oidA)
	if err != nil {
		t.Fatalf("first client: %v", err)
	}
	if r1.Provenance != Broker {
		t.Fatalf("first read provenance = %q, want %q (cold cache must reach the backend)", r1.Provenance, Broker)
	}
	r2, err := second.Object(context.Background(), oidA)
	if err != nil {
		t.Fatalf("second client: %v", err)
	}
	if r2.Provenance != Cache {
		t.Fatalf("second client's provenance = %q, want %q — the whole point of the broker is that a DIFFERENT client reuses the warm entry", r2.Provenance, Cache)
	}
	if string(r1.Data) != string(r2.Data) || r1.OID != r2.OID || r1.Type != r2.Type || r1.Size != r2.Size {
		t.Fatalf("cache hit differs from the live read:\n broker %+v\n cache  %+v", r1.Object, r2.Object)
	}
	if got := backend.count(oidA); got != 1 {
		t.Fatalf("backend saw %d calls for %s, want exactly 1 — the second client must not spawn", got, oidA)
	}
	if got := fallback.count(oidA); got != 0 {
		t.Fatalf("the fallback runner ran %d times; a live broker must not be bypassed", got)
	}
	st := srv.Stats()
	if st.Hits != 1 || st.Misses != 1 || st.Served != 2 {
		t.Fatalf("stats = %+v, want 1 hit / 1 miss / 2 served", st)
	}
}

// TestNonOIDQueriesAreNeverCached holds the rung boundary. `HEAD` is Class B: it
// names something mutable, so answering it from a cache with no invalidation
// path would be wrong the moment anyone commits.
func TestNonOIDQueriesAreNeverCached(t *testing.T) {
	dir := rendezvousDir(t)
	root := filepath.Join(dir, "repo")
	backend := newFakeRunner(map[string]Object{"HEAD": blob(oidA, "tip")})
	srv, err := Serve(Config{RepoRoot: root, Dir: dir, Runner: backend})
	if err != nil {
		t.Fatalf("Serve: %v", err)
	}
	defer srv.Close()

	c := &Client{RepoRoot: root, Dir: dir, Runner: backend}
	for i := 0; i < 3; i++ {
		r, err := c.Object(context.Background(), "HEAD")
		if err != nil {
			t.Fatalf("read %d: %v", i, err)
		}
		if r.Provenance != Broker {
			t.Fatalf("read %d provenance = %q, want %q — a mutable key must be answered live every time", i, r.Provenance, Broker)
		}
	}
	if got := backend.count("HEAD"); got != 3 {
		t.Fatalf("backend saw %d calls for HEAD, want 3 — a non-content-addressed key was cached", got)
	}
	if st := srv.Stats(); st.Entries != 0 || st.Uncached != 3 {
		t.Fatalf("stats = %+v, want 0 cache entries and 3 uncacheable reads", st)
	}
}

// TestDeadBrokerStillAnswers is the fail-open witness: kill the broker mid-run
// and the very next call produces the same object, tagged fallback-spawn. #4603
// is the lesson — a resident helper that dies must cost latency, not answers.
func TestDeadBrokerStillAnswers(t *testing.T) {
	dir := rendezvousDir(t)
	root := filepath.Join(dir, "repo")
	obj := blob(oidA, "identical either way")
	backend := newFakeRunner(map[string]Object{oidA: obj})
	srv, err := Serve(Config{RepoRoot: root, Dir: dir, Runner: backend})
	if err != nil {
		t.Fatalf("Serve: %v", err)
	}
	c := &Client{RepoRoot: root, Dir: dir, Runner: backend}

	live, err := c.Object(context.Background(), oidA)
	if err != nil {
		t.Fatalf("read through the live broker: %v", err)
	}
	if live.Provenance != Broker {
		t.Fatalf("live provenance = %q, want %q", live.Provenance, Broker)
	}

	srv.Close() // the broker dies mid-run

	dead, err := c.Object(context.Background(), oidA)
	if err != nil {
		t.Fatalf("read after the broker died: %v — fail-open means this must still answer", err)
	}
	if dead.Provenance != FallbackSpawn {
		t.Fatalf("post-mortem provenance = %q, want %q — 'the broker is down' must not be spelled the same way as a normal answer", dead.Provenance, FallbackSpawn)
	}
	if string(dead.Data) != string(live.Data) || dead.OID != live.OID || dead.Type != live.Type || dead.Size != live.Size {
		t.Fatalf("a dead broker changed the ANSWER:\n with broker %+v\n after death %+v", live.Object, dead.Object)
	}
}

// TestHungBrokerCannotWedgeAClient stands up a listener that accepts and then
// says nothing forever — the failure mode a plain "is the socket there?" check
// cannot see. The client must be cut off by its own deadline and spawn.
func TestHungBrokerCannotWedgeAClient(t *testing.T) {
	dir := rendezvousDir(t)
	root := filepath.Join(dir, "repo")
	rv := RendezvousIn(dir, root)
	ln, err := net.Listen("unix", rv.Socket)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	if err := os.WriteFile(rv.Token, []byte("any-token"), 0o600); err != nil {
		t.Fatalf("write token: %v", err)
	}
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func() { <-stop; _ = conn.Close() }() // accepted, then silence
		}
	}()

	obj := blob(oidA, "spawned past the hang")
	backend := newFakeRunner(map[string]Object{oidA: obj})
	c := &Client{RepoRoot: root, Dir: dir, Timeout: 150 * time.Millisecond, Runner: backend}

	start := time.Now()
	r, err := c.Object(context.Background(), oidA)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("hung broker propagated an error: %v", err)
	}
	if r.Provenance != FallbackSpawn {
		t.Fatalf("provenance = %q, want %q — a hung broker must not be reported as a served answer", r.Provenance, FallbackSpawn)
	}
	if string(r.Data) != string(obj.Data) {
		t.Fatalf("fallback returned %q, want %q", r.Data, obj.Data)
	}
	if elapsed > 3*time.Second {
		t.Fatalf("client took %s against a hung broker with a 150ms deadline — the fuse did not blow", elapsed)
	}
}

// TestBadTokenFallsBackRatherThanFailing pins that auth failure is ALSO fail-open.
// An unauthorized client is a broker problem, and a broker problem never becomes
// the caller's error.
func TestBadTokenFallsBackRatherThanFailing(t *testing.T) {
	dir := rendezvousDir(t)
	root := filepath.Join(dir, "repo")
	obj := blob(oidA, "auth-independent")
	backend := newFakeRunner(map[string]Object{oidA: obj})
	srv, err := Serve(Config{RepoRoot: root, Dir: dir, Runner: backend})
	if err != nil {
		t.Fatalf("Serve: %v", err)
	}
	defer srv.Close()
	if err := os.WriteFile(srv.Rendezvous().Token, []byte("wrong"), 0o600); err != nil {
		t.Fatalf("clobber token: %v", err)
	}

	c := &Client{RepoRoot: root, Dir: dir, Runner: backend}
	r, err := c.Object(context.Background(), oidA)
	if err != nil {
		t.Fatalf("bad token became a caller error: %v", err)
	}
	if r.Provenance != FallbackSpawn {
		t.Fatalf("provenance = %q, want %q — an unauthorized client must not be served", r.Provenance, FallbackSpawn)
	}
	if st := srv.Stats(); st.Served != 0 {
		t.Fatalf("broker served %d requests to an unauthorized client, want 0", st.Served)
	}
}

// TestStatsDistinguishDownFromIdle is the stallscan rule applied to this
// package's own status surface: a broker that is DOWN must not read the same as
// a broker that is up and has served nothing.
func TestStatsDistinguishDownFromIdle(t *testing.T) {
	dir := rendezvousDir(t)
	root := filepath.Join(dir, "repo")
	c := &Client{RepoRoot: root, Dir: dir}
	if _, ok := c.Stats(context.Background()); ok {
		t.Fatal("Stats reported ok with no broker running")
	}
	srv, err := Serve(Config{RepoRoot: root, Dir: dir, Runner: newFakeRunner(nil)})
	if err != nil {
		t.Fatalf("Serve: %v", err)
	}
	defer srv.Close()
	st, ok := c.Stats(context.Background())
	if !ok {
		t.Fatal("Stats reported not-ok against a live broker")
	}
	if st.Served != 0 {
		t.Fatalf("a freshly started broker reports %d served, want 0", st.Served)
	}
}

// TestServeRefusesToStealALiveRendezvous and its stale-socket twin cover the two
// ways a bind can fail on a path that already exists. Reclaiming a corpse is
// required (a crashed broker would otherwise poison the repo forever); stealing
// a live one is forbidden (it would silently split clients across two brokers).
func TestServeRefusesToStealALiveRendezvous(t *testing.T) {
	dir := rendezvousDir(t)
	root := filepath.Join(dir, "repo")
	first, err := Serve(Config{RepoRoot: root, Dir: dir, Runner: newFakeRunner(nil)})
	if err != nil {
		t.Fatalf("first Serve: %v", err)
	}
	defer first.Close()
	second, err := Serve(Config{RepoRoot: root, Dir: dir, Runner: newFakeRunner(nil)})
	if err == nil {
		second.Close()
		t.Fatal("a second broker bound the same repo's socket — clients would be split across two caches")
	}
}

func TestServeReclaimsAStaleSocket(t *testing.T) {
	dir := rendezvousDir(t)
	root := filepath.Join(dir, "repo")
	rv := RendezvousIn(dir, root)
	// A crashed broker's corpse: the socket node exists, nothing answers it.
	ln, err := net.Listen("unix", rv.Socket)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	// Keep the socket NODE on close so there is a corpse to reclaim; without this
	// Go unlinks it and the case under test cannot arise.
	ln.(*net.UnixListener).SetUnlinkOnClose(false)
	_ = ln.Close()
	if _, err := os.Stat(rv.Socket); err != nil {
		t.Skipf("this platform unlinked the socket node on close (%v); there is no corpse to reclaim", err)
	}
	srv, err := Serve(Config{RepoRoot: root, Dir: dir, Runner: newFakeRunner(nil)})
	if err != nil {
		t.Fatalf("Serve did not reclaim a stale socket: %v", err)
	}
	defer srv.Close()
}

// TestParseBatchRecord covers the cat-file wire shapes without needing git.
func TestParseBatchRecord(t *testing.T) {
	obj, err := parseBatchRecord([]byte(oidA + " blob 5\nhello\n"))
	if err != nil {
		t.Fatalf("well-formed record: %v", err)
	}
	if obj.OID != oidA || obj.Type != "blob" || obj.Size != 5 || string(obj.Data) != "hello" {
		t.Fatalf("parsed %+v, want oid=%s blob/5/hello", obj, oidA)
	}
	if _, err := parseBatchRecord([]byte("deadbeef missing\n")); !errors.Is(err, ErrMissingObject) {
		t.Fatalf("missing record error = %v, want ErrMissingObject", err)
	}
	if _, err := parseBatchRecord([]byte(oidA + " blob 99\nshort\n")); err == nil {
		t.Fatal("a truncated payload parsed as a complete object")
	}
	// Binary payloads must survive byte-for-byte, including embedded newlines
	// and NULs — a blob is not text.
	body := []byte{0x00, '\n', 0xff, 'x'}
	bin, err := parseBatchRecord(append([]byte(oidB+" blob 4\n"), append(body, '\n')...))
	if err != nil {
		t.Fatalf("binary record: %v", err)
	}
	if string(bin.Data) != string(body) {
		t.Fatalf("binary payload = %v, want %v", bin.Data, body)
	}
}

// TestCacheIsBounded proves the Class A store cannot grow without limit, and
// that an oversized single object is simply not cached rather than blowing the
// budget.
func TestCacheIsBounded(t *testing.T) {
	c := newCache(1024) // maxItem = 128
	for i := 0; i < 64; i++ {
		oid := fmt.Sprintf("%040x", i)
		c.put(oid, Object{OID: oid, Size: 100, Data: make([]byte, 100)})
	}
	entries, bytes := c.sizes()
	if bytes > 1024 {
		t.Fatalf("cache holds %d bytes over a 1024 budget (%d entries)", bytes, entries)
	}
	big := strings.Repeat("f", 40)
	c.put(big, Object{OID: big, Size: 512, Data: make([]byte, 512)})
	if _, ok := c.get(big); ok {
		t.Fatal("an object larger than the per-item ceiling was cached anyway")
	}
}

// TestBrokerAndSpawnAgreeOnRealGit is the end-to-end parity witness against the
// actual git binary: broker, cache, and spawn must produce one identical object.
// This is the claim the whole rung rests on, so it is checked against real git
// rather than a fake.
func TestBrokerAndSpawnAgreeOnRealGit(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo, err := os.MkdirTemp("", "gbrepo")
	if err != nil {
		t.Fatalf("temp repo: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(repo) })
	run := func(args ...string) string {
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	run("init", "-q")
	body := "content-addressed\nand immutable\n"
	if err := os.WriteFile(filepath.Join(repo, "f.txt"), []byte(body), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	oid := run("hash-object", "-w", "f.txt")
	if !IsOID(oid) {
		t.Fatalf("git returned %q, which IsOID rejects", oid)
	}

	dir := rendezvousDir(t)
	// Spawn FIRST, with no broker at all: this is the pre-broker path, and it
	// is the reference every other provenance is compared against.
	c := &Client{RepoRoot: repo, Dir: dir}
	spawned, err := c.Object(context.Background(), oid)
	if err != nil {
		t.Fatalf("spawn read: %v", err)
	}
	if spawned.Provenance != FallbackSpawn {
		t.Fatalf("provenance with no broker = %q, want %q", spawned.Provenance, FallbackSpawn)
	}
	if string(spawned.Data) != body {
		t.Fatalf("spawned payload = %q, want %q", spawned.Data, body)
	}

	srv, err := Serve(Config{RepoRoot: repo, Dir: dir})
	if err != nil {
		t.Fatalf("Serve: %v", err)
	}
	defer srv.Close()
	viaBroker, err := c.Object(context.Background(), oid)
	if err != nil {
		t.Fatalf("broker read: %v", err)
	}
	viaCache, err := c.Object(context.Background(), oid)
	if err != nil {
		t.Fatalf("cache read: %v", err)
	}
	if viaBroker.Provenance != Broker || viaCache.Provenance != Cache {
		t.Fatalf("provenance sequence = %q then %q, want %q then %q",
			viaBroker.Provenance, viaCache.Provenance, Broker, Cache)
	}
	for name, got := range map[string]Result{"broker": viaBroker, "cache": viaCache} {
		if got.OID != spawned.OID || got.Type != spawned.Type || got.Size != spawned.Size || string(got.Data) != string(spawned.Data) {
			t.Fatalf("%s answer differs from the pre-broker spawn:\n %s %+v\n spawn %+v", name, name, got.Object, spawned.Object)
		}
	}
}
