package leaseref

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// fakeGit is an in-memory git stand-in implementing the Runner contract: it models a
// blob object store (sha -> bytes) and a ref store (ref -> sha) so the whole leaseref
// algorithm runs with no real git. It also records every argv so a test can assert the
// EXACT plumbing issued (the same discipline witness/safecommit tests use).
type fakeGit struct {
	blobs map[string][]byte // object id -> blob bytes
	refs  map[string]string // ref -> object id
	calls [][]string        // every git argv, in order
	next  int               // synthetic object-id counter
}

func newFakeGit() *fakeGit {
	return &fakeGit{blobs: map[string][]byte{}, refs: map[string]string{}}
}

func (f *fakeGit) run(ctx context.Context, dir string, args ...string) (string, int, error) {
	f.calls = append(f.calls, args)
	switch args[0] {
	case "hash-object":
		// hash-object -w <file> : read the file, store it under a synthetic id.
		path := args[len(args)-1]
		b, err := os.ReadFile(path)
		if err != nil {
			return "", 1, nil
		}
		f.next++
		id := synthID(f.next)
		f.blobs[id] = b
		return id + "\n", 0, nil
	case "update-ref":
		if args[1] == "-d" {
			ref := args[2]
			if _, ok := f.refs[ref]; !ok {
				return "", 1, nil // delete of a missing ref exits non-zero (real git behavior)
			}
			delete(f.refs, ref)
			return "", 0, nil
		}
		f.refs[args[1]] = args[2]
		return "", 0, nil
	case "rev-parse":
		// rev-parse --verify --quiet <ref>
		ref := args[len(args)-1]
		if id, ok := f.refs[ref]; ok {
			return id + "\n", 0, nil
		}
		return "", 1, nil
	case "cat-file":
		// cat-file blob <ref>
		ref := args[2]
		id, ok := f.refs[ref]
		if !ok {
			return "", 1, nil
		}
		return string(f.blobs[id]), 0, nil
	case "for-each-ref":
		// for-each-ref --format=%(refname) <prefix>
		prefix := args[len(args)-1]
		var lines []string
		for ref := range f.refs {
			if strings.HasPrefix(ref, prefix) {
				lines = append(lines, ref)
			}
		}
		if len(lines) == 0 {
			return "", 0, nil // empty namespace is exit 0 with no output
		}
		return strings.Join(lines, "\n") + "\n", 0, nil
	}
	return "", 0, nil
}

// synthID renders a small int as a stable synthetic object id for the fake object store.
func synthID(n int) string { return "obj" + itoa(n) }

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

func ctx() context.Context { return context.Background() }

func TestAcquireGetReleaseRoundTrip(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")

	rec := Record{
		ID:         "kernel-lane",
		TreeGlobs:  []string{"internal/kernel/**"},
		Holder:     "machineA:sess1",
		AcquiredAt: 1000,
		TTLSeconds: 300,
	}
	ref, err := s.Acquire(ctx(), rec)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	if ref != "refs/fak/locks/kernel-lane" {
		t.Fatalf("Acquire ref=%q, want refs/fak/locks/kernel-lane", ref)
	}

	got, ok, err := s.Get(ctx(), "kernel-lane")
	if err != nil || !ok {
		t.Fatalf("Get ok=%v err=%v, want a record", ok, err)
	}
	if got.Holder != "machineA:sess1" || len(got.TreeGlobs) != 1 || got.TreeGlobs[0] != "internal/kernel/**" {
		t.Fatalf("Get returned %+v, want the acquired record", got)
	}

	if err := s.Release(ctx(), "kernel-lane"); err != nil {
		t.Fatalf("Release: %v", err)
	}
	if _, ok, _ := s.Get(ctx(), "kernel-lane"); ok {
		t.Fatalf("Get after Release returned a record, want absent")
	}
}

// TestAcquireIssuesExactPlumbing pins the EXACT git argv: write the blob, then point the
// ref — never a branch, never a force, never HEAD.
func TestAcquireIssuesExactPlumbing(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	if _, err := s.Acquire(ctx(), Record{ID: "x", TreeGlobs: []string{"a"}, Holder: "h", AcquiredAt: 1}); err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	if len(g.calls) != 2 {
		t.Fatalf("Acquire issued %d git calls, want 2 (hash-object, update-ref): %v", len(g.calls), g.calls)
	}
	if g.calls[0][0] != "hash-object" || g.calls[0][1] != "-w" {
		t.Fatalf("first call = %v, want [hash-object -w <file>]", g.calls[0])
	}
	up := g.calls[1]
	if up[0] != "update-ref" || up[1] != "refs/fak/locks/x" {
		t.Fatalf("second call = %v, want [update-ref refs/fak/locks/x <sha>]", up)
	}
	for _, c := range g.calls {
		for _, a := range c {
			if a == "-f" || a == "--force" || a == "HEAD" || strings.HasPrefix(a, "refs/heads/") {
				t.Fatalf("leaseref must never force/touch a branch/HEAD; saw %q in %v", a, c)
			}
		}
	}
}

// TestPeerVisibilityAfterFetch models acquire-on-A -> (push/fetch) -> visible-on-B by
// handing machine B's store the SAME ref state machine A wrote. The point: once the ref
// is in B's local ref store (an ordinary fetch put it there), B's List sees it.
func TestPeerVisibilityAfterFetch(t *testing.T) {
	a := newFakeGit()
	sa := NewWithRunner(a.run, "")
	if _, err := sa.Acquire(ctx(), Record{ID: "shared", TreeGlobs: []string{"docs/**"}, Holder: "A", AcquiredAt: 10, TTLSeconds: 0}); err != nil {
		t.Fatalf("A.Acquire: %v", err)
	}

	// "fetch": copy A's object+ref state into B's store (what git fetch does for the namespace).
	b := newFakeGit()
	for k, v := range a.blobs {
		b.blobs[k] = v
	}
	for k, v := range a.refs {
		b.refs[k] = v
	}
	sb := NewWithRunner(b.run, "")

	recs, err := sb.List(ctx())
	if err != nil {
		t.Fatalf("B.List: %v", err)
	}
	if len(recs) != 1 || recs[0].Holder != "A" || recs[0].ID != "shared" {
		t.Fatalf("B.List = %+v, want the lease A acquired", recs)
	}
}

func TestExpiryReapable(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	// One live (long TTL) and one already-expired (acquired far in the past) lease.
	if _, err := s.Acquire(ctx(), Record{ID: "live", TreeGlobs: []string{"a"}, Holder: "h", AcquiredAt: time.Now().Unix(), TTLSeconds: 3600}); err != nil {
		t.Fatalf("Acquire live: %v", err)
	}
	if _, err := s.Acquire(ctx(), Record{ID: "dead", TreeGlobs: []string{"b"}, Holder: "h", AcquiredAt: 100, TTLSeconds: 10}); err != nil {
		t.Fatalf("Acquire dead: %v", err)
	}

	live, expired, err := s.Live(ctx(), time.Now())
	if err != nil {
		t.Fatalf("Live: %v", err)
	}
	if len(live) != 1 || live[0].ID != "live" {
		t.Fatalf("Live set = %+v, want only [live]", live)
	}
	if len(expired) != 1 || expired[0] != "dead" {
		t.Fatalf("expired set = %v, want [dead]", expired)
	}

	// A peer reaps the expired lease — an ordinary ref delete.
	if err := s.Release(ctx(), "dead"); err != nil {
		t.Fatalf("reap Release: %v", err)
	}
	if _, ok, _ := s.Get(ctx(), "dead"); ok {
		t.Fatalf("reaped lease still present")
	}
}

func TestReleaseMissingIsIdempotent(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	if err := s.Release(ctx(), "never-existed"); err != nil {
		t.Fatalf("Release of a missing lease must be a no-op, got %v", err)
	}
}

func TestInvalidIDRejected(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	for _, bad := range []string{"", "a/b", "../escape", "-flag", ".dot", "has space", "tilde~", "q?"} {
		if _, err := s.Acquire(ctx(), Record{ID: bad, TreeGlobs: []string{"a"}, Holder: "h"}); err == nil {
			t.Fatalf("Acquire(%q) should reject an unsafe ref id", bad)
		}
	}
	// A valid id is accepted.
	if _, err := s.Acquire(ctx(), Record{ID: "ok_lane.1-2", TreeGlobs: []string{"a"}, Holder: "h", AcquiredAt: 1}); err != nil {
		t.Fatalf("Acquire of a valid id failed: %v", err)
	}
}

func TestListSkipsUnparseableBlob(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	if _, err := s.Acquire(ctx(), Record{ID: "good", TreeGlobs: []string{"a"}, Holder: "h", AcquiredAt: 1}); err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	// Inject a corrupt ref pointing at non-JSON bytes — List must skip it, not fail.
	g.blobs["garbage"] = []byte("not json {{{")
	g.refs["refs/fak/locks/bad"] = "garbage"
	recs, err := s.List(ctx())
	if err != nil {
		t.Fatalf("List must not error on a corrupt record: %v", err)
	}
	if len(recs) != 1 || recs[0].ID != "good" {
		t.Fatalf("List = %+v, want only the parseable [good]", recs)
	}
}

func TestRecordExpired(t *testing.T) {
	base := time.Unix(1000, 0)
	noTTL := Record{AcquiredAt: 1000, TTLSeconds: 0}
	if noTTL.Expired(base.Add(1e6 * time.Second)) {
		t.Fatal("a zero TTL must never expire")
	}
	r := Record{AcquiredAt: 1000, TTLSeconds: 60}
	if r.Expired(time.Unix(1059, 0)) {
		t.Fatal("not yet expired at acquired+59")
	}
	if !r.Expired(time.Unix(1060, 0)) {
		t.Fatal("expired at acquired+ttl")
	}
}

// TestLiveLeasesProjection pins the READ-SIDE seam (#825): the live records under
// refs/fak/locks/* project into the dos_arbitrate live_leases shape {lane,lane_kind,tree},
// expired records are dropped, and an empty namespace yields a non-nil empty slice (encodes
// as `[]`, the "nothing held" an arbiter reads).
func TestLiveLeasesProjection(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")

	// Empty namespace -> a non-nil empty slice so JSON renders [].
	empty, err := s.LiveLeases(ctx(), time.Now())
	if err != nil {
		t.Fatalf("LiveLeases (empty): %v", err)
	}
	if empty == nil || len(empty) != 0 {
		t.Fatalf("LiveLeases over an empty namespace = %#v, want a non-nil empty slice", empty)
	}
	if b, _ := json.Marshal(empty); string(b) != "[]" {
		t.Fatalf("empty LiveLeases JSON = %s, want []", b)
	}

	// One live lease and one already-expired lease.
	if _, err := s.Acquire(ctx(), Record{ID: "docs-lane", TreeGlobs: []string{"docs/**"}, Holder: "A:1", AcquiredAt: time.Now().Unix(), TTLSeconds: 3600}); err != nil {
		t.Fatalf("Acquire live: %v", err)
	}
	if _, err := s.Acquire(ctx(), Record{ID: "dead-lane", TreeGlobs: []string{"internal/x/**"}, Holder: "B:2", AcquiredAt: 100, TTLSeconds: 10}); err != nil {
		t.Fatalf("Acquire dead: %v", err)
	}

	leases, err := s.LiveLeases(ctx(), time.Now())
	if err != nil {
		t.Fatalf("LiveLeases: %v", err)
	}
	if len(leases) != 1 {
		t.Fatalf("LiveLeases = %+v, want only the one non-expired lease", leases)
	}
	got := leases[0]
	if got.Lane != "docs-lane" || got.LaneKind != "cluster" {
		t.Fatalf("projection lane/kind = %q/%q, want docs-lane/cluster", got.Lane, got.LaneKind)
	}
	if len(got.Tree) != 1 || got.Tree[0] != "docs/**" {
		t.Fatalf("projection tree = %v, want [docs/**]", got.Tree)
	}

	// The projected element marshals to exactly the arbiter's live_leases entry shape.
	b, _ := json.Marshal(got)
	want := `{"lane":"docs-lane","lane_kind":"cluster","tree":["docs/**"]}`
	if string(b) != want {
		t.Fatalf("ArbiterLease JSON = %s, want %s", b, want)
	}
}

// TestRealGitRoundTrip exercises the package against the REAL git binary in a temp repo.
// Skipped when git is unavailable (e.g. the native-Windows test path); it runs under the
// WSL suite. It proves the actual plumbing — hash-object/update-ref/for-each-ref/cat-file
// — composes into a working acquire->list->release lifecycle.
func TestRealGitRoundTrip(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir := t.TempDir()
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "t@example.com"},
		{"config", "user.name", "t"},
	} {
		c := exec.Command("git", args...)
		c.Dir = dir
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}

	s := NewWithRunner(gitRunner, dir)
	rec := Record{ID: "lane1", TreeGlobs: []string{"internal/x/**"}, Holder: "host:sess", AcquiredAt: 12345, TTLSeconds: 600}
	if _, err := s.Acquire(ctx(), rec); err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	got, ok, err := s.Get(ctx(), "lane1")
	if err != nil || !ok {
		t.Fatalf("Get ok=%v err=%v", ok, err)
	}
	if got.Holder != "host:sess" || got.AcquiredAt != 12345 {
		t.Fatalf("real round-trip record = %+v, want the acquired one", got)
	}
	// Sanity: the on-disk ref really is under refs/fak/locks/.
	listing, _, _ := gitRunner(ctx(), dir, "for-each-ref", "--format=%(refname)", "refs/fak/locks/")
	if !strings.Contains(listing, "refs/fak/locks/lane1") {
		t.Fatalf("ref not under refs/fak/locks/: %q", listing)
	}

	recs, err := s.List(ctx())
	if err != nil || len(recs) != 1 {
		t.Fatalf("List err=%v recs=%+v, want exactly one", err, recs)
	}
	if err := s.Release(ctx(), "lane1"); err != nil {
		t.Fatalf("Release: %v", err)
	}
	if _, ok, _ := s.Get(ctx(), "lane1"); ok {
		t.Fatalf("lease still present after Release")
	}
}

// TestRecordJSONShape pins the on-the-wire record shape so a future field rename can't
// silently break a peer reading an older/newer record.
func TestRecordJSONShape(t *testing.T) {
	b, _ := json.Marshal(Record{ID: "x", TreeGlobs: []string{"a"}, Holder: "h", AcquiredAt: 1, TTLSeconds: 2})
	want := `{"id":"x","tree_globs":["a"],"holder":"h","acquired_unix":1,"ttl_seconds":2}`
	if string(b) != want {
		t.Fatalf("record JSON = %s, want %s", b, want)
	}
}
