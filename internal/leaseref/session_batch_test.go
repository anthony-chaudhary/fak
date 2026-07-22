package leaseref

// session_batch_test.go is the witness for #5355 Half B: ListSessions must read every
// session descriptor through ONE `git cat-file --batch` process instead of one `cat-file
// blob` spawn per ref (the O(N)-spawn cost that made `fak leaseref audit` ~594s over the
// ~14k-ref backlog). The batched path must be BEHAVIORALLY IDENTICAL to the per-ref reader:
// same descriptors, same namespace split, same skip-a-bad-blob resilience, same id-fill.

import (
	"fmt"
	"os/exec"
	"reflect"
	"strings"
	"testing"
)

// batchReadCalls counts the `git cat-file --batch` invocations recorded on the fake — the
// witness that ListSessions drained the whole backlog in ONE process spawn.
func (f *fakeGit) batchReadCalls() int {
	n := 0
	for _, c := range f.calls {
		if len(c) >= 2 && c[0] == "cat-file" && c[1] == "--batch" {
			n++
		}
	}
	return n
}

// perRefBlobCalls counts the serial `git cat-file blob <ref>` spawns — the per-ref read cost
// #5355 Half B removes. Zero of these after a batched ListSessions is the fix's witness.
func (f *fakeGit) perRefBlobCalls() int {
	n := 0
	for _, c := range f.calls {
		if len(c) >= 2 && c[0] == "cat-file" && c[1] == "blob" {
			n++
		}
	}
	return n
}

// TestListSessionsBatchIssuesOneCatFileBatch is the headline #5355 Half B witness at scale:
// a backlog of session descriptors is read in a SINGLE cat-file --batch call — zero per-ref
// cat-file blob spawns.
func TestListSessionsBatchIssuesOneCatFileBatch(t *testing.T) {
	g := newFakeGit()
	s := NewWithStdinRunner(g.run, g.runStdin, "")

	const n = 5
	for i := 0; i < n; i++ {
		if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: fmt.Sprintf("s%d", i), Host: "h", PCBState: "RUNNING", UpdatedAt: 1, TTLSecs: 0}); err != nil {
			t.Fatalf("PublishSession: %v", err)
		}
	}

	g.calls = nil // start the argv witness at the read
	ds, err := s.ListSessions(ctx())
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(ds) != n {
		t.Fatalf("ListSessions returned %d descriptors, want %d", len(ds), n)
	}
	// The fix: ONE process spawn for the whole backlog, not one per ref.
	if got := g.batchReadCalls(); got != 1 {
		t.Fatalf("cat-file --batch calls = %d, want exactly 1 (the whole point of #5355 Half B)", got)
	}
	if got := g.perRefBlobCalls(); got != 0 {
		t.Fatalf("per-ref cat-file blob spawns = %d, want 0 (the O(N)-spawn read cost removed)", got)
	}
}

// TestListSessionsBatchMatchesPerRefPath is the PARITY witness: over the SAME ref/blob state,
// the batched reader (NewWithStdinRunner) returns byte-identical descriptors to the per-ref
// reader (NewWithRunner) — and a coexisting lock lease is EXCLUDED from both views.
func TestListSessionsBatchMatchesPerRefPath(t *testing.T) {
	g := newFakeGit()
	batch := NewWithStdinRunner(g.run, g.runStdin, "")

	want := []SessionDescriptor{
		{ID: "alpha", Host: "n1", PCBState: "RUNNING", UpdatedAt: 100, TTLSecs: 0},
		{ID: "bravo", Host: "n2", PCBState: "PAUSED", UpdatedAt: 200, TTLSecs: 600, AgentUUID: "u-2"},
		{ID: "charlie", Host: "n3", PCBState: "DRAINING", UpdatedAt: 300, TTLSecs: 0},
	}
	for _, d := range want {
		if _, err := batch.PublishSession(ctx(), d); err != nil {
			t.Fatalf("PublishSession %s: %v", d.ID, err)
		}
	}
	// A lock lease under the SAME refs/fak/locks/ prefix must NOT leak into the session view.
	if _, err := batch.Acquire(ctx(), Record{ID: "lease-x", TreeGlobs: []string{"a"}, Holder: "h", AcquiredAt: 1, TTLSeconds: 0}); err != nil {
		t.Fatalf("Acquire: %v", err)
	}

	// The per-ref reader over the SAME fake state is the parity oracle.
	perRef := NewWithRunner(g.run, "")
	wantDs, err := perRef.ListSessions(ctx())
	if err != nil {
		t.Fatalf("per-ref ListSessions: %v", err)
	}
	gotDs, err := batch.ListSessions(ctx())
	if err != nil {
		t.Fatalf("batch ListSessions: %v", err)
	}
	if !reflect.DeepEqual(gotDs, wantDs) {
		t.Fatalf("batch path = %+v, per-ref path = %+v; must be identical", gotDs, wantDs)
	}
	if len(gotDs) != len(want) {
		t.Fatalf("ListSessions returned %d descriptors, want %d (lock lease excluded)", len(gotDs), len(want))
	}
	for i, d := range want {
		if gotDs[i] != d {
			t.Fatalf("descriptor %d = %+v, want %+v", i, gotDs[i], d)
		}
	}
}

// TestListSessionsBatchSkipsUnparseable proves a corrupt/forward-incompatible session ref is
// SKIPPED by the batch reader, not surfaced as an error — the valid descriptor survives.
func TestListSessionsBatchSkipsUnparseable(t *testing.T) {
	g := newFakeGit()
	s := NewWithStdinRunner(g.run, g.runStdin, "")
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "good", Host: "h", PCBState: "RUNNING", UpdatedAt: 1}); err != nil {
		t.Fatalf("PublishSession: %v", err)
	}
	// A session ref pointing at a non-JSON blob — the batch reader must skip it, not fail.
	g.blobs["garbageoid"] = []byte("not json {{{")
	g.refs["refs/fak/locks/session-bad"] = "garbageoid"

	ds, err := s.ListSessions(ctx())
	if err != nil {
		t.Fatalf("ListSessions must not error on a corrupt descriptor: %v", err)
	}
	if len(ds) != 1 || ds[0].ID != "good" {
		t.Fatalf("ListSessions = %+v, want only the parseable [good]", ds)
	}
	if g.batchReadCalls() != 1 {
		t.Fatalf("expected one batched cat-file, got %d", g.batchReadCalls())
	}
}

// TestListSessionsBatchFillsIDFromRef proves the id-fill rule survives the batch path: a
// stored blob that OMITS the id key comes back with ID derived from the ref basename.
func TestListSessionsBatchFillsIDFromRef(t *testing.T) {
	g := newFakeGit()
	s := NewWithStdinRunner(g.run, g.runStdin, "")
	// A stored blob with no id key, pointed at by refs/fak/locks/session-derived.
	g.blobs["noidoid"] = []byte(`{"host":"h","pcb_state":"RUNNING","updated_at":7,"ttl_seconds":0}`)
	g.refs["refs/fak/locks/session-derived"] = "noidoid"

	ds, err := s.ListSessions(ctx())
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(ds) != 1 || ds[0].ID != "derived" {
		t.Fatalf("ListSessions = %+v, want a single descriptor with ID derived from the ref name", ds)
	}
	if ds[0].Host != "h" || ds[0].UpdatedAt != 7 {
		t.Fatalf("id-filled descriptor lost fields: %+v", ds[0])
	}
}

// TestListSessionsBatchEmpty proves an empty store returns an empty slice with no error, and
// spawns NO cat-file at all (there is nothing to read).
func TestListSessionsBatchEmpty(t *testing.T) {
	g := newFakeGit()
	s := NewWithStdinRunner(g.run, g.runStdin, "")

	ds, err := s.ListSessions(ctx())
	if err != nil {
		t.Fatalf("ListSessions over an empty store: %v", err)
	}
	if len(ds) != 0 {
		t.Fatalf("ListSessions over an empty store = %+v, want empty", ds)
	}
	if g.batchReadCalls() != 0 {
		t.Fatalf("empty store issued %d cat-file --batch calls, want 0 (nothing to read)", g.batchReadCalls())
	}
}

// TestListSessionsBatchExcludesNonSessionRefs proves the namespace split holds even when the
// namespace has ONLY lock leases: with no session refs, the batch reader returns an empty
// view and never spawns cat-file (the filter is applied before the batch, as it must be).
func TestListSessionsBatchExcludesNonSessionRefs(t *testing.T) {
	g := newFakeGit()
	s := NewWithStdinRunner(g.run, g.runStdin, "")
	if _, err := s.Acquire(ctx(), Record{ID: "only-a-lease", TreeGlobs: []string{"a"}, Holder: "h", AcquiredAt: 1, TTLSeconds: 0}); err != nil {
		t.Fatalf("Acquire: %v", err)
	}

	g.calls = nil
	ds, err := s.ListSessions(ctx())
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(ds) != 0 {
		t.Fatalf("ListSessions over a lease-only namespace = %+v, want empty (session split)", ds)
	}
	if g.batchReadCalls() != 0 {
		t.Fatalf("a lease-only namespace spawned %d cat-file --batch calls, want 0", g.batchReadCalls())
	}
}

// TestParseSessionBatchSemantics pins the record-stream parser directly: a `missing` status
// line and a non-JSON blob are both SKIPPED, an id-less blob has its ID filled from the ref
// name, and a payload containing a literal newline is read WHOLE by its declared byte count
// (a line-splitting parser would corrupt it). This is the load-bearing correctness of the
// batch decode, isolated from git.
func TestParseSessionBatchSemantics(t *testing.T) {
	refs := []string{
		"refs/fak/locks/session-a",
		"refs/fak/locks/session-gone",
		"refs/fak/locks/session-corrupt",
		"refs/fak/locks/session-noid",
	}
	var b strings.Builder
	rec := func(oid, payload string) { fmt.Fprintf(&b, "%s blob %d\n%s\n", oid, len(payload), payload) }
	rec(strings.Repeat("a", 40), `{"id":"a","host":"h","pcb_state":"RUNNING","updated_at":1,"ttl_seconds":0}`)
	b.WriteString("refs/fak/locks/session-gone missing\n") // no payload
	rec(strings.Repeat("b", 40), "not json {{{")
	// Pretty-printed JSON with REAL newlines and no id: byte-count parsing must read it whole,
	// then the ID is filled from the ref basename.
	rec(strings.Repeat("c", 40), "{\n  \"host\": \"h\",\n  \"pcb_state\": \"PAUSED\"\n}")

	got := parseSessionBatch(b.String(), refs)
	if len(got) != 2 {
		t.Fatalf("parseSessionBatch = %+v, want 2 (missing + corrupt skipped)", got)
	}
	if got[0].ID != "a" || got[0].Host != "h" {
		t.Fatalf("first descriptor = %+v, want id=a host=h", got[0])
	}
	if got[1].ID != "noid" || got[1].PCBState != "PAUSED" || got[1].Host != "h" {
		t.Fatalf("id-less descriptor = %+v, want ID filled to noid, PCBState=PAUSED, host=h", got[1])
	}
}

// TestListSessionsBatchRealGit exercises the batched read against the REAL git binary in a
// temp repo (skipped when git is unavailable). It proves the actual `git cat-file --batch`
// plumbing returns exactly the published descriptors, in sorted order, with a coexisting lock
// lease excluded — the strongest parity witness for #5355 Half B.
func TestListSessionsBatchRealGit(t *testing.T) {
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

	// Wire BOTH the real plain runner and the real stdin runner so ListSessions exercises the
	// actual `git cat-file --batch` plumbing (not the per-ref fallback).
	s := NewWithStdinRunner(gitRunner, gitStdinRunner, dir)
	want := []SessionDescriptor{
		{ID: "alpha", Host: "n1", PCBState: "RUNNING", UpdatedAt: 100, TTLSecs: 0},
		{ID: "bravo", Host: "n2", PCBState: "PAUSED", UpdatedAt: 200, TTLSecs: 600, AgentUUID: "1e21323a-b92d-4b43-a495-1e0c1d46f3ef"},
		{ID: "charlie", Host: "n3", PCBState: "DRAINING", UpdatedAt: 300, TTLSecs: 0},
	}
	for _, d := range want {
		if _, err := s.PublishSession(ctx(), d); err != nil {
			t.Fatalf("PublishSession %s: %v", d.ID, err)
		}
	}
	// A lock lease must be EXCLUDED from the session view even through the batch path.
	if _, err := s.Acquire(ctx(), Record{ID: "lease-1", TreeGlobs: []string{"a"}, Holder: "h", AcquiredAt: 1, TTLSeconds: 0}); err != nil {
		t.Fatalf("Acquire: %v", err)
	}

	got, err := s.ListSessions(ctx())
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("ListSessions returned %d descriptors, want %d (lock lease excluded): %+v", len(got), len(want), got)
	}
	for i, d := range want {
		if got[i] != d {
			t.Fatalf("descriptor %d via real cat-file --batch = %+v, want %+v", i, got[i], d)
		}
	}
}
