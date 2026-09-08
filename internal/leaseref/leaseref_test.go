package leaseref

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"reflect"
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
			cur, ok := f.refs[ref]
			if !ok {
				return "", 1, nil // delete of a missing ref exits non-zero (real git behavior)
			}
			// CAS form `update-ref -d <ref> <old>`: honor the old-value guarded delete the
			// fenced release (casDelete) relies on — a ref that advanced since <old> was
			// read survives the delete (exit non-zero). The 2-arg blind form is unchanged.
			if len(args) >= 4 && cur != args[3] {
				return "", 1, nil
			}
			delete(f.refs, ref)
			return "", 0, nil
		}
		ref, newval := args[1], args[2]
		// CAS form `update-ref <ref> <new> <old>`: honor the old-value compare-and-swap the
		// fence write (casWrite) relies on. An all-zeros <old> is git's "must not exist"
		// sentinel; any other <old> must equal the ref's current object id, else the CAS is
		// lost (exit non-zero). The 3-arg blind form (Acquire) skips this — unchanged.
		if len(args) >= 4 {
			old := args[3]
			cur, exists := f.refs[ref]
			if isAllZeros(old) {
				if exists {
					return "", 1, nil // must-not-exist violated: a peer created it first
				}
			} else if !exists || cur != old {
				return "", 1, nil // old-value mismatch: the ref advanced under the writer
			}
		}
		f.refs[ref] = newval
		return "", 0, nil
	case "rev-parse":
		// rev-parse --show-object-format -> the repo's hash algorithm (zeroOID sizing).
		if len(args) == 2 && args[1] == "--show-object-format" {
			return "sha1\n", 0, nil
		}
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
		blob, ok := f.blobs[id]
		if !ok {
			return "", 1, nil
		}
		return string(blob), 0, nil
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

// runStdin models `git update-ref --stdin`: it records the argv (so a test can prove the
// BATCHED path issued exactly one process, not one spawn per ref) and executes atomic
// multi-command transactions (`verify`, `update`, `create`, `delete`) as well as `cat-file --batch`.
// A failed verification causes the entire transaction to be aborted (changes neither ref).
// A missing ref in an unqualified `delete <ref>` is an idempotent no-op that never aborts.
func (f *fakeGit) runStdin(ctx context.Context, dir, stdin string, args ...string) (string, int, error) {
	f.calls = append(f.calls, args)
	isUpdateRefStdin := false
	if len(args) >= 2 && args[0] == "update-ref" {
		for _, a := range args[1:] {
			if a == "--stdin" {
				isUpdateRefStdin = true
				break
			}
		}
	}
	if isUpdateRefStdin {
		type txOp struct {
			cmd    string
			ref    string
			newOID string
			oldOID string
			hasOld bool
		}

		applyTx := func(ops []txOp) int {
			if len(ops) == 0 {
				return 0
			}
			seenRefs := make(map[string]bool, len(ops))
			for _, op := range ops {
				if seenRefs[op.ref] {
					return 128 // Git rejects multiple updates for the same ref in one transaction
				}
				seenRefs[op.ref] = true
			}

			// Pre-condition verification (all-or-nothing check)
			for _, op := range ops {
				cur, exists := f.refs[op.ref]
				switch op.cmd {
				case "verify":
					if !op.hasOld || isAllZeros(op.oldOID) {
						if exists {
							return 128
						}
					} else {
						if !exists || cur != op.oldOID {
							return 128
						}
					}
				case "create":
					if exists {
						return 128
					}
				case "delete":
					if op.hasOld {
						if isAllZeros(op.oldOID) {
							if exists {
								return 128
							}
						} else {
							if !exists || cur != op.oldOID {
								return 128
							}
						}
					}
				case "update":
					if op.hasOld {
						if isAllZeros(op.oldOID) {
							if exists {
								return 128
							}
						} else {
							if !exists || cur != op.oldOID {
								return 128
							}
						}
					}
				default:
					return 128
				}
			}

			// Apply mutations
			for _, op := range ops {
				switch op.cmd {
				case "create", "update":
					f.refs[op.ref] = op.newOID
				case "delete":
					delete(f.refs, op.ref)
				case "verify":
					// verify mutates nothing
				}
			}
			return 0
		}

		var currentTx []txOp
		inExplicitTx := false
		var out strings.Builder

		for _, rawLine := range strings.Split(stdin, "\n") {
			line := strings.TrimSpace(rawLine)
			if line == "" {
				continue
			}
			fields := strings.Fields(line)
			switch fields[0] {
			case "start":
				inExplicitTx = true
				currentTx = nil
				out.WriteString("start: ok\n")
			case "prepare":
				out.WriteString("prepare: ok\n")
			case "commit":
				code := applyTx(currentTx)
				currentTx = nil
				inExplicitTx = false
				if code != 0 {
					return out.String(), code, nil
				}
				out.WriteString("commit: ok\n")
			case "abort":
				currentTx = nil
				inExplicitTx = false
				out.WriteString("abort: ok\n")
			case "option":
				continue
			case "update":
				if len(fields) < 3 || len(fields) > 4 {
					return "", 128, nil
				}
				op := txOp{cmd: "update", ref: fields[1], newOID: fields[2]}
				if len(fields) == 4 {
					op.oldOID = fields[3]
					op.hasOld = true
				}
				currentTx = append(currentTx, op)
			case "create":
				if len(fields) != 3 {
					return "", 128, nil
				}
				currentTx = append(currentTx, txOp{cmd: "create", ref: fields[1], newOID: fields[2]})
			case "delete":
				if len(fields) < 2 || len(fields) > 3 {
					return "", 128, nil
				}
				op := txOp{cmd: "delete", ref: fields[1]}
				if len(fields) == 3 {
					op.oldOID = fields[2]
					op.hasOld = true
				}
				currentTx = append(currentTx, op)
			case "verify":
				if len(fields) < 2 || len(fields) > 3 {
					return "", 128, nil
				}
				op := txOp{cmd: "verify", ref: fields[1]}
				if len(fields) == 3 {
					op.oldOID = fields[2]
					op.hasOld = true
				}
				currentTx = append(currentTx, op)
			default:
				return "", 128, nil
			}
		}

		if !inExplicitTx && len(currentTx) > 0 {
			code := applyTx(currentTx)
			if code != 0 {
				return "", code, nil
			}
		}
		return out.String(), 0, nil
	}
	// cat-file --batch: read one ref/oid per stdin line, emit the real git --batch record
	// format so the batched session reader is exercised end to end. A resolvable ref yields
	// `<oid> blob <size>\n<payload>\n`; an unresolvable one yields `<object> missing\n` with
	// no payload — exactly what real git streams, in the same order the inputs were fed.
	if len(args) >= 2 && args[0] == "cat-file" && args[1] == "--batch" {
		var b strings.Builder
		for _, line := range strings.Split(stdin, "\n") {
			ref := strings.TrimSpace(line)
			if ref == "" {
				continue
			}
			id, ok := f.refs[ref]
			if !ok {
				b.WriteString(ref + " missing\n")
				continue
			}
			blob, ok := f.blobs[id]
			if !ok {
				b.WriteString(ref + " missing\n")
				continue
			}
			fmt.Fprintf(&b, "%s blob %d\n", id, len(blob))
			b.Write(blob)
			b.WriteByte('\n')
		}
		return b.String(), 0, nil
	}
	return "", 0, nil
}

// stdinDeleteCalls counts the `git update-ref --stdin` (batched) invocations recorded on
// the fake, the witness that the reaper drained the backlog in ONE process spawn.
func (f *fakeGit) stdinDeleteCalls() int {
	n := 0
	for _, c := range f.calls {
		if len(c) >= 2 && c[0] == "update-ref" && c[1] == "--stdin" {
			n++
		}
	}
	return n
}

// perRefDeleteCalls counts the serial `git update-ref -d <ref>` spawns — the per-ref cost
// #4990 root cause #2 removes. Zero of these after a batched reap is the fix's witness.
func (f *fakeGit) perRefDeleteCalls() int {
	n := 0
	for _, c := range f.calls {
		if len(c) >= 2 && c[0] == "update-ref" && c[1] == "-d" {
			n++
		}
	}
	return n
}

// isAllZeros reports whether s is a non-empty all-zeros object id — git's update-ref
// "must not exist" CAS sentinel. Used by the fake's update-ref to model the create CAS.
func isAllZeros(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] != '0' {
			return false
		}
	}
	return true
}

// synthID renders a small int as a stable synthetic object id for the fake object store.
func synthID(n int) string { return strings.Repeat("0", 39-len(itoa(n))) + "1" + itoa(n) }

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

func TestAcquireRejectsInvalidHashObjectIDBeforeUpdateRef(t *testing.T) {
	var calls [][]string
	run := func(_ context.Context, _ string, args ...string) (string, int, error) {
		calls = append(calls, append([]string(nil), args...))
		if args[0] == "hash-object" {
			return string(make([]byte, 41)), 0, nil
		}
		return "", 0, nil
	}
	_, err := NewWithRunner(run, "").Acquire(context.Background(), Record{ID: "writer-invalid-oid", Holder: "test"})
	if err == nil || !strings.Contains(err.Error(), "invalid object id") {
		t.Fatalf("Acquire error = %v, want invalid object id", err)
	}
	for _, call := range calls {
		if call[0] == "update-ref" {
			t.Fatalf("invalid oid reached update-ref: %v", call)
		}
	}
}

// TestStoreLiveBatchedCatFile witnesses that Store.Live (and Store.List) reads a backlog
// of 100+ synthetic lease refs in O(1) git processes: exactly one for-each-ref and one
// cat-file --batch (zero per-ref cat-file blob spawns). It verifies identical record values,
// expiration checks, malformed ref skipping, and namespace partitioning against the reference
// fallback per-ref reader.
func TestStoreLiveBatchedCatFile(t *testing.T) {
	g := newFakeGit()
	now := time.Unix(2000000000, 0)

	// Populate 100+ synthetic lease refs:
	// - 70 active lease records (future TTL)
	// - 35 expired lease records (past AcquiredAt + short TTL)
	// - 10 active records with no ID in stored JSON (proves ID is populated from ref name)
	// - 5 malformed / non-JSON blobs (proves corrupt refs are skipped cleanly)
	// - 5 missing refs (refs pointing to non-existent blob OIDs, proves missing is skipped)
	// - 15 non-lease refs under the same refs/fak/locks/ prefix (session-, intent-, contract-)
	// Total lock lease refs = 125.
	for i := 0; i < 70; i++ {
		id := fmt.Sprintf("lease-active-%03d", i)
		rec := Record{
			ID:         id,
			TreeGlobs:  []string{fmt.Sprintf("internal/pkg%d/**", i)},
			Holder:     fmt.Sprintf("node-a:worker-%d", i),
			AcquiredAt: now.Unix(),
			TTLSeconds: 3600,
		}
		b, err := json.Marshal(rec)
		if err != nil {
			t.Fatalf("marshal active record: %v", err)
		}
		oid := fmt.Sprintf("oid-active-%03d", i)
		g.blobs[oid] = b
		g.refs[refPrefix+id] = oid
	}

	for i := 0; i < 35; i++ {
		id := fmt.Sprintf("lease-expired-%03d", i)
		rec := Record{
			ID:         id,
			TreeGlobs:  []string{fmt.Sprintf("internal/old%d/**", i)},
			Holder:     fmt.Sprintf("node-b:worker-%d", i),
			AcquiredAt: now.Unix() - 500,
			TTLSeconds: 60,
		}
		b, err := json.Marshal(rec)
		if err != nil {
			t.Fatalf("marshal expired record: %v", err)
		}
		oid := fmt.Sprintf("oid-expired-%03d", i)
		g.blobs[oid] = b
		g.refs[refPrefix+id] = oid
	}

	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("lease-noid-%03d", i)
		b := []byte(fmt.Sprintf(`{"tree_globs":["internal/noid%d/**"],"holder":"node-c:worker-%d","acquired_unix":%d,"ttl_seconds":3600}`, i, i, now.Unix()))
		oid := fmt.Sprintf("oid-noid-%03d", i)
		g.blobs[oid] = b
		g.refs[refPrefix+id] = oid
	}

	for i := 0; i < 5; i++ {
		id := fmt.Sprintf("lease-corrupt-%03d", i)
		oid := fmt.Sprintf("oid-corrupt-%03d", i)
		g.blobs[oid] = []byte("not valid json {{{")
		g.refs[refPrefix+id] = oid
	}

	for i := 0; i < 5; i++ {
		id := fmt.Sprintf("lease-missing-%03d", i)
		g.refs[refPrefix+id] = fmt.Sprintf("oid-missing-%03d", i) // not in g.blobs
	}

	// Non-lease refs in the same namespace partition
	for i := 0; i < 5; i++ {
		g.blobs[fmt.Sprintf("oid-session-%d", i)] = []byte(`{"id":"s"}`)
		g.refs[fmt.Sprintf("refs/fak/locks/session-s%d", i)] = fmt.Sprintf("oid-session-%d", i)
		g.blobs[fmt.Sprintf("oid-intent-%d", i)] = []byte(`{"target_key":"k"}`)
		g.refs[fmt.Sprintf("refs/fak/locks/intent-i%d", i)] = fmt.Sprintf("oid-intent-%d", i)
		g.blobs[fmt.Sprintf("oid-contract-%d", i)] = []byte(`{"ticket_id":"t"}`)
		g.refs[fmt.Sprintf("refs/fak/locks/contract-c%d", i)] = fmt.Sprintf("oid-contract-%d", i)
	}

	// Counting runner wrapping fakeGit
	var procCount, forEachCalls, batchCalls, perRefBlobCalls int
	countingRun := func(ctx context.Context, dir string, args ...string) (string, int, error) {
		procCount++
		if len(args) >= 1 && args[0] == "for-each-ref" {
			forEachCalls++
		}
		if len(args) >= 2 && args[0] == "cat-file" && args[1] == "blob" {
			perRefBlobCalls++
		}
		return g.run(ctx, dir, args...)
	}
	countingStdin := func(ctx context.Context, dir, stdin string, args ...string) (string, int, error) {
		procCount++
		if len(args) >= 2 && args[0] == "cat-file" && args[1] == "--batch" {
			batchCalls++
		}
		return g.runStdin(ctx, dir, stdin, args...)
	}

	store := NewWithStdinRunner(countingRun, countingStdin, "")

	live, expired, err := store.Live(ctx(), now)
	if err != nil {
		t.Fatalf("Live failed: %v", err)
	}

	// O(1) process witness: exactly 1 for-each-ref + 1 cat-file --batch, 0 per-ref cat-file blob
	if forEachCalls != 1 {
		t.Fatalf("for-each-ref calls = %d, want exactly 1", forEachCalls)
	}
	if batchCalls != 1 {
		t.Fatalf("cat-file --batch calls = %d, want exactly 1 (batched pipeline)", batchCalls)
	}
	if perRefBlobCalls != 0 {
		t.Fatalf("per-ref cat-file blob calls = %d, want 0 (eliminated per-ref spawns)", perRefBlobCalls)
	}
	if procCount != 2 {
		t.Fatalf("total git processes spawned = %d, want exactly 2", procCount)
	}

	// Verify counts:
	// 70 active + 10 noid (which are active and had ID populated) = 80 live
	// 35 expired
	// 5 corrupt skipped, 5 missing skipped, 15 non-lease refs excluded
	if len(live) != 80 {
		t.Fatalf("live count = %d, want 80", len(live))
	}
	if len(expired) != 35 {
		t.Fatalf("expired count = %d, want 35", len(expired))
	}

	// Verify that live records are sorted by ID
	for i := 1; i < len(live); i++ {
		if live[i-1].ID >= live[i].ID {
			t.Fatalf("live records not sorted: live[%d]=%s >= live[%d]=%s", i-1, live[i-1].ID, i, live[i].ID)
		}
	}

	// Verify that noid records had ID properly filled from the ref name
	var noidCount int
	for _, r := range live {
		if strings.HasPrefix(r.ID, "lease-noid-") {
			noidCount++
			if r.Holder == "" || len(r.TreeGlobs) == 0 {
				t.Fatalf("noid record lost fields: %+v", r)
			}
		}
	}
	if noidCount != 10 {
		t.Fatalf("noid records found = %d, want 10", noidCount)
	}

	// Verify parity with the per-ref fallback path
	fallbackStore := NewWithRunner(g.run, "")
	wantLive, wantExpired, err := fallbackStore.Live(ctx(), now)
	if err != nil {
		t.Fatalf("fallback Live failed: %v", err)
	}
	if !reflect.DeepEqual(live, wantLive) {
		t.Fatalf("batched live records mismatch fallback path:\ngot  %+v\nwant %+v", live, wantLive)
	}
	if !reflect.DeepEqual(expired, wantExpired) {
		t.Fatalf("batched expired records mismatch fallback path:\ngot  %+v\nwant %+v", expired, wantExpired)
	}

	// Integration subtest against real git
	t.Run("RealGit", func(t *testing.T) {
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

		// Write blobs and create 100+ refs using update-ref --stdin
		sBlob, err := (&Store{run: gitRunner, dir: dir}).writeBlob(ctx(), []byte(`{"tree_globs":["a/**"],"holder":"h","acquired_unix":2000000000,"ttl_seconds":3600}`))
		if err != nil {
			t.Fatalf("writeBlob active: %v", err)
		}
		sExpBlob, err := (&Store{run: gitRunner, dir: dir}).writeBlob(ctx(), []byte(`{"tree_globs":["b/**"],"holder":"h","acquired_unix":100,"ttl_seconds":10}`))
		if err != nil {
			t.Fatalf("writeBlob expired: %v", err)
		}
		sCorruptBlob, err := (&Store{run: gitRunner, dir: dir}).writeBlob(ctx(), []byte("not json {{{"))
		if err != nil {
			t.Fatalf("writeBlob corrupt: %v", err)
		}

		var updateStdin strings.Builder
		for i := 0; i < 70; i++ {
			fmt.Fprintf(&updateStdin, "create refs/fak/locks/lease-rg-active-%03d %s\n", i, sBlob)
		}
		for i := 0; i < 35; i++ {
			fmt.Fprintf(&updateStdin, "create refs/fak/locks/lease-rg-expired-%03d %s\n", i, sExpBlob)
		}
		for i := 0; i < 5; i++ {
			fmt.Fprintf(&updateStdin, "create refs/fak/locks/lease-rg-corrupt-%03d %s\n", i, sCorruptBlob)
		}
		// Also non-lease session refs
		for i := 0; i < 5; i++ {
			fmt.Fprintf(&updateStdin, "create refs/fak/locks/session-rg-%03d %s\n", i, sBlob)
		}

		c := exec.Command("git", "update-ref", "--stdin")
		c.Dir = dir
		c.Stdin = strings.NewReader(updateStdin.String())
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("update-ref --stdin: %v\n%s", err, out)
		}

		var realProcCount, realForEach, realBatch, realPerRef int
		rgRun := func(ctx context.Context, d string, args ...string) (string, int, error) {
			realProcCount++
			if len(args) >= 1 && args[0] == "for-each-ref" {
				realForEach++
			}
			if len(args) >= 2 && args[0] == "cat-file" && args[1] == "blob" {
				realPerRef++
			}
			return gitRunner(ctx, d, args...)
		}
		rgStdin := func(ctx context.Context, d, stdin string, args ...string) (string, int, error) {
			realProcCount++
			if len(args) >= 2 && args[0] == "cat-file" && args[1] == "--batch" {
				realBatch++
			}
			return gitStdinRunner(ctx, d, stdin, args...)
		}

		rgStore := NewWithStdinRunner(rgRun, rgStdin, dir)
		rgLive, rgExpired, err := rgStore.Live(ctx(), now)
		if err != nil {
			t.Fatalf("real git Live: %v", err)
		}
		if realForEach != 1 || realBatch != 1 || realPerRef != 0 || realProcCount != 2 {
			t.Fatalf("real git process counts: for-each=%d, batch=%d, per-ref=%d, total=%d; want 1, 1, 0, 2", realForEach, realBatch, realPerRef, realProcCount)
		}
		if len(rgLive) != 70 {
			t.Fatalf("real git live count = %d, want 70", len(rgLive))
		}
		if len(rgExpired) != 35 {
			t.Fatalf("real git expired count = %d, want 35", len(rgExpired))
		}
	})
}

// TestParseRecordBatchSemantics pins the record-stream parser directly: a `missing` status
// line and a non-JSON blob are both SKIPPED, an id-less blob has its ID filled from the ref
// name, and a payload containing a literal newline is read WHOLE by its declared byte count
// (a line-splitting parser would corrupt it).
func TestParseRecordBatchSemantics(t *testing.T) {
	refs := []string{
		"refs/fak/locks/lease-a",
		"refs/fak/locks/lease-gone",
		"refs/fak/locks/lease-corrupt",
		"refs/fak/locks/lease-noid",
	}
	var b strings.Builder
	rec := func(oid, payload string) { fmt.Fprintf(&b, "%s blob %d\n%s\n", oid, len(payload), payload) }
	rec(strings.Repeat("a", 40), `{"id":"lease-a","holder":"h1","tree_globs":["pkg/a/**"],"acquired_unix":1,"ttl_seconds":0}`)
	b.WriteString("refs/fak/locks/lease-gone missing\n")
	rec(strings.Repeat("b", 40), "not valid json {{{")
	rec(strings.Repeat("c", 40), "{\n  \"holder\": \"h2\",\n  \"tree_globs\": [\"pkg/b/**\"],\n  \"acquired_unix\": 2\n}")

	got := parseRecordBatch(b.String(), refs)
	if len(got) != 2 {
		t.Fatalf("parseRecordBatch returned %d records, want 2 (missing + corrupt skipped)", len(got))
	}
	if got[0].ID != "lease-a" || got[0].Holder != "h1" || len(got[0].TreeGlobs) != 1 || got[0].TreeGlobs[0] != "pkg/a/**" {
		t.Fatalf("first record = %+v, want lease-a with tree_globs", got[0])
	}
	if got[1].ID != "lease-noid" || got[1].Holder != "h2" || len(got[1].TreeGlobs) != 1 || got[1].TreeGlobs[0] != "pkg/b/**" {
		t.Fatalf("id-less record = %+v, want ID filled to lease-noid", got[1])
	}
}

// TestLeaseRunnerAtomicTransactionParity compares a real Git transaction (git update-ref --stdin)
// with the fakeGit fixture runner (g.runStdin):
// - A failed expected-OID check (verify <ref> <wrong-oid>) changes neither ref (atomic rollback / no commit).
// - A valid transaction (update <ref1> <new1> <old1>, update <ref2> <new2> <old2>) commits both changes atomically.
func TestLeaseRunnerAtomicTransactionParity(t *testing.T) {
	t.Run("FakeGit", func(t *testing.T) {
		g := newFakeGit()
		s := NewWithStdinRunner(g.run, g.runStdin, "")
		testAtomicTransactionParity(t, s)
	})
	t.Run("RealGit", func(t *testing.T) {
		dir := initRealGitRepo(t)
		s := NewInDir(dir)
		testAtomicTransactionParity(t, s)
	})
}

func testAtomicTransactionParity(t *testing.T, s *Store) {
	t.Helper()
	c := ctx()

	// Write 4 distinct blobs to point refs to.
	oid1, err := s.writeBlob(c, []byte("val-1"))
	if err != nil {
		t.Fatalf("writeBlob oid1: %v", err)
	}
	oid2, err := s.writeBlob(c, []byte("val-2"))
	if err != nil {
		t.Fatalf("writeBlob oid2: %v", err)
	}
	new1, err := s.writeBlob(c, []byte("val-new-1"))
	if err != nil {
		t.Fatalf("writeBlob new1: %v", err)
	}
	new2, err := s.writeBlob(c, []byte("val-new-2"))
	if err != nil {
		t.Fatalf("writeBlob new2: %v", err)
	}

	ref1 := "refs/fak/locks/parity-lane-1"
	ref2 := "refs/fak/locks/parity-lane-2"

	// Initial setup: point ref1 -> oid1 and ref2 -> oid2.
	initPayload := fmt.Sprintf("update %s %s\nupdate %s %s\n", ref1, oid1, ref2, oid2)
	out, code, err := s.runStdin(c, s.dir, initPayload, "update-ref", "--stdin")
	if err != nil || code != 0 {
		t.Fatalf("initial setup failed: code=%d err=%v out=%s", code, err, out)
	}

	cur1, ok1, err := s.currentOID(c, ref1)
	if err != nil || !ok1 || cur1 != oid1 {
		t.Fatalf("ref1 not initialized: ok=%v cur=%q want=%q err=%v", ok1, cur1, oid1, err)
	}
	cur2, ok2, err := s.currentOID(c, ref2)
	if err != nil || !ok2 || cur2 != oid2 {
		t.Fatalf("ref2 not initialized: ok=%v cur=%q want=%q err=%v", ok2, cur2, oid2, err)
	}

	// 1. A failed expected-OID check (verify <ref> <wrong-oid>) changes neither ref (atomic rollback / no commit).
	wrongOID := strings.Repeat("a", len(oid2))
	if wrongOID == oid2 {
		wrongOID = strings.Repeat("b", len(oid2))
	}
	failTx := fmt.Sprintf("update %s %s %s\nverify %s %s\n", ref1, new1, oid1, ref2, wrongOID)
	out, code, err = s.runStdin(c, s.dir, failTx, "update-ref", "--stdin")
	if err != nil {
		t.Fatalf("runStdin unexpected execution error: %v", err)
	}
	if code == 0 {
		t.Fatalf("expected non-zero exit code for failed verify in transaction, got 0; out=%s", out)
	}

	// Neither ref must have changed.
	cur1, ok1, err = s.currentOID(c, ref1)
	if err != nil || !ok1 || cur1 != oid1 {
		t.Fatalf("atomic rollback failed: ref1 changed to %q (want %q), ok=%v err=%v", cur1, oid1, ok1, err)
	}
	cur2, ok2, err = s.currentOID(c, ref2)
	if err != nil || !ok2 || cur2 != oid2 {
		t.Fatalf("atomic rollback failed: ref2 changed to %q (want %q), ok=%v err=%v", cur2, oid2, ok2, err)
	}

	// 2. A valid transaction (update <ref1> <new1> <old1>, update <ref2> <new2> <old2>) commits both changes atomically.
	validTx := fmt.Sprintf("update %s %s %s\nupdate %s %s %s\n", ref1, new1, oid1, ref2, new2, oid2)
	out, code, err = s.runStdin(c, s.dir, validTx, "update-ref", "--stdin")
	if err != nil || code != 0 {
		t.Fatalf("valid transaction failed: code=%d err=%v out=%s", code, err, out)
	}

	cur1, ok1, err = s.currentOID(c, ref1)
	if err != nil || !ok1 || cur1 != new1 {
		t.Fatalf("valid transaction commit failed: ref1 is %q (want %q), ok=%v err=%v", cur1, new1, ok1, err)
	}
	cur2, ok2, err = s.currentOID(c, ref2)
	if err != nil || !ok2 || cur2 != new2 {
		t.Fatalf("valid transaction commit failed: ref2 is %q (want %q), ok=%v err=%v", cur2, new2, ok2, err)
	}
}
