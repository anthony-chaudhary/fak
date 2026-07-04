package session

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
)

// rewindTax is the dos.toml lane taxonomy the rewind tests arbitrate over
// (session non-exclusive with a tree, release/global exclusive — the same shape
// laneadmit_test exercises).
func rewindTax() laneadmit.Taxonomy {
	return laneadmit.Taxonomy{
		Loaded:    true,
		Exclusive: map[string]bool{"release": true, "global": true, "abi": true, "dos": true},
		Trees: map[string][]string{
			"session": {"internal/session/**"},
			"gateway": {"internal/gateway/**"},
			"release": {"VERSION", "docs/releases/**"},
		},
	}
}

// spyApplier is the bulk-write step a rewind would perform: it writes one file
// (the checkpoint's version of a workspace path) and records that it ran. A
// refusal must leave called=false and the workspace untouched.
type spyApplier struct {
	dir     string
	relpath string
	content []byte
	called  bool
}

func (s *spyApplier) Apply() error {
	s.called = true
	return os.WriteFile(filepath.Join(s.dir, s.relpath), s.content, 0o644)
}

type sliceJournal struct{ events []RewindEvent }

func (j *sliceJournal) Record(e RewindEvent) error { j.events = append(j.events, e); return nil }

// treeDigest hashes the sorted (name, content) pairs under dir — a stable
// workspace fingerprint. A refused restore must leave it byte-identical.
func treeDigest(t *testing.T, dir string) string {
	t.Helper()
	var files []string
	err := filepath.Walk(dir, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(dir, p)
		files = append(files, rel)
		return nil
	})
	if err != nil {
		t.Fatalf("walk %s: %v", dir, err)
	}
	sort.Strings(files)
	h := sha256.New()
	for _, rel := range files {
		b, err := os.ReadFile(filepath.Join(dir, rel))
		if err != nil {
			t.Fatalf("read %s: %v", rel, err)
		}
		h.Write([]byte(rel))
		h.Write([]byte{0})
		h.Write(b)
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

// TestRewindRefusedUnderPeerLease is the #2427 acceptance witness: with a live
// lease over an INTERSECTING tree, a workspace restore returns the arbiter's
// closed lane-conflict refusal (COLLISION_RISK) naming the holder, and zero
// files are modified — the tree digest is byte-identical before and after.
func TestRewindRefusedUnderPeerLease(t *testing.T) {
	dir := t.TempDir()
	peerWork := []byte("peer's just-pushed bytes")
	if err := os.WriteFile(filepath.Join(dir, "restore.go"), peerWork, 0o644); err != nil {
		t.Fatal(err)
	}
	digestBefore := treeDigest(t, dir)

	applier := &spyApplier{dir: dir, relpath: "restore.go", content: []byte("STALE checkpoint bytes")}
	journal := &sliceJournal{}

	// A live peer lease over an intersecting tree (the change set the restore
	// would touch is held by a fleet-mate — the #2320 failure class).
	v, err := Rewind(RewindInput{
		Holder:  "operator",
		Tree:    []string{"internal/session/restore.go"},
		Leases:  []laneadmit.Lease{{ID: "loop-lane-session", Lane: "session", Tree: []string{"internal/session/**"}, Holder: "peer-7"}},
		Taxonomy: rewindTax(),
		Applier: applier,
		Journal: journal,
		Now:     time.Unix(1_700_000_000, 0),
	})
	if err != nil {
		t.Fatalf("Rewind error: %v", err)
	}
	if v.Admit {
		t.Fatal("restore over a live intersecting lease must be REFUSED, got admit")
	}
	if v.Reason != laneadmit.ReasonCollisionRisk {
		t.Fatalf("refusal reason = %q, want the arbiter's closed lane-conflict reason %q",
			v.Reason, laneadmit.ReasonCollisionRisk)
	}
	if len(v.Conflicts) == 0 || v.Conflicts[0].Holder != "peer-7" {
		t.Fatalf("refusal must name the holding peer, got conflicts %+v", v.Conflicts)
	}
	if applier.called {
		t.Fatal("Applier ran on a refusal — the restore mutated the tree it was not admitted to")
	}
	if got := treeDigest(t, dir); got != digestBefore {
		t.Fatal("tree digest changed after a refused restore — zero files must be modified")
	}
	// The refusal was journaled with the closed reason.
	if len(journal.events) != 1 || journal.events[0].Kind != EvRewindRefused ||
		journal.events[0].Reason != laneadmit.ReasonCollisionRisk {
		t.Fatalf("journal = %+v, want one refused event carrying COLLISION_RISK", journal.events)
	}
}

// TestRewindProceedsDisjoint is the #2427 second acceptance witness: a lease
// over a DISJOINT tree does not block the restore, and the Applier runs.
func TestRewindProceedsDisjoint(t *testing.T) {
	dir := t.TempDir()
	applier := &spyApplier{dir: dir, relpath: "restore.go", content: []byte("fresh checkpoint bytes")}
	journal := &sliceJournal{}

	// A live peer lease over a disjoint tree (gateway, while the restore touches
	// session) — a fleet-mate is working, just not where the restore lands.
	v, err := Rewind(RewindInput{
		Holder:   "operator",
		Tree:     []string{"internal/session/restore.go"},
		Leases:   []laneadmit.Lease{{ID: "loop-lane-gateway", Lane: "gateway", Tree: []string{"internal/gateway/**"}, Holder: "peer-9"}},
		Taxonomy: rewindTax(),
		Applier:  applier,
		Journal:  journal,
		Now:      time.Unix(1_700_000_001, 0),
	})
	if err != nil {
		t.Fatalf("Rewind error: %v", err)
	}
	if !v.Admit {
		t.Fatalf("restore over a disjoint lease must proceed, got refusal %+v", v)
	}
	if !applier.called {
		t.Fatal("Applier must run once the change set is admitted")
	}
	got, err := os.ReadFile(filepath.Join(dir, "restore.go"))
	if err != nil || string(got) != "fresh checkpoint bytes" {
		t.Fatalf("restore applied %q, err=%v; want the checkpoint bytes written", got, err)
	}
	if len(journal.events) != 1 || journal.events[0].Kind != EvRewindAdmitted {
		t.Fatalf("journal = %+v, want one admitted event", journal.events)
	}
}

// TestRewindRefusalReasonResolvesInClosedVocabulary proves the #2427 acceptance
// "the refusal reason resolves via dos_check_reason": the rewind refuses with the
// arbiter's closed lane-conflict token, and that exact token is a declared
// [reasons.*] block in the repo dos.toml (the table dos_check_reason consults).
func TestRewindRefusalReasonResolvesInClosedVocabulary(t *testing.T) {
	if laneadmit.ReasonCollisionRisk != "COLLISION_RISK" {
		t.Fatalf("the rewind refuses with %q, want the literal COLLISION_RISK token", laneadmit.ReasonCollisionRisk)
	}
	data, err := os.ReadFile(filepath.Join("..", "..", "dos.toml"))
	if err != nil {
		t.Skipf("repo dos.toml unavailable: %v", err)
	}
	want := "[" + "reasons." + laneadmit.ReasonCollisionRisk + "]"
	if !strings.Contains(string(data), want) {
		t.Fatalf("repo dos.toml must declare %s so dos_check_reason resolves the rewind refusal reason", want)
	}
}

// TestRewindForceClearsOverlapButNotExclusive pins the operator force path: a
// force clears a geometric / same-lane conflict (the restore proceeds), but
// still refuses over a live EXCLUSIVE lane (the arbiter's force semantics).
func TestRewindForceClearsOverlapButNotExclusive(t *testing.T) {
	dir := t.TempDir()
	tax := rewindTax()

	// (1) Force over a plain tree-overlap clears it — the restore proceeds.
	overlapApplier := &spyApplier{dir: dir, relpath: "a.txt", content: []byte("ok")}
	v, err := Rewind(RewindInput{
		Holder: "operator",
		Tree:   []string{"internal/session/restore.go"},
		Leases: []laneadmit.Lease{{ID: "loop-lane-session", Lane: "session", Tree: []string{"internal/session/**"}, Holder: "peer"}},
		Taxonomy: tax,
		Force:   true,
		Applier: overlapApplier,
	})
	if err != nil {
		t.Fatalf("force overlap error: %v", err)
	}
	if !v.Admit || !v.Forced {
		t.Fatalf("force must clear a tree-overlap and proceed, got %+v", v)
	}
	if !overlapApplier.called {
		t.Fatal("forced-overlap Applier must run")
	}

	// (2) Force over a live EXCLUSIVE lane still refuses — a release lane runs
	// alone, and a forced restore must never race it.
	exclusiveApplier := &spyApplier{dir: dir, relpath: "b.txt", content: []byte("nope")}
	v, err = Rewind(RewindInput{
		Holder: "operator",
		Lane:   "release",
		Tree:   []string{"VERSION"},
		Leases: []laneadmit.Lease{{ID: "loop-lane-docs", Lane: "session", Tree: []string{"internal/session/**"}, Holder: "peer"}},
		Taxonomy: tax,
		Force:   true,
		Applier: exclusiveApplier,
	})
	if err != nil {
		t.Fatalf("force exclusive error: %v", err)
	}
	if v.Admit {
		t.Fatal("force over a live exclusive lane must STILL refuse")
	}
	if v.Reason != laneadmit.ReasonCollisionRisk {
		t.Fatalf("exclusive refusal reason = %q, want %q", v.Reason, laneadmit.ReasonCollisionRisk)
	}
	if exclusiveApplier.called {
		t.Fatal("exclusive-lane refusal must not run the Applier")
	}
}

// TestRewindOwnLeaseDoesNotConflict mirrors laneadmit's own-lease rule: a live
// lease whose id matches the rewind's LeaseID is the caller's own (a re-entrant
// restore) and never blocks.
func TestRewindOwnLeaseDoesNotConflict(t *testing.T) {
	dir := t.TempDir()
	applier := &spyApplier{dir: dir, relpath: "c.txt", content: []byte("own")}
	v, err := Rewind(RewindInput{
		Holder:  "operator",
		LeaseID: "restore-lane-session",
		Tree:    []string{"internal/session/restore.go"},
		Leases:  []laneadmit.Lease{{ID: "restore-lane-session", Lane: "session", Tree: []string{"internal/session/**"}, Holder: "operator"}},
		Taxonomy: rewindTax(),
		Applier: applier,
	})
	if err != nil {
		t.Fatalf("Rewind error: %v", err)
	}
	if !v.Admit {
		t.Fatalf("a caller's own lease must not conflict, got %+v", v)
	}
	if !applier.called {
		t.Fatal("own-lease restore must proceed")
	}
}
