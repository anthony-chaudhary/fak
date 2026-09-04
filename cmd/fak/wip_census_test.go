package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wiprecon"
	"github.com/anthony-chaudhary/fak/internal/wipref"
)

// wipCensusClassOf runs the full census over a real repo and returns the class the
// census assigned to session sess (or "" if that session has no verdict).
func wipCensusClassOf(t *testing.T, dir, sess string) wipref.CensusClass {
	t.Helper()
	rep, err := wipCensus(context.Background(), dir)
	if err != nil {
		t.Fatalf("wipCensus: %v", err)
	}
	for _, v := range rep.Verdicts {
		if v.Session == sess {
			return v.Class
		}
	}
	return ""
}

// commitWorkingTree stages every change and moves HEAD to a fresh plumbing commit —
// the way a session "lands" its delta into HEAD (verbatim or with further edits).
func commitWorkingTree(t *testing.T, dir, msg string) {
	t.Helper()
	ctx := context.Background()
	if _, err := gitWipOut(ctx, dir, nil, "add", "-A"); err != nil {
		t.Fatalf("git add: %v", err)
	}
	head, err := gitWipOut(ctx, dir, nil, "rev-parse", "HEAD")
	if err != nil {
		t.Fatalf("rev-parse HEAD: %v", err)
	}
	tree, err := gitWipOut(ctx, dir, nil, "write-tree")
	if err != nil {
		t.Fatalf("write-tree: %v", err)
	}
	commit, err := gitWipOut(ctx, dir, nil, "commit-tree", tree, "-p", head, "-m", msg)
	if err != nil {
		t.Fatalf("commit-tree: %v", err)
	}
	if _, err := gitWipOut(ctx, dir, nil, "update-ref", "HEAD", commit); err != nil {
		t.Fatalf("update-ref HEAD: %v", err)
	}
}

// TestWipCensusClassifiesTheVocabulary drives the read-only census end to end over a
// real repo holding one checkpoint of each interesting kind, and asserts each lands in
// the right owner-state — with the CLOSED_DIRTY_RECOVERABLE safety case front and
// center. It also proves the census NEVER deletes: every ref survives the pass.
//
// Each scenario uses a DISTINCT file so a later commit to one never disturbs another's
// verbatim-landing check (wipCheckpoint snapshots the whole tree, not one file), and the
// tree is returned to a clean state between checkpoints so each captures only its file.
func TestWipCensusClassifiesTheVocabulary(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipTestRepo(t)
	write := func(name, body string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	// --- LANDED: a checkpoint whose file is later committed to HEAD verbatim. ---
	write("la.txt", "L\n")
	if _, err := wipCheckpoint(ctx, dir, "landed", true, 1000); err != nil {
		t.Fatalf("checkpoint landed: %v", err)
	}
	commitWorkingTree(t, dir, "land la.txt verbatim") // HEAD now carries la.txt="L\n"; tree clean

	// --- CLOSED_DIRTY_RECOVERABLE: an unlanded dead-session delta (a new file that ---
	// --- never lands). The safety invariant: it MUST be kept, never called clean.    ---
	write("dr.txt", "recoverable\n")
	if _, err := wipCheckpoint(ctx, dir, "dirty", true, 1000); err != nil {
		t.Fatalf("checkpoint dirty: %v", err)
	}
	if err := os.Remove(filepath.Join(dir, "dr.txt")); err != nil { // never lands; tree clean again
		t.Fatal(err)
	}

	// --- DIVERGED (subsumed lines, but bytes differ): a checkpoint whose added ---
	// --- line lands in HEAD alongside an extra unrelated line, so it is NOT a        ---
	// --- verbatim landing (not LANDED) and payload bytes differ from HEAD.           ---
	write("su.txt", "S\n")
	if _, err := wipCheckpoint(ctx, dir, "subsumed", true, 1000); err != nil {
		t.Fatalf("checkpoint subsumed: %v", err)
	}
	write("su.txt", "S\nextra\n")                               // HEAD gets "S" plus an unrelated line
	commitWorkingTree(t, dir, "land su.txt with an extra line") // not byte-identical to the checkpoint

	// --- CLOSED_CLEAN_ESTIMATE: parentless checkpoint whose tree is byte-identical to HEAD. ---
	tree, err := gitWipOut(ctx, dir, nil, "rev-parse", "HEAD^{tree}")
	if err != nil {
		t.Fatal(err)
	}
	cleanObj, err := gitWipOut(ctx, dir, nil, "commit-tree", tree, "-m", "clean parentless")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := gitWipOut(ctx, dir, nil, "update-ref", "refs/fak/wip/clean", cleanObj); err != nil {
		t.Fatal(err)
	}

	classes := map[string]wipref.CensusClass{
		"landed":   wipCensusClassOf(t, dir, "landed"),
		"dirty":    wipCensusClassOf(t, dir, "dirty"),
		"subsumed": wipCensusClassOf(t, dir, "subsumed"),
		"clean":    wipCensusClassOf(t, dir, "clean"),
	}
	want := map[string]wipref.CensusClass{
		"landed":   wipref.CensusLanded,
		"dirty":    wipref.CensusClosedDirtyRecoverable,
		"subsumed": wipref.CensusDiverged,
		"clean":    wipref.CensusClosedCleanEstimate,
	}
	for sess, w := range want {
		if classes[sess] != w {
			t.Errorf("session %q classified %s, want %s", sess, classes[sess], w)
		}
	}

	// Read-only: the census deleted nothing — all four refs still resolve.
	for _, sess := range []string{"landed", "dirty", "subsumed", "clean"} {
		if _, has, err := wipCurrentOID(ctx, dir, wipref.SessionRef(sess)); err != nil || !has {
			t.Errorf("census deleted ref for %q (has=%v err=%v) — census must be read-only", sess, has, err)
		}
	}
}

// TestWipCensusJSONShape proves the --census --json wiring: exit 0, valid JSON, counts
// that sum to the total, and the headline CLOSED_CLEAN_ESTIMATE field present.
func TestWipCensusJSONShape(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipTestRepo(t)

	// One dead-session unlanded checkpoint -> exactly one CLOSED_DIRTY_RECOVERABLE.
	fresh := filepath.Join(dir, "fresh.txt")
	if err := os.WriteFile(fresh, []byte("base line\nrecoverable\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, dir, "solo", true, 1000); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if err := os.Remove(fresh); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	if rc := runWipReap(&out, &errb, []string{"-C", dir, "--census", "--json"}); rc != 0 {
		t.Fatalf("census --json rc=%d stderr=%s", rc, errb.String())
	}
	var rep wipref.CensusReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("census JSON invalid: %v\n%s", err, out.String())
	}
	c := rep.Counts
	if c.Total != 1 || c.ClosedDirtyRecoverable != 1 {
		t.Fatalf("counts = %+v, want Total=1 ClosedDirtyRecoverable=1", c)
	}
	sum := c.Landed + c.Live + c.ClosedCleanEstimate + c.ClosedDirtyRecoverable + c.Diverged + c.Unknown
	if sum != c.Total {
		t.Errorf("per-class counts sum to %d, want Total=%d", sum, c.Total)
	}
	if len(rep.Verdicts) != 1 || rep.Verdicts[0].Session != "solo" {
		t.Errorf("verdicts = %+v, want one for session 'solo'", rep.Verdicts)
	}
}

// TestWipCensusEmptyDeltaIsCleanEstimate proves a dead-session checkpoint that no
// longer differs from HEAD (its whole delta landed as part of a larger commit, so the
// checkpoint-tree-vs-HEAD content is fully present) is a CLOSED_CLEAN_ESTIMATE, and a
// brand-new untracked file that never landed stays CLOSED_DIRTY_RECOVERABLE.
func TestWipCensusNewFileIsRecoverable(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipTestRepo(t)

	// A checkpoint whose only content is a brand-new file that never lands.
	fresh := filepath.Join(dir, "brandnew.txt")
	if err := os.WriteFile(fresh, []byte("net-new unlanded work\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, dir, "newfile", true, 1000); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if err := os.Remove(fresh); err != nil {
		t.Fatal(err)
	}

	if got := wipCensusClassOf(t, dir, "newfile"); got != wipref.CensusClosedDirtyRecoverable {
		t.Fatalf("a never-landed new file classified %s, want CLOSED_DIRTY_RECOVERABLE (kept)", got)
	}
}

func TestRunWIPCensusNamesRecoverableBacklogAsActionNotGarbage(t *testing.T) {
	dir, _ := wipTestRepo(t)
	fresh := filepath.Join(dir, "recoverable.txt")
	if err := os.WriteFile(fresh, []byte("base line\nrecover me\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if r, err := wipCheckpoint(context.Background(), dir, "recoverable-wording", true, time.Now().Unix()); err != nil {
		t.Fatalf("checkpoint: %v (%+v)", err, r)
	}
	if err := os.Remove(fresh); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	if code := runWipCensus(context.Background(), &out, &stderr, dir, false); code != 0 {
		t.Fatalf("runWIPCensus code=%d stderr=%s", code, stderr.String())
	}
	text := out.String()
	for _, want := range []string{"recoverable unlanded deliverable", "ACTION REQUIRED", "recovery backlog (not reapable): 1", "reconcile --reclaim"} {
		if !strings.Contains(text, want) {
			t.Fatalf("census output missing %q:\n%s", want, text)
		}
	}
}

// TestWipPayloadReadingUnscopedWithParent proves an unscoped checkpoint with a parent
// derives touched paths from obj^..obj and correctly classifies absent, diverged, and landed paths.
func TestWipPayloadReadingUnscopedWithParent(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipTestRepo(t)

	landedFile := filepath.Join(dir, "landed.txt")
	divergedFile := filepath.Join(dir, "diverged.txt")
	absentFile := filepath.Join(dir, "absent.txt")

	if err := os.WriteFile(landedFile, []byte("landed v1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(divergedFile, []byte("diverged v1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	commitWorkingTree(t, dir, "base files")

	// Modify working tree for checkpoint:
	// - landed.txt modified to "landed v2\n"
	// - diverged.txt modified to "diverged checkpoint\n"
	// - absent.txt created with "absent checkpoint\n"
	if err := os.WriteFile(landedFile, []byte("landed v2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(divergedFile, []byte("diverged checkpoint\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(absentFile, []byte("absent checkpoint\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	res, err := wipCheckpoint(ctx, dir, "unscoped-parent", true, 1000)
	if err != nil {
		t.Fatalf("wipCheckpoint: %v", err)
	}

	// Advance HEAD:
	// - landed.txt updated to "landed v2\n" (matches checkpoint -> Landed)
	// - diverged.txt updated to "diverged head\n" (differs from checkpoint -> Diverged)
	// - absent.txt removed (does not exist in HEAD -> Absent)
	if err := os.WriteFile(divergedFile, []byte("diverged head\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(absentFile); err != nil {
		t.Fatal(err)
	}
	commitWorkingTree(t, dir, "advance HEAD")

	rec := wipref.RefRecord{
		Ref:    res.Ref,
		Object: res.Object,
		Stamp:  wipref.Stamp{Scope: nil},
	}
	reading := wipPayloadReading(ctx, dir, rec)
	if !reading.Read {
		t.Fatalf("reading unreadable: %s", reading.Unreadable)
	}
	payload := wipref.BuildPayloadCensus(reading)
	if !payload.Read {
		t.Fatalf("payload unreadable: %s", payload.Unreadable)
	}
	if payload.Files != 3 {
		t.Fatalf("payload.Files = %d, want 3; paths = %v", payload.Files, reading.Paths)
	}
	if payload.StateOf("absent.txt") != wipref.PayloadAbsent {
		t.Errorf("absent.txt state = %s, want %s", payload.StateOf("absent.txt"), wipref.PayloadAbsent)
	}
	if payload.StateOf("diverged.txt") != wipref.PayloadDiverged {
		t.Errorf("diverged.txt state = %s, want %s", payload.StateOf("diverged.txt"), wipref.PayloadDiverged)
	}
	if payload.StateOf("landed.txt") != wipref.PayloadLanded {
		t.Errorf("landed.txt state = %s, want %s", payload.StateOf("landed.txt"), wipref.PayloadLanded)
	}
	if payload.Landed != 1 {
		t.Errorf("payload.Landed = %d, want 1", payload.Landed)
	}
}

// TestWipPayloadReadingUnscopedParentless proves an unscoped parentless checkpoint derives
// its path list from git ls-tree -r rather than returning an empty reading.
func TestWipPayloadReadingUnscopedParentless(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "wip@test.local"},
		{"config", "user.name", "wip test"},
	} {
		if _, err := gitWipOut(ctx, dir, nil, args...); err != nil {
			t.Fatalf("git %v: %v", args, err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "p1.txt"), []byte("parentless 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "p2.txt"), []byte("parentless 2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := gitWipOut(ctx, dir, nil, "add", "p1.txt", "p2.txt"); err != nil {
		t.Fatal(err)
	}
	tree, err := gitWipOut(ctx, dir, nil, "write-tree")
	if err != nil {
		t.Fatal(err)
	}
	obj, err := gitWipOut(ctx, dir, nil, "commit-tree", tree, "-m", "parentless")
	if err != nil {
		t.Fatal(err)
	}

	rec := wipref.RefRecord{
		Object: obj,
		Stamp:  wipref.Stamp{Scope: nil},
	}
	reading := wipPayloadReading(ctx, dir, rec)
	if !reading.Read {
		t.Fatalf("reading unreadable: %s", reading.Unreadable)
	}
	payload := wipref.BuildPayloadCensus(reading)
	if !payload.Read {
		t.Fatalf("payload unreadable: %s", payload.Unreadable)
	}
	if payload.Files != 2 {
		t.Fatalf("payload.Files = %d, want 2; paths = %v", payload.Files, reading.Paths)
	}
	if payload.StateOf("p1.txt") != wipref.PayloadAbsent || payload.StateOf("p2.txt") != wipref.PayloadAbsent {
		t.Fatalf("expected both files absent from empty HEAD; got %+v", payload)
	}
}

// TestWipPayloadReadingExplicitScopeRemainsNarrowed proves explicit scope takes precedence
// over full commit diff / tree derivation.
func TestWipPayloadReadingExplicitScopeRemainsNarrowed(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipTestRepo(t)

	f1 := filepath.Join(dir, "scope1.txt")
	f2 := filepath.Join(dir, "scope2.txt")
	if err := os.WriteFile(f1, []byte("s1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(f2, []byte("s2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	res, err := wipCheckpointScoped(ctx, dir, "narrow-scope", true, 1000, []string{"scope1.txt"})
	if err != nil {
		t.Fatalf("wipCheckpointScoped: %v", err)
	}

	rec := wipref.RefRecord{
		Ref:    res.Ref,
		Object: res.Object,
		Stamp:  wipref.Stamp{Scope: []string{"scope1.txt"}},
	}
	reading := wipPayloadReading(ctx, dir, rec)
	if !reading.Read {
		t.Fatalf("reading unreadable: %s", reading.Unreadable)
	}
	if len(reading.Paths) != 1 || reading.Paths[0] != "scope1.txt" {
		t.Fatalf("reading.Paths = %v, want exactly ['scope1.txt']", reading.Paths)
	}
	payload := wipref.BuildPayloadCensus(reading)
	if payload.Files != 1 {
		t.Fatalf("payload.Files = %d, want 1", payload.Files)
	}
}

// TestWipReconcileQuarantineEmitsReapCensusJSON proves a QUARANTINE decision with no
// divergent/absent review commands directs the operator to `fak wip reap --census --json`.
func TestWipReconcileQuarantineEmitsReapCensusJSON(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipTestRepo(t)

	tree, err := gitWipOut(ctx, dir, nil, "rev-parse", "HEAD^{tree}")
	if err != nil {
		t.Fatal(err)
	}
	obj, err := gitWipOut(ctx, dir, nil, "commit-tree", tree, "-m", "parentless identical")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := gitWipOut(ctx, dir, nil, "update-ref", "refs/fak/wip/quar-reap", obj); err != nil {
		t.Fatal(err)
	}

	res, err := wipReconcileAt(ctx, dir, time.Now())
	if err != nil {
		t.Fatalf("wipReconcileAt: %v", err)
	}
	if len(res.Decisions) != 1 {
		t.Fatalf("decisions count = %d, want 1", len(res.Decisions))
	}
	d := res.Decisions[0]
	if d.Action != wiprecon.ActQuarantine {
		t.Fatalf("decision action = %s, want QUARANTINE", d.Action)
	}
	if len(d.ReviewCommands) != 0 {
		t.Fatalf("review commands = %v, want empty", d.ReviewCommands)
	}
	wantNext := "fak wip reap --census --json"
	if d.NextCommand != wantNext {
		t.Fatalf("NextCommand = %q, want %q", d.NextCommand, wantNext)
	}
}
