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

	// --- CLOSED_CLEAN_ESTIMATE (subsumed, not verbatim): a checkpoint whose added ---
	// --- line lands in HEAD alongside an extra unrelated line, so it is NOT a        ---
	// --- verbatim landing (not LANDED) yet every added line is present in HEAD.       ---
	write("su.txt", "S\n")
	if _, err := wipCheckpoint(ctx, dir, "subsumed", true, 1000); err != nil {
		t.Fatalf("checkpoint subsumed: %v", err)
	}
	write("su.txt", "S\nextra\n")                               // HEAD gets "S" plus an unrelated line
	commitWorkingTree(t, dir, "land su.txt with an extra line") // not byte-identical to the checkpoint

	classes := map[string]wipref.CensusClass{
		"landed":   wipCensusClassOf(t, dir, "landed"),
		"dirty":    wipCensusClassOf(t, dir, "dirty"),
		"subsumed": wipCensusClassOf(t, dir, "subsumed"),
	}
	want := map[string]wipref.CensusClass{
		"landed":   wipref.CensusLanded,
		"dirty":    wipref.CensusClosedDirtyRecoverable,
		"subsumed": wipref.CensusClosedCleanEstimate,
	}
	for sess, w := range want {
		if classes[sess] != w {
			t.Errorf("session %q classified %s, want %s", sess, classes[sess], w)
		}
	}

	// Read-only: the census deleted nothing — all three refs still resolve.
	for _, sess := range []string{"landed", "dirty", "subsumed"} {
		if _, has, err := wipCurrentOID(ctx, dir, wipref.SessionRef(sess)); err != nil || !has {
			t.Errorf("census deleted ref for %q (has=%v err=%v) — census must be read-only", sess, has, err)
		}
	}
}

// TestWipCensusJSONShape proves the --census --json wiring: exit 0, valid JSON, counts
// that sum to the total, and the headline CLOSED_CLEAN_ESTIMATE field present.
func TestWipCensusJSONShape(t *testing.T) {
	ctx := context.Background()
	dir, file := wipTestRepo(t)

	// One dead-session unlanded checkpoint -> exactly one CLOSED_DIRTY_RECOVERABLE.
	if err := os.WriteFile(file, []byte("base line\nrecoverable\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, dir, "solo", true, 1000); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if _, err := gitWipOut(ctx, dir, nil, "checkout", "--", "."); err != nil {
		t.Fatalf("checkout reset: %v", err)
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
	sum := c.Landed + c.Live + c.ClosedCleanEstimate + c.ClosedDirtyRecoverable + c.Unknown
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
	dir, file := wipTestRepo(t)
	if err := os.WriteFile(file, []byte("base line\nrecover me\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if r, err := wipCheckpoint(context.Background(), dir, "recoverable-wording", true, time.Now().Unix()); err != nil {
		t.Fatalf("checkpoint: %v (%+v)", err, r)
	}
	if err := os.WriteFile(file, []byte("base line\n"), 0o644); err != nil {
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
