package logvault

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// rewriteCapture rewrites the source file and captures, advancing mtime so the
// capture is never taken for a same-content touch.
func rewriteCapture(t *testing.T, v *Vault, log, content string, at time.Time) {
	t.Helper()
	writeFile(t, log, content)
	os.Chtimes(log, at, at)
	if _, err := v.Capture(); err != nil {
		t.Fatal(err)
	}
}

func TestGCProposeByDefaultThenApplyUnderLive(t *testing.T) {
	srcDir := t.TempDir()
	log := filepath.Join(srcDir, "loops.jsonl")
	v := testVault(t, Source{ID: "s", Root: srcDir})

	// 1 full + 3 rewrites: .history/ ends with 3 superseded versions (A,B,C);
	// D is the current mirror.
	base := time.Now()
	rewriteCapture(t, v, log, "AAAA\n", base)
	rewriteCapture(t, v, log, "BBBBBB\n", base.Add(2*time.Second))
	rewriteCapture(t, v, log, "CCCCCCCC\n", base.Add(4*time.Second))
	rewriteCapture(t, v, log, "DDDDDDDDDD\n", base.Add(6*time.Second))

	histDir := filepath.Join(v.Dir, "by-source", "s", ".history")
	if ents, _ := os.ReadDir(histDir); len(ents) != 3 {
		t.Fatalf("history = %d versions, want 3 before GC", len(ents))
	}
	manRowsBefore := func() int {
		rows, _ := ReadManifestRows(filepath.Join(v.Dir, ManifestName))
		return len(rows)
	}
	rowsBefore := manRowsBefore()

	// PROPOSE (default, live=false): depth 1 keeps the newest version, proposes
	// the oldest two — and must NOT delete anything or append a manifest row.
	rep, err := v.GC(GCPolicy{HistoryDepth: 1}, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Candidates) != 2 {
		t.Fatalf("propose: %d candidates, want 2 (depth 1 of 3)", len(rep.Candidates))
	}
	if rep.Applied {
		t.Fatal("propose pass must report Applied=false")
	}
	if rep.ReclaimBytes != int64(len("AAAA\n")+len("BBBBBB\n")) {
		t.Fatalf("reclaim bytes = %d, want A+B", rep.ReclaimBytes)
	}
	if ents, _ := os.ReadDir(histDir); len(ents) != 3 {
		t.Fatal("propose deleted history files — it must only propose")
	}
	if got := manRowsBefore(); got != rowsBefore {
		t.Fatalf("propose grew the manifest %d -> %d — no row may be written without -live", rowsBefore, got)
	}

	// The kept version must be the NEWEST (C), the proposed ones the two oldest.
	for _, c := range rep.Candidates {
		if c.RelPath != "loops.jsonl" {
			t.Fatalf("candidate rel = %q, want loops.jsonl", c.RelPath)
		}
	}

	// APPLY (live=true): the two proposed files are deleted and each prune is
	// witnessed by a gc-prune manifest row; the chain still verifies.
	rep, err = v.GC(GCPolicy{HistoryDepth: 1}, true)
	if err != nil {
		t.Fatal(err)
	}
	if !rep.Applied || len(rep.Candidates) != 2 {
		t.Fatalf("apply: Applied=%v candidates=%d, want true/2", rep.Applied, len(rep.Candidates))
	}
	if ents, _ := os.ReadDir(histDir); len(ents) != 1 {
		t.Fatalf("history after apply = %d, want 1 (newest kept)", len(ents))
	}
	rows, _ := ReadManifestRows(filepath.Join(v.Dir, ManifestName))
	if got := len(rows) - rowsBefore; got != 2 {
		t.Fatalf("apply appended %d rows, want 2 gc-prune rows", got)
	}
	prunes := 0
	for _, r := range rows {
		if r.Op == OpGCPrune {
			prunes++
			if r.SHA256 != "" {
				t.Fatal("gc-prune row must carry no capture sha (must not advance mirror state)")
			}
		}
	}
	if prunes != 2 {
		t.Fatalf("gc-prune rows = %d, want 2", prunes)
	}
	if _, err := VerifyManifest(filepath.Join(v.Dir, ManifestName)); err != nil {
		t.Fatalf("chain broken after gc-prune: %v", err)
	}
	// The current mirror is untouched: Verify re-hashes it clean.
	if _, _, problems, err := v.Verify(0); err != nil || len(problems) != 0 {
		t.Fatalf("verify after gc: problems=%v err=%v", problems, err)
	}
	if got := readFile(t, v.mirrorPath("s", "loops.jsonl")); got != "DDDDDDDDDD\n" {
		t.Fatalf("current mirror = %q, want the newest content (GC never touches the mirror)", got)
	}

	// Idempotent: a second apply at the same depth has nothing left to prune.
	rep, err = v.GC(GCPolicy{HistoryDepth: 1}, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Candidates) != 0 || rep.Applied {
		t.Fatalf("second apply: candidates=%d applied=%v, want 0/false (nothing to do)", len(rep.Candidates), rep.Applied)
	}
}

func TestGCDepthZeroKeepsEverything(t *testing.T) {
	srcDir := t.TempDir()
	log := filepath.Join(srcDir, "a.jsonl")
	v := testVault(t, Source{ID: "s", Root: srcDir})
	base := time.Now()
	rewriteCapture(t, v, log, "one\n", base)
	rewriteCapture(t, v, log, "twoo\n", base.Add(2*time.Second))
	rewriteCapture(t, v, log, "three\n", base.Add(4*time.Second))

	rep, err := v.GC(GCPolicy{HistoryDepth: 0}, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Candidates) != 0 {
		t.Fatalf("depth 0 proposed %d candidates, want 0 (unlimited retention)", len(rep.Candidates))
	}
}

func TestAdoptColdRoundTripAndDeterministicRepeat(t *testing.T) {
	// Fixture cold tree with nested files.
	cold := t.TempDir()
	writeFile(t, filepath.Join(cold, "head.bin"), "the head part\n")
	writeFile(t, filepath.Join(cold, "iso", "a.txt"), "alpha\n")
	writeFile(t, filepath.Join(cold, "iso", "b.txt"), "beta\n")

	v := &Vault{Dir: t.TempDir()}

	rep1, err := v.AdoptCold(cold)
	if err != nil {
		t.Fatal(err)
	}
	if rep1.Deduped {
		t.Fatal("first adopt must not be a dedup")
	}
	if rep1.Files != 3 {
		t.Fatalf("packed %d files, want 3", rep1.Files)
	}
	arc1 := filepath.Join(v.Dir, filepath.FromSlash(rep1.ArchiveRel))
	if _, err := os.Stat(arc1); err != nil {
		t.Fatalf("archive not banked: %v", err)
	}
	// The manifest witnessed the adoption; Verify re-hashes the banked archive.
	rows, _ := ReadManifestRows(filepath.Join(v.Dir, ManifestName))
	if len(rows) != 1 || rows[0].Op != OpColdAdopt || rows[0].SHA256 != rep1.SHA256 {
		t.Fatalf("manifest row = %+v, want one cold-adopt row carrying the archive sha", rows)
	}
	if _, _, problems, err := v.Verify(0); err != nil || len(problems) != 0 {
		t.Fatalf("verify after adopt: problems=%v err=%v", problems, err)
	}

	// The source tree is untouched — the tool deletes nothing.
	if _, err := os.Stat(filepath.Join(cold, "head.bin")); err != nil {
		t.Fatal("adopt deleted source data — it must never do so")
	}
	if rep1.DeleteCmd == "" {
		t.Fatal("adopt must print an operator delete command")
	}

	first := mustRead(t, arc1)

	// Re-adopt identical content: same content address, byte-identical archive,
	// reported as a dedup with no new manifest row.
	rep2, err := v.AdoptCold(cold)
	if err != nil {
		t.Fatal(err)
	}
	if !rep2.Deduped {
		t.Fatal("re-adopt of identical content must dedup")
	}
	if rep2.SHA256 != rep1.SHA256 || rep2.ArchiveRel != rep1.ArchiveRel {
		t.Fatalf("dedup mismatch: %s vs %s", rep2.ArchiveRel, rep1.ArchiveRel)
	}
	if got := mustRead(t, filepath.Join(v.Dir, filepath.FromSlash(rep2.ArchiveRel))); !bytes.Equal(got, first) {
		t.Fatal("repeat adoption is not byte-identical — deterministic-tar property broken")
	}
	rows2, _ := ReadManifestRows(filepath.Join(v.Dir, ManifestName))
	if len(rows2) != 1 {
		t.Fatalf("dedup appended a duplicate row: %d rows, want 1", len(rows2))
	}
}

func TestAdoptColdDeterministicAcrossTrees(t *testing.T) {
	// Two DISTINCT directories with identical file contents (and differing
	// mtimes) must pack to the same content address — mtimes are pinned.
	mk := func(mtime time.Time) string {
		d := t.TempDir()
		writeFile(t, filepath.Join(d, "x.txt"), "same\n")
		writeFile(t, filepath.Join(d, "y.txt"), "content\n")
		os.Chtimes(filepath.Join(d, "x.txt"), mtime, mtime)
		os.Chtimes(filepath.Join(d, "y.txt"), mtime, mtime)
		return d
	}
	a := mk(time.Unix(1_000_000, 0))
	b := mk(time.Unix(9_000_000, 0))

	var bufA, bufB bytes.Buffer
	if _, err := packTree(a, &bufA); err != nil {
		t.Fatal(err)
	}
	if _, err := packTree(b, &bufB); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(bufA.Bytes(), bufB.Bytes()) {
		t.Fatal("identical content with different mtimes packed to different bytes — dedup would leak")
	}
}

func mustRead(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
