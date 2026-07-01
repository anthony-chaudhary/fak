package rehome

import (
	"os"
	"path/filepath"
	"testing"
)

func mkTranscript(t *testing.T, cfg, project, sid, content string) {
	t.Helper()
	dir := filepath.Join(cfg, "projects", project)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, sid+".jsonl"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestRehomeTranscriptMissingSource(t *testing.T) {
	src := t.TempDir()
	dst := t.TempDir()
	if RehomeTranscript(src, dst, "C--work-fak", "nope", nil) {
		t.Fatal("RehomeTranscript with missing source = true, want false")
	}
}

func TestRehomeTranscriptCopiesTranscriptAndSidecar(t *testing.T) {
	src := t.TempDir()
	dst := t.TempDir()
	sid := "sess1"
	project := "C--work-fak"
	mkTranscript(t, src, project, sid, "turn-one\n")
	// sidecar <sid>/ dir with a nested file.
	sideFile := filepath.Join(src, "projects", project, sid, "workflows", "wf.json")
	if err := os.MkdirAll(filepath.Dir(sideFile), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(sideFile, []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}

	if !RehomeTranscript(src, dst, project, sid, nil) {
		t.Fatal("RehomeTranscript = false, want true")
	}
	got, err := os.ReadFile(filepath.Join(dst, "projects", project, sid+".jsonl"))
	if err != nil {
		t.Fatalf("dst transcript not copied: %v", err)
	}
	if string(got) != "turn-one\n" {
		t.Fatalf("dst transcript content = %q, want %q", got, "turn-one\n")
	}
	if _, err := os.Stat(filepath.Join(dst, "projects", project, sid, "workflows", "wf.json")); err != nil {
		t.Fatalf("sidecar not copied: %v", err)
	}
}

func TestRehomeTranscriptExtraDestSlugs(t *testing.T) {
	src := t.TempDir()
	dst := t.TempDir()
	sid := "sess2"
	owner := "C--work-fak"
	cwdSlug := "C--work-slack-helpers"
	mkTranscript(t, src, owner, sid, "x")

	if !RehomeTranscript(src, dst, owner, sid, []string{cwdSlug}) {
		t.Fatal("RehomeTranscript = false, want true")
	}
	for _, slug := range []string{owner, cwdSlug} {
		if _, err := os.Stat(filepath.Join(dst, "projects", slug, sid+".jsonl")); err != nil {
			t.Fatalf("transcript missing under slug %q: %v", slug, err)
		}
	}
}

func TestRehomeTranscriptSelfCopyIsNoOpSuccess(t *testing.T) {
	cfg := t.TempDir()
	sid := "sess3"
	project := "C--work-fak"
	mkTranscript(t, cfg, project, sid, "same")
	// Mirroring within the owner account under its own slug: dst == src.
	if !RehomeTranscript(cfg, cfg, project, sid, nil) {
		t.Fatal("self-copy RehomeTranscript = false, want true (no-op success)")
	}
	got, err := os.ReadFile(filepath.Join(cfg, "projects", project, sid+".jsonl"))
	if err != nil || string(got) != "same" {
		t.Fatalf("self-copy clobbered source: got %q err %v", got, err)
	}
}
