package rehome

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/memorycotravel"
)

// TestMain points the memory co-travel ledger at a temp file: every real re-home in this
// package appends a default-shadow row, which must never reach the operator's ~/.claude.
func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "fak-rehome-cotravel-")
	if err != nil {
		panic(err)
	}
	os.Setenv("FAK_MEMORY_COTRAVEL_LEDGER", filepath.Join(dir, "ledger.jsonl"))
	code := m.Run()
	_ = os.RemoveAll(dir)
	os.Exit(code)
}

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

func TestRehomeTranscriptCotravelsMemoryUnderGate(t *testing.T) {
	const sid, project = "sess4", "C--work-fak"
	for _, tc := range []struct {
		name, gate string
		wantCopy   bool
		wantGate   string
	}{
		{name: "live", gate: "live", wantCopy: true, wantGate: "on"},
		{name: "default-shadow", gate: "", wantCopy: false, wantGate: "shadow"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("FAK_MEMORY_COTRAVEL", tc.gate)
			t.Setenv("FAK_MEMORY_COTRAVEL_LEDGER", filepath.Join(t.TempDir(), "ledger.jsonl"))
			src := t.TempDir()
			dst := t.TempDir()
			mkTranscript(t, src, project, sid, "x")
			srcMem := filepath.Join(src, "projects", project, "memory")
			if err := os.MkdirAll(srcMem, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(srcMem, "a.md"), []byte("remember"), 0o644); err != nil {
				t.Fatal(err)
			}

			if !RehomeTranscript(src, dst, project, sid, nil) {
				t.Fatal("RehomeTranscript = false, want true")
			}
			got, err := os.ReadFile(filepath.Join(dst, "projects", project, "memory", "a.md"))
			if tc.wantCopy && (err != nil || string(got) != "remember") {
				t.Fatalf("dst memory = %q err %v, want %q", got, err, "remember")
			}
			if !tc.wantCopy && !os.IsNotExist(err) {
				t.Fatalf("dst memory err = %v, want not-exist (gate copies nothing)", err)
			}
			rows := memorycotravel.ReadLedger()
			if len(rows) != 1 || rows[0]["gate"] != tc.wantGate {
				t.Fatalf("ledger rows = %v, want exactly one row with gate=%s", rows, tc.wantGate)
			}
		})
	}
}
