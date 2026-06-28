package dojo

import (
	"os"
	"path/filepath"
	"testing"
)

// writeMarker writes one start-marker into dir exactly as the --dojo writer does:
// a single JSON object on one line.
func writeMarker(t *testing.T, dir, name, command, started string) {
	t.Helper()
	body := `{"mode":"live","command":"` + command + `","started":"` + started + `","cwd":"/x","workspace":"/x"}` + "\n"
	if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestReadLiveCorpusFailsOpenOnMissingDir(t *testing.T) {
	// A directory that was never created (no --dojo session ever ran) must NOT be
	// an error — it is the honest empty state.
	lc, err := ReadLiveCorpus(filepath.Join(t.TempDir(), "does-not-exist"))
	if err != nil {
		t.Fatalf("a missing corpus must fail open, got error %v", err)
	}
	if lc.Present {
		t.Fatalf("a missing corpus must report Present=false, got %+v", lc)
	}
	if lc.Found != 0 || lc.Scorable != 0 || lc.Missing != "" {
		t.Fatalf("a missing corpus must be an empty zero-found corpus, got %+v", lc)
	}
}

func TestReadLiveCorpusEmptyDirIsNotError(t *testing.T) {
	dir := t.TempDir()
	lc, err := ReadLiveCorpus(dir)
	if err != nil {
		t.Fatalf("an empty corpus must fail open, got %v", err)
	}
	if !lc.Present {
		t.Fatalf("an existing-but-empty dir must report Present=true, got %+v", lc)
	}
	if lc.Found != 0 {
		t.Fatalf("an empty dir has no markers, got Found=%d", lc.Found)
	}
}

func TestReadLiveCorpusDiscoversStartMarkersDegradesGracefully(t *testing.T) {
	dir := t.TempDir()
	// Two real markers, deliberately out of filename order to prove sorting.
	writeMarker(t, dir, "episode_20260627_120000.jsonl", "guard", "2026-06-27T12:00:00Z")
	writeMarker(t, dir, "episode_20260627_090000.jsonl", "serve", "2026-06-27T09:00:00Z")
	// A non-marker file that must be ignored.
	if err := os.WriteFile(filepath.Join(dir, "README.txt"), []byte("not a marker"), 0o600); err != nil {
		t.Fatal(err)
	}

	lc, err := ReadLiveCorpus(dir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !lc.Present {
		t.Fatalf("a populated corpus must report Present=true")
	}
	if lc.Found != 2 {
		t.Fatalf("want 2 markers discovered (README ignored), got %d", lc.Found)
	}
	// DEGRADE GRACEFULLY: start-only markers carry no billed reality, so nothing
	// is scorable and the reader must SAY what is missing rather than invent a score.
	if lc.Scorable != 0 {
		t.Fatalf("start-only markers must be 0 scorable (no fabricated score), got %d", lc.Scorable)
	}
	if lc.Missing == "" {
		t.Fatalf("found-but-unscorable markers must name what is missing, got empty Missing")
	}
	// Markers must be oldest-first by filename (09:00 before 12:00).
	if lc.Markers[0].File != "episode_20260627_090000.jsonl" || lc.Markers[0].Command != "serve" {
		t.Fatalf("markers must sort chronologically by filename, got %+v", lc.Markers)
	}
	if lc.Markers[1].Command != "guard" {
		t.Fatalf("second marker should be the guard session, got %+v", lc.Markers[1])
	}
}

func TestReadLiveCorpusSkipsMalformedMarker(t *testing.T) {
	dir := t.TempDir()
	writeMarker(t, dir, "episode_20260627_120000.jsonl", "guard", "2026-06-27T12:00:00Z")
	// A malformed marker (truncated JSON) must be skipped, not fatal.
	if err := os.WriteFile(filepath.Join(dir, "episode_20260627_130000.jsonl"), []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	// A marker-named file with valid JSON but none of our keys must be skipped too.
	if err := os.WriteFile(filepath.Join(dir, "episode_20260627_140000.jsonl"), []byte(`{"unrelated":1}`), 0o600); err != nil {
		t.Fatal(err)
	}
	lc, err := ReadLiveCorpus(dir)
	if err != nil {
		t.Fatalf("a malformed marker must be skipped, not fatal: %v", err)
	}
	if lc.Found != 1 {
		t.Fatalf("only the one well-formed marker should be folded, got %d", lc.Found)
	}
}

func TestScorableLiveEpisodesInventsNoNumber(t *testing.T) {
	// While the writer is start-only, the scorable adapter must produce nothing —
	// no episode is ever scored off metadata alone.
	dir := t.TempDir()
	writeMarker(t, dir, "episode_20260627_120000.jsonl", "guard", "2026-06-27T12:00:00Z")
	lc, err := ReadLiveCorpus(dir)
	if err != nil {
		t.Fatal(err)
	}
	if ins := ScorableLiveEpisodes(lc); len(ins) != 0 {
		t.Fatalf("start-only markers must yield no scored episodes, got %d", len(ins))
	}
}
