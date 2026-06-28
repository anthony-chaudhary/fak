package vcachesnapshot

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

// A written snapshot must read back as the same turns, so the score CLI sees the exact
// observed window the gateway captured.
func TestWriteReadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sub", DefaultRel) // also proves MkdirAll of a missing parent
	in := []vcacheobserve.Turn{
		{Family: "head", UnixMillis: 1, InputTokens: 100, CacheRead: 900, CacheCreation: 50},
		{Family: "head", UnixMillis: 2, InputTokens: 120, CacheRead: 880},
	}
	if err := Write(path, in); err != nil {
		t.Fatalf("write: %v", err)
	}
	out, ok, err := Read(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !ok || len(out) != len(in) {
		t.Fatalf("round-trip lost turns: ok=%v got %d want %d", ok, len(out), len(in))
	}
	for i := range in {
		if out[i] != in[i] {
			t.Errorf("turn %d round-trip mismatch: got %+v want %+v", i, out[i], in[i])
		}
	}
}

// A missing snapshot is the common case (no session has run yet) — it must NOT error, and
// must signal ok=false so the score falls open to the planned forecast rather than
// reporting a phantom observed 0x.
func TestReadMissingFileFailsOpen(t *testing.T) {
	out, ok, err := Read(filepath.Join(t.TempDir(), "nope.jsonl"))
	if err != nil {
		t.Fatalf("a missing snapshot must not error, got %v", err)
	}
	if ok || out != nil {
		t.Fatalf("a missing snapshot must read as (nil,false), got (%v,%v)", out, ok)
	}
}

// An empty turns slice writes an empty file that reads back as no observed window.
func TestWriteEmptyReadsNoWindow(t *testing.T) {
	path := filepath.Join(t.TempDir(), DefaultRel)
	if err := Write(path, nil); err != nil {
		t.Fatalf("write empty: %v", err)
	}
	if _, ok, err := Read(path); err != nil || ok {
		t.Fatalf("empty snapshot must read as no window, got ok=%v err=%v", ok, err)
	}
}

// A malformed row must be skipped, not fail the whole read, so a partially-written
// snapshot still yields its valid turns.
func TestReadSkipsMalformedLine(t *testing.T) {
	path := filepath.Join(t.TempDir(), DefaultRel)
	body := `{"family":"a","unix_millis":1,"input_tokens":10}` + "\n" +
		"{ this is not json" + "\n" +
		`{"family":"a","unix_millis":2,"input_tokens":20}` + "\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	out, ok, err := Read(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !ok || len(out) != 2 {
		t.Fatalf("malformed line must be skipped, got %d valid turns (ok=%v)", len(out), ok)
	}
}

// DefaultPath is stable and ends in the documented basename.
func TestDefaultPathBasename(t *testing.T) {
	if filepath.Base(DefaultPath()) != DefaultRel {
		t.Fatalf("DefaultPath basename = %q, want %q", filepath.Base(DefaultPath()), DefaultRel)
	}
}
