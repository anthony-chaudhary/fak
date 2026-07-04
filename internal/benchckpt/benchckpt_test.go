package benchckpt

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// cell is a stand-in for a measured grid cell (a prefill/decode point, a fanrun N).
type cell struct {
	Key  string  `json:"key"`
	MS   float64 `json:"ms"`
	TokS float64 `json:"tok_s"`
}

// TestAppendReopenResumeFilter is the acceptance witness for #2382 item 1: append N
// cells, reopen, and the resume filter returns exactly the un-recorded coordinates.
func TestAppendReopenResumeFilter(t *testing.T) {
	path := filepath.Join(t.TempDir(), "grid.jsonl")
	fp := Fingerprint{"grid": "16,64,256", "model": "smollm2-135m", "seed": 7}

	// Full grid the sweep intends to measure.
	grid := []string{"P=16", "P=64", "P=256", "decode", "wl:short"}

	// First run: measure the first 3 cells, then "crash" (close) before the rest.
	l, err := Open(path, fp)
	if err != nil {
		t.Fatalf("Open (fresh): %v", err)
	}
	if got := l.Missing(grid); len(got) != len(grid) {
		t.Fatalf("fresh Missing = %v, want the full grid", got)
	}
	for i, k := range grid[:3] {
		if err := l.Append(k, cell{Key: k, MS: float64(10 * (i + 1)), TokS: float64(i + 1)}); err != nil {
			t.Fatalf("Append %q: %v", k, err)
		}
	}
	if err := l.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Reopen (the --resume path) and assert the filter returns exactly the un-recorded
	// coordinates, in order, and nothing recorded is offered again.
	l2, err := Open(path, fp)
	if err != nil {
		t.Fatalf("Open (resume): %v", err)
	}
	defer l2.Close()
	if l2.Len() != 3 {
		t.Fatalf("reopened Len = %d, want 3", l2.Len())
	}
	missing := l2.Missing(grid)
	want := []string{"decode", "wl:short"}
	if !equalStrings(missing, want) {
		t.Fatalf("Missing after resume = %v, want %v", missing, want)
	}
	for _, k := range grid[:3] {
		if !l2.Has(k) {
			t.Fatalf("recorded cell %q reported missing after reopen", k)
		}
	}

	// The recorded cells survived the reopen byte-for-byte (assembly from checkpoint).
	var got cell
	ok, err := l2.Cell("P=64", &got)
	if err != nil || !ok {
		t.Fatalf("Cell P=64: ok=%v err=%v", ok, err)
	}
	if got.MS != 20 || got.TokS != 2 {
		t.Fatalf("Cell P=64 = %+v, want ms=20 tok_s=2", got)
	}

	// Finish the missing cells; a completed grid then reports nothing missing.
	for _, k := range missing {
		if err := l2.Append(k, cell{Key: k, MS: 99}); err != nil {
			t.Fatalf("Append missing %q: %v", k, err)
		}
	}
	if left := l2.Missing(grid); len(left) != 0 {
		t.Fatalf("after completing grid, Missing = %v, want empty", left)
	}
	if l2.Len() != len(grid) {
		t.Fatalf("final Len = %d, want %d", l2.Len(), len(grid))
	}
}

// TestResumeFingerprintMismatchRefuses is the acceptance witness for #2382 item 5:
// resuming with a mismatched grid/model/seed refuses with a typed error instead of a
// silent merge.
func TestResumeFingerprintMismatchRefuses(t *testing.T) {
	path := filepath.Join(t.TempDir(), "grid.jsonl")
	l, err := Open(path, Fingerprint{"grid": "16,64", "model": "a", "seed": 1})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := l.Append("P=16", cell{Key: "P=16", MS: 1}); err != nil {
		t.Fatalf("Append: %v", err)
	}
	l.Close()

	// Different seed -> refuse.
	if _, err := Open(path, Fingerprint{"grid": "16,64", "model": "a", "seed": 2}); !errors.Is(err, ErrFingerprintMismatch) {
		t.Fatalf("resume with changed seed err = %v, want ErrFingerprintMismatch", err)
	}
	// Different model -> refuse.
	if _, err := Open(path, Fingerprint{"grid": "16,64", "model": "b", "seed": 1}); !errors.Is(err, ErrFingerprintMismatch) {
		t.Fatalf("resume with changed model err = %v, want ErrFingerprintMismatch", err)
	}
	// Identical fingerprint (numbers arriving as float64 after a JSON round-trip must
	// still compare equal) -> accepted.
	l3, err := Open(path, Fingerprint{"grid": "16,64", "model": "a", "seed": 1})
	if err != nil {
		t.Fatalf("resume with identical fingerprint: %v", err)
	}
	if !l3.Has("P=16") {
		t.Fatalf("identical-fingerprint resume lost the recorded cell")
	}
	l3.Close()
}

// TestTornTrailingLineTolerated proves the durability discipline (#2386): a crash that
// tears the final append leaves every earlier cell recoverable.
func TestTornTrailingLineTolerated(t *testing.T) {
	path := filepath.Join(t.TempDir(), "grid.jsonl")
	fp := Fingerprint{"grid": "16,64,256"}
	l, err := Open(path, fp)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	for _, k := range []string{"P=16", "P=64"} {
		if err := l.Append(k, cell{Key: k, MS: 1}); err != nil {
			t.Fatalf("Append %q: %v", k, err)
		}
	}
	l.Close()

	// Simulate a kill -9 mid-append: a partial, unparseable trailing line.
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(`{"key":"P=256","cell":{"key":"P=256","ms":`); err != nil {
		t.Fatal(err)
	}
	f.Close()

	l2, err := Open(path, fp)
	if err != nil {
		t.Fatalf("Open after torn write: %v", err)
	}
	defer l2.Close()
	if l2.Len() != 2 {
		t.Fatalf("after torn write Len = %d, want 2 (torn line skipped)", l2.Len())
	}
	if l2.Has("P=256") {
		t.Fatalf("torn cell P=256 should not be recorded")
	}
	if got := l2.Missing([]string{"P=16", "P=64", "P=256"}); !equalStrings(got, []string{"P=256"}) {
		t.Fatalf("Missing = %v, want [P=256]", got)
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
