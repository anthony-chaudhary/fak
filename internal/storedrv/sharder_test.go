package storedrv

import (
	"os"
	"path/filepath"
	"testing"
)

// TestSpillSharderRootModulo covers the worker_id % len(roots) pick, including ids past
// the end of the list and NEGATIVE ids (Go's % keeps the dividend's sign, so the fold
// back into range is the load-bearing correctness bit).
func TestSpillSharderRootModulo(t *testing.T) {
	base := t.TempDir()
	roots := []string{
		filepath.Join(base, "a"),
		filepath.Join(base, "b"),
		filepath.Join(base, "c"),
	}
	s, err := NewSpillSharder(roots)
	if err != nil {
		t.Fatalf("NewSpillSharder: %v", err)
	}
	if s.Len() != 3 {
		t.Fatalf("Len() = %d, want 3", s.Len())
	}
	cleaned := s.Roots()
	cases := []struct {
		workerID int
		want     string
	}{
		{0, cleaned[0]},
		{1, cleaned[1]},
		{2, cleaned[2]},
		{3, cleaned[0]}, // wraps
		{4, cleaned[1]},
		{5, cleaned[2]},
		{-1, cleaned[2]}, // -1 %% 3 == -1 -> +3 -> 2
		{-2, cleaned[1]},
		{-3, cleaned[0]},
		{-4, cleaned[2]},
	}
	for _, c := range cases {
		if got := s.Root(c.workerID); got != c.want {
			t.Errorf("Root(%d) = %q, want %q", c.workerID, got, c.want)
		}
	}
}

// TestSpillSharderDeterministic asserts the same worker id maps to the same root across
// repeated calls (Root does no I/O and is a pure index into the frozen list).
func TestSpillSharderDeterministic(t *testing.T) {
	base := t.TempDir()
	s, err := NewSpillSharder([]string{filepath.Join(base, "x"), filepath.Join(base, "y")})
	if err != nil {
		t.Fatalf("NewSpillSharder: %v", err)
	}
	for id := -5; id <= 5; id++ {
		first := s.Root(id)
		if second := s.Root(id); first != second {
			t.Errorf("Root(%d) not deterministic: %q then %q", id, first, second)
		}
	}
}

// TestSpillSharderValidation covers the three hard-error refusals: an empty list, a list
// whose only entry is blank (drops to empty), and a duplicate root (which would fold two
// workers onto one mount, defeating the fan).
func TestSpillSharderValidation(t *testing.T) {
	base := t.TempDir()
	dup := filepath.Join(base, "same")
	cases := []struct {
		name  string
		roots []string
	}{
		{"nil list", nil},
		{"empty list", []string{}},
		{"all-blank", []string{"", "   ", "\t"}},
		{"duplicate", []string{dup, dup}},
		{"duplicate after clean", []string{dup, dup + string(filepath.Separator)}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := NewSpillSharder(c.roots); err == nil {
				t.Fatalf("NewSpillSharder(%v) = nil error, want refusal", c.roots)
			}
		})
	}
}

// TestSpillSharderDropsBlankEntries asserts a blank entry (e.g. a trailing comma in
// "disk:/a,/b,") is DROPPED, not errored — the documented tolerance — while the real
// roots survive.
func TestSpillSharderDropsBlankEntries(t *testing.T) {
	base := t.TempDir()
	a := filepath.Join(base, "a")
	b := filepath.Join(base, "b")
	s, err := NewSpillSharder([]string{a, "", b, "  "})
	if err != nil {
		t.Fatalf("NewSpillSharder with trailing blanks: %v", err)
	}
	if s.Len() != 2 {
		t.Fatalf("Len() = %d, want 2 (blanks dropped)", s.Len())
	}
}

// TestSpillSharderEagerMkdir asserts every validated root is created ON CONSTRUCTION
// (os.MkdirAll), not lazily on first write — the property that turns a missing mount into
// a boot error instead of a runtime spill loss.
func TestSpillSharderEagerMkdir(t *testing.T) {
	base := t.TempDir()
	roots := []string{
		filepath.Join(base, "m0"),
		filepath.Join(base, "nested", "m1"), // MkdirAll must create the parent too
	}
	for _, r := range roots {
		if _, err := os.Stat(r); !os.IsNotExist(err) {
			t.Fatalf("precondition: %q should not exist yet (err=%v)", r, err)
		}
	}
	s, err := NewSpillSharder(roots)
	if err != nil {
		t.Fatalf("NewSpillSharder: %v", err)
	}
	for _, r := range s.Roots() {
		info, err := os.Stat(r)
		if err != nil {
			t.Fatalf("root %q not created eagerly: %v", r, err)
		}
		if !info.IsDir() {
			t.Fatalf("root %q exists but is not a directory", r)
		}
	}
}

// TestSpillSharderRootsDefensiveCopy asserts mutating the slice returned by Roots cannot
// corrupt the sharder's internal pick.
func TestSpillSharderRootsDefensiveCopy(t *testing.T) {
	base := t.TempDir()
	a := filepath.Join(base, "a")
	b := filepath.Join(base, "b")
	s, err := NewSpillSharder([]string{a, b})
	if err != nil {
		t.Fatalf("NewSpillSharder: %v", err)
	}
	got := s.Roots()
	got[0] = "/tmp/evil"
	if s.Root(0) == "/tmp/evil" {
		t.Fatal("Roots() leaked the internal slice — mutation corrupted the pick")
	}
}
