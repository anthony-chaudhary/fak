package treedoctor

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestClassifyWIP is the pure land-or-park fold, exhaustively: liveness beats every other
// signal (a mid-refactor file that does not build yet must NOT be surfaced for park), an
// aged non-building file is the crash-loop poison, an aged unowned-but-building file is
// abandoned, a live owner or a fresh mtime keeps a file resident.
func TestClassifyWIP(t *testing.T) {
	const abandonSec = 3600
	cases := []struct {
		name                                      string
		f                                         WIPFile
		wantClass                                 string
		wantAbandoned, wantLandOrPark, wantPoison bool
	}{
		{
			// The load-bearing safety case: live wins even when the package won't compile.
			name:          "live non-building is kept, never parked",
			f:             WIPFile{AgeSeconds: 10000, Live: true, BuildProbed: true, Builds: false},
			wantClass:     "live",
			wantAbandoned: false, wantLandOrPark: false, wantPoison: true,
		},
		{
			name:          "aged non-building is poison, surfaced",
			f:             WIPFile{AgeSeconds: 10000, Live: false, BuildProbed: true, Builds: false},
			wantClass:     "poison",
			wantAbandoned: true, wantLandOrPark: true, wantPoison: true,
		},
		{
			name:          "young non-building is still poison but not yet abandoned",
			f:             WIPFile{AgeSeconds: 100, Live: false, BuildProbed: true, Builds: false},
			wantClass:     "poison",
			wantAbandoned: false, wantLandOrPark: true, wantPoison: true,
		},
		{
			name:          "aged, builds, no live owner is abandoned",
			f:             WIPFile{AgeSeconds: 10000, Live: false, BuildProbed: true, Builds: true},
			wantClass:     "abandoned",
			wantAbandoned: true, wantLandOrPark: true, wantPoison: false,
		},
		{
			name:          "aged but a live owner holds it stays resident",
			f:             WIPFile{AgeSeconds: 10000, Live: false, OwnerAlive: true, BuildProbed: true, Builds: true},
			wantClass:     "resident",
			wantAbandoned: false, wantLandOrPark: false, wantPoison: false,
		},
		{
			name:          "fresh-ish, not live, builds stays resident",
			f:             WIPFile{AgeSeconds: 100, Live: false, BuildProbed: true, Builds: true},
			wantClass:     "resident",
			wantAbandoned: false, wantLandOrPark: false, wantPoison: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := tc.f
			classifyWIP(&f, abandonSec)
			if f.Class != tc.wantClass {
				t.Errorf("class = %q, want %q", f.Class, tc.wantClass)
			}
			if f.Abandoned != tc.wantAbandoned {
				t.Errorf("abandoned = %v, want %v", f.Abandoned, tc.wantAbandoned)
			}
			if f.LandOrPark != tc.wantLandOrPark {
				t.Errorf("landOrPark = %v, want %v", f.LandOrPark, tc.wantLandOrPark)
			}
			if f.Poison != tc.wantPoison {
				t.Errorf("poison = %v, want %v", f.Poison, tc.wantPoison)
			}
		})
	}
}

// TestDiagnoseWIPSurfacesAbandonedNotLive is the acceptance witness (#3210): a genuinely
// abandoned untracked file — and a stale build-poison file — are SURFACED (never removed),
// while a live session's in-flight file is NOT flagged abandoned. It runs the real gatherer
// over real temp files (for the mtime reads) with git ls-files and the build probe injected.
func TestDiagnoseWIPSurfacesAbandonedNotLive(t *testing.T) {
	root := t.TempDir()
	now := time.Now()

	mustWriteWIP(t, root, "pkg/live.go", "package pkg\n")
	mustWriteWIP(t, root, "pkg/abandoned.go", "package pkg\n")
	mustWriteWIP(t, root, "broken/poison.go", "package broken\n")
	setMtimeWIP(t, root, "pkg/live.go", now.Add(-1*time.Minute)) // within live window
	setMtimeWIP(t, root, "pkg/abandoned.go", now.Add(-3*time.Hour))
	setMtimeWIP(t, root, "broken/poison.go", now.Add(-3*time.Hour))

	run := func(_ context.Context, _ string, args ...string) (string, int, error) {
		if len(args) >= 1 && args[0] == "ls-files" {
			return "pkg/live.go\npkg/abandoned.go\nbroken/poison.go\n", 0, nil
		}
		return "", 0, nil
	}
	wopts := WIPOptions{
		AbandonAfter: time.Hour,
		// Only package "broken" fails to compile — the shared-trunk poison.
		BuildProbe: func(pkgDir string) bool { return pkgDir != "broken" },
	}

	rep := Diagnose(context.Background(), run, Options{
		RepoRoot: root, Now: now, LiveWindow: 10 * time.Minute, WIP: wopts,
	})

	byPath := map[string]WIPFile{}
	for _, f := range rep.WIP {
		byPath[f.Path] = f
	}
	if len(byPath) != 3 {
		t.Fatalf("expected 3 WIP files, got %d: %+v", len(byPath), rep.WIP)
	}

	if live := byPath["pkg/live.go"]; live.Abandoned || live.LandOrPark || live.Class != "live" {
		t.Errorf("live in-flight file mis-flagged (must not be abandoned/park): %+v", live)
	}
	if ab := byPath["pkg/abandoned.go"]; !ab.Abandoned || !ab.LandOrPark || ab.Class != "abandoned" || ab.Poison {
		t.Errorf("abandoned file not surfaced correctly: %+v", ab)
	}
	if p := byPath["broken/poison.go"]; !p.Poison || !p.LandOrPark || p.Class != "poison" {
		t.Errorf("build-poison file not surfaced correctly: %+v", p)
	}

	// Read-only guarantee: the surface never removes untracked source.
	for _, rel := range []string{"pkg/live.go", "pkg/abandoned.go", "broken/poison.go"} {
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel))); err != nil {
			t.Errorf("tree-doctor removed untracked source %s (must be read-only): %v", rel, err)
		}
	}
}

func mustWriteWIP(t *testing.T, root, rel, body string) {
	t.Helper()
	abs := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(abs, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func setMtimeWIP(t *testing.T, root, rel string, mt time.Time) {
	t.Helper()
	abs := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.Chtimes(abs, mt, mt); err != nil {
		t.Fatal(err)
	}
}
