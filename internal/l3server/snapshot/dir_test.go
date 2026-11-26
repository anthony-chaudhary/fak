package snapshot

import (
	"path/filepath"
	"runtime"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
// Estimate only: pure path arithmetic, no filesystem IO.
func TestConfineDir(t *testing.T) {
	t.Parallel()
	base := filepath.Join(t.TempDir(), "snapshots")

	t.Run("base_itself", func(t *testing.T) {
		for _, req := range []string{base, base + string(filepath.Separator), "."} {
			got, err := ConfineDir(base, req)
			if err != nil {
				t.Fatalf("ConfineDir(%q, %q): %v", base, req, err)
			}
			if got != base {
				t.Fatalf("ConfineDir(%q, %q) = %q, want %q", base, req, got, base)
			}
		}
	})

	t.Run("contained_subdirs", func(t *testing.T) {
		for _, req := range []string{"sub", filepath.Join("sub", "deeper"), "sub" + string(filepath.Separator)} {
			got, err := ConfineDir(base, req)
			if err != nil {
				t.Fatalf("ConfineDir(%q, %q): %v", base, req, err)
			}
			if want := filepath.Join(base, filepath.Clean(req)); got != want {
				t.Fatalf("ConfineDir(%q, %q) = %q, want %q", base, req, got, want)
			}
		}
	})

	t.Run("escapes_refused", func(t *testing.T) {
		for _, req := range []string{"..", filepath.Join("..", "sibling"), filepath.Join("..", "..", "etc")} {
			if got, err := ConfineDir(base, req); err == nil {
				t.Fatalf("ConfineDir(%q, %q) = %q, want refusal", base, req, got)
			}
		}
	})

	t.Run("absolute_outside_base_refused", func(t *testing.T) {
		// An absolute path in a different tree than the base is refused: Rel
		// reduces it to a ".."-prefixed escape.
		abs := filepath.Join(t.TempDir(), "elsewhere")
		if got, err := ConfineDir(base, abs); err == nil {
			t.Fatalf("ConfineDir(%q, %q) = %q, want refusal", base, abs, got)
		}
	})

	t.Run("absolute_under_base_allowed", func(t *testing.T) {
		// The normal client shape: the absolute path of a directory inside
		// the configured base is accepted and normalized to that path.
		abs := filepath.Join(base, "sub", "deeper")
		got, err := ConfineDir(base, abs)
		if err != nil {
			t.Fatalf("ConfineDir(%q, %q): %v", base, abs, err)
		}
		if got != abs {
			t.Fatalf("ConfineDir(%q, %q) = %q, want %q", base, abs, got, abs)
		}
	})

	t.Run("empty_base_refused", func(t *testing.T) {
		if got, err := ConfineDir("", "sub"); err == nil {
			t.Fatalf("ConfineDir(\"\", \"sub\") = %q, want refusal", got)
		}
	})

	t.Run("cleans_dot_segments", func(t *testing.T) {
		req := filepath.Join("sub", "..", "sub2")
		got, err := ConfineDir(base, req)
		if err != nil {
			t.Fatalf("ConfineDir(%q, %q): %v", base, req, err)
		}
		if want := filepath.Join(base, "sub2"); got != want {
			t.Fatalf("ConfineDir(%q, %q) = %q, want %q", base, req, got, want)
		}
	})

	if runtime.GOOS == "windows" {
		t.Run("windows_cross_volume_refused", func(t *testing.T) {
			// A request on another drive than the base cannot be made relative
			// and must be refused, not silently joined.
			if got, err := ConfineDir(`C:\snapshots`, `D:\elsewhere`); err == nil {
				t.Fatalf("ConfineDir cross-volume = %q, want refusal", got)
			}
		})
	}
}
