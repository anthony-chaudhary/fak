package policy

// loaderror_test.go — witnesses for the typed LoadRuntime failure.
//
// The type exists so a CALLER can react to a floor that would not load, not just
// print it. `fak serve --policy <bad>` used to die with a bare
// `fak: policy floor.json: ...` and no next step; cmd/fak's must() now recognizes
// *LoadError and renders the knob, the check, and the recovery. That upgrade is
// only sound if two things hold, which is what these tests hold onto: the error
// is ALWAYS this type (so the caller's errors.As cannot silently miss), and the
// message is unchanged (so nothing matching on text regressed when it was added).

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

func TestLoadRuntimeReturnsTypedReadFailure(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "absent.json")

	_, err := LoadRuntime(missing)
	if err == nil {
		t.Fatal("loading an absent floor must fail")
	}
	var le *LoadError
	if !errors.As(err, &le) {
		t.Fatalf("error is %T, not *LoadError — cmd/fak's must() dispatches on this type, so an untyped failure silently loses its next step", err)
	}
	if le.Op != LoadOpRead {
		t.Errorf("Op = %q, want %q: an unreadable path and an invalid manifest need different next steps", le.Op, LoadOpRead)
	}
	if le.Path != missing {
		t.Errorf("Path = %q, want %q — the bail names this path as the knob to fix", le.Path, missing)
	}
	// The wrapped cause has to survive: a caller checking os.IsNotExist, or any
	// errors.Is against a filesystem sentinel, must still work through the wrap.
	if !errors.Is(err, os.ErrNotExist) {
		t.Errorf("wrapped read error lost its cause: %v", err)
	}
	// Message shape is load-bearing for anything that matched on it before the
	// type existed.
	if got := err.Error(); !strings.HasPrefix(got, "policy: ") {
		t.Errorf("read failure message = %q, want the original \"policy: …\" prefix", got)
	}
}

func TestLoadRuntimeReturnsTypedParseFailure(t *testing.T) {
	path := filepath.Join(t.TempDir(), "floor.json")
	if err := os.WriteFile(path, []byte("this is not a manifest"), 0o600); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	_, err := LoadRuntime(path)
	if err == nil {
		t.Fatal("loading an unparseable floor must fail — a floor that will not parse is never downgraded to a permissive default")
	}
	var le *LoadError
	if !errors.As(err, &le) {
		t.Fatalf("error is %T, not *LoadError", err)
	}
	if le.Op != LoadOpParse {
		t.Errorf("Op = %q, want %q", le.Op, LoadOpParse)
	}
	if le.Path != path {
		t.Errorf("Path = %q, want %q", le.Path, path)
	}
	if got := err.Error(); !strings.HasPrefix(got, "policy "+path+": ") {
		t.Errorf("parse failure message = %q, want the original \"policy <path>: …\" shape", got)
	}
}

// A floor that loads must not be reported as a failure — the guard against a
// typed-error refactor that accidentally fails closed on the happy path.
func TestLoadRuntimeSucceedsOnTheDefaultManifest(t *testing.T) {
	path := filepath.Join(t.TempDir(), "default.json")
	// The same bytes `fak policy --dump` emits, which is what the recovery tells
	// an operator to diff their file against.
	if err := os.WriteFile(path, FromPolicy(adjudicator.DefaultPolicy()).JSON(), 0o600); err != nil {
		t.Fatalf("write default manifest: %v", err)
	}
	if _, err := LoadRuntime(path); err != nil {
		t.Fatalf("the default manifest must load: %v", err)
	}
}
