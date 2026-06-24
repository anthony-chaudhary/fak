package policy

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// TestPresetsRoundTrip is the CI gate for the curated preset pack
// (examples/presets/, issue #578). Every shipped preset must (a) load through
// ParseRuntime — i.e. pass `fak policy --check`, every deny citing a
// closed-vocabulary reason — and (b) round-trip EXACTLY: the manifest, loaded to
// a Policy and re-rendered with FromPolicy (the same path `fak policy --dump`
// uses), must reproduce the SAME floor when re-parsed, AND the re-rendered bytes
// must equal the file on disk. The byte-equality rung is the "a preset can't
// rot" guard: a hand-edit that drifts from canonical form, or that introduces a
// field the manifest loader does not carry, fails the build instead of silently
// shipping a floor different from the one reviewed.
func TestPresetsRoundTrip(t *testing.T) {
	paths, err := filepath.Glob("../../examples/presets/*.json")
	if err != nil {
		t.Fatalf("glob presets: %v", err)
	}
	if len(paths) == 0 {
		t.Fatal("no presets found under examples/presets/ — the pack is missing")
	}
	for _, path := range paths {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read %s: %v", path, err)
			}

			// (a) loads — passes `fak policy --check`.
			rt, err := ParseRuntime(raw)
			if err != nil {
				t.Fatalf("%s fails --check: %v", path, err)
			}

			// (b) round-trips at the Policy level: load -> FromPolicy -> ToPolicy
			// reconstructs the same floor (no field is silently dropped).
			rt2, err := FromPolicy(rt.Adjudicator).ToRuntime()
			if err != nil {
				t.Fatalf("re-parse of FromPolicy output: %v", err)
			}
			if !reflect.DeepEqual(rt.Adjudicator, rt2.Adjudicator) {
				t.Fatalf("policy round-trip drift for %s:\n want=%+v\n got =%+v",
					path, rt.Adjudicator, rt2.Adjudicator)
			}

			// (b, byte rung) the canonical re-render equals the file on disk.
			canon := FromPolicy(rt.Adjudicator).JSON()
			if string(canon) != string(raw) {
				t.Fatalf("%s is not in canonical form (round-trip not exact).\n"+
					"Run `fak policy`-equivalent canonicalization: FromPolicy(Parse(bytes)).JSON().\n"+
					"--- canonical (%d bytes) ---\n%s\n--- file (%d bytes) ---\n%s",
					path, len(canon), string(canon), len(raw), string(raw))
			}
		})
	}
}
