package modelroute

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestRoutingPresetsRoundTrip is the CI gate for the named routing-preset pack
// (examples/routing-presets/, issue #614). It is the routing analogue of
// internal/policy.TestPresetsRoundTrip: routing presets are a DIFFERENT schema +
// loader (fak-route/v1, modelroute.ParseManifest) from the policy-floor pack in
// examples/presets/ (fak-policy/v1), so they live under their own directory and
// are witnessed HERE, not by the policy test. Every shipped preset must
// (a) load through ParseManifest — i.e. pass `fak route --check` — and (b)
// round-trip EXACTLY: the manifest parsed off disk, re-rendered with JSON() (the
// same path `fak route --dump` uses), must reproduce the SAME bytes. The
// byte-equality rung is the "a preset can't rot" guard: a hand-edit that drifts
// from canonical form, or that introduces a field the manifest loader does not
// carry, fails the build instead of silently shipping a route different from the
// one reviewed.
func TestRoutingPresetsRoundTrip(t *testing.T) {
	paths, err := filepath.Glob("../../examples/routing-presets/*.json")
	if err != nil {
		t.Fatalf("glob routing presets: %v", err)
	}
	if len(paths) == 0 {
		t.Fatal("no presets found under examples/routing-presets/ — the pack is missing")
	}
	for _, path := range paths {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read %s: %v", path, err)
			}

			// A tombstoned preset (its routing shape retired with its provider,
			// e.g. the hive-hosted ops-desk desk) is not a live manifest: the
			// stub contract is witnessed here and the parse/round-trip rungs
			// are skipped — there is no canonical manifest form left to rot.
			if stub, ok := tombstoneStub(raw); ok {
				for _, k := range []string{"tombstoned", "reason", "see"} {
					if stub[k] == "" {
						t.Fatalf("%s is a tombstone stub but is missing a non-empty %q field", path, k)
					}
				}
				return
			}

			// (a) loads — passes `fak route --check`.
			m, err := ParseManifest(raw)
			if err != nil {
				t.Fatalf("%s fails --check: %v", path, err)
			}
			if err := m.Validate(); err != nil {
				t.Fatalf("%s fails validate: %v", path, err)
			}

			// (b) byte-exact round trip: re-rendering the parsed manifest
			// reproduces the file on disk (no field rots or drifts from the
			// canonical form `fak route --dump` emits).
			canon := m.JSON()
			if string(canon) != string(raw) {
				t.Fatalf("%s is not in canonical form (round-trip not exact).\n"+
					"Run `fak route`-equivalent canonicalization: ParseManifest(bytes).JSON().\n"+
					"--- canonical (%d bytes) ---\n%s\n--- file (%d bytes) ---\n%s",
					path, len(canon), string(canon), len(raw), string(raw))
			}
		})
	}
}

// tombstoneStub reports whether raw is a retirement tombstone — a stub body
// carrying a "tombstoned" field rather than a live fak-route/v1 manifest — and
// returns its string fields.
func tombstoneStub(raw []byte) (map[string]string, bool) {
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, false
	}
	if _, ok := doc["tombstoned"]; !ok {
		return nil, false
	}
	out := make(map[string]string, 3)
	for _, k := range []string{"tombstoned", "reason", "see"} {
		s, _ := doc[k].(string)
		out[k] = s
	}
	return out, true
}
