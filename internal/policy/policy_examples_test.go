package policy

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// isPolicyManifest reports whether an examples/*.json file is a fak POLICY manifest
// (this package's schema) rather than a different schema that merely shares the
// examples/ dir — e.g. examples/model-routing.example.json is a `fak-route/v1` model
// routing config, which a policy parser correctly rejects ("unknown field"). The
// glob below must only validate POLICY files; a sibling schema dropped into examples/
// would otherwise fail this test for being the wrong (but valid) kind of file. An
// untagged manifest defaults to the current policy version, so it still counts.
func isPolicyManifest(b []byte) bool {
	var probe struct {
		Version string `json:"version"`
	}
	if err := json.Unmarshal(b, &probe); err != nil {
		return true // not parseable as JSON-with-version: let ParseRuntime report it
	}
	return probe.Version == "" || strings.HasPrefix(probe.Version, "fak-policy/")
}

func TestExamplePoliciesParse(t *testing.T) {
	paths, err := filepath.Glob("../../examples/*.json")
	if err != nil {
		t.Fatalf("glob examples: %v", err)
	}
	if len(paths) == 0 {
		t.Fatal("no example policies found")
	}
	for _, path := range paths {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			b, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read %s: %v", path, err)
			}
			if !isPolicyManifest(b) {
				t.Skipf("%s is not a fak-policy manifest (different schema sharing examples/)", filepath.Base(path))
			}
			if _, err := ParseRuntime(b); err != nil {
				t.Fatalf("parse %s: %v", path, err)
			}
		})
	}
}
