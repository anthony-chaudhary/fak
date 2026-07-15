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
		Version      string          `json:"version"`
		Schema       string          `json:"schema"`
		RequiredTier *int            `json:"required_tier"`
		Candidate    string          `json:"candidate"`
		Policies     json.RawMessage `json:"policies"`
		Observations json.RawMessage `json:"observations"`
	}
	if err := json.Unmarshal(b, &probe); err != nil {
		return true // not parseable as JSON-with-version: let ParseRuntime report it
	}
	if probe.Schema != "" {
		// A top-level "schema" key is a different family's self-tag (e.g.
		// fak.resume-source-policy.v1) — policy manifests tag with "version".
		return false
	}
	// The modelops canary input predates an explicit schema tag. Its structural
	// discriminator keeps that sibling example out of the policy parser while
	// preserving fail-closed parsing for ordinary unversioned policy manifests.
	if probe.RequiredTier != nil && probe.Candidate != "" && len(probe.Policies) != 0 && len(probe.Observations) != 0 {
		return false
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
