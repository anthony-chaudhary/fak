package main

import (
	"testing"
)

// fak-test:runtime fast est=1s lane=default
func TestAgentRepoMapOptIn(t *testing.T) {
	fs, af := newAgentFlagSet()
	if err := fs.Parse([]string{"--native", "--repo-map-file", "a.go", "--repo-map-file", "b.go", "--repo-map-tokens", "512"}); err != nil {
		t.Fatal(err)
	}
	opt, err := resolveAgentRepoMapOption(af, true, false)
	if err != nil || opt == nil {
		t.Fatalf("opt-in unavailable: %v", err)
	}
	if len(af.repoMapFiles) != 2 || *af.repoMapTokens != 512 {
		t.Fatal("flags not wired")
	}
	if _, err := resolveAgentRepoMapOption(af, false, false); err == nil {
		t.Fatal("raw/dual accepted")
	}
	*af.resume = "old"
	if _, err := resolveAgentRepoMapOption(af, true, false); err == nil {
		t.Fatal("resume accepted")
	}
	_, plain := newAgentFlagSet()
	if opt, err := resolveAgentRepoMapOption(plain, false, true); err != nil || opt != nil {
		t.Fatal("default changed")
	}
	for _, p := range []string{"../x.go", "/x.go", "a\\b.go", ".secret/x.go", "x.txt"} {
		var files repoMapFileList
		if files.Set(p) == nil {
			t.Fatalf("bad selected path %q", p)
		}
	}
}
