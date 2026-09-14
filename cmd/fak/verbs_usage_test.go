package main

// C3 of epic #1287 (issue #1290): the drift gate over the generated runtime verb
// index. `cmd/fak/verbs_gen.go` is rendered from the devindex catalog by
// `fak-dev index verbs --write-usage`. These tests make a stale artifact and an
// uncataloged dispatch verb deterministic build failures, which is the C3
// acceptance criterion ("a gate reds when a main.go case has no manifest entry").
//
// Test-only: this file may import internal/devindex to regenerate the artifact.
// Production runtime help must not (help.go); usageVerbs honors that by reading
// the committed local slice, never the catalog.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

// TestGeneratedVerbUsageIsCurrent regenerates the runtime verb index from the
// live tree and requires the committed cmd/fak/verbs_gen.go to match byte for
// byte. A failure means someone edited the dispatch switch or the curated
// manifest without re-running `fak-dev index verbs --write-usage`.
func TestGeneratedVerbUsageIsCurrent(t *testing.T) {
	root := devindex.FindRoot(".")
	want, err := devindex.RenderVerbUsage(root)
	if err != nil {
		t.Fatalf("RenderVerbUsage(%q): %v", root, err)
	}
	got, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(devindex.VerbUsagePath)))
	if err != nil {
		t.Fatalf("read %s: %v", devindex.VerbUsagePath, err)
	}
	if string(got) != string(want) {
		t.Fatalf("%s is stale; run `fak-dev index verbs --write-usage --root %s`",
			devindex.VerbUsagePath, root)
	}
}

// TestGeneratedVerbIndexCoversDispatchSwitch is the issue's explicit gate: every
// verb the binary's top-level dispatch switch routes must appear in the generated
// index. devindex's shared brace-depth scan is the coverage authority (the same
// parser the freshness detector uses), so the gate and the detector cannot
// disagree on what a verb is.
func TestGeneratedVerbIndexCoversDispatchSwitch(t *testing.T) {
	root := devindex.FindRoot(".")
	src, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "main.go"))
	if err != nil {
		t.Fatalf("read main.go: %v", err)
	}
	have := map[string]bool{}
	for _, v := range generatedVerbIndex {
		have[v.Name] = true
		for _, a := range v.Aliases {
			have[a] = true
		}
	}
	var missing []string
	for _, tok := range devindex.DispatchVerbs(src) {
		if !have[tok] {
			missing = append(missing, tok)
		}
	}
	if len(missing) != 0 {
		t.Fatalf("dispatch switch routes %d verb(s) absent from the generated index: %v; "+
			"regenerate with `fak-dev index verbs --write-usage`", len(missing), missing)
	}
}

// TestUsageVerbsPrintsTheGeneratedIndex proves `fak help --verbs` renders the
// committed slice (not the authored wall): it must contain a known frontdoor
// verb's generated synopsis and must stay one-line-per-verb.
func TestUsageVerbsPrintsTheGeneratedIndex(t *testing.T) {
	var b strings.Builder
	usageVerbs(&b)
	out := b.String()
	if !strings.Contains(out, "verb index (generated from the devindex catalog") {
		t.Fatalf("fak help --verbs lacks its generated header:\n%s", out)
	}
	for _, v := range generatedVerbIndex {
		if !strings.Contains(out, "fak "+v.Name) {
			t.Errorf("fak help --verbs omitted %q", v.Name)
		}
	}
}