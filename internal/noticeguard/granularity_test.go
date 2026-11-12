// Package noticeguard is a test-only regression guard for the public license notices.
// The notice surfaces (NOTICE.md, THIRD_PARTY_NOTICES.md, third_party/*/NOTICE) must stay at
// the license-required minimum: donor identity + repo URL + pinned revisions, verbatim
// copyright lines, full license texts, and the "Changes were made" notice. The guard refuses
// over-specific attribution — fak-internal paths, adaptation narratives, internal env vars,
// Go test identifiers, donor-side source-mapping lists — that would reveal internal build
// structure instead of just the donor provenance the license requires.
package noticeguard

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// bannedTokens is the closed table of over-specific markers a license-minimal notice must not
// contain; add future tokens here, one line each.
var bannedTokens = []struct {
	name string         // what the marker reveals (reported on failure)
	re   *regexp.Regexp // case-sensitive matcher
}{
	{name: "fak-internal path", re: regexp.MustCompile(`internal/`)},
	{name: "adaptation narrative", re: regexp.MustCompile(`(?:Changed|Changes) for Fak`)},
	{name: "internal env var", re: regexp.MustCompile(`FAK_`)},
	{name: "Go test identifier", re: regexp.MustCompile(`_test\.go|\bTest[A-Z][A-Za-z0-9_]*\b`)},
	{name: "donor-side source-mapping list", re: regexp.MustCompile(`Source units`)},
}

// findRepoRootForTest walks up from the test working directory to the module root (the
// directory containing go.mod), mirroring the idiom in internal/compute/scaffold_test.go.
func findRepoRootForTest(t *testing.T, start string) string {
	t.Helper()
	dir := start
	for i := 0; i < 8; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Fatalf("could not find go.mod walking up from %s", start)
	return ""
}

// fak-test:runtime fast est=10ms lane=default
func TestNoticesStayLicenseMinimal(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	root := findRepoRootForTest(t, wd)

	surfaces := []string{
		filepath.Join(root, "NOTICE.md"),
		filepath.Join(root, "THIRD_PARTY_NOTICES.md"),
	}
	globbed, err := filepath.Glob(filepath.Join(root, "third_party", "*", "NOTICE"))
	if err != nil {
		t.Fatal(err)
	}
	surfaces = append(surfaces, globbed...)

	scanned := 0
	for _, p := range surfaces {
		data, err := os.ReadFile(p)
		if err != nil {
			if os.IsNotExist(err) {
				continue // partial checkout: skip surfaces not materialized on disk
			}
			t.Fatal(err)
		}
		scanned++
		rel, err := filepath.Rel(root, p)
		if err != nil {
			rel = p
		}
		rel = filepath.ToSlash(rel)
		t.Run(rel, func(t *testing.T) {
			for i, line := range strings.Split(string(data), "\n") {
				for _, tok := range bannedTokens {
					if loc := tok.re.FindStringIndex(line); loc != nil {
						t.Errorf("%s line %d: license-minimal notice must not contain %s (%q)", rel, i+1, tok.name, line[loc[0]:loc[1]])
					}
				}
			}
		})
	}
	if scanned == 0 {
		t.Fatalf("no notice surface found under %s (expected NOTICE.md, THIRD_PARTY_NOTICES.md, or third_party/*/NOTICE)", root)
	}
}
