package architest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ciyaml"
)

// TestIssueTemplatesAreWellFormed is the always-on gate for .github/ISSUE_TEMPLATE.
//
// It exists because the directory had NO gate at all and shipped broken. The
// don't-make-me-think problem-map rollout (45fafabb35) added a "Primary fak
// problem" dropdown to feature-request.yml and welded its first line onto the
// previous field's `required: true`, producing `required: true  - type: dropdown`.
// That is unparseable YAML, and GitHub's response to an unparseable issue form is
// to DROP it from the chooser — no error, no red build, nothing anywhere in the
// repo that looks wrong. The feature-request form was simply absent for everyone
// who tried to file one, and the only signal was the absence itself.
//
// The two failure modes below are the two ways that happens, and neither is caught
// by anything else in `go test ./...`:
//
//  1. STRUCTURE — the file does not parse, so the form vanishes. ciyaml.Check is
//     the repo's stdlib-only structural checker (no YAML library is vendored, and
//     that dependency floor is a standing contract), and this is its first caller:
//     the package was written for .github and, until now, nothing ran it.
//  2. SHAPE — the file parses but lacks a key GitHub requires of an issue FORM
//     (`name`, `description`, `body`). GitHub rejects it just as silently.
//
// The input set is WALKED, never enumerated: a template added tomorrow is covered
// the moment it lands. config.yml is a different schema in the same directory (the
// chooser's contact links, not a form), so it gets its own floor rather than a skip.
func TestIssueTemplatesAreWellFormed(t *testing.T) {
	root := filepath.Dir(internalDir(t)) // repo root = parent of internal/
	dir := filepath.Join(root, ".github", "ISSUE_TEMPLATE")
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read issue-template dir %s: %v", dir, err)
	}

	// A GitHub issue FORM needs all three; without them the template is ignored.
	formKeys := []string{"name", "description", "body"}

	checked := 0
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if ext := strings.ToLower(filepath.Ext(name)); ext != ".yml" && ext != ".yaml" {
			continue
		}
		rel := ".github/ISSUE_TEMPLATE/" + name
		data, readErr := os.ReadFile(filepath.Join(dir, name))
		if readErr != nil {
			t.Errorf("read %s: %v", rel, readErr)
			continue
		}
		checked++

		if issues := ciyaml.Check(rel, data); len(issues) > 0 {
			t.Errorf("%s is structurally invalid YAML — GitHub drops an unparseable issue "+
				"template from the chooser silently, so the form is simply gone with no error "+
				"anywhere:\n%s", rel, issues)
		}

		if name == "config.yml" || name == "config.yaml" {
			// The chooser config, not a form: different required shape.
			if !ciyaml.HasTopLevelKey(data, "contact_links") {
				t.Errorf("%s is the issue-chooser config but has no top-level \"contact_links\" key", rel)
			}
			continue
		}
		for _, key := range formKeys {
			if !ciyaml.HasTopLevelKey(data, key) {
				t.Errorf("%s is missing required top-level key %q — GitHub ignores an issue form "+
					"that lacks it, with no error shown anywhere", rel, key)
			}
		}
	}

	// Fail closed: zero files checked means the directory moved or the extension
	// filter drifted, and a gate that examined nothing passes green either way.
	if checked == 0 {
		t.Fatal("no issue templates were checked — this gate would be silently inert; " +
			"the .github/ISSUE_TEMPLATE layout changed")
	}
	t.Logf("issue templates checked under .github/ISSUE_TEMPLATE: %d", checked)
}
