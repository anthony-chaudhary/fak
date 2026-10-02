package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1s
func TestConceptAdmissionCommittedRange(t *testing.T) {
	_, root := conceptCLIFixture(t)
	emptyHooks := t.TempDir()
	git := func(args ...string) string {
		t.Helper()
		prefix := []string{"-c", "core.hooksPath=" + emptyHooks, "-c", "user.name=fak-test", "-c", "user.email=fak@example.invalid"}
		out, err := runGitAt(root, append(prefix, args...)...)
		if err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
		return strings.TrimSpace(out)
	}
	write := func(path, text string) {
		t.Helper()
		path = filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(text), 0600); err != nil {
			t.Fatal(err)
		}
	}
	commit := func(message string) string {
		t.Helper()
		git("add", ".")
		git("commit", "-m", message)
		return git("rev-parse", "HEAD")
	}
	const source = "internal/demo/demo.go"
	const rows = "tools/concept_disambiguation_scorecard.data/rows-cache.json"
	git("init")
	write(source, "package demo\nconst CacheExisting = 1\n")
	write(rows, `{"rows":[]}`)
	base := commit("base")
	write(source, "package demo\nconst CacheExisting = 1\nconst CacheAdded = 2\n")
	introduced := commit("introduce unpositioned token")
	write("note.txt", "unrelated intermediate change\n")
	middle := commit("advance past the introduction")
	write(source, "package demo\nconst CacheExisting = 1\nconst CacheAdded = 2\nfunc value() int { return CacheExisting }\n")
	tip := commit("later use of an established token")
	if got := git("rev-parse", tip+"^"); got != middle || got == introduced || got == base {
		t.Fatalf("fixture must introduce CacheAdded before tip's parent: parent=%s base=%s", got, base)
	}
	write(rows, `{"rows":[{"id":"cache-added","family":"cache","grounding":"CacheAdded"}]}`)
	positioned := commit("position the new token")

	// Neither staged classification nor working-tree source may change the
	// verdict for the already committed base..tip objects.
	write("tools/concept_disambiguation_scorecard.data/_meta.json", `{"families":[{"id":"cache","roots":["cache"],"ignore":["cacheadded"]}]}`)
	git("add", "tools/concept_disambiguation_scorecard.data/_meta.json")
	write(source, "package demo\nconst CacheWorktreeOnly = 3\n")
	t.Chdir(root)

	for _, tc := range []struct {
		name string
		args []string
		code int
		json bool
		find bool
	}{
		{"earlier_introduction_json", []string{"--base", base, "--tip", tip, "--json"}, 1, true, true},
		{"earlier_introduction_text", []string{"--base", base, "--tip", tip}, 1, false, true},
		{"positioned_at_tip", []string{"--base", base, "--tip", positioned, "--json"}, 0, true, false},
		{"empty_range", []string{"--base", base, "--tip", base, "--json"}, 0, true, false},
		{"invalid_base", []string{"--base", "missing-base-ref", "--tip", tip, "--json"}, 1, false, false},
		{"invalid_tip", []string{"--base", base, "--tip", "missing-tip-ref", "--json"}, 1, false, false},
		{"missing_tip", []string{"--base", base}, 2, false, false},
		{"missing_base", []string{"--tip", tip}, 2, false, false},
		{"empty_range_arguments", []string{"--base", "", "--tip", ""}, 2, false, false},
		{"mixed_paths", []string{"--base", base, "--tip", tip, "--paths", source}, 2, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out, errb bytes.Buffer
			code := runConcept(&out, &errb, append([]string{"admission"}, tc.args...))
			if code != tc.code {
				t.Fatalf("code=%d want=%d; stdout=%s stderr=%s", code, tc.code, &out, &errb)
			}
			if tc.json {
				var result struct {
					Schema   string `json:"schema"`
					OK       bool   `json:"ok"`
					Findings []struct {
						File   string `json:"file"`
						Detail string `json:"detail"`
					} `json:"findings"`
				}
				if err := json.Unmarshal(out.Bytes(), &result); err != nil {
					t.Fatalf("JSON result: %v: %s", err, &out)
				}
				if result.Schema != "fak.concept_admission.v1" || result.OK == tc.find {
					t.Fatalf("unexpected result: %s", &out)
				}
				if tc.find && (len(result.Findings) != 1 || result.Findings[0].File != source || !strings.Contains(result.Findings[0].Detail, "token=CacheAdded ")) {
					t.Fatalf("want only the requested range's new token: %s", &out)
				}
				if !tc.find && len(result.Findings) != 0 {
					t.Fatalf("finding-free range: %s", &out)
				}
			} else if tc.find {
				if !strings.Contains(out.String(), "CONCEPT_ADMISSION "+source+":") || !strings.Contains(out.String(), "token=CacheAdded ") {
					t.Fatalf("missing text refusal: %s", &out)
				}
			} else if errb.Len() == 0 {
				t.Fatalf("invalid arguments/ref must explain the refusal: stdout=%s", &out)
			}
		})
	}
}
