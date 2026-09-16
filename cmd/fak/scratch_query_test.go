package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// scratchQueryCLI runs the testable `fak scratch` core with explicit streams.
func scratchQueryCLI(argv ...string) (string, string, int) {
	var out, errb bytes.Buffer
	code := runScratch(&out, &errb, argv)
	return out.String(), errb.String(), code
}

// writeCLIScratchpad lays out <root>/<project>/<session>/scratchpad with files.
func writeCLIScratchpad(t *testing.T, root, project, session string, files map[string]string) {
	t.Helper()
	pad := filepath.Join(root, project, session, "scratchpad")
	if err := os.MkdirAll(pad, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(pad, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// TestRunScratchQueryContentMatches drives content mode through the real dispatch:
// the transcript resolves the scratchpad address via deriveScratchpadPath, and the
// bounded content match reports the hit.
func TestRunScratchQueryContentMatches(t *testing.T) {
	root := t.TempDir()
	projDir := filepath.Join(t.TempDir(), "projects", "project-slug")
	if err := os.MkdirAll(projDir, 0o755); err != nil {
		t.Fatal(err)
	}
	transcript := filepath.Join(projDir, "abc123.jsonl")
	if err := os.WriteFile(transcript, []byte("{}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	writeCLIScratchpad(t, root, "project-slug", "abc123", map[string]string{"note.md": "irrelevant\nneedle here\n"})

	stdout, stderr, code := scratchQueryCLI("query", "--session", "abc123", "--transcript", transcript, "--root", root, "--pattern", "needle")
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %s", code, stderr)
	}
	if !bytes.Contains([]byte(stdout), []byte("needle here")) {
		t.Fatalf("stdout = %s, want the matched line", stdout)
	}
}

// TestRunScratchQueryPromotableAttachesDiagnosis is the #13069 acceptance: across
// dead sessions in a transcript store, parked artifacts that look like session
// flags are surfaced WITH the session resume diagnosis attached, read-only.
func TestRunScratchQueryPromotableAttachesDiagnosis(t *testing.T) {
	store := writeStore(t) // crashed0 (rate limit, unresumed), cleanab1, othererr
	scratchRoot := t.TempDir()
	writeCLIScratchpad(t, scratchRoot, "project-slug", "crashed0", map[string]string{
		"leftover.md": "// unfinished-spine: could not ship the end-to-end path\n",
	})
	writeCLIScratchpad(t, scratchRoot, "project-slug", "cleanab1", map[string]string{
		"note.md": "ordinary note, no marker\n",
	})

	stdout, stderr, code := scratchQueryCLI("query", "--promotable", "--store", store, "--root", scratchRoot, "--json")
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %s", code, stderr)
	}
	var doc struct {
		Taxonomy   []string `json:"taxonomy"`
		Candidates []struct {
			Session   string `json:"session_id"`
			Kind      string `json:"kind"`
			File      string `json:"file"`
			Diagnosis struct {
				Crash     string `json:"crash"`
				Unresumed bool   `json:"unresumed"`
			} `json:"diagnosis"`
		} `json:"candidates"`
		Advisory bool `json:"advisory"`
		ReadOnly bool `json:"read_only"`
	}
	if err := json.Unmarshal([]byte(stdout), &doc); err != nil {
		t.Fatalf("decode %q: %v", stdout, err)
	}
	if len(doc.Taxonomy) != 5 {
		t.Fatalf("taxonomy = %#v, want 5 closed kinds", doc.Taxonomy)
	}
	if len(doc.Candidates) != 1 {
		t.Fatalf("candidates = %#v, want the one flag artifact", doc.Candidates)
	}
	c := doc.Candidates[0]
	if c.Kind != "unfinished-spine" || c.File != "leftover.md" || c.Session != "crashed0" {
		t.Fatalf("candidate = %#v", c)
	}
	if c.Diagnosis.Crash != "rate_limit" || !c.Diagnosis.Unresumed {
		t.Fatalf("diagnosis not attached: %#v", c.Diagnosis)
	}
	if !doc.Advisory || !doc.ReadOnly {
		t.Fatalf("promotable surface must be advisory+read-only: %#v", doc)
	}
	if _, err := os.Stat(filepath.Join(scratchRoot, "project-slug", "crashed0", "scratchpad", "leftover.md")); err != nil {
		t.Fatalf("query mutated the scratchpad: %v", err)
	}
}

// TestRunScratchQueryPromotableMissingStoreIsUsageError pins the closed exit code.
func TestRunScratchQueryPromotableMissingStoreIsUsageError(t *testing.T) {
	_, _, code := scratchQueryCLI("query", "--promotable")
	if code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
}
