package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fixtureMemoryStore writes a markdown memory store whose three notes exercise
// the three read-time verdicts against the REAL DefaultArtifactVerifier run in
// this checkout: a note naming a path that exists (fresh), a note naming a
// deleted package + unresolvable commit (stale → withheld), and a prose note
// with nothing checkable (unverified, rendered hedged).
func fixtureMemoryStore(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"MEMORY.md": "# Memory index\n\n" +
			"- [Gate helper](fresh.md) — still true\n" +
			"- [Moved fix](stale.md) — names a gone artifact\n" +
			"- [Preference](prose.md) — prose only\n",
		"fresh.md": "---\nname: gate-helper\ndescription: where the algebra lives\nmetadata:\n  type: feedback\n---\n\nThe memory algebra executor lives in internal/memq/exec.go.\n",
		"stale.md": "---\nname: moved-fix\ndescription: an old location\nmetadata:\n  type: project\n---\n\nThe fix lives in internal/gonepkg/gone.go.\n",
		"prose.md": "---\nname: preference\ndescription: terse answers\nmetadata:\n  type: user\n---\n\nThe user prefers the outcome stated first.\n",
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// #2347 done-condition at the CLI: `fak memory recall` renders the still-true
// note tagged fresh, renders the prose note hedged, and WITHHOLDS the stale note
// with the failing claim named — the orientation block a loop turn injects.
func TestMemoryRecall_verifiedOrientationBlock(t *testing.T) {
	dir := fixtureMemoryStore(t)
	var out, errb strings.Builder
	code := runMemoryRecall(&out, &errb, []string{"--store", dir, "--intent", "memory algebra gate"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "fresh.md) [fresh]") {
		t.Errorf("still-true note must render tagged [fresh]; got:\n%s", s)
	}
	if !strings.Contains(s, "prose.md) [unverified]") {
		t.Errorf("prose-only note must render hedged [unverified]; got:\n%s", s)
	}
	if strings.Contains(s, "internal/gonepkg/gone.go.\n") || strings.Contains(s, "stale.md) [") {
		t.Errorf("stale note body must never render; got:\n%s", s)
	}
	if !strings.Contains(s, "stale.md [withheld:stale_recall_artifact]") {
		t.Errorf("stale note must be withheld with the reason named; got:\n%s", s)
	}
	if !strings.Contains(s, "internal/gonepkg/gone.go") {
		t.Errorf("the withheld line must name the failing claim as evidence; got:\n%s", s)
	}
}

// The JSON envelope parses and carries the same verdicts (the machine surface a
// loop harness consumes).
func TestMemoryRecall_jsonEnvelope(t *testing.T) {
	dir := fixtureMemoryStore(t)
	var out, errb strings.Builder
	if code := runMemoryRecall(&out, &errb, []string{"--store", dir, "--intent", "memory algebra gate", "--json"}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	var env recallEnvelope
	if err := json.Unmarshal([]byte(out.String()), &env); err != nil {
		t.Fatalf("envelope must parse: %v\n%s", err, out.String())
	}
	verdicts := map[string]string{}
	for _, n := range env.Rendered {
		verdicts[n.ID] = n.Verdict
		if n.Body == "" {
			t.Errorf("rendered note %s must carry its body", n.ID)
		}
	}
	for _, n := range env.Withheld {
		verdicts[n.ID] = n.Verdict
		if n.Body != "" {
			t.Errorf("withheld note %s must never carry a body", n.ID)
		}
	}
	if verdicts["fresh.md"] != "fresh" || verdicts["prose.md"] != "unverified" || verdicts["stale.md"] != "withheld:stale_recall_artifact" {
		t.Fatalf("verdicts = %+v", verdicts)
	}
}

// A missing store yields an empty block and exit 0 — the loop's recall step must
// fail open on a fresh node, never refuse the turn.
func TestMemoryRecall_missingStoreFailsOpen(t *testing.T) {
	var out, errb strings.Builder
	code := runMemoryRecall(&out, &errb, []string{"--store", filepath.Join(t.TempDir(), "nope"), "--intent", "anything"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "no notes rendered") {
		t.Errorf("empty store must say so; got:\n%s", out.String())
	}
}
