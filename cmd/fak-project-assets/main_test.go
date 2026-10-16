// fak-test:runtime fast est=1s
package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/projectassets"
)

const fixtureManifest = `{
 "schema":"fak-project-assets/1",
 "skills":{"canonical_root":".claude/skills","codex_root":".agents/skills","include":["SKILL.md"],"exclude":[]},
 "memories":{"canonical_root":".claude/memory","include":["*.md"],"exclude":[],"startup_command":"CLAUDE.md"},
 "goal_prompts":{"canonical_root":".claude/goal-prompts","include":["*.md"],"exclude":[]},
 "harnesses":{"claude":{"skills":"native","memories":"native","goal_prompts":"native"},"codex":{"skills":"generated-adapter","memories":"AGENTS.md","goal_prompts":"native"},"fak-native":{"skills":"native-loader","memories":"AGENTS.md","goal_prompts":"native-loader"},"opencode":{"skills":"generated-adapter","memories":"AGENTS.md","goal_prompts":"native"}}
}`

func writeFixture(t *testing.T, root, rel, body string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func assetsRoot(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	writeFixture(t, root, projectassets.ManifestPath, fixtureManifest)
	writeFixture(t, root, ".claude/skills/alpha/SKILL.md", "---\nname: alpha\ndescription: alpha skill\n---\nbody\n")
	writeFixture(t, root, ".claude/memory/base.md", "memory\n")
	writeFixture(t, root, ".claude/goal-prompts/base.md", "prompt\n")
	return root
}

func decodeReceipt(t *testing.T, out []byte) projectassets.Receipt {
	t.Helper()
	var r projectassets.Receipt
	if err := json.Unmarshal(out, &r); err != nil {
		t.Fatalf("stdout is not a JSON receipt: %v\n%s", err, out)
	}
	return r
}

func TestRunUsageExitCodes(t *testing.T) {
	for _, args := range [][]string{nil, {"bogus"}, {"parity", "--nope"}} {
		var out bytes.Buffer
		if code := runProjectAssets(&out, &bytes.Buffer{}, args); code != 2 {
			t.Fatalf("args %q: exit = %d, want 2", args, code)
		}
		if out.Len() != 0 {
			t.Fatalf("args %q: stdout = %q, want empty", args, out.String())
		}
	}
}

func TestRunMissingManifestExitsOne(t *testing.T) {
	var out bytes.Buffer
	if code := runProjectAssets(&out, &bytes.Buffer{}, []string{"parity", "--dir", t.TempDir()}); code != 1 {
		t.Fatalf("exit = %d, want 1", code)
	}
	if out.Len() != 0 {
		t.Fatalf("stdout on build error = %q, want empty", out.String())
	}
}

func TestRunParityThenSyncClosesGaps(t *testing.T) {
	root := assetsRoot(t)

	var before bytes.Buffer
	if code := runProjectAssets(&before, &bytes.Buffer{}, []string{"parity", "--dir", root, "--json"}); code != 1 {
		t.Fatalf("parity before sync: exit = %d, want 1", code)
	}
	if r := decodeReceipt(t, before.Bytes()); r.ZeroUnexplainedGaps || r.Schema != "fak-project-assets-parity/1" {
		t.Fatalf("parity before sync receipt = %+v", r)
	}

	var synced bytes.Buffer
	if code := runProjectAssets(&synced, &bytes.Buffer{}, []string{"sync", "--dir", root, "--json"}); code != 0 {
		t.Fatalf("sync: exit = %d, want 0; stdout=%s", code, synced.String())
	}
	if r := decodeReceipt(t, synced.Bytes()); !r.ZeroUnexplainedGaps {
		t.Fatalf("sync receipt = %+v", r)
	}
	if _, err := os.Stat(filepath.Join(root, ".agents", "skills", "alpha", "SKILL.md")); err != nil {
		t.Fatalf("sync did not write the codex adapter: %v", err)
	}

	var text bytes.Buffer
	if code := runProjectAssets(&text, &bytes.Buffer{}, []string{"parity", "--dir", root}); code != 0 {
		t.Fatalf("parity after sync: exit = %d, want 0", code)
	}
	lines := strings.Split(strings.TrimSpace(text.String()), "\n")
	if len(lines) != 5 || !strings.HasPrefix(lines[0], "PROJECT_ASSETS zero_unexplained_gaps=true ") {
		t.Fatalf("text receipt shape:\n%s", text.String())
	}
	for i, h := range []string{"claude", "codex", "fak-native", "opencode"} {
		if !strings.HasPrefix(lines[i+1], h+" canonical=") {
			t.Fatalf("line %d = %q, want %s harness row", i+1, lines[i+1], h)
		}
	}
}
