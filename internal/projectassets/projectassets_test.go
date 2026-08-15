package projectassets

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func write(t *testing.T, root, path, body string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(path))
	if e := os.MkdirAll(filepath.Dir(p), 0755); e != nil {
		t.Fatal(e)
	}
	if e := os.WriteFile(p, []byte(body), 0644); e != nil {
		t.Fatal(e)
	}
}
func fixture(t *testing.T) string {
	r := t.TempDir()
	m := Manifest{Schema: "fak-project-assets/1", Skills: SkillPolicy{CanonicalRoot: ".claude/skills", CodexRoot: ".agents/skills", Include: []string{"SKILL.md"}}, Memories: Policy{CanonicalRoot: ".claude/memory", Include: []string{"*.md"}, Exclude: []Exclusion{{"MEMORY.md", "index"}}, StartupCommand: "fak memory recall --intent <task> --json"}, GoalPrompts: Policy{CanonicalRoot: ".claude/goal-prompts", Include: []string{"template.md"}, Exclude: []Exclusion{{"resolve-[0-9]*.md", "run fuel"}}}, Harnesses: map[string]Harness{"codex": {}, "fak-native": {}}}
	b, _ := json.Marshal(m)
	write(t, r, ManifestPath, string(b))
	write(t, r, ".claude/skills/verify/SKILL.md", "---\nname: verify\ndescription: Verify.\n---\nbody\n")
	write(t, r, ".claude/memory/MEMORY.md", "index\n")
	write(t, r, ".claude/memory/durable.md", "memory\n")
	write(t, r, ".claude/goal-prompts/template.md", "template\n")
	write(t, r, ".claude/goal-prompts/resolve-123.md", "fuel\n")
	return r
}
func TestBuildSyncsWindowsSafeCodexAdapterAndParity(t *testing.T) {
	r := fixture(t)
	before, e := Build(r, false)
	if e != nil {
		t.Fatal(e)
	}
	if before.ZeroUnexplainedGaps {
		t.Fatal("parity passed before missing Codex adapter was generated")
	}
	after, e := Build(r, true)
	if e != nil {
		t.Fatal(e)
	}
	if !after.ZeroUnexplainedGaps {
		t.Fatalf("parity after sync = %#v", after.Harnesses["codex"])
	}
	p := filepath.Join(r, ".agents", "skills", "verify", "SKILL.md")
	info, e := os.Lstat(p)
	if e != nil {
		t.Fatal(e)
	}
	if info.Mode()&os.ModeSymlink != 0 {
		t.Fatal("adapter is a symlink; Windows checkout compatibility requires a regular file")
	}
	b, _ := os.ReadFile(p)
	if string(b) == "body\n" {
		t.Fatal("adapter copied maintained skill body")
	}
}
func TestBuildClassifiesReusableAndEphemeralPrompts(t *testing.T) {
	r := fixture(t)
	if _, e := Build(r, true); e != nil {
		t.Fatal(e)
	}
	receipt, e := Build(r, false)
	if e != nil {
		t.Fatal(e)
	}
	native := receipt.Harnesses["fak-native"]
	foundTemplate, foundExcluded, foundRecall := false, false, false
	for _, p := range native.Imported {
		if p == ".claude/goal-prompts/template.md" {
			foundTemplate = true
		}
		if p == "fak memory recall --intent <task> --json" {
			foundRecall = true
		}
	}
	for _, x := range native.Excluded {
		if x.Path == ".claude/goal-prompts/resolve-123.md" && x.Reason == "run fuel" {
			foundExcluded = true
		}
	}
	if !foundTemplate || !foundExcluded || !foundRecall {
		t.Fatalf("classification/import missing: template=%v excluded=%v recall=%v receipt=%#v", foundTemplate, foundExcluded, foundRecall, native)
	}
}
func TestBuildRejectsUnclassifiedPrompt(t *testing.T) {
	r := fixture(t)
	write(t, r, ".claude/goal-prompts/mystery.txt", "?\n")
	if _, e := Build(r, false); e == nil {
		t.Fatal("unclassified prompt unexpectedly passed")
	}
}
func TestBuildRejectsDuplicateSkillNames(t *testing.T) {
	r := fixture(t)
	write(t, r, ".claude/skills/other/SKILL.md", "---\nname: verify\ndescription: duplicate\n---\n")
	receipt, e := Build(r, true)
	if e != nil {
		t.Fatal(e)
	}
	if receipt.ZeroUnexplainedGaps || len(receipt.Harnesses["codex"].Duplicate) != 1 {
		t.Fatalf("duplicate not reported: %#v", receipt.Harnesses["codex"])
	}
}

func TestTrackedArchiveParity(t *testing.T) {
	root := fixture(t)
	if _, err := Build(root, true); err != nil {
		t.Fatal(err)
	}
	var archive bytes.Buffer
	tw := tar.NewWriter(&archive)
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(root, path)
		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		h := &tar.Header{Name: filepath.ToSlash(rel), Mode: 0644, Size: int64(len(b))}
		if err = tw.WriteHeader(h); err != nil {
			return err
		}
		_, err = tw.Write(b)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	if err = tw.Close(); err != nil {
		t.Fatal(err)
	}
	clean := t.TempDir()
	tr := tar.NewReader(bytes.NewReader(archive.Bytes()))
	for {
		h, e := tr.Next()
		if e == io.EOF {
			break
		}
		if e != nil {
			t.Fatal(e)
		}
		p := filepath.Join(clean, filepath.FromSlash(h.Name))
		if e = os.MkdirAll(filepath.Dir(p), 0755); e != nil {
			t.Fatal(e)
		}
		b, e := io.ReadAll(tr)
		if e != nil {
			t.Fatal(e)
		}
		if e = os.WriteFile(p, b, 0644); e != nil {
			t.Fatal(e)
		}
	}
	receipt, err := Build(clean, false)
	if err != nil {
		t.Fatal(err)
	}
	if !receipt.ZeroUnexplainedGaps {
		t.Fatalf("clean archive parity failed: %#v", receipt.Harnesses["codex"])
	}
}

func TestGeneratedAdapterLoadsCanonicalWorkflowWithoutCopyingIt(t *testing.T) {
	root := t.TempDir()
	write(t, root, ManifestPath, baseManifest())
	canonical := "---\nname: alpha\ndescription: alpha skill\n---\n# Alpha\n\nUse ISSUE_OWNER, LEAF_CHILD, and independent read-back.\n"
	write(t, root, ".claude/skills/alpha/SKILL.md", canonical)
	write(t, root, ".claude/memory/base.md", "memory\n")
	write(t, root, ".claude/goal-prompts/base.md", "prompt\n")
	if _, err := Build(root, true); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, ".agents", "skills", "alpha", "SKILL.md")
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	got := string(body)
	if !strings.Contains(got, "../../../.claude/skills/alpha/SKILL.md") {
		t.Fatalf("adapter does not load canonical skill:\n%s", got)
	}
	if strings.Contains(got, "ISSUE_OWNER") || strings.Contains(got, "LEAF_CHILD") {
		t.Fatalf("adapter copied semantic workflow instead of linking it:\n%s", got)
	}
	if !strings.Contains(got, "must not fork, summarize, or translate") {
		t.Fatalf("adapter lacks portability invariant:\n%s", got)
	}
}

func baseManifest() string {
	return `{
 "schema":"fak-project-assets/1",
 "skills":{"canonical_root":".claude/skills","codex_root":".agents/skills","include":["SKILL.md"],"exclude":[]},
 "memories":{"canonical_root":".claude/memory","include":["*.md"],"exclude":[],"startup_command":"CLAUDE.md"},
 "goal_prompts":{"canonical_root":".claude/goal-prompts","include":["*.md"],"exclude":[]},
 "harnesses":{"claude":{"skills":"native","memories":"native","goal_prompts":"native"},"codex":{"skills":"generated-adapter","memories":"AGENTS.md","goal_prompts":"native"},"fak-native":{"skills":"native-loader","memories":"AGENTS.md","goal_prompts":"native-loader"}}
}`
}
