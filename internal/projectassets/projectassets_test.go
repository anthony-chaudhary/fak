package projectassets

import (
	"archive/tar"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
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
	m := Manifest{Schema: "fak-project-assets/1", Skills: SkillPolicy{CanonicalRoot: ".claude/skills", CodexRoot: ".agents/skills", Include: []string{"SKILL.md"}}, Memories: Policy{CanonicalRoot: ".claude/memory", Include: []string{"*.md"}, Exclude: []Exclusion{{"MEMORY.md", "index"}}, StartupCommand: "fak memory recall --intent <task> --json"}, GoalPrompts: Policy{CanonicalRoot: ".claude/goal-prompts", Include: []string{"template.md"}, Exclude: []Exclusion{{"resolve-[0-9]*.md", "run fuel"}}}, Harnesses: map[string]Harness{"codex": {}, "fak-native": {}, "opencode": {}}}
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
		t.Fatal("parity passed before missing Codex mirror was generated")
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
		t.Fatal("mirror is a symlink; Windows checkout compatibility requires a regular file")
	}
	b, _ := os.ReadFile(p)
	if string(b) == "body\n" {
		t.Fatal("mirror is a bare body copy without frontmatter provenance")
	}
	if !strings.Contains(string(b), "description:") || !strings.Contains(string(b), "Verify.") {
		t.Fatalf("mirror lost canonical description:\n%s", b)
	}
	if !strings.Contains(string(b), "generated-by: fak project-assets sync") || !strings.Contains(string(b), "../../../.claude/skills/verify/SKILL.md") {
		t.Fatalf("mirror lacks provenance metadata:\n%s", b)
	}
	if !strings.Contains(string(b), "\nbody\n") {
		t.Fatalf("mirror lost the canonical body:\n%s", b)
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

func TestGeneratedMirrorCarriesFullCanonicalBody(t *testing.T) {
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
	if !strings.Contains(got, "Use ISSUE_OWNER, LEAF_CHILD, and independent read-back.") {
		t.Fatalf("mirror lost the full canonical body:\n%s", got)
	}
	if !strings.Contains(got, "../../../.claude/skills/alpha/SKILL.md") {
		t.Fatalf("mirror frontmatter lacks the canonical provenance path:\n%s", got)
	}
	if !strings.Contains(got, "generated-by: fak project-assets sync") {
		t.Fatalf("mirror frontmatter lacks generated-by provenance:\n%s", got)
	}
	if strings.Contains(got, "Load and follow") {
		t.Fatalf("mirror still carries the old pointer-adapter marker:\n%s", got)
	}
}

func TestMirrorInjectsProvenanceAndPreservesFrontmatter(t *testing.T) {
	root := t.TempDir()

	// No metadata block: provenance appended at the end of the frontmatter,
	// every other frontmatter line preserved verbatim.
	write(t, root, "plain.md", "---\nname: plain\ndescription: Plain skill.\nuser-invocable: true\nargument-hint: \"[x]\"\n---\n# Plain\n\nBody line one.\n")
	plain, err := mirror(filepath.Join(root, "plain.md"), "plain", "../../../.claude/skills/plain/SKILL.md")
	if err != nil {
		t.Fatal(err)
	}
	wantPlain := "---\nname: plain\ndescription: Plain skill.\nuser-invocable: true\nargument-hint: \"[x]\"\nmetadata:\n  generated-by: fak project-assets sync\n  canonical: ../../../.claude/skills/plain/SKILL.md\n---\n# Plain\n\nBody line one.\n"
	if string(plain) != wantPlain {
		t.Fatalf("no-metadata mirror mismatch:\ngot:\n%s\nwant:\n%s", plain, wantPlain)
	}

	// Existing metadata block: merged without duplicates, other fields intact.
	write(t, root, "meta.md", "---\nname: meta\ndescription: Meta skill.\nmetadata:\n  opencode: claude-only\n---\n# Meta\n\nBody line two.\n")
	withMeta, err := mirror(filepath.Join(root, "meta.md"), "meta", "../../../.claude/skills/meta/SKILL.md")
	if err != nil {
		t.Fatal(err)
	}
	wantMeta := "---\nname: meta\ndescription: Meta skill.\nmetadata:\n  generated-by: fak project-assets sync\n  canonical: ../../../.claude/skills/meta/SKILL.md\n  opencode: claude-only\n---\n# Meta\n\nBody line two.\n"
	if string(withMeta) != wantMeta {
		t.Fatalf("metadata merge mismatch:\ngot:\n%s\nwant:\n%s", withMeta, wantMeta)
	}
	if n := strings.Count(string(withMeta), "generated-by:"); n != 1 {
		t.Fatalf("mirror carries %d generated-by lines, want 1", n)
	}
	if n := strings.Count(string(withMeta), "canonical:"); n != 1 {
		t.Fatalf("mirror carries %d canonical lines, want 1", n)
	}

	// Idempotency: generating twice yields identical bytes, and regenerating a
	// file that already carries provenance stays byte-identical.
	again, err := mirror(filepath.Join(root, "meta.md"), "meta", "../../../.claude/skills/meta/SKILL.md")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(withMeta, again) {
		t.Fatalf("mirror generation is not idempotent:\nfirst:\n%s\nsecond:\n%s", withMeta, again)
	}
	write(t, root, "seeded.md", string(withMeta))
	seeded, err := mirror(filepath.Join(root, "seeded.md"), "meta", "../../../.claude/skills/meta/SKILL.md")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(withMeta, seeded) {
		t.Fatalf("regenerating a provenance-carrying file drifted:\noriginal:\n%s\nregenerated:\n%s", withMeta, seeded)
	}

	// CRLF canonical input normalizes to LF output with the body preserved.
	write(t, root, "crlf.md", "---\r\nname: crlf\r\ndescription: CRLF skill.\r\n---\r\nCRLF body line.\r\n")
	crlf, err := mirror(filepath.Join(root, "crlf.md"), "crlf", "../../../.claude/skills/crlf/SKILL.md")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(crlf), "\r") {
		t.Fatalf("mirror output carries CR bytes:\n%s", crlf)
	}
	if !strings.HasSuffix(string(crlf), "---\nCRLF body line.\n") {
		t.Fatalf("CRLF mirror lost the verbatim body:\n%s", crlf)
	}
}

func TestMirrorKeepsLongDescriptionUntruncated(t *testing.T) {
	root := t.TempDir()
	write(t, root, ManifestPath, baseManifest())
	desc := strings.Repeat("portable trigger ", 18) // 306 runes, above the old 220 truncation floor
	if got := len([]rune(desc)); got <= 220 {
		t.Fatalf("fixture description is %d runes, want > 220", got)
	}
	write(t, root, ".claude/skills/longdesc/SKILL.md", "---\nname: longdesc\ndescription: "+desc+"\n---\n# Long\n")
	write(t, root, ".claude/memory/base.md", "memory\n")
	write(t, root, ".claude/goal-prompts/base.md", "prompt\n")
	if _, err := Build(root, true); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(root, ".agents", "skills", "longdesc", "SKILL.md"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), "description: "+desc+"\n") {
		t.Fatalf("mirror truncated or mangled the 306-rune description:\n%s", b)
	}
	if strings.Contains(string(b), "...") {
		t.Fatalf("mirror carries a truncation ellipsis:\n%s", b)
	}
}

func TestBuildRejectsOversizeDescription(t *testing.T) {
	root := t.TempDir()
	write(t, root, ManifestPath, baseManifest())
	long := strings.Repeat("oversize trigger ", 100) // 1500 runes, above the Agent Skills limit
	if got := len([]rune(long)); got <= 1024 {
		t.Fatalf("fixture description is %d runes, want > 1024", got)
	}
	path := ".claude/skills/oversize/SKILL.md"
	write(t, root, path, "---\nname: oversize\ndescription: "+long+"\n---\n")
	write(t, root, ".claude/memory/base.md", "memory\n")
	write(t, root, ".claude/goal-prompts/base.md", "prompt\n")
	_, err := Build(root, true)
	if err == nil {
		t.Fatal("Build accepted a description above the Agent Skills limit")
	}
	if !strings.Contains(err.Error(), path) {
		t.Fatalf("error does not name the skill path %q: %v", path, err)
	}
}

func TestStaleDetectionFlagsDriftedMirror(t *testing.T) {
	root := t.TempDir()
	write(t, root, ManifestPath, baseManifest())
	write(t, root, ".claude/skills/drift/SKILL.md", "---\nname: drift\ndescription: Drift.\n---\n# Drift\n\nstable body line\n")
	write(t, root, ".claude/memory/base.md", "memory\n")
	write(t, root, ".claude/goal-prompts/base.md", "prompt\n")
	if _, err := Build(root, true); err != nil {
		t.Fatal(err)
	}
	clean, err := Build(root, false)
	if err != nil {
		t.Fatal(err)
	}
	if !clean.ZeroUnexplainedGaps {
		t.Fatalf("expected parity before tamper: %#v", clean.Harnesses["codex"])
	}
	target := filepath.Join(root, ".agents", "skills", "drift", "SKILL.md")
	b, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	i := bytes.Index(b, []byte("stable body line"))
	if i < 0 {
		t.Fatalf("fixture body line missing from mirror:\n%s", b)
	}
	b[i] = 'X' // tamper one byte inside the mirrored body
	if err := os.WriteFile(target, b, 0644); err != nil {
		t.Fatal(err)
	}
	drifted, err := Build(root, false)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, s := range drifted.Harnesses["codex"].Stale {
		if s == ".agents/skills/drift/SKILL.md" {
			found = true
		}
	}
	if !found {
		t.Fatalf("tampered mirror not flagged stale: %v", drifted.Harnesses["codex"].Stale)
	}
}

func baseManifest() string {
	return `{
 "schema":"fak-project-assets/1",
 "skills":{"canonical_root":".claude/skills","codex_root":".agents/skills","include":["SKILL.md"],"exclude":[]},
 "memories":{"canonical_root":".claude/memory","include":["*.md"],"exclude":[],"startup_command":"CLAUDE.md"},
 "goal_prompts":{"canonical_root":".claude/goal-prompts","include":["*.md"],"exclude":[]},
 "harnesses":{"claude":{"skills":"native","memories":"native","goal_prompts":"native"},"codex":{"skills":"generated-adapter","memories":"AGENTS.md","goal_prompts":"native"},"fak-native":{"skills":"native-loader","memories":"AGENTS.md","goal_prompts":"native-loader"},"opencode":{"skills":"generated-adapter","memories":"AGENTS.md","goal_prompts":"native"}}
}`
}

func TestSyncPreservesDeliberateNativeCodexAdapter(t *testing.T) {
	root := t.TempDir()
	write(t, root, ManifestPath, baseManifest())
	write(t, root, ".claude/skills/fleet-wave/SKILL.md", "---\nname: fleet-wave\ndescription: canonical\n---\nClaude workflow.\n")
	write(t, root, ".claude/memory/base.md", "memory\n")
	write(t, root, ".claude/goal-prompts/base.md", "prompt\n")
	native := "---\nname: fleet-wave\ndescription: Codex-native wave\n---\nUse fak dispatch wave --backend=codex.\n"
	write(t, root, ".agents/skills/fleet-wave/SKILL.md", native)
	if receipt, err := Build(root, true); err != nil || !receipt.ZeroUnexplainedGaps {
		t.Fatalf("sync failed: receipt=%+v err=%v", receipt, err)
	}
	body, err := os.ReadFile(filepath.Join(root, ".agents", "skills", "fleet-wave", "SKILL.md"))
	if err != nil {
		t.Fatal(err)
	}
	if string(body) != native {
		t.Fatalf("native adapter was overwritten:\n%s", body)
	}
}

func TestSkillDescriptionNormalizesYAMLScalars(t *testing.T) {
	tests := []struct {
		name   string
		scalar string
		want   string
	}{
		{name: "quoted", scalar: `"Use when a quoted description is canonical."`, want: "Use when a quoted description is canonical."},
		{name: "unquoted", scalar: "Use when a plain description is canonical.", want: "Use when a plain description is canonical."},
		{name: "escaped", scalar: `"Say \"go\" from C:\\work."`, want: `Say "go" from C:\work.`},
		{name: "YAML escapes", scalar: `"slash\/ nul\0 esc\e nbsp\_ nel\N line\L para\P"`, want: "slash/ nul\x00 esc\x1b nbsp\u00a0 nel\u0085 line\u2028 para\u2029"},
		{name: "YAML hex escapes", scalar: `"latin\xE9 pair\xC3\xA9"`, want: "latin\u00e9 pair\u00c3\u00a9"},
		{name: "colon", scalar: "Use when input has a colon: preserve it.", want: "Use when input has a colon: preserve it."},
		{name: "newline", scalar: `"First line\nsecond line."`, want: "First line\nsecond line."},
		{name: "single quoted", scalar: `'It''s portable.'`, want: "It's portable."},
		{name: "quoted whitespace", scalar: `"  preserve semantic spacing  "`, want: "  preserve semantic spacing  "},
		{name: "unterminated quote", scalar: `"Repair this legacy description`, want: "Repair this legacy description"},
		{name: "unterminated escaped quote", scalar: `"Repair the trailing \"`, want: `Repair the trailing "`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			root := t.TempDir()
			path := ".claude/skills/example/SKILL.md"
			write(t, root, path, "---\nname: example\ndescription: "+tt.scalar+"\n---\n")
			got, err := skillDescription(root, path)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("description = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestDecodeYAMLScalarSingleQuotedEscapes(t *testing.T) {
	if got, want := decodeYAMLScalar(`'agent''s trigger'`), "agent's trigger"; got != want {
		t.Fatalf("decodeYAMLScalar = %q, want %q", got, want)
	}
}

func TestVerifyOpenCodeSnapshot(t *testing.T) {
	root := filepath.Clean(filepath.Join("..", ".."))
	if err := VerifyOpenCodeSnapshot(root); err != nil {
		t.Fatalf("VerifyOpenCodeSnapshot on repo root failed: %v", err)
	}

	// Test valid snapshot: false in temp dir
	tmp := t.TempDir()
	write(t, tmp, "opencode.json", `{"snapshot": false}`)
	if err := VerifyOpenCodeSnapshot(tmp); err != nil {
		t.Fatalf("expected valid opencode.json to pass, got: %v", err)
	}

	// Test BOM-prefixed valid snapshot: false in temp dir
	bomTmp := t.TempDir()
	write(t, bomTmp, "opencode.json", "\uFEFF"+`{"snapshot": false}`)
	if err := VerifyOpenCodeSnapshot(bomTmp); err != nil {
		t.Fatalf("expected BOM-prefixed opencode.json to pass, got: %v", err)
	}

	// Test missing file
	emptyTmp := t.TempDir()
	if err := VerifyOpenCodeSnapshot(emptyTmp); err == nil {
		t.Fatal("expected error for missing opencode.json, got nil")
	}

	// Test missing snapshot key
	noSnap := t.TempDir()
	write(t, noSnap, "opencode.json", `{"instructions": ["CONTRIBUTING.md"]}`)
	if err := VerifyOpenCodeSnapshot(noSnap); err == nil {
		t.Fatal("expected error for missing snapshot key, got nil")
	}

	// Test snapshot: true
	trueSnap := t.TempDir()
	write(t, trueSnap, "opencode.json", `{"snapshot": true}`)
	if err := VerifyOpenCodeSnapshot(trueSnap); err == nil {
		t.Fatal("expected error for snapshot: true, got nil")
	}

	// Test non-boolean snapshot
	strSnap := t.TempDir()
	write(t, strSnap, "opencode.json", `{"snapshot": "false"}`)
	if err := VerifyOpenCodeSnapshot(strSnap); err == nil {
		t.Fatal("expected error for non-boolean snapshot, got nil")
	}

	// Test repo root opencode.json directly
	if err := VerifyOpenCodeSnapshot("../.."); err != nil {
		t.Fatalf("repo root opencode.json failed verification: %v", err)
	}
}

func TestEnsureAutoSync(t *testing.T) {
	// 1. autoSync=false when not in parity returns receipt with ZeroUnexplainedGaps=false without syncing
	r := fixture(t)
	receipt, err := Ensure(r, false)
	if err != nil {
		t.Fatalf("unexpected error with autoSync=false: %v", err)
	}
	if receipt.ZeroUnexplainedGaps {
		t.Fatal("expected ZeroUnexplainedGaps=false on desynchronized fixture")
	}
	if len(receipt.Harnesses["codex"].Stale) == 0 {
		t.Fatal("expected stale codex adapters before sync")
	}
	if len(receipt.Harnesses["opencode"].Stale) == 0 {
		t.Fatal("expected stale opencode adapters before sync")
	}
	adapterPath := filepath.Join(r, ".agents", "skills", "verify", "SKILL.md")
	if _, err := os.Stat(adapterPath); !os.IsNotExist(err) {
		t.Fatal("expected adapter to not exist before sync")
	}

	// 2. autoSync=true when not in parity synchronizes adapters and returns receipt with ZeroUnexplainedGaps=true
	receiptSync, err := Ensure(r, true)
	if err != nil {
		t.Fatalf("unexpected error with autoSync=true: %v", err)
	}
	if !receiptSync.ZeroUnexplainedGaps {
		t.Fatalf("expected ZeroUnexplainedGaps=true after autoSync, got %#v", receiptSync.Harnesses["opencode"])
	}
	if len(receiptSync.Harnesses["codex"].Stale) != 0 {
		t.Fatalf("expected 0 stale codex adapters after autoSync, got %d", len(receiptSync.Harnesses["codex"].Stale))
	}
	if len(receiptSync.Harnesses["opencode"].Stale) != 0 {
		t.Fatalf("expected 0 stale opencode adapters after autoSync, got %d", len(receiptSync.Harnesses["opencode"].Stale))
	}
	if _, err := os.Stat(adapterPath); err != nil {
		t.Fatalf("expected adapter to exist after autoSync: %v", err)
	}

	// 3. autoSync=false on already synchronized workspace succeeds and returns parity
	receiptClean, err := Ensure(r, false)
	if err != nil {
		t.Fatalf("unexpected error on synchronized workspace: %v", err)
	}
	if !receiptClean.ZeroUnexplainedGaps {
		t.Fatal("expected ZeroUnexplainedGaps=true on synchronized workspace")
	}
	if len(receiptClean.Harnesses["codex"].Stale) != 0 || len(receiptClean.Harnesses["opencode"].Stale) != 0 {
		t.Fatal("expected no stale adapters on synchronized workspace")
	}

	// 4. Ensure returns error when manifest cannot be loaded
	badDir := t.TempDir()
	if _, err := Ensure(badDir, false); err == nil {
		t.Fatal("expected error for missing manifest with autoSync=false")
	}
	if _, err := Ensure(badDir, true); err == nil {
		t.Fatal("expected error for missing manifest with autoSync=true")
	}
}

func TestEnsureSync(t *testing.T) {
	r := fixture(t)

	// 1. Initial run on fixture without adapters: synced should be true, err nil, ZeroUnexplainedGaps true after sync
	receiptSync, synced, err := EnsureSync(r)
	if err != nil {
		t.Fatalf("unexpected error on EnsureSync: %v", err)
	}
	if !synced {
		t.Fatal("expected synced=true when adapters are missing")
	}
	if !receiptSync.ZeroUnexplainedGaps {
		t.Fatalf("expected ZeroUnexplainedGaps=true after EnsureSync, got %#v", receiptSync.Harnesses["opencode"])
	}

	// 2. Second run when already in parity: synced should be false, err nil
	receiptClean, synced2, err2 := EnsureSync(r)
	if err2 != nil {
		t.Fatalf("unexpected error on second EnsureSync: %v", err2)
	}
	if synced2 {
		t.Fatal("expected synced=false when workspace is already in parity")
	}
	if !receiptClean.ZeroUnexplainedGaps {
		t.Fatal("expected ZeroUnexplainedGaps=true on clean workspace")
	}

	// 3. Error case when manifest is missing
	badDir := t.TempDir()
	if _, synced3, err3 := EnsureSync(badDir); err3 == nil || !synced3 {
		t.Fatalf("expected error and synced=true for badDir, got err=%v synced=%v", err3, synced3)
	}
}

func TestOpenCodeHarnessReceiptAndZeroUnexplainedGaps(t *testing.T) {
	r := fixture(t)
	receipt, err := Ensure(r, true)
	if err != nil {
		t.Fatalf("Ensure failed: %v", err)
	}
	if !receipt.ZeroUnexplainedGaps {
		t.Fatalf("expected ZeroUnexplainedGaps=true, got false")
	}

	opencode, ok := receipt.Harnesses["opencode"]
	if !ok {
		t.Fatal("expected opencode harness receipt to be present")
	}
	codex, ok := receipt.Harnesses["codex"]
	if !ok {
		t.Fatal("expected codex harness receipt to be present")
	}

	if len(opencode.Canonical) != len(codex.Canonical) {
		t.Fatalf("canonical mismatch: opencode=%d, codex=%d", len(opencode.Canonical), len(codex.Canonical))
	}
	for i := range opencode.Canonical {
		if opencode.Canonical[i] != codex.Canonical[i] {
			t.Fatalf("canonical item %d mismatch: %q vs %q", i, opencode.Canonical[i], codex.Canonical[i])
		}
	}
	if len(opencode.Imported) != len(codex.Imported) {
		t.Fatalf("imported mismatch: opencode=%d, codex=%d", len(opencode.Imported), len(codex.Imported))
	}
	for i := range opencode.Imported {
		if opencode.Imported[i] != codex.Imported[i] {
			t.Fatalf("imported item %d mismatch: %q vs %q", i, opencode.Imported[i], codex.Imported[i])
		}
	}
	if len(opencode.Excluded) != len(codex.Excluded) {
		t.Fatalf("excluded mismatch: opencode=%d, codex=%d", len(opencode.Excluded), len(codex.Excluded))
	}
	if len(opencode.Duplicate) != 0 {
		t.Fatalf("expected 0 duplicates, got %d", len(opencode.Duplicate))
	}
	if len(opencode.Stale) != 0 {
		t.Fatalf("expected 0 stale, got %d", len(opencode.Stale))
	}

	// Verify missing opencode in manifest causes ZeroUnexplainedGaps to be false
	noOpenCodeDir := t.TempDir()
	noOpenCodeManifest := Manifest{
		Schema:      "fak-project-assets/1",
		Skills:      SkillPolicy{CanonicalRoot: ".claude/skills", CodexRoot: ".agents/skills", Include: []string{"SKILL.md"}},
		Memories:    Policy{CanonicalRoot: ".claude/memory", Include: []string{"*.md"}, StartupCommand: "cmd"},
		GoalPrompts: Policy{CanonicalRoot: ".claude/goal-prompts", Include: []string{"*.md"}},
		Harnesses:   map[string]Harness{"codex": {}, "fak-native": {}},
	}
	b, _ := json.Marshal(noOpenCodeManifest)
	write(t, noOpenCodeDir, ManifestPath, string(b))
	write(t, noOpenCodeDir, ".claude/skills/demo/SKILL.md", "---\nname: demo\ndescription: demo\n---\n")
	write(t, noOpenCodeDir, ".claude/memory/m.md", "mem\n")
	write(t, noOpenCodeDir, ".claude/goal-prompts/g.md", "goal\n")
	noOpenCodeReceipt, err := Build(noOpenCodeDir, true)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	if noOpenCodeReceipt.ZeroUnexplainedGaps {
		t.Fatal("expected ZeroUnexplainedGaps=false when opencode is missing from manifest")
	}

	// Verify repo root project-assets manifest includes opencode. EnsureSync heals
	// generated mirrors in place when the generator contract moves (same self-repair
	// pattern as TestOpenCodePluginAssetByteParity); hand-written native skills are
	// never touched.
	repoRoot := filepath.Clean(filepath.Join("..", ".."))
	repoReceipt, _, err := EnsureSync(repoRoot)
	if err != nil {
		t.Fatalf("EnsureSync repo root failed: %v", err)
	}
	if !repoReceipt.ZeroUnexplainedGaps {
		t.Fatalf("expected ZeroUnexplainedGaps=true on repoRoot, got false: %#v", repoReceipt.Harnesses)
	}
	if _, ok := repoReceipt.Harnesses["opencode"]; !ok {
		t.Fatal("repo root receipt missing opencode harness")
	}
}

func TestOpenCodePluginAssetByteParity(t *testing.T) {
	repoRoot := filepath.Clean(filepath.Join("..", ".."))
	canonicalPath := filepath.Join(repoRoot, filepath.FromSlash(OpenCodePluginPath))
	b, err := os.ReadFile(canonicalPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) || os.IsNotExist(err) ||
			strings.Contains(err.Error(), "no such file or directory") ||
			strings.Contains(err.Error(), "The system cannot find the path specified") ||
			strings.Contains(err.Error(), "The system cannot find the file specified") {
			t.Skipf("skipping: %s does not exist in worktree", canonicalPath)
		}
		t.Fatalf("failed to read canonical plugin at %s: %v", canonicalPath, err)
	}
	canonical := strings.ReplaceAll(string(b), "\r\n", "\n")
	embedded := strings.ReplaceAll(DefaultOpenCodePlugin, "\r\n", "\n")
	if canonical != embedded {
		if err := SyncOpenCodePlugin(repoRoot); err != nil {
			t.Fatalf("failed to sync canonical plugin at %s: %v", canonicalPath, err)
		}
		b, err = os.ReadFile(canonicalPath)
		if err != nil {
			t.Fatalf("failed to read canonical plugin after sync: %v", err)
		}
		canonical = strings.ReplaceAll(string(b), "\r\n", "\n")
		if canonical != embedded {
			t.Fatalf("embedded DefaultOpenCodePlugin does not match disk asset %s", canonicalPath)
		}
	}
}

func TestVerifyOpenCodePlugin(t *testing.T) {
	repoRoot := filepath.Clean(filepath.Join("..", ".."))
	if err := VerifyOpenCodePlugin(repoRoot); err != nil {
		if errors.Is(err, os.ErrNotExist) || os.IsNotExist(err) ||
			strings.Contains(err.Error(), "no such file or directory") ||
			strings.Contains(err.Error(), "The system cannot find the path specified") ||
			strings.Contains(err.Error(), "The system cannot find the file specified") {
			t.Skipf("skipping: plugin does not exist in repoRoot in isolated worktree: %v", err)
		} else {
			t.Fatalf("VerifyOpenCodePlugin failed on repo root: %v", err)
		}
	}

	// Test missing plugin in empty temp directory
	tmp := t.TempDir()
	if err := VerifyOpenCodePlugin(tmp); err == nil {
		t.Fatal("expected error for missing plugin, got nil")
	}

	// Test empty plugin file
	write(t, tmp, OpenCodePluginPath, "   \n\t  ")
	if err := VerifyOpenCodePlugin(tmp); err == nil || !strings.Contains(err.Error(), "empty") {
		t.Fatalf("expected empty plugin error, got: %v", err)
	}

	// Test missing execution hook
	write(t, tmp, OpenCodePluginPath, "export default function() {}")
	if err := VerifyOpenCodePlugin(tmp); err == nil || !strings.Contains(err.Error(), "tool.execute.before") {
		t.Fatalf("expected missing tool.execute.before error, got: %v", err)
	}

	// Test missing mutation interception
	write(t, tmp, OpenCodePluginPath, "export default function() { return { 'tool.execute.before': async () => {} }; }")
	if err := VerifyOpenCodePlugin(tmp); err == nil || !strings.Contains(err.Error(), "write") {
		t.Fatalf("expected missing mutation tool interception error, got: %v", err)
	}

	// Test missing FAK lease checks
	write(t, tmp, OpenCodePluginPath, "export default function() { return { 'tool.execute.before': async () => { const m = ['write', 'edit', 'apply_patch']; } }; }")
	if err := VerifyOpenCodePlugin(tmp); err == nil || !strings.Contains(err.Error(), "leaseref") {
		t.Fatalf("expected missing leaseref error, got: %v", err)
	}

	// Test missing DOS arbitration checks
	write(t, tmp, OpenCodePluginPath, "export default function() { return { 'tool.execute.before': async () => { const m = ['write', 'edit', 'apply_patch']; fak('leaseref'); fak('loop'); } }; }")
	if err := VerifyOpenCodePlugin(tmp); err == nil || !strings.Contains(err.Error(), "live_leases") {
		t.Fatalf("expected missing live_leases error, got: %v", err)
	}

	// Test missing budgeted-preflight markers. The fixture must carry the COMPLIANT
	// live_leases probe (`live_leases(cfg, expire_dead=True)`), otherwise verification
	// stops at the earlier live_leases check and this fixture never reaches the budget
	// assertion it is testing.
	write(t, tmp, OpenCodePluginPath, "export default function() { return { 'tool.execute.before': async () => { const m = ['write', 'edit', 'apply_patch']; fak('leaseref'); fak('loop'); live_leases(cfg, expire_dead=True); arbitrate; } }; }")
	if err := VerifyOpenCodePlugin(tmp); err == nil || !strings.Contains(err.Error(), "FAK_WRITE_GUARD_BUDGET_MS") {
		t.Fatalf("expected missing FAK_WRITE_GUARD_BUDGET_MS error, got: %v", err)
	}

	// Test missing Windows platform binary extension check
	write(t, tmp, OpenCodePluginPath, "export default function() { return { 'tool.execute.before': async () => { const m = ['write', 'edit', 'apply_patch']; fak('leaseref'); fak('loop'); live_leases(cfg, expire_dead=True); arbitrate; FAK_WRITE_GUARD_BUDGET_MS; FAK_WRITE_GUARD_STRICT; fak.queueus.pending.v1; } }; }")
	if err := VerifyOpenCodePlugin(tmp); err == nil || !strings.Contains(err.Error(), "win32") {
		t.Fatalf("expected missing win32 error, got: %v", err)
	}
}

func TestDefaultOpenCodePluginDOSProbeHasNoStrictKwarg(t *testing.T) {
	if strings.Contains(DefaultOpenCodePlugin, "strict=") {
		t.Fatal("DefaultOpenCodePlugin must not emit strict= (incompatible with DOS >= " + DOSMinVersion + ")")
	}
	if !strings.Contains(DefaultOpenCodePlugin, "live_leases(cfg, expire_dead=True)") {
		t.Fatal("DefaultOpenCodePlugin must emit the strict-free live_leases(cfg, expire_dead=True) probe")
	}
}

func TestVerifyOpenCodePluginRejectsStaleStrictProbe(t *testing.T) {
	stale := strings.Replace(DefaultOpenCodePlugin, "live_leases(cfg, expire_dead=True)", "live_leases(cfg, strict=True, expire_dead=True)", 1)
	if !strings.Contains(stale, "strict=") {
		t.Fatal("fixture construction failed: stale probe lacks strict=")
	}
	tmp := t.TempDir()
	write(t, tmp, OpenCodePluginPath, stale)
	err := VerifyOpenCodePlugin(tmp)
	if err == nil {
		t.Fatal("expected VerifyOpenCodePlugin to reject a stale strict=True probe, got nil")
	}
	if !strings.Contains(err.Error(), "strict") {
		t.Fatalf("expected rejection error to mention strict, got: %v", err)
	}
}

func TestSyncOpenCodePluginRoundTripOmitsStrict(t *testing.T) {
	tmp := t.TempDir()
	if err := SyncOpenCodePlugin(tmp); err != nil {
		t.Fatalf("SyncOpenCodePlugin failed: %v", err)
	}
	if err := VerifyOpenCodePlugin(tmp); err != nil {
		t.Fatalf("VerifyOpenCodePlugin failed after SyncOpenCodePlugin: %v", err)
	}
	b, err := os.ReadFile(filepath.Join(tmp, filepath.FromSlash(OpenCodePluginPath)))
	if err != nil {
		t.Fatalf("read synced plugin: %v", err)
	}
	if strings.Contains(string(b), "strict=") {
		t.Fatal("synced plugin must not contain strict=")
	}
}

func TestSyncAndEnsureOpenCodePlugin(t *testing.T) {
	tmp := t.TempDir()

	// Initially missing
	if err := EnsureOpenCodePlugin(tmp, false); err == nil {
		t.Fatal("expected EnsureOpenCodePlugin with autoSync=false to fail on empty dir")
	}

	// autoSync=true writes and verifies the plugin
	if err := EnsureOpenCodePlugin(tmp, true); err != nil {
		t.Fatalf("EnsureOpenCodePlugin with autoSync=true failed: %v", err)
	}
	if err := VerifyOpenCodePlugin(tmp); err != nil {
		t.Fatalf("VerifyOpenCodePlugin failed after autoSync: %v", err)
	}

	// Explicit SyncOpenCodePlugin also succeeds
	tmp2 := t.TempDir()
	if err := SyncOpenCodePlugin(tmp2); err != nil {
		t.Fatalf("SyncOpenCodePlugin failed: %v", err)
	}
	if err := VerifyOpenCodePlugin(tmp2); err != nil {
		t.Fatalf("VerifyOpenCodePlugin failed after SyncOpenCodePlugin: %v", err)
	}

	// Corrupted plugin repaired by autoSync
	write(t, tmp, OpenCodePluginPath, "corrupted plugin missing all checks")
	if err := EnsureOpenCodePlugin(tmp, false); err == nil {
		t.Fatal("expected EnsureOpenCodePlugin with autoSync=false to fail on corrupted plugin")
	}
	if err := EnsureOpenCodePlugin(tmp, true); err != nil {
		t.Fatalf("EnsureOpenCodePlugin with autoSync=true failed to repair corrupted plugin: %v", err)
	}
	if err := VerifyOpenCodePlugin(tmp); err != nil {
		t.Fatalf("VerifyOpenCodePlugin failed after repair: %v", err)
	}
}

func TestEnsureVerifiesOpenCodePluginWithoutBreaking(t *testing.T) {
	// 1. Repo root sync works cleanly without breaking; EnsureSync heals generated
	// mirrors in place when the generator contract moves (never touches hand-written
	// native skills).
	repoRoot := filepath.Clean(filepath.Join("..", ".."))
	repoReceipt, _, err := EnsureSync(repoRoot)
	if err != nil {
		t.Fatalf("EnsureSync(repoRoot) failed: %v", err)
	}
	if !repoReceipt.ZeroUnexplainedGaps {
		t.Fatalf("expected ZeroUnexplainedGaps=true on repoRoot, got false: %#v", repoReceipt.Harnesses)
	}

	// 2. Fixture without .opencode directory: Ensure works without breaking
	r := fixture(t)
	receiptNoOpenCode, err := Ensure(r, true)
	if err != nil {
		t.Fatalf("Ensure on fixture without .opencode failed: %v", err)
	}
	if !receiptNoOpenCode.ZeroUnexplainedGaps {
		t.Fatal("expected fixture to have ZeroUnexplainedGaps=true after Ensure")
	}
	// Verify that .opencode was not spuriously created when absent
	if _, err := os.Stat(filepath.Join(r, ".opencode")); !os.IsNotExist(err) {
		t.Fatal("expected .opencode to not be created on workspace without .opencode directory")
	}

	// 3. Workspace with .opencode directory: Ensure and EnsureSync synchronize and verify the plugin
	rWithOpenCode := fixture(t)
	if err := os.MkdirAll(filepath.Join(rWithOpenCode, ".opencode"), 0755); err != nil {
		t.Fatal(err)
	}

	// Ensure(autoSync=false) on workspace with .opencode directory but missing plugin must fail
	if _, err := Ensure(rWithOpenCode, false); err == nil {
		t.Fatal("expected Ensure(autoSync=false) to fail on missing OpenCode plugin in workspace with .opencode")
	}

	// EnsureSync on workspace with .opencode directory and missing plugin must sync the plugin and return synced=true
	receiptSyncOpenCode, syncedOpenCode, err := EnsureSync(rWithOpenCode)
	if err != nil {
		t.Fatalf("EnsureSync failed on workspace with .opencode: %v", err)
	}
	if !syncedOpenCode {
		t.Fatal("expected EnsureSync to return synced=true when OpenCode plugin was missing")
	}
	if !receiptSyncOpenCode.ZeroUnexplainedGaps {
		t.Fatal("expected workspace with .opencode to have ZeroUnexplainedGaps=true after EnsureSync")
	}
	if err := VerifyOpenCodePlugin(rWithOpenCode); err != nil {
		t.Fatalf("VerifyOpenCodePlugin failed on workspace after EnsureSync: %v", err)
	}

	// Second run of EnsureSync on clean workspace returns synced=false
	_, syncedAgain, err := EnsureSync(rWithOpenCode)
	if err != nil {
		t.Fatalf("EnsureSync failed on clean workspace: %v", err)
	}
	if syncedAgain {
		t.Fatal("expected EnsureSync to return synced=false when workspace is already in parity")
	}

	receiptWithOpenCode, err := Ensure(rWithOpenCode, true)
	if err != nil {
		t.Fatalf("Ensure on workspace with .opencode failed: %v", err)
	}
	if !receiptWithOpenCode.ZeroUnexplainedGaps {
		t.Fatal("expected workspace with .opencode to have ZeroUnexplainedGaps=true")
	}
	if err := VerifyOpenCodePlugin(rWithOpenCode); err != nil {
		t.Fatalf("VerifyOpenCodePlugin failed on workspace after Ensure(autoSync=true): %v", err)
	}

	// Ensure(autoSync=false) on workspace with valid plugin succeeds
	cleanReceipt, err := Ensure(rWithOpenCode, false)
	if err != nil {
		t.Fatalf("Ensure(autoSync=false) on clean workspace failed: %v", err)
	}
	if !cleanReceipt.ZeroUnexplainedGaps {
		t.Fatal("expected cleanReceipt.ZeroUnexplainedGaps=true")
	}

	// If the plugin is corrupted, Ensure(autoSync=false) catches the defect
	write(t, rWithOpenCode, OpenCodePluginPath, "corrupted plugin")
	if _, err := Ensure(rWithOpenCode, false); err == nil {
		t.Fatal("expected Ensure(autoSync=false) to fail on corrupted plugin in workspace")
	}

	// EnsureSync heals the corrupted plugin and returns synced=true
	healedSyncReceipt, syncedHealed, err := EnsureSync(rWithOpenCode)
	if err != nil {
		t.Fatalf("EnsureSync failed to heal corrupted plugin: %v", err)
	}
	if !syncedHealed {
		t.Fatal("expected EnsureSync to return synced=true when healing corrupted plugin")
	}
	if !healedSyncReceipt.ZeroUnexplainedGaps {
		t.Fatal("expected healedSyncReceipt.ZeroUnexplainedGaps=true")
	}
	if err := VerifyOpenCodePlugin(rWithOpenCode); err != nil {
		t.Fatalf("VerifyOpenCodePlugin failed after EnsureSync heal: %v", err)
	}

	// Ensure(autoSync=true) succeeds when already in parity
	healedReceipt, err := Ensure(rWithOpenCode, true)
	if err != nil {
		t.Fatalf("Ensure(autoSync=true) failed on healed workspace: %v", err)
	}
	if !healedReceipt.ZeroUnexplainedGaps {
		t.Fatal("expected healedReceipt.ZeroUnexplainedGaps=true")
	}
	if err := VerifyOpenCodePlugin(rWithOpenCode); err != nil {
		t.Fatalf("VerifyOpenCodePlugin failed after heal: %v", err)
	}
}

func TestGeneratedMirrorPreservesHarnessMetadataFences(t *testing.T) {
	root := t.TempDir()
	write(t, root, ManifestPath, baseManifest())
	canonical := "---\nname: commit-clean\ndescription: Commit finished work cleanly.\nmetadata:\n  opencode: claude-only\n---\n# Workflow body\n"
	write(t, root, ".claude/skills/commit-clean/SKILL.md", canonical)
	write(t, root, ".claude/memory/base.md", "memory\n")
	write(t, root, ".claude/goal-prompts/base.md", "prompt\n")

	// 1. Build(root, true) generates mirror preserving metadata fence
	receipt, err := Build(root, true)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	if !receipt.ZeroUnexplainedGaps {
		t.Fatalf("expected ZeroUnexplainedGaps=true, got %#v", receipt.Harnesses["opencode"])
	}

	mirrorPath := filepath.Join(root, ".agents", "skills", "commit-clean", "SKILL.md")
	body, err := os.ReadFile(mirrorPath)
	if err != nil {
		t.Fatalf("read mirror: %v", err)
	}
	content := string(body)
	if !strings.Contains(content, "opencode: claude-only") {
		t.Fatalf("mirror frontmatter missing 'opencode: claude-only':\n%s", content)
	}
	if !strings.Contains(content, "generated-by: fak project-assets sync") {
		t.Fatalf("mirror frontmatter missing 'generated-by':\n%s", content)
	}
	if !strings.Contains(content, "# Workflow body\n") {
		t.Fatalf("mirror lost the canonical body:\n%s", content)
	}

	// 2. Parity check with Build(root, false) succeeds
	checkReceipt, err := Build(root, false)
	if err != nil {
		t.Fatalf("Build(false) failed: %v", err)
	}
	if !checkReceipt.ZeroUnexplainedGaps {
		t.Fatalf("expected parity to hold after sync: %#v", checkReceipt.Harnesses["opencode"])
	}

	// 3. Modifying canonical metadata flags mirror as stale
	modifiedCanonical := "---\nname: commit-clean\ndescription: Commit finished work cleanly.\nmetadata:\n  opencode: agent-permission\n---\n# Workflow body\n"
	write(t, root, ".claude/skills/commit-clean/SKILL.md", modifiedCanonical)
	staleReceipt, err := Build(root, false)
	if err != nil {
		t.Fatalf("Build(false) failed: %v", err)
	}
	if staleReceipt.ZeroUnexplainedGaps {
		t.Fatal("expected ZeroUnexplainedGaps=false when canonical metadata changed")
	}
	foundStale := false
	for _, s := range staleReceipt.Harnesses["opencode"].Stale {
		if s == ".agents/skills/commit-clean/SKILL.md" {
			foundStale = true
			break
		}
	}
	if !foundStale {
		t.Fatalf("expected .agents/skills/commit-clean/SKILL.md to be stale, got: %v", staleReceipt.Harnesses["opencode"].Stale)
	}
}

func TestOpenCodePluginWindowsExtensionAndPreflightBudget(t *testing.T) {
	// Verify DefaultOpenCodePlugin contains Windows command extension resolution and the
	// budgeted-preflight contract (budget env, strict escape hatch, QUEUEUS ledger schema).
	if !strings.Contains(DefaultOpenCodePlugin, `process.platform === "win32"`) &&
		!strings.Contains(DefaultOpenCodePlugin, `process.platform === 'win32'`) {
		t.Fatal("DefaultOpenCodePlugin missing win32 platform check")
	}
	if !strings.Contains(DefaultOpenCodePlugin, `.endsWith(".exe")`) &&
		!strings.Contains(DefaultOpenCodePlugin, `.endsWith('.exe')`) {
		t.Fatal("DefaultOpenCodePlugin missing .endsWith('.exe') check")
	}
	for _, marker := range []string{"FAK_WRITE_GUARD_BUDGET_MS", "FAK_WRITE_GUARD_STRICT", "fak.queueus.pending.v1"} {
		if !strings.Contains(DefaultOpenCodePlugin, marker) {
			t.Fatalf("DefaultOpenCodePlugin missing budgeted-guard marker %q", marker)
		}
	}
	if strings.Contains(DefaultOpenCodePlugin, "timeout: 30000") {
		t.Fatal("DefaultOpenCodePlugin must not contain obsolete 30000ms timeout")
	}
	if strings.Contains(DefaultOpenCodePlugin, `if (tool === "edit" || tool === "write")`) ||
		strings.Contains(DefaultOpenCodePlugin, `if (tool === 'edit' || tool === 'write')`) {
		t.Fatal("DefaultOpenCodePlugin tool.execute.after omits apply_patch")
	}
	if !strings.Contains(DefaultOpenCodePlugin, "apply_patch") {
		t.Fatal("DefaultOpenCodePlugin tool.execute.after missing apply_patch handling")
	}
	if !strings.Contains(DefaultOpenCodePlugin, `tool === "apply_patch"`) &&
		!strings.Contains(DefaultOpenCodePlugin, `tool === 'apply_patch'`) &&
		!strings.Contains(DefaultOpenCodePlugin, "mutations.has(tool)") {
		t.Fatal("DefaultOpenCodePlugin tool.execute.after missing apply_patch interceptor")
	}
	if strings.Contains(DefaultOpenCodePlugin, "console.log") {
		t.Fatal("DefaultOpenCodePlugin must contain zero console.log statements")
	}

	// Also verify the disk asset .opencode/plugins/dos-proof-guard.js
	repoRoot := filepath.Clean(filepath.Join("..", ".."))
	canonicalPath := filepath.Join(repoRoot, filepath.FromSlash(OpenCodePluginPath))
	b, err := os.ReadFile(canonicalPath)
	if err != nil {
		if os.IsNotExist(err) {
			t.Skipf("skipping: %s does not exist in worktree", canonicalPath)
		}
		t.Fatalf("failed to read %s: %v", canonicalPath, err)
	}
	content := string(b)
	if !strings.Contains(content, `process.platform === "win32"`) &&
		!strings.Contains(content, `process.platform === 'win32'`) {
		t.Fatal("disk plugin missing win32 platform check")
	}
	if !strings.Contains(content, `.endsWith(".exe")`) &&
		!strings.Contains(content, `.endsWith('.exe')`) {
		t.Fatal("disk plugin missing .endsWith('.exe') check")
	}
	for _, marker := range []string{"FAK_WRITE_GUARD_BUDGET_MS", "FAK_WRITE_GUARD_STRICT", "fak.queueus.pending.v1"} {
		if !strings.Contains(content, marker) {
			t.Fatalf("disk plugin missing budgeted-guard marker %q", marker)
		}
	}
	if strings.Contains(content, "timeout: 30000") {
		t.Fatal("disk plugin must not contain obsolete 30000ms timeout")
	}
	if strings.Contains(content, `if (tool === "edit" || tool === "write")`) ||
		strings.Contains(content, `if (tool === 'edit' || tool === 'write')`) {
		t.Fatal("disk plugin tool.execute.after omits apply_patch")
	}
	if !strings.Contains(content, "apply_patch") {
		t.Fatal("disk plugin tool.execute.after missing apply_patch handling")
	}
	if !strings.Contains(content, `tool === "apply_patch"`) &&
		!strings.Contains(content, `tool === 'apply_patch'`) &&
		!strings.Contains(content, "mutations.has(tool)") {
		t.Fatal("disk plugin tool.execute.after missing apply_patch interceptor")
	}
	if strings.Contains(content, "console.log") {
		t.Fatal("disk plugin must contain zero console.log statements")
	}
}

// seedWitnessTargets creates the named workspace-relative files under root so
// completed mutations in the JS witnesses are never phantom writes (the guard
// notes PHANTOM_WRITE instead of the proof reminder when the target is absent
// on disk).
func seedWitnessTargets(t *testing.T, root string, rels ...string) {
	t.Helper()
	for _, rel := range rels {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("seed\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func TestOpenCodeMutationPathsAndProofReminders(t *testing.T) {
	if _, err := exec.LookPath("node"); err != nil {
		t.Skip("node is required for OpenCode plugin witness")
	}
	root := t.TempDir()
	plugin := filepath.Join(root, "proof.mjs")
	if err := os.WriteFile(plugin, []byte(DefaultOpenCodePlugin), 0644); err != nil {
		t.Fatal(err)
	}
	seedWitnessTargets(t, root, "internal/compute/example.go", "docs/test.md")
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "node", "--input-type=module", "-", plugin, root)
	cmd.Stdin = strings.NewReader(`
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';

const hooks = await (await import(pathToFileURL(process.argv[2]).href)).default.server({directory:process.argv[3]});

const callAfter = async (tool, args, output) => {
  await hooks['tool.execute.after']({tool, args, sessionID:'witness', callID:'test'}, output);
  return output;
};

// 1. Alternative path keys trigger proof reminder and halo hardware validation when touching hardware paths
const rFilePath = await callAfter('write', {file_path: 'internal/compute/example.go'}, {result: 'initial-result'});
assert.ok(typeof rFilePath.result === 'string' && rFilePath.result.includes('[dos-proof-guard]'));
assert.ok(rFilePath.result.includes('fak-dev amd-strix-probe'));

const rPath = await callAfter('edit', {path: 'internal/compute/example.go'}, {text: 'initial-text'});
assert.ok(typeof rPath.text === 'string' && rPath.text.includes('[dos-proof-guard]'));
assert.ok(rPath.text.includes('fak-dev amd-strix-probe'));

const rTarget = await callAfter('write', {target: 'docs/test.md'}, {result: 'ok'});
assert.ok(rTarget.result.includes('[dos-proof-guard]'));
assert.ok(!rTarget.result.includes('fak-dev amd-strix-probe'));

// 2. Standard unified git diff headers (diff --git, ---, +++)
const diffGitPatch = 'diff --git a/internal/compute/example.go b/internal/compute/example.go\n' +
  '--- a/internal/compute/example.go\n' +
  '+++ b/internal/compute/example.go\n' +
  '@@ -1 +1 @@\n-old\n+new';
const rDiffGit = await callAfter('apply_patch', {patchText: diffGitPatch}, {result: 'applied'});
assert.ok(rDiffGit.result.includes('fak-dev amd-strix-probe'));

// 3. Standard unified diff without diff --git
const unifiedPatch = '--- a/internal/compute/example.go\n+++ b/internal/compute/example.go\n@@ -1 +1 @@\n-old\n+new';
const rUnified = await callAfter('apply_patch', {patchText: unifiedPatch}, {text: 'applied'});
assert.ok(rUnified.text.includes('fak-dev amd-strix-probe'));

// 4. New file unified diff (--- /dev/null, +++ b/...)
const newFilePatch = '--- /dev/null\n+++ b/internal/compute/example.go\n@@ -0,0 +1 @@\n+new';
const rNew = await callAfter('apply_patch', {patchText: newFilePatch}, {output: 'applied'});
assert.ok(rNew.output.includes('fak-dev amd-strix-probe'));

// 5. Deleted file unified diff (--- a/..., +++ /dev/null)
const delFilePatch = '--- a/internal/compute/example.go\n+++ /dev/null\n@@ -1 +0,0 @@\n-del';
const rDel = await callAfter('apply_patch', {patchText: delFilePatch}, {output: 'applied'});
assert.ok(rDel.output.includes('fak-dev amd-strix-probe'));

// 6. Proof reminder appended to output.result as array
const rArray = await callAfter('write', {filePath: 'docs/test.md'}, {result: [{type:'text', text:'existing'}]});
assert.equal(rArray.result.length, 2);
assert.ok(rArray.result[1].text.includes('[dos-proof-guard]'));

// 7. Pre-mutation lease admission rejects empty/null mutation paths
await assert.rejects(
  hooks['tool.execute.before']({tool: 'write', sessionID: 'sess'}, {args: {filePath: '', file_path: null, path: undefined, target: '   '}}),
  /Explicit mutation paths are required for lease admission/
);
await assert.rejects(
  hooks['tool.execute.before']({tool: 'write', sessionID: 'sess'}, {args: {}}),
  /Explicit mutation paths are required for lease admission/
);
await assert.rejects(
  hooks['tool.execute.before']({tool: 'apply_patch', sessionID: 'sess'}, {args: {patchText: ''}}),
  /Explicit mutation paths are required for lease admission/
);

console.log(JSON.stringify({ok: true}));
`)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("JS hook execution failed: %v\n%s", err, out)
	}
	var receipt struct {
		OK bool `json:"ok"`
	}
	if err := json.Unmarshal(out, &receipt); err != nil || !receipt.OK {
		t.Fatalf("unexpected output: %s", out)
	}
}

func TestOpenCodePluginAlternativePathKeysAndUnifiedDiffs(t *testing.T) {
	// Issue #12071: verify DefaultOpenCodePlugin supports alternative path keys (file_path, path, target)
	// and standard unified diff formats in mutationPaths.
	for _, key := range []string{"args?.filePath", "args?.file_path", "args?.path", "args?.target"} {
		if !strings.Contains(DefaultOpenCodePlugin, key) {
			t.Fatalf("DefaultOpenCodePlugin missing alternative path key %q (#12071)", key)
		}
	}
	for _, pattern := range []string{"diff --git", "^---", "^\\+\\+\\+"} {
		if !strings.Contains(DefaultOpenCodePlugin, pattern) {
			t.Fatalf("DefaultOpenCodePlugin missing unified diff header pattern %q (#12071)", pattern)
		}
	}
	for _, prop := range []string{"result.output", "result.content"} {
		if !strings.Contains(DefaultOpenCodePlugin, prop) {
			t.Fatalf("DefaultOpenCodePlugin missing proof reminder target property %q (#12071)", prop)
		}
	}
}

func TestOpenCodeGrepExecutableResolutionAndFallback(t *testing.T) {
	root := t.TempDir()
	if err := SyncOpenCodePlugin(root); err != nil {
		t.Fatalf("SyncOpenCodePlugin failed: %v", err)
	}
	if err := VerifyOpenCodePlugin(root); err != nil {
		t.Fatalf("VerifyOpenCodePlugin failed: %v", err)
	}

	requiredPatterns := []struct {
		pattern string
		desc    string
	}{
		{"findRipgrep", "executable path resolution for ripgrep"},
		{"Git", "Git standard directory resolution"},
		{"usr", "Git usr directory resolution"},
		{"Microsoft VS Code", "VS Code ripgrep directory resolution"},
		{"@vscode/ripgrep", "VS Code bundled ripgrep resolution"},
		{".cargo", "Cargo bin resolution"},
		{"findGit", "git fallback resolution"},
		{"git grep", "git grep fallback"},
		{"--no-index", "git grep --no-index flag"},
		{"ENOENT", "child ENOENT error handling"},
		{"rg_not_found", "typed rg_not_found error metadata"},
		{"No ripgrep binary found; install ripgrep or run git grep", "actionable error output message"},
	}
	for _, req := range requiredPatterns {
		if !strings.Contains(defaultOpenCodeGrep, req.pattern) {
			t.Errorf("defaultOpenCodeGrep missing %s (%q)", req.desc, req.pattern)
		}
	}
}

func TestOpenCodeGrepExecutionWitness(t *testing.T) {
	node, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node is required for OpenCode grep witness")
	}

	root := t.TempDir()
	testFile := filepath.Join(root, "sample.txt")
	if err := os.WriteFile(testFile, []byte("alpha target\nbeta line\ngamma target\n"), 0644); err != nil {
		t.Fatal(err)
	}

	pluginPath := filepath.Join(root, "fak-grep.js")
	if err := os.WriteFile(pluginPath, []byte(defaultOpenCodeGrep), 0644); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, node, "--input-type=module", "-", pluginPath, root)
	cmd.Stdin = strings.NewReader(`
import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';

const [pluginFile, root] = process.argv.slice(2);
const code = await readFile(pluginFile, 'utf8');
const mockPlugin = 'data:text/javascript;base64,' + Buffer.from('export const tool = Object.assign((s) => s, { schema: { string: () => ({ optional: () => ({}) }) } });').toString('base64');
const patched = code.replace('@opencode-ai/plugin', mockPlugin);

// 1. Live execution (using whatever binary is resolved or git fallback)
{
  const mod = await import('data:text/javascript;base64,' + Buffer.from(patched).toString('base64'));
  const plugin = await mod.default();
  const ctx = { directory: root, abort: new AbortController().signal, ask: async () => {} };
  const res = await plugin.tool.grep.execute({ pattern: 'target', path: 'sample.txt' }, ctx);
  assert.equal(res.metadata.matches, 2);
  assert.match(res.output, /sample\.txt:1:alpha target/);
  assert.match(res.output, /sample\.txt:3:gamma target/);

  const resNotFound = await plugin.tool.grep.execute({ pattern: 'nonexistent-pattern', path: 'sample.txt' }, ctx);
  assert.equal(resNotFound.metadata.matches, 0);
  assert.equal(resNotFound.output, 'No matches found');
}

// 2. When neither rg nor git is available, returns clean typed fallback without throwing ENOENT
{
  const noTools = patched
    .replace('function findRipgrep() {', 'function findRipgrep() { return null;')
    .replace('function findGit() {', 'function findGit() { return null;');
  const mod = await import('data:text/javascript;base64,' + Buffer.from(noTools).toString('base64'));
  const plugin = await mod.default();
  const ctx = { directory: root, abort: new AbortController().signal, ask: async () => {} };
  const res = await plugin.tool.grep.execute({ pattern: 'alpha' }, ctx);
  assert.equal(res.metadata.error, 'rg_not_found');
  assert.equal(res.output, 'No ripgrep binary found; install ripgrep or run git grep');
  assert.equal(res.metadata.matches, 0);
}

// 3. When child spawn fails with ENOENT, resolves gracefully with typed fallback instead of reject
{
  const spawnFail = patched
    .replace('function findRipgrep() {', "function findRipgrep() { return 'nonexistent-rg-bin.exe';")
    .replace('function findGit() {', 'function findGit() { return null;');
  const mod = await import('data:text/javascript;base64,' + Buffer.from(spawnFail).toString('base64'));
  const plugin = await mod.default();
  const ctx = { directory: root, abort: new AbortController().signal, ask: async () => {} };
  const res = await plugin.tool.grep.execute({ pattern: 'alpha' }, ctx);
  assert.equal(res.metadata.error, 'rg_not_found');
  assert.equal(res.output, 'No ripgrep binary found; install ripgrep or run git grep');
}

console.log(JSON.stringify({ ok: true }));
`)

	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Node grep witness failed: %v\n%s", err, out)
	}
	var receipt struct {
		OK bool `json:"ok"`
	}
	if err := json.Unmarshal(out, &receipt); err != nil || !receipt.OK {
		t.Fatalf("unexpected witness output: %s", out)
	}
}

func TestOpenCodeFastFailSteerageAndCircuitBreaker(t *testing.T) {
	node, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node is required for OpenCode fast-fail and circuit breaker test")
	}
	root := t.TempDir()
	pluginPath := filepath.Join(root, "dos-proof-guard.js")
	if err := os.WriteFile(pluginPath, []byte(DefaultOpenCodePlugin), 0644); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, node, "--no-warnings", "--input-type=module", "-", pluginPath, root)
	cmd.Stdin = strings.NewReader(`
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [pluginFile, root] = process.argv.slice(2);
const plugin = (await import(pathToFileURL(pluginFile).href)).default;
const hooks = await plugin.server({ directory: root });

// 1. Out-of-workspace scratch writes are ADMITTED; non-scratch out-of-tree
//    writes refuse with the structured OUT_OF_LEASE marker and no shell steerage.
{
  const scratchTarget = path.join(root, '..', 'AppData', 'Local', 'Temp', 'opencode', 'scratch.txt');
  await assert.doesNotReject(
    hooks['tool.execute.before']({ tool: 'write', sessionID: 'sess-scratch' }, { args: { filePath: scratchTarget } })
  );

  const normalOut = path.join(root, '..', 'sibling', 'file.txt');
  await assert.rejects(
    hooks['tool.execute.before']({ tool: 'write', sessionID: 'sess-scratch' }, { args: { filePath: normalOut } }),
    (err) => {
      assert.ok(err.message.includes('[dos-proof-guard:OUT_OF_LEASE]'));
      assert.ok(err.message.includes('do NOT re-attempt the same write through a different tool'));
      assert.ok(!/use shell commands/i.test(err.message));
      return true;
    }
  );
}

// 2. Structured refusal steerage on COLLISION_RISK containing [RECOVERY]
{
  const code = await readFile(pluginFile, 'utf8');
  let execCallCount = 0;
  const mockExec = 'let execCallCount = 0;\n' +
    'const execute = async () => { ' +
    'execCallCount++; ' +
    'const err = new Error("Command failed"); ' +
    'err.stdout = JSON.stringify({schema:"fak.loop-region.v1",admit:false,reason:"COLLISION_RISK",conflict:{holder:"worker-alpha"}}); ' +
    'throw err; ' +
    '};';
  const patched = code.replace('const execute = promisify(execFile);', mockExec) +
    '\nexport { execCallCount };';
  const mod = await import('data:text/javascript;base64,' + Buffer.from(patched).toString('base64'));
  const collisionPlugin = await mod.default.server({ directory: root });

  const collisionSession = 'sess-collision';
  const collisionPath = 'collision-file.go';

  // The mock crashes every admission subprocess: an AVAILABILITY failure. The
  // budgeted guard degrades those by default; the strict escape hatch restores
  // the fail-closed steerage this section witnesses, so scope it here.
  process.env.FAK_WRITE_GUARD_STRICT = '1';

  // Failure 1: encounters COLLISION_RISK and receives [RECOVERY] steerage
  await assert.rejects(
    collisionPlugin['tool.execute.before']({ tool: 'edit', sessionID: collisionSession }, { args: { filePath: collisionPath } }),
    (err) => {
      assert.ok(err.message.includes('COLLISION_RISK'));
      assert.ok(err.message.includes('[RECOVERY]'));
      assert.ok(err.message.includes("Lane conflict detected with holder 'worker-alpha'"));
      assert.ok(err.message.includes("Run 'fak recover COLLISION_RISK' or select a tree-disjoint task."));
      return true;
    }
  );

  // Failure 2: encounters COLLISION_RISK again
  await assert.rejects(
    collisionPlugin['tool.execute.before']({ tool: 'edit', sessionID: collisionSession }, { args: { filePath: collisionPath } }),
    (err) => {
      assert.ok(err.message.includes('COLLISION_RISK'));
      assert.ok(err.message.includes('[RECOVERY]'));
      return true;
    }
  );

  const countBefore3rd = mod.execCallCount;

  // Attempt 3: 3rd consecutive failing edit triggers circuit breaker refusal without dispatching subprocess commands
  await assert.rejects(
    collisionPlugin['tool.execute.before']({ tool: 'edit', sessionID: collisionSession }, { args: { filePath: collisionPath } }),
    (err) => {
      assert.ok(err.message.includes('[dos-proof-guard-circuit-breaker]'));
      assert.ok(err.message.includes('Refusing 3rd consecutive failing edit on ' + collisionPath));
      assert.ok(err.message.includes("Stop retrying. Inspect the file with 'read' or pivot to an alternate lane."));
      return true;
    }
  );

  // Verify child CLI command was not dispatched on 3rd attempt
  assert.equal(mod.execCallCount, countBefore3rd, 'Circuit breaker must refuse without dispatching child CLI commands');
  delete process.env.FAK_WRITE_GUARD_STRICT;

  // 2b. New default (non-strict): the same availability failure degrades to an
  //     ADMISSION_UNVERIFIED admit recorded in the QUEUEUS pending ledger.
  const degradedSession = 'sess-degraded';
  await assert.doesNotReject(
    collisionPlugin['tool.execute.before']({ tool: 'edit', sessionID: degradedSession }, { args: { filePath: 'degraded-target.go' } })
  );
  const degradedResult = { output: 'ok' };
  await collisionPlugin['tool.execute.after'](
    { tool: 'edit', sessionID: degradedSession, args: { filePath: 'degraded-target.go' } },
    degradedResult
  );
  assert.ok(degradedResult.output.includes('[dos-proof-guard:ADMISSION_UNVERIFIED]'), 'degraded note missing: ' + degradedResult.output);
  const pending = JSON.parse(await readFile(path.join(root, '.fak', 'queueus', 'admission-pending.json'), 'utf8'));
  assert.equal(pending.schema, 'fak.queueus.pending.v1');
  assert.equal(pending.records.length, 1);
  assert.equal(pending.records[0].kind, 'admission_unverified');
}

// 3. Circuit breaker on string-matching failure and reset on success
{
  const session = 'sess-cb-reset';
  const targetFile = 'circuit-reset-target.go';

  // Failure 1 in tool.execute.after
  await hooks['tool.execute.after'](
    { tool: 'edit', sessionID: session, args: { filePath: targetFile } },
    { output: 'Error: Could not find oldString in file' }
  );

  // Failure 2 in tool.execute.after
  await hooks['tool.execute.after'](
    { tool: 'edit', sessionID: session, args: { filePath: targetFile } },
    { output: 'Error: Found multiple matches for oldString' }
  );

  // Attempt 3: trips circuit breaker in tool.execute.before
  await assert.rejects(
    hooks['tool.execute.before']({ tool: 'edit', sessionID: session }, { args: { filePath: targetFile } }),
    (err) => {
      assert.ok(err.message.includes('[dos-proof-guard-circuit-breaker]'));
      assert.ok(err.message.includes('Refusing 3rd consecutive failing edit on ' + targetFile));
      return true;
    }
  );

  // Successful edit resets the failure counter for that path
  await hooks['tool.execute.after'](
    { tool: 'edit', sessionID: session, args: { filePath: targetFile } },
    { output: 'Successfully replaced 1 instance' }
  );

  // Failure 1 after reset
  await hooks['tool.execute.after'](
    { tool: 'edit', sessionID: session, args: { filePath: targetFile } },
    { output: 'Error: Could not find oldString in file' }
  );

  // Next attempt: failure count is 1, so breaker MUST NOT trip
  let tripped = false;
  try {
    await hooks['tool.execute.before']({ tool: 'edit', sessionID: session }, { args: { filePath: targetFile } });
  } catch (err) {
    if (err.message.includes('[dos-proof-guard-circuit-breaker]')) {
      tripped = true;
    }
  }
  assert.equal(tripped, false, 'Circuit breaker must not trip after failure counter reset');
}

// 4. Pre-flight fast-fail rejects null byte in path
{
  await assert.rejects(
    hooks['tool.execute.before']({ tool: 'write', sessionID: 'sess-malformed' }, { args: { filePath: 'foo\0bar.txt' } }),
    /Malformed mutation path: contains null byte/
  );
}

console.log(JSON.stringify({ ok: true }));
`)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("OpenCode fast-fail & circuit breaker test failed: %v\n%s", err, out)
	}
	var receipt struct {
		OK bool `json:"ok"`
	}
	if err := json.Unmarshal(out, &receipt); err != nil || !receipt.OK {
		t.Fatalf("unexpected witness output: %s", out)
	}
}
