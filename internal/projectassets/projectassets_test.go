package projectassets

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strconv"
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
	if !strings.Contains(string(b), "description:") || !strings.Contains(string(b), "Verify.") {
		t.Fatalf("adapter lost canonical discovery description:\n%s", b)
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

func TestAdapterDescriptionHonorsAgentSkillsLimit(t *testing.T) {
	long := strings.Repeat("discovery trigger ", 100)
	got := adapterDescription(long)
	if chars := len([]rune(got)); chars > maxSkillDescriptionChars {
		t.Fatalf("description has %d characters, want at most %d", chars, maxSkillDescriptionChars)
	}
	if !strings.HasSuffix(got, "...") {
		t.Fatalf("truncated description = %q, want ellipsis", got)
	}
	if got != adapterDescription(long) {
		t.Fatal("description projection is not deterministic")
	}
	short := "Use when a project needs portable skill discovery."
	if got := adapterDescription(short); got != short {
		t.Fatalf("short description = %q, want %q", got, short)
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

func TestAdapterEmitsRoundTrippableYAMLScalars(t *testing.T) {
	tests := []struct {
		name string
		text string
		want string
	}{
		{name: "plain", text: "Portable skill discovery.", want: "Portable skill discovery."},
		{name: "colon", text: "Use when input has a colon: preserve it.", want: strconv.Quote("Use when input has a colon: preserve it.")},
		{name: "escaped", text: `Say "go" from C:\work.`, want: strconv.Quote(`Say "go" from C:\work.`)},
		{name: "newline", text: "First line\nsecond line.", want: strconv.Quote("First line\nsecond line.")},
		{name: "semantic whitespace", text: "  preserve semantic spacing  ", want: strconv.Quote("  preserve semantic spacing  ")},
		{name: "implicit positive number", text: "+1.0", want: strconv.Quote("+1.0")},
		{name: "implicit fractional number", text: ".5", want: strconv.Quote(".5")},
		{name: "implicit special float", text: ".nan", want: strconv.Quote(".nan")},
		{name: "YAML line separator", text: "before\u2028after", want: strconv.Quote("before\u2028after")},
		{name: "YAML paragraph separator", text: "before\u2029after", want: strconv.Quote("before\u2029after")},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := adapter("example", tt.text, "../../../.claude/skills/example/SKILL.md")
			line := ""
			for _, candidate := range strings.Split(body, "\n") {
				if strings.HasPrefix(candidate, "description: ") {
					line = strings.TrimPrefix(candidate, "description: ")
					break
				}
			}
			if line != tt.want {
				t.Fatalf("emitted scalar = %q, want %q\n%s", line, tt.want, body)
			}
			if got := normalizeYAMLScalar(line); got != tt.text {
				t.Fatalf("round trip = %q, want %q", got, tt.text)
			}
		})
	}
}

func TestQuotedDescriptionIsNormalizedBeforeTruncation(t *testing.T) {
	semantic := strings.Repeat("quoted discovery trigger ", 20)
	root := t.TempDir()
	path := ".claude/skills/example/SKILL.md"
	write(t, root, ManifestPath, baseManifest())
	write(t, root, path, "---\nname: example\ndescription: "+strconv.Quote(semantic)+"\n---\n")
	write(t, root, ".claude/memory/base.md", "memory\n")
	write(t, root, ".claude/goal-prompts/base.md", "prompt\n")

	description, err := skillDescription(root, path)
	if err != nil {
		t.Fatal(err)
	}
	got := adapterDescription(description)
	if chars := len([]rune(got)); chars > maxSkillDescriptionChars {
		t.Fatalf("description has %d characters, want at most %d", chars, maxSkillDescriptionChars)
	}
	if strings.HasPrefix(got, `"`) || !strings.HasSuffix(got, "...") {
		t.Fatalf("truncated semantic description = %q", got)
	}

	if _, err = Build(root, true); err != nil {
		t.Fatal(err)
	}
	bodyBytes, err := os.ReadFile(filepath.Join(root, ".agents", "skills", "example", "SKILL.md"))
	if err != nil {
		t.Fatal(err)
	}
	body := string(bodyBytes)
	line := strings.SplitN(strings.SplitN(body, "description: ", 2)[1], "\n", 2)[0]
	if roundTrip := normalizeYAMLScalar(line); roundTrip != got {
		t.Fatalf("emitted description round trip = %q, want %q\n%s", roundTrip, got, body)
	}
}

func TestCodexAdapterDescriptionsCutResidentFloorByThree(t *testing.T) {
	const baselineChars = 50000
	root := filepath.Clean(filepath.Join("..", ".."))
	files, err := skillFiles(root, filepath.ToSlash(filepath.Join(".agents", "skills")))
	if err != nil {
		t.Fatal(err)
	}
	var total int
	for _, path := range files {
		description, err := skillDescription(root, path)
		if err != nil {
			t.Fatal(err)
		}
		if description == "" {
			t.Fatalf("%s has an empty description", path)
		}
		total += len([]rune(description))
	}
	if total*3 > baselineChars {
		t.Fatalf("adapter descriptions total %d characters, want at most one third of baseline %d", total, baselineChars)
	}
}

func TestAdapterDecodesAndRequotesYAMLDescriptions(t *testing.T) {
	root := t.TempDir()
	write(t, root, ".claude/skills/quoted/SKILL.md", "---\nname: quoted\ndescription: \"Use a colon: safely and say \\\"hello\\\".\"\n---\n")
	description, err := skillDescription(root, ".claude/skills/quoted/SKILL.md")
	if err != nil {
		t.Fatal(err)
	}
	if want := `Use a colon: safely and say "hello".`; description != want {
		t.Fatalf("description = %q, want %q", description, want)
	}
	body := adapter("quoted", strings.Repeat(description+" ", 20), "../../../.claude/skills/quoted/SKILL.md")
	line := strings.Split(body, "\n")[2]
	if !strings.HasPrefix(line, `description: "`) || !strings.HasSuffix(line, `..."`) {
		t.Fatalf("adapter description is not a complete quoted scalar: %q", line)
	}
	var decoded string
	if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "description: ")), &decoded); err != nil {
		t.Fatalf("generated description is invalid JSON/YAML quoting: %v", err)
	}
	if len([]rune(decoded)) > maxSkillDescriptionChars {
		t.Fatalf("decoded description has %d characters", len([]rune(decoded)))
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
