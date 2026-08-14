package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"
)

// writeSkill writes .claude/skills/<name>/SKILL.md with the given frontmatter
// body inside the temp root, and returns the skill directory.
func writeSkill(t *testing.T, root, name, body string) string {
	t.Helper()
	dir := filepath.Join(root, ".claude", "skills", name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", dir, err)
	}
	if err := os.WriteFile(filepath.Join(dir, "SKILL.md"), []byte(body), 0o644); err != nil {
		t.Fatalf("write SKILL.md: %v", err)
	}
	return dir
}

// skillFrontmatter builds a SKILL.md body with the given frontmatter keys plus a
// padded body so the file is larger than the at-rest card bytes.
func skillFrontmatter(name, version, description string) string {
	return "---\nname: " + name + "\nversion: \"" + version + "\"\ndescription: " + description +
		"\ntags: [test]\n---\n\n# " + name + "\n\n" +
		"This is the full skill body. It is intentionally longer than the small at-rest\n" +
		"card JSON so the loader's paging check (card bytes < body bytes) holds.\n"
}

// chdirRepo chdirs into a temp dir that carries a go.mod so repoRoot() resolves
// to it; the framework restores the original working directory on cleanup.
func chdirRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module test\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}
	t.Chdir(root)
	return root
}

func decodeJSON(t *testing.T, b []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("decode JSON %q: %v", string(b), err)
	}
	return m
}

// --- loader dimension (scorecard) -----------------------------------------

func TestLoaderDebtZeroForCleanCatalog(t *testing.T) {
	root := chdirRepo(t)
	writeSkill(t, root, "clean", skillFrontmatter("clean", "1.0.0", "Use when you need to clean things"))

	debt, queryable, pages, inSync := collectLoaderDebt(root)
	if debt != 0 || queryable != 0 || pages != 0 || inSync != 0 {
		t.Fatalf("clean catalog: debt=%d (queryable=%d pages=%d inSync=%d), want all zero",
			debt, queryable, pages, inSync)
	}

	p := collectSkillEffectivenessScorecard(root)
	if p["verdict"] != "OK" {
		t.Fatalf("clean catalog verdict=%v, want OK", p["verdict"])
	}
	corpus := p["corpus"].(map[string]any)
	if corpus["loader_debt"] != 0 {
		t.Fatalf("clean catalog loader_debt=%v, want 0", corpus["loader_debt"])
	}
}

func TestLoaderDebtQueryableOnEmptyTrigger(t *testing.T) {
	root := chdirRepo(t)
	// No description frontmatter → the card's trigger is empty → un-queryable.
	writeSkill(t, root, "silent", "---\nname: silent\n---\n\n# silent\n\nbody\n")

	_, queryable, _, _ := collectLoaderDebt(root)
	if queryable == 0 {
		t.Fatalf("a skill with an empty description should score a queryable debt unit")
	}
}

func TestLoaderDebtInSyncIdempotent(t *testing.T) {
	root := chdirRepo(t)
	writeSkill(t, root, "stable", skillFrontmatter("stable", "2.0.0", "Use when you need stability"))

	_, _, _, inSync := collectLoaderDebt(root)
	if inSync != 0 {
		t.Fatalf("re-syncing an unchanged catalog must be idempotent; got inSync=%d", inSync)
	}
}

// --- query -----------------------------------------------------------------

func TestSkillQueryRanksAndFaultsWinner(t *testing.T) {
	root := chdirRepo(t)
	writeSkill(t, root, "frobnicate", skillFrontmatter("frobnicate", "1.0.0", "Use when you need to frobnicate widgets"))
	writeSkill(t, root, "unrelated", skillFrontmatter("unrelated", "1.0.0", "Use when you need to bake bread"))

	var out bytes.Buffer
	code := runSkillQuery(&out, io.Discard, []string{"--budget", "1", "--json", "frobnicate widgets"})
	if code != 0 {
		t.Fatalf("runSkillQuery exit=%d", code)
	}
	m := decodeJSON(t, out.Bytes())
	ws := m["working_set"].([]any)
	if len(ws) == 0 {
		t.Fatalf("working set empty for a matching intent")
	}
	first := ws[0].(map[string]any)
	if first["name"] != "frobnicate" {
		t.Fatalf("top-ranked card = %v, want frobnicate", first["name"])
	}
	winners := m["winners"].([]any)
	if len(winners) != 1 {
		t.Fatalf("winners=%d, want 1 (budget=1)", len(winners))
	}
	if cost := m["cost_bytes"]; cost.(float64) <= 0 {
		t.Fatalf("cost_bytes=%v, want >0 (the winner body was faulted)", cost)
	}
}

func TestSkillQueryRequiresIntent(t *testing.T) {
	root := chdirRepo(t)
	writeSkill(t, root, "x", skillFrontmatter("x", "1.0.0", "Use when x"))

	var errb bytes.Buffer
	code := runSkillQuery(io.Discard, &errb, []string{"--json"})
	if code != 2 {
		t.Fatalf("empty intent: exit=%d, want 2", code)
	}
}

// TestSkillQueryFlagsAfterIntent pins the documented syntax
// `fak skill query <intent> [--budget N]`: flags must still parse when they
// follow the free-form positional intent (Go's stdlib flag stops at the first
// positional, so partitionArgs reorders them).
func TestSkillQueryFlagsAfterIntent(t *testing.T) {
	root := chdirRepo(t)
	writeSkill(t, root, "frob", skillFrontmatter("frob", "1.0.0", "Use when you need to frob"))

	var out bytes.Buffer
	code := runSkillQuery(&out, io.Discard, []string{"frob widgets", "--budget", "1", "--json"})
	if code != 0 {
		t.Fatalf("exit=%d", code)
	}
	m := decodeJSON(t, out.Bytes())
	if m["intent"] != "frob widgets" {
		t.Fatalf("intent=%v, want 'frob widgets' (--budget must not leak into intent)", m["intent"])
	}
	if m["budget"] != float64(1) {
		t.Fatalf("budget=%v, want 1 (flag parsed after positional)", m["budget"])
	}
}

// --- residency -------------------------------------------------------------

func TestSkillResidencyListsCardsAndPins(t *testing.T) {
	root := chdirRepo(t)
	writeSkill(t, root, "alpha", skillFrontmatter("alpha", "1.0.0", "Use when you need alpha"))

	var out bytes.Buffer
	if code := runSkillResidency(&out, io.Discard, []string{"--json"}); code != 0 {
		t.Fatalf("runSkillResidency exit=%d", code)
	}
	m := decodeJSON(t, out.Bytes())
	cards := m["cards"].([]any)
	if len(cards) != 1 {
		t.Fatalf("residency cards=%d, want 1", len(cards))
	}
	if pt, ok := m["page_table"].(map[string]any); !ok || len(pt) != 0 {
		t.Fatalf("page_table=%v, want empty before any swap", m["page_table"])
	}
}

// --- footprint (resident-floor scorecard, #3234) ---------------------------

func TestSkillFootprintSumsResidentBytes(t *testing.T) {
	root := chdirRepo(t)
	writeSkill(t, root, "heavy", skillFrontmatter("heavy", "1.0.0", "Use when you need a very very very long description that dominates the resident floor"))
	writeSkill(t, root, "light", skillFrontmatter("light", "1.0.0", "Use when light"))

	var out bytes.Buffer
	if code := runSkillFootprint(&out, io.Discard, []string{"--json"}); code != 0 {
		t.Fatalf("runSkillFootprint exit=%d", code)
	}
	m := decodeJSON(t, out.Bytes())
	if m["skill_count"].(float64) != 2 {
		t.Fatalf("skill_count=%v, want 2", m["skill_count"])
	}
	floor := m["description_floor_bytes"].(float64)
	if floor <= 0 {
		t.Fatalf("description_floor_bytes=%v, want >0", floor)
	}
	entries := m["entries"].([]any)
	if len(entries) != 2 {
		t.Fatalf("entries=%d, want 2", len(entries))
	}
	// Heaviest resident description first: 'heavy' has the longer description.
	if first := entries[0].(map[string]any); first["name"] != "heavy" {
		t.Fatalf("heaviest entry=%v, want heavy", first["name"])
	}
	// The floor total equals the sum of the per-entry description bytes — the
	// resident tax is exactly the sum of the frontmatter description fields.
	sum := 0.0
	for _, e := range entries {
		sum += e.(map[string]any)["description_bytes"].(float64)
	}
	if sum != floor {
		t.Fatalf("description_floor_bytes %v != sum of entry desc bytes %v", floor, sum)
	}
}

func TestSkillFootprintTopLimits(t *testing.T) {
	root := chdirRepo(t)
	writeSkill(t, root, "a", skillFrontmatter("a", "1.0.0", "Use when you need aaaa long enough to rank"))
	writeSkill(t, root, "b", skillFrontmatter("b", "1.0.0", "Use when you need bbbb an even longer description here to outrank"))
	writeSkill(t, root, "c", skillFrontmatter("c", "1.0.0", "Use when c"))

	var out bytes.Buffer
	if code := runSkillFootprint(&out, io.Discard, []string{"--top", "2", "--json"}); code != 0 {
		t.Fatalf("runSkillFootprint exit=%d", code)
	}
	m := decodeJSON(t, out.Bytes())
	if h := m["heaviest"].([]any); len(h) != 2 {
		t.Fatalf("heaviest=%d, want 2 (--top 2)", len(h))
	}
	if e := m["entries"].([]any); len(e) != 3 {
		t.Fatalf("entries=%d, want 3 (--top bounds the heaviest list, not entries)", len(e))
	}
}

// --- headless name-only profile (#3612) ------------------------------------

// TestSkillFootprintHeadlessProfileIsSmaller pins the #3612 acceptance: a
// headless dispatch worker's resident skills floor is name-only and strictly
// smaller than the interactive name+description floor (#3234), the interactive
// profile is unchanged, and every skill name survives — so a skill is still
// invocable by name from a headless worker.
func TestSkillFootprintHeadlessProfileIsSmaller(t *testing.T) {
	root := chdirRepo(t)
	writeSkill(t, root, "alpha", skillFrontmatter("alpha", "1.0.0", "Use when you need a long alpha description that dominates the resident floor"))
	writeSkill(t, root, "beta", skillFrontmatter("beta", "1.0.0", "Use when you need an even longer beta description here that also dominates the floor"))

	// Interactive (default) profile keeps #3234 behavior: resident floor == description floor.
	var interOut bytes.Buffer
	if code := runSkillFootprint(&interOut, io.Discard, []string{"--json"}); code != 0 {
		t.Fatalf("interactive runSkillFootprint exit=%d", code)
	}
	inter := decodeJSON(t, interOut.Bytes())
	if inter["profile"] != "interactive" {
		t.Fatalf("default profile=%v, want interactive", inter["profile"])
	}
	if inter["resident_floor_bytes"] != inter["description_floor_bytes"] {
		t.Fatalf("interactive resident_floor_bytes=%v != description_floor_bytes=%v (must keep #3234 behavior)",
			inter["resident_floor_bytes"], inter["description_floor_bytes"])
	}

	// Headless profile: the resident floor is name-only and strictly smaller.
	var headOut bytes.Buffer
	if code := runSkillFootprint(&headOut, io.Discard, []string{"--profile", "headless", "--json"}); code != 0 {
		t.Fatalf("headless runSkillFootprint exit=%d", code)
	}
	head := decodeJSON(t, headOut.Bytes())
	if head["profile"] != "headless" {
		t.Fatalf("profile=%v, want headless", head["profile"])
	}
	nameFloor := head["name_floor_bytes"].(float64)
	descFloor := head["description_floor_bytes"].(float64)
	if head["resident_floor_bytes"].(float64) != nameFloor {
		t.Fatalf("headless resident_floor_bytes=%v, want name_floor_bytes=%v", head["resident_floor_bytes"], nameFloor)
	}
	if !(nameFloor > 0 && nameFloor < descFloor) {
		t.Fatalf("headless name-only floor=%v must be >0 and < description floor=%v", nameFloor, descFloor)
	}

	// Every skill name is still present in the rendered index → invocable by name.
	names := map[string]bool{}
	for _, e := range head["entries"].([]any) {
		names[e.(map[string]any)["name"].(string)] = true
	}
	for _, want := range []string{"alpha", "beta"} {
		if !names[want] {
			t.Fatalf("headless index missing skill name %q — not invocable by name", want)
		}
	}
}

// TestSkillFootprintRejectsUnknownProfile pins the closed profile vocabulary:
// an unrecognized --profile is a usage error, not a silent fallback.
func TestSkillFootprintRejectsUnknownProfile(t *testing.T) {
	root := chdirRepo(t)
	writeSkill(t, root, "x", skillFrontmatter("x", "1.0.0", "Use when x"))
	_ = root

	var errb bytes.Buffer
	if code := runSkillFootprint(io.Discard, &errb, []string{"--profile", "bogus", "--json"}); code != 2 {
		t.Fatalf("unknown profile exit=%d, want 2: %s", code, errb.String())
	}
}

// --- swap ------------------------------------------------------------------

func TestSkillSwapPersistsAndResidencyReadsBack(t *testing.T) {
	root := chdirRepo(t)
	writeSkill(t, root, "swappable", skillFrontmatter("swappable", "1.0.0", "Use when you need to swap"))

	var swapOut bytes.Buffer
	if code := runSkillSwap(&swapOut, io.Discard, []string{"--json", "swappable", "2.0.0"}); code != 0 {
		t.Fatalf("runSkillSwap exit=%d: %s", code, swapOut.String())
	}
	m := decodeJSON(t, swapOut.Bytes())
	if m["to_version"] != "2.0.0" {
		t.Fatalf("swap to_version=%v, want 2.0.0", m["to_version"])
	}

	// The page-table file was persisted.
	pins := loadSkillPageTable(root)
	if pins["swappable"] != "2.0.0" {
		t.Fatalf("persisted pin=%q, want 2.0.0", pins["swappable"])
	}

	// Residency reads the pin back.
	var resOut bytes.Buffer
	runSkillResidency(&resOut, io.Discard, []string{"--json"})
	res := decodeJSON(t, resOut.Bytes())
	pt := res["page_table"].(map[string]any)
	if pt["swappable"] != "2.0.0" {
		t.Fatalf("residency page_table swappable=%v, want 2.0.0", pt["swappable"])
	}
}

func TestSkillSwapFromGuardRefuses(t *testing.T) {
	root := chdirRepo(t)
	writeSkill(t, root, "guarded", skillFrontmatter("guarded", "1.0.0", "Use when guarded"))

	var errb bytes.Buffer
	// --from says 9.9.9 but the skill declares 1.0.0 → the Swap guard refuses.
	code := runSkillSwap(io.Discard, &errb, []string{"--from", "9.9.9", "guarded", "2.0.0"})
	if code != 1 {
		t.Fatalf("guarded swap exit=%d, want 1 (refused): %s", code, errb.String())
	}
}

func TestRunSkillCompileRequiresExplicitExposure(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "SKILL.md")
	if err := os.WriteFile(path, []byte(skillCompileFixture), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := runSkillCompile(&stdout, &stderr, []string{"--json", path}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var got struct {
		Registration struct {
			Program struct {
				Name string `json:"name"`
			} `json:"program"`
		} `json:"registration"`
		ModelView struct {
			Tools   []json.RawMessage `json:"tools"`
			Omitted []struct {
				Reason string `json:"reason"`
			} `json:"omitted"`
		} `json:"model_view"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Registration.Program.Name != "repo_search" || len(got.ModelView.Tools) != 0 || len(got.ModelView.Omitted) != 1 || got.ModelView.Omitted[0].Reason != "NOT_SELECTED" {
		t.Fatalf("registration leaked into model availability: %s", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	if code := runSkillCompile(&stdout, &stderr, []string{"--json", "--dialect", "codex", "--expose", "repo_search", path}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte(`"name": "functions.shell_command"`)) {
		t.Fatalf("dialect-native tool name missing: %s", stdout.String())
	}
}

const skillCompileFixture = `---
name: repo_search
description: Search repository text.
---
# Search
` + "```fak-program" + `
{"version":"fak.skill-program/v1","name":"repo_search","input_schema":{"type":"object","properties":{"query":{"type":"string"}}},"executor":{"argv":["fak","code","search","--json"]},"aliases":{"codex":"functions.shell_command"}}
` + "```" + `
`
