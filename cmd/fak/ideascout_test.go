package main

import (
	"bytes"
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

// TestIdeaScoutVerbIsDispatched is the reachability pin of #5546.
//
// runIdeaScout compiled, was fully flag-parsed, and had a usage block from the day
// it landed -- and for months answered `fak: unknown verb "idea-scout"`, because the
// one thing it never had was a case in the cmd/fak/main.go dispatch switch. Its
// `cmdIdeaScout(argv) { os.Exit(runIdeaScout(...)) }` wrapper was then swept as dead
// code (#1419), which was correct about the symptom and wrong about the cure: the
// caller it was missing was the dispatch arm. Meanwhile docs/idea-scout.md and two
// .claude/skills/question-loop/SKILL.md lines told agents to run the verb.
//
// TestRunIdeaScoutDryRunWithFixtureCandidates cannot catch that class of bug: it
// calls runIdeaScout directly, so it stays green whether or not any user can reach
// it. This test asserts the two rungs a USER actually traverses instead:
//
//	rung 1  cmd/fak/main.go has a `case "idea-scout":` whose body calls runIdeaScout
//	        -- checked over the parsed AST, so a case that dispatched somewhere else
//	        (or an arm deleted by a future sweep) reds here, not in the field.
//	rung 2  devindex.DispatchVerbs -- the SAME parser the VERB_UNTIERED pre-push gate
//	        and the `fak help --all` / `fak dev` catalog read -- sees the token, and it
//	        carries a tier. A verb the catalog cannot see is one no help surface lists.
//
// It scans the working tree (HEAD in CI) and asserts a property of correctly-wired
// code, not a snapshot, so it behaves identically in both.
func TestIdeaScoutVerbIsDispatched(t *testing.T) {
	const verb = "idea-scout"
	src, err := os.ReadFile("main.go")
	if err != nil {
		t.Skipf("cmd/fak/main.go unreadable (%v); dispatch reachability is only checkable in-tree", err)
	}

	// rung 1: the case arm exists AND routes to runIdeaScout.
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "main.go", src, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse cmd/fak/main.go: %v", err)
	}
	var arm *ast.CaseClause
	ast.Inspect(file, func(n ast.Node) bool {
		cc, ok := n.(*ast.CaseClause)
		if !ok {
			return true
		}
		for _, e := range cc.List {
			lit, ok := e.(*ast.BasicLit)
			if ok && lit.Kind == token.STRING && strings.Trim(lit.Value, `"`) == verb {
				arm = cc
				return false
			}
		}
		return true
	})
	if arm == nil {
		t.Fatalf("cmd/fak/main.go has no `case %q:` -- `fak %s` answers \"unknown verb\" no matter what runIdeaScout does. "+
			"Add the arm to the dispatch switch (see the `boundary` arm for the same fix).", verb, verb)
	}
	called := false
	ast.Inspect(arm, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && id.Name == "runIdeaScout" {
			called = true
		}
		return true
	})
	if !called {
		t.Errorf("the `case %q:` arm at %s does not call runIdeaScout -- the verb dispatches somewhere else",
			verb, fset.Position(arm.Pos()))
	}

	// rung 2: the catalog/tier parser agrees, so help and the tier gate see it too.
	dispatched := false
	for _, tok := range devindex.DispatchVerbs(src) {
		if tok == verb {
			dispatched = true
			break
		}
	}
	if !dispatched {
		t.Errorf("devindex.DispatchVerbs does not see %q in cmd/fak/main.go -- `fak help --all`, `fak dev`, "+
			"and the VERB_UNTIERED gate all read that parser, so the verb would stay invisible to every help surface", verb)
	}
	if tier, ok := devindex.TierOf(verb); !ok {
		t.Errorf("%q has no tier in internal/devindex/tiers.go -- classify it in one tier block or the pre-push VERB_UNTIERED gate reds the tree", verb)
	} else if tier != devindex.TierDev {
		t.Errorf("%q is tiered %q; it is internal fleet tooling, not product surface (the frontdoor tier is ceiling-gated)", verb, tier)
	}
}

// TestIdeaScoutCarriesAHelpRow pins the other half of a reachable verb: `fak help
// idea-scout` must answer. A dev-tier verb has no cmd/fak/help.go overviewGroups line
// by construction (the overview is frontdoor-ONLY, gated by TestOverviewIsExactlyFrontdoor),
// so its help row is the devindex catalog entry -- which is also what `fak help --all`
// and `fak dev` list. Without it, an agent that follows the skill and then reaches for
// help gets nothing.
func TestIdeaScoutCarriesAHelpRow(t *testing.T) {
	var out bytes.Buffer
	if !printVerbHelp(&out, "idea-scout") {
		t.Fatal("`fak help idea-scout` knows nothing about the verb -- add a verbManifest entry in internal/devindex/verbs.go")
	}
	got := out.String()
	// Dev-tier verbs are introduced by their canonical `fak dev <verb>` spelling.
	if !strings.Contains(got, "fak dev idea-scout") {
		t.Errorf("help header does not name the canonical dev spelling:\n%s", got)
	}
	// The help row must carry the --live blast radius forward, not just a neutral blurb:
	// this is the one verb in the family whose opt-in files public GitHub issues.
	if !strings.Contains(got, "--live") {
		t.Errorf("help row does not mention --live, the only flag with an effect outside the process:\n%s", got)
	}
}

func TestRunIdeaScoutDryRunWithFixtureCandidates(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "topics.json")
	if err := os.WriteFile(configPath, []byte(`{
		"topics": [{"key":"fixture","github":"fixture","terms":["agent","tool","policy"],"area":"trust-floor"}],
		"thresholds": {"min_score": 1, "max_issues": 1}
	}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	candidatesPath := filepath.Join(dir, "candidates.json")
	if err := os.WriteFile(candidatesPath, []byte(`[
		{"source":"github","source_id":"github:o/r","url":"https://github.com/o/r","title":"o/r","summary":"agent tool policy","published":"2026-06-29T00:00:00Z","topic":"fixture","extra":{"stars":300,"pushed_at":"2026-06-30T00:00:00Z","language":"Go"}},
		{"source":"github","source_id":"github:o/dupe","url":"https://github.com/o/dupe","title":"dupe","summary":"agent tool policy","published":"2026-06-29T00:00:00Z","topic":"fixture"}
	]`), 0o644); err != nil {
		t.Fatalf("write candidates: %v", err)
	}
	issuesPath := filepath.Join(dir, "issues.json")
	if err := os.WriteFile(issuesPath, []byte(`[{"number":1,"title":"dupe","body":"manual issue"}]`), 0o644); err != nil {
		t.Fatalf("write issues: %v", err)
	}

	var stdout, stderr bytes.Buffer
	code := runIdeaScout(&stdout, &stderr, []string{
		"--workspace", dir,
		"--config", configPath,
		"--candidates", candidatesPath,
		"--issues", issuesPath,
		"--json",
		"--today", "2026-06-30",
	})
	if code != 0 {
		t.Fatalf("runIdeaScout code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	var result struct {
		Mode               string `json:"mode"`
		CandidatesGathered int    `json:"candidates_gathered"`
		Planned            []struct {
			Title    string   `json:"title"`
			SourceID string   `json:"source_id"`
			Labels   []string `json:"labels"`
		} `json:"planned"`
		Skipped map[string]int `json:"skipped"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("stdout is not JSON: %v\n%s", err, stdout.String())
	}
	if result.Mode != "dry-run" || result.CandidatesGathered != 2 {
		t.Fatalf("result mode/gathered = %#v", result)
	}
	if len(result.Planned) != 1 || result.Planned[0].SourceID != "github:o/r" {
		t.Fatalf("planned = %#v, want only github:o/r", result.Planned)
	}
	if result.Skipped["title-near"] != 1 {
		t.Fatalf("skipped = %#v, want title-near=1", result.Skipped)
	}
	if strings.Contains(stdout.String(), `"body"`) {
		t.Fatalf("JSON plan should omit full issue bodies from the summary output: %s", stdout.String())
	}
	if _, err := os.Stat(filepath.Join(dir, ".idea-scout", "seen.json")); !os.IsNotExist(err) {
		t.Fatalf("dry-run wrote seen cache, stat err=%v", err)
	}
}
