package dispatchtick

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devindex"
	"github.com/anthony-chaudhary/fak/internal/promptlint"
)

// promptrules_test.go — the freshness gate that makes the #3220 rule set self-checking.
//
// The point of moving the guidance from prose to {id, imperative, witness} data is that
// the witnesses become CHECKABLE. This file is the check: it folds every rule's witness
// against the same authoritative registries internal/promptlint lints a rendered prompt
// with, read LIVE from the repo rather than copied into a fixture. Retire a reason token,
// rename a `fak` verb, or invent a witness nothing declares, and these tests red.

// repoRootForRules walks up from the package dir to the checkout root (the dir holding
// dos.toml), so the registry reads below work from any working directory.
func repoRootForRules(t *testing.T) string {
	t.Helper()
	dir, err := filepath.Abs(".")
	if err != nil {
		t.Fatalf("abs: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "dos.toml")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("no dos.toml above %s - cannot locate the repo root", dir)
		}
		dir = parent
	}
}

var (
	dosReasonRe  = regexp.MustCompile(`(?m)^\[reasons\.([A-Z][A-Z0-9_]*)\]`)
	goReasonRe   = regexp.MustCompile(`(?m)^\s*(?:const\s+)?Reason[A-Za-z0-9]*\s*=\s*"([A-Z][A-Z0-9_]*)"`)
	upperTokenRe = regexp.MustCompile(`^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$`)
)

// liveKnown assembles promptlint.Known from fak's OWN live registries, exactly as
// promptlint's package doc specifies the authority: the dos.toml closed reason set, the
// safecommit + pythongate reason vocabularies, and the live cmd/fak dispatch verb catalog
// (devindex derives verb COVERAGE from cmd/fak/main.go, so it cannot lag the binary).
func liveKnown(t *testing.T) promptlint.Known {
	t.Helper()
	root := repoRootForRules(t)

	var reasons []string
	dosToml, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		t.Fatalf("read dos.toml: %v", err)
	}
	for _, m := range dosReasonRe.FindAllStringSubmatch(string(dosToml), -1) {
		reasons = append(reasons, m[1])
	}
	// The commit-gate tokens that live in Go source rather than dos.toml (PATHSPEC_RACE,
	// NEW_PYTHON_TOOL, ...). Scanning the declaring package is what keeps this a LIVE read:
	// delete or rename the constant and the witness citing it stops resolving.
	for _, pkg := range []string{"safecommit", "pythongate"} {
		entries, err := os.ReadDir(filepath.Join(root, "internal", pkg))
		if err != nil {
			t.Fatalf("read internal/%s: %v", pkg, err)
		}
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") || strings.HasSuffix(e.Name(), "_test.go") {
				continue
			}
			src, err := os.ReadFile(filepath.Join(root, "internal", pkg, e.Name()))
			if err != nil {
				t.Fatalf("read internal/%s/%s: %v", pkg, e.Name(), err)
			}
			for _, m := range goReasonRe.FindAllStringSubmatch(string(src), -1) {
				reasons = append(reasons, m[1])
			}
		}
	}
	if len(reasons) < 20 {
		t.Fatalf("only %d reason tokens resolved from the live registries - the scan is broken, "+
			"not the rules", len(reasons))
	}

	cat, err := devindex.Load(root)
	if err != nil {
		t.Fatalf("devindex.Load(%s): %v", root, err)
	}
	var verbs []string
	for _, v := range cat.Verbs() {
		verbs = append(verbs, v.Spellings()...)
	}
	if len(verbs) < 20 {
		t.Fatalf("only %d fak verbs resolved from the live catalog - the scan is broken, "+
			"not the rules", len(verbs))
	}
	return promptlint.NewKnown(verbs, reasons)
}

func allPromptRules() []PromptRule {
	return append(WorkRules(465, "docs"), GitLawRules(465, "docs", "main")...)
}

func TestRetiredThoughtCheckRuleIsAbsentAndRemainingRulesStayOrdered(t *testing.T) {
	wantIDs := []string{
		"lane-lease",
		"refusal-taxonomy",
		"smallest-change",
		"checkpoint-commit",
		"gate-before-done",
		"proof-by-default",
		"browser-display",
		"no-delete",
		"honest-bail",
	}
	rules := WorkRules(9568, "issuecheck")
	if len(rules) != len(wantIDs) {
		t.Fatalf("work rule count = %d, want %d: %#v", len(rules), len(wantIDs), rules)
	}
	for i, rule := range rules {
		if rule.ID != wantIDs[i] {
			t.Errorf("work rule %d id = %q, want %q", i, rule.ID, wantIDs[i])
		}
		for field, value := range map[string]string{
			"id":         rule.ID,
			"imperative": rule.Imperative,
			"witness":    rule.Witness,
		} {
			if strings.Contains(value, "thought-check") {
				t.Errorf("work rule %q %s still names retired thought-check surface: %q", rule.ID, field, value)
			}
		}
	}

	prompt := RenderIssuePrompt(IssuePromptInput{
		Number: 9568, Title: "dispatch prompt", Body: "Keep the live rules intact.",
		Lane: "issuecheck", Workspace: "C:/work/fak",
	})
	for _, id := range wantIDs {
		if !strings.Contains(prompt, "- "+id+":") {
			t.Errorf("rendered prompt missing surviving work rule %q", id)
		}
	}
	for _, stale := range []string{"thought-check", "top-five-thought-check", "fak-issuecheck"} {
		if strings.Contains(prompt, stale) {
			t.Errorf("rendered prompt still names retired surface %q", stale)
		}
	}
}

// Every rule is well-formed data: a stable, unique, kebab-case id, a non-empty imperative,
// and a witness. Without this, "structured" would be a shape nothing enforces.

func TestWorkRulesDefaultBrowserWorkOffDesktop(t *testing.T) {
	for _, rule := range WorkRules(8299, "internal/dispatchtick") {
		if rule.ID != "browser-display" {
			continue
		}
		for _, want := range []string{
			"off the operator desktop",
			"headless browser mode and captured render/screenshot artifacts by default",
			"do not launch or reuse visible Chrome or Edge windows",
			"unless this issue explicitly requires an attended visual witness",
		} {
			if !strings.Contains(rule.Imperative, want) {
				t.Fatalf("browser display rule missing %q: %s", want, rule.Imperative)
			}
		}
		return
	}
	t.Fatal("browser-display rule missing")
}
func TestHonestBailRequiresDurableDeliverable(t *testing.T) {
	var imperative string
	for _, rule := range WorkRules(6574, "tools") {
		if rule.ID == "honest-bail" {
			imperative = rule.Imperative
			break
		}
	}
	for _, want := range []string{
		"durable handoff",
		"gh issue comment <N> --body",
		"final chat report alone is not durable",
		"ignored scratch is not a deliverable",
	} {
		if !strings.Contains(imperative, want) {
			t.Fatalf("honest-bail imperative = %q, want %q", imperative, want)
		}
	}
}

func TestPromptRulesAreWellFormedData(t *testing.T) {
	idRe := regexp.MustCompile(`^[a-z][a-z0-9-]*$`)
	seen := map[string]bool{}
	for _, r := range allPromptRules() {
		if !idRe.MatchString(r.ID) {
			t.Errorf("rule id %q is not kebab-case", r.ID)
		}
		if seen[r.ID] {
			t.Errorf("duplicate rule id %q", r.ID)
		}
		seen[r.ID] = true
		if strings.TrimSpace(r.Imperative) == "" {
			t.Errorf("rule %q has an empty imperative", r.ID)
		}
		if strings.TrimSpace(r.Witness) == "" {
			t.Errorf("rule %q has no witness - an unwitnessed rule is the prose this replaces", r.ID)
		}
	}
	if len(seen) == 0 {
		t.Fatal("no rules at all")
	}
}

// THE gate (#3220): every rule's witness resolves against the live registries. A witness is
// either an UPPER_SNAKE refusal token some reason registry declares, or a command whose
// `fak <verb>` head the live catalog routes. Anything else is unfalsifiable prose.
func TestEveryPromptRuleWitnessResolvesLive(t *testing.T) {
	known := liveKnown(t)
	for _, r := range allPromptRules() {
		w := strings.TrimSpace(r.Witness)
		switch {
		case upperTokenRe.MatchString(w):
			if !known.Reasons[w] {
				t.Errorf("rule %q cites refusal token %q that NO live fak reason registry "+
					"declares (dos.toml / safecommit / pythongate) - retire or repoint the rule",
					r.ID, w)
			}
		case strings.HasPrefix(w, "fak "):
			verb := strings.Fields(w)[1]
			if !known.Verbs[strings.ToLower(verb)] {
				t.Errorf("rule %q cites `fak %s`, which the live cmd/fak dispatch switch does "+
					"not route - the verb was renamed or removed", r.ID, verb)
			}
		case strings.HasPrefix(w, "dos "):
			// A dos-kernel verb: promptlint's extractors are fak-scoped, so assert only the
			// shape here (a real subcommand token), never a fabricated pass on the dos CLI.
			if len(strings.Fields(w)) < 2 {
				t.Errorf("rule %q witness %q names `dos` with no subcommand", r.ID, w)
			}
		default:
			t.Errorf("rule %q witness %q is neither an UPPER_SNAKE refusal token nor a "+
				"`fak`/`dos` command - promptlint cannot keep it honest", r.ID, w)
		}
	}
}

// The witness of a token-shaped rule must survive promptlint's own extractor: a token the
// linter cannot SEE in the rendered packet is not actually watched, however real it is.
func TestPromptRuleWitnessesAreExtractableFromTheRenderedPrompt(t *testing.T) {
	in := sampleIssuePrompt()
	rendered := RenderIssuePrompt(in)
	seen := map[string]bool{}
	for _, m := range promptlint.ExtractRefusalTokens(rendered) {
		seen[m.Token] = true
	}
	for _, r := range append(WorkRules(in.Number, in.Lane), GitLawRules(in.Number, in.Lane, "main")...) {
		if upperTokenRe.MatchString(r.Witness) && !seen[r.Witness] {
			t.Errorf("rule %q witness %q is not extractable from the rendered prompt - the "+
				"freshness monitor would never check it", r.ID, r.Witness)
		}
	}
}

// The whole rendered packet is lint-clean against the live registries — not just the rule
// witnesses. This is what caught the prompt's stale `REASON_NEW_PYTHON_TOOL` citation (the
// Go constant NAME; the token pythongate stamps is NEW_PYTHON_TOOL).
//
// The issue body is supplied as controlled text: a live body is untrusted third-party data
// that may legitimately quote a token fak does not declare, so the gate covers the
// GUIDANCE the renderer authors, which is the only part it can fix.
func TestRenderedPromptIsFreshAgainstLiveRegistries(t *testing.T) {
	known := liveKnown(t)
	for _, lane := range []string{"docs", "tools", "gateway", "cmd", "abi"} {
		in := sampleIssuePrompt()
		in.Lane = lane
		in.Body = "A plain issue body with no refusal tokens in it."
		in.HostOS = "windows" // include the PowerShell block in the linted surface
		if findings := promptlint.Lint(RenderIssuePrompt(in), known); len(findings) != 0 {
			t.Errorf("lane %q: rendered prompt is stale: %+v", lane, findings)
		}
	}
}

// The renderer emits every rule in the one uniform bulleted shape, and emits ALL of them —
// a rule silently dropped from the render is a rule nothing enforces.
func TestRenderPromptRulesEmitsUniformBullets(t *testing.T) {
	rules := GitLawRules(465, "docs", "main")
	block := RenderPromptRules("git laws:", rules)
	lines := strings.Split(block, "\n")
	if lines[0] != "git laws:" {
		t.Fatalf("header = %q", lines[0])
	}
	if got, want := len(lines)-1, len(rules); got != want {
		t.Fatalf("rendered %d bullets for %d rules:\n%s", got, want, block)
	}
	for i, r := range rules {
		want := "- " + r.ID + ": " + r.Imperative + " - witness `" + r.Witness + "`"
		if lines[i+1] != want {
			t.Errorf("bullet %d:\n got %q\nwant %q", i, lines[i+1], want)
		}
	}
}

// pyRuleIDRe matches one rule-table entry head in the Python renderer's mirrored tables:
// the id string literal that opens each `(id, imperative, witness)` tuple.
var pyRuleIDRe = regexp.MustCompile(`(?m)^\s*\("([a-z][a-z0-9-]*)",\s*$`)

// The Python renderer mirrors THIS package's rule set (#3220's "apply the same structure
// there, or generate it from the shared Go spec"). promptrules.go is the spec; the Python
// table restates it, and this test is what stops the two renderers from drifting apart in
// silence — the failure mode that motivated the issue in the first place.
//
// What it guarantees, exactly: the rule ID SETS are equal, and each rule's witness is
// named on the Python side (whole token for a refusal token; the `<tool> <verb>` head for
// a command witness, because the tail interpolates the lane). It does NOT compare the
// imperative wording — the Python source wraps its strings, so a byte comparison there
// would fail on line breaks rather than on meaning.
//
// It SKIPS when the Python renderer is gone: retiring it (the issue's "#B") is a planned
// outcome, and a parity test that reds on a successful retirement would be a false alarm.
func TestPythonRendererMirrorsTheGoRuleSpec(t *testing.T) {
	path := filepath.Join(repoRootForRules(t), "tools", "issue_worker_prompt.py")
	src, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		t.Skip("tools/issue_worker_prompt.py is retired - nothing to keep in parity")
	}
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	// Scope the id scan to the mirrored tables so unrelated tuples elsewhere in the
	// module cannot masquerade as rules.
	text := string(src)
	start := strings.Index(text, "def _work_rules(")
	end := strings.Index(text, "def render_prompt(")
	if start < 0 || end < 0 || end < start {
		t.Fatalf("cannot locate the Python rule tables in %s - the mirror moved", path)
	}
	tables := text[start:end]

	pyIDs := map[string]bool{}
	for _, m := range pyRuleIDRe.FindAllStringSubmatch(tables, -1) {
		pyIDs[m[1]] = true
	}
	goIDs := map[string]bool{}
	for _, r := range allPromptRules() {
		goIDs[r.ID] = true
		if !pyIDs[r.ID] {
			t.Errorf("rule %q exists in the Go spec but NOT in the Python renderer - the "+
				"two worker prompts now state different laws", r.ID)
		}
		// The witness must be named on the Python side too, else the mirrored rule is
		// pointing at a different gate than the one this package vouches for.
		want := r.Witness
		if f := strings.Fields(want); !upperTokenRe.MatchString(want) && len(f) >= 2 {
			want = f[0] + " " + f[1]
		}
		if !strings.Contains(tables, want) {
			t.Errorf("rule %q cites witness %q, which the Python rule tables never name", r.ID, want)
		}
	}
	for id := range pyIDs {
		if !goIDs[id] {
			t.Errorf("the Python renderer states rule %q that the Go spec does not - the "+
				"spec is promptrules.go; add it here or drop it there", id)
		}
	}
}

func TestPythonRendererCarriesNoRetiredThoughtCheckRule(t *testing.T) {
	path := filepath.Join(repoRootForRules(t), "tools", "issue_worker_prompt.py")
	src, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		t.Skip("tools/issue_worker_prompt.py is retired - nothing to keep in parity")
	}
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	text := string(src)
	start := strings.Index(text, "def _work_rules(")
	end := strings.Index(text, "def _git_law_rules(")
	if start < 0 || end < 0 || end < start {
		t.Fatalf("cannot locate Python work-rule table in %s", path)
	}
	workRules := text[start:end]
	for _, stale := range []string{
		"top-five-thought-check",
		"fak thought-check",
		"fak-issuecheck",
	} {
		if strings.Contains(workRules, stale) {
			t.Errorf("Python work-rule mirror still names retired surface %q", stale)
		}
	}
}

// The prose the rules replaced must be gone: the old free-paragraph guidance is what made
// the block unscannable and unobservable, so its distinctive phrasing must not reappear.
func TestGuidanceBlocksCarryNoUnwitnessedProse(t *testing.T) {
	p := RenderIssuePrompt(sampleIssuePrompt())
	for _, stale := range []string{
		"Take the lane lease first if siblings may collide",
		"If the issue is a docs/observability/test ask, that is often a single file.",
		"the verb-led subject + `(fak docs)` trailer is what makes",
		"Never branch / new-worktree (the OFF_TRUNK guard refuses it).",
	} {
		if strings.Contains(p, stale) {
			t.Errorf("prompt still carries the pre-#3220 prose %q", stale)
		}
	}
}
