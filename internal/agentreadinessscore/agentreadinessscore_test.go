package agentreadinessscore

// Golden-parity port of tools/agent_readiness_scorecard_test.py. Every case here mirrors a
// deterministic Python test one-for-one — same inputs, same expected scores/defect-counts/
// groups — so a divergence in the Go pure core reds the same way the Python reference would.
// The two Python live sentinels that assert a ZERO-friction-debt tree are intentionally NOT
// ported as hard gates: they encode a tree-state assertion (agent-readiness of the CURRENT
// tree) that is a separate regression signal, not a property of this port. The tolerant
// TestLivePayloadIsWellFormed covers the live path's shape.

import (
	"encoding/json"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// --- fixtures (ported verbatim from the Python module) ----------------------

var goodAgents = strings.Join([]string{
	"# AGENTS.md",
	"## What this project is",
	"**fak** is an agent kernel.",
	"```bash",
	"go build ./cmd/fak",
	"make test",
	"go run ./cmd/fak preflight --policy p.json --tool t --args \"{}\"",
	"```",
	"Work on the trunk; the trunk guard refuses OFF_TRUNK commits.",
	"Commit by explicit path (`git commit -- <paths>`), never `git add -A`.",
	"Sign off with `git commit -s` (DCO).",
	"Each claim in CLAIMS.md carries a tag. Add a feature as a leaf via fak new-leaf.",
	"Writes outside the repo are refused by the repo-guard (OUT_OF_TREE_WRITE).",
	"See CONTRIBUTING.md. Green = `make ci`.",
}, "\n") + "\n"

var goodCodex = strings.Join([]string{
	"# fak + OpenAI Codex",
	"Codex is OpenAI's coding agent. Current Codex surfaces include the CLI, IDE",
	"extension, Codex app, and cloud tasks.",
	"Codex reads AGENTS.md before it works in this repo.",
	"```bash",
	"codex mcp add fak -- ./fak serve --stdio --policy examples/dev-agent-policy.json",
	"codex exec --json \"Summarize AGENTS.md\"",
	"export OPENAI_BASE_URL=\"http://127.0.0.1:8080/v1\"",
	"```",
	"Responses clients use /v1/responses; fak's current client-facing OpenAI-compatible",
	"surface is Chat Completions, so current Codex users should use MCP first.",
}, "\n") + "\n"

// A main()-shaped switch: only the func main() body's top-level verbs count; the cmdPolicy
// sub-switch must NOT leak in.
const mainGoFixture = `package main

func main() {
	switch os.Args[1] {
	case "serve":
		cmdServe(os.Args[2:])
	case "hook":
		cmdHook()
	case "hooks":
		cmdHooks(os.Args[2:])
	case "version", "-v", "--version":
		fmt.Println(v)
	case guard.TrampolineVerb:
		guard.LandlockTrampoline(os.Args[2:])
	default:
		usage()
	}
}

func cmdPolicy(argv []string) {
	switch argv[0] {
	case "budget":
		runBudget(argv[1:])
	}
}
`

// --- small assertion helpers ------------------------------------------------

func anyContains(ss []string, sub string) bool {
	for _, s := range ss {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

func contains(ss []string, s string) bool {
	for _, x := range ss {
		if x == s {
			return true
		}
	}
	return false
}

// sameSet reports whether a and b hold the same elements (order-independent, multiplicity-aware).
func sameSet(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	m := map[string]int{}
	for _, x := range a {
		m[x]++
	}
	for _, x := range b {
		m[x]--
	}
	for _, v := range m {
		if v != 0 {
			return false
		}
	}
	return true
}

func mustWrite(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func copyFacts(m map[string]int) map[string]int {
	out := make(map[string]int, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// cleanKPIs is every KPI in its zero-defect state — the all-green tree (mirrors _clean_kpis).
func cleanKPIs() []KPI {
	return []KPI{
		kpiAgentsEntrypoint(goodAgents, true),
		kpiAgentConfig(nil),
		kpiAgentConfigValid(nil),
		kpiLLMSMap(true, true),
		kpiIdentityStatement(identityDocs, nil),
		kpiEntryLinksResolve(nil),
		kpiRecipeLinksResolve(nil),
		kpiFirstCommand(true, "AGENTS.md"),
		kpiCommandVerbsResolve(nil),
		kpiFirstCommandRuns(true, true, "examples/p.json", false),
		kpiInstallOneliner(true, "AGENTS.md"),
		kpiHonestyLedger(true, nil),
		kpiIntegrationRecipes(nil),
		kpiCodexRecipeCurrent(nil),
		kpiFencedPathsResolve(nil),
		kpiExtensionScaffold(true, true),
		kpiGuardrailsSurfaced(nil),
		kpiContributorContract(true, true, true),
		kpiPlatformGuidanceConsistent(true, true),
		kpiMachineConsumable(8, 8, nil),
		kpiRefusalRecoveryMapped(nil, 6),
		kpiQuickstartSuccessSignal(true, true),
		kpiToolchainPinned(true, true),
	}
}

// baselineMap round-trips a Payload through JSON exactly as the CLI's --compare read does,
// so corpus numbers arrive as float64 (the shape Compare must tolerate).
func baselineMap(t *testing.T, p Payload) map[string]any {
	t.Helper()
	b, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal baseline: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("unmarshal baseline: %v", err)
	}
	return m
}

// --- the small helpers ------------------------------------------------------

func TestGradeLetterBands(t *testing.T) {
	for _, c := range []struct {
		in   float64
		want string
	}{{100, "A"}, {90, "A"}, {85, "B"}, {72, "C"}, {61, "D"}, {40, "F"}} {
		if got := scorecard.GradeStd(c.in); got != c.want {
			t.Errorf("gradeLetter(%v) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestUntaggedClaimsCountsTags(t *testing.T) {
	text := "- [SHIPPED] real thing\n" +
		"- [SIMULATED] [STUB] two tags is malformed\n" +
		"- [TODO] a bracketed claim with no status tag\n" +
		"- a plain bullet (not a `- [` claim line) is not graded\n" +
		"  - [STUB] indented claim is fine\n" +
		"not a claim line at all\n"
	bad := untaggedClaims(text, true)
	if len(bad) != 2 {
		t.Fatalf("untaggedClaims returned %d bad, want 2: %v", len(bad), bad)
	}
	if !anyContains(bad, "2 status tag") || !anyContains(bad, "0 status tag") {
		t.Errorf("untaggedClaims missing expected tag-count messages: %v", bad)
	}
	if got := untaggedClaims("- [SHIPPED] all good\n- [STUB] also good", true); len(got) != 0 {
		t.Errorf("clean claims => %v, want empty", got)
	}
	if got := untaggedClaims(text, false); len(got) != 0 {
		t.Errorf("absent ledger must abstain, got %v", got)
	}
}

func TestFindFirstCommandRequiresFence(t *testing.T) {
	if ok, _ := findFirstCommand(map[string]string{"README.md": "Run fak preflight to see a denial."}); ok {
		t.Error("a prose token must not count as a first command")
	}
	ok, where := findFirstCommand(map[string]string{"README.md": "Try it:\n```\nfak preflight --policy p.json\n```\n"})
	if !ok || where != "README.md" {
		t.Errorf("findFirstCommand(fenced) = (%v, %q), want (true, README.md)", ok, where)
	}
}

func TestFindInstallOnelinerNeedsBothTokens(t *testing.T) {
	if ok, _ := findInstallOneliner(map[string]string{"README.md": "go install foo/cmd/fak@latest"}); !ok {
		t.Error("go install …@latest is the resolvable one-liner")
	}
	if ok, _ := findInstallOneliner(map[string]string{"README.md": "go install ./cmd/fak"}); ok {
		t.Error("go install without @latest is not the one-liner")
	}
}

func TestFindIdentityRequiredInEveryDoc(t *testing.T) {
	every := map[string]string{
		"AGENTS.md": "**fak** is an agent kernel.",
		llmsFile:    "`fak` is an agent kernel.",
		"README.md": "`fak` is one Go binary you put in front of the agent.",
	}
	present, missing := findIdentity(every)
	if len(missing) != 0 {
		t.Errorf("identity present in all docs => missing %v, want none", missing)
	}
	if !sameSet(present, identityDocs) {
		t.Errorf("present = %v, want %v", present, identityDocs)
	}
	// A match buried past the head window does not count.
	buried := map[string]string{"AGENTS.md": every["AGENTS.md"], llmsFile: every[llmsFile],
		"README.md": strings.Repeat("\n", 60) + "fak is one Go binary"}
	if _, m := findIdentity(buried); !reflect.DeepEqual(m, []string{"README.md"}) {
		t.Errorf("buried identity => missing %v, want [README.md]", m)
	}
	// No identity anywhere => all three orientation docs are missing it.
	if _, m := findIdentity(map[string]string{}); !sameSet(m, identityDocs) {
		t.Errorf("empty => missing %v, want %v", m, identityDocs)
	}
}

func TestFindIdentityRecognizesExistingFrontDoorLanguage(t *testing.T) {
	texts := map[string]string{
		"AGENTS.md": "## What this project is\n\n**fak** is an *agent kernel*: one Go binary.",
		llmsFile:    "fak is an agent kernel.",
		"README.md": "# fak — configure your agents\n\nfak lets you run your agents with a small configuration chosen for this task, while one boundary manages their context, models, tools, and record.",
	}
	if present, missing := findIdentity(texts); len(missing) != 0 || !sameSet(present, identityDocs) {
		t.Fatalf("front-door identity present=%v missing=%v, want all present", present, missing)
	}

	texts["README.md"] = "# fak\n\nFast, safe, and flexible."
	if _, missing := findIdentity(texts); !reflect.DeepEqual(missing, []string{"README.md"}) {
		t.Fatalf("slogan-only README missing=%v, want [README.md]", missing)
	}
}
func TestMissingGuardrailsDetectsGaps(t *testing.T) {
	if got := missingGuardrails(goodAgents); len(got) != 0 {
		t.Errorf("GOOD_AGENTS surfaces every rule, got missing %v", got)
	}
	if got := missingGuardrails("# AGENTS.md\njust build and test, nothing about the rules"); len(got) != len(guardrailClusters) {
		t.Errorf("thin AGENTS.md => %d missing, want %d", len(got), len(guardrailClusters))
	}
}

// --- per-KPI defect triggers + clean cases ----------------------------------

func TestAgentsEntrypointMissingFileAndElements(t *testing.T) {
	k := kpiAgentsEntrypoint("", false)
	if k.Score != 0 || len(k.Defects) != 1 || !strings.Contains(k.Defects[0], "missing") {
		t.Errorf("missing AGENTS.md => %+v", k)
	}
	if k := kpiAgentsEntrypoint("**fak** is an agent kernel. No commands here.", true); len(k.Defects) != 3 {
		t.Errorf("identity but no build/test/run => %d defects, want 3", len(k.Defects))
	}
	if k := kpiAgentsEntrypoint(goodAgents, true); len(k.Defects) != 0 {
		t.Errorf("GOOD_AGENTS => %v, want no defects", k.Defects)
	}
}

func TestAgentConfigMissingAndClean(t *testing.T) {
	k := kpiAgentConfig([]string{"Cursor (AGENTS.md / .cursor/rules)"})
	if len(k.Defects) != 1 || !strings.Contains(k.Defects[0], "Cursor") {
		t.Errorf("missing Cursor => %+v", k)
	}
	if k := kpiAgentConfig(nil); len(k.Defects) != 0 || k.Score != 100 {
		t.Errorf("no missing => %+v, want clean/100", k)
	}
}

func TestLLMSMapHardAndSoft(t *testing.T) {
	missing := kpiLLMSMap(false, false)
	if len(missing.Defects) != 1 || len(missing.Soft) != 1 {
		t.Errorf("both absent => %d hard / %d soft, want 1/1", len(missing.Defects), len(missing.Soft))
	}
	clean := kpiLLMSMap(true, true)
	if len(clean.Defects) != 0 || len(clean.Soft) != 0 {
		t.Errorf("both present => %+v, want clean", clean)
	}
}

func TestIdentityStatementKpi(t *testing.T) {
	one := kpiIdentityStatement([]string{agentsFile, llmsFile}, []string{"README.md"})
	if len(one.Defects) != 1 || one.Score >= 100 || !strings.Contains(one.Defects[0], "README.md") {
		t.Errorf("one missing => %+v", one)
	}
	if all := kpiIdentityStatement(identityDocs, nil); len(all.Defects) != 0 || all.Score != 100 {
		t.Errorf("all present => %+v, want clean/100", all)
	}
}

func TestEntryLinksResolveKpi(t *testing.T) {
	k := kpiEntryLinksResolve([]string{"AGENTS.md -> docs/gone.md", "AGENTS.md -> x.md"})
	if len(k.Defects) != 2 {
		t.Errorf("two dead links => %d defects, want 2", len(k.Defects))
	}
	if k := kpiEntryLinksResolve(nil); len(k.Defects) != 0 {
		t.Errorf("no dead links => %v", k.Defects)
	}
}

func TestFirstCommandKpi(t *testing.T) {
	if k := kpiFirstCommand(false, ""); k.Score != 20 {
		t.Errorf("no first command => score %d, want 20", k.Score)
	}
	if k := kpiFirstCommand(true, "AGENTS.md"); len(k.Defects) != 0 {
		t.Errorf("present => %v", k.Defects)
	}
}

func TestInstallOnelinerKpi(t *testing.T) {
	if k := kpiInstallOneliner(false, ""); len(k.Defects) == 0 {
		t.Error("absent one-liner => want a defect")
	}
	if k := kpiInstallOneliner(true, "README.md"); len(k.Defects) != 0 {
		t.Errorf("present => %v", k.Defects)
	}
}

func TestHonestyLedgerKpi(t *testing.T) {
	if k := kpiHonestyLedger(false, nil); k.Score != 0 {
		t.Errorf("no ledger => score %d, want 0", k.Score)
	}
	untagged := kpiHonestyLedger(true, []string{"CLAIMS.md:5: 0 status tag(s): - foo"})
	if len(untagged.Defects) != 1 || untagged.Score >= 100 {
		t.Errorf("untagged => %+v", untagged)
	}
	if k := kpiHonestyLedger(true, nil); len(k.Defects) != 0 {
		t.Errorf("clean ledger => %v", k.Defects)
	}
}

func TestIntegrationRecipesKpi(t *testing.T) {
	k := kpiIntegrationRecipes([]string{"Cursor", "MCP client"})
	if len(k.Defects) != 2 {
		t.Errorf("two missing recipes => %d defects, want 2", len(k.Defects))
	}
	if k := kpiIntegrationRecipes(nil); k.Score != 100 {
		t.Errorf("none missing => score %d, want 100", k.Score)
	}
}

func TestCodexRecipeCurrentnessKpi(t *testing.T) {
	if gaps := codexRecipeGaps(goodCodex, true); len(gaps) != 0 {
		t.Errorf("GOOD_CODEX => gaps %v, want none", gaps)
	}
	clean := kpiCodexRecipeCurrent(nil)
	if clean.Score != 100 || len(clean.Defects) != 0 || clean.Group != "adopt" {
		t.Errorf("clean codex => %+v", clean)
	}
	stale := "OpenAI has deprecated the standalone Codex API. Use gpt-4-turbo through a generic SDK."
	gaps := codexRecipeGaps(stale, true)
	if !anyContains(gaps, "MCP server path") || !anyContains(gaps, "stale Codex-era copy") {
		t.Errorf("stale codex gaps missing expected messages: %v", gaps)
	}
	if k := kpiCodexRecipeCurrent(gaps); len(k.Defects) != len(gaps) || k.Score >= 100 {
		t.Errorf("stale KPI => %+v (gaps=%d)", k, len(gaps))
	}
}

func TestExtensionScaffoldKpi(t *testing.T) {
	if k := kpiExtensionScaffold(false, false); len(k.Defects) != 2 {
		t.Errorf("both absent => %d defects, want 2", len(k.Defects))
	}
	if k := kpiExtensionScaffold(true, true); len(k.Defects) != 0 {
		t.Errorf("both present => %v", k.Defects)
	}
}

func TestGuardrailsSurfacedKpi(t *testing.T) {
	k := kpiGuardrailsSurfaced([]string{"DCO sign-off"})
	if len(k.Defects) != 1 || k.Score >= 100 {
		t.Errorf("one missing rule => %+v", k)
	}
	if k := kpiGuardrailsSurfaced(nil); k.Score != 100 {
		t.Errorf("none missing => score %d, want 100", k.Score)
	}
}

func TestContributorContractKpi(t *testing.T) {
	k := kpiContributorContract(true, false, false)
	if len(k.Defects) != 2 {
		t.Errorf("present but unlinked + no green gate => %d defects, want 2", len(k.Defects))
	}
	if k := kpiContributorContract(true, true, true); len(k.Defects) != 0 {
		t.Errorf("all present => %v", k.Defects)
	}
}

func TestMachineConsumableIsSoft(t *testing.T) {
	k := kpiMachineConsumable(6, 8, []string{"tools/x_scorecard.py", "tools/y_scorecard.py"})
	if len(k.Defects) != 0 {
		t.Errorf("machine_consumable is SOFT, must never emit hard debt: %v", k.Defects)
	}
	if k.Score != 75 || len(k.Soft) != 2 {
		t.Errorf("6/8 => score %d, soft %d; want 75, 2", k.Score, len(k.Soft))
	}
}

// --- paste-and-run success KPIs ---------------------------------------------

func TestPathOperandsExtractsPathsSkipsNoise(t *testing.T) {
	block := "cd fleet/fak\n" +
		"go build ./cmd/fak\n" +
		"go run ./cmd/fak preflight --policy examples/p.json\n" +
		"curl https://example.com/x\n" +
		"  \"method\": \"tools/call\",\n" +
		"\"self_modify_globs\": [\"internal/\"]\n" +
		"make ci   # Windows: scripts/ci.ps1)\n"
	ops := pathOperands(block)
	for _, want := range []string{"fleet/fak", "./cmd/fak", "examples/p.json"} {
		if !contains(ops, want) {
			t.Errorf("pathOperands missing %q: %v", want, ops)
		}
	}
	if contains(ops, "https://example.com/x") || contains(ops, "tools/call") ||
		anyContains(ops, "internal") || anyContains(ops, "ci.ps1") {
		t.Errorf("pathOperands leaked noise: %v", ops)
	}
}

func TestTemplateSlotsAreNotPaths(t *testing.T) {
	if !isTemplateSlot("<your-policy>") || !isTemplateSlot("YOUR_ENV_VAR") {
		t.Error("template slots must be recognized")
	}
	if isTemplateSlot("examples/p.json") {
		t.Error("a real path is not a template slot")
	}
	if got := pathOperands("cd <your-clone>"); len(got) != 0 {
		t.Errorf("a bracketed cd slot is an adapt-me marker, not a path: %v", got)
	}
}

func TestFencedPathsResolveKpi(t *testing.T) {
	clean := kpiFencedPathsResolve(nil)
	if len(clean.Defects) != 0 || clean.Score != 100 || clean.Group != "adopt" {
		t.Errorf("clean => %+v", clean)
	}
	bad := kpiFencedPathsResolve([]string{
		"docs/integrations/cursor.md: `fleet/fak` — stale private-monorepo path",
		"docs/integrations/cursor.md: `/path/to/…` placeholder in a runnable command",
	})
	if len(bad.Defects) != 2 || bad.Score != 80 {
		t.Errorf("two bad paths => %+v, want 2 defects / score 80", bad)
	}
}

func TestFirstCommandRunsKpi(t *testing.T) {
	ok := kpiFirstCommandRuns(true, true, "examples/p.json", false)
	if len(ok.Defects) != 0 || ok.Score != 100 || ok.Group != "adopt" {
		t.Errorf("clean => %+v", ok)
	}
	miss := kpiFirstCommandRuns(true, false, "examples/gone.json", false)
	if len(miss.Defects) != 1 || !strings.Contains(miss.Defects[0], "doesn't exist") {
		t.Errorf("missing policy => %+v", miss)
	}
	keyed := kpiFirstCommandRuns(true, true, "examples/p.json", true)
	if len(keyed.Defects) != 1 || !strings.Contains(keyed.Defects[0], "key") {
		t.Errorf("secret key => %+v", keyed)
	}
	absent := kpiFirstCommandRuns(false, true, "", false)
	if len(absent.Defects) != 0 || absent.Score != 100 {
		t.Errorf("no first command => abstain, got %+v", absent)
	}
}

func TestPlatformGuidanceConsistentKpi(t *testing.T) {
	if k := kpiPlatformGuidanceConsistent(true, true); len(k.Defects) != 0 {
		t.Errorf("sells make ci + names bridge => %v", k.Defects)
	}
	bad := kpiPlatformGuidanceConsistent(true, false)
	if len(bad.Defects) != 1 || bad.Score != 40 || bad.Group != "build" {
		t.Errorf("make ci, no bridge => %+v", bad)
	}
	if k := kpiPlatformGuidanceConsistent(false, false); len(k.Defects) != 0 {
		t.Errorf("no make ci => nothing to reconcile, got %v", k.Defects)
	}
}

// --- executable-truth KPIs --------------------------------------------------

func TestDispatchVerbsParsesOnlyMainSwitch(t *testing.T) {
	verbs := dispatchVerbs(mainGoFixture)
	for _, v := range []string{"serve", "hook", "hooks", "version", "-v", "--version"} {
		if !verbs[v] {
			t.Errorf("dispatchVerbs missing top-level verb %q: %v", v, verbs)
		}
	}
	if verbs["budget"] {
		t.Error("cmdPolicy sub-switch verb budget must not leak")
	}
	if verbs["TrampolineVerb"] {
		t.Error("a non-string case contributes nothing")
	}
	if got := dispatchVerbs(""); len(got) != 0 {
		t.Errorf("absent main.go => empty set, got %v", got)
	}
	if got := dispatchVerbs("package main\n// no func main here\n"); len(got) != 0 {
		t.Errorf("no func main => empty set, got %v", got)
	}
}

// The real cmd/fak/main.go splits its routing table across several helpers that switch
// on the same top-level verb name. Verbs routed there resolve at runtime just as much as
// main()'s own cases, while subcommand switches such as cmdPolicy must stay excluded.
func TestDispatchVerbsParsesPrimaryVerbSplit(t *testing.T) {
	const fixture = `package main

func main() {
	if dispatchPrimaryVerb(os.Args[1], os.Args[2:], start, &verb) {
		return
	}
	switch os.Args[1] {
	case "serve":
		cmdServe(os.Args[2:])
	default:
		usage()
	}
}

func dispatchCoreVerbA(name string, args []string) bool {
	switch name {
	case "agent":
		cmdAgent(args)
	default:
		return false
	}
	return true
}

func dispatchCoreVerbB(name string, args []string) bool {
	switch name {
	case "hooks":
		cmdHooks(args)
	default:
		return false
	}
	return true
}

func dispatchExtendedVerbA(name string, args []string) bool {
	switch name {
	case "guard":
		cmdGuard(args)
	default:
		return false
	}
	return true
}

func dispatchExtendedVerbB(name string, args []string) bool {
	switch name {
	case "score":
		cmdScore(args)
	default:
		return false
	}
	return true
}

func dispatchPrimaryVerb(name string, args []string, start time.Time, verb *string) bool {
	switch name {
	case "preflight":
		cmdPreflight(args)
	case "commit":
		os.Exit(runObservedGitOperation(start, *verb, args, func() int {
			return runCommitCommand(os.Stdout, os.Stderr, args)
		}))
	case "worktree":
		cmdWorktreeVerb(args)
	default:
		return false
	}
	return true
}

func cmdPolicy(argv []string) {
	switch argv[0] {
	case "budget":
		runBudget(argv[1:])
	}
}
`
	verbs := dispatchVerbs(fixture)
	for _, v := range []string{"serve", "agent", "hooks", "guard", "score", "preflight", "commit", "worktree"} {
		if !verbs[v] {
			t.Errorf("dispatchVerbs missing routed verb %q: %v", v, verbs)
		}
	}
	if verbs["budget"] {
		t.Error("cmdPolicy sub-switch verb budget must not leak in via the second dispatcher")
	}
}

func TestCommandVerbsExtractsCommandContextOnly(t *testing.T) {
	fenced := "```bash\n" +
		"go run ./cmd/fak preflight --policy p.json\n" +
		"cd repo && fak serve --stdio\n" +
		"# fak governs the call   <- a comment, not a command\n" +
		"```\n"
	got := commandVerbs(fenced)
	if !contains(got, "preflight") || !contains(got, "serve") || contains(got, "governs") {
		t.Errorf("fenced => %v", got)
	}
	prose := "Run it via `fak hooks pre-commit` to gate the commit. " +
		"fak governs the call (prose, no backticks). " +
		"In Cursor: `@fak please adjudicate` is a tool mention, not a CLI call."
	got = commandVerbs(prose)
	if !contains(got, "hooks") || contains(got, "governs") || contains(got, "please") {
		t.Errorf("prose => %v", got)
	}
	desync := "```bash\nfak serve\n```\n\nThen `fak hooks pre-commit` runs the gates.\n"
	if !contains(commandVerbs(desync), "hooks") {
		t.Errorf("inline span after a fence must still pair: %v", commandVerbs(desync))
	}
}

func TestCommandVerbsHardeningAgainstAuditVectors(t *testing.T) {
	banner := "```\nfak guard: 131 kernel decision(s) — 121 allowed\nfak summary: done\n```\n"
	if got := commandVerbs(banner); contains(got, "guard") || contains(got, "summary") {
		t.Errorf("output banner is not a command: %v", got)
	}
	if got := commandVerbs("```bash\nFAK_AUDIT_JOURNAL=x.jsonl fak badverb --flag\n```\n"); !contains(got, "badverb") {
		t.Errorf("env-var prefix must not hide the verb: %v", got)
	}
	wrapped := "```bash\ncodex mcp add fak -- ./fak serve --stdio\nfak guard -- claude\n```\n"
	if got := commandVerbs(wrapped); !contains(got, "serve") || contains(got, "claude") {
		t.Errorf("wrapper boundary handling => %v", got)
	}
	win := "```powershell\n.\\fak serve --policy p.json\nfak.exe preflight --tool t\n```\n"
	if got := commandVerbs(win); !contains(got, "serve") || !contains(got, "preflight") {
		t.Errorf("windows forms => %v", got)
	}
	if got := commandVerbs("```\nfak serve   # then fak foo later\n```\n"); !reflect.DeepEqual(got, []string{"serve"}) {
		t.Errorf("trailing comment must not leak: %v", got)
	}
}

func TestCommandVerbsResolveKpi(t *testing.T) {
	clean := kpiCommandVerbsResolve(nil)
	if len(clean.Defects) != 0 || clean.Score != 100 || clean.Group != "adopt" {
		t.Errorf("clean => %+v", clean)
	}
	bad := kpiCommandVerbsResolve([]string{"AGENTS.md: fak hooks", "README.md: fak frobnicate"})
	if len(bad.Defects) != 2 || bad.Score != 60 {
		t.Errorf("two unknown verbs => %+v, want 2 defects / score 60", bad)
	}
}

func TestUnknownCommandVerbsRecognizesFamiliesAndMovedCommands(t *testing.T) {
	docs := map[string]string{"AGENTS.md": "`fak server status`\n`fak plan-audit`\n`fak frobnicate`"}
	got := unknownCommandVerbs(docs, map[string]bool{"serve": true})
	if !reflect.DeepEqual(got, []string{"AGENTS.md: fak frobnicate"}) {
		t.Fatalf("unknown commands=%v, want only genuinely unknown verb", got)
	}
}
func TestUnknownCommandVerbsAbstainsWithoutDispatch(t *testing.T) {
	docs := map[string]string{"AGENTS.md": "Run `fak hooks pre-commit`."}
	if got := unknownCommandVerbs(docs, map[string]bool{}); len(got) != 0 {
		t.Errorf("no dispatch set => abstain, got %v", got)
	}
	out := unknownCommandVerbs(docs, map[string]bool{"hook": true, "serve": true})
	if !reflect.DeepEqual(out, []string{"AGENTS.md: fak hooks"}) {
		t.Errorf("unknown verb => %v, want [AGENTS.md: fak hooks]", out)
	}
	if got := unknownCommandVerbs(map[string]string{"AGENTS.md": "`fak serve`"}, map[string]bool{"serve": true}); len(got) != 0 {
		t.Errorf("dispatched verb not flagged, got %v", got)
	}
}

func TestRecipeLinksResolveKpi(t *testing.T) {
	clean := kpiRecipeLinksResolve(nil)
	if len(clean.Defects) != 0 || clean.Score != 100 || clean.Group != "discover" {
		t.Errorf("clean => %+v", clean)
	}
	bad := kpiRecipeLinksResolve([]string{
		"docs/integrations/cursor.md -> ../gone.md",
		"docs/integrations/claude.md -> missing.md",
	})
	if len(bad.Defects) != 2 || bad.Score != 76 {
		t.Errorf("two dead recipe links => %+v, want 2 defects / score 76", bad)
	}
}

func TestAgentConfigValidKpi(t *testing.T) {
	clean := kpiAgentConfigValid(nil)
	if len(clean.Defects) != 0 || clean.Score != 100 || clean.Group != "discover" {
		t.Errorf("clean => %+v", clean)
	}
	bad := kpiAgentConfigValid([]string{".mcp.json server 'x' names no launch command"})
	if len(bad.Defects) != 1 || bad.Score != 66 {
		t.Errorf("one bad server => %+v, want 1 defect / score 66", bad)
	}
}

func TestAgentConfigIntegrityReadsMcpJson(t *testing.T) {
	root := t.TempDir()
	if got := agentConfigIntegrity(root); len(got) != 0 {
		t.Errorf("absent .mcp.json => no integrity defect, got %v", got)
	}
	write := func(s string) { mustWrite(t, root+"/.mcp.json", s) }
	write("{ not json")
	if !anyContains(agentConfigIntegrity(root), "does not parse") {
		t.Error("malformed JSON => want a parse defect")
	}
	write(`{"mcpServers": {"x": {"args": []}}}`)
	if !anyContains(agentConfigIntegrity(root), "neither a launch command nor a url") {
		t.Error("server with neither command nor url => want a defect")
	}
	write(`{"mcpServers": {"dos": {"command": "dos-mcp"}}}`)
	if got := agentConfigIntegrity(root); len(got) != 0 {
		t.Errorf("well-formed command server => clean, got %v", got)
	}
	write(`{"mcpServers": {"r": {"url": "https://x/sse", "type": "sse"}}}`)
	if got := agentConfigIntegrity(root); len(got) != 0 {
		t.Errorf("remote url server => clean, got %v", got)
	}
	write(`{"mcpServers": {"//": "a note", "dos": {"command": "dos-mcp"}}}`)
	if got := agentConfigIntegrity(root); len(got) != 0 {
		t.Errorf("a // comment key is not a server => clean, got %v", got)
	}
}

// --- refusal-recovery / quickstart signal / toolchain pin -------------------

func TestDosReasonTokensParsesReasonBlocks(t *testing.T) {
	toml := "[reasons.OFF_TRUNK]\nrefusal = true\n\n" +
		"[reasons.ARCH_LAYER_VIOLATION]\nsummary = \"x\"\n\n" +
		"[other.NOT_A_REASON]\nx = 1\n" +
		"  [reasons.INDENTED]\n"
	toks := dosReasonTokens(toml)
	if !reflect.DeepEqual(toks, []string{"ARCH_LAYER_VIOLATION", "OFF_TRUNK"}) {
		t.Errorf("dosReasonTokens => %v, want sorted+deduped [ARCH_LAYER_VIOLATION OFF_TRUNK]", toks)
	}
	if len(dosReasonTokens("")) != 0 || len(dosReasonTokens("no reason blocks here")) != 0 {
		t.Error("empty/absent source => abstain, never invented debt")
	}
}

func TestUnmappedRefusalTokensRequiresNearbyRecoveryCue(t *testing.T) {
	tokens := []string{"OFF_TRUNK", "ARCH_LAYER_VIOLATION", "OUT_OF_DIRECTION"}
	recovery := "## How to recover from a refusal\n" +
		"OFF_TRUNK: commit to main instead.\n" +
		"ARCH_LAYER_VIOLATION: fix the import.\n"
	if got := unmappedRefusalTokens(tokens, recovery); !reflect.DeepEqual(got, []string{"OUT_OF_DIRECTION"}) {
		t.Errorf("=> %v, want [OUT_OF_DIRECTION]", got)
	}
	glossary := "OFF_TRUNK is a thing.\nARCH_LAYER_VIOLATION is another thing.\n"
	if got := unmappedRefusalTokens(tokens, glossary); !sameSet(got, tokens) {
		t.Errorf("a cue-less glossary maps nothing: %v", got)
	}
	if got := unmappedRefusalTokens(tokens, ""); !sameSet(got, tokens) {
		t.Errorf("no surface => all unmapped, got %v", got)
	}
	if got := unmappedRefusalTokens(nil, recovery); len(got) != 0 {
		t.Errorf("no tokens => none unmapped, got %v", got)
	}
}

func TestRefusalRecoveryMappedKpi(t *testing.T) {
	clean := kpiRefusalRecoveryMapped(nil, 6)
	if len(clean.Defects) != 0 || clean.Score != 100 || clean.Group != "build" {
		t.Errorf("clean => %+v", clean)
	}
	bad := kpiRefusalRecoveryMapped([]string{"ARCH_LAYER_VIOLATION", "OUT_OF_DIRECTION"}, 6)
	if len(bad.Defects) != 2 || bad.Score != 67 {
		t.Errorf("4/6 mapped => %+v, want 2 defects / score 67", bad)
	}
	for _, d := range bad.Defects {
		if !strings.Contains(d, "recovery") {
			t.Errorf("defect missing 'recovery': %q", d)
		}
	}
	if k := kpiRefusalRecoveryMapped(nil, 0); k.Score != 100 {
		t.Errorf("unparsable dos.toml (total 0) => abstain clean, got score %d", k.Score)
	}
}

func TestQuickstartSignalFindsSignalInProofBlock(t *testing.T) {
	if f, s := quickstartSignal(map[string]string{"AGENTS.md": "Proof:\n```\nfak preflight --policy p.json   # -> DENY\n```\n"}); !f || !s {
		t.Errorf("proof with -> signal => (%v,%v), want (true,true)", f, s)
	}
	if f, s := quickstartSignal(map[string]string{"AGENTS.md": "Proof:\n```\nfak preflight --policy p.json\n```\n"}); !f || s {
		t.Errorf("proof without signal => (%v,%v), want (true,false)", f, s)
	}
	if f, s := quickstartSignal(map[string]string{"AGENTS.md": "Proof:\n```\nfak preflight --allow-foo --policy p.json\n```\n"}); !f || s {
		t.Errorf("incidental flag is not a success marker => (%v,%v), want (true,false)", f, s)
	}
	if f, s := quickstartSignal(map[string]string{"README.md": "no fenced first command at all"}); f || s {
		t.Errorf("no fenced command => (%v,%v), want (false,false)", f, s)
	}
}

func TestQuickstartSuccessSignalKpi(t *testing.T) {
	ok := kpiQuickstartSuccessSignal(true, true)
	if len(ok.Defects) != 0 || ok.Score != 100 || ok.Group != "adopt" {
		t.Errorf("clean => %+v", ok)
	}
	bad := kpiQuickstartSuccessSignal(true, false)
	if len(bad.Defects) != 1 || bad.Score != 40 || !strings.Contains(bad.Defects[0], "expected") {
		t.Errorf("no signal => %+v", bad)
	}
	if absent := kpiQuickstartSuccessSignal(false, false); len(absent.Defects) != 0 || absent.Score != 100 {
		t.Errorf("no first command => abstain, got %+v", absent)
	}
}

func TestToolchainPinnedKpi(t *testing.T) {
	if ok := kpiToolchainPinned(true, true); len(ok.Defects) != 0 || ok.Score != 100 || ok.Group != "adopt" {
		t.Errorf("clean => %+v", ok)
	}
	if both := kpiToolchainPinned(false, false); len(both.Defects) != 2 || both.Score != 0 {
		t.Errorf("no directive + undocumented => %+v, want 2 defects / score 0", both)
	}
	if one := kpiToolchainPinned(true, false); len(one.Defects) != 1 || one.Score != 50 {
		t.Errorf("directive present but undocumented => %+v, want 1 defect / score 50", one)
	}
}

// --- fold to friction-debt --------------------------------------------------

func TestBuildPayloadZeroDebtIsOk(t *testing.T) {
	p := buildPayload(".", cleanKPIs(), nil, "")
	if !p.OK || p.Verdict != "OK" || p.Finding != "agent_ready" {
		t.Errorf("zero-debt => ok=%v verdict=%q finding=%q", p.OK, p.Verdict, p.Finding)
	}
	if corpusInt(p.Corpus, "friction_debt") != 0 || corpusStr(p.Corpus, "grade") != "A" {
		t.Errorf("corpus => %v", p.Corpus)
	}
	if corpusFloat(p.Corpus, "score") != 100.0 {
		t.Errorf("score = %v, want 100.0", corpusFloat(p.Corpus, "score"))
	}
	var sum float64
	for _, w := range kpiWeights {
		sum += w
	}
	if math.Abs(sum-1.0) > 1e-9 {
		t.Errorf("weights sum to %v, want 1.0", sum)
	}
	got := map[string]bool{}
	for _, k := range cleanKPIs() {
		got[k.Kpi] = true
	}
	if len(got) != len(kpiWeights) {
		t.Errorf("weights cover %d KPIs, clean set has %d", len(kpiWeights), len(got))
	}
	for name := range kpiWeights {
		if !got[name] {
			t.Errorf("weight %q has no KPI in the clean set", name)
		}
	}
}

func TestBuildPayloadDebtDrivesActionWithGroupAttribution(t *testing.T) {
	swap := map[string]KPI{
		"agent_config":        kpiAgentConfig([]string{"Cursor (AGENTS.md / .cursor/rules)"}), // discover
		"integration_recipes": kpiIntegrationRecipes([]string{"MCP client"}),                  // adopt
		"extension_scaffold":  kpiExtensionScaffold(true, false),                              // build
	}
	kpis := make([]KPI, 0, len(cleanKPIs()))
	for _, k := range cleanKPIs() {
		if s, ok := swap[k.Kpi]; ok {
			kpis = append(kpis, s)
		} else {
			kpis = append(kpis, k)
		}
	}
	p := buildPayload(".", kpis, nil, "")
	if p.OK || p.Finding != "friction_debt" {
		t.Errorf("debt => ok=%v finding=%q", p.OK, p.Finding)
	}
	if corpusInt(p.Corpus, "friction_debt") != 3 {
		t.Errorf("friction_debt = %d, want 3", corpusInt(p.Corpus, "friction_debt"))
	}
	dbg := p.Corpus["debt_by_group"].(map[string]int)
	if !reflect.DeepEqual(dbg, map[string]int{"discover": 1, "adopt": 1, "build": 1}) {
		t.Errorf("debt_by_group = %v, want {discover:1 adopt:1 build:1}", dbg)
	}
	if corpusFloat(p.Corpus, "score") >= 100 {
		t.Errorf("score = %v, want < 100", corpusFloat(p.Corpus, "score"))
	}
}

func TestBuildPayloadError(t *testing.T) {
	p := buildPayload(".", nil, nil, "not a git repo")
	if p.OK || p.Verdict != "AUDIT_ERROR" {
		t.Errorf("error payload => ok=%v verdict=%q", p.OK, p.Verdict)
	}
}

// --- the unbounded experience-frontier --------------------------------------

func TestFrontierUnitsArePinned(t *testing.T) {
	want := map[string]int{"integration_recipes": 8, "harness_configs": 10, "refusal_recoveries": 3, "machine_consumable": 2}
	if !reflect.DeepEqual(frontierUnits, want) {
		t.Errorf("frontierUnits = %v, want %v", frontierUnits, want)
	}
	for dim, w := range frontierUnits {
		if w <= 0 {
			t.Errorf("frontier weight %q = %d, must be a positive int", dim, w)
		}
	}
}

func TestFrontierHarnessConfigsSupersetsCore(t *testing.T) {
	core := map[string]bool{}
	for _, lp := range agentConfigs {
		core[lp.label] = true
	}
	breadth := map[string]bool{}
	for _, lp := range frontierHarnessConfigs {
		breadth[lp.label] = true
	}
	for label := range core {
		if !breadth[label] {
			t.Errorf("core config %q not contained in the frontier breadth list", label)
		}
	}
	if len(breadth) <= len(core) {
		t.Errorf("breadth (%d) must exceed core (%d) — real breadth beyond the gate", len(breadth), len(core))
	}
}

func TestExperienceFrontierIsWeightedSum(t *testing.T) {
	facts := map[string]int{"integration_recipes": 20, "harness_configs": 3, "refusal_recoveries": 16, "machine_consumable": 27}
	total, byTerm := experienceFrontier(facts)
	want := map[string]int{"integration_recipes": 160, "harness_configs": 30, "refusal_recoveries": 48, "machine_consumable": 54}
	if !reflect.DeepEqual(byTerm, want) {
		t.Errorf("byTerm = %v, want %v", byTerm, want)
	}
	if total != 292 {
		t.Errorf("total = %d, want 292", total)
	}
}

func TestExperienceFrontierIsUnboundedAbove100(t *testing.T) {
	small, _ := experienceFrontier(map[string]int{"integration_recipes": 20, "harness_configs": 3, "refusal_recoveries": 16, "machine_consumable": 27})
	big, _ := experienceFrontier(map[string]int{"integration_recipes": 200, "harness_configs": 30, "refusal_recoveries": 160, "machine_consumable": 270})
	if small <= 100 {
		t.Errorf("small frontier = %d, want > 100 (unbounded, not a grade)", small)
	}
	if big <= small*9 {
		t.Errorf("10x affordances => big=%d, want > 9x small=%d (no clamp)", big, small)
	}
}

func TestExperienceFrontierIsMonotonic(t *testing.T) {
	base := map[string]int{"integration_recipes": 4, "harness_configs": 3, "refusal_recoveries": 16, "machine_consumable": 27}
	baseTotal, _ := experienceFrontier(base)
	for dim, w := range frontierUnits {
		bumped := map[string]int{}
		for k, v := range base {
			bumped[k] = v
		}
		bumped[dim]++
		got, _ := experienceFrontier(bumped)
		if got != baseTotal+w {
			t.Errorf("bumping %q => %d, want %d (base+weight)", dim, got, baseTotal+w)
		}
	}
}

func TestExperienceFrontierMissingFactIsZero(t *testing.T) {
	total, byTerm := experienceFrontier(map[string]int{})
	if total != 0 {
		t.Errorf("no facts => total %d, want 0 (fail low, never high)", total)
	}
	for dim, v := range byTerm {
		if v != 0 {
			t.Errorf("missing fact %q => %d, want 0", dim, v)
		}
	}
	if len(byTerm) != len(frontierUnits) {
		t.Errorf("byTerm reports %d dims, want all %d", len(byTerm), len(frontierUnits))
	}
}

func TestBuildPayloadCarriesFrontier(t *testing.T) {
	facts := map[string]int{"integration_recipes": 20, "harness_configs": 3, "refusal_recoveries": 16, "machine_consumable": 27}
	p := buildPayload(".", cleanKPIs(), facts, "")
	if corpusInt(p.Corpus, "experience_frontier") != 292 {
		t.Errorf("experience_frontier = %d, want 292", corpusInt(p.Corpus, "experience_frontier"))
	}
	if bt := p.Corpus["frontier_by_term"].(map[string]int); bt["integration_recipes"] != 160 {
		t.Errorf("frontier_by_term[integration_recipes] = %d, want 160", bt["integration_recipes"])
	}
	if fu := p.Corpus["frontier_units"].(map[string]int); !reflect.DeepEqual(fu, frontierUnits) {
		t.Errorf("frontier_units = %v, want %v", fu, frontierUnits)
	}
	if corpusFloat(p.Corpus, "score") != 100.0 || corpusInt(p.Corpus, "friction_debt") != 0 {
		t.Error("the unbounded headline must ride alongside the bounded gate")
	}
}

func TestBuildPayloadWithoutFactsIsBackCompatible(t *testing.T) {
	p := buildPayload(".", cleanKPIs(), nil, "")
	if corpusInt(p.Corpus, "experience_frontier") != 0 {
		t.Errorf("no facts => frontier %d, want 0", corpusInt(p.Corpus, "experience_frontier"))
	}
	if corpusFloat(p.Corpus, "score") != 100.0 || corpusStr(p.Corpus, "grade") != "A" || corpusInt(p.Corpus, "friction_debt") != 0 {
		t.Error("score/grade/friction-debt unchanged without facts")
	}
}

func TestIsSubstantiveRecipeRequiresRealContent(t *testing.T) {
	if isSubstantiveRecipe("") || isSubstantiveRecipe("# Title only\n") || isSubstantiveRecipe(strings.Repeat("x", 300)) {
		t.Error("a stub / title-only / fenceless-linkless body is not substantive")
	}
	longLink := "# fak + Foo\n" + strings.Repeat("Point your agent at fak. ", 20) + "\n[setup](./s.md)\n"
	if !isSubstantiveRecipe(longLink) {
		t.Error("length + a link is substantive")
	}
	longFence := "# fak + Bar\n" + strings.Repeat("Run the proof. ", 20) + "\n```\nfak preflight\n```\n"
	if !isSubstantiveRecipe(longFence) {
		t.Error("length + a fence is substantive")
	}
}

// --- render_compare ---------------------------------------------------------

func TestRenderCompareReports35pctFrontierGoal(t *testing.T) {
	baseFacts := map[string]int{"integration_recipes": 10, "harness_configs": 2, "refusal_recoveries": 4, "machine_consumable": 5}
	base := buildPayload(".", cleanKPIs(), baseFacts, "")
	up := copyFacts(baseFacts)
	up["integration_recipes"] += 8
	curUp := buildPayload(".", cleanKPIs(), up, "")
	if out := Compare(curUp, baselineMap(t, base)); !strings.Contains(out, "+35% achieved") {
		t.Errorf("a >=35%% climb must read achieved:\n%s", out)
	}
	small := copyFacts(baseFacts)
	small["integration_recipes"]++
	curSmall := buildPayload(".", cleanKPIs(), small, "")
	if out := Compare(curSmall, baselineMap(t, base)); !strings.Contains(out, "not yet +35%") {
		t.Errorf("a small climb must read not-yet:\n%s", out)
	}
}

func TestRenderCompareZeroBaselineAndGateLine(t *testing.T) {
	cur := buildPayload(".", cleanKPIs(), map[string]int{"integration_recipes": 5, "harness_configs": 1, "refusal_recoveries": 2, "machine_consumable": 3}, "")
	base0 := buildPayload(".", cleanKPIs(), nil, "")
	out := Compare(cur, baselineMap(t, base0))
	if !strings.Contains(out, "no prior frontier") {
		t.Errorf("zero-baseline verdict branch missing:\n%s", out)
	}
	if !strings.Contains(out, "(gate)") {
		t.Errorf("friction-debt gate line missing:\n%s", out)
	}
}

func TestRenderCompareSignedScoreDeltaOnRegression(t *testing.T) {
	hi := buildPayload(".", cleanKPIs(), nil, "") // score 100
	loKPIs := make([]KPI, 0, len(cleanKPIs()))
	for _, k := range cleanKPIs() {
		if k.Kpi == "first_command" {
			loKPIs = append(loKPIs, kpiFirstCommand(false, ""))
		} else {
			loKPIs = append(loKPIs, k)
		}
	}
	lo := buildPayload(".", loKPIs, nil, "")
	if out := Compare(lo, baselineMap(t, hi)); strings.Contains(out, "(+-") {
		t.Errorf("a score regression must render a single-signed delta, never (+-):\n%s", out)
	}
}

// --- the tolerant live smoke ------------------------------------------------

// TestLivePayloadIsWellFormed exercises the real disk+git gather path and pins the payload
// SHAPE (schema, the weighted KPI set, per-KPI control-pane fields, a summed frontier). It
// deliberately does NOT assert zero friction-debt: that is a tree-state regression sentinel
// (owned by the Python live tests), not a property of this port.
func TestLivePayloadIsWellFormed(t *testing.T) {
	// Go runs package tests from this directory; Build expects the repository root.
	p := Build(filepath.Join("..", ".."))
	if p.Schema == "" || p.Verdict == "" || p.Finding == "" || p.Reason == "" || p.NextAction == "" {
		t.Errorf("live payload missing a top-level field: %+v", p)
	}
	if p.Corpus == nil {
		t.Fatal("live payload has no corpus")
	}
	if len(p.KPIs) != len(kpiWeights) {
		t.Fatalf("live payload has %d KPIs, want the weighted set of %d", len(p.KPIs), len(kpiWeights))
	}
	for _, k := range p.KPIs {
		if k.Kpi == "" || k.Group == "" {
			t.Errorf("KPI missing kpi/group: %+v", k)
		}
		if k.Defects == nil || k.Soft == nil {
			t.Errorf("KPI %q must carry non-nil defects/soft slices (JSON [] not null)", k.Kpi)
		}
	}
	frontier := corpusInt(p.Corpus, "experience_frontier")
	byTerm, ok := p.Corpus["frontier_by_term"].(map[string]int)
	if !ok {
		t.Fatalf("frontier_by_term has type %T, want map[string]int", p.Corpus["frontier_by_term"])
	}
	sum := 0
	for _, v := range byTerm {
		sum += v
	}
	if sum != frontier {
		t.Errorf("frontier_by_term sums to %d, want experience_frontier %d", sum, frontier)
	}
	if len(byTerm) != len(frontierUnits) {
		t.Errorf("frontier_by_term has %d dims, want %d", len(byTerm), len(frontierUnits))
	}
}
