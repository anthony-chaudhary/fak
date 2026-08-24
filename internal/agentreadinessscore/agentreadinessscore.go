// Package agentreadinessscore grades the ONE thing the sibling scorecards do not: can an
// autonomous coding agent — Claude Code, OpenAI Codex, Cursor, an MCP client — (1) DISCOVER
// fak, (2) WANT to adopt and build on it, and (3) do so effectively and easily? The other
// inward sticks grade a surface a human reviewer cares about (the tree's shape, the Go
// module, a doc's prose); this one grades agent attractiveness, the number an agent-first
// project lives or dies on.
//
// It scores the git-tracked tree on twenty-three mechanical KPIs in three groups — the exact
// three steps an agent walks (DISCOVER / ADOPT / BUILD) — folds them into a weighted score
// and an A-F grade, and reports two headline numbers, because agent-readiness is two
// questions:
//
//   - friction_debt (the HARD gate, lower = better, floor 0): the count of concrete,
//     re-derivable defects that make the BASELINE affordances missing or broken — a dead
//     link, an un-runnable first command, an un-tagged claim. It saturates: drive it to zero
//     and there is nothing left to fix. The 0-100 score / A-F grade is the same
//     baseline-coverage question in percent form.
//   - experience_frontier (the headline, higher = better, UNBOUNDED): the weighted count of
//     real, WORKING agent affordances the tree provides — each integration recipe, each
//     zero-setup harness config, each kernel refusal an agent can recover from, each tool an
//     agent drives via --json. Agent experience is a never-done program, so it has no ceiling
//     and is tracked as a frontier + a trend, the deliberate mirror of the operator-heaviness
//     gauge. You move it by ADDING a real affordance, never by gaming a substring.
//
// This is the Go port of the former tools/agent_readiness_scorecard.py. It is a faithful,
// self-contained re-implementation (Idiom B, like internal/focusscore): the default / --json
// / --markdown / --stamp / --compare surfaces are preserved so the committed snapshot and the
// control-pane fold are unchanged. Deterministic + read-only: it reads the git-tracked tree
// (two clones of the same commit score identically) and edits nothing.
package agentreadinessscore

import (
	"encoding/json"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/devhandoff"
	"github.com/anthony-chaudhary/fak/internal/windowgate"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

// Schema is the control-pane schema id (unchanged from the Python tool).
const Schema = "fak-agent-readiness-scorecard/2"

// ---------------------------------------------------------------------------
// The contract an agent expects. Each constant is a deliberate, named affordance
// an autonomous agent reaches for — never a hand-picked file list where a rule would do.
// ---------------------------------------------------------------------------

const (
	agentsFile    = "AGENTS.md"
	llmsFile      = "llms.txt"
	llmsFullFile  = "llms-full.txt"
	claimsFile    = "CLAIMS.md"
	leafScaffold  = "cmd/fak/newleaf.go"
	extendingFile = "EXTENDING.md"
	contribFile   = "CONTRIBUTING.md"
	codexFile     = "docs/integrations/openai-codex.md"
	mcpConfigFile = ".mcp.json"
	dosToml       = "dos.toml"
	mainGo        = "cmd/fak/main.go"
	goModFile     = "go.mod"
	pasteDocsGlob = "docs/integrations"
)

var claimTags = []string{"[SHIPPED]", "[SIMULATED]", "[STUB]"}

// labeledPaths is (harness/family label, candidate paths); the affordance is present if ANY
// path exists.
type labeledPaths struct {
	label string
	paths []string
}

// agentConfigs — the zero-setup configs an agent's harness auto-discovers on entry (the
// REQUIRED core the gate enforces).
var agentConfigs = []labeledPaths{
	{"MCP clients (.mcp.json)", []string{".mcp.json", "examples/mcp/.mcp.json"}},
	{"Cursor (AGENTS.md / .cursor/rules)", []string{"AGENTS.md", ".cursor/rules"}},
	{"GitHub Copilot (copilot-instructions)", []string{".github/copilot-instructions.md"}},
}

// frontierHarnessConfigs — every named harness fak ships a zero-setup config for (the
// experience-frontier's harness_configs term; a superset of agentConfigs so the core never
// double-counts). OPTIONAL surface to climb, never required.
var frontierHarnessConfigs = append(append([]labeledPaths{}, agentConfigs...), []labeledPaths{
	{"Cline (AGENTS.md / .clinerules)", []string{"AGENTS.md", ".clinerules"}},
	{"Windsurf (AGENTS.md / .windsurf/rules)", []string{"AGENTS.md", ".windsurf/rules"}},
	{"Gemini CLI (GEMINI.md)", []string{"GEMINI.md"}},
	{"Amp / AGENT.md convention", []string{"AGENT.md"}},
	{"Aider (.aider.conf.yml)", []string{".aider.conf.yml"}},
	{"Zed (AGENTS.md / .rules)", []string{"AGENTS.md", ".rules"}},
	{"JetBrains Junie (.junie/guidelines.md)", []string{".junie/guidelines.md"}},
	{"Continue (.continue/rules)", []string{".continue/rules/fak.md", ".continue/rules"}},
}...)

// requiredRecipes — one per-agent integration recipe for each family an agent identifies with.
var requiredRecipes = []labeledPaths{
	{"Claude Code / Anthropic", []string{"docs/integrations/claude.md"}},
	{"OpenAI Codex / OpenAI", []string{"docs/integrations/openai-codex.md"}},
	{"Cursor", []string{"docs/integrations/cursor.md"}},
	{"MCP client", []string{"examples/mcp/README.md", "docs/integrations/mcp.md"}},
}

// codexCluster is (label, tokens); a Codex-currentness cluster is satisfied only when EVERY
// token is present.
type codexCluster struct {
	label  string
	tokens []string
}

var codexRecipeClusters = []codexCluster{
	{"current Codex product surface", []string{"coding agent", "CLI", "IDE", "app", "cloud"}},
	{"AGENTS.md instruction path", []string{"AGENTS.md", "reads"}},
	{"MCP server path", []string{"codex mcp", "fak serve --stdio"}},
	{"machine-readable codex exec path", []string{"codex exec", "--json"}},
	{"OpenAI-compatible proxy path", []string{"OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"}},
	{"Responses honesty fence", []string{"Responses", "/v1/responses", "Chat Completions"}},
}

var staleCodexRecipeTokens = []string{
	"deprecated the standalone Codex API",
	"gpt-4-turbo",
	"gpt-3.5-turbo",
	"o1-preview",
}

var guardrailClusters = []labeledPaths{
	{"trunk-only (no feature branch)", []string{"off_trunk", "on the trunk", "feature branch", "trunk guard"}},
	{"commit by explicit path (no add -A)", []string{"git add -a", "explicit path", "commit -- <", "commit by explicit"}},
	{"DCO sign-off", []string{"git commit -s", "sign off", "sign-off", "dco"}},
	{"tagged claims ledger", []string{"claims.md", "claims-lint", "[shipped]"}},
	{"leaf / frozen-ABI discipline", []string{"new-leaf", "as a leaf", "frozen abi", "additive-only"}},
	{"out-of-tree write guard", []string{"out_of_tree", "out-of-tree", "fak_repo_guard", "repo-guard", "outside the repo", "outside this repo"}},
}

var (
	firstCommandTokens = []string{"fak preflight", "fak agent --offline", "preflight --policy"}
	firstCommandDocs   = []string{agentsFile, "README.md", "START-HERE.md", "GETTING-STARTED.md"}
	installTokens      = []string{"go install", "@latest"}
	installDocs        = []string{agentsFile, "README.md", "GETTING-STARTED.md", "INSTALL.md"}
	greenGateTokens    = []string{"make ci", "make test", "make test-fast", "scripts/ci.ps1", "./test.ps1"}
	orientationDocs    = []string{agentsFile, "docs/integrations/README.md"}
	pasteDocs          = []string{agentsFile, "README.md", "GETTING-STARTED.md", "START-HERE.md"}
)

var (
	stalePathPrefixes  = []string{"fleet/", "fak/fak/", "../fleet/", "fleet\\"}
	placeholderLiteral = []string{"/path/to/", "path/to/your", "/PATH/TO/"}
	repoTopDirs        = []string{"examples/", "cmd/", "internal/", "docs/", "tools/", "scripts/", "pkg/", "test/", "testdata/"}
)

var (
	proofNeedsKeyTokens = []string{"--api-key-env", "api.openai.com", "OPENAI_API_KEY", "--provider "}
	windowsBridgeTokens = []string{"scripts/ci.ps1", "ci.ps1", "test.ps1", "wsl"}
	identityDocs        = []string{agentsFile, llmsFile, "README.md"}
	successSignalTokens = []string{"-> ", "=> ", "→ ", "exit code", "expected", "you should see", "prints", "outputs", "yes->no"}
	refusalRecoveryDocs = []string{agentsFile, "docs/repo-guard.md"}
	recoveryCues        = []string{"recover", "refus", "fix", "floor", "instead", "soften"}
	toolchainDocs       = []string{agentsFile, "README.md", "GETTING-STARTED.md"}
	commandFamilyVerbs  = []string{"server"}
	cmdPrefixes         = []string{"go run ./cmd/fak ", "go run cmd/fak ", "./cmd/fak ", ".\\fak.exe ", ".\\fak ", "fak.exe ", "./fak ", "fak "}
)

const (
	identityHeadLines = 50
	recoveryWindow    = 20
	recipeMinChars    = 200
	configMinChars    = 24
)

// groups is the ordered agent journey; groupIndex keeps deterministic ordering.
var groups = []string{"discover", "adopt", "build"}

var kpiGroup = map[string]string{
	"agents_entrypoint": "discover", "agent_config": "discover", "agent_config_valid": "discover",
	"llms_map": "discover", "identity_statement": "discover", "entry_links_resolve": "discover",
	"recipe_links_resolve": "discover", "first_command": "adopt", "command_verbs_resolve": "adopt",
	"install_oneliner": "adopt", "honesty_ledger": "adopt", "integration_recipes": "adopt",
	"codex_recipe_current": "adopt", "extension_scaffold": "build", "guardrails_surfaced": "build",
	"contributor_contract": "build", "machine_consumable": "build", "fenced_paths_resolve": "adopt",
	"first_command_runs": "adopt", "platform_guidance_consistent": "build",
	"refusal_recovery_mapped": "build", "quickstart_success_signal": "adopt", "toolchain_pinned": "adopt",
}

// kpiWeightOrder is the EXACT insertion order the Python KPI_WEIGHTS dict carries — the score
// sum walks it in this order so the floating-point fold is bit-identical.
var kpiWeightOrder = []string{
	"agents_entrypoint", "agent_config", "agent_config_valid", "llms_map", "entry_links_resolve",
	"recipe_links_resolve", "identity_statement",
	"fenced_paths_resolve", "command_verbs_resolve", "first_command", "first_command_runs",
	"honesty_ledger", "integration_recipes", "codex_recipe_current", "install_oneliner",
	"quickstart_success_signal", "toolchain_pinned",
	"guardrails_surfaced", "contributor_contract", "extension_scaffold",
	"platform_guidance_consistent", "machine_consumable", "refusal_recovery_mapped",
}

var kpiWeights = map[string]float64{
	"agents_entrypoint": 0.08, "agent_config": 0.05, "agent_config_valid": 0.03, "llms_map": 0.04,
	"entry_links_resolve": 0.04, "recipe_links_resolve": 0.03, "identity_statement": 0.03,
	"fenced_paths_resolve": 0.06, "command_verbs_resolve": 0.05, "first_command": 0.03,
	"first_command_runs": 0.04, "honesty_ledger": 0.04, "integration_recipes": 0.04,
	"codex_recipe_current": 0.04, "install_oneliner": 0.03, "quickstart_success_signal": 0.03,
	"toolchain_pinned": 0.03, "guardrails_surfaced": 0.06, "contributor_contract": 0.05,
	"extension_scaffold": 0.05, "platform_guidance_consistent": 0.05, "machine_consumable": 0.05,
	"refusal_recovery_mapped": 0.05,
}

// frontierDims is the ordered set of unbounded-frontier dimensions; frontierUnits the
// points-per-unit weight of each.
var frontierDims = []string{"integration_recipes", "harness_configs", "refusal_recoveries", "machine_consumable"}

var frontierUnits = map[string]int{
	"integration_recipes": 8, "harness_configs": 10, "refusal_recoveries": 3, "machine_consumable": 2,
}

// ---- compiled regexes (mirror the Python module-level patterns) -------------------

var (
	claimLineRe   = regexp.MustCompile(`^\s*- \[`)
	bracketSlotRe = regexp.MustCompile(`<[^>]+>`)
	envSlotRe     = regexp.MustCompile(`^[A-Z][A-Z0-9_]{2,}$`)
	proofPolicyRe = regexp.MustCompile(`--policy\s+(\S+)`)
	identityRe    = regexp.MustCompile(`(?i)\bfak\b[^.\n]{0,60}?\b(?:is\b[^.\n]{0,80}?(?:kernel|firewall|gate|gateway|proxy|binary)|(?:lets|helps|enables)\s+you\b[^.\n]{0,120}?(?:agent|context|model|tool|permission))`)
	reasonRe      = regexp.MustCompile(`(?m)^\[reasons\.([A-Z][A-Z0-9_]+)\]`)
	verbTokenRe   = regexp.MustCompile(`^([a-z][a-z0-9-]+)`)
	inlineCodeRe  = regexp.MustCompile("`([^`]+)`")
	goDirectiveRe = regexp.MustCompile(`(?m)^go\s+\d+\.\d+`)
	goVersDocRe   = regexp.MustCompile(`(?i)go\s*1\.\d+`)
	linkRe        = regexp.MustCompile(`\[([^\]]+)\]\(([^)]+)\)`)
	fenceRe       = regexp.MustCompile("^(```|~~~)")
	caseRe        = regexp.MustCompile(`^\s*case\s+(.+):`)
	quotedRe      = regexp.MustCompile(`"([^"]+)"`)
	wsHashRe      = regexp.MustCompile(`\s+#\s`)
	wsRe          = regexp.MustCompile(`\s+`)
	cdRe          = regexp.MustCompile(`^cd\s+(\S+)`)
	segSepRe      = regexp.MustCompile(`&&|\|\||;|\||\s--\s`)
	promptRe      = regexp.MustCompile(`^[\$>]\s+`)
	envPrefixRe   = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*=\S*\s+`)
)

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

// KPI is one graded criterion. JSON shape mirrors the Python kpi dict exactly.
type KPI struct {
	Kpi     string   `json:"kpi"`
	Group   string   `json:"group"`
	Score   int      `json:"score"`
	Detail  string   `json:"detail"`
	Defects []string `json:"defects"`
	Soft    []string `json:"soft"`
}

// breakdownRow is one worst-first row of corpus.breakdown (a struct so the JSON key order
// matches the Python dict).
type breakdownRow struct {
	Kpi    string `json:"kpi"`
	Group  string `json:"group"`
	Score  int    `json:"score"`
	Debt   int    `json:"debt"`
	Detail string `json:"detail"`
}

// Payload is the full control-pane payload. Field order mirrors the Python dict insertion order.
type Payload struct {
	Schema     string         `json:"schema"`
	OK         bool           `json:"ok"`
	Verdict    string         `json:"verdict"`
	Finding    string         `json:"finding"`
	Reason     string         `json:"reason"`
	NextAction string         `json:"next_action"`
	Workspace  string         `json:"workspace"`
	Corpus     map[string]any `json:"corpus"`
	KPIs       []KPI          `json:"kpis"`
}

// ---------------------------------------------------------------------------
// Small pure helpers (the testable core).
// ---------------------------------------------------------------------------

// round1 rounds to one decimal the way Python round(x, 1) does (correctly-rounded,
// half-to-even) by round-tripping through fixed-precision formatting.
func round1(x float64) float64 {
	v, _ := strconv.ParseFloat(strconv.FormatFloat(x, 'f', 1, 64), 64)
	return v
}

// experienceFrontier is the UNBOUNDED agent-experience frontier: sum of weight*count over
// every dimension, plus the per-dimension breakdown. A missing fact counts as zero.
func experienceFrontier(facts map[string]int) (int, map[string]int) {
	byTerm := make(map[string]int, len(frontierDims))
	total := 0
	for _, dim := range frontierDims {
		v := facts[dim]
		if v < 0 {
			v = 0
		}
		byTerm[dim] = frontierUnits[dim] * v
		total += byTerm[dim]
	}
	return total, byTerm
}

// has reports whether text (case-insensitive) contains any of the tokens.
func has(text string, tokens ...string) bool {
	if text == "" {
		return false
	}
	low := strings.ToLower(text)
	for _, t := range tokens {
		if strings.Contains(low, strings.ToLower(t)) {
			return true
		}
	}
	return false
}

// fencedBlocks returns the contents of every fenced code block.
func fencedBlocks(text string) []string {
	var blocks []string
	var cur []string
	inFence := false
	for _, raw := range strings.Split(text, "\n") {
		if fenceRe.MatchString(strings.TrimSpace(raw)) {
			if inFence {
				blocks = append(blocks, strings.Join(cur, "\n"))
				cur = nil
			}
			inFence = !inFence
			continue
		}
		if inFence {
			cur = append(cur, raw)
		}
	}
	return blocks
}

// proseOutsideFences returns the document with fenced blocks (and fence lines) removed.
func proseOutsideFences(text string) string {
	var out []string
	inFence := false
	for _, raw := range strings.Split(text, "\n") {
		if fenceRe.MatchString(strings.TrimSpace(raw)) {
			inFence = !inFence
			continue
		}
		if !inFence {
			out = append(out, raw)
		}
	}
	return strings.Join(out, "\n")
}

func isTemplateSlot(tok string) bool {
	return bracketSlotRe.MatchString(tok) || envSlotRe.MatchString(tok)
}

// splitOnceWSHash returns the segment before the first ` # ` inline comment.
func splitOnceWSHash(s string) string {
	if loc := wsHashRe.FindStringIndex(s); loc != nil {
		return s[:loc[0]]
	}
	return s
}

// pathOperands returns the repo-relative-looking path operands in a fenced command block.
func pathOperands(block string) []string {
	var ops []string
	for _, raw := range strings.Split(block, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		var code string
		if strings.HasPrefix(strings.TrimLeft(line, " \t"), "\"") {
			code = line
		} else {
			code = splitOnceWSHash(line)
		}
		for _, tok := range wsRe.Split(code, -1) {
			rawTok := strings.TrimSpace(tok)
			t := strings.Trim(rawTok, "\"'`,;\\")
			if t == "" || isTemplateSlot(t) || strings.ContainsAny(rawTok, "\"[]{}") {
				continue
			}
			low := strings.ToLower(t)
			isRepoRel := (strings.HasPrefix(low, "./") && strings.Contains(low[2:], "/"))
			if !isRepoRel {
				for _, d := range repoTopDirs {
					if strings.HasPrefix(low, d) {
						isRepoRel = true
						break
					}
				}
			}
			if !isRepoRel {
				for _, p := range stalePathPrefixes {
					if strings.HasPrefix(low, p) {
						isRepoRel = true
						break
					}
				}
			}
			if isRepoRel {
				ops = append(ops, t)
			}
		}
		if m := cdRe.FindStringSubmatch(code); m != nil {
			tgt := strings.Trim(m[1], "\"'`")
			if tgt != "" && !isTemplateSlot(tgt) && !strings.HasPrefix(tgt, "/") &&
				!strings.HasPrefix(tgt, "~") && !strings.HasPrefix(tgt, "$") {
				ops = append(ops, tgt)
			}
		}
	}
	return ops
}

// findFirstCommand reports whether a no-key/no-model/no-GPU first command sits inside a
// fenced block of an adoption doc, and where.
func findFirstCommand(texts map[string]string) (bool, string) {
	for _, doc := range firstCommandDocs {
		for _, block := range fencedBlocks(texts[doc]) {
			if has(block, firstCommandTokens...) {
				return true, doc
			}
		}
	}
	return false, ""
}

func findInstallOneliner(texts map[string]string) (bool, string) {
	for _, doc := range installDocs {
		t := texts[doc]
		all := true
		for _, tok := range installTokens {
			if !has(t, tok) {
				all = false
				break
			}
		}
		if all {
			return true, doc
		}
	}
	return false, ""
}

func findIdentity(texts map[string]string) (present, missing []string) {
	present, missing = []string{}, []string{}
	for _, doc := range identityDocs {
		lines := strings.Split(texts[doc], "\n")
		if len(lines) > identityHeadLines {
			lines = lines[:identityHeadLines]
		}
		head := strings.Join(lines, "\n")
		if identityRe.MatchString(head) {
			present = append(present, doc)
		} else {
			missing = append(missing, doc)
		}
	}
	return present, missing
}

func untaggedClaims(claimsText string, present bool) []string {
	bad := []string{}
	if !present || claimsText == "" {
		return bad
	}
	for i, line := range strings.Split(claimsText, "\n") {
		if !claimLineRe.MatchString(line) {
			continue
		}
		n := 0
		for _, tag := range claimTags {
			n += strings.Count(line, tag)
		}
		if n != 1 {
			snippet := strings.TrimSpace(line)
			if len(snippet) > 80 {
				snippet = snippet[:80]
			}
			bad = append(bad, "CLAIMS.md:"+strconv.Itoa(i+1)+": "+strconv.Itoa(n)+" status tag(s) (need exactly 1): "+snippet)
		}
	}
	return bad
}

func missingRecipes(present map[string]bool) []string {
	out := []string{}
	for _, r := range requiredRecipes {
		if !present[r.label] {
			out = append(out, r.label)
		}
	}
	return out
}

func missingAgentConfigs(present map[string]bool) []string {
	out := []string{}
	for _, c := range agentConfigs {
		if !present[c.label] {
			out = append(out, c.label)
		}
	}
	return out
}

func missingGuardrails(agentsText string) []string {
	out := []string{}
	for _, g := range guardrailClusters {
		if !has(agentsText, g.paths...) {
			out = append(out, g.label)
		}
	}
	return out
}

func codexRecipeGaps(text string, present bool) []string {
	if !present {
		return []string{"missing " + codexFile + " — the Codex/OpenAI recipe an agent follows"}
	}
	gaps := []string{}
	for _, cl := range codexRecipeClusters {
		var missing []string
		for _, tok := range cl.tokens {
			if !has(text, tok) {
				missing = append(missing, tok)
			}
		}
		if len(missing) > 0 {
			gaps = append(gaps, codexFile+" missing "+cl.label+": "+strings.Join(missing, ", "))
		}
	}
	for _, tok := range staleCodexRecipeTokens {
		if has(text, tok) {
			gaps = append(gaps, codexFile+" still carries stale Codex-era copy: "+tok)
		}
	}
	return gaps
}

// dispatchVerbFns names the top-level verb-routing functions main() delegates to in
// cmd/fak/main.go. Each listed helper switches on the same top-level verb name; they
// are routing-table splits, not subcommand switches
// like cmdPolicy's argv[0] switch — those must NOT leak in). If the routing table is
// split across a new helper, add its `func <name>(` header here so real verbs keep
// resolving.
var dispatchVerbFns = []string{
	"func main()",
	"func dispatchCoreVerbA(",
	"func dispatchCoreVerbB(",
	"func dispatchExtendedVerbA(",
	"func dispatchExtendedVerbB(",
	"func dispatchPrimaryVerb(",
}

// dispatchVerbs is the set of top-level verbs the binary dispatches, parsed from every
// top-level routing switch in cmd/fak/main.go (func main() plus the routing-table
// helpers it delegates to).
func dispatchVerbs(mainGoText string) map[string]bool {
	verbs := map[string]bool{}
	if mainGoText == "" {
		return verbs
	}
	lines := strings.Split(mainGoText, "\n")
	for _, fn := range dispatchVerbFns {
		collectCaseVerbs(lines, fn, verbs)
	}
	return verbs
}

// collectCaseVerbs adds every quoted `case "verb":` label in the body of the function
// whose header line begins with fnPrefix. The body runs from the header to the first
// column-0 `}` (the function's own closing brace); nested switch/closure braces are
// indented, so they never end the scan early.
func collectCaseVerbs(lines []string, fnPrefix string, verbs map[string]bool) {
	start := -1
	for i, ln := range lines {
		if strings.HasPrefix(ln, fnPrefix) {
			start = i
			break
		}
	}
	if start < 0 {
		return
	}
	for _, ln := range lines[start+1:] {
		if ln == "}" {
			break
		}
		m := caseRe.FindStringSubmatch(ln)
		if m == nil {
			continue
		}
		for _, sm := range quotedRe.FindAllStringSubmatch(m[1], -1) {
			verbs[sm[1]] = true
		}
	}
}

// commandVerbs returns every CLI verb an agent would paste from this doc, in appearance order.
func commandVerbs(text string) []string {
	var verbs []string
	fromSegment := func(seg string) {
		s := strings.TrimSpace(splitOnceWSHash(strings.TrimSpace(seg)))
		s = promptRe.ReplaceAllString(s, "")
		for {
			loc := envPrefixRe.FindStringIndex(s)
			if loc == nil {
				break
			}
			s = s[loc[1]:]
		}
		for _, pre := range cmdPrefixes {
			if strings.HasPrefix(s, pre) {
				rest := strings.TrimLeft(s[len(pre):], " \t")
				m := verbTokenRe.FindStringSubmatch(rest)
				if m != nil {
					end := len(m[1])
					next := ""
					if end < len(rest) {
						next = rest[end : end+1]
					}
					if next != ":" {
						verbs = append(verbs, m[1])
					}
				}
				return
			}
		}
	}
	for _, block := range fencedBlocks(text) {
		for _, line := range strings.Split(block, "\n") {
			if strings.HasPrefix(strings.TrimLeft(line, " \t"), "#") {
				continue
			}
			for _, seg := range segSepRe.Split(line, -1) {
				fromSegment(seg)
			}
		}
	}
	for _, m := range inlineCodeRe.FindAllStringSubmatch(proseOutsideFences(text), -1) {
		fromSegment(m[1])
	}
	return verbs
}

func dosReasonTokens(dosText string) []string {
	if dosText == "" {
		return []string{}
	}
	set := map[string]bool{}
	for _, m := range reasonRe.FindAllStringSubmatch(dosText, -1) {
		set[m[1]] = true
	}
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func queryBackedRefusalRecovery(tokens []string, dosText, recoveryText string) []string {
	if !has(recoveryText, "dos man wedge", "--explain") || !has(recoveryText, "fak recover") {
		return unmappedRefusalTokens(tokens, recoveryText)
	}
	unmapped := []string{}
	for _, token := range tokens {
		section := "[reasons." + token + "]"
		start := strings.Index(dosText, section)
		if start < 0 {
			unmapped = append(unmapped, token)
			continue
		}
		end := strings.Index(dosText[start+len(section):], "\n[reasons.")
		block := dosText[start:]
		if end >= 0 {
			block = dosText[start : start+len(section)+end]
		}
		if !regexp.MustCompile(`(?m)^summary\s*=`).MatchString(block) || !regexp.MustCompile(`(?m)^fix\s*=`).MatchString(block) {
			unmapped = append(unmapped, token)
		}
	}
	return unmapped
}
func unmappedRefusalTokens(tokens []string, recoveryText string) []string {
	if recoveryText == "" {
		return append([]string{}, tokens...)
	}
	lines := strings.Split(recoveryText, "\n")
	var cueLines []int
	for i, ln := range lines {
		if has(ln, recoveryCues...) {
			cueLines = append(cueLines, i)
		}
	}
	unmapped := []string{}
	for _, t := range tokens {
		var hits []int
		for i, ln := range lines {
			if strings.Contains(ln, t) {
				hits = append(hits, i)
			}
		}
		mapped := false
		for _, h := range hits {
			for _, c := range cueLines {
				d := h - c
				if d < 0 {
					d = -d
				}
				if d <= recoveryWindow {
					mapped = true
					break
				}
			}
			if mapped {
				break
			}
		}
		if !mapped {
			unmapped = append(unmapped, t)
		}
	}
	return unmapped
}

// quickstartSignal returns (found, hasSignal) for the 60-second proof block.
func quickstartSignal(texts map[string]string) (bool, bool) {
	for _, doc := range firstCommandDocs {
		for _, block := range fencedBlocks(texts[doc]) {
			if has(block, firstCommandTokens...) {
				return true, has(block, successSignalTokens...)
			}
		}
	}
	return false, false
}

// ---------------------------------------------------------------------------
// Per-KPI pure checks. Each returns a KPI (defects = HARD friction-debt units; soft =
// score-only judgment nudges). Slices are always non-nil so JSON emits [] not null.
// ---------------------------------------------------------------------------

func kpiAgentsEntrypoint(agentsText string, present bool) KPI {
	defects := []string{}
	if !present || strings.TrimSpace(agentsText) == "" {
		return KPI{"agents_entrypoint", "discover", 0, "no " + agentsFile + " at the repo root",
			[]string{"missing " + agentsFile + " — the agents.md machine-read entry point (agents.md)"}, []string{}}
	}
	if !has(agentsText, "what this project is", "what this is", "is an agent", "**fak**") {
		defects = append(defects, agentsFile+" does not state what the project is (a 'what this is' line)")
	}
	if !has(agentsText, "go build", "make build", "make test-fast") {
		defects = append(defects, agentsFile+" has no build command (go build / make …)")
	}
	if !has(agentsText, "make test", "go test", "test.ps1", "make ci") {
		defects = append(defects, agentsFile+" has no test command (make test / go test / ./test.ps1)")
	}
	if !has(agentsText, "go run ./cmd/fak", "fak preflight", "fak serve", "fak agent") {
		defects = append(defects, agentsFile+" has no runnable first verb (go run ./cmd/fak … / fak …)")
	}
	detail := "AGENTS.md states identity + build/test/run"
	if len(defects) > 0 {
		detail = strconv.Itoa(len(defects)) + " missing entry-point element(s)"
	}
	return KPI{"agents_entrypoint", "discover", mathx.ClampScore(100 - 22*float64(len(defects))), detail, defects, []string{}}
}

// kpiCoverage renders the "N of M families are covered" KPI shape: one defect per
// missing label, score = covered/total. `defect` and `noun` carry each caller's own
// wording so the rendered strings are unchanged.
func kpiCoverage(kpi, group string, total int, missing []string, defect func(label string) string, noun string) KPI {
	defects := []string{}
	for _, label := range missing {
		defects = append(defects, defect(label))
	}
	covered := total - len(missing)
	return KPI{kpi, group,
		mathx.ClampScore(100 * float64(covered) / float64(max1(total))),
		strconv.Itoa(covered) + "/" + strconv.Itoa(total) + " " + noun,
		defects, []string{}}
}

func kpiAgentConfig(missing []string) KPI {
	return kpiCoverage("agent_config", "discover", len(agentConfigs), missing,
		func(label string) string {
			return "no auto-discovered config for " + label + " — add it so that harness drops in with no setup"
		},
		"agent harnesses have a zero-setup config")
}

func kpiLLMSMap(llmsPresent, llmsFullPresent bool) KPI {
	defects := []string{}
	soft := []string{}
	if !llmsPresent {
		defects = append(defects, "missing "+llmsFile+" — the agent / answer-engine doc-map")
	}
	if !llmsFullPresent {
		soft = append(soft, "no "+llmsFullFile+" (the inlined full doc-map an answer engine ingests)")
	}
	detail := llmsFile + " present"
	if len(defects) > 0 {
		detail = "no " + llmsFile
	}
	return KPI{"llms_map", "discover", mathx.ClampScore(100 - 60*float64(len(defects)) - 8*float64(len(soft))), detail, defects, soft}
}

func kpiIdentityStatement(presentIn, missingFrom []string) KPI {
	defects := []string{}
	for _, doc := range missingFrom {
		defects = append(defects, "no plain-English statement of what fak is or does near the top of "+doc+" — add a quotable one-liner")
	}
	score := 100
	var detail string
	if len(missingFrom) == 0 {
		detail = "identity resolves near the top of all " + strconv.Itoa(len(identityDocs)) + " orientation docs (" + strings.Join(presentIn, ", ") + ")"
	} else {
		score = int(math.RoundToEven(100 * float64(len(presentIn)) / float64(len(identityDocs))))
		pres := strings.Join(presentIn, ", ")
		if pres == "" {
			pres = "none"
		}
		detail = "identity missing from " + strings.Join(missingFrom, ", ") + " (present in " + pres + ")"
	}
	return KPI{"identity_statement", "discover", score, detail, defects, []string{}}
}

func kpiEntryLinksResolve(dead []string) KPI {
	sorted := append([]string{}, dead...)
	sort.Strings(sorted)
	defects := []string{}
	for _, d := range sorted {
		defects = append(defects, "dead orientation link: "+d)
	}
	detail := "every orientation link resolves"
	if len(defects) > 0 {
		detail = strconv.Itoa(len(defects)) + " dead orientation link(s)"
	}
	return KPI{"entry_links_resolve", "discover", mathx.ClampScore(100 - 14*float64(len(defects))), detail, defects, []string{}}
}

// kpiPresence renders the "the one runnable thing is either there or it is not" KPI
// shape: 100 and a locating detail when found, otherwise a fixed floor plus one defect
// naming what to add. `absentScore` keeps each caller's own floor — 20 for a missing
// first command, 40 for a missing install one-liner — which is the only number the two
// KPIs disagree on.
func kpiPresence(kpi string, found bool, absentScore int, absentDefect, absentDetail, presentDetail string) KPI {
	defects := []string{}
	if !found {
		defects = append(defects, absentDefect)
	}
	score := absentScore
	detail := absentDetail
	if found {
		score = 100
		detail = presentDetail
	}
	return KPI{kpi, "adopt", score, detail, defects, []string{}}
}

func kpiFirstCommand(found bool, where string) KPI {
	return kpiPresence("first_command", found, 20,
		"no copy-pasteable no-key/no-model/no-GPU first command in a fenced block of "+strings.Join(firstCommandDocs, ", ")+" (e.g. `fak preflight …`)",
		"no runnable no-setup first command",
		"first command present in "+where)
}

func kpiInstallOneliner(found bool, where string) KPI {
	return kpiPresence("install_oneliner", found, 40,
		"no one-line install (`go install …@latest`) in "+strings.Join(installDocs, ", ")+" — give an agent the one-command install",
		"no one-line install",
		"install one-liner present in "+where)
}

func kpiHonestyLedger(present bool, untagged []string) KPI {
	defects := []string{}
	if !present {
		defects = append(defects, "missing "+claimsFile+" — the honesty ledger an agent trusts (every claim tagged shipped/simulated/stub)")
	} else {
		limit := untagged
		if len(limit) > 8 {
			limit = limit[:8]
		}
		defects = append(defects, limit...)
	}
	soft := []string{}
	if present && len(untagged) > 8 {
		soft = append(soft, "... and "+strconv.Itoa(len(untagged)-8)+" more untagged claim line(s)")
	}
	base := 0.0
	nPresentDefects := 0
	if present {
		base = 100
		nPresentDefects = len(defects)
	}
	detail := "no " + claimsFile
	if present {
		detail = claimsFile + " present, " + strconv.Itoa(len(untagged)) + " untagged claim(s)"
	}
	return KPI{"honesty_ledger", "adopt", mathx.ClampScore(base - 12*float64(nPresentDefects)), detail, defects, soft}
}

func kpiIntegrationRecipes(missing []string) KPI {
	return kpiCoverage("integration_recipes", "adopt", len(requiredRecipes), missing,
		func(label string) string {
			return "no integration recipe for " + label + " — add one under docs/integrations/"
		},
		"agent families have an integration recipe")
}

func kpiCodexRecipeCurrent(gaps []string) KPI {
	n := len(gaps)
	detail := "Codex recipe covers MCP, AGENTS.md, exec JSON, proxy URL, and Responses fence"
	if n > 0 {
		detail = strconv.Itoa(n) + " Codex currentness gap(s)"
	}
	return KPI{"codex_recipe_current", "adopt", mathx.ClampScore(100 - 15*float64(n)), detail, gaps, []string{}}
}

func kpiExtensionScaffold(scaffold, extending bool) KPI {
	defects := []string{}
	if !scaffold {
		defects = append(defects, "no "+leafScaffold+" — the leaf scaffolder an agent runs to add a feature additively")
	}
	if !extending {
		defects = append(defects, "no "+extendingFile+" — the doc that teaches the plug-in/prove-it path")
	}
	detail := "leaf scaffolder + EXTENDING.md present"
	if len(defects) > 0 {
		detail = strconv.Itoa(len(defects)) + " missing extension affordance(s)"
	}
	return KPI{"extension_scaffold", "build", mathx.ClampScore(100 - 50*float64(len(defects))), detail, defects, []string{}}
}

func kpiGuardrailsSurfaced(missing []string) KPI {
	defects := []string{}
	for _, label := range missing {
		defects = append(defects, "enforced rule not surfaced in "+agentsFile+": "+label)
	}
	covered := len(guardrailClusters) - len(missing)
	return KPI{"guardrails_surfaced", "build",
		mathx.ClampScore(100 * float64(covered) / float64(max1(len(guardrailClusters)))),
		strconv.Itoa(covered) + "/" + strconv.Itoa(len(guardrailClusters)) + " enforced rules surfaced up front",
		defects, []string{}}
}

func kpiContributorContract(contributing, linked, greenGate bool) KPI {
	defects := []string{}
	if !contributing {
		defects = append(defects, "no "+contribFile+" — the contributor contract")
	} else if !linked {
		defects = append(defects, contribFile+" exists but is not linked from "+agentsFile+"/README — an agent can't find it")
	}
	if !greenGate {
		defects = append(defects, "no one-command green gate documented (make ci / make test / ./test.ps1) — the feedback loop an agent runs before shipping")
	}
	detail := "CONTRIBUTING linked + green gate documented"
	if len(defects) > 0 {
		detail = strconv.Itoa(len(defects)) + " missing contract/feedback affordance(s)"
	}
	return KPI{"contributor_contract", "build", mathx.ClampScore(100 - 30*float64(len(defects))), detail, defects, []string{}}
}

func kpiMachineConsumable(jsonTools, totalTools int, missing []string) KPI {
	soft := []string{}
	lim := missing
	if len(lim) > 8 {
		lim = lim[:8]
	}
	for _, m := range lim {
		soft = append(soft, "tool without a --json surface: "+m)
	}
	rate := 1.0
	if totalTools != 0 {
		rate = float64(jsonTools) / float64(totalTools)
	}
	detail := "no measurement tools found"
	if totalTools != 0 {
		detail = strconv.Itoa(jsonTools) + "/" + strconv.Itoa(totalTools) + " measurement tools expose --json (" + pctString(rate) + ")"
	}
	return KPI{"machine_consumable", "build", mathx.ClampScore(math.RoundToEven(100 * rate)), detail, []string{}, soft}
}

func kpiCommandVerbsResolve(unknown []string) KPI {
	defects := []string{}
	for _, u := range unknown {
		defects = append(defects, "unknown CLI verb an agent would paste: "+u)
	}
	n := len(defects)
	detail := "every pasted `fak <verb>` resolves to a real dispatched verb"
	if n > 0 {
		detail = strconv.Itoa(n) + " pasted `fak <verb>` command(s) don't resolve to a dispatched verb"
	}
	return KPI{"command_verbs_resolve", "adopt", mathx.ClampScore(100 - 20*float64(n)), detail, defects, []string{}}
}

func kpiRecipeLinksResolve(dead []string) KPI {
	sorted := append([]string{}, dead...)
	sort.Strings(sorted)
	defects := []string{}
	for _, d := range sorted {
		defects = append(defects, "dead recipe link: "+d)
	}
	detail := "every link inside every integration recipe resolves"
	if len(defects) > 0 {
		detail = strconv.Itoa(len(defects)) + " dead link(s) inside the integration recipes"
	}
	return KPI{"recipe_links_resolve", "discover", mathx.ClampScore(100 - 12*float64(len(defects))), detail, defects, []string{}}
}

func kpiAgentConfigValid(bad []string) KPI {
	defects := append([]string{}, bad...)
	detail := mcpConfigFile + " parses and every server names a launch command"
	if len(defects) > 0 {
		detail = strconv.Itoa(len(defects)) + " agent-config integrity defect(s)"
	}
	return KPI{"agent_config_valid", "discover", mathx.ClampScore(100 - 34*float64(len(defects))), detail, defects, []string{}}
}

func kpiFencedPathsResolve(badPaths []string) KPI {
	defects := []string{}
	for _, b := range badPaths {
		defects = append(defects, "unresolvable fenced path: "+b)
	}
	n := len(defects)
	detail := "every fenced command path resolves from a clean clone"
	if n > 0 {
		detail = strconv.Itoa(n) + " fenced command path(s) don't resolve from a clean clone"
	}
	return KPI{"fenced_paths_resolve", "adopt", mathx.ClampScore(100 - 10*float64(n)), detail, defects, []string{}}
}

func kpiFirstCommandRuns(found, policyOK bool, policyRef string, needsKey bool) KPI {
	if !found {
		return KPI{"first_command_runs", "adopt", 100, "no first command to check (see first_command)", []string{}, []string{}}
	}
	defects := []string{}
	if !policyOK {
		ref := policyRef
		if ref == "" {
			ref = "<none parsed>"
		}
		defects = append(defects, "the first command names a policy that doesn't exist on disk: "+ref+" — an agent that pastes it hits a missing-file error")
	}
	if needsKey {
		defects = append(defects, "the first command sold as the no-setup proof secretly needs a key (--api-key-env / --provider) — it is not the no-key/no-model/no-GPU form an agent can run cold")
	}
	detail := "the first command runs cold (policy " + policyRef + " resolves, no key)"
	if len(defects) > 0 {
		detail = strconv.Itoa(len(defects)) + " runnability gap(s) in the first command"
	}
	return KPI{"first_command_runs", "adopt", mathx.ClampScore(100 - 50*float64(len(defects))), detail, defects, []string{}}
}

func kpiPlatformGuidanceConsistent(sellsMake, hasBridge bool) KPI {
	defects := []string{}
	if sellsMake && !hasBridge {
		defects = append(defects, "AGENTS.md sells `make ci` as the green gate but names no native-Windows bridge (scripts/ci.ps1 / ./test.ps1 under WSL) — a Windows agent can't run the gate it's told to run")
	}
	score := 100
	detail := "the green gate names its native-Windows bridge"
	if len(defects) > 0 {
		score = 40
		detail = "make ci sold without a native-Windows bridge"
	}
	return KPI{"platform_guidance_consistent", "build", score, detail, defects, []string{}}
}

func kpiRefusalRecoveryMapped(unmapped []string, total int) KPI {
	defects := []string{}
	for _, t := range unmapped {
		defects = append(defects, "refusal token has no query-backed recovery: "+t+" — add summary/fix metadata to its dos.toml [reasons] block and keep the AGENTS.md query commands usable")
	}
	mapped := total - len(unmapped)
	score := 100
	detail := "no dos.toml [reasons.*] parsed — abstain"
	if total != 0 {
		score = mathx.ClampScore(100 * float64(mapped) / float64(total))
		detail = strconv.Itoa(mapped) + "/" + strconv.Itoa(total) + " kernel refusal tokens have an agent-facing recovery"
	}
	return KPI{"refusal_recovery_mapped", "build", score, detail, defects, []string{}}
}

func kpiQuickstartSuccessSignal(found, hasSignal bool) KPI {
	if !found {
		return KPI{"quickstart_success_signal", "adopt", 100, "no first command to check (see first_command)", []string{}, []string{}}
	}
	defects := []string{}
	if !hasSignal {
		defects = append(defects, "the first-command proof block shows no expected-result marker (`-> DENY/ALLOW`, an exit code, a result line) — an agent can run it but can't tell whether it succeeded; annotate each command with its expected outcome")
	}
	detail := "the proof block shows an observable success signal"
	if len(defects) > 0 {
		detail = "the proof block names no expected outcome"
	}
	return KPI{"quickstart_success_signal", "adopt", mathx.ClampScore(100 - 60*float64(len(defects))), detail, defects, []string{}}
}

func kpiToolchainPinned(hasDirective, docNamed bool) KPI {
	defects := []string{}
	if !hasDirective {
		defects = append(defects, goModFile+" has no `go <version>` directive — an agent can't know which Go toolchain to provision")
	}
	if !docNamed {
		defects = append(defects, "no entry doc (AGENTS.md / README / GETTING-STARTED) names the required Go version — pin it so an agent provisions the right toolchain")
	}
	detail := "go.mod pins the Go version and an entry doc names it"
	if len(defects) > 0 {
		detail = strconv.Itoa(len(defects)) + " toolchain-pin gap(s)"
	}
	return KPI{"toolchain_pinned", "adopt", mathx.ClampScore(100 - 50*float64(len(defects))), detail, defects, []string{}}
}

// ---------------------------------------------------------------------------
// Fold: KPIs -> composite score, grade, friction-debt, control-pane payload.
// ---------------------------------------------------------------------------

func buildPayload(workspace string, kpis []KPI, facts map[string]int, errText string) Payload {
	if errText != "" {
		return Payload{Schema: Schema, OK: false, Verdict: "AUDIT_ERROR", Finding: "tooling_error",
			Reason: errText, NextAction: "fix the read (run from repo ROOT, with git), then re-run",
			Workspace: workspace, Corpus: map[string]any{}, KPIs: []KPI{}}
	}
	byName := make(map[string]KPI, len(kpis))
	for _, k := range kpis {
		byName[k.Kpi] = k
	}
	var scoreSum float64
	for _, n := range kpiWeightOrder {
		if k, ok := byName[n]; ok {
			scoreSum += kpiWeights[n] * float64(k.Score)
		}
	}
	score := round1(scoreSum)
	frictionDebt := 0
	nSoft := 0
	for _, k := range kpis {
		frictionDebt += len(k.Defects)
		nSoft += len(k.Soft)
	}
	grade := scorecard.GradeStd(score)
	frontier, frontierByTerm := experienceFrontier(facts)

	debtByGroup := map[string]int{"discover": 0, "adopt": 0, "build": 0}
	for _, k := range kpis {
		debtByGroup[k.Group] += len(k.Defects)
	}
	scoreByGroup := map[string]float64{"discover": 0, "adopt": 0, "build": 0}
	wsumByGroup := map[string]float64{"discover": 0, "adopt": 0, "build": 0}
	for _, k := range kpis {
		w := kpiWeights[k.Kpi]
		scoreByGroup[k.Group] += w * float64(k.Score)
		wsumByGroup[k.Group] += w
	}
	groupScores := map[string]float64{}
	for _, g := range groups {
		if wsumByGroup[g] != 0 {
			groupScores[g] = round1(scoreByGroup[g] / wsumByGroup[g])
		} else {
			groupScores[g] = 0.0
		}
	}

	breakdown := make([]breakdownRow, 0, len(kpis))
	for _, k := range kpis {
		breakdown = append(breakdown, breakdownRow{k.Kpi, k.Group, k.Score, len(k.Defects), k.Detail})
	}
	sort.SliceStable(breakdown, func(i, j int) bool {
		if breakdown[i].Debt != breakdown[j].Debt {
			return breakdown[i].Debt > breakdown[j].Debt
		}
		return breakdown[i].Score < breakdown[j].Score
	})

	kpiScores := map[string]int{}
	debtByKpi := map[string]int{}
	for _, k := range kpis {
		kpiScores[k.Kpi] = k.Score
		debtByKpi[k.Kpi] = len(k.Defects)
	}

	frontierUnitsCopy := map[string]int{}
	for k, v := range frontierUnits {
		frontierUnitsCopy[k] = v
	}
	corpus := map[string]any{
		"experience_frontier": frontier,
		"frontier_by_term":    frontierByTerm,
		"frontier_units":      frontierUnitsCopy,
		"score":               score,
		"grade":               grade,
		"friction_debt":       frictionDebt,
		"soft_signals":        nSoft,
		"group_scores":        groupScores,
		"debt_by_group":       debtByGroup,
		"kpi_scores":          kpiScores,
		"debt_by_kpi":         debtByKpi,
		"breakdown":           breakdown,
	}

	standing := "discover " + fmt0(groupScores["discover"]) + " · adopt " + fmt0(groupScores["adopt"]) + " · build " + fmt0(groupScores["build"])
	var ok bool
	var verdict, finding, reason, nextAction string
	if frictionDebt == 0 {
		ok, verdict, finding = true, "OK", "agent_ready"
		reason = "experience-frontier " + strconv.Itoa(frontier) + " (unbounded; higher = better) · baseline score " +
			ftoa(score) + "/100 (grade " + grade + "), zero friction-debt across " + strconv.Itoa(len(kpis)) +
			" KPIs (" + standing + "; " + strconv.Itoa(nSoft) + " advisory). An agent can discover, adopt, and build on fak with no missing affordance — and the frontier still has headroom: onboard a harness, map a refusal, expose a --json surface"
		nextAction = "climb the frontier — add the next real affordance (a new integration recipe / harness config, a refusal mapped to a recovery, a tool given --json); hold friction-debt at 0; re-run to prove the climb"
	} else {
		ok, verdict, finding = false, "ACTION", "friction_debt"
		worst := breakdown[0]
		reason = "experience-frontier " + strconv.Itoa(frontier) + " (unbounded) · " + strconv.Itoa(frictionDebt) + " unit(s) of friction-debt; baseline score " +
			ftoa(score) + "/100 (grade " + grade + "); heaviest: " + worst.Kpi + " (" + strconv.Itoa(worst.Debt) + " defect(s)); standing " + standing
		nextAction = "retire friction-debt worst-first (see corpus.breakdown + per-KPI defects): fix the agents.md entry point, the doc-map, the quotable identity, dead orientation links, the first command, the install one-liner, the tagged ledger, the per-agent recipes, the leaf scaffold, the surfaced guardrails, the contributor contract; re-run to prove the drop"
	}

	return Payload{Schema: Schema, OK: ok, Verdict: verdict, Finding: finding, Reason: reason,
		NextAction: nextAction, Workspace: workspace, Corpus: corpus, KPIs: kpis}
}

// ---------------------------------------------------------------------------
// Disk + git gathering (the impure shell around the pure KPIs).
// ---------------------------------------------------------------------------

func gitLines(args []string, root string) []string {
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil
	}
	var lines []string
	for _, ln := range strings.Split(string(out), "\n") {
		if strings.TrimSpace(ln) != "" {
			lines = append(lines, ln)
		}
	}
	return lines
}

func safeRead(root, rel string) string {
	b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
	if err != nil {
		return ""
	}
	return string(b)
}

func fileExists(root, rel string) bool {
	_, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel)))
	return err == nil
}

// localLink is one local link in a doc: the target as written + whether it resolves on disk.
type localLink struct {
	target string
	exists bool
}

func localLinks(text, docRel, root string) []localLink {
	base := filepath.Dir(filepath.Join(root, filepath.FromSlash(docRel)))
	var out []localLink
	seen := map[string]bool{}
	for _, m := range linkRe.FindAllStringSubmatch(text, -1) {
		t := strings.TrimSpace(m[2])
		if strings.HasPrefix(t, "http://") || strings.HasPrefix(t, "https://") ||
			strings.HasPrefix(t, "mailto:") || strings.HasPrefix(t, "#") || strings.HasPrefix(t, "tel:") {
			continue
		}
		pathPart := strings.TrimSpace(strings.SplitN(strings.SplitN(t, "#", 2)[0], "?", 2)[0])
		if pathPart == "" || seen[pathPart] {
			continue
		}
		seen[pathPart] = true
		var resolved string
		if strings.HasPrefix(pathPart, "/") {
			resolved = filepath.Join(root, filepath.FromSlash(strings.TrimLeft(pathPart, "/")))
		} else {
			resolved = filepath.Join(base, filepath.FromSlash(pathPart))
		}
		_, err := os.Stat(resolved)
		out = append(out, localLink{pathPart, err == nil})
	}
	return out
}

func toolHasJSON(text string) bool {
	return strings.Contains(text, `"--json"`) || strings.Contains(text, "'--json'")
}

func isSubstantiveRecipe(text string) bool {
	t := strings.TrimSpace(text)
	if len(t) < recipeMinChars {
		return false
	}
	return strings.Contains(t, "```") || strings.Contains(t, "](")
}

// badFencedPaths flags every fenced path operand that won't resolve in a clean clone.
func badFencedPaths(docTexts map[string]string, root string) []string {
	var bad []string
	for _, doc := range sortedKeys(docTexts) {
		text := docTexts[doc]
		if text == "" {
			continue
		}
		for _, block := range fencedBlocks(text) {
			for _, line := range strings.Split(block, "\n") {
				low := strings.ToLower(line)
				for _, lit := range placeholderLiteral {
					if strings.Contains(low, strings.ToLower(lit)) {
						bad = append(bad, doc+": `"+lit+"…` placeholder in a runnable command — make it a real path or an <angle-bracket> slot")
						break
					}
				}
			}
			for _, op := range pathOperands(block) {
				low := strings.ToLower(op)
				stale := false
				for _, p := range stalePathPrefixes {
					if strings.HasPrefix(low, strings.ToLower(p)) {
						stale = true
						break
					}
				}
				if stale {
					bad = append(bad, doc+": `"+op+"` — stale private-monorepo path (a clean clone has no fleet/ parent; the module IS the repo root)")
					continue
				}
				clean := op
				if strings.HasPrefix(op, "./") {
					clean = op[2:]
				}
				isRepoRel := false
				for _, d := range repoTopDirs {
					if strings.HasPrefix(strings.ToLower(clean), d) {
						isRepoRel = true
						break
					}
				}
				if isRepoRel && !fileExists(root, clean) {
					bad = append(bad, doc+": `"+op+"` — repo-relative path does not exist on disk")
				}
			}
		}
	}
	return dedup(bad)
}

// firstCommandFacts resolves the runnability of the no-setup first command.
func firstCommandFacts(texts map[string]string, root string) (found, policyOK bool, policyRef string, needsKey bool) {
	for _, doc := range firstCommandDocs {
		for _, block := range fencedBlocks(texts[doc]) {
			if !has(block, firstCommandTokens...) {
				continue
			}
			policyRef = ""
			if m := proofPolicyRe.FindStringSubmatch(block); m != nil {
				policyRef = strings.Trim(m[1], "\"'`")
			}
			policyOK = true
			if policyRef != "" && !isTemplateSlot(policyRef) {
				policyOK = fileExists(root, policyRef)
			}
			needsKey = has(block, proofNeedsKeyTokens...)
			return true, policyOK, policyRef, needsKey
		}
	}
	return false, true, "", false
}

func unknownCommandVerbs(docTexts map[string]string, verbs map[string]bool) []string {
	if len(verbs) == 0 {
		return []string{}
	}
	var out []string
	seen := map[string]bool{}
	for _, verb := range commandFamilyVerbs {
		verbs[verb] = true
	}
	for _, command := range devhandoff.Commands {
		verbs[command.Name] = true
	}
	for _, doc := range sortedKeys(docTexts) {
		text := docTexts[doc]
		if text == "" {
			continue
		}
		for _, v := range commandVerbs(text) {
			if verbs[v] {
				continue
			}
			key := doc + ": fak " + v
			if !seen[key] {
				seen[key] = true
				out = append(out, key)
			}
		}
	}
	return out
}

func deadRecipeLinks(root string, recipeTexts map[string]string) []string {
	var dead []string
	for _, rel := range sortedKeys(recipeTexts) {
		text := recipeTexts[rel]
		if text == "" {
			continue
		}
		for _, l := range localLinks(text, rel, root) {
			if !l.exists {
				dead = append(dead, rel+" -> "+l.target)
			}
		}
	}
	return dead
}

func agentConfigIntegrity(root string) []string {
	var bad []string
	if !fileExists(root, mcpConfigFile) {
		return bad
	}
	raw := safeRead(root, mcpConfigFile)
	var data any
	if err := json.Unmarshal([]byte(raw), &data); err != nil {
		return []string{mcpConfigFile + " is present but does not parse as JSON: " + err.Error()}
	}
	obj, isObj := data.(map[string]any)
	if !isObj {
		bad = append(bad, mcpConfigFile+" has no mcpServers map — the harness finds no server to start")
		return bad
	}
	servers, ok := obj["mcpServers"].(map[string]any)
	if !ok || len(servers) == 0 {
		bad = append(bad, mcpConfigFile+" has no mcpServers map — the harness finds no server to start")
		return bad
	}
	for _, name := range sortedKeys(anyMapToStr(servers)) {
		spec := servers[name]
		if strings.HasPrefix(strings.TrimSpace(name), "//") {
			continue
		}
		specMap, isMap := spec.(map[string]any)
		if !isMap {
			bad = append(bad, mcpConfigFile+" server '"+name+"' is not an object — the harness can't read it")
			continue
		}
		if strings.TrimSpace(anyToStr(specMap["command"])) == "" && strings.TrimSpace(anyToStr(specMap["url"])) == "" {
			bad = append(bad, mcpConfigFile+" server '"+name+"' names neither a launch command nor a url — the harness can't start it")
		}
	}
	return bad
}

// Build scores whether an agent can discover, adopt, and build on fak, re-derived from the
// git-tracked tree at root. It is the collect() entry point.
func Build(root string) Payload {
	abs, err := filepath.Abs(root)
	if err != nil || abs == "" {
		abs = root
	}
	if !fileExists(abs, ".git") && gitLines([]string{"rev-parse", "--git-dir"}, abs) == nil {
		return buildPayload(abs, nil, nil, "not a git repo at "+abs+" — run from the repo ROOT")
	}
	kpis, facts := gather(abs)
	return buildPayload(abs, kpis, facts, "")
}
