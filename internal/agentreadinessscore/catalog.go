package agentreadinessscore

import "regexp"

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
	"platform_guidance_consistent", "machine_consumable", "refusal_recovery_mapped", "launch_entry_contract",
}

var kpiWeights = map[string]float64{
	"agents_entrypoint": 0.08, "agent_config": 0.05, "agent_config_valid": 0.03, "llms_map": 0.04,
	"entry_links_resolve": 0.04, "recipe_links_resolve": 0.03, "identity_statement": 0.03,
	"fenced_paths_resolve": 0.06, "command_verbs_resolve": 0.05, "first_command": 0.03,
	"first_command_runs": 0.04, "honesty_ledger": 0.04, "integration_recipes": 0.04,
	"codex_recipe_current": 0.04, "install_oneliner": 0.03, "quickstart_success_signal": 0.03,
	"toolchain_pinned": 0.03, "guardrails_surfaced": 0.06, "contributor_contract": 0.05,
	"extension_scaffold": 0.05, "platform_guidance_consistent": 0.05, "machine_consumable": 0.05,
	"refusal_recovery_mapped": 0.04, "launch_entry_contract": 0.01,
}

// frontierDims is the ordered set of unbounded-frontier dimensions; frontierUnits the
// points-per-unit weight of each.
var frontierDims = []string{"integration_recipes", "harness_configs", "refusal_recoveries", "machine_consumable"}

var frontierUnits = map[string]int{
	"integration_recipes": 8, "harness_configs": 10, "refusal_recoveries": 3, "machine_consumable": 2,
}

// ---- compiled regexes (mirror the Python module-level patterns) -------------------

var (
	bracketSlotRe        = regexp.MustCompile(`<[^>]+>`)
	envSlotRe            = regexp.MustCompile(`^[A-Z][A-Z0-9_]{2,}$`)
	proofPolicyRe        = regexp.MustCompile(`--policy\s+(\S+)`)
	identityDefinitionRe = regexp.MustCompile(`(?i)\bfak\b.{0,80}\b(?:is|acts as|serves as)\b.{0,120}\b(?:agent\s+kernel|kernel|firewall|gate|gateway|proxy|binary|runtime|boundary)\b`)
	identityEnableRe     = regexp.MustCompile(`(?i)\bfak\b.{0,80}\b(?:lets|helps|enables)\s+(?:you\s+)?(?:run|configure|manage|guard|route|build)\b.{0,120}\b(?:agent|task|context|model|tool|permission)\b`)
	identityTaskConfigRe = regexp.MustCompile(`(?i)\bfak\b.{0,80}\bgives\b.{0,40}\b(?:task|agent)s?\b.{0,60}\b(?:configuration|policy|profile)\b`)
	identityBoundaryRe   = regexp.MustCompile(`(?i)\b(?:one|a|the)\s+boundary\b.{0,80}\b(?:manages|handles|routes|records)\b.{0,160}\b(?:context|models?|tools?|permissions?|record)\b`)
	reasonRe             = regexp.MustCompile(`(?m)^\[reasons\.([A-Z][A-Z0-9_]+)\]`)
	verbTokenRe          = regexp.MustCompile(`^([a-z][a-z0-9-]+)`)
	inlineCodeRe         = regexp.MustCompile("`([^`]+)`")
	goDirectiveRe        = regexp.MustCompile(`(?m)^go\s+\d+\.\d+`)
	goVersDocRe          = regexp.MustCompile(`(?i)go\s*1\.\d+`)
	linkRe               = regexp.MustCompile(`\[([^\]]+)\]\(([^)]+)\)`)
	fenceRe              = regexp.MustCompile("^(```|~~~)")
	caseRe               = regexp.MustCompile(`^\s*case\s+(.+):`)
	quotedRe             = regexp.MustCompile(`"([^"]+)"`)
	wsHashRe             = regexp.MustCompile(`\s+#\s`)
	wsRe                 = regexp.MustCompile(`\s+`)
	cdRe                 = regexp.MustCompile(`^cd\s+(\S+)`)
	segSepRe             = regexp.MustCompile(`&&|\|\||;|\||\s--\s`)
	promptRe             = regexp.MustCompile(`^[\$>]\s+`)
	envPrefixRe          = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*=\S*\s+`)
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
