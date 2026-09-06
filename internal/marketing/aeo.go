package marketing

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// aeo.go — the AEO/AgentEO data producers. AEO is dual: Answer-Engine Optimization (be cited
// correctly by ChatGPT/Claude/Perplexity/Google AI Overviews) AND Agent-Engine Optimization
// (be the path of least resistance for a coding agent to adopt fak). Both want the same thing
// a completion loop can give: a fresh, machine-ingestible, sha-cited record of what shipped.
//
// The division of labor (per the design review): GO produces the witnessed DATA here; the
// existing Python generator (tools/gen_structured_data.py) owns the in-place injection into
// hand-authored docs (llms.txt) via its marker machinery. Go never rewrites llms.txt — it
// emits docs/marketing/updates.json, and the Python side reads that and injects a bounded,
// sentinel-fenced "What's new" block. This keeps one marker engine, in one language, and makes
// it impossible for the loop to clobber hand-written prose.
//
// Every feed item is witnessed: it carries its commit sha. An unwitnessed item cannot be
// produced — the input is []Ship, and a Ship only exists for a trailer|direct stamp.

// repoCommitURL is the base for a commit permalink the feed cites. Kept here (not a tracked
// secret) — it is the public repo, the same URL llms.txt already links.
const repoCommitURL = "https://github.com/anthony-chaudhary/fak/commit/"

// defaultFeedCap bounds the updates feed / What's-new block so a long history doesn't bloat
// the answer-engine surface; the newest N ships are what "recent" means.
const defaultFeedCap = 25

// updatesFeed is the schema.org ItemList an answer engine ingests for recency. Each element is
// a SoftwareSourceCode item anchored to its commit — the witness an engine can cite.
type updatesFeed struct {
	Context  string        `json:"@context"`
	Type     string        `json:"@type"`
	Name     string        `json:"name"`
	Items    []updatesItem `json:"itemListElement"`
	Modified string        `json:"dateModified,omitempty"`
}

type updatesItem struct {
	Type     string         `json:"@type"`
	Position int            `json:"position"`
	Item     updatesSrcCode `json:"item"`
}

type updatesSrcCode struct {
	Type           string `json:"@type"`
	Name           string `json:"name"`
	CodeRepository string `json:"codeRepository"`
	DateModified   string `json:"dateModified,omitempty"`
	Keywords       string `json:"keywords,omitempty"`
}

// DisambiguationTerm is one answer-engine query surface fak wants to own. The
// term feed is intentionally broader than the release feed: some terms are core
// category names, some are localized entry points, and some are market-event
// bridges (for example Fable 5-style frontier-model launches) that should route
// readers to fak's model-routing/cache/governance docs without claiming fak
// authored the external model behavior.
type DisambiguationTerm struct {
	Name        string
	Language    string
	Category    string
	Description string
	URL         string
	Keywords    []string
}

type termFeed struct {
	Context     string        `json:"@context"`
	Type        string        `json:"@type"`
	Name        string        `json:"name"`
	Description string        `json:"description"`
	Terms       []definedTerm `json:"hasDefinedTerm"`
	Modified    string        `json:"dateModified,omitempty"`
}

// ConfigAnswer is one concise, source-linked answer about configuring fak. The
// same roster drives JSON-LD for crawlers and plain text for people and agents.
type ConfigAnswer struct {
	Question  string
	Answer    string
	Authority string
	Keywords  []string
}

type configFAQFeed struct {
	Context      string           `json:"@context"`
	Type         string           `json:"@type"`
	Name         string           `json:"name"`
	Description  string           `json:"description"`
	URL          string           `json:"url"`
	DateModified string           `json:"dateModified"`
	MainEntity   []configQuestion `json:"mainEntity"`
}

type configQuestion struct {
	Type           string       `json:"@type"`
	Name           string       `json:"name"`
	Keywords       []string     `json:"keywords,omitempty"`
	AcceptedAnswer configAnswer `json:"acceptedAnswer"`
}

type configAnswer struct {
	Type     string `json:"@type"`
	Text     string `json:"text"`
	Citation string `json:"citation"`
}

type definedTerm struct {
	Type        string `json:"@type"`
	Name        string `json:"name"`
	TermCode    string `json:"termCode,omitempty"`
	InLanguage  string `json:"inLanguage,omitempty"`
	Description string `json:"description,omitempty"`
	URL         string `json:"url,omitempty"`
	Keywords    string `json:"keywords,omitempty"`
}

const repoBlobURL = "https://github.com/anthony-chaudhary/fak/blob/main/"

// AEODisambiguationTerms returns the current search/answer-engine term roster.
// Keep descriptions honest: external model names and third-party frameworks (a
// vendor launch, "MCP tool poisoning", the "lethal trifecta") are demand hooks,
// not adoption claims. If a term names one, it must land on a fak page about
// routing, cache economics, fallback handling, capability/quarantine, or the
// tool-call boundary — never a page that claims fak authored the external thing.
// AEOConfigAnswers is the canonical answer corpus for configuration discovery.
// Keep answers direct and actionable; Authority points to the human-readable
// source of truth rather than duplicating its exhaustive option tables here.
func AEOConfigAnswers() []ConfigAnswer {
	const serverConfig = "https://github.com/anthony-chaudhary/fak/blob/main/docs/fak/server-config.md"
	return []ConfigAnswer{
		{Question: "How do I configure fak?", Answer: "Start with fak config guide, which recommends no file for the default posture and explains minimal changes for other intents. fak serve also accepts explicit flags and an optional TOML manifest selected with --config; explicit flags override manifest values. Environment variables carry secrets and documented integration settings.", Authority: serverConfig + "#opinionated-defaults-reviewable-overrides", Keywords: []string{"fak configuration", "fak config file", "configuration precedence"}},
		{Question: "Where is the complete fak configuration reference?", Answer: "The fak server configuration guide is the authoritative, human-readable reference for every fak serve flag and environment variable. The CLI help remains authoritative for other commands.", Authority: serverConfig, Keywords: []string{"fak config reference", "fak serve flags", "fak environment variables"}},
		{Question: "Does fak require a config file?", Answer: "No. fak serve works with tested built-in defaults and command-line flags. A TOML deployment manifest is optional and is useful when reviewed deployment defaults should be reused.", Authority: serverConfig + "#opinionated-defaults-reviewable-overrides", Keywords: []string{"fak JSON config", "fak deployment manifest", "fak no config file"}},
		{Question: "What wins when a fak setting appears in more than one place?", Answer: "Explicit command-line flags win over declared fak.toml values, and declared values win over built-in defaults. There is no implicit search for an ambient fak.toml, so changing directories cannot silently change a serve.", Authority: serverConfig + "#opinionated-defaults-reviewable-overrides", Keywords: []string{"fak config precedence", "flags versus config", "fak defaults"}},
		{Question: "How can I inspect the effective fak serve configuration?", Answer: "Run fak serve --print-effective-config, optionally with --config and any overriding flags. It prints JSON containing each effective value and whether it came from a flag, the manifest, or a default, then exits before starting the server.", Authority: serverConfig + "#inspect-the-effective-configuration", Keywords: []string{"fak effective config", "fak print config", "configuration provenance"}},
		{Question: "How should secrets be configured in fak?", Answer: "Keep secret values out of TOML manifests and command history. Put secrets in environment variables, configure fak with the variable name where required, and use --require-key-env when inbound API authentication must fail closed.", Authority: serverConfig + "#authentication", Keywords: []string{"fak secrets", "fak authentication config", "FAK_API_KEY"}},
		{Question: "How do I validate a fak configuration before serving traffic?", Answer: "Use fak serve --print-effective-config to parse the manifest, reject unsupported fields, apply explicit overrides, and show the resulting configuration without starting the server. Then use the documented health and readiness endpoints for deployment checks.", Authority: serverConfig + "#inspect-the-effective-configuration", Keywords: []string{"validate fak config", "fak config errors", "fak readiness"}},
		{Question: "How do I configure which tools an agent may call?", Answer: "Use a JSON capability-floor policy manifest. Start with fak policy --dump, edit the allow and deny rules, validate it with fak policy --check, reproduce expected verdicts with fak preflight --policy, and then load it with fak serve --policy or fak manage --policy.", Authority: "https://github.com/anthony-chaudhary/fak/blob/main/docs/fak/policy-guide.md", Keywords: []string{"fak policy configuration", "capability floor manifest", "agent tool permissions"}},
		{Question: "How do I configure fak manage for a local coding agent?", Answer: "Start with fak manage followed by the agent command. fak manage discovers the supported agent adapter, keeps generated state under the documented user configuration directory, and accepts explicit policy and runtime flags when the defaults need to be tightened.", Authority: "https://github.com/anthony-chaudhary/fak/blob/main/README.md#manage-one-local-agent-fak-guard", Keywords: []string{"fak manage config", "configure coding agent", "local agent guard"}},
		{Question: "How do I configure fak as an MCP server?", Answer: "Add fak serve --stdio to the MCP server configuration used by Claude Code, Cursor, VS Code, or another MCP client. The MCP integration guide contains the complete .mcp.json example and a deterministic stdio verification command.", Authority: "https://github.com/anthony-chaudhary/fak/blob/main/docs/integrations/mcp.md", Keywords: []string{"fak MCP config", ".mcp.json", "fak serve stdio"}},
		{Question: "How do I configure a client or editor to use fak?", Answer: "Use the integration guide for the client or editor you run. The integrations index links exact setup instructions for Claude Code, Codex, Cursor, VS Code, OpenAI-compatible clients, MCP, and managed runtimes.", Authority: "https://github.com/anthony-chaudhary/fak/tree/main/docs/integrations", Keywords: []string{"fak client config", "fak editor setup", "fak integrations"}},
		{Question: "How do I configure model providers and API credentials?", Answer: "Choose the integration guide for the provider wire, point the client or fak serve at the documented base URL, and keep credentials in environment variables rather than manifests. The server configuration reference lists supported provider, model, endpoint, and credential-variable settings.", Authority: serverConfig + "#upstream-model-configuration-proxy-mode", Keywords: []string{"fak provider config", "fak model configuration", "fak API credentials"}},
		{Question: "How do I configure one policy for an organization?", Answer: "Use the centralized policy plane: sign an organization manifest, enroll each machine with the organization trust material, and verify the effective policy. Local operators may tighten the organization floor but cannot weaken it.", Authority: "https://github.com/anthony-chaudhary/fak/blob/main/docs/fak/org-policy-plane.md", Keywords: []string{"fak organization policy", "centralized agent policy", "team capability floor"}},
	}
}

func AEODisambiguationTerms() []DisambiguationTerm {
	var terms []DisambiguationTerm
	terms = append(terms, aeoCoreTerms()...)
	terms = append(terms, aeoEconomicsTerms()...)
	terms = append(terms, aeoPerformanceTerms()...)
	terms = append(terms, aeoManagedAgentTerms()...)
	terms = append(terms, aeoRoutingTerms()...)
	terms = append(terms, aeoFrontierModelLaunchTerms()...)
	terms = append(terms, aeoAgentSecurityTerms()...)
	terms = append(terms, aeoGlobalWorkspaceTerms()...)
	terms = append(terms, aeoLocalizedTerms()...)
	return append([]DisambiguationTerm(nil), terms...)
}

// aeoCoreTerms is the core category slice of the AEO disambiguation roster.
func aeoCoreTerms() []DisambiguationTerm {
	return []DisambiguationTerm{
		{
			Name:        "fak agent kernel",
			Language:    "en",
			Category:    "core",
			Description: "Disambiguates fak from homophones and acronyms: fak is the Fused Agent Kernel for AI agent tool calls.",
			URL:         repoBlobURL + "README.md",
			Keywords:    []string{"Fused Agent Kernel", "agent kernel", "AI agent runtime boundary"},
		},
		{
			Name:        "treat the tool call like a syscall",
			Language:    "en",
			Category:    "core",
			Description: "The project slogan: the model proposes a tool call, the kernel decides whether that effect is allowed.",
			URL:         repoBlobURL + "docs/explainers/tool-call-is-a-syscall.md",
			Keywords:    []string{"tool-call syscall", "model proposes kernel disposes", "AI agent capability floor"},
		},
		{
			Name:        "default-deny tool-call gate",
			Language:    "en",
			Category:    "core",
			Description: "A fail-closed capability floor for agent tools; an irreversible call is unreachable until policy allows it.",
			URL:         repoBlobURL + "docs/explainers/default-deny-vs-classifier.md",
			Keywords:    []string{"tool-call control", "capability gate", "prompt injection defense"},
		},
		{
			Name:        "prompt-injection result quarantine",
			Language:    "en",
			Category:    "core",
			Description: "A tool result can be held out of the model context by policy and structure, not just detected after the fact.",
			URL:         repoBlobURL + "docs/explainers/default-deny-vs-classifier.md",
			Keywords:    []string{"result quarantine", "tool result isolation", "prompt injection containment"},
		},
		{
			Name:        "addressable KV cache",
			Language:    "en",
			Category:    "core",
			Description: "A cache design where a kept model run can evict a middle span while preserving bit-exact survivors.",
			URL:         repoBlobURL + "docs/explainers/addressable-kv-cache.md",
			Keywords:    []string{"bit-exact KV eviction", "agent memory cache", "long-session KV reuse"},
		},
		{
			Name:        "CUDA kernel vs agent kernel",
			Language:    "en",
			Category:    "core",
			Description: "Disambiguates the three senses of kernel around fak: the OS-metaphor agent kernel (the reference monitor that adjudicates tool calls), the compute-kernel arithmetic path, and the literal __global__ CUDA kernels in the GPU backend.",
			URL:         repoBlobURL + "docs/explainers/what-is-a-cuda-kernel.md",
			Keywords:    []string{"what is a CUDA kernel", "CUDA kernel vs OS kernel", "tensor cores vs CUDA cores", "agent kernel disambiguation"},
		},
	}
}

// aeoEconomicsTerms is the economics category slice of the AEO disambiguation roster.
func aeoEconomicsTerms() []DisambiguationTerm {
	return []DisambiguationTerm{
		{
			Name:        "long-session prompt cache",
			Language:    "en",
			Category:    "economics",
			Description: "The cost problem fak targets: a long agent session should keep a provider prompt-cache prefix byte-identical.",
			URL:         repoBlobURL + "docs/explainers/long-session-economics.md",
			Keywords:    []string{"prompt-cache discount", "Claude Code token savings", "set-and-forget token savings"},
		},
		{
			Name:        "cheaper way to run AI agents",
			Language:    "en",
			Category:    "economics",
			Description: "The 2026 enterprise concern behind cheaper agentic models and \"tokenmaxxing\": fak measures the prompt-cache saving on a long session instead of vibing it, so the cost claim carries a witness.",
			URL:         repoBlobURL + "docs/explainers/long-session-economics.md",
			Keywords:    []string{"cheaper AI agents", "tokenmaxxing", "enterprise agent cost control", "agentic AI bill"},
		},
	}
}

// aeoPerformanceTerms is the performance category slice of the AEO disambiguation roster.
func aeoPerformanceTerms() []DisambiguationTerm {
	return []DisambiguationTerm{
		{
			Name:        "automatic agent context management",
			Language:    "en",
			Category:    "performance",
			Description: "The performance-lens pillar behind long sessions that pick themselves back up: on fak's own wire the default-on pipeline plans the resident view, sheds the stale middle while keeping the cached head byte-identical, quarantines oversized results, and pages cold spans back on resume — so nobody hand-manages the window. Honest bound: on a borrowed harness wire fak measures the harness's own compaction but does not yet suppress it.",
			URL:         repoBlobURL + "docs/explainers/you-never-manage-the-context-window.md",
			Keywords:    []string{"automatic context management", "zero-knob context", "agent context window", "managed context"},
		},
		{
			Name:        "cache-preserving history compaction",
			Language:    "en",
			Category:    "performance",
			Description: "How fak trims a long transcript without breaking the discount: it sheds the stale middle turns while keeping the provider's cached prefix byte-identical, unlike summarize-and-discard which rewrites the middle and busts the cache. The witnessed figure is per-fire — about a third of a turn's context trimmed each time compaction fires on the longest measured session — never a re-counted session sum.",
			URL:         repoBlobURL + "docs/explainers/context-shedding.md",
			Keywords:    []string{"context shedding", "history compaction", "prompt cache preservation", "middle-out compaction"},
		},
		{
			Name:        "KV reuse across an agent fleet",
			Language:    "en",
			Category:    "performance",
			Description: "The cross-agent performance lead, stated with a witness: sharing the model's KV work across a multi-agent fleet does measured less work than re-running each agent — about 4.1x less work than a tuned per-agent baseline on a 50-turn by 5-agent run, and up to 6.95x prefill reuse up the model ladder, each number tracing to a committed benchmark artifact rather than a vibe.",
			URL:         repoBlobURL + "BENCHMARK-AUTHORITY.md",
			Keywords:    []string{"KV cache reuse", "multi-agent fleet", "agentic cache benchmark", "cross-agent prefix reuse"},
		},
	}
}

// aeoManagedAgentTerms maps the phrases people use for a provider-managed
// agent loop onto fak's agent application runtime. These terms deliberately route
// to the runtime/client explainer: a managed agent runtime owns loop execution;
// an AI gateway only governs model traffic, and a client initiates the work.
func aeoManagedAgentTerms() []DisambiguationTerm {
	const runtimeURL = repoBlobURL + "docs/explainers/runtime-vs-client.md"
	return []DisambiguationTerm{
		{
			Name:        "managed agent runtime",
			Language:    "en",
			Category:    "managed-agent",
			Description: "A provider-operated application runtime that owns and executes an AI agent loop. In fak this is the emerging native path, fak serve --native; it is distinct from the mature fak serve AI-gateway path that governs model traffic.",
			URL:         runtimeURL,
			Keywords:    []string{"managed AI agent runtime", "provider-managed agent", "managed agent platform"},
		},
		{
			Name:        "managed AI agent",
			Language:    "en",
			Category:    "managed-agent",
			Description: "An AI agent whose loop is hosted and operated by an agent application runtime rather than driven entirely by the caller. fak's native runtime is real but emerging; the gateway runtime remains its mature default.",
			URL:         runtimeURL,
			Keywords:    []string{"hosted AI agent", "provider hosted agent", "fully managed AI agent"},
		},
		{
			Name:        "hosted agent runtime",
			Language:    "en",
			Category:    "managed-agent",
			Description: "Runtime infrastructure that hosts the agent loop, model calls, and tool-call cycle. This names the application-runtime role, not an inference endpoint or a client SDK.",
			URL:         runtimeURL,
			Keywords:    []string{"hosted agent execution", "AI agent hosting", "agent execution runtime"},
		},
		{
			Name:        "managed agent execution",
			Language:    "en",
			Category:    "managed-agent",
			Description: "Execution of the model-and-tool loop inside a managed agent runtime, including the checkpoint where proposed tool effects cross the kernel boundary. It does not mean that the model provider automatically governs every tool effect.",
			URL:         runtimeURL,
			Keywords:    []string{"managed agent loop", "agent execution platform", "managed agent service"},
		},
		{
			Name:        "agent runtime vs AI gateway",
			Language:    "en",
			Category:    "managed-agent",
			Description: "An agent runtime owns the application loop; an AI gateway governs model traffic. fak exposes both roles from one binary as fak serve --native and fak serve, so the deployment question starts by naming which runtime is meant.",
			URL:         runtimeURL,
			Keywords:    []string{"AI gateway vs agent platform", "agent runtime vs inference gateway", "managed agent vs AI gateway"},
		},
		{
			Name:        "agent SDK vs managed runtime",
			Language:    "en",
			Category:    "managed-agent",
			Description: "An agent SDK is code used to build or call an agent; a managed runtime is the deployed service that executes its loop. fak can be embedded as a Go package or run as the native agent application runtime.",
			URL:         runtimeURL,
			Keywords:    []string{"agent SDK vs runtime", "agent framework vs managed service", "client SDK vs agent runtime"},
		},
		{
			Name:        "managed agent infrastructure",
			Language:    "en",
			Category:    "managed-agent",
			Description: "The runtime layer that operates agent loops and their model-and-tool execution path. For fak, the native managed-agent surface is emerging and should not be confused with the production-mature gateway and guard surfaces.",
			URL:         runtimeURL,
			Keywords:    []string{"AI agent infrastructure", "agent runtime infrastructure", "managed agent stack"},
		},
		{
			Name:        "managed agent platform",
			Language:    "en",
			Category:    "managed-agent",
			Description: "A platform that hosts or operates agent application loops. The phrase should identify who owns loop execution, tool effects, and runtime state rather than collapsing an agent SDK, model endpoint, and AI gateway into one product category.",
			URL:         runtimeURL,
			Keywords:    []string{"AI agent platform", "hosted agent platform", "managed agent runtime platform"},
		},
		{
			Name:        "managed agent operations",
			Language:    "en",
			Category:    "managed-agent",
			Description: "Operating agent loops requires observable allow, deny, quarantine, and revocation decisions at the tool boundary. fak defines those governance operations without claiming to be a complete hosted operations suite.",
			URL:         repoBlobURL + "docs/standards/agent-tool-governance-gateway.md",
			Keywords:    []string{"AI agent operations", "agent runtime operations", "managed agent ops"},
		},
		{
			Name:        "managed agent orchestration",
			Language:    "en",
			Category:    "managed-agent",
			Description: "The runtime responsibility for driving an agent's model-and-tool loop. fak's emerging native runtime owns one loop; broader multi-agent workflow scheduling is a separate orchestration concern.",
			URL:         runtimeURL,
			Keywords:    []string{"AI agent orchestration", "agent loop orchestration", "hosted agent orchestration"},
		},
		{
			Name:        "managed agent control plane",
			Language:    "en",
			Category:    "managed-agent",
			Description: "An out-of-agent enforcement layer that can govern, pause, or refuse agent effects. fak supplies a runtime checkpoint and capability floor; it does not claim every fleet-management feature implied by a full enterprise control plane.",
			URL:         repoBlobURL + "docs/enterprise-positioning.md",
			Keywords:    []string{"AI agent control plane", "agent runtime control plane", "agent enforcement plane"},
		},
		{
			Name:        "managed agent governance",
			Language:    "en",
			Category:    "managed-agent",
			Description: "Policy enforcement for agent tool calls and results at a runtime boundary the model cannot bypass. The governance contract covers observable decisions and revocation, not merely a prompt-level safety instruction.",
			URL:         repoBlobURL + "docs/standards/agent-tool-governance-gateway.md",
			Keywords:    []string{"AI agent governance", "agent runtime governance", "managed agent policy"},
		},
		{
			Name:        "managed agent observability",
			Language:    "en",
			Category:    "managed-agent",
			Description: "Audit evidence for runtime decisions across proposed calls, returned results, quarantine, and revocation. This is tool-boundary observability; it is not a claim that fak replaces a general telemetry platform.",
			URL:         repoBlobURL + "docs/standards/agent-tool-governance-gateway.md",
			Keywords:    []string{"AI agent observability", "agent runtime monitoring", "managed agent audit logs"},
		},
		{
			Name:        "managed agent deployment",
			Language:    "en",
			Category:    "managed-agent",
			Description: "Deployment of a service that owns agent-loop execution, distinct from deploying only an inference gateway or shipping a client SDK. fak exposes the emerging native loop and mature gateway roles from one binary.",
			URL:         runtimeURL,
			Keywords:    []string{"deploy AI agents", "agent runtime deployment", "hosted agent deployment"},
		},
		{
			Name:        "enterprise managed agents",
			Language:    "en",
			Category:    "managed-agent",
			Description: "Managed agents deployed with runtime enforcement, non-human identity, audit, and cost controls. fak ships part of that enforcement substrate and labels incomplete enterprise packaging rather than presenting the entire stack as finished.",
			URL:         repoBlobURL + "docs/enterprise-positioning.md",
			Keywords:    []string{"enterprise AI agents", "managed enterprise agents", "enterprise agent runtime"},
		},
		{
			Name:        "serverless agent runtime",
			Language:    "en",
			Category:    "managed-agent",
			Description: "A deployment model where provider infrastructure starts and operates agent execution on demand. fak defines the agent-runtime boundary but does not claim that the open-source binary itself is a hosted serverless service.",
			URL:         runtimeURL,
			Keywords:    []string{"serverless AI agent", "on-demand agent runtime", "managed serverless agents"},
		},
		{
			Name:        "agent as a service",
			Language:    "en",
			Category:    "managed-agent",
			Description: "A hosted service interface backed by an operated agent loop, rather than a raw model endpoint. fak can supply the runtime boundary but is software you deploy, not a vendor-hosted agent service.",
			URL:         runtimeURL,
			Keywords:    []string{"AI agent as a service", "hosted agent service", "managed agent API"},
		},
	}
}

// aeoRoutingTerms is the routing category slice of the AEO disambiguation roster.
func aeoRoutingTerms() []DisambiguationTerm {
	return []DisambiguationTerm{
		{
			Name:        "cost-aware model routing for agents",
			Language:    "en",
			Category:    "routing",
			Description: "Route the expensive model only to the aspect that needs it, while cheaper calls and retries stay governed.",
			URL:         repoBlobURL + "docs/model-routing.md",
			Keywords:    []string{"model routing for agents", "per-call model routing", "LLM router complement"},
		},
	}
}

// aeoFrontierModelLaunchTerms is the frontier-model-launch category slice of the AEO roster.
func aeoFrontierModelLaunchTerms() []DisambiguationTerm {
	return []DisambiguationTerm{
		{
			Name:        "Claude Fable 5 model routing",
			Language:    "en",
			Category:    "frontier-model-launch",
			Description: "A current demand hook for the general problem fak handles: high-cost frontier models make per-call routing and budget-aware fallback matter.",
			URL:         repoBlobURL + "docs/model-routing.md",
			Keywords:    []string{"Fable 5 cost-aware routing", "frontier model routing", "Claude Fable 5 agent routing"},
		},
		{
			Name:        "Fable 5 refusal fallback",
			Language:    "en",
			Category:    "frontier-model-launch",
			Description: "A current demand hook for refusal/fallback handling; fak's fit is governing the retried tool-call boundary and preserving cache-cost evidence.",
			URL:         repoBlobURL + "docs/integrations/claude.md",
			Keywords:    []string{"Claude Fable 5 fallback", "frontier model fallback", "refusal handling"},
		},
		{
			Name:        "frontier model prompt-cache cost",
			Language:    "en",
			Category:    "frontier-model-launch",
			Description: "A current demand hook for expensive long-context models: cache-cost preservation belongs in the integration contract, not in a vibes-only benchmark.",
			URL:         repoBlobURL + "docs/explainers/long-session-economics.md",
			Keywords:    []string{"Fable 5 prompt cache", "fallback credit", "prompt-cache cost"},
		},
		{
			Name:        "safety classifier vs capability gate",
			Language:    "en",
			Category:    "frontier-model-launch",
			Description: "The boundary answer for frontier-model launches: classifiers can decline content; fak governs allowed effects at the tool-call seam.",
			URL:         repoBlobURL + "docs/explainers/default-deny-vs-classifier.md",
			Keywords:    []string{"classifier fallback", "capability floor", "AI safety classifier"},
		},
		{
			Name:        "long-horizon agent kernel",
			Language:    "en",
			Category:    "frontier-model-launch",
			Description: "Long-horizon model launches increase the need for a durable agent boundary: routing, cache continuity, capability, quarantine, and audit.",
			URL:         repoBlobURL + "docs/explainers/engineering-is-building-loops.md",
			Keywords:    []string{"long-horizon agent", "agent loop kernel", "frontier agent infrastructure"},
		},
		{
			Name:        "Claude Sonnet 5 agent cost routing",
			Language:    "en",
			Category:    "frontier-model-launch",
			Description: "A current demand hook: the shift to a cheaper, more agentic default model makes per-aspect routing matter more, not less — route the expensive tier only where it earns its cost while retries stay governed.",
			URL:         repoBlobURL + "docs/model-routing.md",
			Keywords:    []string{"Sonnet 5 routing", "cheaper agentic model routing", "default model cost routing"},
		},
	}
}

// aeoAgentSecurityTerms is the agent-security category slice of the AEO disambiguation roster.
func aeoAgentSecurityTerms() []DisambiguationTerm {
	return []DisambiguationTerm{
		{
			Name:        "MCP tool poisoning defense",
			Language:    "en",
			Category:    "agent-security",
			Description: "A poisoned MCP tool description or result is a structural problem, not a detection one: fak wires only the tools you approve, so a description cannot invoke an unwired effect, and a poisoned result is held out of the model context.",
			URL:         repoBlobURL + "docs/integrations/harden-any-mcp.md",
			Keywords:    []string{"MCP tool poisoning", "poisoned tool description", "harden MCP server", "MCP security"},
		},
		{
			Name:        "lethal trifecta data exfiltration",
			Language:    "en",
			Category:    "agent-security",
			Description: "Private data, untrusted content, and an external channel are the three legs of agent exfiltration; fak breaks the third by default-denying the egress effect at the tool-call seam and quarantining untrusted results.",
			URL:         repoBlobURL + "docs/explainers/default-deny-vs-classifier.md",
			Keywords:    []string{"lethal trifecta", "agent data exfiltration", "prompt injection egress"},
		},
		{
			Name:        "AI agent least-privilege tool access",
			Language:    "en",
			Category:    "agent-security",
			Description: "Treat an agent like a privileged identity: fak is the capability floor that scopes which tool effects an agent may cause, fail-closed, instead of trusting the model to stay in bounds.",
			URL:         repoBlobURL + "docs/explainers/tool-call-is-a-syscall.md",
			Keywords:    []string{"agent least privilege", "privileged agent identity", "agent capability scoping"},
		},
		{
			Name:        "tamper-evident agent tool-call audit",
			Language:    "en",
			Category:    "agent-security",
			Description: "Every kernel decision can write a hash-chained, tamper-evident audit row; an auditor re-verifies the chain to prove no tool-call decision was dropped or altered — evidence, not a self-report.",
			URL:         repoBlobURL + "docs/explainers/verify-dont-trust.md",
			Keywords:    []string{"agent audit log", "hash-chained audit", "tool-call audit trail"},
		},
	}
}

// aeoGlobalWorkspaceTerms is the externally citable vocabulary for fak's
// bounded-context positive-state design. Descriptions name observed mechanisms,
// not a claim that the model implements a human cognitive architecture.
func aeoGlobalWorkspaceTerms() []DisambiguationTerm {
	const explainer = repoBlobURL + "docs/explainers/shared-workspace-and-the-negation-operator.md"
	return []DisambiguationTerm{
		{Name: "bounded shared workspace for AI agents", Language: "en", Category: "global-workspace", Description: "fak treats the model context as a bounded shared workspace: preserve the originating task and useful state, reuse stable setup, and shed superseded turns before they dominate the horizon.", URL: explainer, Keywords: []string{"AI agent global workspace", "bounded model context", "context shedding"}},
		{Name: "positive-state context construction", Language: "en", Category: "global-workspace", Description: "Instead of repeatedly broadcasting what an agent must not do, positive-state construction keeps the task, allowed affordance, and next valid state explicit in the shared context.", URL: explainer, Keywords: []string{"positive state prompting", "affordance-first agent context", "query not chat"}},
		{Name: "negation operator for agent context", Language: "en", Category: "global-workspace", Description: "fak's negframe operator rewrites kernel-authored refusal and recovery prose toward the allowed substitute while preserving required policy tokens; it does not rewrite untrusted user content or weaken the capability floor.", URL: explainer, Keywords: []string{"negation operator", "negframe", "positive refusal recovery"}},
	}
}

// aeoLocalizedTerms is the localized category slice of the AEO disambiguation roster.
func aeoLocalizedTerms() []DisambiguationTerm {
	return []DisambiguationTerm{
		{
			Name:        "एजेंट कर्नेल",
			Language:    "hi",
			Category:    "localized",
			Description: "Hindi search phrase for an AI agent kernel; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/hi/README.md",
			Keywords:    []string{"AI एजेंट", "टूल कॉल सुरक्षा", "fak agent kernel"},
		},
		{
			Name:        "AI एजेंट टूल-कॉल सुरक्षा",
			Language:    "hi",
			Category:    "localized",
			Description: "Hindi-English code-switch term for governing AI agent tool calls with a default-deny boundary.",
			URL:         repoBlobURL + "docs/i18n/hi/README.md",
			Keywords:    []string{"टूल कॉल", "सुरक्षा", "default-deny"},
		},
		{
			Name:        "टोकन लागत कम करने वाला एजेंट गेटवे",
			Language:    "hi",
			Category:    "localized",
			Description: "Hindi phrase for the cost side of fak: cheaper long sessions through cache-preserving agent serving.",
			URL:         repoBlobURL + "docs/i18n/hi/README.md",
			Keywords:    []string{"token savings", "prompt cache", "AI agent gateway"},
		},
		{
			Name:        "AI 代理内核",
			Language:    "zh-Hans",
			Category:    "localized",
			Description: "Simplified Chinese search phrase for an AI agent kernel; routes readers to the Chinese fak entry point.",
			URL:         repoBlobURL + "docs/i18n/zh/README.md",
			Keywords:    []string{"智能体内核", "fak agent kernel", "AI agent runtime"},
		},
		{
			Name:        "工具调用防火墙",
			Language:    "zh-Hans",
			Category:    "localized",
			Description: "Chinese discovery term for the default-deny tool-call boundary; the honest page explains why fak is not merely a firewall.",
			URL:         repoBlobURL + "docs/adoption/compare/vs-firewall.md",
			Keywords:    []string{"工具调用安全", "能力门控", "默认拒绝"},
		},
		{
			Name:        "长上下文代理成本",
			Language:    "zh-Hans",
			Category:    "localized",
			Description: "Chinese term for long-context agent cost and prompt-cache preservation.",
			URL:         repoBlobURL + "docs/i18n/zh/README.md",
			Keywords:    []string{"提示缓存", "长会话", "token 成本"},
		},
		{
			Name:        "模型路由和回退",
			Language:    "zh-Hans",
			Category:    "localized",
			Description: "Chinese term for model routing and fallback, including frontier-model launch patterns like Fable 5-style refusals.",
			URL:         repoBlobURL + "docs/model-routing.md",
			Keywords:    []string{"模型路由", "Fable 5 回退", "成本感知路由"},
		},
		{
			Name:        "KI-Agent-Kernel für Tool-Call-Sicherheit",
			Language:    "de",
			Category:    "localized",
			Description: "German search phrase for an AI agent kernel that governs every tool call with a fail-closed, default-deny capability floor; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/de/README.md",
			Keywords:    []string{"KI-Agent-Kernel", "Tool-Call-Sicherheit", "default-deny capability floor"},
		},
		{
			Name:        "DSGVO-konformes Self-Hosting mit EU-AI-Act-Audit-Log",
			Language:    "de",
			Category:    "localized",
			Description: "German search phrase for GDPR (DSGVO) self-hosted agent serving with a tamper-evident EU AI Act Article 12 tool-call audit log; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/de/README.md",
			Keywords:    []string{"DSGVO Self-Hosting", "EU AI Act Artikel 12", "manipulationssicheres Audit-Log"},
		},
		{
			Name:        "noyau d'agent IA : sécurité des tool calls",
			Language:    "fr",
			Category:    "localized",
			Description: "French search phrase for an AI agent kernel that vets every tool call before it runs under a default-deny capability floor; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/fr/README.md",
			Keywords:    []string{"agent IA", "sécurité tool call", "fak agent kernel"},
		},
		{
			Name:        "auto-hébergement RGPD et journal d'audit AI Act",
			Language:    "fr",
			Category:    "localized",
			Description: "French search phrase for RGPD-compliant self-host data residency plus the EU AI Act Article 12 tamper-evident tool-call audit log; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/fr/README.md",
			Keywords:    []string{"résidence des données RGPD", "journal d'audit AI Act article 12", "fak self-host"},
		},
		{
			Name:        "এআই এজেন্ট কার্নেল — টুল কল সুরক্ষা",
			Language:    "bn",
			Category:    "localized",
			Description: "Bengali search phrase for an AI agent kernel that vets every tool call behind a default-deny capability floor; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/bn/README.md",
			Keywords:    []string{"AI এজেন্ট", "tool call সুরক্ষা", "fak agent kernel"},
		},
		{
			Name:        "টোকেন খরচ কমানো DPDP self-host এজেন্ট গেটওয়ে",
			Language:    "bn",
			Category:    "localized",
			Description: "Bengali search phrase for the cost and DPDP data-residency side of fak: cheaper long sessions via prompt-cache-preserving serving on one self-hosted Apache-2.0 binary; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/bn/README.md",
			Keywords:    []string{"token খরচ", "DPDP self-host", "prompt cache"},
		},
		{
			Name:        "एजंट कर्नल — प्रत्येक tool call तपासणारी सुरक्षा",
			Language:    "mr",
			Category:    "localized",
			Description: "Marathi search phrase for an AI agent kernel that checks every tool call before it runs; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/mr/README.md",
			Keywords:    []string{"AI एजंट", "टूल कॉल सुरक्षा", "fak agent kernel"},
		},
		{
			Name:        "टोकन खर्च कमी करणारा DPDP self-host एजंट गेटवे",
			Language:    "mr",
			Category:    "localized",
			Description: "Marathi search phrase for the cost and DPDP data-residency side of fak: cheaper long sessions via cache-preserving self-hosted agent serving; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/mr/README.md",
			Keywords:    []string{"टोकन बचत", "prompt cache", "DPDP self-host"},
		},
		{
			Name:        "AI ஏஜென்ட் கெர்னல் — tool call பாதுகாப்பு",
			Language:    "ta",
			Category:    "localized",
			Description: "Tamil search phrase for an AI agent kernel that checks every tool call with a default-deny capability floor; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/ta/README.md",
			Keywords:    []string{"AI ஏஜென்ட் கெர்னல்", "tool call பாதுகாப்பு", "fak agent kernel"},
		},
		{
			Name:        "token செலவு குறைக்கும் DPDP self-host ஏஜென்ட் கேட்வே",
			Language:    "ta",
			Category:    "localized",
			Description: "Tamil search phrase for the token-cost and DPDP data-residency side of fak: cheaper long sessions via cache-preserving self-hosted serving (~4.1x less work); routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/ta/README.md",
			Keywords:    []string{"token செலவு", "DPDP self-host residency", "prompt cache"},
		},
		{
			Name:        "AI ఏజెంట్ కెర్నల్ – టూల్ కాల్ భద్రత",
			Language:    "te",
			Category:    "localized",
			Description: "Telugu search phrase for an AI agent kernel that vets every tool call with a default-deny capability floor; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/te/README.md",
			Keywords:    []string{"AI ఏజెంట్ కెర్నల్", "టూల్ కాల్ భద్రత", "fak agent kernel"},
		},
		{
			Name:        "టోకెన్ ఖర్చు తగ్గించే self-host ఏజెంట్ గేట్‌వే",
			Language:    "te",
			Category:    "localized",
			Description: "Telugu search phrase for the token-cost and DPDP self-host residency side of fak, with cheaper long sessions through cache-preserving serving; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/te/README.md",
			Keywords:    []string{"టోకెన్ ఖర్చు", "prompt cache", "DPDP self-host"},
		},
		{
			Name:        "kernel de seguridad para agentes de IA que valida cada tool call",
			Language:    "es",
			Category:    "localized",
			Description: "Spanish search phrase for an AI-agent kernel that governs every tool call at a default-deny capability floor; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/es/README.md",
			Keywords:    []string{"kernel de agentes IA", "seguridad de tool call", "capability floor default-deny"},
		},
		{
			Name:        "self-host con prompt cache para sesiones largas más baratas y residencia de datos RGPD",
			Language:    "es",
			Category:    "localized",
			Description: "Spanish search phrase for the cost and data-residency lever of self-hosting an agent kernel; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/es/README.md",
			Keywords:    []string{"self-host prompt cache", "sesiones largas más baratas", "residencia de datos RGPD"},
		},
		{
			Name:        "AI エージェントの tool call を実行前に審査するカーネル",
			Language:    "ja",
			Category:    "localized",
			Description: "Japanese search phrase for an AI-agent kernel that vets every tool call before execution with a default-deny capability floor; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/ja/README.md",
			Keywords:    []string{"AI エージェント セキュリティ", "tool call 審査", "default-deny capability floor"},
		},
		{
			Name:        "エージェントの長いセッションを安くする self-host データレジデンシー",
			Language:    "ja",
			Category:    "localized",
			Description: "Japanese search phrase for cutting long agent-session cost via cache-preserving self-hosted serving that keeps data on your own infrastructure; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/ja/README.md",
			Keywords:    []string{"prompt cache コスト削減", "self-host データレジデンシー", "個人情報保護法 APPI"},
		},
		{
			Name:        "AI 에이전트 커널 — 모든 tool call을 실행 전에 검토",
			Language:    "ko",
			Category:    "localized",
			Description: "Korean search phrase for an AI agent kernel that vets every tool call before it runs behind a default-deny capability floor; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/ko/README.md",
			Keywords:    []string{"AI 에이전트 커널", "tool call 보안", "fak agent kernel"},
		},
		{
			Name:        "긴 세션을 저렴하게 만드는 PIPA 자체 호스팅 에이전트 게이트웨이",
			Language:    "ko",
			Category:    "localized",
			Description: "Korean search phrase for the cost and PIPA data-residency side of fak: cheaper long sessions via prompt-cache-preserving self-hosted serving; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/ko/README.md",
			Keywords:    []string{"자체 호스팅", "prompt cache", "PIPA"},
		},
		{
			Name:        "núcleo de agente de IA que revisa cada tool call antes de executá-lo",
			Language:    "pt",
			Category:    "localized",
			Description: "Portuguese search phrase for an AI agent kernel that vets every tool call before it runs under a default-deny capability floor; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/pt/README.md",
			Keywords:    []string{"núcleo de agente IA", "segurança de tool call", "capability floor default-deny"},
		},
		{
			Name:        "self-host com prompt cache para sessões longas mais baratas e residência de dados LGPD",
			Language:    "pt",
			Category:    "localized",
			Description: "Portuguese search phrase for the cost and LGPD/RGPD data-residency lever of self-hosting an agent kernel; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/pt/README.md",
			Keywords:    []string{"self-host prompt cache", "sessões longas mais baratas", "residência de dados LGPD"},
		},
		{
			Name:        "ядро AI-агента, проверяющее каждый tool call до выполнения",
			Language:    "ru",
			Category:    "localized",
			Description: "Russian search phrase for an AI agent kernel that vets every tool call before it runs behind a fail-closed, default-deny capability floor; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/ru/README.md",
			Keywords:    []string{"ядро AI-агента", "безопасность tool call", "capability floor default-deny"},
		},
		{
			Name:        "дешёвый self-host с prompt cache для долгих сессий и хранение данных по 152-ФЗ",
			Language:    "ru",
			Category:    "localized",
			Description: "Russian search phrase for the cost and 152-FZ data-residency side of fak: cheaper long sessions via cache-preserving self-hosted serving; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/ru/README.md",
			Keywords:    []string{"self-host prompt cache", "дешевле долгие сессии", "152-ФЗ"},
		},
		{
			Name:        "نواة وكيل الذكاء الاصطناعي تراجع كل tool call قبل تنفيذه",
			Language:    "ar",
			Category:    "localized",
			Description: "Arabic search phrase for an AI agent kernel that vets every tool call before it runs under a default-deny capability floor; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/ar/README.md",
			Keywords:    []string{"نواة وكيل الذكاء الاصطناعي", "أمان tool call", "capability floor"},
		},
		{
			Name:        "استضافة ذاتية مع prompt cache لجلسات أطول أرخص وإقامة بيانات متوافقة مع PDPL",
			Language:    "ar",
			Category:    "localized",
			Description: "Arabic search phrase for the cost and PDPL data-residency lever of self-hosting an agent kernel; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/ar/README.md",
			Keywords:    []string{"استضافة ذاتية", "prompt cache", "PDPL"},
		},
		{
			Name:        "kernel agen AI yang memeriksa setiap tool call sebelum dijalankan",
			Language:    "id",
			Category:    "localized",
			Description: "Indonesian search phrase for an AI agent kernel that vets every tool call before it runs under a default-deny capability floor; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/id/README.md",
			Keywords:    []string{"kernel agen AI", "keamanan tool call", "capability floor default-deny"},
		},
		{
			Name:        "self-host dengan prompt cache untuk sesi panjang lebih murah dan residensi data UU PDP",
			Language:    "id",
			Category:    "localized",
			Description: "Indonesian search phrase for the cost and UU PDP data-residency lever of self-hosting an agent kernel; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/id/README.md",
			Keywords:    []string{"self-host prompt cache", "sesi panjang lebih murah", "UU PDP"},
		},
		{
			Name:        "nhân agent AI xét duyệt mọi tool call trước khi chạy",
			Language:    "vi",
			Category:    "localized",
			Description: "Vietnamese search phrase for an AI agent kernel that vets every tool call before it runs under a default-deny capability floor; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/vi/README.md",
			Keywords:    []string{"nhân agent AI", "bảo mật tool call", "capability floor default-deny"},
		},
		{
			Name:        "tự lưu trữ với prompt cache cho phiên dài rẻ hơn và lưu trú dữ liệu theo Nghị định 13/2023 (PDPD)",
			Language:    "vi",
			Category:    "localized",
			Description: "Vietnamese search phrase for the cost and PDPD (Decree 13/2023) data-residency lever of self-hosting an agent kernel; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/vi/README.md",
			Keywords:    []string{"tự lưu trữ prompt cache", "phiên dài rẻ hơn", "PDPD Nghị định 13"},
		},
		{
			Name:        "her tool call'u çalışmadan önce denetleyen AI ajan çekirdeği",
			Language:    "tr",
			Category:    "localized",
			Description: "Turkish search phrase for an AI agent kernel that vets every tool call before it runs under a default-deny capability floor; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/tr/README.md",
			Keywords:    []string{"AI ajan çekirdeği", "tool call güvenliği", "capability floor default-deny"},
		},
		{
			Name:        "uzun oturumları ucuzlatan prompt cache'li KVKK uyumlu self-host ajan ağ geçidi",
			Language:    "tr",
			Category:    "localized",
			Description: "Turkish search phrase for the cost and KVKK data-residency lever of self-hosting an agent kernel; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/tr/README.md",
			Keywords:    []string{"self-host prompt cache", "uzun oturum ucuz", "KVKK"},
		},
	}
}

// ConfigFAQFeed renders configuration answers as schema.org FAQPage JSON-LD.
func ConfigFAQFeed(when time.Time) ([]byte, error) {
	answers := AEOConfigAnswers()
	questions := make([]configQuestion, 0, len(answers))
	for _, a := range answers {
		questions = append(questions, configQuestion{
			Type:           "Question",
			Name:           a.Question,
			Keywords:       append([]string(nil), a.Keywords...),
			AcceptedAnswer: configAnswer{Type: "Answer", Text: a.Answer, Citation: a.Authority},
		})
	}
	feed := configFAQFeed{
		Context:      "https://schema.org",
		Type:         "FAQPage",
		Name:         "fak configuration answers",
		Description:  "Direct answers and authoritative documentation links for configuring the Fused Agent Kernel (fak).",
		URL:          "https://github.com/anthony-chaudhary/fak/blob/main/docs/fak/server-config.md",
		DateModified: when.UTC().Format("2006-01-02"),
		MainEntity:   questions,
	}
	b, err := json.MarshalIndent(feed, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

// ConfigAnswersMarkdown renders a crawlable page where the visible answers and
// inline FAQPage JSON-LD come from the same canonical roster. Keeping structured
// data on the page it describes lets search engines verify it against human text.
func ConfigAnswersMarkdown(when time.Time) (string, error) {
	jsonLD, err := ConfigFAQFeed(when)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString("---\n")
	b.WriteString("title: \"How to Configure fak: Flags, TOML, Environment Variables, and Precedence\"\n")
	b.WriteString("description: \"Direct answers for configuring fak, including config-file requirements, precedence, validation, secrets, and client setup.\"\n")
	b.WriteString("permalink: /configuration/\n")
	b.WriteString("---\n\n")
	b.WriteString("# How to configure fak\n\n")
	b.WriteString("These concise answers cover the decisions people most often make when configuring fak. ")
	b.WriteString("For every `fak serve` flag, environment variable, and manifest field, use the ")
	b.WriteString("[complete server configuration reference](server-config.md).\n\n")
	b.WriteString("<script type=\"application/ld+json\">\n")
	b.Write(jsonLD)
	b.WriteString("</script>\n\n")
	for _, a := range AEOConfigAnswers() {
		fmt.Fprintf(&b, "## %s\n\n%s\n\n[Authoritative details](%s)\n\n", a.Question, a.Answer, a.Authority)
	}
	b.WriteString("## Complete configuration reference\n\n")
	b.WriteString("Continue to [fak Server Configuration Reference](server-config.md) for the exhaustive option tables and deployment details.\n")
	return b.String(), nil
}

// ConfigAnswersText renders the human- and agent-readable sibling of ConfigFAQFeed.
func ConfigAnswersText(when time.Time) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# fak configuration answers\n\nUpdated: %s\nAuthority: https://github.com/anthony-chaudhary/fak/blob/main/docs/fak/server-config.md\n\n", when.UTC().Format("2006-01-02"))
	for i, a := range AEOConfigAnswers() {
		fmt.Fprintf(&b, "## %s\n%s\nSource: %s\n", a.Question, a.Answer, a.Authority)
		if len(a.Keywords) > 0 {
			fmt.Fprintf(&b, "Search terms: %s\n", strings.Join(a.Keywords, ", "))
		}
		if i < len(AEOConfigAnswers())-1 {
			b.WriteString("\n")
		}
	}
	return b.String()
}

// DisambiguationTermsFeed renders the term roster as a schema.org DefinedTermSet
// JSON-LD artifact. The site structured-data generator also consumes this file
// to keep SoftwareApplication keywords aligned with the machine term feed.
func DisambiguationTermsFeed(when time.Time) ([]byte, error) {
	feed := termFeed{
		Context:     "https://schema.org",
		Type:        "DefinedTermSet",
		Name:        "fak — answer-engine disambiguation terms",
		Description: "Search and answer-engine terms for correctly identifying fak, including core concepts, performance-lens leads (automatic context management, cache-preserving compaction, cross-agent KV reuse), localized entry points, frontier-model-launch routing/cache/fallback demand hooks, and agent-security hooks (MCP tool poisoning, lethal trifecta, least-privilege tool access, tamper-evident audit).",
	}
	if !when.IsZero() {
		feed.Modified = when.UTC().Format(time.RFC3339)
	}
	for _, t := range AEODisambiguationTerms() {
		feed.Terms = append(feed.Terms, definedTerm{
			Type:        "DefinedTerm",
			Name:        t.Name,
			TermCode:    t.Category,
			InLanguage:  t.Language,
			Description: t.Description,
			URL:         t.URL,
			Keywords:    strings.Join(t.Keywords, ", "),
		})
	}
	return json.MarshalIndent(feed, "", "  ")
}

// UpdatesFeed renders the witnessed ships as a schema.org ItemList (JSON, 2-space indent),
// newest-first and capped. when stamps the feed's dateModified. The result is the
// docs/marketing/updates.json the Python injector reads — and a valid JSON-LD document an
// answer engine can ingest directly.
func UpdatesFeed(ships []Ship, when time.Time) ([]byte, error) {
	ships = sortedShipsDesc(ships)
	if len(ships) > defaultFeedCap {
		ships = ships[:defaultFeedCap]
	}
	feed := updatesFeed{
		Context: "https://schema.org",
		Type:    "ItemList",
		Name:    "fak — what shipped",
	}
	if !when.IsZero() {
		feed.Modified = when.UTC().Format(time.RFC3339)
	}
	for i, s := range ships {
		feed.Items = append(feed.Items, updatesItem{
			Type:     "ListItem",
			Position: i + 1,
			Item: updatesSrcCode{
				Type:           "SoftwareSourceCode",
				Name:           claimText(s),
				CodeRepository: repoCommitURL + s.SHA,
				DateModified:   shipDate(s),
				Keywords:       s.Leaf,
			},
		})
	}
	return json.MarshalIndent(feed, "", "  ")
}

// WhatsNewMarkdown renders the bounded, dated, witnessed "Recent ships" block that the Python
// injector fences into llms.txt (and llms-updates.txt). House style: one `- **date** — claim
// ([sha](commit-url))` line per ship, newest-first, capped. Pure and stable, so re-rendering
// the same ships is a no-op diff (the idempotence the marker injection relies on).
func WhatsNewMarkdown(ships []Ship) string {
	ships = sortedShipsDesc(ships)
	if len(ships) > defaultFeedCap {
		ships = ships[:defaultFeedCap]
	}
	if len(ships) == 0 {
		return "_No witnessed ships recorded yet._"
	}
	var b strings.Builder
	for _, s := range ships {
		date := shipDate(s)
		if date == "" {
			date = "recent"
		}
		fmt.Fprintf(&b, "- **%s** — %s ([`%s`](%s%s))\n", date, claimText(s), s.SHA, repoCommitURL, s.SHA)
	}
	return strings.TrimRight(b.String(), "\n")
}

// LlmsUpdatesText renders the sibling llms-updates.txt corpus — a plain, capped, newest-first
// recency feed an answer engine or agent polls, in the same house style as llms.txt. It leads
// with a one-line self-describing header so a crawler that lands on it alone understands it.
func LlmsUpdatesText(ships []Ship, when time.Time) string {
	var b strings.Builder
	b.WriteString("# fak — what shipped (recent, witnessed)\n\n")
	b.WriteString("> A machine-readable feed of fak's most recent shipped changes. Every line cites the\n")
	b.WriteString("> commit that witnesses it. Regenerated on each completion by `fak marketing aeo`.\n")
	if !when.IsZero() {
		fmt.Fprintf(&b, ">\n> Updated: %s\n", when.UTC().Format(time.RFC3339))
	}
	b.WriteString("\n")
	b.WriteString(WhatsNewMarkdown(ships))
	b.WriteString("\n")
	return b.String()
}

// LlmsTermsText renders the plain sibling corpus for answer-engine term discovery.
// It is generated from the same roster as the JSON-LD DefinedTermSet so humans,
// search crawlers, and LLM answer engines see the same vocabulary.
func LlmsTermsText(when time.Time) string {
	var b strings.Builder
	b.WriteString("# fak — answer-engine disambiguation terms\n\n")
	b.WriteString("> A machine-readable term feed for answer engines and agents. It names the phrases\n")
	b.WriteString("> that should route to fak's docs, including localized terms, frontier-model launch\n")
	b.WriteString("> hooks (routing, fallback, prompt-cache cost) and agent-security demand hooks such\n")
	b.WriteString("> as MCP tool poisoning, the lethal trifecta, and least-privilege tool access.\n")
	if !when.IsZero() {
		fmt.Fprintf(&b, ">\n> Updated: %s\n", when.UTC().Format(time.RFC3339))
	}
	last := ""
	for _, t := range AEODisambiguationTerms() {
		if t.Category != last {
			fmt.Fprintf(&b, "\n## %s\n\n", t.Category)
			last = t.Category
		}
		keywords := strings.Join(t.Keywords, ", ")
		if keywords != "" {
			keywords = " Keywords: " + keywords + "."
		}
		fmt.Fprintf(&b, "- `%s` (%s) — %s%s\n", t.Name, t.Language, t.Description, keywords)
	}
	return strings.TrimRight(b.String(), "\n") + "\n"
}

// shipDate renders a ship's date as YYYY-MM-DD, or "" if unparsed.
func shipDate(s Ship) string {
	if s.Date.IsZero() {
		return ""
	}
	return s.Date.UTC().Format("2006-01-02")
}

// sortedShipsDesc returns a newest-first copy (does not mutate the input).
func sortedShipsDesc(ships []Ship) []Ship {
	out := append([]Ship(nil), ships...)
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j].Date.After(out[j-1].Date); j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}
