package gateway

// coretool_admit.go — the MCP-catalog-over-new-core-tool admission gate (#2926).
//
// Rung 5 of the Footprint Ladder (AGENTS.md:198) says: prefer adding a capability as
// an MCP server in the catalog over growing the core toolset, because a new core tool's
// schema ships on EVERY API call — a per-turn tax paid forever. Rung 6 (a new entry in
// MCPFloorToolDefs) is the last resort. Upstream (the Hermes reviewer discipline) leaves
// that preference to human judgment; the reviewer who forgets it lets the floor grow
// past what MCP already covers.
//
// This file makes the preference a kernel check. AdmitCoreTool maps a PROPOSED core
// tool onto fak's already-registered MCP catalog (toolDescriptors) and REFUSES the
// proposal when an existing MCP capability already reaches it, naming the specific
// sidestep. It is the rung-5-over-rung-6 companion to mcpfootprint.CheckFloor (#2924,
// the rung-6 token BUDGET gate): CheckFloor refuses a tool that grows the floor past a
// ceiling; this refuses a tool that need not join the floor at all because MCP covers
// it. The two are separable — a tool can be cheap yet redundant, or novel yet fat. It
// is also distinct from the service-gated admission of sibling #2925 (admit only when
// the backing service is configured); this gate is a flat "the catalog already reaches
// this, do not add it".
//
// The matching rule is deliberately PRECISE over fuzzy (issue #2926's stated concern: a
// fuzzy name match yields false-positive refusals that block genuinely novel tools). It
// is structural: a proposal is refused only when one side's significant capability
// tokens are WHOLLY carried by the other's — either the catalog tool is at least as
// general (proposal ⊆ catalog), or the proposal spells out a catalog capability in full
// and adds words (catalog ⊆ proposal, the shape a free-text Capability phrase produces).
// Containment either way means one name fully spans the other; a mere partial overlap
// does not, and still admits. A proposal that
// introduces a distinctive token no catalog tool carries is admitted — the gate never
// blocks a novel capability, it only refuses one the catalog already reaches. A
// proper-subset match must further rest on at least one DISTINCTIVE (non-generic) token,
// so a bare shared verb like "search"/"read" never triggers a refusal on its own.

import (
	"fmt"
	"sort"
	"strings"
)

// ReasonMCPCatalogCovers is the closed-vocabulary refusal token this gate names, in the
// same spirit as mcpfootprint's FLOOR_BUDGET_* reasons and the dos.toml [reasons.*]
// blocks a guard refusal cites.
const ReasonMCPCatalogCovers = "MCP_CATALOG_COVERS"

// ProposedCoreTool is a candidate for rung 6 — a new always-sent entry in
// MCPFloorToolDefs(). Name is the tool name it would register under; Capability is an
// optional free phrase of what it would do, whose significant terms widen the match so a
// proposer can state intent even when the name is opaque.
type ProposedCoreTool struct {
	Name            string
	Capability      string
	DestructiveHint bool
	OpenWorldHint   bool
}

// CoreToolAdmissionError is a structured refusal: the proposed tool, the specific MCP
// catalog tool(s) that already reach its capability (the named sidestep), and the shared
// capability tokens the decision rests on. Callers branch on Reason rather than
// string-matching the message.
type CoreToolAdmissionError struct {
	Reason    string   // ReasonMCPCatalogCovers
	Proposed  string   // the proposed tool name
	Sidesteps []string // existing MCP catalog tools that already reach the capability
	Shared    []string // the significant capability tokens the refusal rests on
}

func (e *CoreToolAdmissionError) Error() string {
	return fmt.Sprintf(
		"%s: the proposed core tool %q would ship on EVERY API call, but its capability (%s) is already reachable through the MCP catalog via %s. "+
			"Rung 5 of the Footprint Ladder prefers the existing MCP surface over growing the always-sent core floor (rung 6) — route through the catalog tool "+
			"instead of adding a new core-schema entry, or, if the capability is genuinely distinct, name the token that makes it distinct.",
		e.Reason, e.Proposed, strings.Join(e.Shared, ", "), strings.Join(e.Sidesteps, ", "))
}

// AdmitCoreTool refuses a proposed rung-6 core tool whose capability is already
// reachable through the MCP catalog (rung 5). It returns nil when the proposal is
// genuinely novel and a *CoreToolAdmissionError naming the sidestep otherwise. The
// catalog is fak's own registered MCP tools (toolDescriptors) — the same descriptors
// MCPFloorToolDefs() prices as the always-sent floor.
func AdmitCoreTool(p ProposedCoreTool) error {
	return admitCoreToolAgainst(p, catalogCapabilities())
}

// catalogCapability is one MCP catalog entry reduced to its capability signature: the
// tool name and the significant tokens of that name.
type catalogCapability struct {
	name   string
	tokens map[string]bool
}

// catalogCapabilities reduces the live MCP tool descriptors to their capability
// signatures. Tokens come from the NAME only (curated and precise) — never the long
// description, whose prose would inflate the set and manufacture false matches.
func catalogCapabilities() []catalogCapability {
	descs := toolDescriptors()
	caps := make([]catalogCapability, 0, len(descs))
	for _, td := range descs {
		name, _ := td["name"].(string)
		if strings.TrimSpace(name) == "" {
			continue
		}
		caps = append(caps, catalogCapability{name: name, tokens: significantTokens(name)})
	}
	return caps
}

// admitCoreToolAgainst is AdmitCoreTool with the catalog injected, so a test can drive
// the decision against a synthetic catalog without depending on the live registry.
func admitCoreToolAgainst(p ProposedCoreTool, catalog []catalogCapability) error {
	proposed := significantTokens(p.Name)
	for t := range significantTokens(p.Capability) {
		proposed[t] = true
	}
	if len(proposed) == 0 {
		return nil // no capability signature to match — cannot claim the catalog covers it
	}

	var sidesteps []string
	shared := map[string]bool{}
	for _, c := range catalog {
		if !catalogCovers(proposed, c.tokens) {
			continue
		}
		sidesteps = append(sidesteps, c.name)
		for t := range sharedTokens(proposed, c.tokens) {
			shared[t] = true
		}
	}
	if len(sidesteps) == 0 {
		return nil
	}
	sort.Strings(sidesteps)
	return &CoreToolAdmissionError{
		Reason:    ReasonMCPCatalogCovers,
		Proposed:  p.Name,
		Sidesteps: sidesteps,
		Shared:    tokenSlice(shared),
	}
}

// catalogCovers decides whether a catalog tool's capability tokens already reach the
// proposed tool's. It fires when the two name the SAME capability (equal token sets) or
// when one side's tokens are wholly contained in the other's — in EITHER direction:
//
//   - proposed ⊊ catalog — the catalog tool is strictly more general, so it already
//     reaches the narrower thing being proposed ("fak_index" under "fak_index_lane").
//   - catalog ⊊ proposed — the proposal spells out a capability the catalog names
//     exactly, plus extra words. This is the direction a free-text Capability phrase
//     produces, and it is the whole reason that field exists: "fak_unlease" is opaque on
//     its own, but the phrase "revoke a witness" contributes the token that shows
//     fak_revoke already covers it. Checking only the first direction made the phrase
//     actively counterproductive — every word it added pushed the proposal further from
//     being a subset, so stating intent more fully made a redundant tool LOOK novel.
//
// Either way the match must rest on at least one DISTINCTIVE (non-generic) token from
// the SHARED (smaller) set, so a bare shared verb like "search"/"read" never triggers a
// refusal on its own — that is the precision the issue demands over a fuzzy name overlap.
// The asymmetry is intentional: containment in either direction means one name fully
// spans the other, which is a structural claim; a mere partial OVERLAP still admits,
// because a token neither side accounts for is exactly what a novel capability looks like.
func catalogCovers(proposed, catalog map[string]bool) bool {
	if subsetOf(proposed, catalog) {
		if len(proposed) == len(catalog) {
			return true // equal sets: the same capability by name
		}
		return hasDistinctive(proposed) // shared set is the proposal's
	}
	if subsetOf(catalog, proposed) {
		return hasDistinctive(catalog) // shared set is the catalog tool's
	}
	return false
}

// hasDistinctive reports whether a token set carries at least one non-generic token —
// something naming a DOMAIN and not just an action.
func hasDistinctive(tokens map[string]bool) bool {
	for t := range tokens {
		if !genericToken[t] {
			return true
		}
	}
	return false
}

// sharedTokens is the intersection of a proposal's capability signature with a matched
// catalog tool's. The refusal reports the intersection rather than the whole proposal
// because the two are no longer the same set once a match can run catalog ⊊ proposed: a
// message listing every word of the proposer's phrase as "shared" would overstate what
// the decision actually rests on.
func sharedTokens(proposed, catalog map[string]bool) map[string]bool {
	out := map[string]bool{}
	for t := range proposed {
		if catalog[t] {
			out[t] = true
		}
	}
	return out
}

// subsetOf reports whether every token in a is present in b.
func subsetOf(a, b map[string]bool) bool {
	for t := range a {
		if !b[t] {
			return false
		}
	}
	return true
}

// genericToken is the small set of connective VERBS that name an action but not a
// domain. A proper-subset match resting ONLY on these is too weak to trust as "the
// catalog already covers this", so it does not by itself trigger a refusal.
var genericToken = map[string]bool{
	"get": true, "list": true, "read": true, "run": true, "show": true,
	"search": true, "fetch": true, "query": true, "view": true, "call": true,
	"use": true, "make": true, "check": true,
}

// stopword is dropped entirely from a capability signature: vendor prefixes and
// connectives carry no capability meaning.
var stopword = map[string]bool{
	"fak": true, "mcp": true,
	"a": true, "an": true, "the": true, "to": true, "of": true, "for": true,
	"and": true, "or": true, "by": true, "on": true, "in": true, "with": true,
	"is": true, "it": true, "this": true, "that": true, "its": true,
}

// significantTokens reduces a tool name or capability phrase to its set of meaningful
// lowercased word tokens: split on any non-alphanumeric boundary, drop vendor prefixes
// and connective stopwords and single-character noise. Both "fak_index_lane" and
// "resolve a path to its lane" reduce to their capability nouns/verbs this way.
func significantTokens(s string) map[string]bool {
	out := map[string]bool{}
	for _, tok := range strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9')
	}) {
		if stopword[tok] || len(tok) <= 1 {
			continue
		}
		out[tok] = true
	}
	return out
}

// tokenSlice is the sorted slice form of a token set, for stable error messages.
func tokenSlice(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for t := range m {
		out = append(out, t)
	}
	sort.Strings(out)
	return out
}

// CoreToolsPalette defines the core tools palette annotated with granular capability
// hints (DestructiveHint for mutating operations and OpenWorldHint for network or
// arbitrary shell execution tools).
var CoreToolsPalette = []MCPToolDescriptor{
	{
		Name:            "read",
		Description:     "Read file contents without modifying the workspace",
		DestructiveHint: false,
		OpenWorldHint:   false,
	},
	{
		Name:            "write",
		Description:     "Write or overwrite file contents in the workspace",
		DestructiveHint: true,
		OpenWorldHint:   false,
	},
	{
		Name:            "edit",
		Description:     "Apply exact string replacements in existing files",
		DestructiveHint: true,
		OpenWorldHint:   false,
	},
	{
		Name:            "bash",
		Description:     "Execute arbitrary shell commands in persistent session",
		DestructiveHint: true,
		OpenWorldHint:   true,
	},
	{
		Name:            "exec",
		Description:     "Execute arbitrary command in the host environment",
		DestructiveHint: true,
		OpenWorldHint:   true,
	},
	{
		Name:            "web_fetch",
		Description:     "Fetch content from a remote URL over the network",
		DestructiveHint: false,
		OpenWorldHint:   true,
	},
	{
		Name:            "fetch",
		Description:     "Fetch network resource",
		DestructiveHint: false,
		OpenWorldHint:   true,
	},
	{
		Name:            "fak_syscall",
		Description:     "Adjudicate and execute tool call through fak kernel",
		DestructiveHint: true,
		OpenWorldHint:   true,
	},
	{
		Name:            "fak_admit",
		Description:     "Admit out-of-band tool results into context",
		DestructiveHint: true,
		OpenWorldHint:   false,
	},
	{
		Name:            "fak_revoke",
		Description:     "Evict and refute world-state witness",
		DestructiveHint: true,
		OpenWorldHint:   false,
	},
	{
		Name:            "fak_context_change",
		Description:     "Mutate persisted recall core image context",
		DestructiveHint: true,
		OpenWorldHint:   false,
	},
	{
		Name:            "fak_session_reset",
		Description:     "Reset and rearm session carryover window",
		DestructiveHint: true,
		OpenWorldHint:   false,
	},
	{
		Name:            "fak_memory_run",
		Description:     "Run memory query with applied mutations",
		DestructiveHint: true,
		OpenWorldHint:   false,
	},
	{
		Name:            "fak_read",
		Description:     "Read files with verified-fresh cache reuse",
		DestructiveHint: false,
		OpenWorldHint:   false,
	},
	{
		Name:            "fak_adjudicate",
		Description:     "Adjudicate proposed tool call without executing",
		DestructiveHint: false,
		OpenWorldHint:   false,
	},
	{
		Name:            "fak_tools_search",
		Description:     "Search and retrieve tool schemas",
		DestructiveHint: false,
		OpenWorldHint:   false,
	},
}

// LookupCoreTool finds a core tool in CoreToolsPalette by name.
func LookupCoreTool(name string) (MCPToolDescriptor, bool) {
	for _, tool := range CoreToolsPalette {
		if tool.Name == name {
			return tool, true
		}
	}
	return MCPToolDescriptor{}, false
}

// CoreToolsPaletteProposed returns the core tools palette mapped to []ProposedCoreTool.
func CoreToolsPaletteProposed() []ProposedCoreTool {
	out := make([]ProposedCoreTool, 0, len(CoreToolsPalette))
	for _, tool := range CoreToolsPalette {
		out = append(out, ProposedCoreTool{
			Name:            tool.Name,
			Capability:      tool.Description,
			DestructiveHint: tool.DestructiveHint,
			OpenWorldHint:   tool.OpenWorldHint,
		})
	}
	return out
}
