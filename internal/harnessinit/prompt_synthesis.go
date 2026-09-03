package harnessinit

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/promptcomp"
)

// AgentScale classifies the operational tier and context footprint of an agent.
type AgentScale uint8

const (
	// ScaleCoordinator is a large orchestrating agent responsible for task decomposition,
	// worker fan-out, collision pricing, and receipt aggregation. Target: 2,500–4,000 tokens.
	ScaleCoordinator AgentScale = iota
	// ScaleLeafWorker is an S0/S1 bounded worker focused on a single observable deliverable
	// and exactly one witness command. Target: <800 tokens.
	ScaleLeafWorker
	// ScaleValidator is a micro-filter / evaluator producing strict structured verdicts. Target: <200 tokens.
	ScaleValidator
)

func (s AgentScale) String() string {
	switch s {
	case ScaleCoordinator:
		return "coordinator"
	case ScaleLeafWorker:
		return "leaf"
	case ScaleValidator:
		return "validator"
	default:
		return "unknown"
	}
}

// PromptSpec configures dynamic system prompt synthesis.
type PromptSpec struct {
	Scale         AgentScale
	ModelFamily   string                  // e.g. "qwen3.8", "qwen2.5-coder-7b", "claude-3-7-sonnet"
	IsSmallLocal  bool                    // True for 7B-14B models requiring concise contracts
	ContextBudget int                     // Total available context budget tokens
	WireFormat    string                  // "openai", "gguf", "anthropic"
	ExtraParts    []promptcomp.PromptPart // Additional caller-supplied fragments
}

// Canonical fragments provided by harnessinit.
const (
	PartIDSpineCore         = "spine.fak-core"
	PartIDSafetyFloor       = "safety.floor"
	PartIDSafetyFull        = "safety.full"
	PartIDContractCoord     = "contract.coordinator"
	PartIDContractLeaf      = "contract.leaf-concise"
	PartIDContractValidator = "contract.validator-rubric"
	PartIDToolsFull         = "tools.full-catalog"
	PartIDToolsMinimal      = "tools.minimal-fileops"
	PartIDOverlayDelegation = "overlay.delegation"
	PartIDOverlayGuidance   = "overlay.guidance"
)

// DefaultParts returns the canonical set of prompt fragments conditioned by AgentScale and budget.
func DefaultParts(scale AgentScale) []promptcomp.PromptPart {
	spine := promptcomp.PromptPart{
		ID:      PartIDSpineCore,
		Content: "You are an agent operating inside the fak kernel runtime. You interact with repository state and tools under explicit capability admission. Self-reports are not facts; ground all claims in verifiable tool receipts and git witnesses.",
		Kind:    promptcomp.KindSpine,
		Rank:    0,
	}

	safetyFloor := promptcomp.PromptPart{
		ID: PartIDSafetyFloor,
		Content: "CAPABILITY FLOOR & SAFETY RULES:\n" +
			"- Default-deny capability discipline: all mutating operations require tool-level policy clearance.\n" +
			"- Concurrency safety: acquire disjoint lane leases before modifying shared workspace paths.\n" +
			"- Output safety: never log, echo, or emit credentials, private tokens, or secret keys.",
		Kind:      promptcomp.KindSafety,
		Rank:      10,
		DependsOn: []string{PartIDSpineCore},
		Predicate: func(e promptcomp.Env) bool {
			return e.AgentTier != "coordinator"
		},
	}

	safetyFull := promptcomp.PromptPart{
		ID: PartIDSafetyFull,
		Content: "CAPABILITY FLOOR & ORCHESTRATION SAFETY RULES:\n" +
			"- Default-deny capability discipline: all mutating operations require tool-level policy clearance.\n" +
			"- Concurrency safety: acquire disjoint lane leases before modifying shared workspace paths.\n" +
			"- Output safety: never log, echo, or emit credentials, private tokens, or secret keys.\n" +
			"- Reversible operations: dry-run or preview multi-file sweeps before commit.\n" +
			"- Non-interference: preserve peer WIP on shared trunk; never force-push or branch off trunk.",
		Kind:          promptcomp.KindSafety,
		Rank:          10,
		DependsOn:     []string{PartIDSpineCore},
		ConflictsWith: []string{PartIDSafetyFloor},
		Predicate: func(e promptcomp.Env) bool {
			return e.AgentTier == "coordinator"
		},
	}

	contractCoord := promptcomp.PromptPart{
		ID: PartIDContractCoord,
		Content: "ROLE: COORDINATOR AGENT\n" +
			"- Decompose incoming goals into bounded, tree-disjoint S0/S1 subagent packets.\n" +
			"- Supervise worker lifecycle, price lane collisions via dos arbitrate, and aggregate completion receipts.\n" +
			"- Keep coordinator context clean: delegate investigation, implementation, and long command executions to workers.\n" +
			"- Verify all child outputs independently before landing or reporting done.",
		Kind:      promptcomp.KindContract,
		Rank:      20,
		DependsOn: []string{PartIDSafetyFull},
		Predicate: func(e promptcomp.Env) bool {
			return e.AgentTier == "coordinator"
		},
	}

	contractLeaf := promptcomp.PromptPart{
		ID: PartIDContractLeaf,
		Content: "ROLE: S0/S1 LEAF WORKER\n" +
			"- Restrict execution to a single observable deliverable and exactly one witness command.\n" +
			"- Limit active write surface to 1-3 closely related files.\n" +
			"- Direct imperative action: Read target files before Edit. Verify via deterministic test.\n" +
			"- Emit zero conversational preambles, apologies, or speculative commentary.",
		Kind:      promptcomp.KindContract,
		Rank:      20,
		DependsOn: []string{PartIDSafetyFloor},
		Predicate: func(e promptcomp.Env) bool {
			return e.AgentTier == "leaf"
		},
	}

	contractValidator := promptcomp.PromptPart{
		ID: PartIDContractValidator,
		Content: "ROLE: VALIDATOR / PREDICATE EVALUATOR\n" +
			"- Evaluate candidate output strictly against defined acceptance rubric.\n" +
			"- Output ONLY structured evaluation verdict: {\"verdict\": \"PASS\"|\"FAIL\", \"reason\": \"...\"}.\n" +
			"- No tools permitted. Do not generate code or commentary.",
		Kind:      promptcomp.KindContract,
		Rank:      20,
		DependsOn: []string{PartIDSafetyFloor},
		Predicate: func(e promptcomp.Env) bool {
			return e.AgentTier == "validator"
		},
	}

	toolsFull := promptcomp.PromptPart{
		ID: PartIDToolsFull,
		Content: "TOOL CATALOG & GRAMMAR:\n" +
			"- File operations: Read, Edit, Write, Glob, Grep.\n" +
			"- Terminal operations: Bash (safe execution with directory verification).\n" +
			"- Worker orchestration: task, dos_arbitrate, dos_verify, dos_status.\n" +
			"- Tool discovery: fak_tools_search for dynamic schema retrieval.",
		Kind:      promptcomp.KindTools,
		Rank:      30,
		DependsOn: []string{PartIDContractCoord},
		Predicate: func(e promptcomp.Env) bool {
			return e.AgentTier == "coordinator"
		},
	}

	toolsMinimal := promptcomp.PromptPart{
		ID: PartIDToolsMinimal,
		Content: "TOOL CATALOG:\n" +
			"- Available tools: Read, Edit, Write, Bash, Glob, Grep.\n" +
			"- Use Read before Edit. Confirm exact string match before replacement.\n" +
			"- Run verification command via Bash before declaring task complete.",
		Kind:      promptcomp.KindTools,
		Rank:      30,
		DependsOn: []string{PartIDContractLeaf},
		Predicate: func(e promptcomp.Env) bool {
			return e.AgentTier == "leaf"
		},
	}

	overlayDelegation := promptcomp.PromptPart{
		ID: PartIDOverlayDelegation,
		Content: "DELEGATION & COLLISION PROTOCOL:\n" +
			"- Enforce tree-disjointness across child worktrees.\n" +
			"- Price child fan-out before launch; gate execution if concurrency budget is exceeded.\n" +
			"- Reconcile child state upon exit; release leases immediately after receipt capture.",
		Kind:      promptcomp.KindOverlay,
		Rank:      40,
		DependsOn: []string{PartIDContractCoord},
		Predicate: func(e promptcomp.Env) bool {
			return e.AgentTier == "coordinator"
		},
	}

	overlayGuidance := promptcomp.PromptPart{
		ID: PartIDOverlayGuidance,
		Content: "CONVENTIONS:\n" +
			"- Conventional Commits: feat(scope): ..., fix(scope): ... with (fak <leaf>) trailer.\n" +
			"- Maintain existing style, typing, and architectural patterns.",
		Kind:      promptcomp.KindOverlay,
		Rank:      40,
		DependsOn: []string{PartIDContractLeaf},
		Predicate: func(e promptcomp.Env) bool {
			// Automatically drop secondary guidance when budget is tightly constrained (< 16,000 tokens)
			return e.AgentTier == "leaf" && e.ContextBudget >= 16000
		},
	}

	return []promptcomp.PromptPart{
		spine,
		safetyFloor,
		safetyFull,
		contractCoord,
		contractLeaf,
		contractValidator,
		toolsFull,
		toolsMinimal,
		overlayDelegation,
		overlayGuidance,
	}
}

// ResolvePrompt dynamically compiles the optimal prompt matching the given specification.
func ResolvePrompt(spec PromptSpec) (*promptcomp.CompiledPrompt, error) {
	parts := DefaultParts(spec.Scale)
	if len(spec.ExtraParts) > 0 {
		parts = append(parts, spec.ExtraParts...)
	}

	env := promptcomp.Env{
		ModelFamily:   spec.ModelFamily,
		IsSmallLocal:  spec.IsSmallLocal,
		AgentTier:     spec.Scale.String(),
		ContextBudget: spec.ContextBudget,
		WireFormat:    spec.WireFormat,
		Extra: map[string]any{
			"scale": spec.Scale,
		},
	}

	compiled, err := promptcomp.CompileParts(parts, env)
	if err != nil {
		return nil, fmt.Errorf("harnessinit: failed to compile prompt for scale %s: %w", spec.Scale, err)
	}

	return compiled, nil
}
