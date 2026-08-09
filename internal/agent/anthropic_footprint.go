package agent

// anthropic_footprint.go — the structural token AUDIT of one inbound Anthropic
// Messages request: where do the input tokens Claude Code (or any Messages client)
// sends on this turn actually GO?
//
// EstimateAnthropicTokens already walks the same three surfaces — the system prompt,
// the tool definitions, and the message history — but folds them into ONE opaque
// number. That number answers "how big is the request"; it cannot answer "which
// slice is the bloat." RequestFootprint is the bucketed twin: the SAME char-walk,
// partitioned into named slices, so an operator (or the #2924 tool-schema-footprint
// gate) can see the split instead of the sum.
//
// The load-bearing derived number is the FLOOR = system + tools: the fixed per-call
// tax paid on EVERY turn regardless of how long the conversation has grown. The floor
// is the "clean minimal baseline" — the irreducible cost of a request with an empty
// history — and it is where prompt/tool-schema distillation pays back once per turn.
// History + tail are the part that grows (and that compaction/elision shed).
//
// Provenance: ESTIMATED. Like EstimateAnthropicTokens this uses the ~4-char/token
// house heuristic (bytesPerTokenEstimate), NOT the provider's OBSERVED usage counters
// (those live in the gateway's ctxvalue arm and report resident tokens as one number).
// The two compose: OBSERVED says how FULL the window is; this ESTIMATED split says
// WHERE the bytes went. Computing a footprint is audit-only — it mutates nothing the
// model sees, so it is lossless by construction.

// FootprintProvenance labels every RequestFootprint: the ~4-char/token estimate, not
// a provider-relayed count. Kept as a const so every surface that renders a footprint
// prints the same owner label (Law A2 — every value carries its provenance).
const FootprintProvenance = "ESTIMATED"

// tokenDivisor is a rational characters-per-token calibration. Keeping it rational
// makes the estimator deterministic without rounding a measured ratio to an integer.
type tokenDivisor struct {
	chars  int
	tokens int
}

var (
	// These calibrations are pinned by TestAnthropicTokenDivisorsAgainstRecordedFixtures.
	// Prose and JSON Schema are deliberately separate surfaces: one shared divisor can
	// turn an accounting error into a bad headroom/eviction decision.
	proseTokenDivisor      = tokenDivisor{chars: 15, tokens: 4} // 3.75 chars/token
	jsonSchemaTokenDivisor = tokenDivisor{chars: 9, tokens: 2}  // 4.50 chars/token
)

func estimateTokens(bytes int, divisor tokenDivisor) int {
	if bytes <= 0 {
		return 0
	}
	return bytes * divisor.tokens / divisor.chars
}

// FootprintBucket is one labeled slice of a request's estimated input-token cost.
// Bytes is the exact, additive quantity (bucket bytes sum to Total.Bytes); Tokens is
// the derived Bytes/bytesPerTokenEstimate, independently floored per bucket, so the
// per-bucket Tokens may sum to slightly under Total.Tokens (floor-of-sum ≥ sum-of-
// floors). Pct is the bucket's share of Total.Tokens.
type FootprintBucket struct {
	Bytes  int     `json:"bytes"`
	Tokens int     `json:"tokens"`
	Pct    float64 `json:"pct"`
}

// ToolFootprint is one tool schema's per-call cost — the exact primitive #2924's
// tool-schema-footprint gate needs: the token tax each registered tool adds to every
// API call, whether or not the tool is ever selected.
type ToolFootprint struct {
	Name   string `json:"name"`
	Bytes  int    `json:"bytes"`
	Tokens int    `json:"tokens"`
}

// Footprint is the estimated structural decomposition of one inbound Anthropic
// Messages request. System + Tools + History + Tail partition the whole request; Floor
// (= System + Tools) and Total are derived roll-ups carried for convenience so a reader
// never re-adds them by hand.
type Footprint struct {
	Provenance string `json:"provenance"` // always ESTIMATED

	System  FootprintBucket `json:"system"`  // the system prompt (harness spine + any injected memory/CLAUDE.md)
	Tools   FootprintBucket `json:"tools"`   // all tool definitions (names + descriptions + JSON-Schema parameters)
	History FootprintBucket `json:"history"` // every message EXCEPT the most-recent one (the cacheable body)
	Tail    FootprintBucket `json:"tail"`    // the most-recent message (the volatile suffix)

	// Floor = System + Tools: the fixed per-call tax paid every turn regardless of
	// history depth — the "clean minimal baseline" a distillation pass drives down.
	Floor FootprintBucket `json:"floor"`
	// Total = System + Tools + History + Tail. Total.Tokens == EstimateAnthropicTokens(req).
	Total FootprintBucket `json:"total"`

	PerTool      []ToolFootprint `json:"per_tool,omitempty"` // #2924: per-tool schema cost, largest first not guaranteed
	ToolCount    int             `json:"tool_count"`
	MessageCount int             `json:"message_count"`
}

// bucketFromEstimate builds a bucket from exact byte and estimated-token counts
// against a shared token total (so every bucket's Pct has the SAME denominator).
func bucketFromEstimate(bytes, tokens, totalTokens int) FootprintBucket {
	b := FootprintBucket{Bytes: bytes, Tokens: tokens}
	if totalTokens > 0 {
		b.Pct = float64(b.Tokens) * 100 / float64(totalTokens)
	}
	return b
}

// messageBytes is the exact char cost EstimateAnthropicTokens charges one message:
// its text content plus every tool call's name and raw arguments. Kept identical to
// the estimator's inner loop so the footprint stays a faithful partition of it.
func messageBytes(m Message) int {
	n := len(m.Content)
	for _, tc := range m.ToolCalls {
		n += len(tc.Function.Name) + len(tc.Function.Arguments)
	}
	return n
}

// toolBytes is the exact char cost EstimateAnthropicTokens charges one tool def: its
// name, description, and serialized JSON-Schema parameters.
func toolBytes(t ToolDef) int {
	return len(t.Function.Name) + len(t.Function.Description) + len(t.Function.Parameters)
}

// DeFoldSystemRequest corrects the folded-system double-count that is specific to a DECODED
// live request. DecodeAnthropicMessagesRequest prepends the system prompt as a leading
// RoleSystem message into req.Messages AND keeps req.System, so a naive RequestFootprint
// counts the system twice (the System bucket AND a History/Tail message). This returns a
// shallow copy whose leading folded-system duplicate is dropped, so System is counted once
// and Floor == System + Tools matches /context.
//
// The original req is NEVER mutated — the gateway hot path forwards the same pointer
// verbatim after pricing it, so a mutation here would silently drop the system prompt from
// the request actually sent upstream. A nil req returns nil.
//
// It lives beside RequestFootprint because it is that function's precondition, and it is
// exported because every caller that prices a decoded request needs it: the gateway's
// per-trace footprint observer and `fak footprint-audit` both used to carry a
// byte-identical private copy of it.
func DeFoldSystemRequest(req *AnthropicMessagesRequest) *AnthropicMessagesRequest {
	if req == nil {
		return nil
	}
	if req.System != "" && len(req.Messages) > 0 &&
		req.Messages[0].Role == RoleSystem && req.Messages[0].Content == req.System {
		cp := *req
		cp.Messages = req.Messages[1:]
		return &cp
	}
	return req
}

// RequestFootprint decomposes req into the estimated per-slice token audit. It is the
// bucketed twin of EstimateAnthropicTokens: Total.Tokens == EstimateAnthropicTokens(req)
// by construction (same char-walk, same divisor). A nil req returns a zero footprint
// (still labeled ESTIMATED) rather than panicking, so a caller can render it unguarded.
func RequestFootprint(req *AnthropicMessagesRequest) Footprint {
	fp := Footprint{Provenance: FootprintProvenance}
	if req == nil {
		return fp
	}

	systemBytes := len(req.System)

	toolsBytes := 0
	perTool := make([]ToolFootprint, 0, len(req.Tools))
	for _, t := range req.Tools {
		tb := toolBytes(t)
		toolsBytes += tb
		perTool = append(perTool, ToolFootprint{
			Name:   t.Function.Name,
			Bytes:  tb,
			Tokens: estimateTokens(tb, jsonSchemaTokenDivisor),
		})
	}

	// History is every message except the last; Tail is the last message (the volatile
	// suffix that breaks the cache prefix). A single-message request is all Tail, no
	// History; an empty request is neither.
	historyBytes, tailBytes := 0, 0
	if n := len(req.Messages); n > 0 {
		for i := 0; i < n-1; i++ {
			historyBytes += messageBytes(req.Messages[i])
		}
		tailBytes = messageBytes(req.Messages[n-1])
	}

	totalBytes := systemBytes + toolsBytes + historyBytes + tailBytes
	systemTokens := estimateTokens(systemBytes, proseTokenDivisor)
	toolsTokens := estimateTokens(toolsBytes, jsonSchemaTokenDivisor)
	historyTokens := estimateTokens(historyBytes, proseTokenDivisor)
	tailTokens := estimateTokens(tailBytes, proseTokenDivisor)
	totalTokens := systemTokens + toolsTokens + historyTokens + tailTokens

	fp.System = bucketFromEstimate(systemBytes, systemTokens, totalTokens)
	fp.Tools = bucketFromEstimate(toolsBytes, toolsTokens, totalTokens)
	fp.History = bucketFromEstimate(historyBytes, historyTokens, totalTokens)
	fp.Tail = bucketFromEstimate(tailBytes, tailTokens, totalTokens)
	fp.Floor = bucketFromEstimate(systemBytes+toolsBytes, systemTokens+toolsTokens, totalTokens)
	fp.Total = bucketFromEstimate(totalBytes, totalTokens, totalTokens)
	fp.PerTool = perTool
	fp.ToolCount = len(req.Tools)
	fp.MessageCount = len(req.Messages)
	return fp
}
