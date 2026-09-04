// Package turnkind classifies the latest user turn of an agentic conversation
// from message STRUCTURE alone — which content-block types the last user
// message carries, never their content. It is the pure-sensor first rung of
// #3307 (lower reasoning effort on mechanical tool-result continuations): on
// the guard/serve Anthropic passthrough most turns are the model resuming
// after a routine tool call — the last user message holds only clean
// tool_result blocks, no fresh instruction, no error — and those turns do not
// need the full reasoning effort the harness pinned. A downstream actuator
// (lower-only, never injecting, shadow-first) keys off KindMechanical; that
// gateway wiring is the follow-on rung under the same issue, not this package.
//
// The classifier is deterministic and content-free by construction: it looks
// at block types and the tool_result error flag only, so a shadow log of its
// verdicts is safe. Error turns and new asks keep full effort by construction —
// a failure needs reasoning. Clean-room Go reimplementation of the turn
// classifier in headroomlabs-ai/headroom output_shaper.py (Apache-2.0, pinned
// commit 38074888); sibling idiom: internal/compactcohere's per-turn
// structural Classify for the cache-prefix axis.
package turnkind

// Anthropic content-block type strings this classifier distinguishes. Any
// other type (tool_use, or a future block kind) is structurally unrecognized
// and pushes the turn toward KindUnknown rather than a guessed verdict.
const (
	// BlockText is a text block — a genuinely new instruction from the user.
	BlockText = "text"
	// BlockImage is an image block — likewise a new ask (fresh input to reason over).
	BlockImage = "image"
	// BlockDocument is a document block — also fresh input, treated as a new ask
	// (the headroom source classifies text/image/document identically).
	BlockDocument = "document"
	// BlockToolResult is a tool_result block — the harness returning tool output.
	BlockToolResult = "tool_result"
)

// Block is the minimal, dependency-free shape of one content block in the last
// user message. Callers project whatever richer representation they hold (the
// gateway's raw JSON, an SDK struct) down to this; the classifier deliberately
// depends on nothing in internal/agent or internal/gateway so it stays a pure
// leaf.
type Block struct {
	// Type is the Anthropic content-block type: "text", "image", "document",
	// "tool_result", "tool_use", ...
	Type string
	// IsError mirrors tool_result's is_error flag. Only meaningful when
	// Type == BlockToolResult; ignored on every other block type.
	IsError bool
}

// Kind is the structural classification of the latest user turn.
type Kind int

const (
	// KindUnknown: no blocks, or a block mix the classifier does not
	// recognize as one of the three known shapes. The safe verdict — an
	// actuator must treat it as "do not touch effort".
	KindUnknown Kind = iota
	// KindNewAsk: the last user message carries a text, image, or document
	// block — a genuinely new instruction that deserves full reasoning
	// effort, regardless of what else rides along in the same message.
	KindNewAsk
	// KindErrorContinuation: at least one tool_result block has is_error set —
	// the model is recovering from a failed tool call. A failure needs
	// reasoning, so effort is never lowered here.
	KindErrorContinuation
	// KindMechanical: the message is ONLY clean tool_result blocks — a
	// mechanical continuation carrying just tool output. This is the one
	// verdict the effort-lowering actuator may act on.
	KindMechanical
)

// String returns a stable lowercase label for k, suitable for shadow-log
// lines like "turnkind=mechanical effort:high->low".
func (k Kind) String() string {
	switch k {
	case KindNewAsk:
		return "new_ask"
	case KindErrorContinuation:
		return "error_continuation"
	case KindMechanical:
		return "mechanical"
	default:
		return "unknown"
	}
}

// Invariant: turn kind classification is fail-closed and deterministic.
// Guard: any unrecognized block shape, empty input, or ambiguous sequence safely
// resolves to KindUnknown without inspecting or leaking message content.
// Precondition: caller provides the sequence of content blocks from the latest user message.
// Postcondition: returns a deterministic Kind based strictly on structural block types.
//
// Classify returns the structural kind of the last user message given its
// content blocks. Precedence is fixed and order-independent of the slice:
//
//  1. any text/image/document block  → KindNewAsk (a new ask wins outright)
//  2. else any errored tool_result   → KindErrorContinuation
//  3. else only tool_result blocks   → KindMechanical
//  4. else (empty, or an unrecognized mix) → KindUnknown
//
// Rung 3 requires purity: a single unrecognized block type alongside clean
// tool_results demotes the turn to KindUnknown rather than risking a lowered
// effort on a shape this classifier has never seen.
func Classify(lastUserMessageBlocks []Block) Kind {
	if len(lastUserMessageBlocks) == 0 {
		return KindUnknown
	}
	sawError := false
	onlyToolResults := true
	for _, b := range lastUserMessageBlocks {
		switch b.Type {
		case BlockText, BlockImage, BlockDocument:
			return KindNewAsk
		case BlockToolResult:
			if b.IsError {
				sawError = true
			}
		default:
			onlyToolResults = false
		}
	}
	if sawError {
		return KindErrorContinuation
	}
	if onlyToolResults {
		return KindMechanical
	}
	return KindUnknown
}
