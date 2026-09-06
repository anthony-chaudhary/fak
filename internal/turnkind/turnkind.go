// Package turnkind classifies the latest user turn of an agentic conversation
// by content-block structure without inspecting message content.
package turnkind

const (
	// BlockText indicates a text block representing user instruction.
	BlockText = "text"
	// BlockImage indicates an image input block provided by the user.
	BlockImage = "image"
	// BlockDocument indicates a document input block provided by the user.
	BlockDocument = "document"
	// BlockToolResult indicates tool execution output returned to the conversation.
	BlockToolResult = "tool_result"
)

// Block represents a single content block from the latest user message.
type Block struct {
	// Type identifies the content block category (e.g. text, tool_result).
	Type string
	// IsError indicates whether a tool_result block represents an execution failure.
	IsError bool
}

// Kind is the structural classification of the latest user turn.
type Kind int

const (
	// KindUnknown indicates an empty message or an unrecognized block structure.
	KindUnknown Kind = iota
	// KindNewAsk indicates new user input requiring reasoning (text, image, or document).
	KindNewAsk
	// KindErrorContinuation indicates recovery from an errored tool result.
	KindErrorContinuation
	// KindMechanical indicates an automated continuation containing only successful tool results.
	KindMechanical
)

// String returns a lowercase string label for the turn kind.
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

// Classify returns the structural kind of the latest user message given its content blocks.
// Evaluation order:
//  1. Any text, image, or document block -> KindNewAsk
//  2. Any errored tool_result block -> KindErrorContinuation
//  3. Exclusively clean tool_result blocks -> KindMechanical
//  4. Empty or unrecognized blocks -> KindUnknown
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
