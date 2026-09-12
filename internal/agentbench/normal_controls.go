package agentbench

import (
	"fmt"
	"strings"
)

func applyNormalControl(corpus *normalCorpus, name string, request plannedRequest, session int, base referenceRequest) referenceRequest {
	switch name {
	case "same-prefix-forks":
		// Each fork diverges only after the same substantial, frozen branch point.
		base.Messages = append(base.Messages, chatMessage{Role: "user", Content: corpus.SystemTools.content + "\n" + strings.Repeat("frozen branch context ", 320) + fmt.Sprintf("fork-%d", request.turn)})
	case "new-area-system-warm":
		if request.turn > 1 {
			base.Messages[2].Content = corpus.Areas["B"].content
		}
		base.Messages = append(base.Messages, chatMessage{Role: "user", Content: "inspect the area while retaining the system prefix"})
	case "no-share":
		base.Messages[0].Content = fmt.Sprintf("nonshare-%d-%d\n", session, request.turn) + base.Messages[0].Content
	case "source-edit-reread":
		before := "func retryDelay(attempt int) int { return attempt * 250 }"
		after := "func retryDelay(attempt int) int { return (attempt + 1) * 250 }"
		base.Messages = append(base.Messages,
			chatMessage{Role: "assistant", ToolCalls: []toolCall{{ID: "source-edit", Type: "function", Function: toolFunction{Name: "Edit", Arguments: fmt.Sprintf(`{"file_path":"retry.go","old_string":%q,"new_string":%q}`, before, after)}}}},
			chatMessage{Role: "tool", ToolCallID: "source-edit", Content: "changed bytes: " + after},
			chatMessage{Role: "assistant", ToolCalls: []toolCall{{ID: "source-reread", Type: "function", Function: toolFunction{Name: "Read", Arguments: `{"file_path":"retry.go"}`}}}},
			chatMessage{Role: "tool", ToolCallID: "source-reread", Content: after})
	case "area-eviction-revisit":
		area := corpus.Areas["A"].content
		if request.turn == 2 {
			area = corpus.Areas["B"].content
		}
		if request.turn == 3 {
			area = "explicit logical cache pressure material\n" + corpus.RepositoryMap.content + corpus.SystemTools.content
		}
		base.Messages[2].Content = area
		base.Messages = append(base.Messages, chatMessage{Role: "user", Content: fmt.Sprintf("logical area sequence step %d; backend residency remains unknown", request.turn)})
	case "tool-completion-burst":
		id := fmt.Sprintf("burst-%d-%d", session, request.turn)
		base.Messages = append(base.Messages, chatMessage{Role: "assistant", ToolCalls: []toolCall{{ID: id, Type: "function", Function: toolFunction{Name: "Read", Arguments: `{"file_path":"README.md"}`}}}}, chatMessage{Role: "tool", ToolCallID: id, Content: "independently scheduled recorded tool completion"})
	}
	return base
}
