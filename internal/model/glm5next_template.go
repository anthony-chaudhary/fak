package model

import (
	"fmt"
	"strings"
)

const (
	GLM5NextTokenGMask       = "[gMASK]"
	GLM5NextTokenSop         = "<sop>"
	GLM5NextTokenSystem      = "<|system|>"
	GLM5NextTokenUser        = "<|user|>"
	GLM5NextTokenAssistant   = "<|assistant|>"
	GLM5NextTokenObservation = "<|observation|>"
	GLM5NextTokenThought     = "<|thought|>"
	GLM5NextTokenBeginImage  = "<|begin_of_image|>"
	GLM5NextTokenEndImage    = "<|end_of_image|>"
	GLM5NextTokenImagePad    = "<|image_pad|>"
)

// GLM5NextMessage represents a single message turn in the GLM-5.3-Flash format.
type GLM5NextMessage struct {
	Role     string // "system", "user", "assistant", "observation" (tool)
	Content  string
	Thinking string // optional reasoning content for assistant turns
}

// FormatGLM5NextPrompt renders a slice of messages into the canonical GLM-5.3-Flash prompt string.
// Every prompt begins with "[gMASK]<sop>".
// Messages are rendered with role delimiters (<|system|>, <|user|>, <|assistant|>, <|observation|>).
// If addAssistantPrefix is true, an open "<|assistant|>\n" token is appended for completion generation.
func FormatGLM5NextPrompt(messages []GLM5NextMessage, addAssistantPrefix bool) string {
	var sb strings.Builder
	sb.WriteString(GLM5NextTokenGMask)
	sb.WriteString(GLM5NextTokenSop)

	for _, msg := range messages {
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		switch role {
		case "system":
			sb.WriteString(GLM5NextTokenSystem)
			sb.WriteString("\n")
			sb.WriteString(msg.Content)
		case "user":
			sb.WriteString(GLM5NextTokenUser)
			sb.WriteString("\n")
			sb.WriteString(msg.Content)
		case "assistant":
			sb.WriteString(GLM5NextTokenAssistant)
			sb.WriteString("\n")
			if msg.Thinking != "" {
				sb.WriteString(GLM5NextTokenThought)
				sb.WriteString("\n")
				sb.WriteString(msg.Thinking)
				sb.WriteString("\n")
				sb.WriteString(GLM5NextTokenThought)
			}
			if msg.Content != "" {
				if msg.Thinking != "" {
					sb.WriteString("\n")
				}
				sb.WriteString(msg.Content)
			}
		case "observation", "tool":
			sb.WriteString(GLM5NextTokenObservation)
			sb.WriteString("\n")
			sb.WriteString(msg.Content)
		default:
			sb.WriteString(fmt.Sprintf("<|%s|>\n", role))
			sb.WriteString(msg.Content)
		}
	}

	if addAssistantPrefix {
		sb.WriteString(GLM5NextTokenAssistant)
		sb.WriteString("\n")
	}

	return sb.String()
}

// ParseGLM5NextAssistantResponse parses raw generated text from the model,
// separating reasoning content inside <|thought|>...<|thought|> from final content.
func ParseGLM5NextAssistantResponse(raw string) (thinking, content string) {
	if strings.Contains(raw, GLM5NextTokenThought) {
		parts := strings.Split(raw, GLM5NextTokenThought)
		if len(parts) >= 3 {
			thinking = strings.TrimSpace(parts[1])
			content = strings.TrimSpace(strings.Join(parts[2:], GLM5NextTokenThought))
			return thinking, content
		} else if len(parts) == 2 {
			thinking = strings.TrimSpace(parts[1])
			return thinking, ""
		}
	}
	return "", strings.TrimSpace(raw)
}
