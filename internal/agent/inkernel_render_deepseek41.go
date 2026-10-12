package agent

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// DeepSeek V4.1 control tokens, as the GGUF vocab spells them.
const (
	deepSeekV41BOS       = "<｜begin▁of▁sentence｜>"
	deepSeekV41EOS       = "<｜end▁of▁sentence｜>"
	deepSeekV41User      = "<｜User｜>"
	deepSeekV41Assistant = "<｜Assistant｜>"
	deepSeekV41ThinkEnd  = "</think>"
)

const deepSeekV41ChatRendererID = "deepseek41-chat-nothink-v1"

// renderDeepSeekV41Chat renders a chat turn in DeepSeek V4.1's own template, the
// non-thinking branch: BOS, system text, then <｜User｜>/<｜Assistant｜> turns
// (vcruz305/llama.cpp src/llama-chat.cpp:548-561@5210c7c, DeepSeek-V3 layout),
// with </think> closing the empty reasoning block before each assistant reply as
// the GGUF's tokenizer.chat_template does. Tool results render as user turns; tool
// specs are not rendered.
func renderDeepSeekV41Chat(messages []Message) string {
	var b strings.Builder
	b.WriteString(deepSeekV41BOS)
	for _, m := range messages {
		switch m.Role {
		case RoleSystem:
			b.WriteString(m.Content)
		case RoleAssistant:
			b.WriteString(deepSeekV41Assistant + deepSeekV41ThinkEnd + m.Content + deepSeekV41EOS)
		default:
			b.WriteString(deepSeekV41User + m.Content)
		}
	}
	b.WriteString(deepSeekV41Assistant + deepSeekV41ThinkEnd)
	return b.String()
}

// inKernelRenderPrompt is the text a native planner encodes for one request: the
// raw V4.1 completion text, the V4.1 chat template, or ChatML for every other family.
func inKernelRenderPrompt(messages []Message, tools []ToolDef, cfg model.Config, sp SampleParams) string {
	switch {
	case inKernelRawV41Prompt(cfg, sp):
		var b strings.Builder
		for _, m := range messages {
			b.WriteString(m.Content)
		}
		return b.String()
	case cfg.IsDeepSeekV41():
		return renderDeepSeekV41Chat(messages)
	default:
		return renderInKernelChatMLRequest(messages, tools, cfg, sp.ResponseFormat, sp.ToolChoice, sp)
	}
}
