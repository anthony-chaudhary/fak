package gateway

// upstream_template_render.go — the typed client envelope for an upstream chat-template
// render failure (#13784). llama.cpp answers a message shape its Jinja template refuses with a
// bare 500 ("Jinja Exception: Unexpected message role."); the agent fails that fast, and this
// gives the client a branchable code plus the template's own reason on BOTH the buffered and
// the pre-first-byte streamed paths (both fold through upstreamErrorStatus).

// codeUpstreamTemplateRender is the OpenAI-style error code for a chat-template render failure.
const codeUpstreamTemplateRender = "upstream_template_render"

const templateRenderMessagePrefix = "upstream chat template rejected this request's message shape (not retryable as-is; fix the message roles/order)"

// templateRenderMessage builds the client message for a template render failure: a fixed
// literal plus the upstream's error message (not the raw body), scrubbed of token-shaped runs
// and bounded by scrubForbiddenDetail, so a credential echoed into the body can never ride out.
func templateRenderMessage(body string) string {
	if detail := scrubForbiddenDetail(parseUpstream400(body).Message); detail != "" {
		return templateRenderMessagePrefix + ": " + detail
	}
	return templateRenderMessagePrefix
}
