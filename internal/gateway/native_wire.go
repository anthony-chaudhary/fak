package gateway

// native_wire.go — the CONVERSION half of the native wire seam (#6657).
//
// `fak serve --native` hands a served /v1/messages turn to fak's own agent loop
// (native_serve.go). Until this file, that hand-off discarded most of the client
// contract: the loop was seeded with lastUserText(req.Messages) — a single string — and
// req.Tools was never read at all, so every native run advertised the built-in
// agent.ToolCatalog() fixture no matter what the caller declared.
//
// nativeWireSeed is the one conversion BOTH native handlers (buffered and streamed)
// run, so they cannot drift: it carries the ordered conversation and the request-scoped
// tool catalog across the seam as agent.RunOptions. It is deliberately a pure function
// of the decoded request — no provider round trip, no re-parse of req.Raw — because the
// data it moves was already parsed by DecodeAnthropicMessagesRequest.
//
// Fail-closed (P3): a declaration this seam cannot honor is refused with a closed reason
// token BEFORE the loop runs. A tool that is silently dropped, or silently replaced by
// the fixture, would let a caller believe its catalog was in play when it never was.
//
// Honest fence: this issue is the WIRE half — the loop now receives and advertises the
// caller's declarations. EXECUTING a caller-declared tool (routing its call back out to
// the client, or registering it with the kernel) is the separate follow-on tracked for
// the coding-engine / #1380 work; a call to a tool the kernel does not implement still
// comes back as the kernel's typed unknown-tool receipt, which the model can adapt to.

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// The closed reason vocabulary this seam refuses with. Each names exactly what the
// caller must change; none of them is a free-text message.
const (
	// nativeReasonToolNameMissing: a tools[] entry carried no name, so nothing could be
	// advertised for it and no call could ever be attributed back to it.
	nativeReasonToolNameMissing = "NATIVE_TOOL_NAME_MISSING"
	// nativeReasonToolNameDuplicate: two tools[] entries share a name — a model tool call
	// naming it could not be resolved to one declaration.
	nativeReasonToolNameDuplicate = "NATIVE_TOOL_NAME_DUPLICATE"
	// nativeReasonToolSchemaInvalid: input_schema was present but is not a JSON object,
	// so it is not a JSON Schema the model can be given.
	nativeReasonToolSchemaInvalid = "NATIVE_TOOL_SCHEMA_INVALID"
	// Unsupported message roles are rejected before the loop runs.
	// nativeReasonMessageRoleUnsupported: a message role cannot be represented by
	// the owned loop; the caller must use a supported native message role.
	nativeReasonMessageRoleUnsupported = "NATIVE_MESSAGE_ROLE_UNSUPPORTED"
)

// nativeEmptyToolSchema is the schema substituted for a declaration that omits
// input_schema entirely: an object with no properties, the Anthropic default for a
// no-argument tool. This is a stated normalization, not a silent one — an input_schema
// that IS present but malformed is refused rather than replaced.
const nativeEmptyToolSchema = `{"type":"object","properties":{}}`

// nativeWireError is the typed fail-closed refusal of a served request the native seam
// cannot honor. Reason is drawn from the closed vocabulary above; Detail names the
// offending declaration so the caller can fix it without guessing.
type nativeWireError struct {
	Reason string
	Detail string
}

func (e *nativeWireError) Error() string { return e.Reason + ": " + e.Detail }

// writeNativeWireErr renders a seam refusal as a 400 carrying the closed reason as the
// error code — the same error envelope every other gateway refusal uses.
func writeNativeWireErr(w http.ResponseWriter, err *nativeWireError) {
	writeErrCode(w, http.StatusBadRequest, err.Reason, err.Detail)
}

// nativeWireSeed is one served request, converted for the owned loop.
type nativeWireSeed struct {
	// Conversation is the caller's ordered transcript (roles and content preserved),
	// spliced into the loop after its own system prompt. Empty for a request with no
	// messages, which leaves the historical Task-only seed.
	Conversation []agent.Message
	// Tools is the request-scoped catalog. nil means the caller declared none, which
	// deliberately leaves the kernel-owned agent.ToolCatalog() standing — the existing
	// no-tools native run, unchanged.
	Tools []agent.ToolDef
	// Task is the last user message, kept as the loop's task string for the metrics and
	// trace fields that name it, and as the seed when Conversation is empty.
	Task string
}

// newNativeWireSeed converts a decoded served request into the loop seed, or refuses it
// with a typed reason. It is the single conversion both native handlers run. A nil
// request yields the zero seed, which drives the byte-for-byte historical loop.
func newNativeWireSeed(req *agent.AnthropicMessagesRequest) (nativeWireSeed, *nativeWireError) {
	if req == nil {
		return nativeWireSeed{}, nil
	}
	seed := nativeWireSeed{Task: lastUserText(req.Messages)}
	conv, err := nativeConversation(req.Messages)
	if err != nil {
		return nativeWireSeed{}, err
	}
	seed.Conversation = conv
	tools, err := nativeRequestTools(req.Tools)
	if err != nil {
		return nativeWireSeed{}, err
	}
	seed.Tools = tools
	return seed, nil
}

// nativeConversation validates and copies the served transcript. DecodeAnthropicMessages
// Request has already lowered Anthropic content blocks onto the canonical Message shape
// (tool_use → ToolCalls, tool_result → RoleTool), so this preserves order and roles
// rather than re-deriving them; its job is to refuse a role the loop has no seat for.
func nativeConversation(messages []agent.Message) ([]agent.Message, *nativeWireError) {
	if len(messages) == 0 {
		return nil, nil
	}
	out := make([]agent.Message, 0, len(messages))
	for i, m := range messages {
		switch m.Role {
		case agent.RoleSystem, agent.RoleUser, agent.RoleAssistant, agent.RoleTool:
		default:
			return nil, &nativeWireError{
				Reason: nativeReasonMessageRoleUnsupported,
				Detail: fmt.Sprintf("messages[%d] has role %q; supported roles are system, user, assistant, tool", i, m.Role),
			}
		}
		out = append(out, m)
	}
	return out, nil
}

// nativeRequestTools converts the request's declarations into the loop's request-scoped
// catalog. A caller that declared nothing gets nil (the kernel catalog stands); a caller
// that declared something gets exactly that, schema-checked, with no fixture blended in.
func nativeRequestTools(tools []agent.ToolDef) ([]agent.ToolDef, *nativeWireError) {
	if len(tools) == 0 {
		return nil, nil
	}
	seen := make(map[string]struct{}, len(tools))
	out := make([]agent.ToolDef, 0, len(tools))
	for i, td := range tools {
		name := td.Function.Name
		if name == "" {
			return nil, &nativeWireError{
				Reason: nativeReasonToolNameMissing,
				Detail: fmt.Sprintf("tools[%d] declares no name", i),
			}
		}
		if _, dup := seen[name]; dup {
			return nil, &nativeWireError{
				Reason: nativeReasonToolNameDuplicate,
				Detail: fmt.Sprintf("tools[%d] re-declares the name %q; a tool call naming it could not be resolved", i, name),
			}
		}
		seen[name] = struct{}{}

		schema := td.Function.Parameters
		if len(schema) == 0 {
			schema = json.RawMessage(nativeEmptyToolSchema)
		} else {
			var probe map[string]json.RawMessage
			if err := json.Unmarshal(schema, &probe); err != nil {
				return nil, &nativeWireError{
					Reason: nativeReasonToolSchemaInvalid,
					Detail: fmt.Sprintf("tools[%d] (%q) input_schema is not a JSON object: %v", i, name, err),
				}
			}
		}
		out = append(out, agent.ToolDef{
			Type: "function",
			Function: agent.ToolDefFunction{
				Name:        name,
				Description: td.Function.Description,
				Parameters:  schema,
			},
		})
	}
	return out, nil
}
