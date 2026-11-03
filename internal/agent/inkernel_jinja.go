package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// ErrNativeChatTemplate marks an explicit template contract refusal. Template,
// message, path and tool bytes are never included in its diagnostics.
var ErrNativeChatTemplate = errors.New("native chat template not admitted")

// NativeChatTemplate is immutable after construction and can be shared by
// planners. Identity binds exact source bytes and a caller-selected model ID;
// it does not assert an upstream release or numerical qualification.
type NativeChatTemplate struct {
	program      *nativeJinjaProgram
	sourceSHA256 string
	modelID      string
}

func (t *NativeChatTemplate) String() string {
	if t == nil {
		return "native-chat-template:none"
	}
	return "native-chat-template:llama-jinja-v1:sha256:" + t.sourceSHA256
}
func (t *NativeChatTemplate) SHA256() string {
	if t == nil {
		return ""
	}
	return t.sourceSHA256
}

// NewNativeChatTemplate compiles a bounded text template against an independently
// selected SHA-256 and exact model identity. No file or network is accessed.
func NewNativeChatTemplate(source, expectedSHA256, modelID string) (*NativeChatTemplate, error) {
	if len(source) == 0 || len(source) > jinjaMaxSource || len(modelID) > 4096 || strings.TrimSpace(modelID) == "" {
		return nil, fmt.Errorf("%w: source or model binding", ErrNativeChatTemplate)
	}
	if len(expectedSHA256) != 64 {
		return nil, fmt.Errorf("%w: digest binding", ErrNativeChatTemplate)
	}
	decoded, err := hex.DecodeString(expectedSHA256)
	if err != nil || len(decoded) != 32 || expectedSHA256 != strings.ToLower(expectedSHA256) {
		return nil, fmt.Errorf("%w: digest binding", ErrNativeChatTemplate)
	}
	sum := sha256.Sum256([]byte(source))
	if hex.EncodeToString(sum[:]) != expectedSHA256 {
		return nil, fmt.Errorf("%w: digest mismatch", ErrNativeChatTemplate)
	}
	program, err := compileNativeJinja(source)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrNativeChatTemplate, err)
	}
	return &NativeChatTemplate{program: program, sourceSHA256: expectedSHA256, modelID: modelID}, nil
}

// LoadNativeChatTemplateFile reads one explicitly selected operator-owned file.
// It performs no environment discovery, URL fetch, directory scan or reload.
// Callers must handle refusal before constructing a selected native planner.
func LoadNativeChatTemplateFile(path, expectedSHA256, modelID string) (*NativeChatTemplate, error) {
	if path == "" || len(path) > 4096 || len(modelID) > 4096 || strings.TrimSpace(modelID) == "" {
		return nil, fmt.Errorf("%w: incomplete file selection", ErrNativeChatTemplate)
	}
	if len(expectedSHA256) != 64 {
		return nil, fmt.Errorf("%w: digest binding", ErrNativeChatTemplate)
	}
	decoded, err := hex.DecodeString(expectedSHA256)
	if err != nil || len(decoded) != 32 || expectedSHA256 != strings.ToLower(expectedSHA256) {
		return nil, fmt.Errorf("%w: digest binding", ErrNativeChatTemplate)
	}
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > jinjaMaxSource {
		return nil, fmt.Errorf("%w: selected file", ErrNativeChatTemplate)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("%w: selected file", ErrNativeChatTemplate)
	}
	opened, err := f.Stat()
	if err != nil || !opened.Mode().IsRegular() || !os.SameFile(info, opened) {
		_ = f.Close()
		return nil, fmt.Errorf("%w: changed selected file", ErrNativeChatTemplate)
	}
	raw, readErr := io.ReadAll(io.LimitReader(f, jinjaMaxSource+1))
	closeErr := f.Close()
	if readErr != nil || closeErr != nil || len(raw) > jinjaMaxSource || int64(len(raw)) != info.Size() {
		return nil, fmt.Errorf("%w: selected file bytes", ErrNativeChatTemplate)
	}
	return NewNativeChatTemplate(string(raw), expectedSHA256, modelID)
}

func (p *InKernelPlanner) admitNativeChatTemplate(messages []Message, tools []ToolDef, sp SampleParams) error {
	t := p.nativeChatTemplate
	if t == nil {
		return nil
	}
	if p.m == nil || p.tok == nil || t.program == nil || t.modelID != p.modelID {
		return fmt.Errorf("%w: model binding", ErrNativeChatTemplate)
	}
	if identity, known := p.m.AuthenticatedArtifactIdentity(); known && identity.ChatTemplateSHA256 != "" && identity.ChatTemplateSHA256 != t.sourceSHA256 {
		return fmt.Errorf("%w: authenticated template differs", ErrNativeChatTemplate)
	}
	if x, ok := any(p.tok).(interface{ IsSPM() bool }); ok && x.IsSPM() {
		return fmt.Errorf("%w: tokenizer segmented contract unavailable", ErrNativeChatTemplate)
	}
	if sp.RawPrompt || p.deepSeekV41ChatProfile != "" || p.dflash2 != nil || len(sp.ChatTemplateKwargs) != 0 {
		return fmt.Errorf("%w: incompatible request profile", ErrNativeChatTemplate)
	}
	if len(messages) > 4096 || len(tools) > 1024 {
		return fmt.Errorf("%w: input count", ErrNativeChatTemplate)
	}
	total := 0
	items := len(messages) + len(tools)
	add := func(n int) bool {
		if n < 0 || n > jinjaMaxInput-total {
			return false
		}
		total += n
		return true
	}
	for _, m := range messages {
		if m.ContentWasStructured || m.VisionPrompt != "" || len(m.VisionImages) != 0 || m.FunctionCall != nil || m.Thinking != "" || m.ThinkingSignature != "" || len(m.RedactedThinking) != 0 {
			return fmt.Errorf("%w: text-only message contract", ErrNativeChatTemplate)
		}
		if !add(len(m.Role) + len(m.Content) + len(m.Name) + len(m.ToolCallID) + len(m.ReasoningContent)) {
			return fmt.Errorf("%w: input bytes", ErrNativeChatTemplate)
		}
		if len(m.ToolCalls) > 1024 || len(m.ToolCalls) > 16384-items {
			return fmt.Errorf("%w: tool call count", ErrNativeChatTemplate)
		}
		items += len(m.ToolCalls)
		for _, tc := range m.ToolCalls {
			if tc.ThoughtSignature != "" || tc.Function.Namespace != "" || (tc.Type != "" && tc.Type != "function") {
				return fmt.Errorf("%w: tool-call profile", ErrNativeChatTemplate)
			}
			if !add(len(tc.ID) + len(tc.Type) + len(tc.Function.Name) + len(tc.Function.Arguments)) {
				return fmt.Errorf("%w: input bytes", ErrNativeChatTemplate)
			}
		}
	}
	for _, t := range tools {
		if len(t.ResponsesWire) != 0 || (t.Type != "" && t.Type != "function") {
			return fmt.Errorf("%w: function tools only", ErrNativeChatTemplate)
		}
		if !add(len(t.Type) + len(t.Function.Name) + len(t.Function.Description) + len(t.Function.Parameters)) {
			return fmt.Errorf("%w: input bytes", ErrNativeChatTemplate)
		}
	}
	return nil
}

// renderNativeChatTemplate runs only after exact option/profile admission. All
// request-origin strings, including roles, schema keys and tool arguments, are
// marked ordinary. Only source literals and model-declared BOS/EOS are trusted.
func (p *InKernelPlanner) renderNativeChatTemplate(ctx context.Context, messages []Message, tools []ToolDef, sp SampleParams) ([]tokenizer.Segment, error) {
	if err := p.admitNativeChatTemplate(messages, tools, sp); err != nil {
		return nil, err
	}
	if p.nativeChatTemplate == nil {
		return nil, fmt.Errorf("%w: no explicit template", ErrNativeChatTemplate)
	}
	// Preserve ordinary tool_choice:none filtering after original-shape admission.
	var choice string
	if json.Unmarshal(sp.ToolChoice, &choice) == nil && choice == "none" {
		tools = nil
	}
	instruction := inKernelResponseFormatInstruction(sp.ResponseFormat)
	if forced := inKernelForcedToolInstruction(sp.ToolChoice, tools); forced != "" {
		if instruction != "" {
			instruction += "\n"
		}
		instruction += forced
	}
	if instruction != "" {
		messages = append([]Message{{Role: RoleSystem, Content: instruction}}, messages...)
	}
	if inKernelEffectiveToolName(sp.ToolChoice, tools) != "" {
		tools = nil
	}
	// Synthesized existing constraint instructions are still data to this template.
	if err := p.admitNativeChatTemplate(messages, tools, sp); err != nil {
		return nil, err
	}
	parts, err := p.nativeChatTemplate.program.render(ctx, func(s *jinjaState) *jinjaScope {
		root := &jinjaScope{vars: map[string]*jinjaValue{}}
		msgs := s.value("array")
		msgs.tainted = true
		for _, m := range messages {
			v := s.obj(false)
			v.tainted = true
			s.setObject(v.object, "role", s.text(m.Role, false), 0)
			s.setObject(v.object, "content", s.text(m.Content, false), 0)
			if m.ToolResultIsError {
				s.setObject(v.object, "tool_result_is_error", s.boolean(true, true), 0)
			}
			if m.Name != "" {
				s.setObject(v.object, "name", s.text(m.Name, false), 0)
			}
			if m.ToolCallID != "" {
				s.setObject(v.object, "tool_call_id", s.text(m.ToolCallID, false), 0)
			}
			if m.ReasoningContent != "" {
				s.setObject(v.object, "reasoning_content", s.text(m.ReasoningContent, false), 0)
			}
			if len(m.ToolCalls) > 0 {
				calls := s.value("array")
				calls.tainted = true
				for _, tc := range m.ToolCalls {
					call := s.obj(false)
					call.tainted = true
					fn := s.obj(false)
					fn.tainted = true
					s.setObject(call.object, "id", s.text(tc.ID, false), 0)
					s.setObject(call.object, "type", s.text(tc.Type, false), 0)
					s.setObject(fn.object, "name", s.text(tc.Function.Name, false), 0)
					arg := s.obj(false)
					arg.tainted = true
					if strings.TrimSpace(tc.Function.Arguments) != "" {
						arg = s.fromJSON([]byte(tc.Function.Arguments), 0)
						if arg.kind != "object" {
							s.fail("tool-arguments-object", 0)
						}
					}
					s.setObject(fn.object, "arguments", arg, 0)
					s.setObject(call.object, "function", fn, 0)
					s.charge(0, 1, 0)
					calls.array = append(calls.array, call)
				}
				s.setObject(v.object, "tool_calls", calls, 0)
			}
			s.charge(0, 1, 0)
			msgs.array = append(msgs.array, v)
		}
		root.vars["messages"] = msgs
		defs := s.value("array")
		defs.tainted = true
		for _, tool := range tools {
			v := s.obj(false)
			v.tainted = true
			fn := s.obj(false)
			fn.tainted = true
			kind := tool.Type
			if kind == "" {
				kind = "function"
			}
			s.setObject(v.object, "type", s.text(kind, false), 0)
			s.setObject(fn.object, "name", s.text(tool.Function.Name, false), 0)
			s.setObject(fn.object, "description", s.text(tool.Function.Description, false), 0)
			params := s.obj(false)
			params.tainted = true
			if len(tool.Function.Parameters) > 0 {
				params = s.fromJSON(tool.Function.Parameters, 0)
				if params.kind != "object" {
					s.fail("tool-schema-object", 0)
				}
			}
			s.setObject(fn.object, "parameters", params, 0)
			s.setObject(v.object, "function", fn, 0)
			s.charge(0, 1, 0)
			defs.array = append(defs.array, v)
		}
		root.vars["tools"] = defs
		root.vars["add_generation_prompt"] = s.boolean(true, false)
		root.vars["enable_thinking"] = s.boolean(!inKernelSuppressQwenThinking(p.m.Cfg, sp), false)
		special := p.tok.SpecialTokens()
		if p.m.Cfg.BOSTokenID != nil {
			if text, ok := special[*p.m.Cfg.BOSTokenID]; ok {
				root.vars["bos_token"] = s.text(text, true)
			}
		}
		if text, ok := special[p.m.Cfg.EOSTokenID]; ok {
			root.vars["eos_token"] = s.text(text, true)
		}
		return root
	})
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return nil, err
		}
		return nil, fmt.Errorf("%w: %v", ErrNativeChatTemplate, err)
	}
	return parts, nil
}

// A request-local stream observer receives the exact prepared renderer identity.
// It carries no authority and is never inherited by another request.
type nativeChatTemplatePreparedKey struct{}
