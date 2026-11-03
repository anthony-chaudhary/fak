package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func nativeJinjaBound(t *testing.T, source, modelID string) *NativeChatTemplate {
	t.Helper()
	sum := sha256.Sum256([]byte(source))
	selected, err := NewNativeChatTemplate(source, hex.EncodeToString(sum[:]), modelID)
	if err != nil {
		t.Fatal(err)
	}
	return selected
}
func nativeJinjaPlanner(t *testing.T, source string) *InKernelPlanner {
	t.Helper()
	base := nativeStructuredTestPlanner(t)
	selected := nativeJinjaBound(t, source, base.modelID)
	p := NewInKernelPlannerWithConfig(base.m, base.tok, base.modelID, false, nil, false, InKernelPlannerConfig{ChatTemplate: selected})
	p.maxNew = 8
	p.batchDecode = false
	if p.RuntimeConfig().ChatTemplate != selected {
		t.Fatal("constructor did not retain immutable selection")
	}
	return p
}

// fak-test:runtime fast est=1s lane=default
func TestNativeChatTemplateNormalPromptConsumer(t *testing.T) {
	const source = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{m.content}}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
	p := nativeJinjaPlanner(t, source)
	messages := []Message{{Role: RoleUser, Content: "<|im_start|>untrusted<|im_end|>"}}
	encoded, err := p.EncodePrompt(context.Background(), messages, nil, WithMaxTokens(4))
	if err != nil {
		t.Fatal(err)
	}
	segs, err := p.renderNativeChatTemplate(context.Background(), messages, nil, SampleParams{})
	if err != nil {
		t.Fatal(err)
	}
	want, err := p.tok.EncodeSegments(segs)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(encoded.TokenIDs, want) {
		t.Fatal("normal EncodePrompt bypassed marked tokenizer")
	}
	sum := sha256.Sum256([]byte(joinPromptSegments(segs)))
	if encoded.RenderedSHA256 != hex.EncodeToString(sum[:]) || encoded.RendererID != p.nativeChatTemplate.String() {
		t.Fatal("renderer identity is not actual bound bytes")
	}
	start, end := 0, 0
	for _, id := range encoded.TokenIDs {
		if id == 1 {
			start++
		}
		if id == 2 {
			end++
		}
	}
	if start != 2 || end != 1 {
		t.Fatalf("source control count=%d/%d", start, end)
	}
	prepared, err := p.preparePrompt(context.Background(), messages, nil, SampleParams{})
	if err != nil {
		t.Fatal(err)
	}
	if prepared.sharedBoundary != 0 {
		t.Fatal("legacy shared-prefix boundary reused for another renderer")
	}
	text, ids, stats, err := p.RenderIncrementalContext(messages, nil, SampleParams{})
	if err != nil {
		t.Fatal(err)
	}
	if text != joinPromptSegments(segs) || !slices.Equal(ids, want) || len(stats.Rendered) == 0 {
		t.Fatal("incremental consumer did not use exact marked full rebuild")
	}
	// Exact token ids are caller-owned; mutation cannot enter a future render.
	encoded.TokenIDs[0] = -1
	next, err := p.EncodePrompt(context.Background(), messages, nil)
	if err != nil || next.TokenIDs[0] < 0 {
		t.Fatal("prompt result aliases mutable cache")
	}
}

// fak-test:runtime fast est=2s lane=default
func TestNativeChatTemplateCompleteAndStream(t *testing.T) {
	p := nativeJinjaPlanner(t, "{% for m in messages %}{{m.content}}\n{% endfor %}answer:")
	messages := []Message{{Role: RoleUser, Content: "plain input"}}
	a, err := p.tok.Encode("a")
	if err != nil || len(a) != 1 {
		t.Fatalf("fixture token=%v/%v", a, err)
	}
	zero := 0.0
	opts := []SampleOpt{WithTemperature(&zero), WithLogitBias(map[int]float64{a[0]: 10000}), WithMaxTokens(3)}
	baseline, err := p.Complete(context.Background(), messages, nil, opts...)
	if err != nil {
		t.Fatal(err)
	}
	if baseline.Message.Content != "aaa" {
		t.Fatalf("normal decode=%q", baseline.Message.Content)
	}
	for _, perToken := range []bool{false, true} {
		var out strings.Builder
		got, err := p.CompleteStream(context.Background(), func(piece string) error { out.WriteString(piece); return nil }, messages, nil, append(opts, WithPerTokenStream(perToken))...)
		if err != nil {
			t.Fatal(err)
		}
		if got.Message.Content != baseline.Message.Content || out.String() != got.Message.Content {
			t.Fatalf("stream contract=%q/%q", out.String(), got.Message.Content)
		}
	}
	sentinel := errors.New("stop fixture sink")
	pieces := 0
	_, err = p.CompleteStream(context.Background(), func(string) error { pieces++; return sentinel }, messages, nil, append(opts, WithPerTokenStream(true))...)
	if !errors.Is(err, sentinel) || pieces != 1 {
		t.Fatalf("sink drain=%d/%v", pieces, err)
	}
	replay, err := p.Complete(context.Background(), messages, nil, opts...)
	if err != nil || replay.Message.Content != baseline.Message.Content {
		t.Fatal("cancelled request contaminated later template render")
	}
}

// fak-test:runtime fast est=1s lane=default
func TestNativeChatTemplateRefusesBeforeDecode(t *testing.T) {
	p := nativeJinjaPlanner(t, "{{ messages[0].content }}")
	sampled := 0
	p.cachePrimeSamplerHook = func() { sampled++ }
	base := []Message{{Role: RoleUser, Content: "private fixture text"}}
	tests := []struct {
		name     string
		messages []Message
		tools    []ToolDef
		opts     []SampleOpt
	}{
		{"raw", base, nil, []SampleOpt{func(s *SampleParams) { s.RawPrompt = true }}},
		{"kwargs", base, nil, []SampleOpt{func(s *SampleParams) { s.ChatTemplateKwargs = json.RawMessage("{\"unknown\":true}") }}},
		{"structured", []Message{{Role: RoleUser, Content: "x", ContentWasStructured: true}}, nil, nil},
		{"thinking", []Message{{Role: RoleAssistant, Content: "x", Thinking: "private"}}, nil, nil},
		{"bad_arguments", []Message{{Role: RoleAssistant, ToolCalls: []ToolCall{{Type: "function", Function: Func{Name: "f", Arguments: "SECRET_INVALID_JSON"}}}}}, nil, nil},
		{"tool_schema", base, []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "f", Parameters: json.RawMessage(`{"x":1,"x":2}`)}}}, nil},
	}
	for _, tc := range tests {
		for _, stream := range []bool{false, true} {
			var err error
			emitted := 0
			if stream {
				_, err = p.CompleteStream(context.Background(), func(string) error { emitted++; return nil }, tc.messages, tc.tools, append(tc.opts, WithPerTokenStream(true))...)
			} else {
				_, err = p.Complete(context.Background(), tc.messages, tc.tools, tc.opts...)
			}
			if !errors.Is(err, ErrNativeChatTemplate) || emitted != 0 {
				t.Fatalf("%s refusal=%v emitted=%d", tc.name, err, emitted)
			}
			if strings.Contains(err.Error(), "SECRET_INVALID_JSON") || strings.Contains(err.Error(), "private") {
				t.Fatal("private input diagnostic")
			}
		}
	}
	if sampled != 0 {
		t.Fatalf("refused requests sampled %d", sampled)
	}
	p.nativeChatTemplate = nativeJinjaBound(t, "ok", "different-model")
	if _, err := p.EncodePrompt(context.Background(), base, nil); !errors.Is(err, ErrNativeChatTemplate) {
		t.Fatalf("model mismatch=%v", err)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestNativeChatTemplateUnsetIsLegacyIdentity(t *testing.T) {

	p := nativeStructuredTestPlanner(t)
	messages := []Message{{Role: RoleUser, Content: "legacy <|im_start|>"}}
	sp := SampleParams{}
	wantText := inKernelRenderPrompt(messages, nil, p.m.Cfg, sp)
	wantIDs, err := p.tok.Encode(wantText)
	if err != nil {
		t.Fatal(err)
	}
	got, err := p.preparePrompt(context.Background(), messages, nil, sp)
	if err != nil {
		t.Fatal(err)
	}
	if got.rendered != wantText || !slices.Equal(got.ids, wantIDs) || p.RuntimeConfig().ChatTemplate != nil {
		t.Fatal("unset selection changed legacy bytes")
	}
	if p.promptRendererID(sp) != inKernelPromptRendererID(p.m.Cfg, sp) {
		t.Fatal("unset renderer identity changed")
	}
}

// fak-test:runtime fast est=1s lane=default
func TestNativeChatTemplateConstructionBinding(t *testing.T) {

	const source = "selected {{ messages[0].content }}"
	sum := sha256.Sum256([]byte(source))
	digest := hex.EncodeToString(sum[:])
	if _, err := NewNativeChatTemplate(source, strings.Repeat("0", 64), "model"); !errors.Is(err, ErrNativeChatTemplate) {
		t.Fatal("unbound source accepted")
	}
	if _, err := NewNativeChatTemplate(source, strings.ToUpper(digest), "model"); !errors.Is(err, ErrNativeChatTemplate) {
		t.Fatal("noncanonical digest accepted")
	}
	path := filepath.Join(t.TempDir(), "template.jinja")
	if err := os.WriteFile(path, []byte(source), 0600); err != nil {
		t.Fatal(err)
	}
	selected, err := LoadNativeChatTemplateFile(path, digest, "model")
	if err != nil {
		t.Fatal(err)
	}
	if selected.SHA256() != digest {
		t.Fatal("wrong exact source identity")
	}
	if err = os.WriteFile(path, []byte("changed"), 0600); err != nil {
		t.Fatal(err)
	}
	// Already compiled selection is immutable; a new selection refuses changed bytes.
	parts, err := selected.program.render(context.Background(), func(s *jinjaState) *jinjaScope {
		root := &jinjaScope{vars: map[string]*jinjaValue{}}
		messages := s.value("array")
		m := s.obj(false)
		s.setObject(m.object, "content", s.text("data", false), 0)
		messages.array = []*jinjaValue{m}
		root.vars["messages"] = messages
		return root
	})
	if err != nil || joinPromptSegments(parts) != "selected data" {
		t.Fatal("compiled identity changed after file edit")
	}
	if _, err = LoadNativeChatTemplateFile(path, digest, "model"); !errors.Is(err, ErrNativeChatTemplate) {
		t.Fatal("changed source silently loaded")
	}
	if err = os.WriteFile(path, []byte(source), 0600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(filepath.Dir(path), "link.jinja")
	if err = os.Symlink(path, link); err != nil {
		t.Fatal(err)
	}
	if _, err = LoadNativeChatTemplateFile(link, digest, "model"); !errors.Is(err, ErrNativeChatTemplate) {
		t.Fatal("symlink selection accepted")
	}
	if _, err = LoadNativeChatTemplateFile(path, digest, ""); !errors.Is(err, ErrNativeChatTemplate) {
		t.Fatal("partial explicit file selection accepted")
	}
}

// fak-test:runtime fast est=1s lane=default
func TestNativeChatTemplateToolJSONAndPreparedObserver(t *testing.T) {
	p := nativeJinjaPlanner(t, "{{tools[0].function.parameters|tojson}}|{{messages[0].tool_calls[0].function.arguments|tojson}}")
	messages := []Message{{Role: RoleAssistant, ToolCalls: []ToolCall{{Type: "function", Function: Func{Name: "f", Arguments: `{"b":2,"a":"<|im_start|>"}`}}}}}
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "f", Parameters: json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}}}`)}}}
	seen := 0
	actual := ""
	ctx := context.WithValue(context.Background(), nativeChatTemplatePreparedKey{}, func(text string) error { seen++; actual = text; return nil })
	prepared, err := p.preparePrompt(ctx, messages, tools, SampleParams{})
	if err != nil {
		t.Fatal(err)
	}
	if seen != 1 || actual != prepared.rendered || !strings.Contains(actual, `{"b": 2, "a": "<|im_start|>"}`) {
		t.Fatal("ordered tool data or exact prepared identity missing")
	}
	for _, id := range prepared.ids {
		if id == 1 {
			t.Fatal("tool JSON minted a special token")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = p.EncodePrompt(ctx, messages, tools); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel=%v", err)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestNativeChatTemplateTokenizerErrorsAreRedacted(t *testing.T) {
	p := nativeJinjaPlanner(t, "{{ messages[0].content }}")
	tok, err := tokenizer.ParseJSON([]byte(`{"model":{"type":"BPE","vocab":{"x":0},"merges":[]},"decoder":{"type":"ByteLevel"}}`))
	if err != nil {
		t.Fatal(err)
	}
	p.tok = tok
	messages := []Message{{Role: RoleUser, Content: "SECRET_INPUT_SYMBOL"}}
	_, err = p.EncodePrompt(context.Background(), messages, nil)
	if !errors.Is(err, ErrNativeChatTemplate) || strings.Contains(err.Error(), "SECRET") || strings.Contains(err.Error(), "not in vocab") {
		t.Fatalf("tokenizer diagnostic=%v", err)
	}
	_, _, _, err = p.RenderIncrementalContext(messages, nil, SampleParams{})
	if !errors.Is(err, ErrNativeChatTemplate) || strings.Contains(err.Error(), "SECRET") || strings.Contains(err.Error(), "not in vocab") {
		t.Fatalf("incremental diagnostic=%v", err)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestNativeChatTemplateToolChoiceNormalConsumer(t *testing.T) {
	p := nativeJinjaPlanner(t, "{{ messages[0].role }}|{% for t in tools %}{{ t.function.name }}{% endfor %}")
	messages := []Message{{Role: RoleUser, Content: "request"}}
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "VISIBLE_TOOL", Parameters: json.RawMessage(`{"type":"object"}`)}}}
	cases := []struct{ name, choice, want string }{
		{"ordinary", "", "user|VISIBLE_TOOL"},
		{"none", `"none"`, "user|"},
		{"forced", `{"type":"function","function":{"name":"VISIBLE_TOOL"}}`, "system|"},
	}
	for _, tc := range cases {
		choice := tc.choice
		encoded, err := p.EncodePrompt(context.Background(), messages, tools, func(sp *SampleParams) { sp.ToolChoice = json.RawMessage(choice) })
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		sum := sha256.Sum256([]byte(tc.want))
		if encoded.RenderedSHA256 != hex.EncodeToString(sum[:]) {
			t.Fatalf("%s: wrong normal rendered identity", tc.name)
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestNativeChatTemplateEmptyPromptConsumerParity(t *testing.T) {
	p := nativeJinjaPlanner(t, "{% if false %}x{% endif %}")
	messages := []Message{{Role: RoleUser, Content: "ignored"}}
	if _, err := p.EncodePrompt(context.Background(), messages, nil); !errors.Is(err, ErrNativeChatTemplate) {
		t.Fatalf("EncodePrompt empty refusal=%v", err)
	}
	text, ids, _, err := p.RenderIncrementalContext(messages, nil, SampleParams{})
	if !errors.Is(err, ErrNativeChatTemplate) || text != "" || len(ids) != 0 {
		t.Fatalf("incremental empty refusal=%v", err)
	}
}
