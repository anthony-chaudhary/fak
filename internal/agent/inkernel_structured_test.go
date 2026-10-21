package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

var nativeStructuredTestFormat = json.RawMessage(`{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object","properties":{"answer":{"type":"string","enum":["a","b"]}},"required":["answer"],"additionalProperties":false}}}`)

func nativeStructuredTestPlanner(t *testing.T) *InKernelPlanner {
	t.Helper()
	cfg := tinyConcurrencyConfig()
	cfg.EOSTokenID = 0
	m := model.NewSynthetic(cfg)
	m.Quantize()
	p := NewInKernelPlanner(m, loadProbeTok(t), "synthetic-structured-response", false, nil, false)
	p.maxNew = 48
	p.batchDecode = false
	return p
}

type nativeStructuredFixedCoin struct{}

func (nativeStructuredFixedCoin) Seed(int64)   {}
func (nativeStructuredFixedCoin) Int63() int64 { return 1<<62 | 1<<61 }

// fak-test:runtime fast est=20ms lane=default
func TestNativeStructuredPreservesSamplerControls(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	encode := func(text string) []int {
		ids, err := p.tok.Encode(text)
		if err != nil {
			t.Fatal(err)
		}
		return ids
	}
	a, b, invalid := encode("a")[0], encode("b")[0], encode("x")[0]
	for _, tc := range []struct {
		name              string
		temperature, topP float64
		topK, want        int
	}{
		{"stochastic", 1, 0, 0, b},
		{"greedy", 0, 0, 0, a},
		{"top_k", 1, 0, 1, a},
		{"top_p", 1, 0.5, 0, a},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stops := StopIDs(p.tok, p.m.Cfg)
			c, err := p.newNativeStructuredConstraint(nativeStructuredTestFormat, stops)
			if err != nil {
				t.Fatal(err)
			}
			c.history = encode(`{"answer":"`)
			logits := make([]float32, p.m.Cfg.VocabSize)
			logits[a], logits[b], logits[invalid] = 2, 1, 1000
			got := -1
			lane := &decodeLane{structured: c, logits: logits, temp: tc.temperature, topP: tc.topP, topK: tc.topK, rng: rand.New(nativeStructuredFixedCoin{}), maxNew: 1, stops: stops, emit: func(token int) bool { got = token; return false }, greedyReady: true, greedyNext: invalid}
			if lane.deviceGreedy() {
				t.Fatal("structured lane admitted device argmax")
			}
			lane.decodeOne(context.Background())
			if lane.err != nil {
				t.Fatal(lane.err)
			}
			if got != tc.want {
				t.Fatalf("sampled token %d, want %d", got, tc.want)
			}
			if logits[invalid] != 1000 {
				t.Fatal("sampler mutated source logits")
			}
		})
	}
}

func nativeStructuredCheckAnswer(t *testing.T, content string) {
	t.Helper()
	var value map[string]string
	if err := json.Unmarshal([]byte(content), &value); err != nil {
		t.Fatalf("native response is not JSON: %q: %v", content, err)
	}
	if len(value) != 1 || (value["answer"] != "a" && value["answer"] != "b") {
		t.Fatalf("native response violates requested schema: %q", content)
	}
}

// fak-test:runtime fast est=150ms lane=default
func TestNativeStructuredCompleteAndStream(t *testing.T) {
	for _, temperature := range []float64{0, 0.8} {
		t.Run(string(rune('0'+int(temperature*10))), func(t *testing.T) {
			p := nativeStructuredTestPlanner(t)
			topK, topP := 2, 0.8
			opts := []SampleOpt{WithResponseFormat(nativeStructuredTestFormat), WithTemperature(&temperature), WithTopK(&topK), WithTopP(&topP)}
			messages := []Message{{Role: RoleUser, Content: "Answer with the requested object."}}
			for range 2 {
				completion, err := p.Complete(context.Background(), messages, nil, opts...)
				if err != nil {
					t.Fatal(err)
				}
				nativeStructuredCheckAnswer(t, completion.Message.Content)
			}
			var streamed strings.Builder
			completion, err := p.CompleteStream(context.Background(), func(delta string) error {
				streamed.WriteString(delta)
				return nil
			}, messages, nil, append(opts, WithPerTokenStream(true))...)
			if err != nil {
				t.Fatal(err)
			}
			nativeStructuredCheckAnswer(t, completion.Message.Content)
			if streamed.String() != completion.Message.Content {
				t.Fatalf("streamed %q, completion %q", streamed.String(), completion.Message.Content)
			}
		})
	}
}

// fak-test:runtime fast est=600ms lane=default
func TestNativeStructuredOptionalScalarCompleteAndStream(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	encodeOne := func(text string) int {
		t.Helper()
		ids, err := p.tok.Encode(text)
		if err != nil || len(ids) != 1 {
			t.Fatalf("encode %q: ids=%v err=%v", text, ids, err)
		}
		return ids[0]
	}
	comma, closeObject := encodeOne(","), encodeOne("}")
	seed := int64(-17)
	temperature := 0.8
	messages := []Message{{Role: RoleUser, Content: "Return the requested object."}}

	tests := []struct {
		name       string
		definition string
		check      func(*testing.T, any)
	}{
		{"string", `{"type":"string","enum":["optional"]}`, func(t *testing.T, value any) {
			if value != "optional" {
				t.Fatalf("optional string = %#v", value)
			}
		}},
		{"boolean", `{"type":"boolean"}`, func(t *testing.T, value any) {
			if _, ok := value.(bool); !ok {
				t.Fatalf("optional boolean has type %T", value)
			}
		}},
		{"integer", `{"type":"integer"}`, func(t *testing.T, value any) {
			n, ok := value.(json.Number)
			if !ok {
				t.Fatalf("optional integer has type %T", value)
			}
			if _, err := n.Int64(); err != nil {
				t.Fatalf("optional integer %q: %v", n, err)
			}
		}},
		{"number", `{"type":"number"}`, func(t *testing.T, value any) {
			n, ok := value.(json.Number)
			if !ok {
				t.Fatalf("optional number has type %T", value)
			}
			if _, err := n.Float64(); err != nil {
				t.Fatalf("optional number %q: %v", n, err)
			}
		}},
	}

	check := func(t *testing.T, content string, included bool, checkOptional func(*testing.T, any)) {
		t.Helper()
		decoder := json.NewDecoder(strings.NewReader(content))
		decoder.UseNumber()
		var value map[string]any
		if err := decoder.Decode(&value); err != nil {
			t.Fatalf("decode %q: %v", content, err)
		}
		if value["answer"] != "kept" {
			t.Fatalf("required answer missing from %q", content)
		}
		optional, present := value["extra"]
		if present != included {
			t.Fatalf("optional presence in %q = %v, want %v", content, present, included)
		}
		wantProperties := 1
		if included {
			wantProperties++
		}
		if len(value) != wantProperties {
			t.Fatalf("unexpected properties in %q", content)
		}
		if included {
			checkOptional(t, optional)
		}
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			format := json.RawMessage(`{"type":"json_schema","json_schema":{"strict":true,"schema":{"type":"object","properties":{"answer":{"type":"string","const":"kept"},"extra":` + tc.definition + `},"required":["answer"],"additionalProperties":false}}}`)
			for _, branch := range []struct {
				name     string
				bias     map[int]float64
				included bool
			}{
				{"omit", map[int]float64{closeObject: 100}, false},
				{"include", map[int]float64{comma: 100, closeObject: 99}, true},
			} {
				t.Run(branch.name, func(t *testing.T) {
					opts := []SampleOpt{
						WithResponseFormat(format),
						WithSeed(&seed),
						WithTemperature(&temperature),
						WithLogitBias(branch.bias),
						WithMaxTokens(96),
					}
					buffered, err := p.Complete(context.Background(), messages, nil, opts...)
					if err != nil {
						t.Fatal(err)
					}
					if buffered.FinishReason != "stop" {
						t.Fatalf("buffered finish reason = %q, want stop", buffered.FinishReason)
					}
					check(t, buffered.Message.Content, branch.included, tc.check)
					if tc.name == "string" {
						want := `{"answer":"kept"}`
						if branch.included {
							want = `{"answer":"kept","extra":"optional"}`
						}
						if buffered.Message.Content != want {
							t.Fatalf("forced %s output = %q, want %q", branch.name, buffered.Message.Content, want)
						}
					}
					replayed, err := p.Complete(context.Background(), messages, nil, opts...)
					if err != nil {
						t.Fatal(err)
					}
					if replayed.Message.Content != buffered.Message.Content {
						t.Fatalf("same-seed replay = %q, want %q", replayed.Message.Content, buffered.Message.Content)
					}
					if replayed.FinishReason != "stop" {
						t.Fatalf("replay finish reason = %q, want stop", replayed.FinishReason)
					}

					var streamed strings.Builder
					stream, err := p.CompleteStream(context.Background(), func(piece string) error {
						streamed.WriteString(piece)
						return nil
					}, messages, nil, append(opts, WithPerTokenStream(true))...)
					if err != nil {
						t.Fatal(err)
					}
					if stream.FinishReason != "stop" {
						t.Fatalf("stream finish reason = %q, want stop", stream.FinishReason)
					}
					if stream.Message.Content != buffered.Message.Content || streamed.String() != stream.Message.Content {
						t.Fatalf("seeded buffered/stream mismatch: buffered=%q streamed=%q completion=%q", buffered.Message.Content, streamed.String(), stream.Message.Content)
					}
					check(t, stream.Message.Content, branch.included, tc.check)
				})
			}
		})
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestNativeStructuredIntegerEnumCompleteAndStream(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	encodeOne := func(text string) int {
		t.Helper()
		ids, err := p.tok.Encode(text)
		if err != nil || len(ids) != 1 {
			t.Fatalf("encode %q: ids=%v err=%v", text, ids, err)
		}
		return ids[0]
	}
	format := json.RawMessage(`{"type":"json_schema","json_schema":{"strict":true,"schema":{"type":"object","properties":{"color":{"type":"string","enum":["blue","red"]},"count":{"type":"integer","enum":[1,10,100]}},"required":["color"],"additionalProperties":false}}}`)
	seed := int64(37)
	temperature := 0.8
	opts := []SampleOpt{
		WithResponseFormat(format),
		WithSeed(&seed),
		WithTemperature(&temperature),
		WithLogitBias(map[int]float64{encodeOne(","): 100, encodeOne("0"): 100}),
		WithMaxTokens(96),
	}
	messages := []Message{{Role: RoleUser, Content: "Choose a color and count."}}
	check := func(t *testing.T, completion *Completion) {
		t.Helper()
		if completion.FinishReason != "stop" {
			t.Fatalf("finish reason = %q, want stop", completion.FinishReason)
		}
		decoder := json.NewDecoder(strings.NewReader(completion.Message.Content))
		decoder.UseNumber()
		var value map[string]any
		if err := decoder.Decode(&value); err != nil {
			t.Fatalf("decode %q: %v", completion.Message.Content, err)
		}
		color, ok := value["color"].(string)
		if !ok || (color != "blue" && color != "red") {
			t.Fatalf("color = %#v, want declared string member", value["color"])
		}
		count, ok := value["count"].(json.Number)
		if !ok {
			t.Fatalf("count has type %T, want json.Number", value["count"])
		}
		integer, err := count.Int64()
		if err != nil {
			t.Fatalf("count %q is not an integer: %v", count, err)
		}
		if integer != 100 {
			t.Fatalf("count = %d, want forced prefix-related member 100", integer)
		}
		if len(value) != 2 {
			t.Fatalf("response properties = %#v, want color and count", value)
		}
	}

	buffered, err := p.Complete(context.Background(), messages, nil, opts...)
	if err != nil {
		t.Fatal(err)
	}
	check(t, buffered)
	replayed, err := p.Complete(context.Background(), messages, nil, opts...)
	if err != nil {
		t.Fatal(err)
	}
	check(t, replayed)
	if replayed.Message.Content != buffered.Message.Content {
		t.Fatalf("same-seed replay = %q, want %q", replayed.Message.Content, buffered.Message.Content)
	}

	var streamed strings.Builder
	stream, err := p.CompleteStream(context.Background(), func(piece string) error {
		streamed.WriteString(piece)
		return nil
	}, messages, nil, append(opts, WithPerTokenStream(true))...)
	if err != nil {
		t.Fatal(err)
	}
	check(t, stream)
	if stream.Message.Content != buffered.Message.Content || streamed.String() != stream.Message.Content {
		t.Fatalf("seeded buffered/stream mismatch: buffered=%q streamed=%q completion=%q", buffered.Message.Content, streamed.String(), stream.Message.Content)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestNativeStructuredTextFormatPreservesOrdinaryOutput(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	messages := []Message{{Role: RoleUser, Content: "plain response"}}
	plain, err := p.Complete(context.Background(), messages, nil, WithMaxTokens(3))
	if err != nil {
		t.Fatal(err)
	}
	text, err := p.Complete(context.Background(), messages, nil, WithMaxTokens(3), WithResponseFormat(json.RawMessage(`{"type":"text"}`)))
	if err != nil {
		t.Fatal(err)
	}
	if plain.Message.Content != text.Message.Content || plain.Usage.CompletionTokens != text.Usage.CompletionTokens {
		t.Fatalf("explicit text changed ordinary output: %q/%d versus %q/%d", plain.Message.Content, plain.Usage.CompletionTokens, text.Message.Content, text.Usage.CompletionTokens)
	}
}

// fak-test:runtime fast est=30ms lane=default
func TestNativeStructuredUnsupportedRequestRefusesBeforeSampling(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	sampled := 0
	p.cachePrimeSamplerHook = func() { sampled++ }
	cases := []json.RawMessage{
		json.RawMessage(`{"type":"json_object"}`),
		json.RawMessage(`{"type":"regex","schema":".*"}`),
		json.RawMessage(`{"type":"json_schema","json_schema":{"schema":{"type":"array"}}}`),
		json.RawMessage(`{"type":"json_schema","json_schema":{"strict":false,"schema":{}}}`),
		json.RawMessage(`{"type":"json_schema","json_schema":{}}`),
		json.RawMessage(`{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"answer":{"type":"string"},"nested":{"type":"object","properties":{}}},"required":["answer"],"additionalProperties":false}}}`),
		json.RawMessage(`{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"answer":{"type":"string"},"extra":{"type":"string","minLength":1}},"required":["answer"],"additionalProperties":false}}}`),
	}
	integerSetFormat := func(constraint string) json.RawMessage {
		return json.RawMessage(`{"type":"json_schema","json_schema":{"strict":true,"schema":{"type":"object","properties":{"count":{"type":"integer",` + constraint + `}},"required":["count"],"additionalProperties":false}}}`)
	}
	cases = append(cases,
		integerSetFormat(`"enum":[]`),
		integerSetFormat(`"enum":[1,"2"]`),
		integerSetFormat(`"enum":[null]`),
		integerSetFormat(`"enum":[1.5]`),
		integerSetFormat(`"enum":[1e2]`),
		integerSetFormat(`"enum":[9223372036854775808]`),
		integerSetFormat(`"const":null`),
	)
	tooManyIntegerMembers := make([]string, 33)
	for i := range tooManyIntegerMembers {
		tooManyIntegerMembers[i] = fmt.Sprintf("%d", i)
	}
	cases = append(cases, integerSetFormat(`"enum":[`+strings.Join(tooManyIntegerMembers, ",")+`]`))
	properties := make(map[string]any, 17)
	required := make([]string, 0, 17)
	for i := range 17 {
		name := fmt.Sprintf("p%02d", i)
		properties[name] = map[string]any{"type": "boolean"}
		required = append(required, name)
	}
	cases = append(cases, nativeStructuredFormatForSchema(t, map[string]any{
		"type":                 "object",
		"properties":           properties,
		"required":             required,
		"additionalProperties": false,
	}))
	cases = append(cases, nativeStructuredFormatForSchema(t, map[string]any{
		"type": "object",
		"properties": map[string]any{
			"answer": map[string]any{"type": "string"},
			"o1":     map[string]any{"type": "boolean"},
			"o2":     map[string]any{"type": "boolean"},
			"o3":     map[string]any{"type": "boolean"},
			"o4":     map[string]any{"type": "boolean"},
			"o5":     map[string]any{"type": "boolean"},
		},
		"required":             []string{"answer"},
		"additionalProperties": false,
	}))
	cases = append(cases, nativeStructuredFormatForSchema(t, map[string]any{
		"type":                 "object",
		"description":          strings.Repeat("x", 65536),
		"properties":           map[string]any{"answer": map[string]any{"type": "string"}},
		"required":             []string{"answer"},
		"additionalProperties": false,
	}))
	for _, format := range cases {
		_, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "refuse"}}, nil, WithResponseFormat(format))
		if !errors.Is(err, ErrNativeStructuredUnsupported) {
			t.Fatalf("format %s: got %v", format, err)
		}
	}
	if sampled != 0 {
		t.Fatalf("unsupported request sampled %d times", sampled)
	}
}

func nativeStructuredFormatForSchema(t *testing.T, schema any) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(map[string]any{
		"type": "json_schema",
		"json_schema": map[string]any{
			"strict": true,
			"schema": schema,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// fak-test:runtime fast est=20ms lane=default
func TestNativeStructuredMaskDeadEndAndStopSafety(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	stops := StopIDs(p.tok, p.m.Cfg)
	c, err := p.newNativeStructuredConstraint(nativeStructuredTestFormat, stops)
	if err != nil {
		t.Fatal(err)
	}
	logits := make([]float32, p.m.Cfg.VocabSize)
	for i := range logits {
		logits[i] = float32(i % 7)
	}
	original := slices.Clone(logits)
	effective, err := c.maskLogits(logits, stops)
	if err != nil {
		t.Fatal(err)
	}
	for id, enabled := range stops {
		if enabled && id >= 0 && id < len(effective) && !math.IsInf(float64(effective[id]), -1) {
			t.Fatalf("stop token %d admitted before complete object", id)
		}
	}
	if !slices.Equal(logits, original) {
		t.Fatal("schema mask mutated source logits")
	}
	invalid, err := p.tok.Encode("invalid")
	if err != nil {
		t.Fatal(err)
	}
	c.history = invalid
	if _, err := c.maskLogits(logits, stops); !errors.Is(err, ErrNativeStructuredNoToken) {
		t.Fatalf("dead end: got %v", err)
	}
	if err := c.accept(-1, effective); !errors.Is(err, ErrNativeStructuredNoToken) {
		t.Fatalf("invalid selected token: got %v", err)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestNativeStructuredUnsupportedModesRefuse(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	messages := []Message{{Role: RoleUser, Content: "structured"}}
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "lookup", Parameters: json.RawMessage(`{"type":"object","properties":{}}`)}}}
	_, err := p.Complete(context.Background(), messages, tools, WithResponseFormat(nativeStructuredTestFormat))
	if !errors.Is(err, ErrNativeStructuredUnsupported) {
		t.Fatalf("tools+schema: %v", err)
	}
	_, err = p.Complete(context.Background(), messages, nil, WithResponseFormat(nativeStructuredTestFormat), WithNativeInferenceReceipt(true))
	var unsupported *model.NativeInferenceReceiptUnsupportedError
	if !errors.As(err, &unsupported) || !errors.Is(err, ErrNativeStructuredUnsupported) {
		t.Fatalf("unmodified receipt+schema: %v", err)
	}
	p.dflash2 = &DFlash2Target{}
	_, err = p.Complete(context.Background(), messages, nil, WithResponseFormat(nativeStructuredTestFormat))
	if !errors.Is(err, ErrNativeStructuredUnsupported) {
		t.Fatalf("dflash2+schema: %v", err)
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestNativeStructuredPreservesJSONShapedLikeToolsAndReasoning(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	format := json.RawMessage(`{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"name":{"type":"string","const":"lookup"},"arguments":{"type":"string","const":"<think>value</think>"}},"required":["name","arguments"],"additionalProperties":false}}}`)
	completion, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "Return the required JSON."}}, nil, WithResponseFormat(format), WithMaxTokens(96))
	if err != nil {
		t.Fatal(err)
	}
	const want = `{"arguments":"<think>value</think>","name":"lookup"}`
	if completion.Message.Content != want || len(completion.Message.ToolCalls) != 0 || completion.Message.Thinking != "" {
		t.Fatalf("schema response was transformed: content=%q tools=%v thinking=%q", completion.Message.Content, completion.Message.ToolCalls, completion.Message.Thinking)
	}
	var streamed strings.Builder
	completion, err = p.CompleteStream(context.Background(), func(piece string) error {
		streamed.WriteString(piece)
		return nil
	}, []Message{{Role: RoleUser, Content: "Return the required JSON."}}, nil, WithResponseFormat(format), WithMaxTokens(96), WithPerTokenStream(true))
	if err != nil {
		t.Fatal(err)
	}
	if streamed.String() != want || completion.Message.Content != want {
		t.Fatalf("raw schema stream changed JSON: %q / %q", streamed.String(), completion.Message.Content)
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestNativeStructuredStreamCancellationDoesNotLeakRequestState(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	interrupted := errors.New("stop structured stream")
	pieces := 0
	messages := []Message{{Role: RoleUser, Content: "Return the required object."}}
	_, err := p.CompleteStream(context.Background(), func(string) error {
		pieces++
		return interrupted
	}, messages, nil, WithResponseFormat(nativeStructuredTestFormat), WithPerTokenStream(true))
	if !errors.Is(err, interrupted) || pieces != 1 {
		t.Fatalf("sink cancellation: pieces=%d err=%v", pieces, err)
	}
	completion, err := p.Complete(context.Background(), messages, nil, WithResponseFormat(nativeStructuredTestFormat))
	if err != nil {
		t.Fatal(err)
	}
	nativeStructuredCheckAnswer(t, completion.Message.Content)
}

// fak-test:runtime fast est=20ms lane=default
func TestNativeStructuredNonfiniteAllowedLogitsRefuse(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	stops := StopIDs(p.tok, p.m.Cfg)
	open, err := p.tok.Encode("{")
	if err != nil || len(open) != 1 {
		t.Fatalf("encode brace: %v %v", open, err)
	}
	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1))} {
		c, err := p.newNativeStructuredConstraint(nativeStructuredTestFormat, stops)
		if err != nil {
			t.Fatal(err)
		}
		logits := make([]float32, p.m.Cfg.VocabSize)
		logits[open[0]] = value
		if _, err := c.maskLogits(logits, stops); !errors.Is(err, ErrNativeStructuredNoToken) {
			t.Fatalf("nonfinite allowed score %v: %v", value, err)
		}
	}
}

func nativeStructuredReasoningPlanner(t *testing.T, seeded bool) *InKernelPlanner {
	t.Helper()
	tok := atomicThinkingTokenizer(t)
	cfg := tinyConcurrencyConfig()
	if seeded {
		cfg = model.Config{
			ModelType: "qwen3_5_text", HiddenSize: 32, NumLayers: 4, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
			IntermediateSize: 64, RMSNormEps: 1e-5, RopeTheta: 10000, TieWordEmbeddings: true,
			LayerTypes:          []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
			LinearConvKernelDim: 3, LinearKeyHeadDim: 8, LinearNumKeyHeads: 2, LinearValueHeadDim: 8, LinearNumValueHeads: 4,
			AttnOutputGate: true, FullAttentionInterval: 4, NormGain1p: true,
		}
	}
	cfg.VocabSize = tok.Vocab()
	cfg.EOSTokenID = 0
	m := model.NewSynthetic(cfg)
	m.Quantize()
	p := NewInKernelPlanner(m, tok, "synthetic-structured-reasoning", false, nil, false)
	p.maxNew = 48
	p.batchDecode = false
	return p
}

func nativeStructuredReasoningIDs(t *testing.T, p *InKernelPlanner) (open, close, body, eos int) {
	t.Helper()
	one := func(text string) int {
		ids, err := p.tok.Encode(text)
		if err != nil || len(ids) != 1 {
			t.Fatalf("encode %q: ids=%v err=%v", text, ids, err)
		}
		return ids[0]
	}
	return one(thinkOpen), one(thinkClose), one("x"), 0
}

// fak-test:runtime fast est=200ms lane=default
func TestNativeStructuredReasoningBudgetThenSchema(t *testing.T) {
	p := nativeStructuredReasoningPlanner(t, false)
	open, close, body, eos := nativeStructuredReasoningIDs(t, p)
	zero := 0.0
	opts := []SampleOpt{
		WithResponseFormat(nativeStructuredTestFormat), WithThinkingBudget(2), WithTemperature(&zero),
		WithLogitBias(map[int]float64{body: 100, open: 60, close: -100, eos: 200}),
		WithMaxTokens(48), WithDecodeTrace(true), WithNativeDecodeTokenIDs(true),
	}
	messages := []Message{{Role: RoleUser, Content: "Think briefly, then answer."}}
	check := func(t *testing.T, completion *Completion) {
		t.Helper()
		if completion.FinishReason != "stop" {
			t.Fatalf("finish reason = %q, want stop", completion.FinishReason)
		}
		nativeStructuredCheckAnswer(t, completion.Message.Content)
		if completion.Message.ReasoningContent != "xx" {
			t.Fatalf("reasoning = %q, want the two budgeted reasoning tokens", completion.Message.ReasoningContent)
		}
		ids := completion.NativeDecodeTokenIDs
		if ids == nil || len(ids.TokenIDs) < 5 || !slices.Equal(ids.TokenIDs[:4], []int{open, body, body, close}) {
			t.Fatalf("token IDs = %v, want opener, two reasoning tokens, then the forced close", ids)
		}
		if slices.Contains(ids.TokenIDs, eos) {
			t.Fatalf("EOS escaped the reasoning span or the schema: %v", ids.TokenIDs)
		}
	}
	buffered, err := p.Complete(context.Background(), messages, nil, opts...)
	if err != nil {
		t.Fatal(err)
	}
	check(t, buffered)

	var streamed strings.Builder
	stream, err := p.CompleteStream(context.Background(), func(piece string) error {
		streamed.WriteString(piece)
		return nil
	}, messages, nil, append(opts, WithPerTokenStream(true))...)
	if err != nil {
		t.Fatal(err)
	}
	check(t, stream)
	if streamed.String() != stream.Message.Content || stream.Message.Content != buffered.Message.Content {
		t.Fatalf("reasoning leaked into the schema stream: streamed=%q content=%q buffered=%q", streamed.String(), stream.Message.Content, buffered.Message.Content)
	}
}

// fak-test:runtime fast est=150ms lane=default
func TestNativeStructuredReasoningBudgetMaySkipReasoning(t *testing.T) {
	p := nativeStructuredReasoningPlanner(t, false)
	open, _, body, _ := nativeStructuredReasoningIDs(t, p)
	zero := 0.0
	completion, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "Answer."}}, nil,
		WithResponseFormat(nativeStructuredTestFormat), WithThinkingBudget(4), WithTemperature(&zero),
		WithLogitBias(map[int]float64{body: 100, open: -100}), WithMaxTokens(48))
	if err != nil {
		t.Fatal(err)
	}
	nativeStructuredCheckAnswer(t, completion.Message.Content)
	if completion.Message.ReasoningContent != "" {
		t.Fatalf("reasoning = %q, want none when the model answers directly", completion.Message.ReasoningContent)
	}
}

// fak-test:runtime fast est=150ms lane=default
func TestNativeStructuredBudgetIgnoresThinkTextInsideJSON(t *testing.T) {
	p := nativeStructuredReasoningPlanner(t, false)
	open, _, _, _ := nativeStructuredReasoningIDs(t, p)
	format := json.RawMessage(`{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"note":{"type":"string","const":"<think>abcdef</think>"}},"required":["note"],"additionalProperties":false}}}`)
	completion, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "Answer."}}, nil,
		WithResponseFormat(format), WithThinkingBudget(1), WithLogitBias(map[int]float64{open: -100}), WithMaxTokens(64))
	if err != nil {
		t.Fatal(err)
	}
	if want := `{"note":"<think>abcdef</think>"}`; completion.Message.Content != want || completion.Message.ReasoningContent != "" {
		t.Fatalf("schema text was treated as reasoning: content=%q reasoning=%q", completion.Message.Content, completion.Message.ReasoningContent)
	}
}

// fak-test:runtime fast est=30ms lane=default
func TestNativeStructuredBudgetCountsSpelledCloseAsReasoning(t *testing.T) {
	p := nativeStructuredReasoningPlanner(t, false)
	stops := StopIDs(p.tok, p.m.Cfg)
	structured, err := p.newNativeStructuredConstraint(nativeStructuredTestFormat, stops)
	if err != nil {
		t.Fatal(err)
	}
	if err := structured.armReasoning(p, true, stops); err != nil {
		t.Fatal(err)
	}
	const limit = 12
	thinking, err := p.newNativeThinkingConstraint(limit, true, stops)
	if err != nil {
		t.Fatal(err)
	}
	thinking.spanByToken = true
	_, closeID, body, _ := nativeStructuredReasoningIDs(t, p)
	var reasoning []int
	for _, b := range []byte(thinkClose) {
		ids, err := p.tok.Encode(string(b))
		if err != nil || len(ids) != 1 || ids[0] == closeID {
			t.Fatalf("encode %q: %v %v", b, ids, err)
		}
		reasoning = append(reasoning, ids[0])
	}
	for len(reasoning) < limit {
		reasoning = append(reasoning, body)
	}
	oneHot := func(id int) []float32 {
		logits := make([]float32, p.m.Cfg.VocabSize)
		logits[id] = 100
		return logits
	}
	ln := &decodeLane{structured: structured, thinking: thinking, maxNew: 64, stops: stops, rng: rand.New(rand.NewSource(1))}
	for i, id := range reasoning {
		ln.logits = oneHot(id)
		if next, advance := ln.decodeOne(context.Background()); !advance || next != id || ln.err != nil {
			t.Fatalf("reasoning token %d: next=%d advance=%v err=%v", i, next, advance, ln.err)
		}
	}
	ln.logits = oneHot(body)
	if next, advance := ln.decodeOne(context.Background()); !advance || next != closeID {
		t.Fatalf("after %d reasoning tokens including a spelled close, next=%d advance=%v err=%v; want the forced atomic close", limit, next, advance, ln.err)
	}
	if structured.reasoningActive() {
		t.Fatal("forced close did not start the schema response")
	}
}

// fak-test:runtime fast est=150ms lane=default
func TestNativeStructuredStopStringIgnoresReasoning(t *testing.T) {
	p := nativeStructuredReasoningPlanner(t, true)
	_, close, body, eos := nativeStructuredReasoningIDs(t, p)
	zero := 0.0
	completion, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "Think, then answer."}}, nil,
		WithResponseFormat(nativeStructuredTestFormat), WithThinkingBudget(3), WithTemperature(&zero), WithStop([]string{"xx"}),
		WithLogitBias(map[int]float64{body: 100, close: -100, eos: 200}), WithMaxTokens(48))
	if err != nil {
		t.Fatal(err)
	}
	nativeStructuredCheckAnswer(t, completion.Message.Content)
	if completion.FinishReason != "stop" || completion.Message.ReasoningContent != "xxx" {
		t.Fatalf("stop string matched hidden reasoning: finish=%q reasoning=%q content=%q", completion.FinishReason, completion.Message.ReasoningContent, completion.Message.Content)
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestNativeStructuredSeededReasoningClosesOnTriggerToken(t *testing.T) {
	zero := 0.0
	t.Run("budget forces the trigger", func(t *testing.T) {
		p := nativeStructuredReasoningPlanner(t, true)
		_, close, body, eos := nativeStructuredReasoningIDs(t, p)
		completion, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "Think, then answer."}}, nil,
			WithResponseFormat(nativeStructuredTestFormat), WithThinkingBudget(3), WithTemperature(&zero),
			WithLogitBias(map[int]float64{body: 100, close: -100, eos: 200}), WithMaxTokens(48),
			WithDecodeTrace(true), WithNativeDecodeTokenIDs(true))
		if err != nil {
			t.Fatal(err)
		}
		nativeStructuredCheckAnswer(t, completion.Message.Content)
		if completion.Message.ReasoningContent != "xxx" {
			t.Fatalf("reasoning = %q, want three budgeted tokens", completion.Message.ReasoningContent)
		}
		if ids := completion.NativeDecodeTokenIDs; ids == nil || len(ids.TokenIDs) < 4 || !slices.Equal(ids.TokenIDs[:4], []int{body, body, body, close}) {
			t.Fatalf("token IDs = %v, want three reasoning tokens then the close trigger", ids)
		}
	})
	t.Run("model closes without a budget", func(t *testing.T) {
		t.Setenv("FAK_INKERNEL_ENABLE_THINKING", "1")
		p := nativeStructuredReasoningPlanner(t, true)
		_, close, _, _ := nativeStructuredReasoningIDs(t, p)
		completion, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "Answer."}}, nil,
			WithResponseFormat(nativeStructuredTestFormat), WithTemperature(&zero),
			WithLogitBias(map[int]float64{close: 100}), WithMaxTokens(48))
		if err != nil {
			t.Fatal(err)
		}
		nativeStructuredCheckAnswer(t, completion.Message.Content)
		if completion.Message.ReasoningContent != "" {
			t.Fatalf("reasoning = %q, want an empty span", completion.Message.ReasoningContent)
		}
	})
	t.Run("unclosed span is reasoning, never content", func(t *testing.T) {
		t.Setenv("FAK_INKERNEL_ENABLE_THINKING", "1")
		p := nativeStructuredReasoningPlanner(t, true)
		_, close, body, eos := nativeStructuredReasoningIDs(t, p)
		completion, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "Answer."}}, nil,
			WithResponseFormat(nativeStructuredTestFormat), WithTemperature(&zero),
			WithLogitBias(map[int]float64{body: 100, close: -100, eos: 200}), WithMaxTokens(4))
		if err != nil {
			t.Fatal(err)
		}
		if completion.FinishReason != "length" || completion.Message.Content != "" || completion.Message.ReasoningContent != "xxxx" {
			t.Fatalf("unclosed reasoning: finish=%q content=%q reasoning=%q", completion.FinishReason, completion.Message.Content, completion.Message.ReasoningContent)
		}
	})
}
