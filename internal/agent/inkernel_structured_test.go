package agent

import (
	"context"
	"encoding/json"
	"errors"
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
	}
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
