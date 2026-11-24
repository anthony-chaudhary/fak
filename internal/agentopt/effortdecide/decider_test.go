package effortdecide

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/agentopt"
)

func fixed(scores map[string]float64) Reader {
	return ReaderFunc(func(context.Context, string, []string) (map[string]float64, error) { return scores, nil })
}

func mustNew(t *testing.T, cfg Config) *Decider {
	t.Helper()
	d, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func routineTurn() agentopt.TurnContext {
	return agentopt.TurnContext{LastRole: "tool", ToolName: "read_file", ToolOutput: "package main"}
}

func TestDecideModelOverridesHeuristic(t *testing.T) {
	d := mustNew(t, Config{Reader: fixed(map[string]float64{"A": -4, "B": -3, "C": -2, "D": -0.05})})
	got := d.Decide(context.Background(), routineTurn())
	if got.Source != SourceModel || got.Effort != agentopt.EffortHigh {
		t.Fatalf("got source=%s effort=%s reason=%s", got.Source, got.Effort, got.FallbackReason)
	}
	if got.AllocatedBudget != agentopt.EffortHigh.ThinkingBudget() || got.ThinkingBudget != got.AllocatedBudget {
		t.Fatalf("budget = %d/%d", got.ThinkingBudget, got.AllocatedBudget)
	}
	var sum float64
	for _, p := range got.Probabilities {
		sum += p
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Fatalf("probabilities sum %v", sum)
	}
}

func TestDecideFallsBack(t *testing.T) {
	cases := map[string]struct {
		reader Reader
		want   string
	}{
		"read error":    {ReaderFunc(func(context.Context, string, []string) (map[string]float64, error) { return nil, errors.New("boom") }), "read_error"},
		"one option":    {fixed(map[string]float64{"A": -0.1}), "insufficient_options"},
		"non-finite":    {fixed(map[string]float64{"A": math.NaN(), "B": math.Inf(1)}), "insufficient_options"},
		"low confident": {fixed(map[string]float64{"A": -1.4, "B": -1.4, "C": -1.4, "D": -1.3}), "low_confidence"},
		"undeclared":    {fixed(map[string]float64{"E": 0, "Z": 0}), "insufficient_options"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			d := mustNew(t, Config{Reader: tc.reader})
			turn := routineTurn()
			got := d.Decide(context.Background(), turn)
			heur := agentopt.NewIntraModelEffortRouter().Classify(turn)
			if got.Source != SourceHeuristic || got.Effort != heur.Effort || !strings.HasPrefix(got.FallbackReason, tc.want) {
				t.Fatalf("got source=%s effort=%s reason=%q; heuristic=%s", got.Source, got.Effort, got.FallbackReason, heur.Effort)
			}
		})
	}
}

func TestDecideTimeout(t *testing.T) {
	slow := ReaderFunc(func(ctx context.Context, _ string, _ []string) (map[string]float64, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	})
	d := mustNew(t, Config{Reader: slow, Timeout: 20 * time.Millisecond})
	start := time.Now()
	got := d.Decide(context.Background(), routineTurn())
	if got.Source != SourceHeuristic || got.FallbackReason != "timeout" {
		t.Fatalf("got %+v", got)
	}
	if time.Since(start) > time.Second {
		t.Fatal("timeout not enforced")
	}
}

func TestErrorRecoveryNotLowered(t *testing.T) {
	turn := agentopt.TurnContext{LastRole: "tool", ToolName: "bash", ToolOutput: "--- FAIL: TestX", TestFailure: true, HasError: true}
	heur := agentopt.NewIntraModelEffortRouter().Classify(turn)
	if heur.Category != agentopt.CategoryErrorRecovery {
		t.Skipf("heuristic category %s; fixture no longer error recovery", heur.Category)
	}
	none := fixed(map[string]float64{"A": -0.01, "B": -6, "C": -6, "D": -6})
	if got := mustNew(t, Config{Reader: none}).Decide(context.Background(), turn); got.Effort != heur.Effort {
		t.Fatalf("guarded effort = %s, want heuristic %s", got.Effort, heur.Effort)
	}
	if got := mustNew(t, Config{Reader: none, AllowBelowHeuristicOnError: true}).Decide(context.Background(), turn); got.Effort != agentopt.EffortNone {
		t.Fatalf("unguarded effort = %s, want none", got.Effort)
	}
}

func TestBuildPromptDeclaresEveryOptionAndTruncates(t *testing.T) {
	p := BuildPrompt(agentopt.TurnContext{ToolName: "bash", ToolOutput: strings.Repeat("x", 5000), CompilerError: true})
	for _, l := range Labels() {
		if !strings.Contains(p, "\n"+l+") ") {
			t.Fatalf("prompt missing option %s", l)
		}
	}
	if !strings.Contains(p, "[truncated]") || !strings.Contains(p, "compiler_error") || len(p) > 3000 {
		t.Fatalf("prompt len=%d", len(p))
	}
}

func TestOpenAIReader(t *testing.T) {
	var gotReq map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer k" {
			http.Error(w, "bad", http.StatusBadRequest)
			return
		}
		_ = json.NewDecoder(r.Body).Decode(&gotReq)
		_, _ = w.Write([]byte(`{"choices":[{"logprobs":{"content":[{"top_logprobs":[
			{"token":"C","logprob":-0.2},{"token":" C","logprob":-2.0},
			{"token":"B","logprob":-2.5},{"token":"hello","logprob":-3}]}]}}]}`))
	}))
	defer srv.Close()

	r := &OpenAIReader{BaseURL: srv.URL + "/", Model: "m", APIKey: "k"}
	scores, err := r.ReadOptions(context.Background(), "q", Labels())
	if err != nil {
		t.Fatal(err)
	}
	if gotReq["logprobs"] != true || gotReq["max_tokens"] != float64(1) || gotReq["model"] != "m" {
		t.Fatalf("request %+v", gotReq)
	}
	want := math.Log(math.Exp(-0.2) + math.Exp(-2.0))
	if math.Abs(scores["C"]-want) > 1e-9 || scores["B"] != -2.5 || len(scores) != 2 {
		t.Fatalf("scores %v", scores)
	}

	d := mustNew(t, Config{Reader: r})
	if got := d.Decide(context.Background(), routineTurn()); got.Source != SourceModel || got.Effort != agentopt.EffortMedium {
		t.Fatalf("end to end: %+v", got)
	}
}

func TestOpenAIReaderHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { http.Error(w, "x", http.StatusBadGateway) }))
	defer srv.Close()
	if _, err := (&OpenAIReader{BaseURL: srv.URL, Model: "m"}).ReadOptions(context.Background(), "q", Labels()); err == nil {
		t.Fatal("want error on 502")
	}
}

func TestFromEnv(t *testing.T) {
	env := func(kv map[string]string) func(string) string { return func(k string) string { return kv[k] } }
	if d, err := FromEnv(env(nil)); d != nil || err != nil {
		t.Fatalf("default off: %v %v", d, err)
	}
	if _, err := FromEnv(env(map[string]string{EnvDecider: "decision-model"})); err == nil {
		t.Fatal("want error without URL/model")
	}
	if _, err := FromEnv(env(map[string]string{EnvDecider: "bogus"})); err == nil {
		t.Fatal("want error on unknown mode")
	}
	d, err := FromEnv(env(map[string]string{EnvDecider: "jev", EnvURL: "http://x", EnvModel: "m", EnvMinConfidence: "0.7", EnvTimeout: "300ms"}))
	if err != nil || d == nil || d.cfg.MinConfidence != 0.7 || d.cfg.Timeout != 300*time.Millisecond {
		t.Fatalf("configured: %+v %v", d, err)
	}
	if _, err := FromEnv(env(map[string]string{EnvDecider: "on", EnvURL: "http://x", EnvModel: "m", EnvMinConfidence: "2"})); err == nil {
		t.Fatal("want error on confidence > 1")
	}
}

func TestLowDeclaredMassFallsBack(t *testing.T) {
	// "<think>" holds nearly all mass; stray letters would renormalize to p(A)~0.85.
	d := mustNew(t, Config{Reader: fixed(map[string]float64{"A": -11.2, "D": -12.9})})
	got := d.Decide(context.Background(), routineTurn())
	if got.Source != SourceHeuristic || !strings.HasPrefix(got.FallbackReason, "low_declared_mass") || got.DeclaredMass > 1e-4 {
		t.Fatalf("got %+v", got)
	}
}

func TestReaderIgnoringContextIsBounded(t *testing.T) {
	block := make(chan struct{})
	defer close(block)
	stuck := ReaderFunc(func(context.Context, string, []string) (map[string]float64, error) {
		<-block
		return nil, nil
	})
	start := time.Now()
	got := mustNew(t, Config{Reader: stuck, Timeout: 20 * time.Millisecond}).Decide(context.Background(), routineTurn())
	if got.FallbackReason != "timeout" || time.Since(start) > time.Second {
		t.Fatalf("got %+v after %s", got, time.Since(start))
	}
}

func TestParentCancelIsNotTimeout(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	slow := ReaderFunc(func(ctx context.Context, _ string, _ []string) (map[string]float64, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	})
	if got := mustNew(t, Config{Reader: slow}).Decide(ctx, routineTurn()); got.FallbackReason != "canceled" {
		t.Fatalf("got %q", got.FallbackReason)
	}
}

func TestTruncationKeepsUTF8Valid(t *testing.T) {
	p := BuildPrompt(agentopt.TurnContext{ToolOutput: "x" + strings.Repeat("界", 1000)})
	if !utf8.ValidString(p) {
		t.Fatal("prompt is not valid UTF-8")
	}
}

func TestNormalizeLabel(t *testing.T) {
	for in, want := range map[string]string{"A": "A", " B": "B", "C)": "C", "**D": "D", "a.": "A", "<think>": "<THINK>", "The": "THE"} {
		if got := normalizeLabel(in); got != want {
			t.Errorf("normalizeLabel(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestOpenAIReaderBaseURLAndKnobs(t *testing.T) {
	var body map[string]any
	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = w.Write([]byte(`{"choices":[{"logprobs":{"content":[{"top_logprobs":[{"token":"A)","logprob":-0.1},{"token":"b","logprob":-2.5}]}]}}]}`))
	}))
	defer srv.Close()
	r := &OpenAIReader{BaseURL: srv.URL + "/v1/", Model: "m", MaxTokensField: "max_completion_tokens", DisableThinking: true}
	scores, err := r.ReadOptions(context.Background(), "q", Labels())
	if err != nil {
		t.Fatal(err)
	}
	if path != "/v1/chat/completions" {
		t.Fatalf("path %q", path)
	}
	if _, ok := body["max_tokens"]; ok || body["max_completion_tokens"] != float64(1) {
		t.Fatalf("cap field: %v", body)
	}
	if kw, _ := body["chat_template_kwargs"].(map[string]any); kw["enable_thinking"] != false {
		t.Fatalf("thinking kwargs: %v", body)
	}
	if scores["A"] != -0.1 || scores["B"] != -2.5 {
		t.Fatalf("scores %v", scores)
	}
}

func TestOpenAIReaderErrorCarriesBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, `{"error":"Unsupported parameter: 'max_tokens'"}`, http.StatusBadRequest)
	}))
	defer srv.Close()
	_, err := (&OpenAIReader{BaseURL: srv.URL, Model: "m"}).ReadOptions(context.Background(), "q", Labels())
	if err == nil || !strings.Contains(err.Error(), "max_tokens") {
		t.Fatalf("err %v", err)
	}
}

func TestFromEnvNewKnobs(t *testing.T) {
	env := func(kv map[string]string) func(string) string { return func(k string) string { return kv[k] } }
	base := map[string]string{EnvDecider: "on", EnvURL: "http://x/v1", EnvModel: "m"}
	with := func(k, v string) map[string]string {
		m := map[string]string{}
		for a, b := range base {
			m[a] = b
		}
		m[k] = v
		return m
	}
	d, err := FromEnv(env(with(EnvNoThinking, "true")))
	if err != nil || !d.cfg.Reader.(*OpenAIReader).DisableThinking {
		t.Fatalf("disable thinking: %v", err)
	}
	for k, v := range map[string]string{EnvMaxTokens: "max_output", EnvMinMass: "0", EnvNoThinking: "maybe", EnvURL: "localhost:8000"} {
		if _, err := FromEnv(env(with(k, v))); err == nil {
			t.Errorf("%s=%s: want error", k, v)
		}
	}
	if d, err := FromEnv(env(with(EnvMinMass, "0.3"))); err != nil || d.cfg.MinDeclaredMass != 0.3 {
		t.Fatalf("min mass: %v", err)
	}
}
