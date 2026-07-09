package signals

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"
)

var boolSchema = json.RawMessage(`{"type":"object","required":["holds","evidence"],"properties":{"holds":{"type":"boolean"},"evidence":{"type":"string"}}}`)

func TestSignal_Validate(t *testing.T) {
	cases := []struct {
		name    string
		sig     Signal
		wantErr string
	}{
		{"ok", Signal{Name: "gaveup", Prompt: "did it give up?", Schema: boolSchema, SampleRate: 0.5}, ""},
		{"no name", Signal{Prompt: "q", Schema: boolSchema}, "name is required"},
		{"no prompt", Signal{Name: "x", Schema: boolSchema}, "prompt"},
		{"rate high", Signal{Name: "x", Prompt: "q", Schema: boolSchema, SampleRate: 1.5}, "out of range"},
		{"rate low", Signal{Name: "x", Prompt: "q", Schema: boolSchema, SampleRate: -0.1}, "out of range"},
		{"no schema", Signal{Name: "x", Prompt: "q"}, "schema is required"},
		{"bad schema type", Signal{Name: "x", Prompt: "q", Schema: json.RawMessage(`{"type":"widget"}`)}, "unsupported type"},
		{"required not in props", Signal{Name: "x", Prompt: "q", Schema: json.RawMessage(`{"type":"object","required":["z"],"properties":{}}`)}, "not among properties"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := c.sig.Validate()
			if c.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), c.wantErr) {
				t.Fatalf("got %v, want error containing %q", err, c.wantErr)
			}
		})
	}
}

func TestConfig_Validate_DuplicateNames(t *testing.T) {
	cfg := Config{Signals: []Signal{
		{Name: "dup", Prompt: "q", Schema: boolSchema, SampleRate: 1},
		{Name: "dup", Prompt: "q2", Schema: boolSchema, SampleRate: 1},
	}}
	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("expected duplicate-name error, got %v", err)
	}
}

func TestSampled_DeterministicAndBounded(t *testing.T) {
	sig := Signal{Name: "excessive_apology", SampleRate: 0.3}
	// Determinism: same id => same decision, every call.
	for i := 0; i < 100; i++ {
		id := fmt.Sprintf("turn-%d", i)
		if sig.Sampled(id) != sig.Sampled(id) {
			t.Fatalf("Sampled not deterministic for %q", id)
		}
	}
	// Rate 0 and 1 are hard bounds.
	none := Signal{Name: "s", SampleRate: 0}
	all := Signal{Name: "s", SampleRate: 1}
	for i := 0; i < 50; i++ {
		id := fmt.Sprintf("x-%d", i)
		if none.Sampled(id) {
			t.Fatalf("rate 0 admitted %q", id)
		}
		if !all.Sampled(id) {
			t.Fatalf("rate 1 rejected %q", id)
		}
	}
	// Admitted fraction converges near the rate over many ids.
	n, hit := 20000, 0
	for i := 0; i < n; i++ {
		if sig.Sampled(fmt.Sprintf("item-%d", i)) {
			hit++
		}
	}
	frac := float64(hit) / float64(n)
	if math.Abs(frac-0.3) > 0.02 {
		t.Fatalf("admitted fraction %.3f not near rate 0.30", frac)
	}
}

func TestSalt_DifferentSignalsSampleDifferently(t *testing.T) {
	// The name is the sampling salt: two signals at the same rate must not admit the
	// identical id set, or a whole corpus would be judged by one draw.
	a := Signal{Name: "signal-a", SampleRate: 0.5}
	b := Signal{Name: "signal-b", SampleRate: 0.5}
	diff := 0
	for i := 0; i < 1000; i++ {
		id := fmt.Sprintf("t-%d", i)
		if a.Sampled(id) != b.Sampled(id) {
			diff++
		}
	}
	if diff < 300 {
		t.Fatalf("signals with different names barely diverge (%d/1000) — salt not applied", diff)
	}
}

// fakeEvaluator returns a canned verdict, or a schema-violating one for a flagged item.
type fakeEvaluator struct {
	offSchemaID string
}

func (f fakeEvaluator) Evaluate(sig Signal, item Item) (json.RawMessage, error) {
	if item.ID == f.offSchemaID {
		return json.RawMessage(`{"holds":"yes"}`), nil // holds should be boolean, missing evidence
	}
	return json.RawMessage(`{"holds":true,"evidence":"step 4 abandoned before test"}`), nil
}

func TestRun_SamplesJudgesAndValidates(t *testing.T) {
	cfg := Config{Signals: []Signal{{Name: "gaveup", Prompt: "did it give up before testing?", Schema: boolSchema, SampleRate: 1}}}
	items := []Item{{ID: "a", Text: "..."}, {ID: "b", Text: "..."}, {ID: "bad", Text: "..."}}
	results, err := Run(cfg, items, fakeEvaluator{offSchemaID: "bad"})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 3 {
		t.Fatalf("got %d results, want 3", len(results))
	}
	byID := map[string]Result{}
	for _, r := range results {
		byID[r.ItemID] = r
	}
	if !byID["a"].Sampled || byID["a"].Err != "" || len(byID["a"].Verdict) == 0 {
		t.Fatalf("item a should be judged with a valid verdict: %+v", byID["a"])
	}
	if byID["bad"].Err == "" || !strings.Contains(byID["bad"].Err, "off-schema") {
		t.Fatalf("item bad should be rejected off-schema, got %+v", byID["bad"])
	}
}

func TestRun_NonSampledItemsRecordedNotJudged(t *testing.T) {
	// Rate 0 => every item recorded as not-sampled, evaluator never consulted.
	cfg := Config{Signals: []Signal{{Name: "s", Prompt: "q", Schema: boolSchema, SampleRate: 0}}}
	items := []Item{{ID: "a"}, {ID: "b"}}
	results, err := Run(cfg, items, panicEvaluator{t})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range results {
		if r.Sampled {
			t.Fatalf("rate 0 sampled %q", r.ItemID)
		}
	}
}

type panicEvaluator struct{ t *testing.T }

func (p panicEvaluator) Evaluate(Signal, Item) (json.RawMessage, error) {
	p.t.Fatal("evaluator called for a non-sampled item")
	return nil, nil
}

func TestRenderPrompt_CarriesQuestionSchemaAndItem(t *testing.T) {
	sig := Signal{Name: "s", Prompt: "Did the agent widen scope?", Schema: boolSchema, SampleRate: 1}
	item := Item{ID: "t1", Text: "AGENT: also refactored the logger", Meta: map[string]string{"tool": "edit"}}
	p := RenderPrompt(sig, item)
	for _, want := range []string{"Did the agent widen scope?", "holds", "also refactored the logger", "tool"} {
		if !strings.Contains(p, want) {
			t.Errorf("rendered prompt missing %q:\n%s", want, p)
		}
	}
}

func TestValidateAgainstSchema_Types(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","required":["n","ok","tags"],"properties":{"n":{"type":"integer"},"ok":{"type":"boolean"},"tags":{"type":"array","items":{"type":"string"}},"grade":{"type":"string","enum":["A","B"]}}}`)
	good := json.RawMessage(`{"n":3,"ok":true,"tags":["x","y"],"grade":"A"}`)
	if err := ValidateAgainstSchema(schema, good); err != nil {
		t.Fatalf("good verdict rejected: %v", err)
	}
	bads := map[string]json.RawMessage{
		"missing required": json.RawMessage(`{"n":3,"ok":true}`),
		"wrong type":       json.RawMessage(`{"n":3,"ok":"yes","tags":[]}`),
		"fractional int":   json.RawMessage(`{"n":3.5,"ok":true,"tags":[]}`),
		"bad enum":         json.RawMessage(`{"n":3,"ok":true,"tags":[],"grade":"C"}`),
		"array elem type":  json.RawMessage(`{"n":3,"ok":true,"tags":[1,2]}`),
	}
	for name, v := range bads {
		if err := ValidateAgainstSchema(schema, v); err == nil {
			t.Errorf("%s: expected schema violation, got none", name)
		}
	}
}
