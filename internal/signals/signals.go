package signals

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"strings"
)

// Signal is one plain-English behavioral question to judge over an agent's turns.
type Signal struct {
	// Name is the signal's stable identifier (also the sampling salt). Unique per config.
	Name string `json:"name"`
	// Prompt is the behavioral question in natural language, e.g.
	// "Did the agent abandon the task before running any test?".
	Prompt string `json:"prompt"`
	// Schema is the JSON Schema (a draft-07 subset — see validateSchemaDoc) the judge's
	// verdict must satisfy, so a behavioral answer is a joinable record, not prose.
	Schema json.RawMessage `json:"schema"`
	// SampleRate in [0,1] is the fraction of items to judge. Behavioral judging costs a
	// model call; 1.0 judges every item, 0.0 judges none.
	SampleRate float64 `json:"sample_rate"`
}

// Config is a set of behavioral signals, typically loaded from a signals.json.
type Config struct {
	Signals []Signal `json:"signals"`
}

// Item is one unit a signal is judged over — a turn, a step, or a whole trajectory.
type Item struct {
	ID   string            `json:"id"`             // stable id; the sampling key
	Text string            `json:"text"`           // the content the judge reads
	Meta map[string]string `json:"meta,omitempty"` // optional labels (tool, verdict, ...)
}

// Result is one (signal, item) outcome. A non-sampled item yields Sampled=false and no
// verdict. A sampled item yields either a schema-valid Verdict or an Err.
type Result struct {
	Signal  string          `json:"signal"`
	ItemID  string          `json:"item_id"`
	Sampled bool            `json:"sampled"`
	Verdict json.RawMessage `json:"verdict,omitempty"`
	Err     string          `json:"error,omitempty"`
}

// Evaluator answers a signal's behavioral Prompt for one item, returning a JSON verdict
// that Run then validates against the signal's Schema. Production wraps a model call;
// tests inject a deterministic fake.
type Evaluator interface {
	Evaluate(sig Signal, item Item) (json.RawMessage, error)
}

// Validate checks a signal is well-formed: named, has a behavioral prompt, a sane rate,
// and a schema that is itself a valid (subset) JSON Schema object.
func (s Signal) Validate() error {
	if strings.TrimSpace(s.Name) == "" {
		return fmt.Errorf("signal: name is required")
	}
	if strings.TrimSpace(s.Prompt) == "" {
		return fmt.Errorf("signal %q: prompt (the behavioral question) is required", s.Name)
	}
	if s.SampleRate < 0 || s.SampleRate > 1 {
		return fmt.Errorf("signal %q: sample_rate %g out of range [0,1]", s.Name, s.SampleRate)
	}
	if len(s.Schema) == 0 {
		return fmt.Errorf("signal %q: schema is required (the verdict shape)", s.Name)
	}
	if err := validateSchemaDoc(s.Schema); err != nil {
		return fmt.Errorf("signal %q: %w", s.Name, err)
	}
	return nil
}

// Validate checks the whole config: every signal valid, names unique.
func (c Config) Validate() error {
	seen := map[string]bool{}
	for _, s := range c.Signals {
		if err := s.Validate(); err != nil {
			return err
		}
		if seen[s.Name] {
			return fmt.Errorf("duplicate signal name %q", s.Name)
		}
		seen[s.Name] = true
	}
	return nil
}

// sampleResolution is the fixed-point denominator for deterministic sampling. A rate is
// admitted if hash(name,id) mod resolution < rate*resolution.
const sampleResolution = 1_000_000

// Sampled reports whether itemID is in this signal's sampled set. Deterministic: it
// hashes (Name, itemID) rather than drawing a random number, so the decision is stable
// across runs/machines and a test can assert it exactly. Rate 0 => never, 1 => always.
func (s Signal) Sampled(itemID string) bool {
	if s.SampleRate <= 0 {
		return false
	}
	if s.SampleRate >= 1 {
		return true
	}
	h := fnv.New64a()
	h.Write([]byte(s.Name))
	h.Write([]byte{0})
	h.Write([]byte(itemID))
	bucket := h.Sum64() % sampleResolution
	return bucket < uint64(s.SampleRate*float64(sampleResolution))
}

// RenderPrompt builds the exact judge instruction for (sig, item): the behavioral
// question, the required verdict schema, and the item text. A production Evaluator sends
// this; exposing it makes the judge input inspectable (fak signals plan) without a model.
func RenderPrompt(sig Signal, item Item) string {
	var b strings.Builder
	b.WriteString("You are judging one unit of an agent trajectory for a specific BEHAVIOR.\n\n")
	b.WriteString("Behavioral question:\n")
	b.WriteString(sig.Prompt)
	b.WriteString("\n\nAnswer ONLY with a JSON object matching this schema:\n")
	b.WriteString(string(compactJSON(sig.Schema)))
	if len(item.Meta) > 0 {
		b.WriteString("\n\nItem labels: ")
		b.WriteString(string(mustJSON(item.Meta)))
	}
	b.WriteString("\n\nItem under judgement:\n")
	b.WriteString(item.Text)
	b.WriteString("\n")
	return b.String()
}

// Run evaluates cfg's signals over items: for each signal, it judges exactly the sampled
// items via ev and validates each verdict against the signal's schema. Non-sampled items
// still produce a Result (Sampled=false) so the ledger records what was and wasn't judged.
// Results are ordered by signal (config order) then item (input order) for determinism.
func Run(cfg Config, items []Item, ev Evaluator) ([]Result, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var out []Result
	for _, sig := range cfg.Signals {
		for _, it := range items {
			r := Result{Signal: sig.Name, ItemID: it.ID}
			if !sig.Sampled(it.ID) {
				out = append(out, r)
				continue
			}
			r.Sampled = true
			verdict, err := ev.Evaluate(sig, it)
			if err != nil {
				r.Err = err.Error()
				out = append(out, r)
				continue
			}
			if err := ValidateAgainstSchema(sig.Schema, verdict); err != nil {
				r.Err = fmt.Sprintf("verdict off-schema: %v", err)
				out = append(out, r)
				continue
			}
			r.Verdict = verdict
			out = append(out, r)
		}
	}
	return out, nil
}

// SampledCount reports how many of items a signal would judge — the cost preview.
func (s Signal) SampledCount(items []Item) int {
	n := 0
	for _, it := range items {
		if s.Sampled(it.ID) {
			n++
		}
	}
	return n
}

func compactJSON(raw json.RawMessage) []byte {
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return raw
	}
	return mustJSON(v)
}

func mustJSON(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		return []byte("{}")
	}
	return b
}
