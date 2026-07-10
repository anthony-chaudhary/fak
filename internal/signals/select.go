package signals

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// select.go adds ID-grounded LLM triage: pick relevant candidates BY id, then keep only
// the ids that round-trip against the candidate store. It is the grounding discipline for
// "ask a model which of these events matter" — the failure mode being that a model asked
// to reproduce event content will confidently invent a reference that was never offered.
//
// SelectByID closes that hole. Each candidate is tagged with an [Event ID: <id>] marker,
// the model is forced to answer {summary, event_ids[]} (validated by the same signals
// schema validator), and every returned id is looked up in the store: an id that exists is
// a real selection, an id the model conjured is DROPPED (and surfaced in Dropped for audit,
// never silently swallowed). The model's freedom is reduced to SELECTING from a closed set.

// selectByIDSchema is the forced tool-output shape for a triage: a free-text summary plus
// the ids of the selected events. Reusing the signals schema profile means the same
// ValidateAgainstSchema gate that guards behavioral verdicts guards a selection.
var selectByIDSchema = json.RawMessage(`{"type":"object","required":["summary","event_ids"],"properties":{"summary":{"type":"string"},"event_ids":{"type":"array","items":{"type":"string"}}}}`)

// Selector answers a SelectByID triage: given the rendered prompt (the request, the
// {summary, event_ids[]} instruction, and the candidates tagged with [Event ID: <id>]),
// it returns the raw JSON the model produced. Production wraps a model/tool call; tests
// inject a deterministic fake. The context carries cancellation/deadline to the call.
type Selector interface {
	Select(ctx context.Context, prompt string) (json.RawMessage, error)
}

// Selection is the grounded result of a SelectByID triage.
type Selection struct {
	// Summary is the model's free-text rationale (advisory — not grounded).
	Summary string `json:"summary"`
	// Selected are the candidates the model picked, in the order it returned them,
	// deduplicated, with every fabricated id removed. Each element is a real candidate.
	Selected []Item `json:"selected"`
	// Dropped are the ids the model returned that did NOT round-trip against the store —
	// i.e. fabrications. Surfaced (not swallowed) so a caller can see the model hallucinated.
	Dropped []string `json:"dropped,omitempty"`
}

// RenderSelectPrompt builds the exact triage instruction: the request, the forced output
// schema, and every candidate tagged with its [Event ID: <id>] marker. Exposing it makes
// the model input inspectable without a model call, and it is what SelectByID sends.
func RenderSelectPrompt(candidates []Item, prompt string) string {
	var b strings.Builder
	b.WriteString("You are triaging a set of candidate events. Read the request and select ONLY the events relevant to it.\n\n")
	b.WriteString("Request:\n")
	b.WriteString(prompt)
	b.WriteString("\n\nSelect events by copying the exact id from each relevant [Event ID: <id>] tag below. ")
	b.WriteString("Every id you return MUST be copied verbatim from a tag; do NOT invent ids.\n\n")
	b.WriteString("Answer ONLY with a JSON object matching this schema:\n")
	b.WriteString(string(compactJSON(selectByIDSchema)))
	b.WriteString("\n\nCandidate events:\n")
	for _, c := range candidates {
		fmt.Fprintf(&b, "[Event ID: %s] %s\n", c.ID, c.Text)
	}
	return b.String()
}

// SelectByID asks sel to triage candidates against prompt and returns only the candidates
// whose id round-trips against the input set. It renders the [Event ID: <id>]-tagged
// prompt, validates the model's answer against the {summary, event_ids[]} schema, then
// resolves each returned id in the candidate store: real ids become Selected, fabricated
// ids become Dropped. A nil selector, a selector error, or an off-schema answer is a
// returned error (the triage did not happen); a fabricated id is NOT an error — it is
// grounded away, because dropping invented references is the whole point.
func SelectByID(ctx context.Context, candidates []Item, prompt string, sel Selector) (Selection, error) {
	if sel == nil {
		return Selection{}, fmt.Errorf("signals: SelectByID requires a non-nil Selector")
	}
	raw, err := sel.Select(ctx, RenderSelectPrompt(candidates, prompt))
	if err != nil {
		return Selection{}, fmt.Errorf("signals: selector failed: %w", err)
	}
	if err := ValidateAgainstSchema(selectByIDSchema, raw); err != nil {
		return Selection{}, fmt.Errorf("signals: selection off-schema: %w", err)
	}
	var parsed struct {
		Summary  string   `json:"summary"`
		EventIDs []string `json:"event_ids"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return Selection{}, fmt.Errorf("signals: selection unparseable: %w", err)
	}
	// The store is the closed set of real candidate ids — the only ids a selection may name.
	store := make(map[string]Item, len(candidates))
	for _, c := range candidates {
		store[c.ID] = c
	}
	result := Selection{Summary: parsed.Summary}
	seen := make(map[string]bool, len(parsed.EventIDs))
	for _, id := range parsed.EventIDs {
		if seen[id] {
			continue // a model that lists an id twice selects it once
		}
		seen[id] = true
		if item, ok := store[id]; ok {
			result.Selected = append(result.Selected, item)
		} else {
			result.Dropped = append(result.Dropped, id) // fabricated: never offered as a tag
		}
	}
	return result, nil
}
