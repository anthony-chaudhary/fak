package signals

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// fakeSelector is a deterministic stand-in for the model call: it returns canned JSON and
// records the prompt it was handed, so a test can assert both the grounding tags and the
// round-trip filtering with no network.
type fakeSelector struct {
	raw       json.RawMessage
	err       error
	gotPrompt string
}

func (f *fakeSelector) Select(_ context.Context, prompt string) (json.RawMessage, error) {
	f.gotPrompt = prompt
	return f.raw, f.err
}

func triageCandidates() []Item {
	return []Item{
		{ID: "evt-1", Text: "ran the tests"},
		{ID: "evt-2", Text: "opened a PR"},
	}
}

func TestSelectByID_DropsFabricatedIDs(t *testing.T) {
	// The model returns one REAL id (evt-2) and one FABRICATED id (evt-999).
	stub := &fakeSelector{raw: json.RawMessage(`{"summary":"picked the PR","event_ids":["evt-2","evt-999"]}`)}
	got, err := SelectByID(context.Background(), triageCandidates(), "which event opened a PR?", stub)
	if err != nil {
		t.Fatal(err)
	}
	// The real id survives...
	if len(got.Selected) != 1 || got.Selected[0].ID != "evt-2" {
		t.Fatalf("selected = %+v, want only evt-2", got.Selected)
	}
	// ...the fabricated id is dropped and surfaced for audit, not silently swallowed.
	if len(got.Dropped) != 1 || got.Dropped[0] != "evt-999" {
		t.Fatalf("dropped = %v, want [evt-999]", got.Dropped)
	}
	if got.Summary != "picked the PR" {
		t.Fatalf("summary = %q, want %q", got.Summary, "picked the PR")
	}
	// The prompt must carry the [Event ID: <id>] grounding tags for every candidate.
	for _, tag := range []string{"[Event ID: evt-1]", "[Event ID: evt-2]"} {
		if !strings.Contains(stub.gotPrompt, tag) {
			t.Fatalf("prompt missing grounding tag %q:\n%s", tag, stub.gotPrompt)
		}
	}
}

func TestSelectByID_DedupsAndPreservesModelOrder(t *testing.T) {
	stub := &fakeSelector{raw: json.RawMessage(`{"summary":"both, evt-2 first","event_ids":["evt-2","evt-1","evt-2"]}`)}
	got, err := SelectByID(context.Background(), triageCandidates(), "which matter?", stub)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Selected) != 2 || got.Selected[0].ID != "evt-2" || got.Selected[1].ID != "evt-1" {
		t.Fatalf("selected = %+v, want [evt-2 evt-1] deduped in model order", got.Selected)
	}
	if len(got.Dropped) != 0 {
		t.Fatalf("dropped = %v, want none", got.Dropped)
	}
}

func TestSelectByID_EmptySelectionIsValid(t *testing.T) {
	stub := &fakeSelector{raw: json.RawMessage(`{"summary":"nothing relevant","event_ids":[]}`)}
	got, err := SelectByID(context.Background(), triageCandidates(), "unrelated request", stub)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Selected) != 0 || len(got.Dropped) != 0 {
		t.Fatalf("empty selection = %+v, want no selected/dropped", got)
	}
}

func TestSelectByID_OffSchemaIsError(t *testing.T) {
	cases := map[string]string{
		"event_ids not array": `{"summary":"x","event_ids":"evt-1"}`,
		"missing event_ids":   `{"summary":"x"}`,
		"missing summary":     `{"event_ids":["evt-1"]}`,
		"id not string":       `{"summary":"x","event_ids":[7]}`,
	}
	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			stub := &fakeSelector{raw: json.RawMessage(raw)}
			if _, err := SelectByID(context.Background(), triageCandidates(), "q", stub); err == nil {
				t.Fatalf("off-schema answer %q was accepted", raw)
			}
		})
	}
}

func TestSelectByID_NilSelectorAndSelectorError(t *testing.T) {
	if _, err := SelectByID(context.Background(), triageCandidates(), "q", nil); err == nil {
		t.Fatal("nil selector must error")
	}
	stub := &fakeSelector{err: errors.New("model exploded")}
	if _, err := SelectByID(context.Background(), triageCandidates(), "q", stub); err == nil || !strings.Contains(err.Error(), "model exploded") {
		t.Fatalf("selector error must propagate, got %v", err)
	}
}
