package sessionreset

import (
	"errors"
	"strings"
	"testing"
)

// longTranscript is a session long enough to clear the model distiller's cost gate, with a
// clear objective, an interior decision and blocker (which the EXTRACTIVE distiller cannot
// see), and a recent tail.
func longTranscript() []Msg {
	return []Msg{
		{Role: "system", Content: "You are a helpful coding assistant for the fak repo with a long preamble that establishes conventions and tone for the whole session."},
		{Role: "user", Content: "Help me add a budget-triggered session reset to fak so long sessions carry over cleanly."},
		{Role: "assistant", Content: "We decided to fold pluggable contributors over the drained transcript rather than clearing everything."},
		{Role: "user", Content: "The warm-prefix KV splice is blocked because vcachechain is gated off — note that as the open blocker."},
		{Role: "assistant", Content: "Understood; recording the deferred KV reuse as a follow-on."},
		{Role: "user", Content: "Now wire the model-call distiller behind a cost flag."},
	}
}

// fakeSummarizer records the prompt it was handed and returns a canned recap, so a test can
// assert both that the WHOLE transcript reaches the model and that its output is folded.
type fakeSummarizer struct {
	gotPrompt string
	reply     string
	err       error
}

func (f *fakeSummarizer) summarize(prompt string) (string, error) {
	f.gotPrompt = prompt
	return f.reply, f.err
}

// TestModelDistillFeedsWholeTranscriptAndFoldsSummary proves the contributor sends the
// recap instruction plus the full transcript to the model seam and renders the returned
// "where we are" summary into its Part.
func TestModelDistillFeedsWholeTranscriptAndFoldsSummary(t *testing.T) {
	fake := &fakeSummarizer{reply: "Objective: budget reset. Decision: fold contributors. Blocker: KV splice deferred."}
	c := NewModelDistiller(fake.summarize, 0)

	p, ok := c.Contribute(Input{Messages: longTranscript()})
	if !ok {
		t.Fatal("model distiller declined a long transcript with a working seam")
	}
	// The model's recap is folded verbatim under the recap header.
	if !strings.Contains(p.Text, "Blocker: KV splice deferred") {
		t.Fatalf("model summary not folded into the Part: %q", p.Text)
	}
	if !strings.HasPrefix(p.Text, "Where we are (model recap):") {
		t.Fatalf("recap header missing: %q", p.Text)
	}
	if p.Meta["source"] != "model" {
		t.Fatalf("Meta source = %q, want model", p.Meta["source"])
	}
	if p.Order != ModelDistillOrder {
		t.Fatalf("Order = %d, want %d (just after extractive task_distill)", p.Order, ModelDistillOrder)
	}
	// The instruction AND an interior line the extractive distiller would miss both reach
	// the model — proving the whole transcript is fed, not just first/last.
	if !strings.Contains(fake.gotPrompt, ModelDistillInstruction) {
		t.Fatal("recap instruction not prepended to the model prompt")
	}
	if !strings.Contains(fake.gotPrompt, "vcachechain is gated off") {
		t.Fatalf("interior transcript line missing from model prompt: %q", fake.gotPrompt)
	}
}

// TestModelDistillDeclinesGracefully proves every failure path falls through (ok=false)
// instead of poisoning the seed, so the deterministic taskDistill always covers the slot.
func TestModelDistillDeclinesGracefully(t *testing.T) {
	t.Run("model error", func(t *testing.T) {
		fake := &fakeSummarizer{err: errors.New("timeout")}
		if p, ok := NewModelDistiller(fake.summarize, 0).Contribute(Input{Messages: longTranscript()}); ok {
			t.Fatalf("expected decline on model error, got %+v", p)
		}
	})
	t.Run("empty summary", func(t *testing.T) {
		fake := &fakeSummarizer{reply: "   "}
		if _, ok := NewModelDistiller(fake.summarize, 0).Contribute(Input{Messages: longTranscript()}); ok {
			t.Fatal("expected decline on empty summary")
		}
	})
	t.Run("nil seam", func(t *testing.T) {
		if _, ok := NewModelDistiller(nil, 0).Contribute(Input{Messages: longTranscript()}); ok {
			t.Fatal("expected decline with a nil model seam")
		}
	})
	t.Run("below min chars", func(t *testing.T) {
		fake := &fakeSummarizer{reply: "should never be asked for"}
		short := Input{Messages: []Msg{{Role: "user", Content: "hi"}}}
		p, ok := NewModelDistiller(fake.summarize, DefaultModelDistillMinChars).Contribute(short)
		if ok {
			t.Fatal("expected decline below the cost-gate min chars")
		}
		if fake.gotPrompt != "" {
			t.Fatal("cost gate must short-circuit BEFORE calling the model")
		}
		if p.Meta["skipped"] != "below_min_chars" {
			t.Fatalf("decline reason = %q, want below_min_chars", p.Meta["skipped"])
		}
	})
}

// TestRegisterModelDistillerJoinsFold proves the opt-in path: RegisterModelDistiller adds
// the contributor to the global fold so its recap lands in BuildSeed, while a nil seam
// registers nothing.
func TestRegisterModelDistillerJoinsFold(t *testing.T) {
	if got := RegisterModelDistiller(nil, 0); got != nil {
		t.Fatal("RegisterModelDistiller(nil) must register nothing and return nil")
	}
	before := len(Registered())

	fake := &fakeSummarizer{reply: "Model recap: continuing the budget-reset wiring."}
	if c := RegisterModelDistiller(fake.summarize, 1); c == nil {
		t.Fatal("RegisterModelDistiller returned nil for a real seam")
	}
	if len(Registered()) != before+1 {
		t.Fatalf("registry did not grow: %d -> %d", before, len(Registered()))
	}

	seed := BuildSeed(Input{Messages: longTranscript()})
	if !strings.Contains(seed.Recap, "Model recap: continuing the budget-reset wiring") {
		t.Fatalf("model recap missing from the folded seed: %q", seed.Recap)
	}
}
