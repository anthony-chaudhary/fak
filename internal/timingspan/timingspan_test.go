package timingspan

import (
	"errors"
	"testing"
)

func TestParseSpanAcceptsOnlyClosedVocabulary(t *testing.T) {
	s, err := ParseSpan("prefill_start..prefill_end")
	if err != nil {
		t.Fatalf("ParseSpan(prefill_start..prefill_end) = %v", err)
	}
	if s != PromptSpan {
		t.Fatalf("parsed %v, want PromptSpan %v", s, PromptSpan)
	}
	if _, err := ParseSpan("prefill_start..made_up_point"); err == nil {
		t.Fatal("ParseSpan accepted an endpoint outside the closed vocabulary")
	}
	if _, err := ParseSpan("prefill_start"); err == nil {
		t.Fatal("ParseSpan accepted a span with no end endpoint")
	}
}

func TestSpanStringRoundTrips(t *testing.T) {
	for _, s := range []Span{PromptSpan, DecodeSpan, FirstEmitSpan} {
		got, err := ParseSpan(s.String())
		if err != nil || got != s {
			t.Fatalf("round trip %v: got %v err %v", s, got, err)
		}
	}
}

func TestRequireSameSpanRefusesMismatch(t *testing.T) {
	if err := RequireSameSpan(PromptSpan, PromptSpan); err != nil {
		t.Fatalf("same span refused: %v", err)
	}
	err := RequireSameSpan(PromptSpan, FirstEmitSpan)
	if err == nil {
		t.Fatal("mismatched spans formed a comparison")
	}
	if !errors.Is(err, ErrSpanMismatch) {
		t.Fatalf("error %v does not wrap ErrSpanMismatch", err)
	}
}
