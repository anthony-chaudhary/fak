// Package timingspan names the clock endpoints a timing field measures between,
// so a number always says which span it timed and two spans are never ratioed
// unless they are the same span.
package timingspan

import (
	"errors"
	"fmt"
	"strings"
)

// Endpoint is one named point on a request's timeline. The set is closed.
type Endpoint string

const (
	RequestReceived     Endpoint = "request_received"
	PromptEncoded       Endpoint = "prompt_encoded"
	PrefillStart        Endpoint = "prefill_start"
	PrefillEnd          Endpoint = "prefill_end"
	FirstTokenGenerated Endpoint = "first_token_generated"
	FirstByteSent       Endpoint = "first_byte_sent"
	LastToken           Endpoint = "last_token"
)

var closedVocabulary = map[Endpoint]bool{
	RequestReceived:     true,
	PromptEncoded:       true,
	PrefillStart:        true,
	PrefillEnd:          true,
	FirstTokenGenerated: true,
	FirstByteSent:       true,
	LastToken:           true,
}

// Span is the interval from one endpoint to a later one.
type Span struct {
	From Endpoint
	To   Endpoint
}

// The spans a server timing field may declare.
var (
	PromptSpan    = Span{From: PrefillStart, To: PrefillEnd}
	DecodeSpan    = Span{From: PrefillEnd, To: LastToken}
	FirstEmitSpan = Span{From: RequestReceived, To: FirstByteSent}
)

// ErrSpanMismatch is the typed refusal for forming a ratio across two spans.
var ErrSpanMismatch = errors.New("SPAN_MISMATCH")

// String renders the span as "from..to", the form a timing field declares.
func (s Span) String() string {
	return string(s.From) + ".." + string(s.To)
}

// ParseSpan reads "from..to" and rejects endpoints outside the closed vocabulary.
func ParseSpan(text string) (Span, error) {
	from, to, ok := strings.Cut(text, "..")
	if !ok {
		return Span{}, fmt.Errorf("span %q: want from..to", text)
	}
	s := Span{From: Endpoint(from), To: Endpoint(to)}
	if !closedVocabulary[s.From] || !closedVocabulary[s.To] {
		return Span{}, fmt.Errorf("span %q: endpoint outside the closed vocabulary", text)
	}
	return s, nil
}

// RequireSameSpan reports a SPAN_MISMATCH unless a and b are the same span.
func RequireSameSpan(a, b Span) error {
	if a == b {
		return nil
	}
	return fmt.Errorf("%w: %s vs %s", ErrSpanMismatch, a, b)
}
