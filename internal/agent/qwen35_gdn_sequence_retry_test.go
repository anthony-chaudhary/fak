package agent

import (
	"errors"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// A typed decline of a restored prefix retries fresh; a fresh-session decline
// or an untyped admission error still fails the opt-in request explicitly.
func TestQwen35GDNSequenceRetryFreshOnlyForDeclinedRestoredPrefix(t *testing.T) {
	m := model.NewSynthetic(model.Config{LayerTypes: []string{"linear_attention"}})
	s := m.NewSession()
	defer s.Close()
	declined := s.EnableQwen35MetalGDNPreprojectedSequence()
	var unsupported *model.UnsupportedGDNPreprojectedSequenceError
	if !errors.As(declined, &unsupported) {
		t.Fatalf("plain host session admission err=%v, want typed decline", declined)
	}
	wrapped := fmt.Errorf("admission: %w", declined)
	cases := []struct {
		name    string
		err     error
		matched int
		want    bool
	}{
		{"restored typed decline", declined, 64, true},
		{"restored wrapped typed decline", wrapped, 1, true},
		{"fresh typed decline", declined, 0, false},
		{"restored untyped error", errors.New("projection admission failed"), 64, false},
		{"no error", nil, 64, false},
	}
	for _, tc := range cases {
		if got := qwen35GDNSequenceRetryFresh(tc.err, tc.matched); got != tc.want {
			t.Errorf("%s: retry=%v, want %v", tc.name, got, tc.want)
		}
	}
}
