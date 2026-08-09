package normgate

import "testing"

func TestContainmentSeparatesParaphrasesFromNonPairs(t *testing.T) {
	t.Parallel()
	const threshold = 0.60
	tests := []struct {
		name string
		a, b []string
		near bool
	}{
		{"paraphrase-read", []string{"read", "cached", "tool", "result"}, []string{"reuse", "cached", "tool", "result", "locally"}, true},
		{"paraphrase-quota", []string{"persist", "rate", "limit", "headers"}, []string{"durable", "rate", "limit", "headers", "ledger"}, true},
		{"unrelated", []string{"render", "terminal", "pane"}, []string{"rotate", "account", "credential"}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Containment(tt.a, tt.b)
			if tt.near && got < threshold {
				t.Fatalf("Containment() = %.2f, want >= %.2f", got, threshold)
			}
			if !tt.near && got >= threshold {
				t.Fatalf("Containment() = %.2f, want < %.2f", got, threshold)
			}
		})
	}
}

func TestContainmentSetSemantics(t *testing.T) {
	t.Parallel()
	if got := Containment([]string{"a", "a", "b"}, []string{"a", "b", "c"}); got != 1 {
		t.Fatalf("duplicates changed score: got %v, want 1", got)
	}
	if got := Containment(nil, []string{"a"}); got != 0 {
		t.Fatalf("empty score = %v, want 0", got)
	}
	if ab, ba := Containment([]string{"a", "b"}, []string{"a", "b", "c"}), Containment([]string{"a", "b", "c"}, []string{"a", "b"}); ab != ba {
		t.Fatalf("score not symmetric: %v != %v", ab, ba)
	}
}
