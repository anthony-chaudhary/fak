package strmatch

import "testing"

func TestContainsAny(t *testing.T) {
	cases := []struct {
		s    string
		subs []string
		want bool
	}{
		{"merge conflict in rebase", []string{"auth", "conflict"}, true},
		{"clean run", []string{"auth", "conflict"}, false},
		{"anything", nil, false},
		{"", []string{""}, true}, // strings.Contains(s, "") is true — pinned, matching the folded copies
		{"ERROR: boom", []string{"WARN", "ERROR"}, true},
	}
	for _, tc := range cases {
		if got := ContainsAny(tc.s, tc.subs...); got != tc.want {
			t.Errorf("ContainsAny(%q, %v) = %v, want %v", tc.s, tc.subs, got, tc.want)
		}
	}
}

func TestFirstContained(t *testing.T) {
	phrase, ok := FirstContained("the provider probably has it", []string{"skip re-send", "probably has"})
	if !ok || phrase != "probably has" {
		t.Fatalf("FirstContained = (%q, %v), want (\"probably has\", true)", phrase, ok)
	}
	if p, ok := FirstContained("clean", []string{"dirty"}); ok || p != "" {
		t.Fatalf("FirstContained miss = (%q, %v), want (\"\", false)", p, ok)
	}
	// First match wins in needle order — a caller's witness must be deterministic.
	if p, _ := FirstContained("ab", []string{"b", "a"}); p != "b" {
		t.Fatalf("needle order not honored: got %q, want \"b\"", p)
	}
}
