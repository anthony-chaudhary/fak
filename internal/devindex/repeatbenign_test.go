package devindex

import "testing"

func TestRepeatBenignCatalog(t *testing.T) {
	for _, name := range []string{"Read", "read", "Grep", "Glob", "WebSearch", "web_search", "search_kb"} {
		if !RepeatBenign(name) {
			t.Errorf("RepeatBenign(%q) = false, want true", name)
		}
	}
	for _, name := range []string{"", "Bash", "Edit", "Write", "unknown"} {
		if RepeatBenign(name) {
			t.Errorf("RepeatBenign(%q) = true, want false", name)
		}
	}
}
