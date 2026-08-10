package sessionaudit

import "testing"

func TestRepeatBenignCatalog(t *testing.T) {
	for _, name := range []string{"Read", "read", "Grep", "Glob", "WebSearch", "web_search", "search_kb"} {
		if !repeatBenign(name) {
			t.Errorf("repeatBenign(%q) = false, want true", name)
		}
	}
	for _, name := range []string{"", "Bash", "Edit", "Write", "unknown"} {
		if repeatBenign(name) {
			t.Errorf("repeatBenign(%q) = true, want false", name)
		}
	}
}
