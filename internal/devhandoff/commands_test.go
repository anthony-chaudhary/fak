package devhandoff

import "testing"

func TestCommandsAreUniqueAndSortedByNames(t *testing.T) {
	seen := map[string]bool{}
	for _, command := range Commands {
		if command.Name == "" {
			t.Fatal("empty command name")
		}
		if seen[command.Name] {
			t.Fatalf("duplicate command %q", command.Name)
		}
		seen[command.Name] = true
		if !IsCommand(command.Name) {
			t.Fatalf("IsCommand(%q) = false", command.Name)
		}
	}
	names := Names()
	for i := 1; i < len(names); i++ {
		if names[i-1] >= names[i] {
			t.Fatalf("Names not strictly sorted: %q then %q", names[i-1], names[i])
		}
	}
}
