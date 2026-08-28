package devhandoff

import "testing"

func TestCommandsAreUniqueAndSortedByNames(t *testing.T) {
	seen := map[string]bool{}
	for _, command := range Commands {
		if command.Name == "" {
			t.Fatal("empty command name")
		}
		if command.Owner != "dev" || command.DispatchTarget != "fak-dev" || command.Handler == "" || command.SourceOrigin == "" || command.SourceClass != "dev-only" {
			t.Fatalf("incomplete generated ownership for %q: %+v", command.Name, command)
		}
		if seen[command.Name] {
			t.Fatalf("duplicate command %q", command.Name)
		}
		seen[command.Name] = true
		if !IsCommand(command.Name) {
			t.Fatalf("IsCommand(%q) = false", command.Name)
		}
		for _, alias := range command.Aliases {
			if seen[alias] {
				t.Fatalf("duplicate command alias %q", alias)
			}
			seen[alias] = true
			if !IsCommand(alias) {
				t.Fatalf("IsCommand(%q) = false", alias)
			}
		}
	}
	names := Names()
	for i := 1; i < len(names); i++ {
		if names[i-1] >= names[i] {
			t.Fatalf("Names not strictly sorted: %q then %q", names[i-1], names[i])
		}
	}
}
