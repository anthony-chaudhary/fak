// Package devhandoff defines the compatibility boundary between runtime fak and
// the separately linked fak-dev executable.
package devhandoff

import "sort"

// Command is a repository-development command implemented by fak-dev that
// runtime fak recognizes only to provide an actionable process handoff.
type Command struct {
	Name           string
	Aliases        []string
	Owner          string
	Handler        string
	SourceOrigin   string
	DispatchTarget string
	SourceClass    string
}

var commandSet = func() map[string]struct{} {
	set := make(map[string]struct{}, len(Commands))
	for _, command := range Commands {
		set[command.Name] = struct{}{}
		for _, alias := range command.Aliases {
			set[alias] = struct{}{}
		}
	}
	return set
}()

// IsCommand reports whether name is implemented by the separate fak-dev binary.
func IsCommand(name string) bool {
	_, ok := commandSet[name]
	return ok
}

// Names returns the moved command names in stable lexical order.
func Names() []string {
	names := make([]string, 0, len(commandSet))
	for _, command := range Commands {
		names = append(names, command.Name)
		names = append(names, command.Aliases...)
	}
	sort.Strings(names)
	return names
}
