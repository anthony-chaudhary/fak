package disambiguation

import (
	"sort"
	"strings"
)

// CLISourceSchemaVersion identifies terminology derived from fak's public help.
const CLISourceSchemaVersion = "fak-disambiguation-cli-source/1"

// CLITermKind names the CLI namespace that owns a discovered term.
type CLITermKind string

const (
	CLITermCommand    CLITermKind = "command"
	CLITermSubcommand CLITermKind = "subcommand"
	CLITermFlag       CLITermKind = "flag"
)

// CLITerm is one command, subcommand, or long flag discovered from a public
// usage synopsis. Invocation preserves the command context for overloaded flags.
type CLITerm struct {
	Term       string      `json:"term"`
	Kind       CLITermKind `json:"kind"`
	Invocation string      `json:"invocation"`
}

// CLISourceReport is a read-only projection of the public help source. Stale
// contains prior terms no longer emitted by that source.
type CLISourceReport struct {
	Schema string    `json:"schema"`
	Terms  []CLITerm `json:"terms"`
	Stale  []CLITerm `json:"stale"`
}

// IndexCLISource derives terminology from fak usage synopsis lines and compares
// it with an optional prior snapshot. It never writes the canonical index.
func IndexCLISource(help string, prior []CLITerm) CLISourceReport {
	current := make(map[string]CLITerm)
	for _, line := range strings.Split(help, "\n") {
		if line == strings.TrimLeft(line, " \t") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 || fields[0] != "fak" {
			continue
		}
		invocationParts := []string{"fak"}
		for n, field := range fields[1:] {
			field = strings.Trim(field, " ,;()[]")
			if field == "" {
				continue
			}
			if strings.HasPrefix(field, "--") {
				flag := strings.TrimRight(strings.SplitN(field, "=", 2)[0], "]")
				addCLITerm(current, CLITerm{Term: flag, Kind: CLITermFlag, Invocation: strings.Join(invocationParts, " ")})
				continue
			}
			if strings.ContainsAny(field, "[<|") || strings.HasPrefix(field, "-") || field == strings.ToUpper(field) {
				continue
			}
			kind := CLITermCommand
			if n > 0 {
				kind = CLITermSubcommand
			}
			invocationParts = append(invocationParts, field)
			addCLITerm(current, CLITerm{Term: field, Kind: kind, Invocation: strings.Join(invocationParts, " ")})
		}
	}

	terms := cliTermsSorted(current)
	staleMap := make(map[string]CLITerm)
	for _, term := range prior {
		if _, ok := current[cliTermKey(term)]; !ok {
			staleMap[cliTermKey(term)] = term
		}
	}
	return CLISourceReport{Schema: CLISourceSchemaVersion, Terms: terms, Stale: cliTermsSorted(staleMap)}
}

func addCLITerm(dst map[string]CLITerm, term CLITerm) {
	dst[cliTermKey(term)] = term
}

func cliTermKey(term CLITerm) string {
	return string(term.Kind) + "\x00" + term.Invocation + "\x00" + term.Term
}

func cliTermsSorted(values map[string]CLITerm) []CLITerm {
	result := make([]CLITerm, 0, len(values))
	for _, term := range values {
		result = append(result, term)
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].Invocation != result[j].Invocation {
			return result[i].Invocation < result[j].Invocation
		}
		if result[i].Kind != result[j].Kind {
			return result[i].Kind < result[j].Kind
		}
		return result[i].Term < result[j].Term
	})
	return result
}
