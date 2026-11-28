package agent

import (
	"errors"
	"fmt"
	"strings"
)

//
// Upstream, GLM-4.x/5 tool-call parsing produced different tool names on the
// non-streaming path than on the streaming path because alias/prefix
// normalization was applied on only one of the two parse paths. The fix
// applies the same name-normalization map to non-streaming completions so
// downstream dispatch sees stable identifiers.
//
// This port pins the same contract host-side: one canonical alias/prefix map
// (canonicalToolAliases + canonicalToolPrefixes) is applied on EVERY parse
// path — streaming and non-streaming — so both yield identical canonical
// names for the same raw call.

// ErrUnknownToolAlias is returned when a raw tool name carries an alias
// namespace that the canonical map does not know. Fail-closed: an unknown
// alias is refused rather than passed through, so dispatch never sees a
// half-normalized identifier.
var ErrUnknownToolAlias = errors.New("agent: unknown tool-name alias")

// canonicalToolAliases maps raw (aliased) tool names to their canonical
// identifiers. This is the single source of truth for alias resolution;
// every parse path must route through CanonicalToolName so no path can
// diverge.
var canonicalToolAliases = map[string]string{
	"search_code":        "search",
	"web_search":         "search_web",
	"code_interpreter":   "interpreter",
	"str_replace_editor": "edit_file",
	"write_text_file":    "write_file",
	"browser_use":        "browser",
	"finish":             "finish",
}

// canonicalToolPrefixes are provider/namespace prefixes that some backends
// prepend to tool names (e.g. "functions.search_code"). They are stripped
// before alias lookup, on every parse path.
var canonicalToolPrefixes = []string{
	"functions.",
	"tools.",
	"namespace::",
}

// CanonicalToolName normalizes a raw tool-call name to its canonical
// identifier using the single canonical map. It is the one entry point both
// parse paths use, which is the contract: identical raw input yields
// identical canonical output regardless of which path parsed it.
//
// Rules, applied in order:
//  1. Trim surrounding whitespace; empty names are refused.
//  2. Strip known namespace prefixes (repeatedly, so stacked prefixes
//     collapse).
//  3. Resolve exact aliases through canonicalToolAliases.
//  4. Fail closed: a name that still carries an unrecognized alias namespace
//     (a dot or "::" separator) is refused with ErrUnknownToolAlias instead
//     of being passed through half-normalized.
func CanonicalToolName(raw string) (string, error) {
	name := strings.TrimSpace(raw)
	if name == "" {
		return "", fmt.Errorf("%w: empty tool name", ErrUnknownToolAlias)
	}
	for {
		trimmed := false
		for _, p := range canonicalToolPrefixes {
			if strings.HasPrefix(name, p) {
				name = strings.TrimPrefix(name, p)
				trimmed = true
				break
			}
		}
		if !trimmed {
			break
		}
	}
	if name == "" {
		return "", fmt.Errorf("%w: name is only a known prefix", ErrUnknownToolAlias)
	}
	if canon, ok := canonicalToolAliases[name]; ok {
		return canon, nil
	}
	// Fail-closed: an unrecognized alias namespace means the map does not
	// know this name; refuse it rather than leak a raw identifier to
	// dispatch.
	if strings.Contains(name, "::") || strings.Contains(name, ".") {
		return "", fmt.Errorf("%w: %q", ErrUnknownToolAlias, raw)
	}
	// Plain names with no alias form pass through unchanged: they are
	// already canonical.
	return name, nil
}

// CanonicalToolCall is a parsed tool invocation with its canonical name.
type CanonicalToolCall struct {
	// Name is the canonical tool identifier, stable across parse paths.
	Name string
	// RawName is the raw name as delivered by the backend.
	RawName string
	// Args is the raw JSON argument payload, unmodified.
	Args string
}

// ParseToolCallNonStreaming parses a complete (non-streaming) tool-call
// completion. The name is canonicalized through the shared map.
func ParseToolCallNonStreaming(rawName, args string) (CanonicalToolCall, error) {
	canon, err := CanonicalToolName(rawName)
	if err != nil {
		return CanonicalToolCall{}, err
	}
	return CanonicalToolCall{Name: canon, RawName: rawName, Args: args}, nil
}

// ParseToolCallStreaming parses a tool-call name delivered incrementally as
// streaming deltas. The deltas are joined into the full raw name and then
// canonicalized through the SAME map as the non-streaming path, so both
// paths yield identical canonical names for the same call.
func ParseToolCallStreaming(deltas []string, args string) (CanonicalToolCall, error) {
	raw := strings.Join(deltas, "")
	canon, err := CanonicalToolName(raw)
	if err != nil {
		return CanonicalToolCall{}, err
	}
	return CanonicalToolCall{Name: canon, RawName: raw, Args: args}, nil
}
